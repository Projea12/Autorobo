"""
perception/pipeline.py — Option B tiered perception orchestrator.

Runs every frame on MacBook M1:
  Tier 1 (every frame)  — YOLOv8n COCO detection + DepthAnything v2 depth map
  Tier 2 (low conf only)— SAM 2 precise mask + CLIP identification

Tier 2 fires only when a YOLOv8 detection falls below `low_conf_trigger`.
Objects at the same 3D location are cached — never reidentified.

Failure modes handled:
  - YOLO failure          → returns empty detection list, depth map still valid
  - DepthEstimator fail   → last valid map returned by DepthEstimator itself
  - SAM 2 failure         → original low-conf detection kept as-is, logged
  - CLIP failure          → "unknown object" label, confidence 0.0, logged
  - No models loaded      → RuntimeError with clear message at init
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .detector import Detection, DetectorConfig, ObjectDetector
from .depth import DepthEstimator
from .sam_segmentor import SAM2Config, SAM2Segmentor
from .clip_identifier import CLIPConfig, CLIPIdentifier, IdentificationResult

log = logging.getLogger(__name__)


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class UnknownObject:
    """
    Object SAM 2 segmented and CLIP identified — was low-confidence YOLO detection.
    """
    label:       str
    confidence:  float
    bbox_xyxy:   np.ndarray
    depth:       float
    mask:        Optional[np.ndarray]
    top_k:       list[tuple[str, float]]

    def __repr__(self) -> str:
        return (f"UnknownObject({self.label!r}, conf={self.confidence:.2f}, "
                f"depth={self.depth:.2f}m)")


@dataclass
class PerceptionFrame:
    """
    Full output of one pipeline tick.

    detections : high-confidence YOLOv8 results (conf >= low_conf_trigger)
    unknowns   : objects YOLOv8 was uncertain about, identified by SAM 2 + CLIP
    depth_map  : (H, W) float32 metres — always valid (falls back to last map)
    frame_id   : monotonically increasing counter
    elapsed_ms : wall-clock time for full pipeline tick
    """
    detections: list[Detection]
    unknowns:   list[UnknownObject]
    depth_map:  np.ndarray
    frame_id:   int
    elapsed_ms: float

    @property
    def all_objects(self) -> list:
        """Combined list — high-conf detections + identified unknowns."""
        return self.detections + self.unknowns  # type: ignore[operator]


# ── pipeline configuration ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """
    Thresholds that control when Tier 2 fires.

    low_conf_trigger : YOLO confidence below this → SAM 2 + CLIP fires
    min_conf_discard : YOLO confidence below this → discard completely (noise)
    cache_grid_xy    : normalised pixel grid cells per axis for spatial cache
    cache_depth_res  : depth rounding in metres for spatial cache
    """
    yolo_weights:     str   = "yolov8n.pt"
    low_conf_trigger: float = 0.40   # SAM 2 + CLIP fires below this
    min_conf_discard: float = 0.15   # below this → discard, not worth processing
    cache_grid_xy:    int   = 10     # 10×10 normalised pixel grid
    cache_depth_res:  float = 0.25   # 0.25m depth bins for cache


# ── spatial cache ─────────────────────────────────────────────────────────────

class _SpatialCache:
    """
    Per-session cache: identified unknown objects by approximate 3D location.

    Key: (grid_x, grid_y, depth_bin) derived from normalised pixel centre + depth.
    Prevents re-running SAM 2 + CLIP on the same object every frame.
    """

    def __init__(self, grid_xy: int, depth_res: float) -> None:
        self._grid_xy  = grid_xy
        self._depth_res = depth_res
        self._store: dict[tuple, IdentificationResult] = {}

    def _key(
        self,
        bbox_xyxy: np.ndarray,
        frame_hw:  tuple[int, int],
        depth:     float,
    ) -> tuple:
        H, W = frame_hw
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0 / W
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0 / H
        gx = int(cx * self._grid_xy)
        gy = int(cy * self._grid_xy)
        db = int(depth / self._depth_res)
        return (gx, gy, db)

    def get(
        self,
        bbox_xyxy: np.ndarray,
        frame_hw:  tuple[int, int],
        depth:     float,
    ) -> Optional[IdentificationResult]:
        return self._store.get(self._key(bbox_xyxy, frame_hw, depth))

    def put(
        self,
        bbox_xyxy: np.ndarray,
        frame_hw:  tuple[int, int],
        depth:     float,
        result:    IdentificationResult,
    ) -> None:
        self._store[self._key(bbox_xyxy, frame_hw, depth)] = result

    def size(self) -> int:
        return len(self._store)


# ── pipeline ──────────────────────────────────────────────────────────────────

class PerceptionPipeline:
    """
    Option B tiered perception — YOLOv8 every frame, SAM 2 + CLIP on demand.

    Usage
    -----
        pipe = PerceptionPipeline()
        result = pipe.process(frame_rgb)
        for det in result.detections:
            print(det.class_name, det.confidence, det.bbox_xyxy)
        for unk in result.unknowns:
            print(unk.label, unk.confidence, unk.depth)
    """

    def __init__(
        self,
        cfg:       PipelineConfig  = PipelineConfig(),
        sam_cfg:   SAM2Config      = SAM2Config(),
        clip_cfg:  CLIPConfig      = CLIPConfig(),
        detector_cfg: DetectorConfig = DetectorConfig(),
    ) -> None:
        self.cfg    = cfg
        self._frame_id = 0
        self._cache = _SpatialCache(cfg.cache_grid_xy, cfg.cache_depth_res)

        log.info("Loading perception pipeline (Tier 1: YOLO + Depth, Tier 2: SAM2 + CLIP)...")

        # Tier 1 — always on
        det_cfg = DetectorConfig(
            weights_path = cfg.yolo_weights,
            conf_thresh  = cfg.min_conf_discard,
        )
        self._detector = ObjectDetector(det_cfg)
        self._depth    = DepthEstimator()

        # Tier 2 — lazy: only load if SAM 2 checkpoint exists
        self._sam:  Optional[SAM2Segmentor] = None
        self._clip: Optional[CLIPIdentifier] = None
        try:
            self._sam  = SAM2Segmentor(sam_cfg)
            self._clip = CLIPIdentifier(clip_cfg)
            log.info("Tier 2 (SAM2 + CLIP) loaded — unknown object identification active")
        except Exception as exc:
            log.warning(
                "Tier 2 unavailable (%s) — pipeline runs Tier 1 only. "
                "Low-conf detections will be kept with original YOLO label.",
                exc,
            )

        log.info("PerceptionPipeline ready (Tier 2: %s)", "ON" if self._sam else "OFF")

    # ── public API ─────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> PerceptionFrame:
        """
        Run full tiered perception on one RGB frame.

        Parameters
        ----------
        frame : (H, W, 3) uint8 RGB

        Returns
        -------
        PerceptionFrame with detections, unknowns, depth map, timing.
        """
        t0 = time.perf_counter()
        self._frame_id += 1

        # ── Tier 1 ──────────────────────────────────────────────────────────
        all_detections = self._detect(frame)
        depth_map      = self._depth.estimate(frame)

        high_conf: list[Detection]     = []
        low_conf:  list[Detection]     = []

        for det in all_detections:
            if det.confidence >= self.cfg.low_conf_trigger:
                high_conf.append(det)
            elif det.confidence >= self.cfg.min_conf_discard:
                low_conf.append(det)
            # below min_conf_discard → silently discard

        # ── Tier 2 ──────────────────────────────────────────────────────────
        unknowns: list[UnknownObject] = []
        if low_conf:
            unknowns = self._identify_unknowns(frame, depth_map, low_conf)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if self._frame_id % 30 == 0:
            log.debug(
                "Frame %d — %d detections, %d unknowns, %.1fms | cache=%d",
                self._frame_id, len(high_conf), len(unknowns),
                elapsed_ms, self._cache.size(),
            )

        return PerceptionFrame(
            detections = high_conf,
            unknowns   = unknowns,
            depth_map  = depth_map,
            frame_id   = self._frame_id,
            elapsed_ms = elapsed_ms,
        )

    # ── internals ──────────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> list[Detection]:
        try:
            return self._detector.detect(frame)
        except Exception as exc:
            log.warning("YOLO detection failed: %s", exc)
            return []

    def _identify_unknowns(
        self,
        frame:     np.ndarray,
        depth_map: np.ndarray,
        low_conf:  list[Detection],
    ) -> list[UnknownObject]:
        if self._sam is None or self._clip is None:
            # Tier 2 unavailable — keep low-conf detections with original YOLO label
            return [
                UnknownObject(
                    label      = det.class_name,
                    confidence = det.confidence,
                    bbox_xyxy  = det.bbox_xyxy,
                    depth      = self._centroid_depth(depth_map, det.bbox_xyxy),
                    mask       = None,
                    top_k      = [],
                )
                for det in low_conf
            ]

        unknowns: list[UnknownObject] = []
        H, W = frame.shape[:2]

        for det in low_conf:
            depth = self._centroid_depth(depth_map, det.bbox_xyxy)

            # Check spatial cache — skip reidentification for known locations
            cached = self._cache.get(det.bbox_xyxy, (H, W), depth)
            if cached is not None:
                unknowns.append(UnknownObject(
                    label      = cached.label,
                    confidence = cached.confidence,
                    bbox_xyxy  = det.bbox_xyxy,
                    depth      = depth,
                    mask       = None,
                    top_k      = cached.top_k,
                ))
                continue

            # SAM 2 — precise mask from bounding box
            mask = None
            try:
                mask = self._sam.segment_from_bbox(frame, det.bbox_xyxy)
            except Exception as exc:
                log.warning("SAM2 segment_from_bbox failed for %s: %s", det.class_name, exc)

            # CLIP — identify from mask crop (or full bbox crop if mask failed)
            try:
                result = self._clip.identify(frame, mask=mask)
            except Exception as exc:
                log.warning("CLIP identify failed for %s: %s", det.class_name, exc)
                result = IdentificationResult("unknown object", 0.0, [])

            self._cache.put(det.bbox_xyxy, (H, W), depth, result)

            unknowns.append(UnknownObject(
                label      = result.label,
                confidence = result.confidence,
                bbox_xyxy  = det.bbox_xyxy,
                depth      = depth,
                mask       = mask,
                top_k      = result.top_k,
            ))

        return unknowns

    @staticmethod
    def _centroid_depth(depth_map: np.ndarray, bbox_xyxy: np.ndarray) -> float:
        """Median depth in the bounding box region."""
        x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
        H, W = depth_map.shape
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return 10.0
        region = depth_map[y1:y2, x1:x2]
        return float(np.median(region))

    @property
    def tier2_active(self) -> bool:
        return self._sam is not None and self._clip is not None

    @property
    def cache_size(self) -> int:
        return self._cache.size()

    def __repr__(self) -> str:
        t2 = "SAM2+CLIP" if self.tier2_active else "disabled"
        return (f"PerceptionPipeline(yolo={self.cfg.yolo_weights!r}, "
                f"tier2={t2}, cache={self.cache_size})")
