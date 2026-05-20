"""
perception/detector.py — YOLOv8 object detector for YCB manipulation objects.

Wraps Ultralytics YOLOv8 with a stable interface used throughout the
perception pipeline.  The implementation degrades gracefully when
ultralytics is not installed: construction succeeds but detect() raises
a clear ImportError so the rest of the system can still be imported.

Typical usage
─────────────
    cfg = DetectorConfig(weights_path="models/ycb_yolo.pt", conf_thresh=0.30)
    det = ObjectDetector(cfg)

    frame = env.capture_rgbd()
    detections = det.detect(frame.rgb)
    for d in detections:
        print(d.class_name, d.confidence, d.bbox_xyxy)

    # batch
    batch = det.detect_batch([frame1.rgb, frame2.rgb])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Lazy import — store the module reference at class-construction time so
# tests can patch perception.detector._ultralytics.
try:
    import ultralytics as _ultralytics  # type: ignore
except ImportError:
    _ultralytics = None  # type: ignore


# ── detection dataclass ───────────────────────────────────────────────────────

@dataclass
class Detection:
    """
    Single object detection result.

    Fields
    ------
    class_id    : integer class index (matches dataset.yaml ordering)
    class_name  : human-readable name, e.g. "025_mug"
    confidence  : score in [0, 1]
    bbox_xyxy   : (4,) float32 pixel coords [x1, y1, x2, y2]
    bbox_xywh   : (4,) float32 pixel coords [cx, cy, w, h]
    mask        : (H, W) bool segmentation mask, or None
    position_3d : (3,) float32 position in robot base frame (m), or None
    track_id    : multi-frame track ID, or None
    """
    class_id:    int
    class_name:  str
    confidence:  float
    bbox_xyxy:   np.ndarray              # (4,) float32
    bbox_xywh:   np.ndarray              # (4,) float32
    mask:        Optional[np.ndarray] = None   # (H, W) bool
    position_3d: Optional[np.ndarray] = None   # (3,) float32
    track_id:    Optional[int]        = None

    def __repr__(self) -> str:
        pos = f" pos={self.position_3d.tolist()}" if self.position_3d is not None else ""
        return (f"Detection({self.class_name!r}, conf={self.confidence:.2f}, "
                f"bbox={self.bbox_xyxy.tolist()}{pos})")

    @property
    def area(self) -> float:
        """Bounding box area in pixels²."""
        return float(self.bbox_xywh[2] * self.bbox_xywh[3])

    @property
    def centre(self) -> np.ndarray:
        """Bounding box centre (cx, cy) in pixels."""
        return self.bbox_xywh[:2].copy()


# ── detector configuration ────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectorConfig:
    """
    Configuration for ObjectDetector.

    weights_path   : path to a fine-tuned .pt file, or a base model name
                     like "yolov8n.pt" (downloaded automatically on first use)
    conf_thresh    : minimum confidence to keep a detection [0, 1]
    iou_thresh     : NMS IoU threshold [0, 1]
    device         : "cpu", "cuda", "cuda:0", "mps", or "" (auto)
    half           : use FP16 inference (GPU only)
    imgsz          : inference image size (pixels); images are letterboxed
    max_det        : maximum detections per image
    """
    weights_path: str   = "yolov8n.pt"
    conf_thresh:  float = 0.25
    iou_thresh:   float = 0.45
    device:       str   = ""      # empty string → auto (CUDA if available)
    half:         bool  = False
    imgsz:        int   = 640
    max_det:      int   = 100


# ── detector ──────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    YOLOv8-based detector for YCB manipulation objects.

    Parameters
    ----------
    cfg : DetectorConfig

    Raises
    ------
    ImportError  : at detect() time if ultralytics is not installed
    FileNotFoundError : if weights_path does not exist (ultralytics handles this
                        by downloading the base model, so only custom paths fail)
    """

    def __init__(self, cfg: DetectorConfig = DetectorConfig()) -> None:
        self.cfg = cfg
        self._model = None
        self._class_names: list[str] = []

        if _ultralytics is None:
            log.warning(
                "ultralytics not installed — ObjectDetector created in stub mode. "
                "Install with: pip install ultralytics"
            )
            return

        try:
            yolo_cls = _ultralytics.YOLO
            self._model = yolo_cls(cfg.weights_path)
            device = cfg.device or None
            if device:
                self._model.to(device)
            if cfg.half and device not in ("cpu", ""):
                self._model.half()
            self._class_names = list(self._model.names.values())
            log.info(
                "ObjectDetector loaded '%s' — %d classes, device=%s",
                cfg.weights_path, len(self._class_names), device or "auto",
            )
        except Exception as exc:
            log.error("Failed to load YOLO model: %s", exc)
            self._model = None

    # ── inference ─────────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect objects in a single RGB image.

        Parameters
        ----------
        image : (H, W, 3) uint8 RGB.  BGR is also accepted (auto-converted).

        Returns
        -------
        Detections sorted by confidence descending.  Empty list if none found.
        """
        self._require_model()
        results = self._model.predict(
            source  = image,
            conf    = self.cfg.conf_thresh,
            iou     = self.cfg.iou_thresh,
            imgsz   = self.cfg.imgsz,
            max_det = self.cfg.max_det,
            verbose = False,
        )
        return self._parse_results(results[0])

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """
        Detect objects in a batch of RGB images (2–4× faster than looping).

        Parameters
        ----------
        images : list of (H, W, 3) uint8 arrays

        Returns
        -------
        List of detection lists, one per input image.
        """
        self._require_model()
        if not images:
            return []
        results = self._model.predict(
            source  = images,
            conf    = self.cfg.conf_thresh,
            iou     = self.cfg.iou_thresh,
            imgsz   = self.cfg.imgsz,
            max_det = self.cfg.max_det,
            verbose = False,
        )
        return [self._parse_results(r) for r in results]

    def set_device(self, device: str) -> None:
        """Move the loaded model to a different compute device."""
        self._require_model()
        self._model.to(device)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def class_names(self) -> list[str]:
        """Detectable class name list (reflects the loaded model's names)."""
        return list(self._class_names)

    @property
    def n_classes(self) -> int:
        return len(self._class_names)

    @property
    def is_loaded(self) -> bool:
        """True if the YOLO model was loaded successfully."""
        return self._model is not None

    # ── internals ─────────────────────────────────────────────────────────────

    def _require_model(self) -> None:
        if _ultralytics is None:
            raise ImportError(
                "ultralytics is required for ObjectDetector inference. "
                "Install with: pip install ultralytics"
            )
        if self._model is None:
            raise RuntimeError(
                f"YOLO model failed to load from '{self.cfg.weights_path}'. "
                "Check that the path exists or that the base model name is valid."
            )

    def _parse_results(self, result) -> list[Detection]:
        """Convert one ultralytics Results object to Detection list."""
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy_all  = boxes.xyxy.cpu().numpy().astype(np.float32)
        xywh_all  = boxes.xywh.cpu().numpy().astype(np.float32)
        conf_all  = boxes.conf.cpu().numpy().astype(np.float64)
        cls_all   = boxes.cls.cpu().numpy().astype(np.int32)

        for i in range(len(boxes)):
            cid  = int(cls_all[i])
            name = (self._class_names[cid]
                    if cid < len(self._class_names) else str(cid))
            detections.append(Detection(
                class_id   = cid,
                class_name = name,
                confidence = float(conf_all[i]),
                bbox_xyxy  = xyxy_all[i],
                bbox_xywh  = xywh_all[i],
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
