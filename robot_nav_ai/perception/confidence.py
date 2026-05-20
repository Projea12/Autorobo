"""
perception/confidence.py — Scene confidence aggregator.

Combines detection, depth, and segmentation signals into per-object and
scene-level confidence scores.  These scores feed the UncertaintyGate to
decide whether to act, gather more data, or escalate to human review.

Scores are normalised to [0, 1] and combined as a weighted average:
    combined = (w_det * det_score + w_depth * depth_score + w_seg * seg_score)
               / (w_det + w_depth + w_seg)

Component definitions
─────────────────────
det_score    : YOLO confidence, clipped to [0, 1]
depth_score  : 0.5 × depth_coverage + 0.5 × depth_reliability
                depth_coverage   = min(n_points / depth_n_pts_max, 1)
                depth_reliability = 1 − min(std_z / depth_std_max, 1)
               depth_score = 0.0 when projection is None or n_points == 0
seg_score    : mask_area / bbox_area, clipped to [0, 1]
               seg_score = 0.0 when detection.mask is None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from perception.detector import Detection
from perception.depth_projector import ProjectionResult

log = logging.getLogger(__name__)


# ── per-object confidence ─────────────────────────────────────────────────────

@dataclass
class ObjectConfidence:
    """
    Per-object confidence breakdown.

    Fields
    ------
    detection_score : YOLO confidence in [0, 1]
    depth_score     : depth coverage × reliability in [0, 1]
    seg_score       : SAM mask quality in [0, 1]
    combined        : weighted aggregate of the three components in [0, 1]
    detection       : originating Detection object
    projection      : originating ProjectionResult, or None
    """
    detection_score: float
    depth_score:     float
    seg_score:       float
    combined:        float
    detection:       Detection
    projection:      Optional[ProjectionResult] = None

    def __repr__(self) -> str:
        return (f"ObjectConfidence({self.detection.class_name!r}, "
                f"combined={self.combined:.3f}, "
                f"det={self.detection_score:.2f}, "
                f"depth={self.depth_score:.2f}, "
                f"seg={self.seg_score:.2f})")


# ── scene confidence ──────────────────────────────────────────────────────────

@dataclass
class SceneConfidence:
    """
    Aggregated confidence across all detected objects in one frame.

    Fields
    ------
    objects      : list of per-object ObjectConfidence instances
    global_score : mean of all objects' combined scores; 0.0 if no objects
    n_objects    : number of objects (== len(objects))
    """
    objects:      list[ObjectConfidence]
    global_score: float
    n_objects:    int

    def __repr__(self) -> str:
        return (f"SceneConfidence(global={self.global_score:.3f}, "
                f"n={self.n_objects})")


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AggregatorConfig:
    """
    Weights and scaling parameters for SceneAggregator.

    w_detection     : weight assigned to YOLO detection confidence
    w_depth         : weight assigned to depth quality score
    w_seg           : weight assigned to segmentation mask quality
    depth_n_pts_max : n_points at or above this value saturates coverage = 1.0
    depth_std_max   : depth std (metres) at or above this → reliability = 0.0
    """
    w_detection:     float = 0.4
    w_depth:         float = 0.4
    w_seg:           float = 0.2
    depth_n_pts_max: int   = 500
    depth_std_max:   float = 0.5


# ── aggregator ────────────────────────────────────────────────────────────────

class SceneAggregator:
    """
    Combines detection, depth, and segmentation scores into scene confidence.

    Parameters
    ----------
    cfg : AggregatorConfig
    """

    def __init__(self, cfg: AggregatorConfig = AggregatorConfig()) -> None:
        self.cfg = cfg

    # ── public API ────────────────────────────────────────────────────────────

    def aggregate(
        self,
        detections: list[Detection],
        projections: Optional[list[ProjectionResult]] = None,
    ) -> SceneConfidence:
        """
        Compute ObjectConfidence for each detection and return SceneConfidence.

        Parameters
        ----------
        detections  : YOLO detections; may have .mask set by SAMSegmentor
        projections : one ProjectionResult per detection (parallel list), or
                      None — when None all depth scores are 0.0

        Returns
        -------
        SceneConfidence with per-object breakdown and global_score.

        Raises
        ------
        ValueError : if projections is provided but length != len(detections).
        """
        if projections is not None and len(projections) != len(detections):
            raise ValueError(
                f"len(projections)={len(projections)} != "
                f"len(detections)={len(detections)}"
            )

        objects: list[ObjectConfidence] = []
        for i, det in enumerate(detections):
            proj        = projections[i] if projections is not None else None
            det_score   = float(np.clip(det.confidence, 0.0, 1.0))
            depth_score = self._depth_score(proj)
            seg_score   = self._seg_score(det)
            combined    = self._combined(det_score, depth_score, seg_score)
            objects.append(ObjectConfidence(
                detection_score = det_score,
                depth_score     = depth_score,
                seg_score       = seg_score,
                combined        = combined,
                detection       = det,
                projection      = proj,
            ))

        global_score = (float(np.mean([o.combined for o in objects]))
                        if objects else 0.0)
        return SceneConfidence(
            objects      = objects,
            global_score = global_score,
            n_objects    = len(objects),
        )

    # ── sub-scores ────────────────────────────────────────────────────────────

    def _depth_score(self, proj: Optional[ProjectionResult]) -> float:
        """Depth quality: 0.5 × coverage + 0.5 × reliability, both in [0, 1]."""
        if proj is None or proj.n_points == 0:
            return 0.0
        coverage    = min(proj.n_points / self.cfg.depth_n_pts_max, 1.0)
        std_z       = float(proj.std[2])
        reliability = 1.0 - min(std_z / self.cfg.depth_std_max, 1.0)
        return float(np.clip(0.5 * coverage + 0.5 * reliability, 0.0, 1.0))

    def _seg_score(self, det: Detection) -> float:
        """Mask density: mask_area / bbox_area, clipped to [0, 1]."""
        if det.mask is None:
            return 0.0
        bbox_area = det.area
        if bbox_area <= 0.0:
            return 0.0
        return float(np.clip(float(det.mask.sum()) / bbox_area, 0.0, 1.0))

    def _combined(self, det: float, depth: float, seg: float) -> float:
        """Normalised weighted average of the three component scores."""
        cfg     = self.cfg
        total_w = cfg.w_detection + cfg.w_depth + cfg.w_seg
        if total_w <= 0.0:
            return 0.0
        c = (cfg.w_detection * det + cfg.w_depth * depth + cfg.w_seg * seg) / total_w
        return float(np.clip(c, 0.0, 1.0))

    def __repr__(self) -> str:
        return (f"SceneAggregator(w_det={self.cfg.w_detection}, "
                f"w_depth={self.cfg.w_depth}, w_seg={self.cfg.w_seg})")
