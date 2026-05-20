"""
perception/sam_segmentor.py — SAM-based object segmentation for grasp region estimation.

Wraps Meta's Segment Anything Model (SAM) with bbox-prompted segmentation.
Given detections from ObjectDetector, produces precise per-object masks that
DepthProjector can use for robust 3-D position estimates.

Degrades gracefully if segment_anything is not installed: construction
succeeds in stub mode, inference raises ImportError with a clear message.

Install
───────
    pip install git+https://github.com/facebookresearch/segment-anything.git
    # Download checkpoint (choose one):
    #   vit_b  ~375 MB  sam_vit_b_01ec64.pth   (fastest, lowest quality)
    #   vit_l  ~1.2 GB  sam_vit_l_0b3195.pth
    #   vit_h  ~2.4 GB  sam_vit_h_4b8939.pth   (slowest, best quality)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from perception.detector import Detection

log = logging.getLogger(__name__)

try:
    import segment_anything as _sam_lib   # type: ignore
except ImportError:
    _sam_lib = None  # type: ignore


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SAMConfig:
    """
    Configuration for SAMSegmentor.

    model_type    : SAM architecture — "vit_b", "vit_l", or "vit_h"
    weights_path  : path to the downloaded SAM checkpoint (.pth file)
    device        : "cpu", "cuda", "cuda:0", "mps", or "" (auto-detect)
    score_thresh  : minimum mask quality score to accept in [0, 1];
                    below this threshold segment_from_bbox returns None
                    and segment_detections leaves detection.mask unchanged.
    """
    model_type:   str   = "vit_b"
    weights_path: str   = "sam_vit_b.pth"
    device:       str   = ""
    score_thresh: float = 0.5


# ── segmentor ─────────────────────────────────────────────────────────────────

class SAMSegmentor:
    """
    Bbox-prompted SAM segmentor for YCB manipulation objects.

    Given an RGB image and bbox prompts (from ObjectDetector), returns precise
    per-object segmentation masks.  Masks can then be passed to DepthProjector
    for more accurate 3-D position estimation than bbox-median alone.

    Parameters
    ----------
    cfg : SAMConfig

    Notes
    -----
    • Construction succeeds even if segment_anything is not installed
      (stub mode — segment_from_bbox / segment_detections raise ImportError).
    • If the weights file does not exist, construction succeeds but is_loaded
      will be False and inference will raise RuntimeError.
    • segment_detections encodes the image once and re-uses the embedding for
      all per-detection predict() calls (2–10× faster than encoding per bbox).
    """

    def __init__(self, cfg: SAMConfig = SAMConfig()) -> None:
        self.cfg = cfg
        self._predictor = None

        if _sam_lib is None:
            log.warning(
                "segment_anything not installed — SAMSegmentor in stub mode. "
                "Install with: pip install segment-anything"
            )
            return

        try:
            device = cfg.device or self._auto_device()
            sam    = _sam_lib.sam_model_registry[cfg.model_type](
                checkpoint=cfg.weights_path
            )
            sam.to(device=device)
            self._predictor = _sam_lib.SamPredictor(sam)
            log.info("SAMSegmentor loaded %r on %s", cfg.model_type, device)
        except Exception as exc:
            log.error("Failed to load SAM model: %s", exc)
            self._predictor = None

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True if the SAM model was loaded successfully."""
        return self._predictor is not None

    # ── public API ────────────────────────────────────────────────────────────

    def segment_from_bbox(
        self,
        image: np.ndarray,
        bbox_xyxy: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Segment one object given its axis-aligned bounding box.

        Parameters
        ----------
        image    : (H, W, 3) uint8 RGB image
        bbox_xyxy: (4,) float32 array [x1, y1, x2, y2] in pixel coordinates

        Returns
        -------
        (H, W) bool mask, or None if SAM is unavailable or score < score_thresh.

        Raises
        ------
        ImportError  : if segment_anything is not installed
        RuntimeError : if the SAM weights failed to load
        """
        self._require_predictor()
        self._predictor.set_image(image)
        box           = bbox_xyxy.astype(np.float32)[None]   # (1, 4)
        masks, scores, _ = self._predictor.predict(
            point_coords     = None,
            point_labels     = None,
            box              = box,
            multimask_output = True,
        )
        best = int(np.argmax(scores))
        if float(scores[best]) < self.cfg.score_thresh:
            return None
        return masks[best].astype(bool)

    def segment_detections(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> list[Detection]:
        """
        Run SAM on each detection's bbox and set detection.mask in-place.

        The image is encoded once; each detection runs a separate predict() call
        reusing the cached embedding.  Detections where SAM's best score is below
        score_thresh are left with mask=None.

        Parameters
        ----------
        image      : (H, W, 3) uint8 RGB image
        detections : list of Detection objects (modified in-place)

        Returns
        -------
        The same list, with .mask filled on successful detections.

        Raises
        ------
        ImportError  : if segment_anything is not installed
        RuntimeError : if the SAM weights failed to load
        """
        self._require_predictor()
        if not detections:
            return detections

        self._predictor.set_image(image)

        for det in detections:
            box = det.bbox_xyxy.astype(np.float32)[None]
            try:
                masks, scores, _ = self._predictor.predict(
                    point_coords     = None,
                    point_labels     = None,
                    box              = box,
                    multimask_output = True,
                )
                best = int(np.argmax(scores))
                if float(scores[best]) >= self.cfg.score_thresh:
                    det.mask = masks[best].astype(bool)
            except Exception as exc:
                log.warning("SAM failed for %r: %s", det.class_name, exc)

        return detections

    # ── internals ─────────────────────────────────────────────────────────────

    def _require_predictor(self) -> None:
        if self._predictor is not None:
            return
        if _sam_lib is None:
            raise ImportError(
                "segment_anything is required for SAMSegmentor. "
                "Install with: pip install segment-anything"
            )
        raise RuntimeError(
            f"SAM model failed to load from '{self.cfg.weights_path}'. "
            "Check that the checkpoint path exists."
        )

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "stub"
        return f"SAMSegmentor({self.cfg.model_type!r}, {status})"
