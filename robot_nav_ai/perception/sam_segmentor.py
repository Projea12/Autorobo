"""
perception/sam_segmentor.py — SAM 2 segmentor (Meta, 2024).

Two modes:
  prompted  — given a bounding box from YOLOv8, segments that specific object
  automatic — no prompt, segments every object in the frame (for unknown objects)

Runs on MPS (Apple M1). Falls back to CPU if MPS unavailable.

Failure modes handled:
  - Checkpoint missing       → RuntimeError with download instructions
  - MPS unavailable          → CPU fallback, logged
  - Inference failure        → returns None / empty list, never crashes pipeline
  - Empty image              → ValueError
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

_CHECKPOINT   = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "sam2", "sam2_hiera_tiny.pt"
)
_CONFIG       = "configs/sam2/sam2_hiera_t.yaml"
_SCORE_THRESH = 0.5


@dataclass(frozen=True)
class SAM2Config:
    checkpoint:   str   = _CHECKPOINT
    config:       str   = _CONFIG
    score_thresh: float = _SCORE_THRESH
    device:       str   = ""


class SAM2Segmentor:
    """
    SAM 2 segmentor — prompted and automatic modes.

    Prompted mode:  given bbox from YOLOv8, returns precise object mask.
    Automatic mode: no bbox needed, segments everything in the frame.
                    Used for unknown objects YOLOv8 could not classify.
    """

    def __init__(self, cfg: SAM2Config = SAM2Config()) -> None:
        self.cfg         = cfg
        self._predictor  = None
        self._auto_gen   = None
        self._device     = cfg.device or self._auto_device()

        checkpoint = os.path.abspath(cfg.checkpoint)
        if not os.path.exists(checkpoint):
            raise RuntimeError(
                f"SAM2 checkpoint not found at '{checkpoint}'.\n"
                "Download with:\n"
                "  from huggingface_hub import hf_hub_download\n"
                "  hf_hub_download('facebook/sam2-hiera-tiny', "
                "'sam2_hiera_tiny.pt', local_dir='checkpoints/sam2')"
            )

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            model = build_sam2(cfg.config, checkpoint, device=self._device)
            self._predictor = SAM2ImagePredictor(model)
            self._auto_gen  = SAM2AutomaticMaskGenerator(
                model,
                points_per_side        = 16,
                pred_iou_thresh        = cfg.score_thresh,
                stability_score_thresh = cfg.score_thresh,
            )
            log.info("SAM2Segmentor loaded on %s", self._device)

        except Exception as exc:
            raise RuntimeError(f"SAM2Segmentor failed to load: {exc}") from exc

    # ── public API ─────────────────────────────────────────────────────────────

    def segment_from_bbox(
        self,
        image:     np.ndarray,
        bbox_xyxy: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Segment one object from its bounding box.

        Parameters
        ----------
        image     : (H, W, 3) uint8 RGB
        bbox_xyxy : (4,) float32 [x1, y1, x2, y2]

        Returns
        -------
        (H, W) bool mask, or None if score below threshold or inference fails.
        """
        self._validate_image(image)
        try:
            self._predictor.set_image(image)
            masks, scores, _ = self._predictor.predict(
                box              = bbox_xyxy[None].astype(np.float32),
                multimask_output = True,
            )
            best = int(np.argmax(scores))
            if float(scores[best]) < self.cfg.score_thresh:
                return None
            return masks[best].astype(bool)

        except Exception as exc:
            log.warning("SAM2 prompted segmentation failed: %s", exc)
            return None

    def segment_unknown(
        self,
        image: np.ndarray,
    ) -> list[dict]:
        """
        Automatically segment all objects in frame — no prompt needed.

        Used when YOLOv8 confidence is low and unknown objects are present.

        Parameters
        ----------
        image : (H, W, 3) uint8 RGB

        Returns
        -------
        List of dicts, each with keys:
          'mask'       : (H, W) bool
          'score'      : float — predicted IoU quality
          'bbox'       : [x, y, w, h] pixel coordinates
          'area'       : int — mask area in pixels
        Returns empty list on failure.
        """
        self._validate_image(image)
        try:
            results = self._auto_gen.generate(image)
            return [
                {
                    "mask":  r["segmentation"].astype(bool),
                    "score": float(r["predicted_iou"]),
                    "bbox":  r["bbox"],
                    "area":  int(r["area"]),
                }
                for r in results
                if float(r["predicted_iou"]) >= self.cfg.score_thresh
            ]
        except Exception as exc:
            log.warning("SAM2 automatic segmentation failed: %s", exc)
            return []

    # ── internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) RGB image, got shape {image.shape}"
            )

    @staticmethod
    def _auto_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "failed"
        return f"SAM2Segmentor(device='{self._device}', {status})"
