"""
segmenter.py — SAM2 Object Segmenter (Phase 6)

Generates precise pixel-level segmentation masks for detected objects
using Segment Anything Model 2 (SAM2) by Meta AI.

Prompted by bounding boxes from the YOLOv8 detector (box-prompted mode).
Masks are used by GraspEstimator to compute object geometry for grasp planning.

Usage:
    from perception.segmenter import ObjectSegmenter
    from perception.detector import ObjectDetector

    detector = ObjectDetector(cfg.perception.detector)
    segmenter = ObjectSegmenter(cfg.perception.segmenter)

    detections = detector.detect(rgb_image)
    masks = segmenter.segment(rgb_image, detections)
    for det, mask in zip(detections, masks):
        print(det.class_name, mask.shape, mask.sum(), "pixels")
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from perception.detector import Detection

log = logging.getLogger(__name__)


class ObjectSegmenter:
    """
    SAM2-based instance segmenter for robotic manipulation.

    Takes an RGB image and a list of detected bounding boxes (from YOLOv8),
    and returns a binary segmentation mask for each detected object.

    Configuration via DictConfig (from configs/perception/yolo.yaml):
        model_type: "vit_b", "vit_l", or "vit_h"
        checkpoint: path to SAM2 checkpoint
        device: "cpu", "cuda:0", or "mps"
        prompt_mode: "box" (from YOLO) or "points" (click-based)
    """

    def __init__(self, cfg: Any) -> None:
        """
        Initialise and load the SAM2 model.

        Args:
            cfg: DictConfig with segmenter settings.

        TODO: Phase 6 — implement:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = build_sam2(cfg.model_type, cfg.checkpoint)
            self._predictor = SAM2ImagePredictor(sam2_model)
            self._predictor.model.to(cfg.device)
        """
        self.cfg = cfg
        self._predictor = None  # SAM2ImagePredictor — loaded in Phase 6
        log.info("ObjectSegmenter created (model not yet loaded — TODO: Phase 6)")

    def segment(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> list[np.ndarray]:
        """
        Generate segmentation masks for all detected objects.

        Uses bounding boxes from YOLO detections as SAM2 box prompts.
        For each detection, SAM2 produces a binary mask of the object.

        Args:
            image: RGB image as np.ndarray of shape (H, W, 3), dtype uint8.
            detections: List of Detection objects from ObjectDetector.detect().

        Returns:
            List of binary masks, one per detection.
            Each mask is np.ndarray of shape (H, W), dtype bool.
            True pixels belong to the detected object.
            Empty list if detections is empty.

        TODO: Phase 6 — implement:
            self._predictor.set_image(image)
            masks = []
            for det in detections:
                box = det.bbox_xyxy  # SAM2 expects xyxy format
                mask, score, _ = self._predictor.predict(
                    box=box,
                    multimask_output=False,
                )
                masks.append(mask[0].astype(bool))
                det.mask = mask[0].astype(bool)  # attach mask to detection
            return masks
        """
        if not detections:
            return []
        if self._predictor is None:
            raise RuntimeError(
                "ObjectSegmenter model not loaded. TODO: Phase 6 — load SAM2 in __init__."
            )
        raise NotImplementedError(
            "TODO: Phase 6 — implement segment() using SAM2ImagePredictor "
            "with box prompts from YOLO detections."
        )

    def segment_interactive(
        self,
        image: np.ndarray,
        points: np.ndarray,
        point_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a mask from interactive point prompts (click-based).

        Useful for debugging or when bounding box prompts are unavailable.

        Args:
            image: RGB image, shape (H, W, 3).
            points: Click points, shape (N, 2) in [x, y] pixel coords.
            point_labels: Label per point: 1=foreground, 0=background. Shape (N,).

        Returns:
            Binary mask, shape (H, W), dtype bool.

        TODO: Phase 6 — implement using SAM2ImagePredictor.predict(
            point_coords=points, point_labels=point_labels
        )
        """
        raise NotImplementedError(
            "TODO: Phase 6 — implement interactive point-prompted segmentation."
        )
