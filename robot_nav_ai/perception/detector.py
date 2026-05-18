"""
detector.py — YOLOv8 Object Detector (Phase 6)

Provides real-time object detection on RGB images using YOLOv8.
Detects YCB objects and general household items relevant to manipulation tasks.

The full implementation (Phase 6) will:
- Load YOLOv8n/s/m checkpoint (configurable via configs/perception/yolo.yaml)
- Accept BGR or RGB numpy images
- Return Detection objects with class, confidence, bounding box, and 3D position
- Support fine-tuning on YCB-Video dataset for domain adaptation

Usage:
    from perception.detector import ObjectDetector

    detector = ObjectDetector(cfg.perception.detector)
    detections = detector.detect(rgb_image)
    for det in detections:
        print(det.class_name, det.confidence, det.bbox_xyxy)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    """
    Represents a single object detection result.

    Attributes:
        class_id: Integer class ID from the detector.
        class_name: Human-readable class name (e.g. "025_mug").
        confidence: Detection confidence score in [0.0, 1.0].
        bbox_xyxy: Bounding box [x1, y1, x2, y2] in pixel coordinates.
        bbox_xywh: Bounding box [cx, cy, w, h] in pixel coordinates.
        mask: Optional binary segmentation mask (H, W) bool array.
        position_3d: Optional 3D position (x, y, z) in robot base frame (metres).
        track_id: Optional tracking ID for multi-frame tracking.
    """
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray           # shape (4,), dtype float32
    bbox_xywh: np.ndarray           # shape (4,), dtype float32
    mask: np.ndarray | None = None  # shape (H, W), dtype bool
    position_3d: np.ndarray | None = None  # shape (3,), dtype float32
    track_id: int | None = None

    def __repr__(self) -> str:
        pos = f" pos={self.position_3d}" if self.position_3d is not None else ""
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"bbox={self.bbox_xyxy.tolist()}{pos})"
        )


class ObjectDetector:
    """
    YOLOv8-based object detector for tabletop manipulation.

    Wraps the Ultralytics YOLOv8 model with a stable interface for the
    manipulation pipeline. Handles model loading, inference, and
    post-processing (NMS, class filtering, confidence thresholding).

    Configuration via DictConfig (from configs/perception/yolo.yaml):
        model_path: path to YOLOv8 .pt checkpoint
        confidence_threshold: minimum detection confidence
        iou_threshold: NMS IoU threshold
        device: "cpu", "cuda:0", or "mps"
        classes: list of class names to detect (None = all)
    """

    def __init__(self, cfg: Any) -> None:
        """
        Initialise and load the YOLOv8 model.

        Args:
            cfg: DictConfig with detector settings (from perception.detector).

        TODO: Phase 6 — implement:
            from ultralytics import YOLO
            model_path = cfg.fine_tuned_path if cfg.fine_tuned else cfg.model_path
            self._model = YOLO(model_path)
            self._model.to(cfg.device)
            log.info(f"YOLOv8 loaded from {model_path} on {cfg.device}")
        """
        self.cfg = cfg
        self._model = None  # ultralytics.YOLO — loaded in Phase 6
        self._class_names: list[str] = []
        log.info("ObjectDetector created (model not yet loaded — TODO: Phase 6)")

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Run object detection on a single RGB image.

        Args:
            image: RGB image as np.ndarray of shape (H, W, 3), dtype uint8.
                Accepts both HWC (standard) and CHW format.

        Returns:
            List of Detection objects, sorted by confidence (descending).
            Empty list if no objects detected above threshold.

        TODO: Phase 6 — implement:
            results = self._model.predict(
                source=image,
                conf=self.cfg.confidence_threshold,
                iou=self.cfg.iou_threshold,
                max_det=self.cfg.max_detections,
                verbose=False,
            )
            detections = []
            for box in results[0].boxes:
                det = Detection(
                    class_id=int(box.cls),
                    class_name=self._class_names[int(box.cls)],
                    confidence=float(box.conf),
                    bbox_xyxy=box.xyxy[0].cpu().numpy(),
                    bbox_xywh=box.xywh[0].cpu().numpy(),
                )
                detections.append(det)
            return sorted(detections, key=lambda d: d.confidence, reverse=True)
        """
        if self._model is None:
            raise RuntimeError(
                "ObjectDetector model not loaded. "
                "Call ObjectDetector(cfg) to load the model first. "
                "TODO: Phase 6 — load model in __init__."
            )
        raise NotImplementedError(
            "TODO: Phase 6 — implement detect() using ultralytics YOLO.predict()."
        )

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run detection on a batch of images for efficiency.

        Args:
            images: List of RGB images, each shape (H, W, 3).

        Returns:
            List of detection lists, one per input image.

        TODO: Phase 6 — use YOLO batch inference for 2-4× speedup:
            results = self._model.predict(source=images, ...)
        """
        raise NotImplementedError(
            "TODO: Phase 6 — implement batch detection using YOLO batch inference."
        )

    def set_device(self, device: str) -> None:
        """
        Move the model to a different compute device.

        Args:
            device: "cpu", "cuda:0", "cuda:1", or "mps".

        TODO: Phase 6 — implement:
            self._model.to(device)
            self.cfg.device = device
        """
        raise NotImplementedError(
            f"TODO: Phase 6 — implement set_device({device!r})."
        )

    @property
    def class_names(self) -> list[str]:
        """Return the list of detectable class names."""
        return self._class_names
