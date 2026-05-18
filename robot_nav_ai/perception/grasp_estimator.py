"""
grasp_estimator.py — Grasp Pose Estimator (Phase 6)

Combines object detection (YOLO), segmentation (SAM2), and depth estimation
(DepthAnything v2) to estimate 6D grasp poses for detected objects.

The grasp pose is the target end-effector pose that the arm should move to
in order to grasp the object. It includes:
  - Pre-grasp position: approach position above the object
  - Grasp position: position with fingers around the object
  - Grasp orientation: rotation matrix for optimal grasp approach

Usage:
    from perception.grasp_estimator import GraspEstimator

    estimator = GraspEstimator(cfg.perception)
    grasps = estimator.estimate_grasps(rgb_image, depth_image)
    best_grasp = grasps[0]  # sorted by quality score
    print(best_grasp.position_3d, best_grasp.orientation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.depth import DepthEstimator
from perception.detector import Detection, ObjectDetector
from perception.segmenter import ObjectSegmenter

log = logging.getLogger(__name__)


@dataclass
class GraspCandidate:
    """
    Represents a single grasp candidate pose.

    Attributes:
        detection: The associated Detection object.
        position_3d: 3D grasp position in camera frame, metres. Shape (3,).
        orientation: Grasp orientation as rotation matrix. Shape (3, 3).
        pre_grasp_position: Position to move to before final grasp. Shape (3,).
        quality_score: Grasp quality in [0.0, 1.0]. Higher is better.
        grasp_width: Required gripper opening in metres.
        approach_direction: Unit vector for approach direction. Shape (3,).
    """
    detection: Detection
    position_3d: np.ndarray        # shape (3,)
    orientation: np.ndarray        # shape (3, 3)
    pre_grasp_position: np.ndarray # shape (3,)
    quality_score: float
    grasp_width: float             # metres
    approach_direction: np.ndarray # shape (3,)

    def __repr__(self) -> str:
        return (
            f"GraspCandidate({self.detection.class_name}, "
            f"pos={self.position_3d.tolist()}, "
            f"quality={self.quality_score:.2f}, "
            f"width={self.grasp_width:.3f}m)"
        )


class GraspEstimator:
    """
    Full perception-to-grasp pipeline.

    Chains YOLO detection → SAM2 segmentation → DepthAnything depth →
    3D object pose → grasp candidate generation.

    The grasp candidate generation supports multiple backends:
      - "heuristic": top-down grasp from object centroid (fast, less accurate)
      - "anygrasp": AnyGrasp neural grasp estimator (accurate, requires GPU)

    Configuration via DictConfig (from configs/perception/yolo.yaml):
        grasp_estimator.method: "heuristic" | "anygrasp"
        grasp_estimator.num_grasp_candidates: number of candidates to generate
        grasp_estimator.min_grasp_quality: minimum quality threshold
    """

    def __init__(self, cfg: Any) -> None:
        """
        Initialise the grasp estimator with all perception sub-components.

        Args:
            cfg: DictConfig with full perception settings.

        TODO: Phase 6 — implement:
            self.detector = ObjectDetector(cfg.detector)
            self.segmenter = ObjectSegmenter(cfg.segmenter)
            self.depth_estimator = DepthEstimator(cfg.depth)
            self._camera_intrinsics = self._build_intrinsics(cfg.camera)
        """
        self.cfg = cfg
        self.detector: ObjectDetector | None = None    # loaded in Phase 6
        self.segmenter: ObjectSegmenter | None = None  # loaded in Phase 6
        self.depth_estimator: DepthEstimator | None = None  # loaded in Phase 6
        self._camera_intrinsics: np.ndarray | None = None
        log.info("GraspEstimator created (sub-components not yet loaded — TODO: Phase 6)")

    def estimate_grasps(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray | None = None,
    ) -> list[GraspCandidate]:
        """
        Estimate grasp candidates for all detected objects in an image.

        Full pipeline:
        1. Detect objects with YOLO
        2. Segment each detection with SAM2
        3. Estimate depth (from depth_image if provided, else run DepthAnything)
        4. Back-project mask centroid to 3D
        5. Generate grasp candidates per object
        6. Score and sort candidates by quality

        Args:
            rgb_image: RGB image, shape (H, W, 3), uint8.
            depth_image: Optional depth image, shape (H, W), float32, metres.
                If None, DepthAnything v2 is used to estimate depth.

        Returns:
            List of GraspCandidate objects sorted by quality (best first).
            Empty if no objects detected.

        TODO: Phase 6 — implement full pipeline as described above.
        Target latency: < 200ms on GPU, < 1s on CPU.
        """
        raise NotImplementedError(
            "TODO: Phase 6 — implement estimate_grasps() pipeline: "
            "detect → segment → depth → 3D localise → generate candidates → sort."
        )

    def estimate_grasp_for_object(
        self,
        rgb_image: np.ndarray,
        detection: Detection,
        depth_image: np.ndarray | None = None,
    ) -> list[GraspCandidate]:
        """
        Estimate grasp candidates for a specific, already-detected object.

        Used when the detection has been provided by an upstream component
        (e.g., from world memory or task planner object reference).

        Args:
            rgb_image: RGB image, shape (H, W, 3).
            detection: Pre-computed Detection object.
            depth_image: Optional depth image.

        Returns:
            List of GraspCandidates for this specific object.

        TODO: Phase 6 — implement: segment detection → depth → 3D → candidates.
        """
        raise NotImplementedError(
            "TODO: Phase 6 — implement single-object grasp estimation."
        )

    def _generate_heuristic_grasps(
        self,
        object_centroid_3d: np.ndarray,
        mask: np.ndarray,
        detection: Detection,
    ) -> list[GraspCandidate]:
        """
        Generate top-down grasp candidates using a simple heuristic.

        The heuristic: approach from directly above the object centroid,
        oriented along the object's principal axis (from mask PCA).

        Args:
            object_centroid_3d: 3D centroid in camera frame. Shape (3,).
            mask: Binary object mask. Shape (H, W).
            detection: Associated Detection object.

        Returns:
            List of GraspCandidates (typically 1–3 from the heuristic).

        TODO: Phase 6 — implement:
            1. Compute object orientation via PCA on mask pixels
            2. Generate top-down approach vector (camera -Z direction)
            3. Create pre-grasp position 15cm above centroid
            4. Estimate grasp width from mask bounding box width + depth
        """
        raise NotImplementedError(
            "TODO: Phase 6 — implement heuristic top-down grasp generation."
        )

    def _build_camera_intrinsics(self, cfg: Any) -> np.ndarray:
        """
        Build camera intrinsics matrix K from config.

        Args:
            cfg: Camera config with fx, fy, cx, cy.

        Returns:
            3x3 intrinsics matrix K.
        """
        return np.array([
            [cfg.fx,    0.0, cfg.cx],
            [0.0,    cfg.fy, cfg.cy],
            [0.0,       0.0,    1.0],
        ], dtype=np.float32)
