"""
data/synth/annotator.py — Generate YOLO bounding-box annotations from a
rendered synthetic scene.

For each active object the annotator:
  1. Projects the 8 AABB corners to image space using the verified pinhole model.
  2. Discards objects whose centre is behind the camera.
  3. Discards objects whose projected bbox covers less than min_area_frac of the
     image area (too small or too far away to be useful training signal).
  4. Returns a Detection per visible object.

Detection → YOLO label line
────────────────────────────
  "<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>"
  All bbox values normalised to [0, 1] relative to image width/height.

Class mapping
─────────────
  Class IDs are assigned by sorted YCB canonical name order:
    0 → 002_master_chef_can
    1 → 003_cracker_box
    ...
    20 → 061_foam_brick
  This ordering is stable across training runs and matches dataset.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.ycb.registry import REGISTRY
from .camera import CameraConfig, bbox_2d_from_3d, yolo_xywh
from .scene import ObjectSlot

# Sorted canonical names → class IDs (stable mapping)
_NAME_TO_CLASS: dict[str, int] = {
    name: i for i, name in enumerate(REGISTRY.names())
}
CLASS_NAMES: list[str] = REGISTRY.names()


# ── detection result ──────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single object detection for one image."""
    name:       str           # canonical YCB name
    class_id:   int           # YOLO class index
    bbox_xyxy:  np.ndarray    # (4,) pixel coords [u0, v0, u1, v1]
    bbox_yolo:  np.ndarray    # (4,) normalised [cx, cy, w, h]
    depth_m:    float         # depth of object centre in metres

    def yolo_line(self) -> str:
        """YOLO label line: '<class_id> cx cy w h'."""
        cx, cy, w, h = self.bbox_yolo
        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def __repr__(self) -> str:
        return (f"Detection({self.name!r}, cls={self.class_id}, "
                f"depth={self.depth_m:.2f}m, "
                f"bbox=[{self.bbox_xyxy[0]:.0f},{self.bbox_xyxy[1]:.0f},"
                f"{self.bbox_xyxy[2]:.0f},{self.bbox_xyxy[3]:.0f}])")


# ── annotator ─────────────────────────────────────────────────────────────────

class Annotator:
    """
    Converts active ObjectSlots + camera pose into YOLO-format detections.

    Parameters
    ----------
    cam_cfg       : CameraConfig (intrinsics)
    min_area_frac : minimum bbox area as fraction of image area; objects
                    smaller than this are discarded (default 0.002 = 0.2%)
    max_depth     : discard objects further than this many metres
    """

    def __init__(
        self,
        cam_cfg:       CameraConfig,
        min_area_frac: float = 0.002,
        max_depth:     float = 5.0,
    ) -> None:
        self.cam_cfg       = cam_cfg
        self.min_area_frac = min_area_frac
        self.max_depth     = max_depth
        self._img_area     = cam_cfg.image_w * cam_cfg.image_h

    def annotate(
        self,
        active_slots: list[ObjectSlot],
        data,                                      # mujoco.MjData
        cam_pos:      np.ndarray,
        R:            np.ndarray,
        scene,                                     # SynthScene
    ) -> list[Detection]:
        """
        Build a Detection for each visible active slot.

        Parameters
        ----------
        active_slots : slots returned by SynthScene.reset()
        data         : MjData with current qpos
        cam_pos      : (3,) camera world position
        R            : (3, 3) world-to-camera rotation
        scene        : SynthScene (for reading object positions)

        Returns
        -------
        List of Detection, one per visible object (may be shorter than
        len(active_slots) if some objects are behind the camera or too small).
        """
        detections: list[Detection] = []

        for slot in active_slots:
            centre = scene.object_pos(data, slot)

            bbox_xyxy, depth = bbox_2d_from_3d(
                centre       = centre,
                half_extents = slot.half_extents,
                cam_pos      = cam_pos,
                R            = R,
                cfg          = self.cam_cfg,
            )

            # Reject invisible or too-distant objects
            if bbox_xyxy is None:
                continue
            if not np.isfinite(depth) or depth > self.max_depth or depth <= 0:
                continue

            # Reject tiny projections
            w_px = bbox_xyxy[2] - bbox_xyxy[0]
            h_px = bbox_xyxy[3] - bbox_xyxy[1]
            if w_px * h_px < self.min_area_frac * self._img_area:
                continue

            class_id  = _NAME_TO_CLASS.get(slot.name, -1)
            if class_id < 0:
                continue   # object not in global registry — skip

            bbox_yolo = yolo_xywh(bbox_xyxy,
                                   self.cam_cfg.image_w,
                                   self.cam_cfg.image_h)

            detections.append(Detection(
                name      = slot.name,
                class_id  = class_id,
                bbox_xyxy = bbox_xyxy,
                bbox_yolo = bbox_yolo,
                depth_m   = depth,
            ))

        return detections

    # ── class mapping helpers ─────────────────────────────────────────────────

    @staticmethod
    def class_id(name: str) -> int:
        """Return the YOLO class index for a canonical YCB name."""
        return _NAME_TO_CLASS[name]

    @staticmethod
    def class_name(class_id: int) -> str:
        """Return the canonical YCB name for a YOLO class index."""
        return CLASS_NAMES[class_id]

    @staticmethod
    def n_classes() -> int:
        return len(CLASS_NAMES)
