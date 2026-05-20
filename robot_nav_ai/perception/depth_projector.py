"""
perception/depth_projector.py — Back-project 2-D bounding boxes to 3-D positions.

Converts (Detection bbox or mask, RGBDFrame depth map, camera intrinsics K)
into a 3-D position + uncertainty in the camera frame.

Camera frame convention: X right, Y down, Z forward (into scene).

Typical usage
─────────────
    from perception.depth_projector import DepthProjector, ProjectorConfig
    from perception.detector import ObjectDetector
    from perception.rgbd_camera import RGBDCamera

    frame      = cam.capture(data)
    detections = detector.detect(frame.rgb)
    projector  = DepthProjector()
    results    = projector.annotate_detections(detections, frame)
    for det, res in zip(detections, results):
        print(det.class_name, det.position_3d, "±", res.std)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from perception.rgbd_camera import RGBDFrame
from perception.detector import Detection

log = logging.getLogger(__name__)


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class ProjectionResult:
    """
    3-D position estimate for one detected object.

    Fields
    ------
    xyz      : (3,) float32, camera frame [X right, Y down, Z forward], metres
    std      : (3,) float32, 1-sigma uncertainty per axis, metres
    n_points : number of valid depth pixels used
    method   : "bbox_median", "mask_median", or "center" (single-pixel fallback)
    """
    xyz:      np.ndarray   # (3,) float32
    std:      np.ndarray   # (3,) float32
    n_points: int
    method:   str

    def __repr__(self) -> str:
        x, y, z = self.xyz.tolist()
        return (f"ProjectionResult(xyz=[{x:.3f},{y:.3f},{z:.3f}], "
                f"std={[round(v,4) for v in self.std.tolist()]}, "
                f"n={self.n_points}, method={self.method!r})")


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProjectorConfig:
    """
    Configuration for DepthProjector.

    min_valid_pixels : minimum valid depth pixels required to use bbox/mask median;
                       below this threshold falls back to single center-pixel projection.
    z_min            : discard depth readings below this value (metres)
    z_max            : discard depth readings above this value (metres)
    bbox_shrink      : fraction [0, 0.5) to inset each edge of the bbox before
                       sampling — reduces contamination from background depth at
                       object boundaries.
    """
    min_valid_pixels: int   = 5
    z_min:            float = 0.05
    z_max:            float = 10.0
    bbox_shrink:      float = 0.15


# ── projector ─────────────────────────────────────────────────────────────────

class DepthProjector:
    """
    Back-projects Detection bboxes or SAM masks to 3-D camera-frame positions.

    Parameters
    ----------
    cfg : ProjectorConfig
    """

    def __init__(self, cfg: ProjectorConfig = ProjectorConfig()) -> None:
        self.cfg = cfg

    # ── public API ────────────────────────────────────────────────────────────

    def project(
        self,
        detection: Detection,
        frame: RGBDFrame,
        mask: Optional[np.ndarray] = None,
    ) -> ProjectionResult:
        """
        Project one detection to 3-D.

        Parameters
        ----------
        detection : Detection with bbox_xyxy in image pixel coordinates
        frame     : RGBDFrame supplying the depth map and camera intrinsics K
        mask      : optional (H, W) bool — if provided and non-empty, use masked
                    pixels instead of the bounding box region

        Returns
        -------
        ProjectionResult with xyz in camera frame (X right, Y down, Z forward).
        """
        if mask is not None and mask.any():
            return self._project_masked(detection, frame, mask)
        return self._project_bbox(detection, frame)

    def project_batch(
        self,
        detections: list[Detection],
        frame: RGBDFrame,
    ) -> list[ProjectionResult]:
        """
        Project a list of detections; uses detection.mask when available.

        Returns a list of ProjectionResult, one per detection, preserving order.
        """
        if not detections:
            return []
        return [self.project(d, frame, mask=d.mask) for d in detections]

    def annotate_detections(
        self,
        detections: list[Detection],
        frame: RGBDFrame,
    ) -> list[ProjectionResult]:
        """
        Project each detection and set detection.position_3d in-place.

        Only sets position_3d when n_points > 0 (valid depth was found).

        Returns
        -------
        List of ProjectionResult, one per detection.
        """
        results = self.project_batch(detections, frame)
        for det, res in zip(detections, results):
            if res.n_points > 0:
                det.position_3d = res.xyz
        return results

    # ── internals ─────────────────────────────────────────────────────────────

    def _project_bbox(self, det: Detection, frame: RGBDFrame) -> ProjectionResult:
        x1, y1, x2, y2 = det.bbox_xyxy.astype(np.float64)
        H, W = frame.depth.shape

        sx = (x2 - x1) * self.cfg.bbox_shrink
        sy = (y2 - y1) * self.cfg.bbox_shrink
        c_x1 = int(np.clip(x1 + sx, 0, W - 1))
        c_y1 = int(np.clip(y1 + sy, 0, H - 1))
        c_x2 = int(np.clip(x2 - sx, c_x1 + 1, W))
        c_y2 = int(np.clip(y2 - sy, c_y1 + 1, H))

        patch = frame.depth[c_y1:c_y2, c_x1:c_x2]
        valid = (patch > self.cfg.z_min) & (patch < self.cfg.z_max) & np.isfinite(patch)
        n_valid = int(valid.sum())

        if n_valid >= self.cfg.min_valid_pixels:
            d_vals = patch[valid].astype(np.float64)
            z      = float(np.median(d_vals))
            z_std  = float(np.std(d_vals)) if len(d_vals) > 1 else 0.0
            cx_pix = (x1 + x2) / 2.0
            cy_pix = (y1 + y2) / 2.0
            xyz, std = self._backproject(cx_pix, cy_pix, z, z_std, frame.K)
            return ProjectionResult(xyz=xyz, std=std, n_points=n_valid,
                                    method="bbox_median")

        return self._center_pixel_fallback(det, frame)

    def _project_masked(
        self, det: Detection, frame: RGBDFrame, mask: np.ndarray
    ) -> ProjectionResult:
        d_masked = frame.depth[mask]
        valid    = (d_masked > self.cfg.z_min) & (d_masked < self.cfg.z_max) & np.isfinite(d_masked)
        n_valid  = int(valid.sum())

        if n_valid >= self.cfg.min_valid_pixels:
            d_vals = d_masked[valid].astype(np.float64)
            z      = float(np.median(d_vals))
            z_std  = float(np.std(d_vals)) if len(d_vals) > 1 else 0.0
            ys, xs = np.where(mask)
            cx_pix = float(xs.mean())
            cy_pix = float(ys.mean())
            xyz, std = self._backproject(cx_pix, cy_pix, z, z_std, frame.K)
            return ProjectionResult(xyz=xyz, std=std, n_points=n_valid,
                                    method="mask_median")

        return self._center_pixel_fallback(det, frame)

    def _center_pixel_fallback(self, det: Detection, frame: RGBDFrame) -> ProjectionResult:
        H, W   = frame.depth.shape
        cx_pix = float(np.clip((det.bbox_xyxy[0] + det.bbox_xyxy[2]) / 2.0, 0, W - 1))
        cy_pix = float(np.clip((det.bbox_xyxy[1] + det.bbox_xyxy[3]) / 2.0, 0, H - 1))
        z      = float(frame.depth[int(cy_pix), int(cx_pix)])
        if z <= 0.0 or not np.isfinite(z):
            z = 0.0
        xyz, std = self._backproject(cx_pix, cy_pix, z, 0.0, frame.K)
        return ProjectionResult(xyz=xyz, std=std,
                                n_points=(1 if z > 0.0 else 0),
                                method="center")

    @staticmethod
    def _backproject(
        u: float, v: float, z: float, z_std: float, K: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert pixel (u, v) + metric depth z → 3-D point + uncertainty."""
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        xyz = np.array([x, y, z], dtype=np.float32)
        std_x = abs((u - cx) / fx) * z_std
        std_y = abs((v - cy) / fy) * z_std
        std   = np.array([std_x, std_y, z_std], dtype=np.float32)
        return xyz, std

    def __repr__(self) -> str:
        return (f"DepthProjector(bbox_shrink={self.cfg.bbox_shrink}, "
                f"min_valid_pixels={self.cfg.min_valid_pixels})")
