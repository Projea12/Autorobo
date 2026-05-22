"""
ar/transforms.py — Rigid-body transforms for the Autorobo camera pipeline.

Coordinate conventions
----------------------
Camera frame  (OpenCV standard)
    X = right
    Y = down
    Z = forward (into scene)

TidyBot base frame
    X = right  (lateral)
    Y = forward (robot drives along +Y at heading=0)
    Z = up

Camera mounting (hardcoded, TidyBot default)
    Position in base frame : [x=0.0, y=0.1, z=1.2] metres
        x=0   — centred, no lateral offset
        y=0.1 — 0.1 m forward of the base origin
        z=1.2 — 1.2 m above the floor

    Orientation : facing forward, no tilt (camera Z aligns with base Y)

Rotation derivation (camera → base)
    cam X (right)   → base  X (right)   :  X_base =  X_cam
    cam Y (down)    → base -Z (up=-down) :  Z_base = -Y_cam
    cam Z (forward) → base  Y (forward)  :  Y_base =  Z_cam

    R_cam_to_base = [[1,  0,  0],
                     [0,  0,  1],
                     [0, -1,  0]]

Usage
-----
    from ar.transforms import T_CAM_TO_BASE

    xyz_base = T_CAM_TO_BASE(xyz_cam)          # single point
    pts_base = T_CAM_TO_BASE.apply(pts_cam)    # (N,3) array
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ── SE(3) rigid transform ─────────────────────────────────────────────────────

@dataclass
class RigidTransform:
    """
    SE(3) rigid-body transform: P_out = R @ P_in + t

    Parameters
    ----------
    R : (3,3) rotation matrix  — orthonormal, det=+1
    t : (3,)  translation vector in the output frame
    """
    R: np.ndarray   # (3,3)
    t: np.ndarray   # (3,)

    def __call__(
        self, point: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Transform a single (x, y, z) point."""
        p = self.R @ np.asarray(point, dtype=float) + self.t
        return (float(p[0]), float(p[1]), float(p[2]))

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Transform an (N, 3) array of points.

        Parameters
        ----------
        points : (N, 3) float array in the input frame

        Returns
        -------
        (N, 3) float array in the output frame
        """
        pts = np.asarray(points, dtype=float)
        return (pts @ self.R.T) + self.t


# ── camera → TidyBot base frame ───────────────────────────────────────────────

#  Rotation: maps camera axes to base axes (see module docstring)
_R_CAM_TO_BASE = np.array(
    [[1,  0,  0],   # X_base =  X_cam
     [0,  0,  1],   # Y_base =  Z_cam  (forward = forward)
     [0, -1,  0]],  # Z_base = -Y_cam  (up = -down)
    dtype=float,
)

#  Camera position in base frame [right, forward, up] metres
_T_CAM_IN_BASE = np.array([0.0, 0.1, 1.2], dtype=float)

T_CAM_TO_BASE = RigidTransform(R=_R_CAM_TO_BASE, t=_T_CAM_IN_BASE)
"""
Fixed camera-to-base transform for TidyBot.

Transform a point from camera frame to robot base frame:

    (X, Y, Z)_base = T_CAM_TO_BASE((X, Y, Z)_cam)

For a forward-facing camera with no tilt, an object directly on the
optical axis at depth d maps to Y_base ≈ d  (forward distance from robot).
"""

# Inverse: base frame → camera frame  (P_cam = R^T · (P_base − t))
T_BASE_TO_CAM = RigidTransform(
    R = _R_CAM_TO_BASE.T,
    t = -_R_CAM_TO_BASE.T @ _T_CAM_IN_BASE,
)


def project_to_pixel(
    xyz_base: Tuple[float, float, float],
    intrinsics,
) -> Optional[Tuple[int, int]]:
    """
    Project a 3D point in robot base frame to a 2D image pixel.

    Pipeline:
        1. base frame → camera frame  (T_BASE_TO_CAM)
        2. pin-hole projection:  u = fx·X/Z + cx,  v = fy·Y/Z + cy

    Returns None if the point is behind the camera (Z_cam ≤ 0).

    Parameters
    ----------
    xyz_base   : (3,) point in robot base frame
    intrinsics : object with .fx .fy .cx .cy

    Returns
    -------
    (u, v) pixel coordinates, or None if behind camera
    """
    xyz_cam = T_BASE_TO_CAM(xyz_base)
    Z = xyz_cam[2]
    if Z <= 1e-4:
        return None
    u = int(round(intrinsics.fx * xyz_cam[0] / Z + intrinsics.cx))
    v = int(round(intrinsics.fy * xyz_cam[1] / Z + intrinsics.cy))
    return (u, v)
