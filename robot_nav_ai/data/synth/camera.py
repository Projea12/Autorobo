"""
data/synth/camera.py — Camera intrinsics, pose, and 3-D projection.

All functions are pure-numpy so they can be unit-tested without MuJoCo.

Coordinate conventions
──────────────────────
  World  : right-handed, Z up, same as MuJoCo default.
  Camera : right-handed, X right, Y up, Z BACKWARD (away from scene).
           Object in front of camera → z_cam < 0  → depth = -z_cam > 0.
  Image  : u = column (left→right), v = row (top→bottom).

Pinhole model
─────────────
  u = fx * (x_cam / depth) + cx
  v = cy - fy * (y_cam / depth)      ← minus because image v goes DOWN

MuJoCo mjvCamera spherical → Cartesian
────────────────────────────────────────
  Derivation from MuJoCo source (engine_vis_interact.c):
    cam_pos = lookat + d * [cos(el)*(-sin(az)),
                            -cos(el)*cos(az),
                            -sin(el)]
  where az, el in radians, el < 0 means camera is above the lookat point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ── intrinsics ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CameraConfig:
    """
    Pinhole camera intrinsics + spherical extrinsic parameters.

    image_w / image_h : rendered resolution (pixels)
    fovy              : vertical field of view in degrees
    lookat            : world-frame point the camera orbits (default origin)
    distance_range    : (min, max) camera distance in metres
    azimuth_range     : (min, max) azimuth in degrees
    elevation_range   : (min, max) elevation in degrees (negative = looking down)
    """
    image_w:         int              = 640
    image_h:         int              = 480
    fovy:            float            = 45.0
    lookat:          tuple            = (0.0, 0.0, 0.10)
    distance_range:  tuple            = (0.90, 2.20)
    azimuth_range:   tuple            = (0.0, 360.0)
    elevation_range: tuple            = (-55.0, -15.0)

    # ── derived intrinsics ────────────────────────────────────────────────────

    @property
    def fy(self) -> float:
        """Vertical focal length in pixels."""
        return (self.image_h / 2.0) / math.tan(math.radians(self.fovy / 2.0))

    @property
    def fx(self) -> float:
        """Horizontal focal length (square pixels)."""
        return self.fy

    @property
    def cx(self) -> float:
        return self.image_w / 2.0

    @property
    def cy(self) -> float:
        return self.image_h / 2.0

    @property
    def K(self) -> np.ndarray:
        """3 × 3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0   ],
        ])

    def sample_pose(self, rng: np.random.Generator) -> dict:
        """
        Sample a random camera pose within the configured ranges.

        Returns a dict with keys: lookat, distance, azimuth, elevation.
        These can be passed directly to a mujoco.MjvCamera instance.
        """
        d  = float(rng.uniform(*self.distance_range))
        az = float(rng.uniform(*self.azimuth_range))
        el = float(rng.uniform(*self.elevation_range))
        # Slight lookat jitter so the camera doesn't always centre on origin
        lk = np.array(self.lookat, dtype=np.float64)
        lk[:2] += rng.uniform(-0.05, 0.05, size=2)
        return dict(lookat=lk, distance=d, azimuth=az, elevation=el)


# ── camera pose ───────────────────────────────────────────────────────────────

def camera_pose_from_spherical(
    lookat:    np.ndarray,
    distance:  float,
    azimuth:   float,
    elevation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute camera position and world-to-camera rotation matrix from MuJoCo
    free-camera spherical parameters.

    Parameters
    ----------
    lookat    : (3,) world-frame target point
    distance  : metres from lookat
    azimuth   : degrees, 0 = along +y axis (MuJoCo convention)
    elevation : degrees, negative = camera above looking down

    Returns
    -------
    pos : (3,) camera position in world frame
    R   : (3, 3) rotation matrix s.t.  p_cam = R @ (p_world - pos)
          rows = [right, up, backward]
          camera Z axis (row 2) points AWAY from the scene.
    """
    az  = math.radians(azimuth)
    el  = math.radians(elevation)

    # Camera position (MuJoCo source formula)
    pos = np.asarray(lookat, dtype=np.float64) + distance * np.array([
        math.cos(el) * (-math.sin(az)),
        -math.cos(el) * math.cos(az),
        -math.sin(el),
    ])

    # Camera orientation
    forward = np.asarray(lookat, dtype=np.float64) - pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        forward = np.array([0.0, 1.0, 0.0])
    else:
        forward /= forward_norm

    world_up = np.array([0.0, 0.0, 1.0])
    right    = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        # Camera looking straight up/down — choose arbitrary right
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= right_norm

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # R rows = [right, up, backward] → p_cam = R @ (p_world - pos)
    R = np.stack([right, up, -forward], axis=0)  # (3, 3)
    return pos, R


# ── projection ────────────────────────────────────────────────────────────────

def project_points(
    points:  np.ndarray,
    cam_pos: np.ndarray,
    R:       np.ndarray,
    cfg:     CameraConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project an array of world-frame 3D points to pixel coordinates.

    Parameters
    ----------
    points  : (N, 3) world-frame points
    cam_pos : (3,) camera position
    R       : (3, 3) world-to-camera rotation (rows = right, up, backward)
    cfg     : CameraConfig for intrinsics

    Returns
    -------
    uv     : (N, 2) pixel coordinates [u=col, v=row] (float)
    depth  : (N,)  depth in metres (positive = in front, ≤ 0 = behind/on camera)
    """
    pts    = np.atleast_2d(points)              # (N, 3)
    local  = pts - cam_pos                       # (N, 3)
    p_cam  = (R @ local.T).T                     # (N, 3)

    depth  = -p_cam[:, 2]                        # backward component negated

    # Avoid division by zero for points behind or on the image plane
    safe_depth = np.where(depth > 1e-4, depth, np.nan)

    u = cfg.fx * (p_cam[:, 0] / safe_depth) + cfg.cx
    v = cfg.cy - cfg.fy * (p_cam[:, 1] / safe_depth)

    return np.column_stack([u, v]), depth


def bbox_2d_from_3d(
    centre:       np.ndarray,
    half_extents: np.ndarray,
    cam_pos:      np.ndarray,
    R:            np.ndarray,
    cfg:          CameraConfig,
) -> tuple[np.ndarray | None, float]:
    """
    Project the 8 corners of a 3D axis-aligned bounding box to image space
    and return the 2D axis-aligned bounding box.

    The object's local AABB corners are axis-aligned (no rotation applied to
    the box itself — we rely on the box approximating the object well enough
    for YOLO training purposes where exact tight boxes are not critical).

    Parameters
    ----------
    centre       : (3,) world-frame centre of the object
    half_extents : (3,) half-widths along x, y, z
    cam_pos / R  : camera extrinsics
    cfg          : camera intrinsics

    Returns
    -------
    bbox : (4,) [u_min, v_min, u_max, v_max] in pixels, or None if invisible
    depth_centre : depth of the object centre in metres
    """
    hx, hy, hz = half_extents

    # 8 corners of AABB (axis-aligned in world frame)
    offsets = np.array([
        [+hx, +hy, +hz], [+hx, +hy, -hz],
        [+hx, -hy, +hz], [+hx, -hy, -hz],
        [-hx, +hy, +hz], [-hx, +hy, -hz],
        [-hx, -hy, +hz], [-hx, -hy, -hz],
    ])
    corners = centre + offsets        # (8, 3)

    uv, depths = project_points(corners, cam_pos, R, cfg)

    # Filter corners that are in front of the camera
    valid = np.isfinite(uv).all(axis=1) & (depths > 0.0)
    if valid.sum() < 3:
        return None, float("nan")

    uv_v = uv[valid]

    # Clip to image bounds
    u_min = float(np.clip(uv_v[:, 0].min(), 0, cfg.image_w - 1))
    u_max = float(np.clip(uv_v[:, 0].max(), 0, cfg.image_w - 1))
    v_min = float(np.clip(uv_v[:, 1].min(), 0, cfg.image_h - 1))
    v_max = float(np.clip(uv_v[:, 1].max(), 0, cfg.image_h - 1))

    if u_max <= u_min or v_max <= v_min:
        return None, float("nan")

    _, (depth_c,) = project_points(
        centre[np.newaxis], cam_pos, R, cfg
    )
    return np.array([u_min, v_min, u_max, v_max]), float(depth_c)


def yolo_xywh(
    bbox_xyxy: np.ndarray,
    image_w:   int,
    image_h:   int,
) -> np.ndarray:
    """
    Convert [u_min, v_min, u_max, v_max] pixel bbox to YOLO normalised format
    [cx, cy, w, h] where all values are in [0, 1].
    """
    u0, v0, u1, v1 = bbox_xyxy
    cx = (u0 + u1) / 2.0 / image_w
    cy = (v0 + v1) / 2.0 / image_h
    w  = (u1 - u0) / image_w
    h  = (v1 - v0) / image_h
    return np.array([cx, cy, w, h])
