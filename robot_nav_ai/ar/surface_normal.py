"""
ar/surface_normal.py — Surface normal estimation from a metric depth map.

Given a depth map Z(u, v) and camera intrinsics, the 3-D point at any
pixel is:

    P(u, v) = ( (u-cx)*Z/fx,  (v-cy)*Z/fy,  Z )

Two tangent vectors are the partial derivatives of P with respect to u and v:

    T_u = dP/du = ( Z/fx + (u-cx)*gz_u/fx,   (v-cy)*gz_u/fy,   gz_u )
    T_v = dP/dv = ( (u-cx)*gz_v/fx,   Z/fy + (v-cy)*gz_v/fy,   gz_v )

where gz_u = dZ/du  and  gz_v = dZ/dv  (central-difference depth gradients).

Surface normal:
    n     = T_v × T_u          (cross order → points toward camera)
    n_hat = n / |n|

Usage
-----
    from ar.surface_normal import estimate_normal, normal_map

    n = estimate_normal(depth_map, u=180, v=300, intrinsics=K)
    # n is a (3,) unit vector in camera frame
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ── single-point normal ───────────────────────────────────────────────────────

def estimate_normal(
    depth_map: np.ndarray,
    u: int,
    v: int,
    intrinsics,
) -> np.ndarray:
    """
    Estimate the surface normal at pixel (u, v) in camera frame.

    Computes dZ/du and dZ/dv using central differences, builds 3-D tangent
    vectors from the pin-hole Jacobian, and returns their cross product
    (normalised, oriented toward the camera).

    Parameters
    ----------
    depth_map  : H×W float32 depth map in metres (from DepthAnything V2 Metric)
    u, v       : pixel coordinates (column, row)
    intrinsics : object with .fx .fy .cx .cy

    Returns
    -------
    n_hat : (3,) float64 unit vector in camera frame.
            Camera frame: X=right, Y=down, Z=forward.
            A flat horizontal table returns approximately [0, -1, 0]  (up in world).
            A vertical wall facing the camera returns approximately [0, 0, -1].
    """
    H, W = depth_map.shape[:2]
    u = int(np.clip(u, 1, W - 2))
    v = int(np.clip(v, 1, H - 2))

    Z    = float(depth_map[v, u])
    if Z < 1e-4:
        return np.array([0.0, -1.0, 0.0])   # invalid depth → fallback up

    # Central differences: dZ/du and dZ/dv
    gz_u = (float(depth_map[v,     u + 1]) - float(depth_map[v,     u - 1])) / 2.0
    gz_v = (float(depth_map[v + 1, u    ]) - float(depth_map[v - 1, u    ])) / 2.0

    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    # 3-D tangent vectors (Jacobian of pin-hole back-projection)
    T_u = np.array([
        Z / fx + (u - cx) * gz_u / fx,
        (v - cy) * gz_u / fy,
        gz_u,
    ], dtype=np.float64)

    T_v = np.array([
        (u - cx) * gz_v / fx,
        Z / fy + (v - cy) * gz_v / fy,
        gz_v,
    ], dtype=np.float64)

    # T_v × T_u → normal oriented toward the camera
    n    = np.cross(T_v, T_u)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.array([0.0, -1.0, 0.0])   # degenerate → fallback up
    return n / norm


# ── full normal map (optional, for visualisation) ─────────────────────────────

def normal_map(
    depth_map: np.ndarray,
    intrinsics,
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Compute a dense surface-normal map from a depth map.

    Applies a small Gaussian blur first to reduce depth noise, then
    uses Sobel gradients for efficiency.

    Parameters
    ----------
    depth_map  : H×W float32 depth map in metres
    intrinsics : object with .fx .fy .cx .cy
    blur_ksize : Gaussian blur kernel size (must be odd; 0 = no blur)

    Returns
    -------
    normals : H×W×3 float32 array, each pixel is a unit normal in camera frame.
              Values in [-1, 1].
    """
    if blur_ksize > 0:
        depth_map = cv2.GaussianBlur(depth_map, (blur_ksize, blur_ksize), 0)

    H, W = depth_map.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    # Pixel-grid coordinate maps
    uu = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)   # (H,W)
    vv = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)   # (H,W)
    Z  = depth_map.astype(np.float64)

    # Sobel gradients (more noise-robust than raw central differences)
    gz_u = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gz_v = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3) / 8.0

    # Tangent vectors at every pixel
    Tu_x = Z / fx + (uu - cx) * gz_u / fx
    Tu_y = (vv - cy) * gz_u / fy
    Tu_z = gz_u

    Tv_x = (uu - cx) * gz_v / fx
    Tv_y = Z / fy + (vv - cy) * gz_v / fy
    Tv_z = gz_v

    # Cross product T_v × T_u at every pixel
    nx = Tv_y * Tu_z - Tv_z * Tu_y
    ny = Tv_z * Tu_x - Tv_x * Tu_z
    nz = Tv_x * Tu_y - Tv_y * Tu_x

    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    norm = np.where(norm < 1e-9, 1.0, norm)   # avoid div-by-zero

    normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    return normals.astype(np.float32)


# ── visualisation helper ──────────────────────────────────────────────────────

def draw_normal(
    frame: np.ndarray,
    u: int,
    v: int,
    n_hat: np.ndarray,
    length: int = 40,
    colour: Tuple[int, int, int] = (0, 200, 255),
) -> np.ndarray:
    """
    Draw the surface normal as an arrow on the frame.

    Projects the 3-D normal into the image plane (ignores Z component)
    and draws an arrow from (u, v) in that direction.
    """
    dx = int(n_hat[0] * length)
    dy = int(n_hat[1] * length)
    cv2.arrowedLine(frame, (u, v), (u + dx, v + dy),
                    colour, 2, tipLength=0.3)
    cv2.circle(frame, (u, v), 3, colour, -1)
    return frame
