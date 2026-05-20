"""
ar/floor_detector.py — Automatic floor plane detection from monocular depth.

Pipeline:
    depth map (H×W float32 [0,1])
    → back-project to 3D point cloud (using pinhole camera model)
    → RANSAC plane fit (finds dominant flat surface = floor)
    → floor mask (H×W bool) + plane equation (normal, d)

Usage:
    python ar/floor_detector.py              # live webcam + floor overlay
    python ar/floor_detector.py --no-preview # headless integration mode
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics. MacBook webcam defaults for 640×480."""
    fx: float = 525.0   # focal length x (pixels)
    fy: float = 525.0   # focal length y (pixels)
    cx: float = 320.0   # principal point x
    cy: float = 240.0   # principal point y


@dataclass
class FloorDetectorConfig:
    intrinsics:       CameraIntrinsics = field(default_factory=CameraIntrinsics)
    downsample:       int   = 4       # stride when sampling pixels (speed vs precision)
    ransac_iters:     int   = 120     # number of RANSAC trials
    inlier_thresh:    float = 0.04    # plane distance threshold (normalised units)
    min_inlier_frac:  float = 0.25    # minimum inlier fraction to accept plane
    normal_up_thresh: float = 0.6     # |normal_y| must exceed this → horizontal plane
    depth_min:        float = 0.05    # ignore very close pixels (noise)
    depth_max:        float = 0.95    # ignore very far pixels (background clutter)


@dataclass
class FloorResult:
    found:       bool
    mask:        np.ndarray          # H×W bool — True where floor was detected
    normal:      Optional[np.ndarray] = None   # unit normal vector (3,)
    plane_d:     float               = 0.0     # plane eq: normal·p + d = 0
    inlier_frac: float               = 0.0


# ── detector ──────────────────────────────────────────────────────────────────

class FloorDetector:
    """
    Detects the dominant horizontal floor plane in a normalised depth map.

    The depth map is back-projected into a 3D point cloud using a pinhole
    camera model, then RANSAC fits the best-fitting horizontal plane.

    Parameters
    ----------
    cfg : FloorDetectorConfig
    """

    def __init__(self, cfg: FloorDetectorConfig = FloorDetectorConfig()) -> None:
        self.cfg = cfg
        h, w = 480, 640   # default frame size — updated on first detect() call
        self._pixel_dirs = self._make_pixel_dirs(h, w)
        self._last_hw: Tuple[int, int] = (h, w)

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, depth_norm: np.ndarray) -> FloorResult:
        """
        Detect floor plane from a normalised depth map.

        Parameters
        ----------
        depth_norm : H×W float32, values in [0, 1]

        Returns
        -------
        FloorResult
        """
        h, w = depth_norm.shape[:2]
        if (h, w) != self._last_hw:
            self._pixel_dirs = self._make_pixel_dirs(h, w)
            self._last_hw = (h, w)

        pts, idx = self._backproject(depth_norm)

        if len(pts) < 50:
            return FloorResult(found=False, mask=np.zeros((h, w), bool))

        normal, d, inlier_mask = self._ransac(pts)

        if normal is None:
            return FloorResult(found=False, mask=np.zeros((h, w), bool))

        # Build per-pixel floor mask
        floor_mask = self._build_mask(depth_norm, normal, d, h, w)

        return FloorResult(
            found       = True,
            mask        = floor_mask,
            normal      = normal,
            plane_d     = d,
            inlier_frac = inlier_mask.mean(),
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _make_pixel_dirs(self, h: int, w: int) -> np.ndarray:
        """Pre-compute normalised ray directions for every pixel. Shape: H×W×3."""
        cfg = self.cfg.intrinsics
        ys, xs = np.mgrid[0:h, 0:w]
        dirs = np.stack([
            (xs - cfg.cx) / cfg.fx,
            (ys - cfg.cy) / cfg.fy,
            np.ones((h, w), np.float32),
        ], axis=-1).astype(np.float32)
        return dirs  # H×W×3

    def _backproject(self, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a subset of pixels and back-project to 3D."""
        cfg = self.cfg
        s = cfg.downsample
        depth_s  = depth[::s, ::s]
        dirs_s   = self._pixel_dirs[::s, ::s]  # H'×W'×3

        mask = (depth_s > cfg.depth_min) & (depth_s < cfg.depth_max)
        flat_depth = depth_s[mask]
        flat_dirs  = dirs_s[mask]               # N×3

        pts = flat_dirs * flat_depth[:, None]   # scale rays by depth
        idx = np.where(mask)
        return pts, idx

    def _ransac(
        self, pts: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float, np.ndarray]:
        """RANSAC plane fitting. Returns (normal, d, inlier_bool_array)."""
        cfg = self.cfg
        n = len(pts)
        best_normal     = None
        best_d          = 0.0
        best_inliers    = np.zeros(n, bool)
        best_count      = 0

        rng = np.random.default_rng(42)

        for _ in range(cfg.ransac_iters):
            # Sample 3 random points
            idx = rng.choice(n, 3, replace=False)
            p0, p1, p2 = pts[idx]

            # Plane normal from cross product
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8:
                continue
            normal = normal / norm_len
            d = -float(normal @ p0)

            # Count inliers
            dist = np.abs(pts @ normal + d)
            inliers = dist < cfg.inlier_thresh

            # Prefer planes whose normal points roughly upward (Y axis in cam coords)
            # In OpenCV camera frame: Y points down, so floor normal ≈ (0, -1, 0)
            # We accept |normal[1]| > threshold (horizontal plane)
            if abs(normal[1]) < cfg.normal_up_thresh:
                continue

            count = inliers.sum()
            if count > best_count:
                best_count   = count
                best_normal  = normal.copy()
                best_d       = d
                best_inliers = inliers

        if best_normal is None:
            return None, 0.0, np.zeros(n, bool)

        inlier_frac = best_count / n
        if inlier_frac < cfg.min_inlier_frac:
            return None, 0.0, best_inliers

        # Refit on all inliers for stability
        inlier_pts = pts[best_inliers]
        centroid   = inlier_pts.mean(axis=0)
        _, _, Vt   = np.linalg.svd(inlier_pts - centroid)
        refined_normal = Vt[-1]
        if refined_normal[1] > 0:          # ensure normal points upward (−Y cam)
            refined_normal = -refined_normal
        refined_d = float(-refined_normal @ centroid)

        return refined_normal, refined_d, best_inliers

    def _build_mask(
        self,
        depth: np.ndarray,
        normal: np.ndarray,
        d: float,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Mark every pixel whose 3D point is within inlier_thresh of the plane."""
        # Back-project all pixels (not just sampled)
        pts_all = self._pixel_dirs * depth[:, :, None]   # H×W×3
        dist    = np.abs(pts_all @ normal + d)            # H×W
        valid   = (depth > self.cfg.depth_min) & (depth < self.cfg.depth_max)
        return (dist < self.cfg.inlier_thresh) & valid


# ── preview helper ────────────────────────────────────────────────────────────

def overlay_floor(frame_bgr: np.ndarray, result: FloorResult) -> np.ndarray:
    """Draw floor mask as a green overlay on the BGR frame."""
    out = frame_bgr.copy()
    if result.found:
        green = np.zeros_like(out)
        green[result.mask] = (0, 200, 80)
        out = cv2.addWeighted(out, 0.65, green, 0.35, 0)
        label = (f"Floor detected  |  "
                 f"inliers={result.inlier_frac:.0%}  |  "
                 f"normal=({result.normal[0]:.2f},{result.normal[1]:.2f},{result.normal[2]:.2f})")
    else:
        label = "Floor: not detected"
    cv2.putText(out, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autorobo floor detector — live webcam preview"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ar.depth_estimator import DepthConfig, DepthEstimator

    cfg       = DepthConfig(camera_index=args.camera)
    estimator = DepthEstimator(cfg)
    detector  = FloorDetector()

    if args.no_preview:
        print("FloorDetector ready.")
        return

    estimator.open_camera()
    interval   = 1.0 / cfg.target_fps
    last_depth = np.zeros((cfg.frame_height, cfg.frame_width), np.float32)
    last_result: FloorResult = FloorResult(
        found=False,
        mask=np.zeros((cfg.frame_height, cfg.frame_width), bool),
    )
    last_time   = 0.0
    fps_display = 0.0

    print("\n[FloorDetector] Live preview — green = detected floor.  Q to quit.\n")

    try:
        while True:
            frame = estimator.read_frame()
            if frame is None:
                break

            now = time.perf_counter()
            if now - last_time >= interval:
                t0          = time.perf_counter()
                last_depth  = estimator.estimate(frame)
                last_result = detector.detect(last_depth)
                elapsed     = time.perf_counter() - t0
                fps_display = 1.0 / elapsed if elapsed > 0 else 0.0
                last_time   = now

            # Depth colourised
            depth_u8    = (last_depth * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

            # Floor overlay on webcam frame
            floor_frame = overlay_floor(frame, last_result)

            # FPS
            fps_label = f"FPS: {fps_display:.1f}  |  Q=quit"
            cv2.putText(floor_frame, fps_label, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(depth_color, fps_label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            combined = np.hstack([floor_frame, depth_color])
            cv2.imshow("Autorobo — Floor Detection (Q to quit)", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        estimator.close_camera()
        cv2.destroyAllWindows()
        print("[FloorDetector] Stopped.")


if __name__ == "__main__":
    main()
