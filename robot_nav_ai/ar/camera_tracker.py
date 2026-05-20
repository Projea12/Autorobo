"""
ar/camera_tracker.py — Monocular visual odometry for AR camera tracking.

Tracks camera movement frame-to-frame using:
  • Shi-Tomasi corners + Lucas-Kanade optical flow  (fast, robust)
  • Essential matrix + RANSAC                        (removes bad matches)
  • Depth map scaling                                (resolves monocular scale)
  • Exponential pose smoothing                       (reduces jitter)

Output: 4×4 world-to-camera pose matrix updated every frame.

Usage:
    python ar/camera_tracker.py              # live webcam preview
    python ar/camera_tracker.py --no-preview # headless
"""

from __future__ import annotations

import time
import argparse
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0

    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx,      0, self.cx],
            [     0, self.fy, self.cy],
            [     0,      0,      1],
        ], dtype=np.float64)


@dataclass
class TrackerConfig:
    intrinsics:   CameraIntrinsics = field(default_factory=CameraIntrinsics)

    # Feature detection
    max_features:  int   = 300
    min_features:  int   = 60    # reinitialise if tracking drops below this

    # Lucas-Kanade optical flow
    lk_win_size:   tuple = (21, 21)
    lk_max_level:  int   = 3
    lk_max_iter:   int   = 30
    lk_eps:        float = 0.01

    # RANSAC essential matrix
    ransac_prob:   float = 0.999
    ransac_thresh: float = 1.0   # pixels

    # Depth-based scale estimation
    depth_scale:   float = 3.0   # multiply normalised depth to get rough metres
    depth_min:     float = 0.05
    depth_max:     float = 0.95

    # Pose smoothing (exponential moving average on translation)
    smooth_alpha:  float = 0.5   # 0 = frozen, 1 = raw

    # Maximum translation per frame before we assume a bad estimate
    max_delta_t:   float = 0.3   # metres


@dataclass
class TrackResult:
    pose:          np.ndarray          # 4×4 world pose (float64)
    n_features:    int                 # active feature count
    status:        str                 # "tracking" | "reinit" | "init"
    delta_t:       np.ndarray          # translation since last frame (3,)


# ── tracker ───────────────────────────────────────────────────────────────────

class CameraTracker:
    """
    Frame-to-frame monocular camera tracker.

    Call update(frame_bgr, depth_norm) every frame.
    Returns a TrackResult with the current camera pose.

    The pose matrix P maps world points to camera space:
        p_cam = P @ p_world_homogeneous

    On first call the camera is placed at the origin facing +Z.
    """

    def __init__(self, cfg: TrackerConfig = TrackerConfig()) -> None:
        self.cfg = cfg
        self._K  = cfg.intrinsics.matrix()

        # State
        self._prev_gray:  Optional[np.ndarray] = None
        self._prev_pts:   Optional[np.ndarray] = None   # N×1×2 float32
        self._pose:       np.ndarray = np.eye(4, dtype=np.float64)
        self._smooth_t:   np.ndarray = np.zeros(3, np.float64)
        self._status = "init"

        self._lk_params = dict(
            winSize  = cfg.lk_win_size,
            maxLevel = cfg.lk_max_level,
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                cfg.lk_max_iter,
                cfg.lk_eps,
            ),
        )

    # ── public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame_bgr: np.ndarray,
        depth_norm: Optional[np.ndarray] = None,
    ) -> TrackResult:
        """
        Process one frame and return updated pose.

        Parameters
        ----------
        frame_bgr  : H×W×3 uint8
        depth_norm : H×W float32 in [0,1], or None (disables depth scaling)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_pts is None:
            return self._initialise(gray)

        # Track existing features
        curr_pts, ok, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk_params
        )
        ok = ok.ravel().astype(bool)

        if ok.sum() < self.cfg.min_features:
            return self._initialise(gray)

        prev_good = self._prev_pts[ok]
        curr_good = curr_pts[ok]

        # Essential matrix + RANSAC
        E, inliers = cv2.findEssentialMat(
            curr_good, prev_good, self._K,
            method    = cv2.RANSAC,
            prob      = self.cfg.ransac_prob,
            threshold = self.cfg.ransac_thresh,
        )

        if E is None or inliers is None:
            return self._initialise(gray)

        inliers = inliers.ravel().astype(bool)
        if inliers.sum() < 20:
            return self._initialise(gray)

        _, R, t, _ = cv2.recoverPose(
            E,
            curr_good[inliers],
            prev_good[inliers],
            self._K,
        )
        t = t.ravel()   # (3,)

        # Scale translation using depth at tracked feature locations
        scale = self._estimate_scale(prev_good[inliers], depth_norm)
        t_scaled = t * scale

        # Sanity check — reject huge jumps
        if np.linalg.norm(t_scaled) > self.cfg.max_delta_t:
            t_scaled = t * (self.cfg.max_delta_t / (np.linalg.norm(t_scaled) + 1e-9))

        # Smooth translation
        alpha = self.cfg.smooth_alpha
        self._smooth_t = alpha * t_scaled + (1 - alpha) * self._smooth_t

        # Build delta pose and accumulate
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = R
        delta[:3,  3] = self._smooth_t
        self._pose = delta @ self._pose

        # Refresh features periodically
        n_inliers = int(inliers.sum())
        if n_inliers < self.cfg.min_features * 1.5:
            self._prev_pts = self._detect_features(gray)
        else:
            self._prev_pts = curr_good[inliers].reshape(-1, 1, 2)

        self._prev_gray = gray
        self._status    = "tracking"

        return TrackResult(
            pose       = self._pose.copy(),
            n_features = n_inliers,
            status     = self._status,
            delta_t    = self._smooth_t.copy(),
        )

    def reset(self) -> None:
        """Reset to origin — call when scene changes completely."""
        self._prev_gray = None
        self._prev_pts  = None
        self._pose      = np.eye(4, dtype=np.float64)
        self._smooth_t  = np.zeros(3, np.float64)
        self._status    = "init"

    # ── internals ─────────────────────────────────────────────────────────────

    def _initialise(self, gray: np.ndarray) -> TrackResult:
        self._prev_gray = gray
        self._prev_pts  = self._detect_features(gray)
        self._status    = "reinit" if self._prev_pts is not None else "init"
        return TrackResult(
            pose       = self._pose.copy(),
            n_features = len(self._prev_pts) if self._prev_pts is not None else 0,
            status     = self._status,
            delta_t    = np.zeros(3, np.float64),
        )

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners  = self.cfg.max_features,
            qualityLevel= 0.01,
            minDistance = 10,
            blockSize   = 7,
        )
        return pts   # N×1×2 float32 or None

    def _estimate_scale(
        self,
        pts: np.ndarray,          # N×1×2 or N×2
        depth_norm: Optional[np.ndarray],
    ) -> float:
        if depth_norm is None:
            return self.cfg.depth_scale

        p = pts.reshape(-1, 2).astype(int)
        h, w = depth_norm.shape[:2]
        p[:, 0] = np.clip(p[:, 0], 0, w - 1)
        p[:, 1] = np.clip(p[:, 1], 0, h - 1)

        depths = depth_norm[p[:, 1], p[:, 0]]
        valid  = (depths > self.cfg.depth_min) & (depths < self.cfg.depth_max)
        if valid.sum() < 3:
            return self.cfg.depth_scale

        median_depth = float(np.median(depths[valid]))
        return median_depth * self.cfg.depth_scale


# ── pose helpers ──────────────────────────────────────────────────────────────

def pose_translation(pose: np.ndarray) -> np.ndarray:
    """Extract world-space camera position (x, y, z) from 4×4 pose."""
    return pose[:3, 3]


def draw_tracking_overlay(
    frame: np.ndarray,
    result: TrackResult,
    prev_pts: Optional[np.ndarray] = None,
    curr_pts: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = frame.copy()
    t   = pose_translation(result.pose)
    status_color = (0, 200, 80) if result.status == "tracking" else (0, 140, 255)

    label = (f"{result.status.upper()}  |  "
             f"feats={result.n_features}  |  "
             f"pos=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")
    cv2.putText(out, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 1, cv2.LINE_AA)
    return out


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autorobo camera tracker — live preview"
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
    tracker   = CameraTracker()

    if args.no_preview:
        print("CameraTracker ready.")
        return

    estimator.open_camera()

    h, w      = cfg.frame_height, cfg.frame_width
    last_depth = np.zeros((h, w), np.float32)
    fps_display = 0.0
    lock        = threading.Lock()
    latest: dict = {"frame": None, "result": None}

    def inference_loop() -> None:
        nonlocal last_depth, fps_display
        while not stop_event.is_set():
            with lock:
                frame = latest["frame"]
            if frame is None:
                time.sleep(0.01)
                continue

            t0    = time.perf_counter()
            depth = estimator.estimate(frame)
            result = tracker.update(frame, depth)
            elapsed = time.perf_counter() - t0

            with lock:
                last_depth     = depth
                latest["result"] = result
                fps_display    = 1.0 / elapsed if elapsed > 0 else 0.0

            t = pose_translation(result.pose)
            print(f"[tracker] {result.status:<8}  feats={result.n_features:3d}  "
                  f"pos=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})  "
                  f"{fps_display:.1f}fps  ({elapsed*1000:.0f}ms)")

    stop_event = threading.Event()
    worker     = threading.Thread(target=inference_loop, daemon=True)
    worker.start()

    print("\n[CameraTracker] Move camera slowly — watch position update in terminal.")
    print("Q to quit.\n")

    try:
        while True:
            frame = estimator.read_frame()
            if frame is None:
                break

            with lock:
                latest["frame"] = frame.copy()
                result  = latest["result"]
                depth   = last_depth.copy()
                fps     = fps_display

            if result is not None:
                display = draw_tracking_overlay(frame, result)
            else:
                display = frame.copy()

            cv2.putText(display, f"FPS: {fps:.1f}  Q=quit", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Autorobo — Camera Tracking (Q to quit)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_event.set()
        estimator.close_camera()
        cv2.destroyAllWindows()
        print("[CameraTracker] Stopped.")


if __name__ == "__main__":
    main()
