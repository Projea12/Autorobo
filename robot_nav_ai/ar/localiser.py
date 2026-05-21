"""
ar/localiser.py — Metric 3D localisation from DepthAnything V2.

Runs DepthAnything V2 Metric-Indoor in a background thread and
back-projects detection bounding boxes to metric 3D coordinates in both
the camera frame and the robot base frame.

Pipeline per detection
----------------------
1. Sample inner-50% bbox region of the depth map → median depth d (metres)
2. back_project(u, v, d, K) → (X, Y, Z) in camera frame
3. T_CAM_TO_BASE((X, Y, Z)_cam) → (X, Y, Z) in TidyBot base frame

See ar/transforms.py for the camera mounting geometry and axis conventions.

Usage
-----
    from ar.localiser import Localiser
    from ar.ar_renderer import CameraIntrinsics

    K   = CameraIntrinsics(fx=800, fy=800, cx=320, cy=240)
    loc = Localiser(every_n=5)
    loc.start()

    loc.update(frame)                          # non-blocking feed

    depth = loc.latest_depth()                 # H×W float32 metres
    if depth is not None:
        xyz = loc.localise(detections, depth, K)
        loc.draw_3d(frame, detections, xyz)
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ar.transforms import T_CAM_TO_BASE, RigidTransform


# ── inner-box sampling ────────────────────────────────────────────────────────

INNER_FRAC: float = 0.50   # keep central 50% of each bbox dimension


def _inner_region(
    bbox_xyxy: Tuple[int, int, int, int],
    depth_map: np.ndarray,
) -> np.ndarray:
    """
    Return depth values from the inner 50% of the bounding box.

    Shrinks each edge by 25% so background pixels at box borders
    don't contaminate the estimate.
    """
    x1, y1, x2, y2 = bbox_xyxy
    H, W = depth_map.shape[:2]

    pad_x = max(1, int((x2 - x1) * (1.0 - INNER_FRAC) / 2))
    pad_y = max(1, int((y2 - y1) * (1.0 - INNER_FRAC) / 2))

    ix1 = int(np.clip(x1 + pad_x, 0, W - 1))
    ix2 = int(np.clip(x2 - pad_x, ix1 + 1, W))
    iy1 = int(np.clip(y1 + pad_y, 0, H - 1))
    iy2 = int(np.clip(y2 - pad_y, iy1 + 1, H))

    return depth_map[iy1:iy2, ix1:ix2]


# ── Localiser ─────────────────────────────────────────────────────────────────

class Localiser:
    """
    Metric 3-D localisation from monocular depth.

    Background thread runs DepthAnything V2 Metric-Indoor every `every_n`
    frames.  `localise()` back-projects detection centroids to (X,Y,Z)
    in the camera frame using the pin-hole model.

    Parameters
    ----------
    every_n     : run depth inference on every Nth frame (default 5)
    """

    def __init__(self, every_n: int = 5) -> None:
        self._every_n    = every_n
        self._frame_in: Optional[np.ndarray] = None
        self._depth_out: Optional[np.ndarray] = None
        self._frame_count = 0
        self._lock       = threading.Lock()
        self._stop_evt   = threading.Event()
        self._thread     = threading.Thread(target=self._loop, daemon=True)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()
        print("[Localiser] started — DepthAnything V2 Metric-Indoor loading…")

    def stop(self) -> None:
        self._stop_evt.set()

    # ── feed / read ───────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> None:
        """Feed the latest video frame (non-blocking)."""
        with self._lock:
            self._frame_in = frame

    def latest_depth(self) -> Optional[np.ndarray]:
        """
        Latest depth map (H×W float32, metres).
        Returns None until the first inference completes.
        """
        with self._lock:
            return self._depth_out

    # ── 3-D back-projection ───────────────────────────────────────────────────

    @staticmethod
    def back_project(
        u: float,
        v: float,
        d: float,
        intrinsics,
    ) -> Tuple[float, float, float]:
        """
        Pin-hole back-projection: pixel (u, v) + metric depth d → camera-frame (X, Y, Z).

        Formulae
        --------
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d

        Parameters
        ----------
        u, v        : pixel coordinates (column, row)
        d           : depth in metres (positive)
        intrinsics  : object with .fx .fy .cx .cy  (e.g. CameraIntrinsics)

        Returns
        -------
        (X, Y, Z) in metres, camera frame.
        X=right, Y=down, Z=into scene.
        """
        X = (u - intrinsics.cx) * d / intrinsics.fx
        Y = (v - intrinsics.cy) * d / intrinsics.fy
        Z = d
        return (X, Y, Z)

    @staticmethod
    def to_base_frame(
        xyz_cam: Tuple[float, float, float],
        transform: RigidTransform = T_CAM_TO_BASE,
    ) -> Tuple[float, float, float]:
        """
        Convert a camera-frame point to the robot base frame.

        Uses the fixed camera mounting transform (ar/transforms.py):
            camera at [right=0, forward=0.1m, up=1.2m] on TidyBot.

        Parameters
        ----------
        xyz_cam   : (X, Y, Z) in camera frame — from back_project()
        transform : RigidTransform to use (default T_CAM_TO_BASE)

        Returns
        -------
        (X, Y, Z) in TidyBot base frame.
            X = right (metres),  Y = forward (metres),  Z = up (metres)

        Example
        -------
        Object at 1 m directly in front of the camera (optical axis):
            xyz_cam  = (0.0, 0.0, 1.0)
            xyz_base = (0.0, 1.1, 1.2)   # Y_base ≈ 1.0 m forward ✓
        """
        return transform(xyz_cam)

    def localise(
        self,
        detections,
        depth_map: np.ndarray,
        intrinsics,
        in_base_frame: bool = True,
    ) -> List[Optional[Tuple[float, float, float]]]:
        """
        Back-project each detection into 3-D space.

        Pipeline per detection:
          1. Sample inner-50% bbox region → median depth d (metres)
          2. back_project(u, v, d, K) → (X,Y,Z) in camera frame
          3. to_base_frame() → (X,Y,Z) in TidyBot base frame  [if in_base_frame]

        Parameters
        ----------
        detections     : list[Detection]
        depth_map      : H×W float32 metres (from latest_depth())
        intrinsics     : CameraIntrinsics — needs .fx .fy .cx .cy
        in_base_frame  : if True (default), transform output to robot base frame

        Returns
        -------
        List of (X, Y, Z) in metres, one per detection.
        Camera frame if in_base_frame=False, base frame otherwise.
        Returns None for any detection with no valid depth pixels.
        """
        results = []
        for det in detections:
            patch = _inner_region(det.bbox_xyxy, depth_map)
            valid = patch[patch > 0.01]
            if valid.size == 0:
                results.append(None)
                continue
            d       = float(np.median(valid))
            u, v    = det.centroid_uv
            xyz_cam = self.back_project(u, v, d, intrinsics)
            if in_base_frame:
                xyz = self.to_base_frame(xyz_cam)
            else:
                xyz = xyz_cam
            results.append(tuple(round(c, 3) for c in xyz))
        return results

    # ── drawing ───────────────────────────────────────────────────────────────

    @staticmethod
    def draw_3d(
        frame: np.ndarray,
        detections,
        xyz_list: List[Optional[Tuple[float, float, float]]],
    ) -> np.ndarray:
        """
        Overlay (X, Y, Z) labels just above each detection bounding box.
        Format: `+0.3,−0.1, 1.4m`
        """
        for det, xyz in zip(detections, xyz_list):
            if xyz is None:
                continue
            x1, y1, _, _ = det.bbox_xyxy
            label = f"{xyz[0]:+.1f},{xyz[1]:+.1f},{xyz[2]:.2f}m"
            ty    = max(y1 - 4, 14)
            cv2.putText(frame, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    # ── background inference loop ─────────────────────────────────────────────

    def _loop(self) -> None:
        from ar.depth_estimator import DepthEstimator, DepthConfig

        cfg = DepthConfig(metric=True)
        try:
            estimator = DepthEstimator(cfg)
            print("[Localiser] DepthAnything V2 Metric-Indoor ready.")
        except Exception as exc:
            print(f"[Localiser] Failed to load depth model: {exc}")
            return

        frame_count = 0
        while not self._stop_evt.is_set():
            with self._lock:
                frame = self._frame_in

            if frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            if frame_count % self._every_n != 0:
                time.sleep(0.01)
                continue

            try:
                depth = estimator.estimate(frame)
                with self._lock:
                    self._depth_out = depth
            except Exception as exc:
                print(f"[Localiser] depth error: {exc}")

            time.sleep(0.01)
