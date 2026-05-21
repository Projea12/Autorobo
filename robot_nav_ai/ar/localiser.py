"""
ar/localiser.py — 3D localisation from monocular depth.

Runs DepthAnything V2 in a background thread and back-projects
detection centroids to 3D camera-frame coordinates (X, Y, Z in metres,
approximate scale).

Usage
-----
    from ar.localiser import Localiser
    from ar.ar_renderer import CameraIntrinsics

    K   = CameraIntrinsics(fx=800, fy=800, cx=320, cy=240)
    loc = Localiser(every_n=5)
    loc.start()

    # in display loop:
    loc.update(frame)
    depth = loc.latest_depth()
    if depth is not None:
        xyz_list = loc.localise(detections, depth, K)
        # xyz_list[i] = (X, Y, Z) in metres for detections[i]
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── depth scale ───────────────────────────────────────────────────────────────
# DepthAnything V2 outputs relative depth: 0=far, 1=close.
# We invert and scale by DEPTH_SCALE_M to get approximate metric depth.
# 5 m is a reasonable max for an indoor room scene.
DEPTH_SCALE_M: float = 5.0

# Patch half-size (pixels) used to sample median depth around centroid.
PATCH_HALF: int = 8


# ── Localiser ─────────────────────────────────────────────────────────────────

class Localiser:
    """
    Monocular 3-D localisation.

    Runs DepthAnything V2 (Small) in a background thread and exposes
    `localise()` to convert pixel centroids → (X, Y, Z) camera-frame coords.
    """

    def __init__(self, every_n: int = 5, depth_scale: float = DEPTH_SCALE_M) -> None:
        self._every_n     = every_n
        self._depth_scale = depth_scale

        self._frame_in: Optional[np.ndarray] = None
        self._depth_out: Optional[np.ndarray] = None
        self._tick       = 0
        self._lock       = threading.Lock()
        self._stop_evt   = threading.Event()
        self._thread     = threading.Thread(target=self._loop, daemon=True)
        self._estimator  = None   # lazy-loaded inside thread

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()
        print("[Localiser] started (DepthAnything V2 loading in background…)")

    def stop(self) -> None:
        self._stop_evt.set()

    # ── feed / read ───────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> None:
        """Feed the latest video frame (non-blocking)."""
        with self._lock:
            self._frame_in = frame

    def latest_depth(self) -> Optional[np.ndarray]:
        """
        Return the most recent depth map (H×W float32, 0=far…1=close).
        Returns None until the first inference completes.
        """
        with self._lock:
            return self._depth_out

    # ── 3-D projection ────────────────────────────────────────────────────────

    def localise(
        self,
        detections,
        depth_map: np.ndarray,
        intrinsics,
    ) -> List[Optional[Tuple[float, float, float]]]:
        """
        Back-project each detection centroid into 3-D camera space.

        Parameters
        ----------
        detections  : list[Detection] — from ObjectDetector.latest
        depth_map   : H×W float32    — from Localiser.latest_depth()
        intrinsics  : CameraIntrinsics (has .fx .fy .cx .cy)

        Returns
        -------
        list of (X, Y, Z) tuples in metres (same order as `detections`).
        Returns None for any detection whose centroid is out of frame.
        """
        H, W  = depth_map.shape[:2]
        fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
        results = []

        for det in detections:
            u, v = det.centroid_uv
            u    = int(np.clip(u, 0, W - 1))
            v    = int(np.clip(v, 0, H - 1))

            # Sample depth over a small patch (robust to noise)
            u0, u1 = max(0, u - PATCH_HALF), min(W, u + PATCH_HALF + 1)
            v0, v1 = max(0, v - PATCH_HALF), min(H, v + PATCH_HALF + 1)
            patch  = depth_map[v0:v1, u0:u1]

            if patch.size == 0:
                results.append(None)
                continue

            depth_norm = float(np.median(patch))

            # Invert: 1=close → small depth, 0=far → large depth
            d = (1.0 - depth_norm) * self._depth_scale   # metres (approximate)

            # Pin-hole back-projection into camera frame
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d

            results.append((round(X, 2), round(Y, 2), round(Z, 2)))

        return results

    @staticmethod
    def draw_3d(
        frame: np.ndarray,
        detections,
        xyz_list: List[Optional[Tuple[float, float, float]]],
    ) -> np.ndarray:
        """
        Draw 3-D position labels (X, Y, Z) above each detection box.

        Modifies `frame` in-place and returns it.
        """
        for det, xyz in zip(detections, xyz_list):
            if xyz is None:
                continue
            x1, y1, x2, y2 = det.bbox_xyxy
            label = f"{xyz[0]:+.1f},{xyz[1]:+.1f},{xyz[2]:.1f}m"
            tx    = x1
            ty    = max(y1 - 4, 14)
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    # ── background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        # Lazy import so startup doesn't block
        from ar.depth_estimator import DepthEstimator, DepthConfig

        cfg = DepthConfig()
        try:
            self._estimator = DepthEstimator(cfg)
            print("[Localiser] DepthAnything V2 ready.")
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
                depth = self._estimator.estimate(frame)
                with self._lock:
                    self._depth_out = depth
            except Exception as exc:
                print(f"[Localiser] depth error: {exc}")

            time.sleep(0.01)
