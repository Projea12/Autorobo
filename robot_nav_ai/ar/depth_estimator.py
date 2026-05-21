"""
ar/depth_estimator.py — Real-time depth estimation using DepthAnything V2.

Runs DepthAnything V2 (small, fast variant) on live webcam frames using
PyTorch MPS (M1 GPU acceleration).

What it does:
    • Captures frames from your webcam
    • Passes each frame through DepthAnything V2
    • Returns a normalised depth map (0 = far, 1 = close)
    • Displays webcam + depth side by side in real time

Usage:
    python ar/depth_estimator.py              # runs live preview
    python ar/depth_estimator.py --no-preview # headless, for integration
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


# ── config ────────────────────────────────────────────────────────────────────

METRIC_MODEL   = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
RELATIVE_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass
class DepthConfig:
    """Configuration for the depth estimator."""
    model_name:   str   = METRIC_MODEL
    camera_index: int   = 0          # 0 = built-in MacBook webcam
    frame_width:  int   = 640
    frame_height: int   = 480
    target_fps:   float = 15.0       # depth inference target
    metric:       bool  = True       # True → raw metres output; False → normalise [0,1]


# ── depth estimator ───────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Wraps DepthAnything V2 for real-time monocular depth estimation.

    Uses MPS (M1 GPU) when available, falls back to CPU.

    Parameters
    ----------
    cfg : DepthConfig
    """

    def __init__(self, cfg: DepthConfig = DepthConfig()) -> None:
        self.cfg = cfg
        self._device = self._best_device()
        print(f"[DepthEstimator] Loading {cfg.model_name} on {self._device} ...")
        self._pipe = pipeline(
            task            = "depth-estimation",
            model           = cfg.model_name,
            device          = self._device,
        )
        print("[DepthEstimator] Model ready.")
        self._cap: Optional[cv2.VideoCapture] = None

    # ── public API ────────────────────────────────────────────────────────────

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Estimate depth for a single BGR frame (from OpenCV).

        Parameters
        ----------
        frame_bgr : H×W×3 uint8 numpy array

        Returns
        -------
        If cfg.metric=True  → H×W float32, values in metres (positive = distance)
        If cfg.metric=False → H×W float32, values in [0, 1]  (1=close, 0=far)
        """
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        out   = self._pipe(pil)
        depth = np.array(out["depth"], dtype=np.float32)

        if not self.cfg.metric:
            # Relative model: normalise to [0, 1] with 1=closest
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth = (depth - d_min) / (d_max - d_min)
            else:
                depth = np.zeros_like(depth)
        # Metric model: depth is already in metres — no transformation needed

        # Resize to match input frame size
        depth = cv2.resize(depth, (frame_bgr.shape[1], frame_bgr.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        return depth

    def open_camera(self) -> None:
        """Open the webcam capture."""
        self._cap = cv2.VideoCapture(self.cfg.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.cfg.camera_index}. "
                "Check that your webcam is connected and not in use."
            )
        print(f"[DepthEstimator] Camera {self.cfg.camera_index} opened "
              f"({self.cfg.frame_width}×{self.cfg.frame_height})")

    def close_camera(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame from the webcam. Returns None if failed."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def run_preview(self) -> None:
        """
        Run live side-by-side preview:
            Left  = webcam feed
            Right = depth map (colourised)
        Press Q to quit.
        """
        self.open_camera()
        interval   = 1.0 / self.cfg.target_fps
        last_depth = np.zeros(
            (self.cfg.frame_height, self.cfg.frame_width), dtype=np.float32
        )
        last_time  = 0.0
        fps_display = 0.0

        print("\n[DepthEstimator] Live preview running.")
        print("  Left  = webcam feed")
        print("  Right = depth map (warm = close, cool = far)")
        print("  Press Q to quit.\n")

        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    print("[DepthEstimator] Camera read failed — stopping.")
                    break

                now = time.perf_counter()

                # Only run depth inference at target_fps
                if now - last_time >= interval:
                    t0         = time.perf_counter()
                    last_depth = self.estimate(frame)
                    elapsed    = time.perf_counter() - t0
                    fps_display = 1.0 / elapsed if elapsed > 0 else 0.0
                    last_time  = now

                # Colourize depth map
                depth_u8    = (last_depth * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

                # Overlay FPS and device info
                info = (f"Depth FPS: {fps_display:.1f}  |  "
                        f"Device: {self._device}  |  Q=quit")
                for img in (frame, depth_color):
                    cv2.putText(img, info, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA)

                # Side by side
                combined = np.hstack([frame, depth_color])
                cv2.imshow("Autorobo — Depth Estimation (Q to quit)", combined)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.close_camera()
            cv2.destroyAllWindows()
            print("[DepthEstimator] Stopped.")

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _best_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def __repr__(self) -> str:
        return (f"DepthEstimator(model={self.cfg.model_name}, "
                f"device={self._device})")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autorobo depth estimator — live webcam preview"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default 0)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip live preview (for integration use)")
    args = parser.parse_args()

    cfg       = DepthConfig(camera_index=args.camera)
    estimator = DepthEstimator(cfg)

    if not args.no_preview:
        estimator.run_preview()
    else:
        print(repr(estimator))
        print("Depth estimator ready for integration.")


if __name__ == "__main__":
    main()
