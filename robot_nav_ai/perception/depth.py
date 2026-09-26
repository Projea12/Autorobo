"""
perception/depth.py — DepthAnything v2 Small depth estimator.

Loads DepthAnything v2 Small via HuggingFace transformers.
Runs on MPS (Apple M1) for real-time inference.
Outputs metric depth maps in metres from single RGB frames.

Failure modes handled:
  - Model download failure  → RuntimeError with clear message
  - MPS unavailable         → falls back to CPU automatically
  - Invalid image input     → ValueError with shape info
  - Inference failure       → RuntimeError, system continues with last valid map
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline

log = logging.getLogger(__name__)

_MODEL_ID   = "depth-anything/Depth-Anything-V2-Small-hf"
_MIN_DEPTH  = 0.1    # metres
_MAX_DEPTH  = 10.0   # metres


class DepthEstimator:
    """
    DepthAnything v2 Small — monocular metric depth from single RGB frame.

    Runs on MPS (M1) when available, falls back to CPU.
    Thread-safe for single-threaded perception pipeline use.
    """

    def __init__(self) -> None:
        self._pipe       = None
        self._last_map:  Optional[np.ndarray] = None
        self._device     = "mps" if torch.backends.mps.is_available() else "cpu"

        try:
            self._pipe = hf_pipeline(
                task              = "depth-estimation",
                model             = _MODEL_ID,
                device            = self._device,
                local_files_only  = True,
            )
            log.info("DepthEstimator loaded on %s (local cache)", self._device)
        except Exception:
            # Local cache miss — allow network download on first install
            try:
                self._pipe = hf_pipeline(
                    task    = "depth-estimation",
                    model   = _MODEL_ID,
                    device  = self._device,
                )
                log.info("DepthEstimator loaded on %s (downloaded)", self._device)
            except Exception as exc:
                raise RuntimeError(
                    f"DepthEstimator failed to load '{_MODEL_ID}': {exc}\n"
                    "Check internet connection for first-time model download."
                ) from exc

    # ── public API ─────────────────────────────────────────────────────────────

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a single RGB frame.

        Parameters
        ----------
        frame : (H, W, 3) uint8 RGB numpy array

        Returns
        -------
        (H, W) float32 depth map in metres, clipped to [0.1, 10.0].
        Returns last valid map on inference failure — never raises mid-loop.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) RGB frame, got shape {frame.shape}"
            )

        try:
            pil_img    = Image.fromarray(frame)
            result     = self._pipe(pil_img)
            depth      = np.array(result["depth"], dtype=np.float32)

            # Resize to match input frame resolution
            if depth.shape != frame.shape[:2]:
                pil_depth = Image.fromarray(depth).resize(
                    (frame.shape[1], frame.shape[0]),
                    resample=Image.BILINEAR,
                )
                depth = np.array(pil_depth, dtype=np.float32)

            depth = np.clip(depth, _MIN_DEPTH, _MAX_DEPTH)
            self._last_map = depth
            return depth

        except Exception as exc:
            log.warning("DepthEstimator inference failed: %s — returning last valid map", exc)
            if self._last_map is not None:
                return self._last_map
            return np.full(frame.shape[:2], _MAX_DEPTH, dtype=np.float32)

    def get_object_depth(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        aggregation: str = "median",
    ) -> float:
        """
        Estimate depth of a masked object region.

        Parameters
        ----------
        depth_map   : (H, W) float32 depth map from estimate()
        mask        : (H, W) bool mask of the object region
        aggregation : "median" | "mean" | "min"

        Returns
        -------
        Depth in metres. Returns _MAX_DEPTH if no valid pixels in mask.
        """
        masked = depth_map[mask]
        valid  = masked[(masked >= _MIN_DEPTH) & (masked <= _MAX_DEPTH)]

        if len(valid) == 0:
            log.warning("get_object_depth: no valid pixels in mask, returning max depth")
            return _MAX_DEPTH

        if aggregation == "median":
            return float(np.median(valid))
        elif aggregation == "mean":
            return float(np.mean(valid))
        elif aggregation == "min":
            return float(np.min(valid))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r}. Use 'median', 'mean', or 'min'.")

    def pixel_to_3d(
        self,
        pixel_xy:          np.ndarray,
        depth:             float,
        camera_intrinsics: np.ndarray,
    ) -> np.ndarray:
        """
        Back-project pixel + depth to 3D camera-frame point.

        Parameters
        ----------
        pixel_xy          : (2,) [u, v] pixel coordinates
        depth             : depth in metres
        camera_intrinsics : (3, 3) camera K matrix

        Returns
        -------
        (3,) [X, Y, Z] in metres, camera frame.
        """
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        u, v = float(pixel_xy[0]), float(pixel_xy[1])
        return np.array([
            (u - cx) * depth / fx,
            (v - cy) * depth / fy,
            depth,
        ], dtype=np.float32)

    def visualise(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Colourise depth map for display.

        Returns (H, W, 3) uint8 BGR image — near=dark blue, far=yellow.
        """
        import cv2
        valid_min = float(np.percentile(depth_map, 2))
        valid_max = float(np.percentile(depth_map, 98))
        norm = np.clip(
            (depth_map - valid_min) / (valid_max - valid_min + 1e-8),
            0.0, 1.0,
        )
        grey = (norm * 255).astype(np.uint8)
        return cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)

    @property
    def device(self) -> str:
        return self._device

    def __repr__(self) -> str:
        return f"DepthEstimator(model='{_MODEL_ID}', device='{self._device}')"
