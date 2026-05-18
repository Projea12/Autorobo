"""
depth.py — DepthAnything v2 Depth Estimator (Phase 6)

Estimates metric depth maps from monocular RGB images using DepthAnything v2.
Provides absolute depth values in metres (metric depth mode), required for
accurate 3D object localisation and grasp pose estimation.

The depth map is fused with segmentation masks from ObjectSegmenter to
estimate the 3D centroid of each detected object in the camera frame.

Usage:
    from perception.depth import DepthEstimator

    estimator = DepthEstimator(cfg.perception.depth)
    depth_map = estimator.estimate(rgb_image)  # (H, W) float32, metres
    object_depth = estimator.get_object_depth(depth_map, mask)  # scalar metres
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class DepthEstimator:
    """
    DepthAnything v2 monocular metric depth estimator.

    Predicts dense depth maps from single RGB images. Operates in
    metric depth mode — outputs absolute distances in metres, not
    relative/affine depth that requires scale estimation.

    Configuration via DictConfig (from configs/perception/yolo.yaml):
        model_size: "small", "base", or "large"
        checkpoint: path to .pth model file
        device: "cpu", "cuda:0", or "mps"
        min_depth: minimum valid depth in metres (default 0.1)
        max_depth: maximum valid depth in metres (default 10.0)
    """

    def __init__(self, cfg: Any) -> None:
        """
        Initialise and load the DepthAnything v2 model.

        Args:
            cfg: DictConfig with depth estimator settings.

        TODO: Phase 6 — implement:
            from depth_anything_v2.dpt import DepthAnythingV2
            model_configs = {
                "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "base":  {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }
            self._model = DepthAnythingV2(**model_configs[cfg.model_size])
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
            self._model.load_state_dict(state_dict)
            self._model.to(cfg.device).eval()
        """
        self.cfg = cfg
        self._model = None  # DepthAnythingV2 — loaded in Phase 6
        self._min_depth = getattr(cfg, "min_depth", 0.1)
        self._max_depth = getattr(cfg, "max_depth", 10.0)
        log.info("DepthEstimator created (model not yet loaded — TODO: Phase 6)")

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate a metric depth map from an RGB image.

        Args:
            image: RGB image as np.ndarray of shape (H, W, 3), dtype uint8.

        Returns:
            Depth map as np.ndarray of shape (H, W), dtype float32.
            Values are in metres. Invalid/occluded pixels are np.nan.
            Range: [min_depth, max_depth] metres.

        TODO: Phase 6 — implement:
            import torch
            img_tensor = self._preprocess(image)  # normalise, resize to 518x518
            with torch.inference_mode():
                depth = self._model(img_tensor)
            depth = self._postprocess(depth, original_shape=image.shape[:2])
            depth = np.clip(depth, self._min_depth, self._max_depth)
            return depth
        """
        if self._model is None:
            raise RuntimeError(
                "DepthEstimator model not loaded. TODO: Phase 6 — load model in __init__."
            )
        raise NotImplementedError(
            "TODO: Phase 6 — implement estimate() using DepthAnythingV2 inference."
        )

    def get_object_depth(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        aggregation: str = "median",
    ) -> float:
        """
        Estimate the depth of a masked object region.

        Args:
            depth_map: Depth map from estimate(), shape (H, W), float32.
            mask: Binary mask of the object, shape (H, W), dtype bool.
            aggregation: How to aggregate depth values within the mask.
                "median" (robust to outliers), "mean", or "min" (nearest point).

        Returns:
            Estimated object depth in metres.

        Raises:
            ValueError: If mask has no valid pixels in the depth map.
        """
        masked_depths = depth_map[mask]
        valid_depths = masked_depths[
            ~np.isnan(masked_depths) &
            (masked_depths >= self._min_depth) &
            (masked_depths <= self._max_depth)
        ]

        if len(valid_depths) == 0:
            raise ValueError(
                "No valid depth values in masked region. "
                "Check that the mask overlaps a region with valid depth."
            )

        if aggregation == "median":
            return float(np.median(valid_depths))
        elif aggregation == "mean":
            return float(np.mean(valid_depths))
        elif aggregation == "min":
            return float(np.min(valid_depths))
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation!r}")

    def pixel_to_3d(
        self,
        pixel_xy: np.ndarray,
        depth: float,
        camera_intrinsics: np.ndarray,
    ) -> np.ndarray:
        """
        Back-project a 2D pixel point + depth to a 3D camera-frame point.

        Args:
            pixel_xy: Pixel coordinates [u, v], shape (2,).
            depth: Depth value at this pixel, in metres.
            camera_intrinsics: Camera K matrix, shape (3, 3).
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        Returns:
            3D point [X, Y, Z] in camera frame, in metres. Shape (3,).
        """
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]

        u, v = pixel_xy
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return np.array([x, y, z], dtype=np.float32)

    def visualise_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert a depth map to a colourised RGB image for debugging.

        Args:
            depth_map: Depth map, shape (H, W), float32.

        Returns:
            Colourised depth image, shape (H, W, 3), uint8.
            Uses inferno colormap: dark=near, bright=far.

        TODO: Phase 6 — implement:
            import matplotlib.cm as cm
            normalised = (depth_map - depth_map.min()) / (depth_map.ptp() + 1e-8)
            colourised = (cm.inferno(normalised)[:, :, :3] * 255).astype(np.uint8)
            return colourised
        """
        raise NotImplementedError(
            "TODO: Phase 6 — implement depth visualisation with matplotlib colormap."
        )
