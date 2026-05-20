"""
perception/rgbd_camera.py — RGB-D camera for MuJoCo simulation.

Renders both a colour (RGB) and a linear depth frame from any named camera
already compiled into the MjModel.  Two rendering passes are issued per
capture: one for RGB, one with depth rendering enabled.

Depth conversion
────────────────
MuJoCo's renderer returns raw OpenGL depth buffer values d ∈ [0, 1].
These are converted to linear metric depth with the perspective formula:

    extent = model.stat.extent
    near   = model.vis.map.znear * extent
    far    = model.vis.map.zfar  * extent
    depth_m = near * far / (far - d * (far - near))

Values outside [min_depth, max_depth] are clamped to 0 (invalid).

Intrinsics
──────────
The pinhole K matrix is derived from the camera's fovy angle:

    fy = (H/2) / tan(fovy/2)
    fx = fy          (square pixels — MuJoCo default)
    cx = W/2,  cy = H/2

Point cloud
───────────
Back-projection from depth to 3-D camera frame (X right, Y down, Z forward):

    z = depth_m[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

Usage
─────
    cam = RGBDCamera(RGBDConfig(camera_name="nav_cam"), model)
    frame = cam.capture(data, step=env.step_count)

    rgb   = frame.rgb           # (H, W, 3) uint8
    depth = frame.depth         # (H, W) float32 metres
    pts   = frame.point_cloud() # (N, 3) float32 camera frame
    cam.close()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RGBDConfig:
    """
    Parameters for the RGB-D camera.

    camera_name       : name of the MuJoCo camera in the compiled model
    width, height     : render resolution (pixels)
    min_depth         : pixels closer than this (metres) are set to 0 (invalid)
    max_depth         : pixels farther than this (metres) are set to 0 (invalid)
    depth_noise_sigma : additive Gaussian noise std on depth (metres); 0 = none
    """
    camera_name:        str   = "nav_cam"
    width:              int   = 640
    height:             int   = 480
    min_depth:          float = 0.05    # m — RealSense D435 min range
    max_depth:          float = 8.0     # m
    depth_noise_sigma:  float = 0.0


# ── frame dataclass ───────────────────────────────────────────────────────────

@dataclass
class RGBDFrame:
    """
    One RGB + depth capture.

    rgb   : (H, W, 3) uint8   — colour image (R, G, B channels)
    depth : (H, W) float32    — metric depth in metres; 0 = invalid
    K     : (3, 3) float64    — pinhole intrinsics
    step  : environment step index at capture time
    """
    rgb:   np.ndarray    # (H, W, 3) uint8
    depth: np.ndarray    # (H, W) float32
    K:     np.ndarray    # (3, 3) float64
    step:  int = 0

    # ── convenience accessors ─────────────────────────────────────────────────

    def rgb_float(self) -> np.ndarray:
        """Return RGB as (H, W, 3) float32 in [0, 1]."""
        return self.rgb.astype(np.float32) / 255.0

    def valid_mask(self) -> np.ndarray:
        """Boolean mask (H, W) — True where depth is finite and > 0."""
        return (self.depth > 0) & np.isfinite(self.depth)

    def point_cloud(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Back-project valid depth pixels to 3-D points in camera frame.

        Camera frame convention: X right, Y down, Z forward (into scene).

        Returns
        -------
        (N, 3) float32 — only valid pixels included (N ≤ H*W).
        """
        H, W  = self.depth.shape
        fx    = self.K[0, 0]
        fy    = self.K[1, 1]
        cx    = self.K[0, 2]
        cy    = self.K[1, 2]

        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        mask = self.valid_mask()
        z = self.depth[mask].astype(np.float32)
        x = (uu[mask] - cx) * z / fx
        y = (vv[mask] - cy) * z / fy
        return np.stack([x, y, z], axis=1).astype(np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        """(H, W) image dimensions."""
        return self.rgb.shape[:2]

    def __repr__(self) -> str:
        H, W = self.shape
        n_valid = int(self.valid_mask().sum())
        return (f"RGBDFrame(step={self.step}, {W}×{H}, "
                f"depth_valid={n_valid}/{H*W})")


# ── camera ────────────────────────────────────────────────────────────────────

class RGBDCamera:
    """
    Two-pass RGB-D renderer wrapping mujoco.Renderer.

    One Renderer instance handles both passes: the first renders RGB, the
    second enables depth rendering, renders, then disables it again.  This
    avoids creating two separate GL contexts.

    Parameters
    ----------
    cfg   : RGBDConfig
    model : compiled MjModel that contains a camera named cfg.camera_name
    """

    def __init__(
        self,
        cfg:   RGBDConfig,
        model: mujoco.MjModel,
        rng:   Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg  = cfg
        self._rng  = rng or np.random.default_rng(0)

        # Resolve camera index
        cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.camera_name
        )
        if cam_id < 0:
            raise ValueError(
                f"Camera '{cfg.camera_name}' not found in model. "
                f"Available cameras: {_list_cameras(model)}"
            )
        self._cam_id = cam_id

        # Compute intrinsics from model fovy
        self._K = _build_K(
            fovy_deg = float(model.cam_fovy[cam_id]),
            width    = cfg.width,
            height   = cfg.height,
        )

        # Depth linearisation constants (computed once — model is immutable)
        extent = float(model.stat.extent)
        self._near = float(model.vis.map.znear) * extent
        self._far  = float(model.vis.map.zfar)  * extent

        # Single renderer — reused across captures
        self._renderer = mujoco.Renderer(model, cfg.height, cfg.width)

    # ── public API ────────────────────────────────────────────────────────────

    def capture(
        self,
        data:  mujoco.MjData,
        step:  int = 0,
    ) -> RGBDFrame:
        """
        Render one RGB + depth frame from the current simulation state.

        Parameters
        ----------
        data : current MjData (kinematics must already be consistent with
               the model; call mj_forward if in doubt)
        step : simulation step index stored in the returned frame

        Returns
        -------
        RGBDFrame with rgb (H,W,3 uint8) and depth (H,W float32 metres).
        """
        # ── RGB pass ──────────────────────────────────────────────────────────
        self._renderer.update_scene(data, camera=self._cam_id)
        rgb = self._renderer.render().copy()   # (H, W, 3) uint8

        # ── depth pass ───────────────────────────────────────────────────────
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=self._cam_id)
        raw_depth = self._renderer.render().copy()   # (H, W) float32 in [0,1]
        self._renderer.disable_depth_rendering()

        depth = self._linearise_depth(raw_depth)

        return RGBDFrame(
            rgb   = rgb,
            depth = depth,
            K     = self._K.copy(),
            step  = step,
        )

    def close(self) -> None:
        """Release the MuJoCo renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def K(self) -> np.ndarray:
        """3×3 pinhole intrinsic matrix (float64)."""
        return self._K.copy()

    @property
    def width(self) -> int:
        return self._cfg.width

    @property
    def height(self) -> int:
        return self._cfg.height

    @property
    def fovy(self) -> float:
        """Vertical field of view in degrees."""
        fy = self._K[1, 1]
        return math.degrees(2.0 * math.atan(self._cfg.height / (2.0 * fy)))

    @property
    def cam_id(self) -> int:
        return self._cam_id

    @property
    def cfg(self) -> RGBDConfig:
        return self._cfg

    # ── depth conversion ──────────────────────────────────────────────────────

    def _linearise_depth(self, raw: np.ndarray) -> np.ndarray:
        """
        Convert raw OpenGL depth buffer [0, 1] → metric depth (float32 metres).

        depth_m = near * far / (far - d * (far - near))

        Pixels outside [min_depth, max_depth] are set to 0 (invalid).
        Depth noise is added after conversion when depth_noise_sigma > 0.
        """
        near, far = self._near, self._far
        # Avoid division by zero in degenerate case (d = 1 and near == far)
        denom = far - raw.astype(np.float64) * (far - near)
        # Guard against near-zero denominator (pixels exactly at far plane)
        depth = np.where(
            np.abs(denom) > 1e-9,
            near * far / denom,
            0.0,
        ).astype(np.float32)

        # Validity clipping
        cfg = self._cfg
        depth[(depth < cfg.min_depth) | (depth > cfg.max_depth)] = 0.0

        # Sensor noise
        if cfg.depth_noise_sigma > 0.0:
            noise = self._rng.normal(
                scale=cfg.depth_noise_sigma, size=depth.shape
            ).astype(np.float32)
            valid = depth > 0
            depth[valid] = np.clip(depth[valid] + noise[valid], 0.0, None)

        return depth

    def __repr__(self) -> str:
        return (f"RGBDCamera(cam='{self._cfg.camera_name}', "
                f"{self._cfg.width}×{self._cfg.height}, "
                f"near={self._near:.2f}m, far={self._far:.2f}m)")


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_K(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Compute pinhole intrinsic matrix from vertical fov + image size."""
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg / 2.0))
    fx = fy          # square pixels (MuJoCo default — isotropic projection)
    cx = width  / 2.0
    cy = height / 2.0
    return np.array([
        [fx,  0., cx],
        [ 0., fy, cy],
        [ 0.,  0., 1.],
    ], dtype=np.float64)


def _list_cameras(model: mujoco.MjModel) -> list[str]:
    """Return names of all cameras in the model (for error messages)."""
    names = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name:
            names.append(name)
    return names
