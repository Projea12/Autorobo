"""
mujoco_interface.py — MuJoCo Simulation Interface

Implements BaseRobotInterface by wrapping ManipulationEnv.
All physics, reward, and action-scaling logic lives in ManipulationEnv;
this class adds the camera and lidar rendering required by the interface contract,
and translates between the Gymnasium 5-tuple API and the 4-tuple BaseRobotInterface API.

Sim→real swap: replace this file with ros2_interface.py and nothing else changes.

Usage:
    interface = MuJoCoInterface()
    obs = interface.reset()
    obs, reward, done, info = interface.step(action)
    interface.close()
"""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np

from interfaces.base_interface import BaseRobotInterface
from env.manipulation_env import ManipulationEnv

log = logging.getLogger(__name__)

# ── camera / lidar constants ──────────────────────────────────────────────────

_CAM_H     = 480
_CAM_W     = 640
_CAM_NAME  = "rgbd_cam"
_LIDAR_N   = 360    # one ray per degree, horizontal sweep
_LIDAR_MAX = 10.0   # metres — rays that miss return this value


def _cfg_get(cfg: Any, dotpath: str, default: Any) -> Any:
    """Read a dotted attribute path from cfg, falling back to default."""
    if cfg is None:
        return default
    try:
        obj = cfg
        for key in dotpath.split("."):
            obj = getattr(obj, key)
        return obj
    except AttributeError:
        return default


class MuJoCoInterface(BaseRobotInterface):
    """
    MuJoCo implementation of BaseRobotInterface.

    Wraps ManipulationEnv (physics + reward) and adds:
      - RGB rendering via mujoco.Renderer (rgbd_cam)
      - Depth rendering via mujoco.Renderer with depth enabled (rgbd_cam)
      - 360° 2D lidar simulation via mujoco.mj_ray from lidar_site

    The observation dict returned by reset/step/get_observation is:
        {
            "rgb":            np.ndarray (H, W, 3)    uint8
            "depth":          np.ndarray (H, W)       float32, metres
            "lidar":          np.ndarray (LIDAR_N,)   float32, metres
            "proprioception": np.ndarray (45,)        float32
        }

    cfg is optional. If supplied it may carry:
        cfg.env.episode.max_steps   (int, default 500)
        cfg.env.mujoco.n_substeps   (int, default 5)
    """

    def __init__(self, cfg: Any = None) -> None:
        super().__init__(cfg)

        max_steps  = _cfg_get(cfg, "env.episode.max_steps", 500)
        n_substeps = _cfg_get(cfg, "env.mujoco.n_substeps", 5)

        self._env = ManipulationEnv(
            render_mode=None,
            max_steps=max_steps,
            n_substeps=n_substeps,
        )

        m = self._env._model

        self._rgb_renderer = mujoco.Renderer(m, height=_CAM_H, width=_CAM_W)

        self._depth_renderer = mujoco.Renderer(m, height=_CAM_H, width=_CAM_W)
        self._depth_renderer.enable_depth_rendering()

        self._lidar_site_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, "lidar_site"
        )
        if self._lidar_site_id == -1:
            log.warning("lidar_site not found in model — lidar will return zeros")

        # Precompute horizontal ray directions in site-local frame (x-y plane)
        angles = np.linspace(0.0, 2.0 * np.pi, _LIDAR_N, endpoint=False)
        self._lidar_local_dirs = np.stack(
            [np.cos(angles), np.sin(angles), np.zeros(_LIDAR_N)], axis=1
        )  # (N, 3)

        self._step_count    = 0
        self._episode_count = 0
        log.info("MuJoCoInterface ready")

    # ── BaseRobotInterface API ────────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        """Reset simulation and return the initial observation dict."""
        self._env.reset()
        self._step_count = 0
        self._episode_count += 1
        return self.get_observation()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Apply action, advance physics, return (obs_dict, reward, done, info).

        done = terminated (success) OR truncated (timeout).
        info keys: success, collision, timeout, distance_to_goal, lift, step.
        """
        action = np.asarray(action, dtype=np.float32)
        _, reward, terminated, truncated, gym_info = self._env.step(action)

        self._step_count += 1
        obs  = self.get_observation()
        done = terminated or truncated

        info = {
            "success":          gym_info["success"],
            "collision":        False,
            "timeout":          truncated,
            "distance_to_goal": gym_info["ee_to_target"],
            "lift":             gym_info["lift"],
            "step":             self._step_count,
        }
        return obs, float(reward), done, info

    def get_observation(self) -> dict[str, Any]:
        """
        Build the full observation dict from current simulation state.

        Calls mj_forward to ensure renderers and sensor data are consistent
        before rendering — safe to call without stepping.
        """
        m = self._env._model
        d = self._env._data
        mujoco.mj_forward(m, d)

        self._rgb_renderer.update_scene(d, camera=_CAM_NAME)
        rgb = self._rgb_renderer.render().copy()            # (H, W, 3) uint8

        self._depth_renderer.update_scene(d, camera=_CAM_NAME)
        depth = self._depth_renderer.render().copy()       # (H, W) float32, metres

        lidar  = self._cast_lidar(m, d)                    # (N,) float32
        proprio = self._env._get_obs()                     # (45,) float32

        return {
            "rgb":            rgb,
            "depth":          depth,
            "lidar":          lidar,
            "proprioception": proprio,
        }

    def apply_action(self, action: Any) -> None:
        """
        Write scaled control to the actuators without advancing physics.

        Use this for real-time loops where the caller drives the step cadence.
        """
        ctrl = self._env._scale_action(np.asarray(action, dtype=np.float64))
        np.copyto(self._env._data.ctrl, ctrl)

    def close(self) -> None:
        """Free all renderers and the underlying environment."""
        self._rgb_renderer.close()
        self._depth_renderer.close()
        self._env.close()
        self._is_closed = True
        log.info(
            "MuJoCoInterface closed after %d episodes, %d steps",
            self._episode_count,
            self._step_count,
        )

    # ── Sim-only extras ───────────────────────────────────────────────────────

    def get_ground_truth_pose(self, body_name: str) -> np.ndarray:
        """
        Return [x, y, z, qw, qx, qy, qz] for any named body (sim privilege).
        Only available in simulation — do not call via ROS2Interface.
        """
        m = self._env._model
        d = self._env._data
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model")
        return np.concatenate([d.xpos[body_id].copy(), d.xquat[body_id].copy()])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _cast_lidar(self, m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
        """
        Simulate a 2D horizontal lidar scan by raycasting from lidar_site.

        Returns (LIDAR_N,) float32 array of range readings in metres.
        Rays that hit nothing return _LIDAR_MAX.
        """
        ranges = np.full(_LIDAR_N, _LIDAR_MAX, dtype=np.float32)

        if self._lidar_site_id == -1:
            return ranges

        origin   = d.site_xpos[self._lidar_site_id].copy()              # (3,)
        site_rot = d.site_xmat[self._lidar_site_id].reshape(3, 3)       # (3, 3)

        # Rotate all ray directions from site-local frame to world frame at once
        world_dirs = self._lidar_local_dirs @ site_rot.T                  # (N, 3)

        geomid = np.array([-1], dtype=np.int32)
        for i in range(_LIDAR_N):
            dist = mujoco.mj_ray(m, d, origin, world_dirs[i], None, 1, -1, geomid)
            if dist >= 0:
                ranges[i] = min(float(dist), _LIDAR_MAX)

        return ranges
