"""
mujoco_interface.py — MuJoCo Simulation Interface (Phases 1–15)

Implements BaseRobotInterface using the MuJoCo 3.x physics engine.
This is the primary interface for all training and evaluation in simulation.

Usage:
    from interfaces.mujoco_interface import MuJoCoInterface

    interface = MuJoCoInterface(cfg)
    obs = interface.reset()
    obs, reward, done, info = interface.step(action)
    interface.close()
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from interfaces.base_interface import BaseRobotInterface

log = logging.getLogger(__name__)


class MuJoCoInterface(BaseRobotInterface):
    """
    MuJoCo implementation of BaseRobotInterface.

    Wraps the MuJoCo physics engine to provide a uniform observation/action
    interface compatible with all policy training code.

    The full implementation (Phase 1–3) will:
    - Load the MJCF scene XML from cfg.env.mujoco.scene_xml
    - Initialise MjModel and MjData
    - Set up camera renderers for RGB and depth
    - Configure LiDAR raycast simulation
    - Apply control at cfg.env.mujoco.control_timestep frequency
    - Compute rewards based on cfg.env.reward settings
    """

    def __init__(self, cfg: Any) -> None:
        """
        Initialise the MuJoCo interface.

        Args:
            cfg: Hydra DictConfig with env.mujoco and robot settings.

        TODO: Phase 1 — implement:
            import mujoco
            self.model = mujoco.MjModel.from_xml_path(cfg.env.mujoco.scene_xml)
            self.data = mujoco.MjData(self.model)
            self.renderer = mujoco.Renderer(self.model, ...)
        """
        super().__init__(cfg)
        self._model = None   # mujoco.MjModel — set in Phase 1
        self._data = None    # mujoco.MjData — set in Phase 1
        self._renderer = None  # mujoco.Renderer — set in Phase 1
        self._step_count = 0
        self._episode_count = 0
        log.info("MuJoCoInterface created (not yet initialised — TODO: Phase 1)")

    def reset(self) -> dict[str, Any]:
        """
        Reset MuJoCo simulation to a randomised initial state.

        TODO: Phase 1 — implement:
            mujoco.mj_resetData(self._model, self._data)
            # Randomise object positions within spawn area
            # Randomise robot start pose
            # Step physics to settle objects (5–10 steps)
            return self._build_observation()
        """
        self._step_count = 0
        self._episode_count += 1
        raise NotImplementedError(
            "TODO: Phase 1 — implement MuJoCo reset: "
            "mj_resetData(), randomise object/robot poses, settle physics, "
            "return observation dict."
        )

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Apply action and advance MuJoCo simulation by one control step.

        TODO: Phase 1 — implement:
            # Apply action to actuators
            self._apply_control(action)
            # Step physics for n_substeps
            for _ in range(self.cfg.env.mujoco.n_substeps):
                mujoco.mj_step(self._model, self._data)
            self._step_count += 1
            obs = self._build_observation()
            reward = self._compute_reward(action)
            done = self._check_done()
            info = self._build_info()
            return obs, reward, done, info
        """
        raise NotImplementedError(
            "TODO: Phase 1 — implement MuJoCo step: "
            "apply control, mj_step() × n_substeps, build obs/reward/done/info."
        )

    def get_observation(self) -> dict[str, Any]:
        """
        Build observation from current MuJoCo state without stepping.

        TODO: Phase 2 — implement _build_observation():
            rgb = self._render_rgb()       # (H, W, 3) uint8
            depth = self._render_depth()   # (H, W) float32 in metres
            lidar = self._cast_lidar()     # (N,) float32 range readings
            proprio = self._get_proprio()  # joint positions + velocities
            return {"rgb": rgb, "depth": depth, "lidar": lidar, "proprioception": proprio}
        """
        raise NotImplementedError(
            "TODO: Phase 2 — implement _build_observation() using MuJoCo renderer "
            "for RGB/depth and raycast for LiDAR."
        )

    def apply_action(self, action: Any) -> None:
        """
        Send control commands to MuJoCo actuators without stepping.

        TODO: Phase 1 — implement:
            # For navigation: set wheel velocity actuators
            self._data.ctrl[self._base_actuator_ids] = self._nav_action_to_ctrl(action)
            # For manipulation: set arm joint actuators
            self._data.ctrl[self._arm_actuator_ids] = self._arm_action_to_ctrl(action)
        """
        raise NotImplementedError(
            "TODO: Phase 1 — implement apply_action: "
            "map action vector to mujoco ctrl array entries."
        )

    def close(self) -> None:
        """
        Free MuJoCo model and data memory, close renderer.

        TODO: Phase 1 — implement:
            if self._renderer is not None:
                self._renderer.close()
            # MjModel/MjData are freed by Python GC when dereferenced
            self._model = None
            self._data = None
            self._is_closed = True
        """
        log.info(
            f"MuJoCoInterface.close() called after {self._episode_count} episodes."
        )
        self._is_closed = True
        raise NotImplementedError(
            "TODO: Phase 1 — implement close(): "
            "free renderer, dereference model/data."
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_reward(self, action: Any) -> float:
        """
        Compute step reward based on current state and action.

        TODO: Phase 1/2 — implement per cfg.env.reward settings:
            - distance shaping toward goal
            - collision penalty
            - step penalty
            - success bonus
        """
        raise NotImplementedError(
            "TODO: Phase 1 — implement reward function from cfg.env.reward."
        )

    def _check_done(self) -> bool:
        """
        Check episode termination conditions.

        TODO: Phase 1 — implement:
            success = self._check_success()
            timeout = self._step_count >= self.cfg.env.episode.max_steps
            collision = self._check_collision()
            return success or timeout or collision
        """
        raise NotImplementedError(
            "TODO: Phase 1 — implement done check: success, timeout, collision."
        )

    def _get_ground_truth_pose(self, body_name: str) -> np.ndarray:
        """
        Get ground-truth 6D pose (position + quaternion) of a named body.

        This is only available in simulation — exposes simulator privilege.
        Used by oracle demo collector and for debugging.

        Args:
            body_name: MuJoCo body name from scene XML.

        Returns:
            np.ndarray of shape (7,): [x, y, z, qw, qx, qy, qz]

        TODO: Phase 7 — implement:
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos = self._data.xpos[body_id]
            quat = self._data.xquat[body_id]
            return np.concatenate([pos, quat])
        """
        raise NotImplementedError(
            f"TODO: Phase 7 — get ground truth pose for '{body_name}' "
            "from self._data.xpos and self._data.xquat."
        )
