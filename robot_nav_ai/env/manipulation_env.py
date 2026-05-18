"""
env/manipulation_env.py — Gymnasium environment for mobile manipulation.

Task
────
The robot must drive toward a target box on the floor, reach it with the arm,
grasp it, and lift it ≥ 20 cm above the floor.

Architecture
────────────
  Observation  : 45-dim float32 Box (proprioceptive + target position)
  Action       : 9-dim float32 Box in [−1, 1] (2 wheels + 6 arm + 1 gripper)
  Reward       : dense shaping (navigation → reach → contact → grasp → lift)
                 + sparse terminal bonus on success
  Termination  : success (lift ≥ 20 cm) or timeout (max_steps)

Physics
───────
  dt_env = n_substeps × dt_sim = 5 × 2 ms = 10 ms per env step
  Model is built once via MjSpec (robot.xml + target box freejoint).

Observation layout (45 floats)
───────────────────────────────
  [ 0: 3]  base_pos         world xyz
  [ 3: 4]  base_yaw         yaw extracted from freejoint quat
  [ 4: 7]  base_linvel      world xyz (m/s)
  [ 7:10]  base_angvel      world xyz (rad/s)
  [10:16]  arm_joint_pos    q1..q6, normalised to [-1, 1] by joint limits
  [16:22]  arm_joint_vel    dq1..dq6, normalised to [-1, 1] by vel limits
  [22:24]  finger_pos       [left, right], normalised to [0, 1]
  [24:26]  touch            [left, right] fingertip contact force
  [26:29]  ee_pos           world xyz (m)
  [29:33]  ee_quat          world wxyz
  [33:36]  wrist_force      normalised by wrist_force_max
  [36:39]  wrist_torque     normalised by wrist_torque_max
  [39:42]  target_pos       world xyz (m)
  [42:45]  rel_target       target_pos − ee_pos

Action layout (9 floats, all in [−1, 1])
─────────────────────────────────────────
  [0]    wheel_left    → ± wheel_vel_max rad/s
  [1]    wheel_right   → ± wheel_vel_max rad/s
  [2:8]  arm_j1..j6   → actuator ctrlrange (absolute position target)
  [8]    gripper       → [0, finger_pos_max] m
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robot.constants import ROBOT_XML_PATH, KF_HOME
from robot.workspace import DEFAULT_LIMITS, WorkspaceLimits

# ── observation / action dimensions ──────────────────────────────────────────

OBS_DIM: int = 45
ACT_DIM: int = 9

# ── sensordata slice indices (mirrors sensors.py _S layout exactly) ──────────

_SD_LINVEL   = slice(10, 13)
_SD_ANGVEL   = slice(13, 16)
_SD_JPOS     = slice(16, 22)
_SD_JVEL     = slice(22, 28)
_SD_WFORCE   = slice(28, 31)
_SD_WTORQUE  = slice(31, 34)
_SD_FINGERL  = 34
_SD_FINGERR  = 35
_SD_TOUCHL   = 36
_SD_TOUCHR   = 37
_SD_EEPOS    = slice(38, 41)
_SD_EEQUAT   = slice(41, 45)

# ── qpos slice constants ──────────────────────────────────────────────────────

_BASE_POS  = slice(0, 3)
_BASE_QUAT = slice(3, 7)

# ── physics constants ─────────────────────────────────────────────────────────

_TARGET_FLOOR_Z   = 0.025   # m — box half-height; sits exactly on floor
_LIFT_THRESHOLD   = 0.020   # m above floor → lift reward begins
_SUCCESS_HEIGHT   = 0.200   # m above floor → episode success
_GRASP_CLOSE_MAX  = 0.020   # m — finger ≤ this and touching → "grasped"

# ── reward coefficients ───────────────────────────────────────────────────────

_R_NAV      =  2.0   # per metre of base progress toward target
_R_REACH    =  5.0   # per metre of EE progress toward target
_R_CONTACT  =  0.10  # both fingertips in contact
_R_GRASP    =  0.50  # contact + gripper partially closed
_R_LIFT     =  5.0   # per metre of lift above threshold (dense)
_R_SUCCESS  = 10.0   # terminal bonus
_R_TIME     = -0.01  # per step (encourages efficiency)


# ═══════════════════════════════════════════════════════════════════════════════

class ManipulationEnv(gym.Env):
    """
    Gymnasium environment: AutoRobo v1 picking a box from the floor.

    Parameters
    ----------
    render_mode : str or None
        "rgb_array" returns an (H, W, 3) image from env.render().
    max_steps : int
        Hard episode length limit (default 500 ≈ 5 s at 10 ms/step).
    n_substeps : int
        MuJoCo steps per env step.  5 × 2 ms = 10 ms per policy call.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    _TARGET_X_RANGE = (0.40, 0.85)   # m in front of robot
    _TARGET_Y_RANGE = (-0.30, 0.30)  # m left/right

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 500,
        n_substeps: int = 5,
    ) -> None:
        super().__init__()
        self.render_mode  = render_mode
        self._max_steps   = max_steps
        self._n_substeps  = n_substeps

        self._model = self._build_model()
        self._data  = mujoco.MjData(self._model)
        self._renderer: Optional[mujoco.Renderer] = None

        self._cache_indices()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        # Per-episode state (initialised in reset)
        self._step_count:    int   = 0
        self._prev_ee_dist:  float = 0.0
        self._prev_base_dist: float = 0.0

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self._model, self._data, self._home_kf_id)

        # Randomise target position (in front of robot, on the floor)
        tx = float(self.np_random.uniform(*self._TARGET_X_RANGE))
        ty = float(self.np_random.uniform(*self._TARGET_Y_RANGE))

        qa = self._target_qadr
        self._data.qpos[qa : qa + 3]     = [tx, ty, _TARGET_FLOOR_Z]
        self._data.qpos[qa + 3]           = 1.0   # quaternion w (identity)
        self._data.qpos[qa + 4 : qa + 7] = 0.0   # quaternion xyz = 0
        self._data.qvel[self._target_vadr : self._target_vadr + 6] = 0.0

        mujoco.mj_forward(self._model, self._data)

        self._step_count = 0
        self._prev_ee_dist, self._prev_base_dist = self._current_dists()

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        ctrl = self._scale_action(np.asarray(action, dtype=np.float64))
        np.copyto(self._data.ctrl, ctrl)

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs = self._get_obs()

        reward, terminated = self._compute_reward()
        truncated = self._step_count >= self._max_steps

        self._prev_ee_dist, self._prev_base_dist = self._current_dists()

        return obs, float(reward), terminated, truncated, self._build_info(terminated)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera="rgbd_cam")
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── model construction ────────────────────────────────────────────────────

    @staticmethod
    def _build_model() -> mujoco.MjModel:
        """
        Load robot.xml via MjSpec and programmatically add a pickable target box.
        Keeps robot.xml clean — no scene content in the robot definition.
        """
        spec = mujoco.MjSpec.from_file(str(ROBOT_XML_PATH))

        target       = spec.worldbody.add_body()
        target.name  = "target_object"
        target.pos   = [0.7, 0.0, _TARGET_FLOOR_Z]

        fj           = target.add_freejoint()
        fj.name      = "target_joint"

        geom          = target.add_geom()
        geom.name     = "target_geom"
        geom.type     = mujoco.mjtGeom.mjGEOM_BOX
        geom.size     = [0.025, 0.025, 0.025]   # 5×5×5 cm box
        geom.mass     = 0.20
        geom.rgba     = [1.0, 0.35, 0.05, 1.0]
        geom.friction = [1.5, 0.5, 0.5]

        return spec.compile()

    def _cache_indices(self) -> None:
        m = self._model
        self._home_kf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, KF_HOME)

        tj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
        self._target_qadr = int(m.jnt_qposadr[tj_id])
        self._target_vadr = int(m.jnt_dofadr[tj_id])

        # (6, 2) ctrlrange for arm actuators (indices 2..7 in ctrl)
        self._arm_ctrlrange: np.ndarray = m.actuator_ctrlrange[2:8].copy()

    # ── observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        qpos = self._data.qpos
        sd   = self._data.sensordata
        lim  = DEFAULT_LIMITS

        base_pos    = qpos[_BASE_POS].copy()
        base_yaw    = _quat_to_yaw(qpos[_BASE_QUAT])
        base_linvel = sd[_SD_LINVEL].copy()
        base_angvel = sd[_SD_ANGVEL].copy()

        arm_pos_norm = _normalise_arm_pos(sd[_SD_JPOS], lim)
        arm_vel_norm = np.clip(sd[_SD_JVEL] / lim.joint_vel_max, -1.0, 1.0)

        finger_l = float(sd[_SD_FINGERL]) / lim.finger_pos_max
        finger_r = float(sd[_SD_FINGERR]) / lim.finger_pos_max
        touch_l  = float(sd[_SD_TOUCHL])
        touch_r  = float(sd[_SD_TOUCHR])

        ee_pos  = sd[_SD_EEPOS].copy()
        ee_quat = sd[_SD_EEQUAT].copy()

        wrist_force  = np.clip(sd[_SD_WFORCE]  / lim.wrist_force_max,  -1.0, 1.0)
        wrist_torque = np.clip(sd[_SD_WTORQUE] / lim.wrist_torque_max, -1.0, 1.0)

        qa         = self._target_qadr
        target_pos = qpos[qa : qa + 3].copy()
        rel_target = target_pos - ee_pos

        return np.concatenate([
            base_pos,                        # [0:3]
            [base_yaw],                      # [3]
            base_linvel,                     # [4:7]
            base_angvel,                     # [7:10]
            arm_pos_norm,                    # [10:16]
            arm_vel_norm,                    # [16:22]
            [finger_l, finger_r],            # [22:24]
            [touch_l,  touch_r],             # [24:26]
            ee_pos,                          # [26:29]
            ee_quat,                         # [29:33]
            wrist_force,                     # [33:36]
            wrist_torque,                    # [36:39]
            target_pos,                      # [39:42]
            rel_target,                      # [42:45]
        ]).astype(np.float32)                # total = 45

    # ── reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> tuple[float, bool]:
        sd   = self._data.sensordata
        qpos = self._data.qpos

        ee_pos     = sd[_SD_EEPOS]
        qa         = self._target_qadr
        target_pos = qpos[qa : qa + 3]
        base_pos   = qpos[_BASE_POS]

        ee_dist   = float(np.linalg.norm(target_pos - ee_pos))
        base_dist = float(np.linalg.norm(target_pos[:2] - base_pos[:2]))

        touch_l = float(sd[_SD_TOUCHL])
        touch_r = float(sd[_SD_TOUCHR])
        f_l     = float(sd[_SD_FINGERL])
        f_r     = float(sd[_SD_FINGERR])

        reward: float = 0.0

        # Navigation shaping — base approaching target horizontally
        reward += _R_NAV * (self._prev_base_dist - base_dist)

        # Reach shaping — EE approaching target in 3D
        reward += _R_REACH * (self._prev_ee_dist - ee_dist)

        # Contact bonus
        has_contact = touch_l > 0.05 and touch_r > 0.05
        if has_contact:
            reward += _R_CONTACT

        # Grasp bonus — touching + fingers not fully open
        grasping = has_contact and f_l < _GRASP_CLOSE_MAX and f_r < _GRASP_CLOSE_MAX
        if grasping:
            reward += _R_GRASP

        # Lift reward (dense) — only above the threshold
        lift = float(target_pos[2]) - _TARGET_FLOOR_Z
        if lift > _LIFT_THRESHOLD:
            reward += _R_LIFT * lift

        # Success terminal bonus
        terminated = lift > _SUCCESS_HEIGHT
        if terminated:
            reward += _R_SUCCESS

        reward += _R_TIME
        return reward, terminated

    def _current_dists(self) -> tuple[float, float]:
        sd     = self._data.sensordata
        qpos   = self._data.qpos
        ee_pos = sd[_SD_EEPOS]
        qa     = self._target_qadr
        tgt    = qpos[qa : qa + 3]
        base   = qpos[_BASE_POS]
        return (
            float(np.linalg.norm(tgt     - ee_pos)),
            float(np.linalg.norm(tgt[:2] - base[:2])),
        )

    # ── action scaling ────────────────────────────────────────────────────────

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalised action ∈ [−1, 1]^9 → ctrl vector in physical units."""
        lim  = DEFAULT_LIMITS
        ctrl = np.zeros(self._model.nu, dtype=np.float64)

        # Wheels: [-1, 1] → [−wheel_vel_max, +wheel_vel_max] rad/s
        ctrl[0] = action[0] * lim.wheel_vel_max
        ctrl[1] = action[1] * lim.wheel_vel_max

        # Arm joints: [-1, 1] → actuator ctrlrange (absolute position target)
        for i in range(6):
            lo, hi    = self._arm_ctrlrange[i]
            ctrl[2+i] = lo + (action[2+i] + 1.0) / 2.0 * (hi - lo)

        # Gripper: [-1, 1] → [0, finger_pos_max] m
        ctrl[8] = (action[8] + 1.0) / 2.0 * lim.finger_pos_max

        return ctrl

    # ── info dict ─────────────────────────────────────────────────────────────

    def _build_info(self, terminated: bool) -> dict[str, Any]:
        qa = self._target_qadr
        th = float(self._data.qpos[qa + 2])
        return {
            "success":        terminated,
            "step":           self._step_count,
            "target_height":  th,
            "lift":           th - _TARGET_FLOOR_Z,
            "ee_to_target":   self._prev_ee_dist,
            "base_to_target": self._prev_base_dist,
        }


# ── module-level helpers ──────────────────────────────────────────────────────

def _quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw from quaternion [w, x, y, z] (MuJoCo wxyz convention)."""
    w, x, y, z = q
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _normalise_arm_pos(q: np.ndarray, lim: WorkspaceLimits) -> np.ndarray:
    """Linearly map arm joint angles to [−1, 1] using per-joint limits."""
    mid = (lim.joint_pos_hi + lim.joint_pos_lo) / 2.0
    rng = (lim.joint_pos_hi - lim.joint_pos_lo) / 2.0
    return np.clip((q - mid) / rng, -1.0, 1.0)
