"""
robot/robot_controller.py — Low-level arm joint controller (Block 5.2).

Why a separate controller instead of using the existing ArmController
----------------------------------------------------------------------
The existing manipulation/arm_controller.py depends on BaseRobotInterface
(Phase 8 stub — not yet implemented).  This module is self-contained:
it owns the MuJoCo model/data directly and drives joints by writing to
data.ctrl, matching exactly how the TidyBot physics simulation works.

Interpolation design
--------------------
Real servos have finite velocity limits.  We enforce MAX_RAD_PER_STEP
(0.1 rad/step by default) to prevent teleportation and to produce smooth
joint motion that a real Kinova Gen3 could follow.  Each call to step()
advances all arm joints one step toward their targets:

    Δq = clip(q_target − q_current, −max_step, +max_step)
    q_new = q_current + Δq

At STEP_HZ = 100 Hz and max 0.1 rad/step, the largest single-joint
travel (π rad) takes ≤ 32 steps = 0.32 s — well within the 2-second
acceptance window.

Actuator indices (from tidybot.xml)
------------------------------------
act[0] joint_x, act[1] joint_y, act[2] joint_th   (base — not used here)
act[3] joint_1 … act[9] joint_7                    (Kinova arm)
act[10] fingers_actuator (gripper, ctrl 0–255)

GRIPPER_OPEN  = 0    (fully open)
GRIPPER_CLOSE = 200  (contact force — not full 255 to avoid over-squeeze)

Usage
-----
    from robot.robot_controller import RobotController

    ctrl = RobotController()
    ctrl.set_joint_targets(q_7dof)      # set goal
    while not ctrl.is_at_target():
        ctrl.step()                      # advance 1 physics step
    print(ctrl.get_joints())            # current arm joints
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import mujoco
    _MJ_OK = True
except ImportError:
    _MJ_OK = False

from robot.kinematics import (
    TidyBotKinematics, ARM_QPOS_SLICE, HOME_QPOS, EE_SITE,
)

# ── constants ─────────────────────────────────────────────────────────────────

MAX_RAD_PER_STEP: float = 0.1   # max joint travel per step (0.1 rad)
STEP_HZ:          int   = 100   # simulated control frequency (Hz)
AT_TARGET_TOL:    float = 1e-3  # "at target" threshold (radians)

# Actuator index ranges
_ARM_ACT_SLICE    = slice(3, 10)    # joint_1 … joint_7 actuators
_GRIPPER_ACT_IDX  = 10             # fingers_actuator index

GRIPPER_OPEN:  float = 0.0
GRIPPER_CLOSE: float = 200.0       # ≤ 255 to avoid over-squeeze


# ── controller ────────────────────────────────────────────────────────────────

class RobotController:
    """
    Step-based arm joint controller backed by MuJoCo.

    Maintains a target joint configuration and advances joints toward it
    at most MAX_RAD_PER_STEP per call to step().

    Parameters
    ----------
    kin          : existing TidyBotKinematics (optional — creates one if None)
    max_rad_step : maximum joint change per step  (default 0.1 rad)
    at_tol       : "at target" position tolerance (default 1e-3 rad)
    """

    def __init__(
        self,
        kin:          Optional[TidyBotKinematics] = None,
        max_rad_step: float = MAX_RAD_PER_STEP,
        at_tol:       float = AT_TARGET_TOL,
    ) -> None:
        self.kin          = kin or TidyBotKinematics()
        self.max_rad_step = max_rad_step
        self.at_tol       = at_tol

        # Initialise to home configuration
        self._q_current = HOME_QPOS[ARM_QPOS_SLICE].copy()   # (7,)
        self._q_target  = self._q_current.copy()             # (7,)
        self._gripper   = GRIPPER_OPEN

        # Sync MuJoCo state
        self._sync_to_mujoco()

        # Step counter
        self.total_steps: int = 0

    # ── joint API ─────────────────────────────────────────────────────────────

    def get_joints(self) -> np.ndarray:
        """Return current arm joint angles (7,) in radians."""
        return self._q_current.copy()

    def set_joints(self, q: np.ndarray) -> None:
        """
        Instantly teleport arm to joint configuration q (7,).
        Also updates the internal target to avoid unwanted drift.
        """
        q = np.asarray(q, dtype=np.float64)
        self._q_current = q.copy()
        self._q_target  = q.copy()
        self._sync_to_mujoco()

    def set_joint_targets(self, q_target: np.ndarray) -> None:
        """
        Set the joint target; subsequent step() calls interpolate toward it.

        Parameters
        ----------
        q_target : (7,) desired joint angles (radians)
        """
        self._q_target = np.asarray(q_target, dtype=np.float64).copy()

    # ── gripper API ───────────────────────────────────────────────────────────

    def open_gripper(self) -> None:
        """Command gripper to open (ctrl = 0)."""
        self._gripper = GRIPPER_OPEN
        self._sync_to_mujoco()

    def close_gripper(self) -> None:
        """Command gripper to close (ctrl = 200)."""
        self._gripper = GRIPPER_CLOSE
        self._sync_to_mujoco()

    # ── physics step ──────────────────────────────────────────────────────────

    def step(self, n: int = 1) -> None:
        """
        Advance the controller n steps.

        Each step moves each joint at most MAX_RAD_PER_STEP toward target.
        Writes to MuJoCo data.ctrl so the physics engine tracks the motion.
        """
        for _ in range(n):
            delta = self._q_target - self._q_current
            delta = np.clip(delta, -self.max_rad_step, self.max_rad_step)
            self._q_current = self._q_current + delta
            self._sync_to_mujoco()
            self.total_steps += 1

    def is_at_target(self) -> bool:
        """Return True when all joints are within at_tol of the target."""
        return bool(np.all(np.abs(self._q_target - self._q_current) < self.at_tol))

    def run_until_converged(self, max_steps: int = 500) -> Tuple[bool, int]:
        """
        Step until at target or max_steps exhausted.

        Returns
        -------
        (converged, steps_taken)
        """
        for i in range(max_steps):
            if self.is_at_target():
                return True, i
            self.step()
        return self.is_at_target(), max_steps

    # ── EE query ──────────────────────────────────────────────────────────────

    def get_ee_xyz(self) -> np.ndarray:
        """Current EE position in world frame (3,)."""
        return self.kin.data.site_xpos[self.kin._ee_id].copy()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sync_to_mujoco(self) -> None:
        """Write current joints and gripper to MuJoCo data and run FK."""
        self.kin.data.qpos[ARM_QPOS_SLICE] = self._q_current
        self.kin.data.ctrl[_ARM_ACT_SLICE] = self._q_current
        self.kin.data.ctrl[_GRIPPER_ACT_IDX] = self._gripper
        mujoco.mj_fwdPosition(self.kin.model, self.kin.data)

    def __repr__(self) -> str:
        dist = float(np.max(np.abs(self._q_target - self._q_current)))
        return (
            f"RobotController(at_target={self.is_at_target()}, "
            f"max_err={dist:.4f} rad, steps={self.total_steps})"
        )
