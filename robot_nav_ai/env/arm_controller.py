"""
env/arm_controller.py — Joint-level arm controller for the manipulation layer.

Converts delta end-effector commands (from GraspActionProcessor) into
joint velocity targets using Jacobian pseudo-inverse (damped least squares).
Includes force-torque feedback and self-collision checking.

Architecture
────────────
  Input  : GraspPhysicalAction (delta_pos, delta_euler, gripper_cmd)
  Output : ArmControlResult    (joint velocities, gripper target, safety flags)

IK method  — Damped Least Squares (DLS)
────────────────────────────────────────
  1. Query MuJoCo for the 6×6 Jacobian J at the EE site.
  2. Compute desired EE twist: ξ = [delta_pos/dt, delta_euler/dt]
  3. Joint velocity:  dq = Jᵀ (J Jᵀ + λ²I)⁻¹ ξ   (DLS formula)
  4. Clip dq to joint velocity limits.
  5. Target joint position = current_q + dq × dt

  λ (damping) prevents large joint motions near singularities.
  DLS is preferred over pure pseudo-inverse because it degrades gracefully
  at singularity rather than producing explosive joint commands.

Force-torque feedback
──────────────────────
  The wrist F/T sensor (sensordata[28:34]) is read every step.
  If the resultant force or torque exceeds the safety thresholds defined
  in WorkspaceLimits, the controller:
    1. Returns wrist_safe=False in ArmControlResult.
    2. Sets all joint velocity targets to zero (hard stop).
  The environment (Phase 6 safety layer) then terminates the episode.

Self-collision detection
─────────────────────────
  After physics is stepped, MuJoCo populates data.contact with all active
  contact pairs.  A self-collision is any contact where BOTH geoms belong
  to the robot (geomgroup 1 in the MJCF model).

  If a self-collision is detected:
    1. Returns self_collision=True in ArmControlResult.
    2. Reverses the joint command (zero velocity) to prevent further
       penetration.

Usage
─────
    ctrl = ArmController(model, data, ee_site_id, cfg=ArmControllerConfig())
    ctrl.reset()

    result = ctrl.step(grasp_action)
    if result.wrist_safe and not result.self_collision:
        data.ctrl[arm_ctrl_slice] = result.joint_pos_target
        data.ctrl[gripper_ctrl_id] = result.gripper_target
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from env.grasp_action import GraspPhysicalAction, GripperCmd
from robot.workspace import DEFAULT_LIMITS, WorkspaceLimits, check_wrist_safety


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ArmControllerConfig:
    """
    Configuration for ArmController.

    damping          : DLS damping factor λ (larger = safer near singularities)
    max_joint_vel    : per-joint velocity cap [rad/s] applied after DLS
    dt               : controller timestep in seconds (must match env dt_env)
    robot_geomgroup  : MuJoCo geomgroup index for robot arm geoms
    n_arm_joints     : number of arm joints (6 for our 6-DOF arm)
    arm_ctrl_start   : index of first arm actuator in data.ctrl (after 2 wheels)
    gripper_ctrl_idx : index of gripper actuator in data.ctrl
    gripper_open_pos : gripper target position when OPEN [m]
    gripper_close_pos: gripper target position when CLOSED [m]
    """
    damping:           float = 0.05
    max_joint_vel:     float = 1.5      # rad/s — applied after DLS, before limits
    dt:                float = 0.010    # 10 ms — must match n_substeps × model dt
    robot_geomgroup:   int   = 1        # geomgroup 1 = robot in our MJCF
    n_arm_joints:      int   = 6
    arm_ctrl_start:    int   = 2        # ctrl[0]=wheel_left, ctrl[1]=wheel_right
    gripper_ctrl_idx:  int   = 8
    gripper_open_pos:  float = 0.040    # m — matches finger_pos_max
    gripper_close_pos: float = 0.0      # m — fully closed
    qpos_arm_start:    int   = 7        # qpos offset to first arm joint (7 = after base freejoint)


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class ArmControlResult:
    """
    Output of ArmController.step().

    Fields
    ------
    joint_pos_target : (6,) float64 — target joint positions [rad]
    gripper_target   : float — target finger position [m]
    dq               : (6,) float64 — joint velocity used this step [rad/s]
    wrist_safe       : False if wrist F/T exceeded threshold (hard stop)
    self_collision   : True if self-collision detected this step
    wrist_force_mag  : scalar wrist force magnitude [N]
    wrist_torque_mag : scalar wrist torque magnitude [N·m]
    reason           : human-readable flag reason (empty if ok)
    """
    joint_pos_target: np.ndarray   # (6,) float64
    gripper_target:   float
    dq:               np.ndarray   # (6,) float64
    wrist_safe:       bool
    self_collision:   bool
    wrist_force_mag:  float
    wrist_torque_mag: float
    reason:           str = ""

    def __repr__(self) -> str:
        flags = []
        if not self.wrist_safe:
            flags.append(f"WRIST_UNSAFE({self.wrist_force_mag:.1f}N)")
        if self.self_collision:
            flags.append("SELF_COLLISION")
        flag_str = " ".join(flags) or "ok"
        dq_max = float(np.max(np.abs(self.dq)))
        return (f"ArmControlResult(dq_max={dq_max:.3f}rad/s, "
                f"gripper={self.gripper_target:.3f}m, {flag_str})")


# ── controller ────────────────────────────────────────────────────────────────

class ArmController:
    """
    Jacobian-based arm controller with F/T feedback and self-collision checking.

    Parameters
    ----------
    model       : mujoco.MjModel — simulation model
    data        : mujoco.MjData  — simulation data (mutated by step())
    ee_site_id  : int — MuJoCo site id for the end-effector TCP
    limits      : WorkspaceLimits (defaults to DEFAULT_LIMITS)
    cfg         : ArmControllerConfig
    """

    def __init__(
        self,
        model:      mujoco.MjModel,
        data:       mujoco.MjData,
        ee_site_id: int,
        limits:     WorkspaceLimits   = DEFAULT_LIMITS,
        cfg:        ArmControllerConfig = ArmControllerConfig(),
    ) -> None:
        self._model      = model
        self._data       = data
        self._ee_site_id = ee_site_id
        self._limits     = limits
        self.cfg         = cfg

        nv = model.nv
        self._jacp  = np.zeros((3, nv), dtype=np.float64)
        self._jacr  = np.zeros((3, nv), dtype=np.float64)

        self._arm_slice = slice(
            cfg.arm_ctrl_start,
            cfg.arm_ctrl_start + cfg.n_arm_joints,
        )
        # sensordata slices for wrist F/T (mirrors sensors.py layout)
        self._sd_wforce  = slice(28, 31)
        self._sd_wtorque = slice(31, 34)

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Call once per episode — no persistent state to reset currently."""
        pass

    def step(self, action: GraspPhysicalAction) -> ArmControlResult:
        """
        Compute joint position targets for one env step.

        Parameters
        ----------
        action : GraspPhysicalAction from GraspActionProcessor.process()

        Returns
        -------
        ArmControlResult — apply joint_pos_target to data.ctrl if flags ok.
        """
        # ── 1. read wrist F/T ─────────────────────────────────────────────────
        wrist_force  = self._data.sensordata[self._sd_wforce].copy()
        wrist_torque = self._data.sensordata[self._sd_wtorque].copy()
        ft_check     = check_wrist_safety(wrist_force, wrist_torque, self._limits)

        f_mag = float(np.linalg.norm(wrist_force))
        t_mag = float(np.linalg.norm(wrist_torque))

        if not ft_check.ok:
            q_curr = self._current_q()
            return ArmControlResult(
                joint_pos_target = q_curr,
                gripper_target   = self._gripper_target(action.gripper_cmd),
                dq               = np.zeros(self.cfg.n_arm_joints),
                wrist_safe       = False,
                self_collision   = False,
                wrist_force_mag  = f_mag,
                wrist_torque_mag = t_mag,
                reason           = ft_check.reason,
            )

        # ── 2. Jacobian pseudo-inverse IK (DLS) ───────────────────────────────
        mujoco.mj_jacSite(
            self._model, self._data,
            self._jacp, self._jacr,
            self._ee_site_id,
        )

        n = self.cfg.n_arm_joints
        arm_start = self.cfg.arm_ctrl_start

        Jp = self._jacp[:, arm_start: arm_start + n]   # (3, n)
        Jr = self._jacr[:, arm_start: arm_start + n]   # (3, n)
        J  = np.vstack([Jp, Jr])                        # (6, n)

        xi = np.concatenate([
            action.delta_pos.astype(np.float64) / self.cfg.dt,
            action.delta_euler.astype(np.float64) / self.cfg.dt,
        ])  # (6,) — desired EE twist

        lam2 = self.cfg.damping ** 2
        JJT  = J @ J.T                                  # (6, 6)
        dq   = J.T @ np.linalg.solve(JJT + lam2 * np.eye(6), xi)  # (n,)

        # ── 3. velocity and joint-position clamping ────────────────────────────
        dq = np.clip(dq, -self.cfg.max_joint_vel, self.cfg.max_joint_vel)

        vel_limits = self._limits.joint_vel_max[:n]
        dq = np.clip(dq, -vel_limits, vel_limits)

        q_curr   = self._current_q()
        q_target = q_curr + dq * self.cfg.dt
        q_target = np.clip(
            q_target,
            self._limits.joint_pos_lo[:n],
            self._limits.joint_pos_hi[:n],
        )

        # ── 4. self-collision check ───────────────────────────────────────────
        self_coll, coll_reason = self._check_self_collision()

        if self_coll:
            q_target = q_curr   # freeze arm
            dq       = np.zeros(n)

        # ── 5. gripper target ─────────────────────────────────────────────────
        gripper_pos = self._gripper_target(action.gripper_cmd)

        return ArmControlResult(
            joint_pos_target = q_target,
            gripper_target   = gripper_pos,
            dq               = dq,
            wrist_safe       = True,
            self_collision   = self_coll,
            wrist_force_mag  = f_mag,
            wrist_torque_mag = t_mag,
            reason           = coll_reason,
        )

    def ee_position(self) -> np.ndarray:
        """Return current end-effector position (world frame) as (3,) float64."""
        return self._data.site_xpos[self._ee_site_id].copy()

    def ee_quaternion(self) -> np.ndarray:
        """Return current EE orientation as (4,) float64 quaternion (wxyz)."""
        mat = self._data.site_xmat[self._ee_site_id].reshape(3, 3)
        return _mat_to_quat(mat)

    # ── internals ─────────────────────────────────────────────────────────────

    def _current_q(self) -> np.ndarray:
        """Return current arm joint positions as (n_arm_joints,) float64."""
        n   = self.cfg.n_arm_joints
        s   = self.cfg.qpos_arm_start
        return self._data.qpos[s: s + n].copy()

    def _gripper_target(self, cmd: GripperCmd) -> float:
        """Map GripperCmd enum to a finger position target [m]."""
        if cmd == GripperCmd.OPEN:
            return self.cfg.gripper_open_pos
        return self.cfg.gripper_close_pos

    def _check_self_collision(self) -> tuple[bool, str]:
        """
        Scan active contacts for any pair where both geoms are in the robot
        geomgroup (group == cfg.robot_geomgroup).

        Returns (is_collision, reason_string).
        """
        g = self.cfg.robot_geomgroup
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            g1 = self._model.geom_group[contact.geom1]
            g2 = self._model.geom_group[contact.geom2]
            if g1 == g and g2 == g:
                n1 = mujoco.mj_id2name(
                    self._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                ) or str(contact.geom1)
                n2 = mujoco.mj_id2name(
                    self._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                ) or str(contact.geom2)
                return True, f"self-collision: {n1} ↔ {n2}"
        return False, ""

    def __repr__(self) -> str:
        return (
            f"ArmController(damping={self.cfg.damping}, "
            f"max_joint_vel={self.cfg.max_joint_vel}rad/s, "
            f"dt={self.cfg.dt}s)"
        )


# ── quaternion helper ─────────────────────────────────────────────────────────

def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)
