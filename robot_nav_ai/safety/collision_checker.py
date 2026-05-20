"""
safety/collision_checker.py — Pre-execution trajectory collision checker.

Checks a proposed joint configuration (or interpolated path) for collisions
*before* writing it to data.ctrl.  This prevents the arm from ever entering
a configuration that causes penetration — safer than the post-hoc contact
scan in ArmController._check_self_collision().

How it works
────────────
  1. Save current qpos and ctrl.
  2. Write the proposed q_target to qpos[arm_slice].
  3. Call mujoco.mj_fwdPosition() — recomputes kinematics and contacts
     WITHOUT integrating dynamics (no velocity/force side-effects).
  4. Scan data.contact for any violating pair (self-collision or env-collision).
  5. Restore original qpos and ctrl.

Collision categories
─────────────────────
  SELF      : both contact geoms belong to the robot (geomgroup == robot_geomgroup)
  ENV       : one geom is robot, the other is environment (table, walls, objects)
  NONE      : no collision detected — safe to execute

Trajectory check (check_path)
──────────────────────────────
  Linearly interpolates between q_start and q_end in n_waypoints steps and
  calls check_qpos() at each waypoint.  Returns the first collision found,
  or a NONE result if the whole path is clear.

Usage
─────
    checker = TrajectoryCollisionChecker(model, data, estop)

    # single-point check before applying q_target
    result = checker.check_qpos(q_target)
    if result.collision:
        estop.trigger(result.reason)

    # whole path check
    result = checker.check_path(q_start=current_q, q_end=target_q, n_waypoints=5)
    if result.collision:
        pass  # estop already triggered
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import mujoco
import numpy as np

log = logging.getLogger(__name__)


# ── collision category ────────────────────────────────────────────────────────

class CollisionType(Enum):
    NONE  = "none"
    SELF  = "self_collision"
    ENV   = "env_collision"


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CollisionCheckConfig:
    """
    Configuration for the trajectory collision checker.

    robot_geomgroup  : MuJoCo geomgroup index assigned to all robot geoms
                       (set in the MJCF via <geom group="1"/>).
    check_self       : whether to detect robot–robot contacts.
    check_env        : whether to detect robot–environment contacts.
    n_arm_joints     : number of arm DOF (matches ArmControllerConfig).
    qpos_arm_start   : first arm joint index in qpos (7 = after base freejoint).
    arm_ctrl_start   : first arm actuator index in ctrl (2 = after 2 wheel actuators).
    """
    robot_geomgroup: int   = 1
    check_self:      bool  = True
    check_env:       bool  = True
    n_arm_joints:    int   = 6
    qpos_arm_start:  int   = 7
    arm_ctrl_start:  int   = 2


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class CollisionResult:
    """
    Outcome of a single collision check.

    Fields
    ------
    collision      : True if any collision was detected
    collision_type : SELF, ENV, or NONE
    geom1_name     : name of first colliding geom (empty if no collision)
    geom2_name     : name of second colliding geom
    waypoint_index : which waypoint triggered the collision (-1 = single check)
    reason         : human-readable description
    """
    collision:      bool
    collision_type: CollisionType
    geom1_name:     str  = ""
    geom2_name:     str  = ""
    waypoint_index: int  = -1
    reason:         str  = ""

    @property
    def safe(self) -> bool:
        return not self.collision

    def __repr__(self) -> str:
        if not self.collision:
            return "CollisionResult(safe)"
        return (f"CollisionResult({self.collision_type.value}: "
                f"{self.geom1_name} ↔ {self.geom2_name})")


_SAFE = CollisionResult(collision=False, collision_type=CollisionType.NONE)


# ── checker ───────────────────────────────────────────────────────────────────

class TrajectoryCollisionChecker:
    """
    Pre-execution collision checker using MuJoCo forward kinematics.

    Saves and restores simulation state so collision probing has no side-effects
    on the running episode.

    Parameters
    ----------
    model  : mujoco.MjModel
    data   : mujoco.MjData
    estop  : SimEStop — triggered automatically on collision detection
    cfg    : CollisionCheckConfig
    """

    def __init__(
        self,
        model,
        data,
        estop,
        cfg: CollisionCheckConfig = CollisionCheckConfig(),
    ) -> None:
        self._model  = model
        self._data   = data
        self._estop  = estop
        self.cfg     = cfg

        self._arm_qpos_slice = slice(
            cfg.qpos_arm_start,
            cfg.qpos_arm_start + cfg.n_arm_joints,
        )
        self._check_count     = 0
        self._collision_count = 0

    # ── public API ────────────────────────────────────────────────────────────

    def check_qpos(self, q_proposed: np.ndarray) -> CollisionResult:
        """
        Check whether a proposed arm joint configuration causes a collision.

        The simulation state is fully restored after the probe — calling this
        method has no effect on the ongoing episode.

        Parameters
        ----------
        q_proposed : (n_arm_joints,) joint positions to probe [rad]

        Returns
        -------
        CollisionResult — safe=True if no collision found.
        If a collision is found, estop.trigger() is called automatically.
        """
        self._check_count += 1

        # ── save state ────────────────────────────────────────────────────────
        saved_qpos = self._data.qpos.copy()
        saved_ctrl = self._data.ctrl.copy()

        try:
            # ── probe: set proposed joint positions ───────────────────────────
            self._data.qpos[self._arm_qpos_slice] = q_proposed
            mujoco.mj_fwdPosition(self._model, self._data)

            # ── scan contacts ─────────────────────────────────────────────────
            result = self._scan_contacts()

        finally:
            # ── always restore — even if scan raises ──────────────────────────
            self._data.qpos[:] = saved_qpos
            self._data.ctrl[:] = saved_ctrl
            mujoco.mj_fwdPosition(self._model, self._data)

        if result.collision:
            self._collision_count += 1
            log.warning("Collision detected at proposed qpos: %s", result.reason)
            self._estop.trigger(reason=result.reason, source="collision_checker")

        return result

    def check_path(
        self,
        q_start:     np.ndarray,
        q_end:       np.ndarray,
        n_waypoints: int = 5,
    ) -> CollisionResult:
        """
        Check a linearly interpolated joint-space path for collisions.

        Probes n_waypoints evenly-spaced configurations between q_start and
        q_end (inclusive of both endpoints).  Returns the FIRST collision found,
        or a NONE result if the whole path is clear.

        Parameters
        ----------
        q_start     : (n_arm_joints,) start joint positions [rad]
        q_end       : (n_arm_joints,) target joint positions [rad]
        n_waypoints : number of points to check along the path (min 2)

        Returns
        -------
        CollisionResult from the first colliding waypoint, or safe result.
        """
        n_waypoints = max(2, n_waypoints)
        for i, t in enumerate(np.linspace(0.0, 1.0, n_waypoints)):
            q_probe = q_start + t * (q_end - q_start)
            result  = self.check_qpos(q_probe)
            if result.collision:
                return CollisionResult(
                    collision      = True,
                    collision_type = result.collision_type,
                    geom1_name     = result.geom1_name,
                    geom2_name     = result.geom2_name,
                    waypoint_index = i,
                    reason         = f"path waypoint {i}/{n_waypoints-1}: {result.reason}",
                )
        return _SAFE

    # ── internals ─────────────────────────────────────────────────────────────

    def _scan_contacts(self) -> CollisionResult:
        """
        Scan data.contact for self- or env-collision pairs.

        Robot geoms have geomgroup == cfg.robot_geomgroup.
        Environment geoms have any other group (or group 0 by default).
        """
        g_robot = self.cfg.robot_geomgroup

        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            g1 = int(self._model.geom_group[contact.geom1])
            g2 = int(self._model.geom_group[contact.geom2])

            is_robot_1 = g1 == g_robot
            is_robot_2 = g2 == g_robot

            if is_robot_1 and is_robot_2 and self.cfg.check_self:
                n1 = self._geom_name(contact.geom1)
                n2 = self._geom_name(contact.geom2)
                return CollisionResult(
                    collision      = True,
                    collision_type = CollisionType.SELF,
                    geom1_name     = n1,
                    geom2_name     = n2,
                    reason         = f"self-collision: {n1} ↔ {n2}",
                )

            if (is_robot_1 != is_robot_2) and self.cfg.check_env:
                r_geom = contact.geom1 if is_robot_1 else contact.geom2
                e_geom = contact.geom2 if is_robot_1 else contact.geom1
                rn = self._geom_name(r_geom)
                en = self._geom_name(e_geom)
                return CollisionResult(
                    collision      = True,
                    collision_type = CollisionType.ENV,
                    geom1_name     = rn,
                    geom2_name     = en,
                    reason         = f"env-collision: robot:{rn} ↔ env:{en}",
                )

        return _SAFE

    def _geom_name(self, geom_id: int) -> str:
        name = mujoco.mj_id2name(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        return name if name else str(geom_id)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def check_count(self) -> int:
        """Total number of qpos probes performed."""
        return self._check_count

    @property
    def collision_count(self) -> int:
        """Number of probes that detected a collision."""
        return self._collision_count

    def __repr__(self) -> str:
        return (f"TrajectoryCollisionChecker("
                f"checks={self._check_count}, "
                f"collisions={self._collision_count}, "
                f"self={self.cfg.check_self}, env={self.cfg.check_env})")
