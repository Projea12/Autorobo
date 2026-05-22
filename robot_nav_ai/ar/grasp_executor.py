"""
ar/grasp_executor.py — Grasp execution state machine (Block 5.1).

State machine
-------------
    IDLE
      │  execute() called with a GraspPose
      ▼
    MOVING_TO_PREGRASP  — IK-solve pre-grasp target, set_joint_targets(),
      │                    step until is_at_target()
      ▼
    MOVING_TO_GRASP     — IK-solve grasp target, set_joint_targets(),
      │                    step until is_at_target()
      ▼
    CLOSING             — close_gripper(), wait CLOSE_SETTLE_STEPS
      │
      ▼
    LIFTING             — set_joint_targets(q_lift), step until at target
      │
      ▼
    DONE

Why this ordering
-----------------
Moving to pre-grasp first aligns the arm approach axis before it enters
the object's bounding volume.  If we went straight to the grasp point
from an arbitrary start pose, the gripper might clip the object or the
table on the way in.  The pre-grasp point is 15 cm back along the approach
axis — clear of the object — so any approach path from the home pose is
safe.

Why close before lift
---------------------
Grasping in the z-direction first ensures the object is secured before
any lift force is applied, reducing the chance of the object being pushed
or knocked over by the moving gripper.

Lift definition
---------------
A "lift" is a 10 cm upward displacement of the EE from the grasp pose.
We re-solve IK at (grasp_xyz + [0, 0, 0.10]) to get the lift joint target.
If IK fails (target unreachable), the lift is skipped and state → DONE.

Usage
-----
    from robot.kinematics import TidyBotKinematics
    from robot.robot_controller import RobotController
    from ar.grasp_planner import GraspPlanner
    from ar.grasp_pose import GraspApproach, ApproachType
    from ar.grasp_executor import GraspExecutor

    kin     = TidyBotKinematics()
    ctrl    = RobotController(kin)
    planner = GraspPlanner()
    approach = ...
    pose    = planner.plan(obj_xyz, approach)

    executor = GraspExecutor(ctrl, kin)
    result   = executor.execute(pose)
    print(result)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from ar.grasp_planner import GraspPose
from robot.kinematics  import TidyBotKinematics, ARM_QPOS_SLICE
from robot.robot_controller import RobotController

# ── constants ─────────────────────────────────────────────────────────────────

MAX_STEPS_PER_MOVE:  int   = 500    # step budget per move segment
CLOSE_SETTLE_STEPS:  int   = 20     # steps to hold gripper closed before lift
LIFT_HEIGHT_M:       float = 0.10   # how high to lift above grasp point


# ── state enum ────────────────────────────────────────────────────────────────

class GraspState(Enum):
    IDLE              = auto()
    MOVING_TO_PREGRASP = auto()
    MOVING_TO_GRASP   = auto()
    CLOSING           = auto()
    LIFTING           = auto()
    DONE              = auto()
    FAILED            = auto()


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """
    Summary of one GraspExecutor.execute() run.

    Attributes
    ----------
    success          : True if state machine reached DONE without failure
    final_state      : last GraspState reached
    states_visited   : ordered list of states (for test verification)
    total_steps      : total controller steps consumed
    elapsed_s        : wall-clock time
    fail_reason      : explanation if success is False
    """
    success:        bool
    final_state:    GraspState
    states_visited: List[GraspState]
    total_steps:    int
    elapsed_s:      float
    fail_reason:    str = ""

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED ({self.fail_reason})"
        path   = " → ".join(s.name for s in self.states_visited)
        return (
            f"ExecutionResult [{status}]\n"
            f"  path   : {path}\n"
            f"  steps  : {self.total_steps}\n"
            f"  elapsed: {self.elapsed_s*1000:.1f} ms"
        )


# ── executor ──────────────────────────────────────────────────────────────────

class GraspExecutor:
    """
    State-machine executor for a single grasp attempt.

    Parameters
    ----------
    controller : RobotController — owns the arm joint state
    kin        : TidyBotKinematics — used for IK solves
    max_steps  : max controller steps per move segment (default 500)
    """

    def __init__(
        self,
        controller: RobotController,
        kin:        TidyBotKinematics,
        max_steps:  int = MAX_STEPS_PER_MOVE,
    ) -> None:
        self.ctrl      = controller
        self.kin       = kin
        self.max_steps = max_steps

        self._state: GraspState = GraspState.IDLE

    @property
    def state(self) -> GraspState:
        return self._state

    # ── public API ────────────────────────────────────────────────────────────

    def execute(self, pose: GraspPose) -> ExecutionResult:
        """
        Run the full IDLE → DONE state machine for the given grasp pose.

        Parameters
        ----------
        pose : GraspPose from GraspPlanner.plan()

        Returns
        -------
        ExecutionResult
        """
        t0             = time.perf_counter()
        states_visited : List[GraspState] = [GraspState.IDLE]
        total_steps    = 0

        # ── 1. MOVING_TO_PREGRASP ────────────────────────────────────────────
        self._transition(GraspState.MOVING_TO_PREGRASP)
        states_visited.append(self._state)

        ik_pre = self.kin.ik(pose.pre_grasp_xyz, q_init=self.ctrl.get_joints())
        if not ik_pre.converged:
            return self._fail("IK failed for pre-grasp target",
                              states_visited, total_steps, t0)

        self.ctrl.set_joint_targets(ik_pre.q_arm)
        converged, steps = self.ctrl.run_until_converged(self.max_steps)
        total_steps += steps
        if not converged:
            return self._fail("Timed out reaching pre-grasp",
                              states_visited, total_steps, t0)

        # ── 2. MOVING_TO_GRASP ───────────────────────────────────────────────
        self._transition(GraspState.MOVING_TO_GRASP)
        states_visited.append(self._state)

        ik_grs = self.kin.ik(pose.grasp_xyz, q_init=self.ctrl.get_joints())
        if not ik_grs.converged:
            return self._fail("IK failed for grasp target",
                              states_visited, total_steps, t0)

        self.ctrl.set_joint_targets(ik_grs.q_arm)
        converged, steps = self.ctrl.run_until_converged(self.max_steps)
        total_steps += steps
        if not converged:
            return self._fail("Timed out reaching grasp pose",
                              states_visited, total_steps, t0)

        # ── 3. CLOSING ───────────────────────────────────────────────────────
        self._transition(GraspState.CLOSING)
        states_visited.append(self._state)

        self.ctrl.close_gripper()
        self.ctrl.step(CLOSE_SETTLE_STEPS)   # hold for settle
        total_steps += CLOSE_SETTLE_STEPS

        # ── 4. LIFTING ───────────────────────────────────────────────────────
        self._transition(GraspState.LIFTING)
        states_visited.append(self._state)

        lift_xyz = np.asarray(pose.grasp_xyz) + np.array([0.0, 0.0, LIFT_HEIGHT_M])
        ik_lift  = self.kin.ik(lift_xyz, q_init=self.ctrl.get_joints())

        if ik_lift.converged:
            self.ctrl.set_joint_targets(ik_lift.q_arm)
            _, steps = self.ctrl.run_until_converged(self.max_steps)
            total_steps += steps
        # If lift IK fails we still consider the grasp successful (object secured)

        # ── 5. DONE ──────────────────────────────────────────────────────────
        self._transition(GraspState.DONE)
        states_visited.append(self._state)

        return ExecutionResult(
            success        = True,
            final_state    = GraspState.DONE,
            states_visited = states_visited,
            total_steps    = total_steps,
            elapsed_s      = time.perf_counter() - t0,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _transition(self, new_state: GraspState) -> None:
        self._state = new_state

    def _fail(
        self,
        reason: str,
        states_visited: List[GraspState],
        total_steps: int,
        t0: float,
    ) -> ExecutionResult:
        self._transition(GraspState.FAILED)
        states_visited.append(GraspState.FAILED)
        return ExecutionResult(
            success        = False,
            final_state    = GraspState.FAILED,
            states_visited = states_visited,
            total_steps    = total_steps,
            elapsed_s      = time.perf_counter() - t0,
            fail_reason    = reason,
        )

    def reset(self) -> None:
        """Reset executor to IDLE (does not reset controller joints)."""
        self._state = GraspState.IDLE
