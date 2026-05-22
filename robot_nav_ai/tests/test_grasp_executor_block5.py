"""
tests/test_grasp_executor_block5.py — Acceptance tests for Blocks 5.1 & 5.2.

Block 5.1 Acceptance: state machine transitions correctly through all states
    IDLE → MOVING_TO_PREGRASP → MOVING_TO_GRASP → CLOSING → LIFTING → DONE

Block 5.2 Acceptance: arm moves smoothly to any valid joint target
    within 2 seconds (= 200 steps at STEP_HZ=100)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.kinematics       import TidyBotKinematics, HOME_QPOS, ARM_QPOS_SLICE
from robot.robot_controller import (
    RobotController, MAX_RAD_PER_STEP, AT_TARGET_TOL, STEP_HZ,
    GRIPPER_OPEN, GRIPPER_CLOSE,
)
from ar.grasp_executor import (
    GraspExecutor, GraspState, ExecutionResult,
)
from ar.grasp_planner  import GraspPlanner
from ar.grasp_pose     import GraspApproach, ApproachType

# Standard table-top scenario
_OBJ_XYZ  = np.array([0.0,  0.5,  0.50])
_APP_VEC   = np.array([0.0,  0.0, -1.0])   # downward in base frame


def _make_approach() -> GraspApproach:
    v = _APP_VEC / np.linalg.norm(_APP_VEC)
    return GraspApproach(n_hat=-v, approach_vec=v,
                         approach_type=ApproachType.TOP_DOWN, confidence=1.0)


def _make_pose():
    return GraspPlanner().plan(_OBJ_XYZ, _make_approach())


# ── Block 5.2 — RobotController / waypoint follower ─────────────────────────

def test_controller_initialises_at_home() -> bool:
    """RobotController starts at home joints."""
    ctrl     = RobotController()
    q_home   = HOME_QPOS[ARM_QPOS_SLICE]
    ok       = np.allclose(ctrl.get_joints(), q_home, atol=1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}]  controller initialises at home joints")
    return ok


def test_set_joints_instant() -> bool:
    """set_joints() must update current joints immediately."""
    ctrl  = RobotController()
    q_new = np.zeros(7)
    ctrl.set_joints(q_new)
    ok    = np.allclose(ctrl.get_joints(), q_new, atol=1e-9)
    print(f"  [{'PASS' if ok else 'FAIL'}]  set_joints() is instant")
    return ok


def test_step_bounded_delta() -> bool:
    """Each step must change each joint by at most MAX_RAD_PER_STEP."""
    ctrl    = RobotController()
    ctrl.set_joints(np.zeros(7))
    target  = np.full(7, 3.0)            # large target
    ctrl.set_joint_targets(target)
    q_before = ctrl.get_joints().copy()
    ctrl.step()
    q_after  = ctrl.get_joints()
    max_delta = float(np.max(np.abs(q_after - q_before)))
    ok = max_delta <= MAX_RAD_PER_STEP + 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}]  step bounded: max_delta={max_delta:.4f} rad  "
          f"(limit={MAX_RAD_PER_STEP})")
    return ok


def test_step_direction() -> bool:
    """Steps must move joints toward the target, never away."""
    ctrl   = RobotController()
    ctrl.set_joints(np.zeros(7))
    target = np.array([1.0, -1.0, 0.5, -0.5, 0.8, -0.8, 0.3])
    ctrl.set_joint_targets(target)
    q0 = ctrl.get_joints().copy()
    ctrl.step()
    q1 = ctrl.get_joints()
    err_before = np.abs(target - q0)
    err_after  = np.abs(target - q1)
    ok = bool(np.all(err_after <= err_before + 1e-9))
    print(f"  [{'PASS' if ok else 'FAIL'}]  steps move toward target "
          f"(max err reduction {float(np.max(err_before - err_after)):.4f} rad)")
    return ok


def test_converges_within_200_steps() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 5.2)

    Arm must reach any valid target within 200 steps (2 s at 100 Hz).
    Test with four different targets.
    """
    kin  = TidyBotKinematics()
    targets = [
        np.zeros(7),
        HOME_QPOS[ARM_QPOS_SLICE],
        np.array([-0.4,  0.37,  2.71, -2.45,  0.19,  0.75,  1.57]),
        np.array([ 0.2, -0.30,  2.00, -1.80, -0.10,  0.50,  0.80]),
    ]
    results = []
    for target in targets:
        ctrl = RobotController(TidyBotKinematics())
        ctrl.set_joints(np.zeros(7))
        ctrl.set_joint_targets(target)
        converged, steps = ctrl.run_until_converged(max_steps=200)
        results.append((converged, steps))
        print(f"    target norm={np.linalg.norm(target):.2f}  "
              f"converged={converged}  steps={steps}")
    ok = all(r[0] for r in results)
    print(f"  [{'PASS' if ok else 'FAIL'}]  ALL targets reached within 200 steps  "
          f"({sum(r[0] for r in results)}/{len(results)})")
    return ok


def test_gripper_open_close() -> bool:
    """open_gripper() / close_gripper() must change ctrl without error."""
    ctrl = RobotController()
    ctrl.close_gripper()
    g_closed = ctrl._gripper
    ctrl.open_gripper()
    g_open = ctrl._gripper
    ok = (g_closed == GRIPPER_CLOSE) and (g_open == GRIPPER_OPEN)
    print(f"  [{'PASS' if ok else 'FAIL'}]  gripper open={g_open}  closed={g_closed}")
    return ok


def test_ee_xyz_updates_after_step() -> bool:
    """EE position must change after joints move."""
    ctrl   = RobotController()
    ee0    = ctrl.get_ee_xyz().copy()
    ctrl.set_joint_targets(np.zeros(7))
    ctrl.run_until_converged()
    ee1    = ctrl.get_ee_xyz()
    moved  = float(np.linalg.norm(ee1 - ee0))
    ok     = moved > 0.001   # at least 1 mm of EE motion
    print(f"  [{'PASS' if ok else 'FAIL'}]  EE moved {moved:.4f} m after convergence")
    return ok


# ── Block 5.1 — GraspExecutor state machine ──────────────────────────────────

_EXPECTED_PATH = [
    GraspState.IDLE,
    GraspState.MOVING_TO_PREGRASP,
    GraspState.MOVING_TO_GRASP,
    GraspState.CLOSING,
    GraspState.LIFTING,
    GraspState.DONE,
]


def test_state_machine_full_path() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 5.1)

    execute() must visit all states in order:
    IDLE → MOVING_TO_PREGRASP → MOVING_TO_GRASP → CLOSING → LIFTING → DONE
    """
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    pose     = _make_pose()

    result = executor.execute(pose)

    ok = (result.success and
          result.states_visited == _EXPECTED_PATH and
          result.final_state == GraspState.DONE)

    path_str = " → ".join(s.name for s in result.states_visited)
    print(f"  [{'PASS' if ok else 'FAIL'}]  STATE PATH: {path_str}")
    print(f"         success={result.success}  steps={result.total_steps}  "
          f"elapsed={result.elapsed_s*1000:.1f} ms")
    return ok


def test_result_is_execution_result() -> bool:
    """execute() must return an ExecutionResult."""
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    result   = executor.execute(_make_pose())
    ok       = isinstance(result, ExecutionResult)
    print(f"  [{'PASS' if ok else 'FAIL'}]  execute() returns ExecutionResult")
    return ok


def test_final_state_is_done() -> bool:
    """executor.state must be DONE after a successful execute()."""
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    executor.execute(_make_pose())
    ok       = executor.state == GraspState.DONE
    print(f"  [{'PASS' if ok else 'FAIL'}]  executor.state == DONE  (got {executor.state})")
    return ok


def test_all_states_in_enum() -> bool:
    """GraspState enum must contain all 7 required states."""
    required = {"IDLE", "MOVING_TO_PREGRASP", "MOVING_TO_GRASP",
                "CLOSING", "LIFTING", "DONE", "FAILED"}
    present  = {s.name for s in GraspState}
    ok       = required.issubset(present)
    print(f"  [{'PASS' if ok else 'FAIL'}]  GraspState has all required states  "
          f"missing={required - present}")
    return ok


def test_reset_returns_to_idle() -> bool:
    """reset() must put executor back to IDLE."""
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    executor.execute(_make_pose())
    executor.reset()
    ok       = executor.state == GraspState.IDLE
    print(f"  [{'PASS' if ok else 'FAIL'}]  reset() → IDLE  (got {executor.state})")
    return ok


def test_result_str_renders() -> bool:
    """ExecutionResult.__str__() must render without error."""
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    result   = executor.execute(_make_pose())
    s        = str(result)
    ok       = "SUCCESS" in s or "FAILED" in s
    print(f"  [{'PASS' if ok else 'FAIL'}]  ExecutionResult str renders:\n{s}")
    return ok


def test_gripper_closed_after_grasp() -> bool:
    """Gripper must be closed (ctrl ≥ 100) after CLOSING state."""
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    executor.execute(_make_pose())
    ok       = ctrl._gripper >= 100
    print(f"  [{'PASS' if ok else 'FAIL'}]  gripper closed after execute  "
          f"ctrl={ctrl._gripper}")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Block 5.1 & 5.2 — GraspExecutor + RobotController ──────────────")

    results = [
        # Block 5.2 — controller
        test_controller_initialises_at_home(),
        test_set_joints_instant(),
        test_step_bounded_delta(),
        test_step_direction(),
        test_converges_within_200_steps(),
        test_gripper_open_close(),
        test_ee_xyz_updates_after_step(),
        # Block 5.1 — state machine
        test_state_machine_full_path(),
        test_result_is_execution_result(),
        test_final_state_is_done(),
        test_all_states_in_enum(),
        test_reset_returns_to_idle(),
        test_result_str_renders(),
        test_gripper_closed_after_grasp(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
