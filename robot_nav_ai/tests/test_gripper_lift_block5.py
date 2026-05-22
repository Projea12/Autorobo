"""
tests/test_gripper_lift_block5.py — Acceptance tests for Blocks 5.3, 5.4, 5.5.

Block 5.3 Acceptance:
    Gripper closes to ctrl=200 when no object.
    Stops earlier (object_detected=True) when object resists.

Block 5.4 Acceptance:
    Lift motion executes and EE rises ≥ 5 cm (delta_z > LIFT_SUCCESS_MIN_Z).

Block 5.5 Acceptance:
    parse() recognises PICK command; parse_pick_target() extracts label;
    PICK appears in PLAYBACK with speed=0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.kinematics       import TidyBotKinematics
from robot.robot_controller import (
    RobotController, GripperCloseResult,
    RAMP_STEPS, RESISTANCE_TOL, RESISTANCE_MIN_STEP,
    GRIPPER_CLOSE, GRIPPER_OPEN,
    _RDRIVER_QPOS,
)
from ar.grasp_executor      import (
    GraspExecutor, GraspState, ExecutionResult, LiftResult,
    LIFT_SUCCESS_MIN_Z,
)
from ar.grasp_planner       import GraspPlanner
from ar.grasp_pose          import GraspApproach, ApproachType
from ar.command_interface   import Cmd, parse, parse_pick_target
from ar.video_ar            import PLAYBACK

_OBJ  = np.array([0.0, 0.5, 0.50])
_APPV = np.array([0.0, 0.0, -1.0])


def _make_pose():
    v = _APPV / np.linalg.norm(_APPV)
    app = GraspApproach(n_hat=v, approach_vec=-v,
                        approach_type=ApproachType.TOP_DOWN, confidence=1.0)
    return GraspPlanner().plan(_OBJ, app)


# ── Block 5.3 — gripper close sequence ───────────────────────────────────────

def test_free_close_reaches_200() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 5.3a)

    Without an object in the scene the ramp runs to completion and
    final_ctrl == GRIPPER_CLOSE (200).
    """
    ctrl   = RobotController()
    result = ctrl.close_gripper_ramped()
    ok     = (not result.object_detected and
              abs(result.final_ctrl - GRIPPER_CLOSE) < 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}]  FREE CLOSE → ctrl={result.final_ctrl:.1f}  "
          f"object_detected={result.object_detected}  steps={result.steps_taken}")
    return ok


def test_free_close_steps_equal_ramp() -> bool:
    """Free close must take exactly RAMP_STEPS steps."""
    ctrl   = RobotController()
    result = ctrl.close_gripper_ramped()
    ok     = result.steps_taken == RAMP_STEPS
    print(f"  [{'PASS' if ok else 'FAIL'}]  free close steps={result.steps_taken}  "
          f"(expect {RAMP_STEPS})")
    return ok


def test_resistance_stops_early() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 5.3b)

    When the driver joint is frozen (object resisting), the ramp must
    stop BEFORE reaching ctrl=200 and set object_detected=True.

    Simulation: inject a physics_step_fn that freezes driver qpos
    after the minimum check window — mimicking an object blocking.
    """
    ctrl = RobotController()
    step_count = [0]

    def frozen_step():
        import mujoco
        mujoco.mj_step(ctrl.kin.model, ctrl.kin.data)
        # After the minimum check window, freeze both driver joints
        if step_count[0] >= RESISTANCE_MIN_STEP + 1:
            ctrl.kin.data.qpos[_RDRIVER_QPOS] = 0.25   # stuck mid-way
            ctrl.kin.data.qpos[14]             = 0.25
        step_count[0] += 1

    result = ctrl.close_gripper_ramped(_physics_step_fn=frozen_step)
    ok     = result.object_detected and result.final_ctrl < GRIPPER_CLOSE
    print(f"  [{'PASS' if ok else 'FAIL'}]  OBJECT DETECTED — ctrl stopped at "
          f"{result.final_ctrl:.1f}/200  steps={result.steps_taken}")
    print(f"         driver_pos={result.final_driver_pos:.3f} rad  "
          f"object_detected={result.object_detected}")
    return ok


def test_gripper_close_result_type() -> bool:
    """close_gripper_ramped() must return GripperCloseResult."""
    ctrl   = RobotController()
    result = ctrl.close_gripper_ramped()
    ok     = isinstance(result, GripperCloseResult)
    print(f"  [{'PASS' if ok else 'FAIL'}]  returns GripperCloseResult")
    return ok


def test_gripper_result_str() -> bool:
    """GripperCloseResult.__str__() must render without error."""
    ctrl   = RobotController()
    result = ctrl.close_gripper_ramped()
    s      = str(result)
    ok     = "CLOSE" in s and "ctrl=" in s
    print(f"  [{'PASS' if ok else 'FAIL'}]  GripperCloseResult str: {s}")
    return ok


def test_ramp_is_monotone() -> bool:
    """ctrl must increase monotonically: verify via steps_taken vs final_ctrl."""
    ctrl        = RobotController()
    ctrl_values = []

    def recording_step():
        import mujoco
        ctrl_values.append(ctrl.kin.data.ctrl[10])
        mujoco.mj_step(ctrl.kin.model, ctrl.kin.data)

    ctrl.close_gripper_ramped(_physics_step_fn=recording_step)
    ok = all(ctrl_values[i] <= ctrl_values[i + 1]
             for i in range(len(ctrl_values) - 1))
    print(f"  [{'PASS' if ok else 'FAIL'}]  ctrl ramp is monotone  "
          f"({len(ctrl_values)} steps, min={min(ctrl_values):.1f} max={max(ctrl_values):.1f})")
    return ok


# ── Block 5.4 — lift test ─────────────────────────────────────────────────────

def test_lift_result_present() -> bool:
    """execute() result must include a LiftResult."""
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    result   = executor.execute(_make_pose())
    ok       = result.lift_result is not None
    print(f"  [{'PASS' if ok else 'FAIL'}]  execute() includes LiftResult  "
          f"(got {type(result.lift_result).__name__})")
    return ok


def test_lift_delta_z_positive() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 5.4)

    After lift IK, EE must rise by > LIFT_SUCCESS_MIN_Z (5 cm).
    The LiftResult.success flag must be True.
    """
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    result   = executor.execute(_make_pose())
    lr       = result.lift_result

    ok = lr is not None and lr.success and lr.delta_z > LIFT_SUCCESS_MIN_Z
    dz = lr.delta_z if lr else 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}]  LIFT Δz = {dz*100:.1f} cm  "
          f"(need > {LIFT_SUCCESS_MIN_Z*100:.0f} cm)  success={lr.success if lr else None}")
    return ok


def test_lift_ee_higher_after_lift() -> bool:
    """ee_xyz_post[2] must exceed ee_xyz_pre[2]."""
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    result   = executor.execute(_make_pose())
    lr       = result.lift_result

    ok = lr is not None and lr.ee_xyz_post[2] > lr.ee_xyz_pre[2]
    print(f"  [{'PASS' if ok else 'FAIL'}]  EE lifted: "
          f"pre_z={lr.ee_xyz_pre[2]:.3f}  post_z={lr.ee_xyz_post[2]:.3f}" if lr else
          f"  [FAIL]  LiftResult is None")
    return ok


def test_lift_result_str() -> bool:
    """LiftResult.__str__() renders without error."""
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    result   = executor.execute(_make_pose())
    lr       = result.lift_result
    ok       = lr is not None and ("LIFTED" in str(lr) or "NO LIFT" in str(lr))
    print(f"  [{'PASS' if ok else 'FAIL'}]  LiftResult str: {lr}")
    return ok


def test_gripper_result_in_execution_result() -> bool:
    """ExecutionResult must carry both gripper_result and lift_result."""
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    result   = executor.execute(_make_pose())
    ok       = (isinstance(result.gripper_result, GripperCloseResult) and
                isinstance(result.lift_result, LiftResult))
    print(f"  [{'PASS' if ok else 'FAIL'}]  ExecutionResult carries gripper+lift results")
    return ok


# ── Block 5.5 — command interface integration ─────────────────────────────────

def test_pick_in_cmd_enum() -> bool:
    """Cmd.PICK must exist in the enum."""
    ok = hasattr(Cmd, "PICK")
    print(f"  [{'PASS' if ok else 'FAIL'}]  Cmd.PICK exists in enum")
    return ok


def test_parse_pick_command() -> bool:
    """'pick up the mug' must parse to Cmd.PICK."""
    phrases = [
        "pick up the mug",
        "get me a cup",
        "fetch the bottle",
        "bring me the book",
        "grab the red mug",
    ]
    results = [(p, parse(p)) for p in phrases]
    ok      = all(cmd == Cmd.PICK for _, cmd in results)
    for phrase, cmd in results:
        print(f"    '{phrase}' → {cmd}")
    print(f"  [{'PASS' if ok else 'FAIL'}]  all PICK phrases parsed correctly")
    return ok


def test_parse_pick_target() -> bool:
    """parse_pick_target() must extract object name from pick phrase."""
    cases = [
        ("pick up the mug",     "mug"),
        ("get me a cup",        "cup"),
        ("fetch the bottle",    "bottle"),
        ("grab the red mug",    "red mug"),
    ]
    ok = True
    for text, expected in cases:
        got = parse_pick_target(text)
        match = got == expected
        ok = ok and match
        print(f"    '{text}' → '{got}' (expect '{expected}') {'✓' if match else '✗'}")
    print(f"  [{'PASS' if ok else 'FAIL'}]  parse_pick_target extracts object names")
    return ok


def test_pick_pauses_video() -> bool:
    """PLAYBACK['PICK'] must be 0.0 (video pauses during grasp)."""
    ok = PLAYBACK.get("PICK", -1) == 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}]  PLAYBACK['PICK'] = {PLAYBACK.get('PICK')}  (expect 0.0)")
    return ok


def test_pick_not_confused_with_gripper_close() -> bool:
    """'pick up the mug' must be PICK, not GRIPPER_CLOSE."""
    cmd = parse("pick up the mug")
    ok  = cmd == Cmd.PICK
    print(f"  [{'PASS' if ok else 'FAIL'}]  'pick up the mug' → {cmd}  (not GRIPPER_CLOSE)")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Blocks 5.3 / 5.4 / 5.5 — gripper ramp, lift, PICK command ───────")

    results = [
        # Block 5.3
        test_free_close_reaches_200(),
        test_free_close_steps_equal_ramp(),
        test_resistance_stops_early(),
        test_gripper_close_result_type(),
        test_gripper_result_str(),
        test_ramp_is_monotone(),
        # Block 5.4
        test_lift_result_present(),
        test_lift_delta_z_positive(),
        test_lift_ee_higher_after_lift(),
        test_lift_result_str(),
        test_gripper_result_in_execution_result(),
        # Block 5.5
        test_pick_in_cmd_enum(),
        test_parse_pick_command(),
        test_parse_pick_target(),
        test_pick_pauses_video(),
        test_pick_not_confused_with_gripper_close(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
