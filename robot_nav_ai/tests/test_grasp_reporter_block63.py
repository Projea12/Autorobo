"""
tests/test_grasp_reporter_block63.py — Acceptance tests for Blocks 6.3 & 6.4.

Block 6.3 Acceptance:
    GraspReporter classifies and prints the correct failure reason for
    each simulated failure mode, and draws the right overlay colour.

Block 6.4 Acceptance:
    GraspResult dataclass is defined, correctly populated, and printed.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ar.grasp_reporter import (
    GraspReporter, GraspResult, FailureReason,
)
from ar.grasp_executor import (
    ExecutionResult, GraspState, LiftResult,
)
from robot.robot_controller import GripperCloseResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_exec_result(
    success: bool = True,
    fail_reason: str = "",
    gripper_object_detected: bool = True,
    lift_success: bool = True,
    lift_ik_ok: bool = True,
    elapsed_s: float = 0.5,
) -> ExecutionResult:
    """Build a minimal ExecutionResult for testing."""
    gripper_result = GripperCloseResult(
        final_ctrl       = 200.0 if gripper_object_detected else 200.0,
        steps_taken      = 50,
        object_detected  = gripper_object_detected,
        final_driver_pos = 0.05 if gripper_object_detected else 0.23,
    )

    z_pre  = np.array([0.0, 0.5, 0.5])
    z_post = np.array([0.0, 0.5, 0.5 + (0.1 if lift_success else 0.0)])
    lift_result = LiftResult(
        ee_xyz_pre   = z_pre,
        ee_xyz_post  = z_post,
        delta_z      = float(z_post[2] - z_pre[2]),
        success      = lift_success,
        ik_converged = lift_ik_ok,
    )

    states = [GraspState.IDLE, GraspState.MOVING_TO_PREGRASP,
              GraspState.MOVING_TO_GRASP, GraspState.CLOSING, GraspState.LIFTING]
    states.append(GraspState.DONE if success else GraspState.FAILED)

    return ExecutionResult(
        success        = success,
        final_state    = states[-1],
        states_visited = states,
        total_steps    = 200,
        elapsed_s      = elapsed_s,
        gripper_result = gripper_result,
        lift_result    = lift_result,
        fail_reason    = fail_reason,
    )


def _capture(fn) -> str:
    """Capture stdout from fn() and return it."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


# ── Block 6.3 — reporter ──────────────────────────────────────────────────────

def test_success_prints_secured() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 6.3 — success)

    On success, console must contain '[grasp] SUCCESS' and the label.
    """
    reporter   = GraspReporter()
    exec_res   = _make_exec_result(success=True)
    output     = _capture(lambda: reporter.report(exec_res, label="mug"))
    ok         = "[grasp] SUCCESS" in output and "mug" in output
    print(f"  [{'PASS' if ok else 'FAIL'}]  success prints '[grasp] SUCCESS — mug secured'")
    return ok


def test_failure_no_contact_reason() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 6.3 — NO_CONTACT)

    When gripper closed but object_detected=False, failure reason must
    be NO_CONTACT and console line must mention it.
    """
    reporter = GraspReporter()
    exec_res = _make_exec_result(
        success=True,          # executor reached DONE
        gripper_object_detected=False,
    )
    # Override success to False so reporter treats it as failure
    exec_res = ExecutionResult(
        success        = False,
        final_state    = GraspState.DONE,
        states_visited = exec_res.states_visited,
        total_steps    = exec_res.total_steps,
        elapsed_s      = exec_res.elapsed_s,
        gripper_result = exec_res.gripper_result,
        lift_result    = exec_res.lift_result,
        fail_reason    = "",
    )
    result = reporter.report(exec_res, label="mug")
    ok     = (result.failure_reason == FailureReason.NO_CONTACT and
              not result.success)
    output = _capture(lambda: reporter.report(exec_res, label="mug"))
    ok2    = "no contact" in output.lower()
    print(f"  [{'PASS' if ok and ok2 else 'FAIL'}]  NO_CONTACT classified and printed  "
          f"reason={result.failure_reason}")
    return ok and ok2


def test_failure_object_moved() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 6.3 — OBJECT_MOVED)

    Contact detected (object_detected=True) but lift failed → OBJECT_MOVED.
    """
    reporter = GraspReporter()
    exec_res = _make_exec_result(
        success       = False,
        fail_reason   = "",
        gripper_object_detected = True,
        lift_success  = False,
    )
    result = reporter.report(exec_res, label="mug")
    ok     = result.failure_reason == FailureReason.OBJECT_MOVED and not result.success
    print(f"  [{'PASS' if ok else 'FAIL'}]  OBJECT_MOVED classified  reason={result.failure_reason}")
    return ok


def test_failure_ik_failed_reason() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 6.3 — IK_FAILED)

    fail_reason string containing 'IK failed' → FailureReason.IK_FAILED.
    """
    reporter = GraspReporter()
    exec_res = _make_exec_result(
        success     = False,
        fail_reason = "IK failed for pre-grasp target",
    )
    result = reporter.report(exec_res, label="mug")
    ok     = result.failure_reason == FailureReason.IK_FAILED
    print(f"  [{'PASS' if ok else 'FAIL'}]  IK_FAILED classified  reason={result.failure_reason}")
    return ok


def test_failure_out_of_reach_reason() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 6.3 — OUT_OF_REACH)

    fail_reason containing 'out of reach' → FailureReason.OUT_OF_REACH.
    """
    reporter = GraspReporter()
    exec_res = _make_exec_result(
        success     = False,
        fail_reason = "Target is out of reach (dist=1.50 > max=0.89)",
    )
    result = reporter.report(exec_res, label="mug")
    ok     = result.failure_reason == FailureReason.OUT_OF_REACH
    print(f"  [{'PASS' if ok else 'FAIL'}]  OUT_OF_REACH classified  reason={result.failure_reason}")
    return ok


def test_failure_console_line_format() -> bool:
    """Console failure line must match '[grasp] FAILED — reason: ...'."""
    reporter = GraspReporter()
    exec_res = _make_exec_result(
        success     = False,
        fail_reason = "IK failed for grasp target",
    )
    output = _capture(lambda: reporter.report(exec_res, label="mug"))
    ok     = "[grasp] FAILED — reason:" in output
    print(f"  [{'PASS' if ok else 'FAIL'}]  failure console line format correct")
    return ok


def test_success_returns_none_failure_reason() -> bool:
    """On success, GraspResult.failure_reason must be None."""
    reporter = GraspReporter()
    exec_res = _make_exec_result(success=True)
    result   = reporter.report(exec_res, label="cup")
    ok       = result.failure_reason is None and result.success
    print(f"  [{'PASS' if ok else 'FAIL'}]  success → failure_reason=None  got={result.failure_reason}")
    return ok


def test_overlay_green_on_success() -> bool:
    """
    On success, _draw_banner must write green-ish pixels on the frame.
    (Skipped if cv2 is not available.)
    """
    try:
        import cv2
    except ImportError:
        print("  [SKIP]  cv2 not available — skipping overlay colour check")
        return True

    frame    = np.zeros((480, 640, 3), dtype=np.uint8)
    reporter = GraspReporter()
    exec_res = _make_exec_result(success=True)
    reporter.report(exec_res, label="mug", frame=frame)

    # Bottom banner row — green channel should dominate
    banner_row = frame[470, :, :]          # BGR: (B, G, R)
    mean_g = float(banner_row[:, 1].mean())
    mean_r = float(banner_row[:, 2].mean())
    mean_b = float(banner_row[:, 0].mean())
    ok     = mean_g > mean_r and mean_g > mean_b and mean_g > 10
    print(f"  [{'PASS' if ok else 'FAIL'}]  green banner on success  "
          f"B={mean_b:.0f} G={mean_g:.0f} R={mean_r:.0f}")
    return ok


def test_overlay_red_on_failure() -> bool:
    """On failure, _draw_banner must write red-ish pixels on the frame."""
    try:
        import cv2
    except ImportError:
        print("  [SKIP]  cv2 not available — skipping overlay colour check")
        return True

    frame    = np.zeros((480, 640, 3), dtype=np.uint8)
    reporter = GraspReporter()
    exec_res = _make_exec_result(
        success     = False,
        fail_reason = "IK failed for grasp target",
    )
    reporter.report(exec_res, label="mug", frame=frame)

    banner_row = frame[470, :, :]
    mean_r = float(banner_row[:, 2].mean())
    mean_g = float(banner_row[:, 1].mean())
    mean_b = float(banner_row[:, 0].mean())
    ok     = mean_r > mean_g and mean_r > mean_b and mean_r > 10
    print(f"  [{'PASS' if ok else 'FAIL'}]  red banner on failure  "
          f"B={mean_b:.0f} G={mean_g:.0f} R={mean_r:.0f}")
    return ok


# ── Block 6.4 — GraspResult dataclass ────────────────────────────────────────

def test_grasp_result_type() -> bool:
    """report() must return a GraspResult."""
    reporter = GraspReporter()
    exec_res = _make_exec_result(success=True)
    result   = reporter.report(exec_res, label="mug")
    ok       = isinstance(result, GraspResult)
    print(f"  [{'PASS' if ok else 'FAIL'}]  report() returns GraspResult")
    return ok


def test_grasp_result_fields_success() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 6.4 — success)

    GraspResult fields are populated correctly on success.
    """
    reporter = GraspReporter()
    exec_res = _make_exec_result(success=True, elapsed_s=0.75)
    result   = reporter.report(exec_res, label="bottle")
    ok = (
        result.success and
        result.object_label == "bottle" and
        result.failure_reason is None and
        abs(result.attempt_duration_ms - 750.0) < 1.0
    )
    print(f"  [{'PASS' if ok else 'FAIL'}]  GraspResult fields correct on success  "
          f"label={result.object_label}  duration={result.attempt_duration_ms:.0f} ms")
    return ok


def test_grasp_result_fields_failure() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 6.4 — failure)

    GraspResult fields are populated correctly on failure.
    """
    reporter = GraspReporter()
    exec_res = _make_exec_result(
        success     = False,
        fail_reason = "IK failed for pre-grasp target",
        elapsed_s   = 1.2,
    )
    result = reporter.report(exec_res, label="mug")
    ok = (
        not result.success and
        result.object_label == "mug" and
        result.failure_reason == FailureReason.IK_FAILED and
        abs(result.attempt_duration_ms - 1200.0) < 1.0
    )
    print(f"  [{'PASS' if ok else 'FAIL'}]  GraspResult fields correct on failure  "
          f"reason={result.failure_reason}  duration={result.attempt_duration_ms:.0f} ms")
    return ok


def test_grasp_result_str() -> bool:
    """GraspResult.__str__() renders without error for success and failure."""
    reporter = GraspReporter()
    r_ok  = reporter.report(_make_exec_result(success=True), label="mug")
    r_fail = reporter.report(
        _make_exec_result(success=False, fail_reason="IK failed for grasp target"),
        label="mug",
    )
    ok = (
        "GraspResult" in str(r_ok)  and "SUCCESS" in str(r_ok) and
        "GraspResult" in str(r_fail) and "FAILED"  in str(r_fail)
    )
    print(f"  [{'PASS' if ok else 'FAIL'}]  GraspResult str: ok='{str(r_ok)}'")
    return ok


def test_all_failure_reasons_covered() -> bool:
    """Every FailureReason member must be classifiable from a representative input."""
    reporter = GraspReporter()

    cases = {
        FailureReason.IK_FAILED:    _make_exec_result(success=False, fail_reason="IK failed for pre-grasp target"),
        FailureReason.OUT_OF_REACH: _make_exec_result(success=False, fail_reason="Target is out of reach"),
        FailureReason.NO_CONTACT:   ExecutionResult(
            success=False, final_state=GraspState.DONE,
            states_visited=[GraspState.IDLE, GraspState.DONE],
            total_steps=50, elapsed_s=0.5,
            gripper_result=GripperCloseResult(200.0, 50, False, 0.23),
            lift_result=None, fail_reason="",
        ),
        FailureReason.OBJECT_MOVED: _make_exec_result(
            success=False, fail_reason="",
            gripper_object_detected=True, lift_success=False,
        ),
    }

    ok = True
    for expected_reason, exec_res in cases.items():
        result = reporter.report(exec_res, label="mug")
        match  = result.failure_reason == expected_reason
        ok     = ok and match
        print(f"    {expected_reason.name}: {'✓' if match else '✗'}  "
              f"(got {result.failure_reason})")

    print(f"  [{'PASS' if ok else 'FAIL'}]  all 4 failure reasons correctly classified")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Blocks 6.3 & 6.4 — grasp reporter + Phase-8 handoff ──────────────")

    results = [
        # Block 6.3
        test_success_prints_secured(),
        test_failure_no_contact_reason(),
        test_failure_object_moved(),
        test_failure_ik_failed_reason(),
        test_failure_out_of_reach_reason(),
        test_failure_console_line_format(),
        test_success_returns_none_failure_reason(),
        test_overlay_green_on_success(),
        test_overlay_red_on_failure(),
        # Block 6.4
        test_grasp_result_type(),
        test_grasp_result_fields_success(),
        test_grasp_result_fields_failure(),
        test_grasp_result_str(),
        test_all_failure_reasons_covered(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
