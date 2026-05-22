"""
tests/test_contact_success_block6.py — Acceptance tests for Blocks 6.1 & 6.2.

Block 6.1 Acceptance:
    ContactDetector correctly detects contact when a simulated box object
    is in the gripper path (driver joint frozen below free-close reference).

Block 6.2 Acceptance:
    GraspSuccessChecker correctly distinguishes:
    - SUCCESS: object 3D position moved upward > 5 cm
    - FAILURE: object 3D position unchanged (arm lifted without object)
    Both the 3D check and the pixel centroid check are tested.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.kinematics       import TidyBotKinematics
from robot.robot_controller import (
    RobotController, _RDRIVER_QPOS, RAMP_STEPS, RESISTANCE_MIN_STEP,
)
from ar.contact_detector    import (
    ContactDetector, ContactResult,
    CONTACT_THRESHOLD_RAD, FREE_CLOSE_DRIVER_REF,
    GRIPPER_QPOS_SLICE, RDRIVER_QPOS_IDX,
)
from ar.grasp_success       import (
    GraspSuccessChecker, GraspSuccessResult, MIN_LIFT_Z, MIN_PIXEL_RISE,
)


# ── helpers ───────────────────────────────────────────────────────────────────

@dataclass
class _MockDetection:
    """Minimal Detection stand-in for pixel centroid tests."""
    label:       str
    confidence:  float
    bbox_xyxy:   Tuple[int, int, int, int]
    centroid_uv: Tuple[int, int]


def _mock_det(label: str, u: int, v: int, conf: float = 0.9) -> _MockDetection:
    return _MockDetection(
        label=label, confidence=conf,
        bbox_xyxy=(u-30, v-30, u+30, v+30),
        centroid_uv=(u, v),
    )


# ── Block 6.1 — gripper contact detection ────────────────────────────────────

def test_contact_detector_free_close_no_contact() -> bool:
    """
    After a free close (no object in scene), ContactDetector must report
    contact_detected = False.
    """
    kin     = TidyBotKinematics()
    ctrl    = RobotController(kin)
    ctrl.close_gripper_ramped()      # free close — joints reach reference
    detector = ContactDetector()
    result   = detector.detect(kin.data, commanded_ctrl=200.0)
    ok       = not result.contact_detected
    print(f"  [{'PASS' if ok else 'FAIL'}]  FREE CLOSE → no contact  "
          f"driver={result.driver_pos:.4f}  deficit={result.driver_deficit_rad:.4f}")
    return ok


def test_contact_detector_blocked_detects_contact() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 6.1)

    When driver joints are frozen at a lower angle (simulating a box in the
    gripper path), ContactDetector must report contact_detected = True.
    """
    kin  = TidyBotKinematics()
    ctrl = RobotController(kin)
    step_count = [0]

    # Freeze driver joints mid-way to simulate blocked gripper
    blocked_pos = 0.05   # rad — well below free_close_ref (0.2336)

    def frozen_step():
        import mujoco
        mujoco.mj_step(ctrl.kin.model, ctrl.kin.data)
        if step_count[0] >= RESISTANCE_MIN_STEP:
            ctrl.kin.data.qpos[10] = blocked_pos
            ctrl.kin.data.qpos[14] = blocked_pos
        step_count[0] += 1

    ctrl.close_gripper_ramped(_physics_step_fn=frozen_step)

    detector = ContactDetector()
    result   = detector.detect(kin.data, commanded_ctrl=200.0)
    ok       = result.contact_detected

    print(f"  [{'PASS' if ok else 'FAIL'}]  BLOCKED CLOSE → contact detected  "
          f"driver={result.driver_pos:.4f}  deficit={result.driver_deficit_rad:.4f}  "
          f"threshold={result.threshold_rad:.4f}")
    return ok


def test_contact_result_type() -> bool:
    """detect() must return a ContactResult dataclass."""
    kin     = TidyBotKinematics()
    ctrl    = RobotController(kin)
    ctrl.close_gripper_ramped()
    result  = ContactDetector().detect(kin.data)
    ok      = isinstance(result, ContactResult)
    print(f"  [{'PASS' if ok else 'FAIL'}]  detect() returns ContactResult")
    return ok


def test_contact_result_qpos_shape() -> bool:
    """qpos_gripper in result must be (8,) array."""
    kin     = TidyBotKinematics()
    ctrl    = RobotController(kin)
    ctrl.close_gripper_ramped()
    result  = ContactDetector().detect(kin.data)
    ok      = result.qpos_gripper.shape == (8,)
    print(f"  [{'PASS' if ok else 'FAIL'}]  qpos_gripper shape={result.qpos_gripper.shape}  (expect (8,))")
    return ok


def test_contact_result_str() -> bool:
    """ContactResult.__str__() must render without error."""
    kin     = TidyBotKinematics()
    ctrl    = RobotController(kin)
    ctrl.close_gripper_ramped()
    result  = ContactDetector().detect(kin.data)
    s       = str(result)
    ok      = "ContactResult" in s and ("CONTACT" in s or "FREE" in s)
    print(f"  [{'PASS' if ok else 'FAIL'}]  ContactResult str: {s}")
    return ok


def test_contact_bool_api() -> bool:
    """detect_contact() convenience method returns bool."""
    kin     = TidyBotKinematics()
    ctrl    = RobotController(kin)
    ctrl.close_gripper_ramped()
    result  = ContactDetector().detect_contact(kin.data)
    ok      = isinstance(result, bool)
    print(f"  [{'PASS' if ok else 'FAIL'}]  detect_contact() returns bool  (got {result})")
    return ok


def test_threshold_is_5_units() -> bool:
    """CONTACT_THRESHOLD_RAD must equal 5/255 * 0.8."""
    expected = (5.0 / 255.0) * 0.8
    ok       = abs(CONTACT_THRESHOLD_RAD - expected) < 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}]  threshold = {CONTACT_THRESHOLD_RAD:.6f} rad  "
          f"(= 5/255 * 0.8 = {expected:.6f})")
    return ok


def test_deficit_sign_convention() -> bool:
    """
    Deficit = reference - actual.
    With no object (driver near reference), deficit ≈ 0.
    With object (driver below reference), deficit > 0.
    """
    kin     = TidyBotKinematics()
    ctrl    = RobotController(kin)
    ctrl.close_gripper_ramped()
    result_free = ContactDetector().detect(kin.data)

    # Manually lower driver to simulate contact
    kin.data.qpos[10] = 0.05
    kin.data.qpos[14] = 0.05
    result_contact = ContactDetector().detect(kin.data)

    ok = (abs(result_free.driver_deficit_rad) < 0.05 and
          result_contact.driver_deficit_rad > CONTACT_THRESHOLD_RAD)
    print(f"  [{'PASS' if ok else 'FAIL'}]  deficit sign: "
          f"free={result_free.driver_deficit_rad:.4f}  "
          f"contact={result_contact.driver_deficit_rad:.4f}")
    return ok


# ── Block 6.2 — grasp success checker ────────────────────────────────────────

def test_3d_success_case() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 6.2 — success)

    Object at (0,0.5,0.5) before, (0,0.5,0.6) after → Δz=10cm > 5cm → success.
    """
    checker = GraspSuccessChecker()
    result  = checker.check_3d([0, 0.5, 0.5], [0, 0.5, 0.6])
    ok      = result.success and abs(result.delta_z - 0.1) < 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}]  3D SUCCESS: Δz={result.delta_z*100:.1f} cm  "
          f"success={result.success}")
    return ok


def test_3d_failure_case() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 6.2 — failure)

    Object at (0,0.5,0.5) before, same after → Δz=0 < 5cm → failure.
    """
    checker = GraspSuccessChecker()
    result  = checker.check_3d([0, 0.5, 0.5], [0, 0.5, 0.5])
    ok      = not result.success and abs(result.delta_z) < 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}]  3D FAILURE: Δz={result.delta_z*100:.1f} cm  "
          f"success={result.success}")
    return ok


def test_3d_boundary_cases() -> bool:
    """Exactly at threshold (5cm) → borderline; just over → success."""
    checker = GraspSuccessChecker()
    at_threshold  = checker.check_3d([0,0,0], [0,0, MIN_LIFT_Z - 1e-9])
    just_over     = checker.check_3d([0,0,0], [0,0, MIN_LIFT_Z + 1e-9])
    ok = not at_threshold.success and just_over.success
    print(f"  [{'PASS' if ok else 'FAIL'}]  boundary: "
          f"at_threshold={at_threshold.success}  just_over={just_over.success}")
    return ok


def test_pixel_success_case() -> bool:
    """
    Centroid moves from v=300 → v=250 (rose 50px > threshold) → success.
    """
    checker = GraspSuccessChecker()
    dets_before = [_mock_det("mug", 320, 300)]
    dets_after  = [_mock_det("mug", 318, 250)]
    result = checker.check_detections(dets_before, dets_after, "mug")
    ok     = result.success and result.pixel_rise == 50
    print(f"  [{'PASS' if ok else 'FAIL'}]  PIXEL SUCCESS: rise={result.pixel_rise} px  "
          f"success={result.success}")
    return ok


def test_pixel_failure_case() -> bool:
    """Centroid stays at v=300 → rise=0 < threshold → failure."""
    checker = GraspSuccessChecker()
    dets_before = [_mock_det("mug", 320, 300)]
    dets_after  = [_mock_det("mug", 320, 300)]
    result = checker.check_detections(dets_before, dets_after, "mug")
    ok     = not result.success and result.pixel_rise == 0
    print(f"  [{'PASS' if ok else 'FAIL'}]  PIXEL FAILURE: rise={result.pixel_rise} px  "
          f"success={result.success}")
    return ok


def test_pixel_missing_after_lift() -> bool:
    """Object not re-detected after lift → failure (conservative)."""
    checker = GraspSuccessChecker()
    result  = checker.check_detections([_mock_det("mug", 320, 300)], [], "mug")
    ok      = not result.success and not result.label_matched
    print(f"  [{'PASS' if ok else 'FAIL'}]  not re-detected → failure  "
          f"label_matched={result.label_matched}")
    return ok


def test_pixel_missing_before_lift() -> bool:
    """Object not in pre-grasp detections → failure."""
    checker = GraspSuccessChecker()
    result  = checker.check_detections([], [_mock_det("mug", 320, 250)], "mug")
    ok      = not result.success
    print(f"  [{'PASS' if ok else 'FAIL'}]  not in pre-grasp → failure")
    return ok


def test_combined_check_prefers_3d() -> bool:
    """check() uses 3D when xyz provided, ignores pixel."""
    checker = GraspSuccessChecker()
    result  = checker.check(
        xyz_before=[0,0,0], xyz_after=[0,0,0.1],
        dets_before=[_mock_det("mug", 320, 300)],
        dets_after=[_mock_det("mug", 320, 300)],
        label="mug",
    )
    ok = result.method == "3d" and result.success
    print(f"  [{'PASS' if ok else 'FAIL'}]  combined check prefers 3d  method={result.method}")
    return ok


def test_grasp_success_result_str() -> bool:
    """GraspSuccessResult.__str__() renders without error for both check types."""
    checker = GraspSuccessChecker()
    r3d  = checker.check_3d([0,0,0], [0,0,0.1])
    rpix = checker.check_detections(
        [_mock_det("mug", 320, 300)], [_mock_det("mug", 320, 240)], "mug"
    )
    ok3d  = "SUCCESS" in str(r3d) or "FAILURE" in str(r3d)
    okpix = "SUCCESS" in str(rpix) or "FAILURE" in str(rpix)
    ok    = ok3d and okpix
    print(f"  [{'PASS' if ok else 'FAIL'}]  GraspSuccessResult str renders for 3d and pixel")
    return ok


def test_full_pipeline_contact_then_success() -> bool:
    """
    End-to-end: free close → no contact; forced block → contact; lift 10cm → success.
    Exercises contact detection and 3D success check together.
    """
    checker  = GraspSuccessChecker()

    # 1. Free close → no contact
    kin1  = TidyBotKinematics()
    ctrl1 = RobotController(kin1)
    ctrl1.close_gripper_ramped()
    r_free = ContactDetector().detect(kin1.data)

    # 2. Blocked close → contact
    kin2  = TidyBotKinematics()
    ctrl2 = RobotController(kin2)
    step_n = [0]
    def _block():
        import mujoco
        mujoco.mj_step(ctrl2.kin.model, ctrl2.kin.data)
        if step_n[0] >= RESISTANCE_MIN_STEP:
            ctrl2.kin.data.qpos[10] = 0.05
            ctrl2.kin.data.qpos[14] = 0.05
        step_n[0] += 1
    ctrl2.close_gripper_ramped(_physics_step_fn=_block)
    r_contact = ContactDetector().detect(kin2.data)

    # 3. Lift → 3D success
    r_lift = checker.check_3d([0, 0.5, 0.5], [0, 0.5, 0.6])

    ok = not r_free.contact_detected and r_contact.contact_detected and r_lift.success
    print(f"  [{'PASS' if ok else 'FAIL'}]  pipeline: "
          f"free_no_contact={not r_free.contact_detected}  "
          f"blocked_contact={r_contact.contact_detected}  "
          f"lift_success={r_lift.success}")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Blocks 6.1 & 6.2 — contact detection + grasp success ───────────")

    results = [
        # Block 6.1
        test_contact_detector_free_close_no_contact(),
        test_contact_detector_blocked_detects_contact(),
        test_contact_result_type(),
        test_contact_result_qpos_shape(),
        test_contact_result_str(),
        test_contact_bool_api(),
        test_threshold_is_5_units(),
        test_deficit_sign_convention(),
        # Block 6.2
        test_3d_success_case(),
        test_3d_failure_case(),
        test_3d_boundary_cases(),
        test_pixel_success_case(),
        test_pixel_failure_case(),
        test_pixel_missing_after_lift(),
        test_pixel_missing_before_lift(),
        test_combined_check_prefers_3d(),
        test_grasp_success_result_str(),
        test_full_pipeline_contact_then_success(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
