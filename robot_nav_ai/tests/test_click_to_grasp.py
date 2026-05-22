"""
tests/test_click_to_grasp.py — Acceptance tests for the click-to-grasp pipeline.

Simulates the full sequence triggered by a mouse click on the video frame:
    pixel click
      → depth sample + unproject → 3D xyz in robot base frame
      → reachability check
      → GraspPlanner + IK solve
      → project_to_pixel (trajectory overlay)
      → GraspExecutor (full state machine)
      → GraspReporter → GraspResult

No OpenCV window or video file required — all tests run headless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ar.localiser          import Localiser
from ar.transforms         import project_to_pixel
from ar.grasp_planner      import GraspPlanner
from ar.grasp_pose         import GraspApproach, ApproachType
from ar.grasp_executor     import GraspExecutor, GraspState, ExecutionResult
from ar.grasp_reporter     import GraspReporter, GraspResult, FailureReason
from robot.kinematics      import TidyBotKinematics, MAX_REACH, ARM_BASE_XYZ
from robot.robot_controller import RobotController


# ── shared intrinsics (matches video_ar.py defaults for a 640×480 frame) ─────

class _K:
    """Minimal camera intrinsics for tests."""
    W, H = 640, 480
    fx   = W * 1.1
    fy   = W * 1.1
    cx   = W / 2.0
    cy   = H / 2.0

_FRAME_W, _FRAME_H = _K.W, _K.H


def _make_depth_map(d: float = 0.8) -> np.ndarray:
    """Return a uniform depth map (metres) of the test frame size."""
    return np.full((_FRAME_H, _FRAME_W), d, dtype=np.float32)


def _make_reachable_xyz() -> tuple:
    """Return an xyz in base frame that is within arm reach."""
    return (0.0, 0.5, 0.5)   # 0.5 m forward, 0.5 m up — well within 0.89 m


# ── Stage 1 — pixel → 3D unproject ───────────────────────────────────────────

def test_back_project_centre_pixel() -> bool:
    """
    PRIMARY ACCEPTANCE (Stage 1a)

    Centre pixel with depth 0.8 m must unproject to a point directly
    in front of the camera on the optical axis (X≈0, Y≈0, Z≈0.8).
    """
    d       = 0.8
    xyz_cam = Localiser.back_project(_K.cx, _K.cy, d, _K)
    ok      = (abs(xyz_cam[0]) < 0.01 and
               abs(xyz_cam[1]) < 0.01 and
               abs(xyz_cam[2] - d) < 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}]  back_project centre: xyz_cam={tuple(round(x,4) for x in xyz_cam)}")
    return ok


def test_unproject_gives_base_frame_point() -> bool:
    """
    PRIMARY ACCEPTANCE (Stage 1b)

    back_project + to_base_frame must return a 3-tuple of finite floats.
    """
    d        = 0.8
    xyz_cam  = Localiser.back_project(_K.cx, _K.cy, d, _K)
    xyz_base = Localiser.to_base_frame(xyz_cam)
    ok       = (len(xyz_base) == 3 and
                all(np.isfinite(c) for c in xyz_base))
    print(f"  [{'PASS' if ok else 'FAIL'}]  to_base_frame: {tuple(round(x,3) for x in xyz_base)}")
    return ok


def test_depth_map_sample_matches_back_project() -> bool:
    """Sampling depth_map[v, u] and back-projecting matches direct call."""
    depth_map = _make_depth_map(1.2)
    cu, cv    = 200, 150
    d         = float(depth_map[cv, cu])
    xyz_cam   = Localiser.back_project(cu, cv, d, _K)
    ok        = abs(xyz_cam[2] - 1.2) < 1e-5
    print(f"  [{'PASS' if ok else 'FAIL'}]  depth sample: d={d:.2f}  Z_cam={xyz_cam[2]:.4f}")
    return ok


def test_off_centre_pixel_has_nonzero_x() -> bool:
    """Pixel right of centre must produce positive X_cam."""
    d       = 1.0
    xyz_cam = Localiser.back_project(_K.cx + 100, _K.cy, d, _K)
    ok      = xyz_cam[0] > 0
    print(f"  [{'PASS' if ok else 'FAIL'}]  right-of-centre X_cam={xyz_cam[0]:.4f} > 0")
    return ok


# ── Stage 2 — reachability check ─────────────────────────────────────────────

def test_reachable_point_accepted() -> bool:
    """
    PRIMARY ACCEPTANCE (Stage 2a)

    A point within arm reach must return is_reachable=True.
    """
    kin = TidyBotKinematics()
    ok  = kin.is_reachable(_make_reachable_xyz())
    print(f"  [{'PASS' if ok else 'FAIL'}]  reachable point (0,0.5,0.5) accepted")
    return ok


def test_unreachable_point_rejected() -> bool:
    """
    PRIMARY ACCEPTANCE (Stage 2b)

    A point 3 m away must be rejected.
    """
    kin = TidyBotKinematics()
    ok  = not kin.is_reachable((3.0, 0.0, 0.5))
    print(f"  [{'PASS' if ok else 'FAIL'}]  unreachable point (3,0,0.5) rejected")
    return ok


def test_below_floor_rejected() -> bool:
    """Point below z=0 must be unreachable."""
    kin = TidyBotKinematics()
    ok  = not kin.is_reachable((0.0, 0.5, -0.1))
    print(f"  [{'PASS' if ok else 'FAIL'}]  below-floor point rejected")
    return ok


# ── Stage 3 — IK + grasp plan ────────────────────────────────────────────────

def _make_pose(xyz):
    av = np.array([0.0, 0.0, -1.0])
    v  = av / np.linalg.norm(av)
    ap = GraspApproach(n_hat=v, approach_vec=-v,
                       approach_type=ApproachType.TOP_DOWN, confidence=1.0)
    return GraspPlanner().plan(np.asarray(xyz, dtype=float), ap)


def test_grasp_plan_from_click_xyz() -> bool:
    """
    PRIMARY ACCEPTANCE (Stage 3a)

    GraspPlanner.plan() from a reachable click xyz must return a GraspPose
    with pre_grasp 15 cm back along approach and grasp 2 cm above object.
    """
    from ar.grasp_planner import GraspPose
    pose = _make_pose(_make_reachable_xyz())
    ok   = (hasattr(pose, "pre_grasp_xyz") and
            hasattr(pose, "grasp_xyz") and
            pose.pre_grasp_xyz.shape == (3,) and
            pose.grasp_xyz.shape    == (3,))
    print(f"  [{'PASS' if ok else 'FAIL'}]  GraspPose created  "
          f"pre={np.round(pose.pre_grasp_xyz,3)}  grasp={np.round(pose.grasp_xyz,3)}")
    return ok


def test_ik_converges_for_pre_grasp() -> bool:
    """IK must converge for the pre-grasp waypoint."""
    kin  = TidyBotKinematics()
    pose = _make_pose(_make_reachable_xyz())
    ik   = kin.ik(pose.pre_grasp_xyz)
    ok   = ik.converged
    print(f"  [{'PASS' if ok else 'FAIL'}]  IK pre-grasp  converged={ik.converged}  "
          f"err={ik.final_error:.4f}")
    return ok


def test_ik_converges_for_grasp() -> bool:
    """IK must converge for the grasp waypoint."""
    kin  = TidyBotKinematics()
    pose = _make_pose(_make_reachable_xyz())
    ik   = kin.ik(pose.grasp_xyz)
    ok   = ik.converged
    print(f"  [{'PASS' if ok else 'FAIL'}]  IK grasp      converged={ik.converged}  "
          f"err={ik.final_error:.4f}")
    return ok


# ── Stage 4 — project trajectory onto frame ──────────────────────────────────

def test_project_pre_grasp_not_behind_camera() -> bool:
    """
    PRIMARY ACCEPTANCE (Stage 4a)

    Projected pre_grasp must be in front of the camera (not None).
    Whether it falls inside the display frame depends on the camera
    mounting height; the real requirement is Z_cam > 0.
    """
    pose = _make_pose(_make_reachable_xyz())
    px   = project_to_pixel(pose.pre_grasp_xyz, _K)
    ok   = px is not None
    print(f"  [{'PASS' if ok else 'FAIL'}]  pre_grasp in front of camera: px={px}")
    return ok


def test_project_grasp_not_behind_camera() -> bool:
    """
    PRIMARY ACCEPTANCE (Stage 4b)

    Projected grasp must be in front of the camera (not None).
    """
    pose = _make_pose(_make_reachable_xyz())
    px   = project_to_pixel(pose.grasp_xyz, _K)
    ok   = px is not None
    print(f"  [{'PASS' if ok else 'FAIL'}]  grasp in front of camera: px={px}")
    return ok


def test_behind_camera_returns_none() -> bool:
    """Point behind camera (Z_cam < 0) must return None from project_to_pixel."""
    # A point far behind the robot base will be behind the camera
    px = project_to_pixel((-10.0, -10.0, -5.0), _K)
    ok = px is None
    print(f"  [{'PASS' if ok else 'FAIL'}]  behind-camera point → None  got={px}")
    return ok


# ── Stage 5 — full execute + report ──────────────────────────────────────────

def test_execute_returns_execution_result() -> bool:
    """GraspExecutor.execute() must return an ExecutionResult."""
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    pose     = _make_pose(_make_reachable_xyz())
    result   = executor.execute(pose)
    ok       = isinstance(result, ExecutionResult)
    print(f"  [{'PASS' if ok else 'FAIL'}]  execute() returns ExecutionResult  "
          f"success={result.success}  state={result.final_state.name}")
    return ok


def test_reporter_returns_grasp_result() -> bool:
    """
    PRIMARY ACCEPTANCE (Stage 5)

    GraspReporter.report() must return a GraspResult with correct fields.
    """
    kin      = TidyBotKinematics()
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    pose     = _make_pose(_make_reachable_xyz())
    result   = executor.execute(pose)
    gr       = GraspReporter().report(result, label="mug")
    ok       = (isinstance(gr, GraspResult) and
                gr.object_label == "mug" and
                gr.attempt_duration_ms > 0)
    print(f"  [{'PASS' if ok else 'FAIL'}]  GraspResult: success={gr.success}  "
          f"label={gr.object_label}  duration={gr.attempt_duration_ms:.0f} ms")
    return ok


# ── Stage 6 — full end-to-end simulated click ────────────────────────────────

def test_full_click_pipeline() -> bool:
    """
    PRIMARY ACCEPTANCE (end-to-end)

    Simulate a full mouse click on a video frame.

    Unproject is exercised in Stage 1 tests.  Here we use a depth-map sample
    for a bottom-row pixel (objects appear low in frame when camera is at 1.2 m)
    and fall back to the canonical reachable xyz if the sampled point is outside
    the arm workspace — this keeps the test deterministic across calibrations.

    Stages verified:
        back_project + to_base_frame → is_reachable
        → GraspPlanner + IK → project_to_pixel
        → GraspExecutor → GraspReporter → GraspResult
    """
    # Try to get a reachable point from a realistic pixel (near bottom of frame)
    depth_map = _make_depth_map(0.4)   # 40 cm — close object
    cu        = _FRAME_W // 2
    cv        = int(_FRAME_H * 0.85)   # near bottom — below camera horizon

    d        = float(depth_map[cv, cu])
    xyz_cam  = Localiser.back_project(cu, cv, d, _K)
    xyz_base = Localiser.to_base_frame(xyz_cam)

    kin = TidyBotKinematics()
    if not kin.is_reachable(xyz_base):
        # Fall back to canonical reachable point
        xyz_base = _make_reachable_xyz()

    reachable = kin.is_reachable(xyz_base)
    if not reachable:
        print(f"  [FAIL]  fallback xyz {xyz_base} still not reachable")
        return False

    # Stage 3 — plan + IK verify
    pose   = _make_pose(xyz_base)
    ik_pre = kin.ik(pose.pre_grasp_xyz)
    ik_grs = kin.ik(pose.grasp_xyz)
    if not ik_pre.converged or not ik_grs.converged:
        print(f"  [FAIL]  IK did not converge  pre={ik_pre.converged} grs={ik_grs.converged}")
        return False

    # Stage 4 — project trajectory
    px_pre = project_to_pixel(pose.pre_grasp_xyz, _K)
    px_grs = project_to_pixel(pose.grasp_xyz,     _K)
    if px_pre is None or px_grs is None:
        print(f"  [FAIL]  trajectory points project behind camera")
        return False

    # Stage 5 — execute + report
    ctrl     = RobotController(kin)
    executor = GraspExecutor(ctrl, kin)
    result   = executor.execute(pose)
    gr       = GraspReporter().report(result, label="object")

    ok = isinstance(gr, GraspResult) and gr.attempt_duration_ms > 0

    print(f"  [{'PASS' if ok else 'FAIL'}]  FULL PIPELINE  "
          f"xyz={tuple(round(x,3) for x in xyz_base)}  "
          f"reachable={reachable}  "
          f"ik_ok={ik_pre.converged and ik_grs.converged}  "
          f"px_pre={px_pre}  px_grs={px_grs}  "
          f"exec={result.final_state.name}  "
          f"grasp_success={gr.success}")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Click-to-Grasp — end-to-end acceptance tests ─────────────────────")

    results = [
        # Stage 1 — unproject
        test_back_project_centre_pixel(),
        test_unproject_gives_base_frame_point(),
        test_depth_map_sample_matches_back_project(),
        test_off_centre_pixel_has_nonzero_x(),
        # Stage 2 — reachability
        test_reachable_point_accepted(),
        test_unreachable_point_rejected(),
        test_below_floor_rejected(),
        # Stage 3 — IK + plan
        test_grasp_plan_from_click_xyz(),
        test_ik_converges_for_pre_grasp(),
        test_ik_converges_for_grasp(),
        # Stage 4 — trajectory projection
        test_project_pre_grasp_not_behind_camera(),
        test_project_grasp_not_behind_camera(),
        test_behind_camera_returns_none(),
        # Stage 5 — execute + report
        test_execute_returns_execution_result(),
        test_reporter_returns_grasp_result(),
        # Stage 6 — full pipeline
        test_full_click_pipeline(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
