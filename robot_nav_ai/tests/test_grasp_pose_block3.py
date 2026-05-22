"""
tests/test_grasp_pose_block3.py — Acceptance tests for Block 3.3 grasp pose.

Acceptance criterion (from spec)
---------------------------------
    Top-down grasp generates:
      - pre-grasp 15 cm above the object
      - grasp at 2 cm from the object surface

    Camera frame: X=right, Y=down, Z=forward.
    "Above" = −Y direction.  Top-down approach = [0, +1, 0].

    pre_grasp_y = object_y − 0.15   (15 cm higher = lower Y in camera frame)
    grasp_y     = object_y − 0.02   (2 cm above surface)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ar.grasp_planner import GraspPlanner, _orientation_from_approach
from ar.grasp_pose    import ApproachType, GraspApproach

TOL = 1e-6   # metre tolerance — pure geometry, no model


# ── helpers ───────────────────────────────────────────────────────────────────

def _approach(vec, atype) -> GraspApproach:
    v = np.array(vec, dtype=float)
    v = v / np.linalg.norm(v)
    return GraspApproach(n_hat=-v, approach_vec=v,
                         approach_type=atype, confidence=1.0)

def _is_SO3(R):
    return (np.allclose(R.T @ R, np.eye(3), atol=1e-9)
            and abs(np.linalg.det(R) - 1.0) < 1e-9)


# ── PRIMARY ACCEPTANCE — top-down ─────────────────────────────────────────────

def test_top_down_pre_grasp_15cm_above() -> bool:
    """
    PRIMARY ACCEPTANCE

    Top-down approach [0,1,0]:
      pre_grasp must be 15 cm above object  (Δy = −0.15 m)
      grasp     must be  2 cm above object  (Δy = −0.02 m)
    """
    obj  = np.array([0.10, 0.80, 1.20])
    pose = GraspPlanner().plan(obj, _approach([0,1,0], ApproachType.TOP_DOWN))

    expected_pre = obj + np.array([0.0, -0.15, 0.0])
    expected_grs = obj + np.array([0.0, -0.02, 0.0])

    pre_ok = np.allclose(pose.pre_grasp_xyz, expected_pre, atol=TOL)
    grs_ok = np.allclose(pose.grasp_xyz,    expected_grs, atol=TOL)
    ok     = pre_ok and grs_ok

    pg, g = pose.pre_grasp_xyz, pose.grasp_xyz
    print(f"  [{'PASS' if ok else 'FAIL'}]  TOP-DOWN: pre-grasp 15 cm above, grasp 2 cm above")
    print(f"         object    : ({obj[0]:+.3f}, {obj[1]:+.3f}, {obj[2]:+.3f}) m")
    print(f"         pre_grasp : ({pg[0]:+.3f}, {pg[1]:+.3f}, {pg[2]:+.3f}) m  "
          f"Δy={pg[1]-obj[1]:+.3f}  (expect −0.150)")
    print(f"         grasp     : ({g[0]:+.3f},  {g[1]:+.3f},  {g[2]:+.3f}) m  "
          f"Δy={g[1]-obj[1]:+.3f}  (expect −0.020)")
    return ok


# ── horizontal / shelf grasp ──────────────────────────────────────────────────

def test_horizontal_pre_grasp_15cm_infront() -> bool:
    """
    Horizontal approach [0,0,1]:
      pre_grasp 15 cm in front of object (Δz = −0.15 m)
      grasp      2 cm in front           (Δz = −0.02 m)
    """
    obj  = np.array([0.05, 1.10, 0.80])
    pose = GraspPlanner().plan(obj, _approach([0,0,1], ApproachType.HORIZONTAL))

    expected_pre = obj + np.array([0.0, 0.0, -0.15])
    expected_grs = obj + np.array([0.0, 0.0, -0.02])

    pre_ok = np.allclose(pose.pre_grasp_xyz, expected_pre, atol=TOL)
    grs_ok = np.allclose(pose.grasp_xyz,    expected_grs, atol=TOL)
    ok     = pre_ok and grs_ok

    print(f"  [{'PASS' if ok else 'FAIL'}]  HORIZONTAL: pre-grasp 15 cm in front  "
          f"Δz={pose.pre_grasp_xyz[2]-obj[2]:+.3f} (expect −0.150)")
    return ok


# ── object position preserved ─────────────────────────────────────────────────

def test_object_xyz_preserved() -> bool:
    obj  = np.array([0.33, 0.77, 1.11])
    pose = GraspPlanner().plan(obj, _approach([0,1,0], ApproachType.TOP_DOWN))
    ok   = np.allclose(pose.object_xyz, obj, atol=TOL)
    print(f"  [{'PASS' if ok else 'FAIL'}]  object_xyz preserved in GraspPose")
    return ok


# ── custom offsets ────────────────────────────────────────────────────────────

def test_custom_offsets() -> bool:
    obj     = np.array([0.0, 0.5, 1.0])
    planner = GraspPlanner(pre_grasp_offset=0.20, grasp_offset=0.05)
    pose    = planner.plan(obj, _approach([0,1,0], ApproachType.TOP_DOWN))
    pre_ok  = abs(pose.pre_grasp_xyz[1] - (obj[1] - 0.20)) < TOL
    grs_ok  = abs(pose.grasp_xyz[1]     - (obj[1] - 0.05)) < TOL
    ok      = pre_ok and grs_ok
    print(f"  [{'PASS' if ok else 'FAIL'}]  custom offsets 20 cm / 5 cm respected")
    return ok


# ── orientation matrix ────────────────────────────────────────────────────────

def test_orientation_is_SO3() -> bool:
    """R must be valid SO(3) for any approach direction."""
    approaches = [
        np.array([0., 1., 0.]),
        np.array([0., 0., 1.]),
        np.array([1., 0., 0.]),
        np.array([0.577, 0.577, 0.577]),
    ]
    results = [_is_SO3(_orientation_from_approach(a / np.linalg.norm(a)))
               for a in approaches]
    ok = all(results)
    print(f"  [{'PASS' if ok else 'FAIL'}]  gripper_R is SO(3): "
          f"{sum(results)}/{len(results)} approaches")
    return ok


def test_orientation_z_axis_aligns() -> bool:
    """R[:,2] (gripper Z) must align with approach_vec to <0.01°."""
    cases = [
        (np.array([0., 1., 0.]), "top-down"),
        (np.array([0., 0., 1.]), "horizontal"),
        (np.array([0.408, 0.816, 0.408]), "diagonal"),
    ]
    results = []
    for a_raw, label in cases:
        a     = a_raw / np.linalg.norm(a_raw)
        R     = _orientation_from_approach(a)
        angle = np.degrees(np.arccos(np.clip(np.dot(R[:, 2], a), -1, 1)))
        ok    = angle < 0.01
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}]  Z-axis aligns [{label}]  "
              f"R[:,2]={np.round(R[:,2],3)}  angle={angle:.4f}°")
    return all(results)


# ── string rendering ──────────────────────────────────────────────────────────

def test_str_renders() -> bool:
    pose = GraspPlanner().plan(
        np.array([0.1, 0.5, 1.2]),
        _approach([0, 1, 0], ApproachType.TOP_DOWN),
    )
    s  = str(pose)
    ok = "pre_grasp" in s and "grasp" in s and "top_down" in s
    print(f"  [{'PASS' if ok else 'FAIL'}]  __str__ renders correctly")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Block 3.3 — grasp pose generation ───────────────────────────")

    results = [
        test_top_down_pre_grasp_15cm_above(),
        test_horizontal_pre_grasp_15cm_infront(),
        test_object_xyz_preserved(),
        test_custom_offsets(),
        test_orientation_is_SO3(),
        test_orientation_z_axis_aligns(),
        test_str_renders(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
