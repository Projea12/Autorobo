"""
tests/test_trajectory_block4.py — Acceptance tests for Block 4.5.

Acceptance criterion
--------------------
    Trajectory is collision-free for a standard table-top grasp.

The standard table-top scenario:
    Object at [0.0, 0.5, 0.5] m (0.5 m forward, 0.5 m high)
    Approach: TOP_DOWN  →  approach_vec = [0, 0, -1] (downward in base frame)
    Pre-grasp : [0.0, 0.5, 0.65]  (15 cm above object)
    Grasp     : [0.0, 0.5, 0.52]  ( 2 cm above object surface)
    Start     : home keyframe joints
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.kinematics import TidyBotKinematics, HOME_QPOS, ARM_QPOS_SLICE
from robot.trajectory  import (
    TrajectoryPlanner, JointTrajectory, Waypoint,
    _cosine_ease, _interpolate_segment, N_WAYPOINTS,
)

# Standard table-top grasp scenario
_OBJ_XYZ       = np.array([0.0,  0.5,  0.50])
_PRE_GRASP_XYZ = np.array([0.0,  0.5,  0.65])   # 15 cm above
_GRASP_XYZ     = np.array([0.0,  0.5,  0.52])   #  2 cm above surface
_Q_HOME        = HOME_QPOS[ARM_QPOS_SLICE].copy()


# ── cosine easing unit tests ──────────────────────────────────────────────────

def test_cosine_ease_endpoints() -> bool:
    """s(0) = 0  and  s(1) = 1."""
    s  = _cosine_ease(30)
    ok = abs(s[0]) < 1e-9 and abs(s[-1] - 1.0) < 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}]  cosine_ease endpoints: s[0]={s[0]:.6f}  s[-1]={s[-1]:.6f}")
    return ok


def test_cosine_ease_monotone() -> bool:
    """s must be strictly increasing."""
    s  = _cosine_ease(30)
    ok = bool(np.all(np.diff(s) > 0))
    print(f"  [{'PASS' if ok else 'FAIL'}]  cosine_ease is monotone  "
          f"min_step={float(np.min(np.diff(s))):.5f}")
    return ok


def test_cosine_ease_zero_velocity_at_ends() -> bool:
    """
    First derivative at endpoints must be near zero.
    ds/dt ≈ 0 at t=0 and t=1 (smooth start/stop).
    """
    s   = _cosine_ease(100)
    d0  = s[1] - s[0]       # forward difference at start
    d1  = s[-1] - s[-2]     # backward difference at end
    ok  = d0 < 0.01 and d1 < 0.01
    print(f"  [{'PASS' if ok else 'FAIL'}]  cosine_ease zero-velocity at ends  "
          f"ds_start={d0:.5f}  ds_end={d1:.5f}  (both < 0.01)")
    return ok


def test_interpolate_segment_shape() -> bool:
    """_interpolate_segment returns (N, 7) array."""
    q0  = np.zeros(7)
    q1  = np.ones(7)
    out = _interpolate_segment(q0, q1, 30)
    ok  = out.shape == (30, 7)
    print(f"  [{'PASS' if ok else 'FAIL'}]  _interpolate_segment shape {out.shape}  (expect (30,7))")
    return ok


def test_interpolate_segment_endpoints() -> bool:
    """First row = q_start, last row = q_end."""
    q0  = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    q1  = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    out = _interpolate_segment(q0, q1, 30)
    ok  = np.allclose(out[0], q0) and np.allclose(out[-1], q1)
    print(f"  [{'PASS' if ok else 'FAIL'}]  interpolation endpoints match q_start and q_end")
    return ok


# ── trajectory planner ────────────────────────────────────────────────────────

def test_plan_returns_joint_trajectory() -> bool:
    """plan() must return a JointTrajectory."""
    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin)
    traj    = planner.plan(_Q_HOME, _PRE_GRASP_XYZ, _GRASP_XYZ)
    ok      = isinstance(traj, JointTrajectory)
    print(f"  [{'PASS' if ok else 'FAIL'}]  plan() returns JointTrajectory")
    return ok


def test_trajectory_waypoint_count() -> bool:
    """Total waypoints = 2 × N_WAYPOINTS = 60."""
    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin, n_per_segment=N_WAYPOINTS)
    traj    = planner.plan(_Q_HOME, _PRE_GRASP_XYZ, _GRASP_XYZ)
    expected = 2 * N_WAYPOINTS
    ok       = len(traj.waypoints) == expected
    print(f"  [{'PASS' if ok else 'FAIL'}]  waypoint count = {len(traj.waypoints)}  "
          f"(expect {expected})")
    return ok


def test_trajectory_time_span() -> bool:
    """t values span [0, 1], segment 0 ends where segment 1 begins."""
    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin)
    traj    = planner.plan(_Q_HOME, _PRE_GRASP_XYZ, _GRASP_XYZ)
    t_vals  = [w.t for w in traj.waypoints]
    ok      = abs(t_vals[0]) < 1e-9 and abs(t_vals[-1] - 1.0) < 1e-6
    print(f"  [{'PASS' if ok else 'FAIL'}]  t ∈ [0,1]: t[0]={t_vals[0]:.4f}  t[-1]={t_vals[-1]:.4f}")
    return ok


def test_trajectory_segment_labels() -> bool:
    """First N waypoints are segment 0, last N are segment 1."""
    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin)
    traj    = planner.plan(_Q_HOME, _PRE_GRASP_XYZ, _GRASP_XYZ)
    N   = N_WAYPOINTS
    ok  = (all(w.segment == 0 for w in traj.waypoints[:N]) and
           all(w.segment == 1 for w in traj.waypoints[N:]))
    print(f"  [{'PASS' if ok else 'FAIL'}]  segment labels: first {N} → seg=0, last {N} → seg=1")
    return ok


def test_ik_both_segments_converge() -> bool:
    """IK must converge for both pre-grasp and grasp targets."""
    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin)
    traj    = planner.plan(_Q_HOME, _PRE_GRASP_XYZ, _GRASP_XYZ)
    ok      = traj.ik_pre_grasp.converged and traj.ik_grasp.converged
    print(f"  [{'PASS' if ok else 'FAIL'}]  IK converged: "
          f"pre_grasp={traj.ik_pre_grasp.converged}  "
          f"grasp={traj.ik_grasp.converged}")
    print(f"         pre-grasp err={traj.ik_pre_grasp.final_error*1000:.2f} mm  "
          f"grasp err={traj.ik_grasp.final_error*1000:.2f} mm")
    return ok


def test_q_path_shape() -> bool:
    """q_path() must return (60, 7) array."""
    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin)
    traj    = planner.plan(_Q_HOME, _PRE_GRASP_XYZ, _GRASP_XYZ)
    q_path  = traj.q_path()
    ok      = q_path.shape == (2 * N_WAYPOINTS, 7)
    print(f"  [{'PASS' if ok else 'FAIL'}]  q_path() shape={q_path.shape}  (expect ({2*N_WAYPOINTS},7))")
    return ok


def test_ee_path_starts_at_home() -> bool:
    """First waypoint EE must be close to home FK position."""
    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin)
    traj    = planner.plan(_Q_HOME, _PRE_GRASP_XYZ, _GRASP_XYZ)
    ee_home = kin.fk_home()
    ee_first = traj.waypoints[0].ee_xyz
    dist    = float(np.linalg.norm(ee_first - ee_home))
    ok      = dist < 0.01   # within 1 cm
    print(f"  [{'PASS' if ok else 'FAIL'}]  first EE = {np.round(ee_first,3)}  "
          f"home EE = {np.round(ee_home,3)}  Δ={dist:.4f} m")
    return ok


def test_str_renders() -> bool:
    """JointTrajectory.__str__() must render without error."""
    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin)
    traj    = planner.plan(_Q_HOME, _PRE_GRASP_XYZ, _GRASP_XYZ)
    s       = str(traj)
    ok      = "segment" in s.lower() and "waypoint" in s.lower()
    print(f"  [{'PASS' if ok else 'FAIL'}]  __str__ renders:\n{s}")
    return ok


# ── PRIMARY ACCEPTANCE — collision-free ───────────────────────────────────────

def test_collision_free_table_top_grasp() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 4.5)

    Standard table-top grasp trajectory from home → pre-grasp → grasp
    must have zero collisions across all 60 waypoints.
    """
    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin, n_per_segment=N_WAYPOINTS)
    traj    = planner.plan(_Q_HOME, _PRE_GRASP_XYZ, _GRASP_XYZ)

    n_coll = traj.collision_count()
    ok     = traj.is_collision_free()

    print(f"  [{'PASS' if ok else 'FAIL'}]  COLLISION-FREE table-top grasp  "
          f"({n_coll}/{len(traj.waypoints)} waypoints in collision)")
    if not ok:
        for w in traj.waypoints:
            if w.in_collision:
                print(f"    collision at t={w.t:.3f} seg={w.segment}  "
                      f"ee={np.round(w.ee_xyz,3)}")
    return ok


def test_planning_time() -> bool:
    """Full trajectory planning (IK × 2 + 60 FK + collision checks) < 50 ms."""
    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin)
    t0      = time.perf_counter()
    planner.plan(_Q_HOME, _PRE_GRASP_XYZ, _GRASP_XYZ)
    dt      = (time.perf_counter() - t0) * 1000
    ok      = dt < 50.0
    print(f"  [{'PASS' if ok else 'FAIL'}]  planning time = {dt:.1f} ms  (< 50 ms)")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Block 4.5 — IK trajectory waypoints ────────────────────────────")

    results = [
        # easing
        test_cosine_ease_endpoints(),
        test_cosine_ease_monotone(),
        test_cosine_ease_zero_velocity_at_ends(),
        test_interpolate_segment_shape(),
        test_interpolate_segment_endpoints(),
        # trajectory structure
        test_plan_returns_joint_trajectory(),
        test_trajectory_waypoint_count(),
        test_trajectory_time_span(),
        test_trajectory_segment_labels(),
        test_ik_both_segments_converge(),
        test_q_path_shape(),
        test_ee_path_starts_at_home(),
        test_str_renders(),
        # primary acceptance
        test_collision_free_table_top_grasp(),
        test_planning_time(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
