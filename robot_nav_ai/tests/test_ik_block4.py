"""
tests/test_ik_block4.py — Acceptance tests for Block 4.3 / 4.4.

Block 4.3 Acceptance: IK for target (0.0, 0.5, 0.8) converges in < 5 ms.
Block 4.4 Acceptance: target (3.0, 0.0, 0.5) correctly flagged as unreachable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot.kinematics import (
    TidyBotKinematics, IKResult, ReachabilityError,
    ARM_BASE_XYZ, MAX_REACH,
)


# ── Block 4.3 — IK solver ────────────────────────────────────────────────────

def test_ik_converges_near_target() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 4.3)

    IK for (0.0, 0.5, 0.8) must converge (pos error < 1 mm) in < 5 ms.
    """
    kin    = TidyBotKinematics()
    target = np.array([0.0, 0.5, 0.8])

    t0  = time.perf_counter()
    res = kin.ik(target)
    dt  = (time.perf_counter() - t0) * 1000.0   # ms

    ok = res.converged and dt < 5.0
    print(f"  [{'PASS' if ok else 'FAIL'}]  IK PRIMARY — target (0.0, 0.5, 0.8)")
    print(f"         converged={res.converged}  iters={res.iterations}  "
          f"err={res.final_error*1000:.2f} mm  time={dt:.2f} ms  (< 5 ms)")
    print(f"         EE achieved = {np.round(res.ee_xyz, 4)}")
    return ok


def test_ik_result_type() -> bool:
    """ik() must return an IKResult dataclass."""
    kin = TidyBotKinematics()
    res = kin.ik([0.0, 0.5, 0.8])
    ok  = isinstance(res, IKResult) and res.q_arm.shape == (7,)
    print(f"  [{'PASS' if ok else 'FAIL'}]  ik() returns IKResult with q_arm shape (7,)")
    return ok


def test_ik_joint_limits_respected() -> bool:
    """All returned joint angles must be within model limits."""
    kin    = TidyBotKinematics()
    # Choose a tricky target near the edge of reach
    target = np.array([0.3, 0.4, 0.6])
    res    = kin.ik(target)

    limits = kin._arm_limits
    in_limits = np.all(res.q_arm >= limits[:, 0]) and np.all(res.q_arm <= limits[:, 1])
    ok = in_limits
    print(f"  [{'PASS' if ok else 'FAIL'}]  joint limits respected  "
          f"q={np.round(res.q_arm, 3)}")
    if not ok:
        for i, (q, lo, hi) in enumerate(zip(res.q_arm, limits[:, 0], limits[:, 1])):
            if not (lo <= q <= hi):
                print(f"    joint_{i+1}: {q:.4f} outside [{lo:.4f}, {hi:.4f}]")
    return ok


def test_ik_multiple_targets_converge() -> bool:
    """IK must converge for a range of reachable targets."""
    kin = TidyBotKinematics()
    targets = [
        np.array([0.0,  0.5, 0.8]),
        np.array([0.2,  0.4, 0.7]),
        np.array([-0.2, 0.4, 0.9]),
        np.array([0.0,  0.3, 0.6]),
    ]
    results = []
    for t in targets:
        res = kin.ik(t)
        results.append(res.converged)
        print(f"    target={np.round(t,2)}  converged={res.converged}  "
              f"err={res.final_error*1000:.2f} mm  iters={res.iterations}")
    ok = sum(results) >= 3   # at least 3 of 4 must converge
    print(f"  [{'PASS' if ok else 'FAIL'}]  {sum(results)}/{len(results)} targets converged")
    return ok


def test_ik_str_renders() -> bool:
    """IKResult.__str__() must render without error."""
    kin = TidyBotKinematics()
    res = kin.ik([0.0, 0.5, 0.8])
    s   = str(res)
    ok  = "CONVERGED" in s or "FAILED" in s
    print(f"  [{'PASS' if ok else 'FAIL'}]  IKResult str renders  (first 60 chars: {s[:60]!r})")
    return ok


def test_ik_timing_five_calls() -> bool:
    """Average of 5 IK solves from home init must each be < 5 ms."""
    kin    = TidyBotKinematics()
    target = np.array([0.0, 0.5, 0.8])
    times  = []
    for _ in range(5):
        t0 = time.perf_counter()
        kin.ik(target)
        times.append((time.perf_counter() - t0) * 1000)
    avg = float(np.mean(times))
    ok  = avg < 5.0
    print(f"  [{'PASS' if ok else 'FAIL'}]  IK timing: avg={avg:.2f} ms over 5 calls  "
          f"min={min(times):.2f} max={max(times):.2f}")
    return ok


# ── Block 4.4 — reachability ──────────────────────────────────────────────────

def test_far_target_is_unreachable() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 4.4)

    Target (3.0, 0.0, 0.5) is 2.78 m from the arm base — must raise
    ReachabilityError.
    """
    kin    = TidyBotKinematics()
    target = np.array([3.0, 0.0, 0.5])
    dist   = float(np.linalg.norm(target - ARM_BASE_XYZ))
    try:
        kin.check_reachable(target)
        ok = False
        print(f"  [FAIL]  (3.0, 0.0, 0.5) was NOT flagged (dist={dist:.3f} m)")
    except ReachabilityError as e:
        ok = True
        print(f"  [PASS]  (3.0, 0.0, 0.5) correctly flagged as unreachable")
        print(f"         dist={dist:.3f} m  reason: {e}")
    return ok


def test_below_floor_is_unreachable() -> bool:
    """Target below z=0 must raise ReachabilityError."""
    kin = TidyBotKinematics()
    try:
        kin.check_reachable([0.0, 0.3, -0.1])
        ok = False
        print(f"  [FAIL]  below-floor target was NOT flagged")
    except ReachabilityError as e:
        ok = True
        print(f"  [PASS]  below-floor target flagged: {e}")
    return ok


def test_nearby_target_is_reachable() -> bool:
    """Target (0.0, 0.5, 0.8) is well within reach — must not raise."""
    kin  = TidyBotKinematics()
    dist = float(np.linalg.norm(np.array([0.0, 0.5, 0.8]) - ARM_BASE_XYZ))
    try:
        kin.check_reachable([0.0, 0.5, 0.8])
        ok = True
        print(f"  [PASS]  (0.0, 0.5, 0.8) correctly accepted  dist={dist:.3f} m")
    except ReachabilityError as e:
        ok = False
        print(f"  [FAIL]  (0.0, 0.5, 0.8) wrongly rejected: {e}")
    return ok


def test_is_reachable_bool_api() -> bool:
    """is_reachable() must return True/False without raising."""
    kin = TidyBotKinematics()
    near = kin.is_reachable([0.0, 0.5, 0.8])
    far  = kin.is_reachable([3.0, 0.0, 0.5])
    ok   = near and not far
    print(f"  [{'PASS' if ok else 'FAIL'}]  is_reachable(): near={near}  far={far}")
    return ok


def test_boundary_near_max_reach() -> bool:
    """Target exactly at MAX_REACH distance must be accepted; just beyond must fail."""
    kin = TidyBotKinematics()
    # Point MAX_REACH metres directly forward from arm base
    at_limit  = ARM_BASE_XYZ + np.array([0.0, MAX_REACH - 0.001, 0.0])
    just_over = ARM_BASE_XYZ + np.array([0.0, MAX_REACH + 0.001, 0.0])
    in_ok  = kin.is_reachable(at_limit)
    out_ok = not kin.is_reachable(just_over)
    ok     = in_ok and out_ok
    print(f"  [{'PASS' if ok else 'FAIL'}]  boundary: within={in_ok}  beyond={not out_ok}")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Block 4.3 & 4.4 — IK solver + reachability ─────────────────────")

    results = [
        # Block 4.3
        test_ik_converges_near_target(),
        test_ik_result_type(),
        test_ik_joint_limits_respected(),
        test_ik_multiple_targets_converge(),
        test_ik_str_renders(),
        test_ik_timing_five_calls(),
        # Block 4.4
        test_far_target_is_unreachable(),
        test_below_floor_is_unreachable(),
        test_nearby_target_is_reachable(),
        test_is_reachable_bool_api(),
        test_boundary_near_max_reach(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
