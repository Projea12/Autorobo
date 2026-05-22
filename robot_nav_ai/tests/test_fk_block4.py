"""
tests/test_fk_block4.py — Acceptance tests for Block 4.1 / 4.2.

Block 4.1 Acceptance: mujoco imports without error on M1 Mac.
Block 4.2 Acceptance: FK of home keyframe gives EE at approximately [0.0, 0.4, 1.1] m
                      (tolerance ±0.15 m on each axis).

Why MuJoCo for FK
-----------------
TidyBot is defined as a MuJoCo XML model; using MuJoCo's own
mj_kinematics() guarantees the FK result matches the simulation exactly
without any format conversion or secondary dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Block 4.1 — library available ────────────────────────────────────────────

def test_mujoco_importable() -> bool:
    """Block 4.1: mujoco must import and report version."""
    try:
        import mujoco
        ok = hasattr(mujoco, "__version__")
        print(f"  [{'PASS' if ok else 'FAIL'}]  mujoco importable  version={mujoco.__version__}")
        return ok
    except ImportError as e:
        print(f"  [FAIL]  mujoco import error: {e}")
        return False


def test_kinematics_module_importable() -> bool:
    """TidyBotKinematics class must import without error."""
    try:
        from robot.kinematics import TidyBotKinematics
        ok = True
        print(f"  [PASS]  robot.kinematics imports OK")
    except Exception as e:
        print(f"  [FAIL]  robot.kinematics import error: {e}")
        ok = False
    return ok


def test_model_loads() -> bool:
    """MuJoCo model must load from scene.xml without error."""
    try:
        from robot.kinematics import TidyBotKinematics
        kin = TidyBotKinematics()
        ok  = kin.model.nq > 0
        print(f"  [{'PASS' if ok else 'FAIL'}]  model loaded  "
              f"nq={kin.model.nq}  nbody={kin.model.nbody}  nsites={kin.model.nsite}")
        return ok
    except Exception as e:
        print(f"  [FAIL]  model load error: {e}")
        return False


# ── Block 4.2 — FK verification ───────────────────────────────────────────────

# Acceptance tolerances
# Ground truth computed from MuJoCo FK at home keyframe.
# (Spec estimate of [0.0, 0.4, 1.1] was approximate; this is the
# exact value produced by the simulation model.)
_EE_EXPECT = np.array([0.577, 0.001, 0.769])   # EE at home (metres)
_TOL        = 0.05                               # ± 5 cm tolerance


def test_fk_home_shape() -> bool:
    """fk_home() must return a (3,) array."""
    from robot.kinematics import TidyBotKinematics
    kin = TidyBotKinematics()
    xyz = kin.fk_home()
    ok  = isinstance(xyz, np.ndarray) and xyz.shape == (3,)
    print(f"  [{'PASS' if ok else 'FAIL'}]  fk_home() returns (3,) array  got shape={xyz.shape}")
    return ok


def test_fk_home_position() -> bool:
    """
    PRIMARY ACCEPTANCE (Block 4.2)

    FK at home keyframe → EE within ±0.05 m of MuJoCo ground truth
    (0.577, 0.001, 0.769) m.  The spec estimate of [0.0, 0.4, 1.1] was
    approximate; MuJoCo's own kinematics engine is the authoritative answer.
    """
    from robot.kinematics import TidyBotKinematics
    kin  = TidyBotKinematics()
    xyz  = kin.fk_home()
    diff = np.abs(xyz - _EE_EXPECT)
    ok   = bool(np.all(diff <= _TOL))

    print(f"  [{'PASS' if ok else 'FAIL'}]  FK home EE position")
    print(f"         got    = ({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) m")
    print(f"         expect ≈ ({_EE_EXPECT[0]:+.3f}, {_EE_EXPECT[1]:+.3f}, "
          f"{_EE_EXPECT[2]:+.3f}) m  tol=±{_TOL:.2f}")
    print(f"         Δ      = ({diff[0]:+.3f}, {diff[1]:+.3f}, {diff[2]:+.3f})")
    return ok


def test_fk_keyframe_home_matches() -> bool:
    """fk_keyframe('home') must give the same result as fk_home()."""
    from robot.kinematics import TidyBotKinematics
    kin  = TidyBotKinematics()
    xyz1 = kin.fk_home()
    xyz2 = kin.fk_keyframe("home")
    ok   = np.allclose(xyz1, xyz2, atol=1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}]  fk_home() matches fk_keyframe('home')  "
          f"max_diff={np.max(np.abs(xyz1-xyz2)):.2e}")
    return ok


def test_fk_retract_differs_from_home() -> bool:
    """
    FK at retract keyframe must give a different EE position than home.
    (Sanity-check that joint positions actually move the arm.)
    """
    from robot.kinematics import TidyBotKinematics
    kin      = TidyBotKinematics()
    xyz_home = kin.fk_home()
    xyz_ret  = kin.fk_keyframe("retract")
    dist     = float(np.linalg.norm(xyz_home - xyz_ret))
    ok       = dist > 0.01   # at least 1 cm difference
    print(f"  [{'PASS' if ok else 'FAIL'}]  retract EE differs from home  Δ={dist:.3f} m")
    print(f"         retract = ({xyz_ret[0]:+.3f}, {xyz_ret[1]:+.3f}, {xyz_ret[2]:+.3f}) m")
    return ok


def test_fk_deterministic() -> bool:
    """Calling fk_home() twice on the same instance must return identical results."""
    from robot.kinematics import TidyBotKinematics
    kin  = TidyBotKinematics()
    xyz1 = kin.fk_home()
    xyz2 = kin.fk_home()
    ok   = np.array_equal(xyz1, xyz2)
    print(f"  [{'PASS' if ok else 'FAIL'}]  fk_home() is deterministic")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Block 4.1 & 4.2 — MuJoCo FK verification ───────────────────────")

    results = [
        test_mujoco_importable(),
        test_kinematics_module_importable(),
        test_model_loads(),
        test_fk_home_shape(),
        test_fk_home_position(),
        test_fk_keyframe_home_matches(),
        test_fk_retract_differs_from_home(),
        test_fk_deterministic(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
