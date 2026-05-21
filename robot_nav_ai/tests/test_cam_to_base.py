"""
tests/test_cam_to_base.py — Acceptance test for Block 2.3 camera→base transform.

Acceptance criterion (from spec)
---------------------------------
    Object at 1 m directly in front of the robot maps to
    Y_base ≈ 1.0 m  (Y = forward in TidyBot base frame).

    Note: the spec labels this "Z_base" but TidyBot uses Y=forward (robot
    drives along +Y at heading=0; see command_interface.py:149).
    The forward distance is checked with ±15 % tolerance.

Coordinate conventions (see ar/transforms.py for full derivation)
------------------------------------------------------------------
    Camera frame  : X=right,   Y=down,    Z=forward (OpenCV)
    Base frame    : X=right,   Y=forward, Z=up      (TidyBot / ROS-like)

Camera mounting on TidyBot (hardcoded, T_CAM_TO_BASE)
    Position : [right=0.0, forward=0.1, up=1.2] m
    Rotation : forward-facing, no tilt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ar.transforms import T_CAM_TO_BASE, RigidTransform
from ar.localiser  import Localiser

TOLERANCE = 0.15    # ±15 % on forward distance


# ── helpers ───────────────────────────────────────────────────────────────────

def _check(label: str, xyz_cam, expected_base, tol_abs=1e-9) -> bool:
    got = T_CAM_TO_BASE(xyz_cam)
    ok  = all(abs(g - e) <= tol_abs for g, e in zip(got, expected_base))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  {label}")
    print(f"         cam  = ({xyz_cam[0]:+.3f}, {xyz_cam[1]:+.3f}, {xyz_cam[2]:+.3f})")
    print(f"         base = ({got[0]:+.3f}, {got[1]:+.3f}, {got[2]:+.3f})")
    if not ok:
        print(f"         expected ({expected_base[0]:+.3f}, {expected_base[1]:+.3f}, {expected_base[2]:+.3f})")
    return ok


# ── rotation sanity ───────────────────────────────────────────────────────────

def test_rotation_matrix() -> bool:
    """R must be orthonormal with det=+1."""
    R   = T_CAM_TO_BASE.R
    RRt = R @ R.T
    det = np.linalg.det(R)
    ok  = (np.allclose(RRt, np.eye(3), atol=1e-12)
           and abs(det - 1.0) < 1e-12)
    print(f"  [{'PASS' if ok else 'FAIL'}]  rotation matrix orthonormal, det=+1  "
          f"(got det={det:.6f})")
    return ok


# ── acceptance test ───────────────────────────────────────────────────────────

def test_acceptance_1m_forward() -> bool:
    """
    PRIMARY ACCEPTANCE — spec 2.3

    Object on optical axis at Z_cam=1.0 m → Y_base ≈ 1.0 m (forward).
    Camera is at y=0.1 m forward, so Y_base = 1.0 + 0.1 = 1.1 m.
    That is within 15 % of 1.0 m  → PASS.
    """
    xyz_cam = (0.0, 0.0, 1.0)
    xyz_base = T_CAM_TO_BASE(xyz_cam)
    y_fwd = xyz_base[1]           # Y = forward in base frame
    pct   = abs(y_fwd - 1.0) / 1.0
    ok    = pct <= TOLERANCE

    print(f"  [{'PASS' if ok else 'FAIL'}]  1m forward acceptance")
    print(f"         cam  (0.0, 0.0, 1.0) m")
    print(f"         base ({xyz_base[0]:+.3f}, {xyz_base[1]:+.3f}, {xyz_base[2]:+.3f}) m")
    print(f"         Y_base (forward) = {y_fwd:.3f} m   error = {pct*100:.1f} %  "
          f"(threshold ±{TOLERANCE*100:.0f} %)")
    return ok


# ── geometry tests ────────────────────────────────────────────────────────────

def test_camera_origin_maps_to_mount_position() -> bool:
    """Camera origin (0,0,0)_cam → camera mount position in base frame."""
    # The camera origin in camera frame transforms to the camera's
    # physical position in base frame: [right=0, forward=0.1, up=1.2].
    return _check(
        "camera origin → mount position [0, 0.1, 1.2]",
        xyz_cam   = (0.0, 0.0, 0.0),
        expected_base = (0.0, 0.1, 1.2),
    )


def test_right_of_camera_is_right_in_base() -> bool:
    """
    Object at X_cam=+1 (1m to the right of camera) →
    X_base = +1 (1m to the right in base frame).
    """
    # Camera X aligns directly with base X (both = right).
    # P_cam = (1, 0, 0) → P_base = R@(1,0,0) + t = (1, 0, 0) + (0, 0.1, 1.2)
    return _check(
        "1m right in cam → 1m right in base",
        xyz_cam       = (1.0, 0.0, 0.0),
        expected_base = (1.0, 0.1, 1.2),
    )


def test_down_in_camera_is_down_in_base() -> bool:
    """
    Object 1m below camera (Y_cam=+1, i.e., downward in OpenCV) →
    Z_base = -1 + 1.2 = 0.2 m above ground.
    """
    # Y_cam (down) → -Z_base (Z=up), so Z_base = -1 + 1.2 = 0.2
    return _check(
        "1m below camera → Z_base = 0.2 m (still above floor)",
        xyz_cam       = (0.0, 1.0, 0.0),
        expected_base = (0.0, 0.1, 0.2),
    )


def test_localiser_to_base_frame() -> bool:
    """Localiser.to_base_frame() delegates correctly to T_CAM_TO_BASE."""
    xyz_cam  = (0.0, 0.0, 1.0)
    via_transform  = T_CAM_TO_BASE(xyz_cam)
    via_localiser  = Localiser.to_base_frame(xyz_cam)
    ok = all(abs(a - b) < 1e-9 for a, b in zip(via_transform, via_localiser))
    print(f"  [{'PASS' if ok else 'FAIL'}]  Localiser.to_base_frame() matches T_CAM_TO_BASE")
    return ok


def test_apply_batch() -> bool:
    """RigidTransform.apply() handles (N,3) arrays correctly."""
    pts_cam = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    pts_base = T_CAM_TO_BASE.apply(pts_cam)
    # Each row should match the single-point transform
    for i, row in enumerate(pts_cam):
        expected = T_CAM_TO_BASE(tuple(row))
        got = tuple(pts_base[i])
        if not all(abs(g - e) < 1e-9 for g, e in zip(got, expected)):
            print(f"  [FAIL]  batch apply: row {i} mismatch {got} vs {expected}")
            return False
    print(f"  [PASS]  batch apply (3 points)")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Block 2.3 — camera → base frame transform ────────────────────")

    results = [
        test_rotation_matrix(),
        test_acceptance_1m_forward(),
        test_camera_origin_maps_to_mount_position(),
        test_right_of_camera_is_right_in_base(),
        test_down_in_camera_is_down_in_base(),
        test_localiser_to_base_frame(),
        test_apply_batch(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
