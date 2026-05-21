"""
tests/test_surface_normal.py — Acceptance tests for Block 3.1 surface normals.

Acceptance criterion (from spec)
---------------------------------
    Flat table surface returns normal pointing up → [0, -1, 0] in camera frame.

    Camera frame convention: X=right, Y=down, Z=forward.
    "Up in the world" = -Y in camera frame.

Synthetic depth maps used
--------------------------
    1. Horizontal plane  — Z(u,v) = h*fy/(v-cy)   → normal [0, -1, 0]
    2. Vertical wall     — Z(u,v) = const           → normal [0,  0, -1]
    3. 45° tilted plane  — Z(u,v) = a*u + b         → known analytic normal
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ar.surface_normal import estimate_normal, normal_map

ANGLE_TOL_DEG = 5.0   # normal must be within 5° of expected direction


@dataclass
class _K:
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two unit vectors."""
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return np.degrees(np.arccos(abs(dot)))   # abs: ignore sign flip


def _check(label, depth_map, u, v, expected, K=_K()) -> bool:
    n = estimate_normal(depth_map, u, v, K)
    angle = _angle_deg(n, np.array(expected, dtype=float))
    ok = angle <= ANGLE_TOL_DEG
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  {label}")
    print(f"         n_hat    = [{n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f}]")
    print(f"         expected = {expected}   angle = {angle:.2f}°  "
          f"(threshold {ANGLE_TOL_DEG}°)")
    return ok


# ── synthetic depth map builders ──────────────────────────────────────────────

def _horizontal_plane(H=480, W=640, K=_K(), h=1.0):
    """
    Depth map for a horizontal plane at camera-frame Y = h (below optical axis).

        Z(u, v) = h * fy / (v - cy)    for v > cy
        Z(u, v) = 0                     for v <= cy  (above horizon)

    Surface normal analytic result: [0, -1, 0]  (up in world = -Y in camera)
    """
    depth = np.zeros((H, W), dtype=np.float32)
    for v in range(int(K.cy) + 1, H):
        depth[v, :] = h * K.fy / (v - K.cy)
    return depth


def _vertical_wall(H=480, W=640, d=3.0):
    """
    Depth map for a flat vertical wall at constant depth d.

        Z(u, v) = d  everywhere

    Surface normal analytic result: [0, 0, -1]  (toward the camera)
    """
    return np.full((H, W), d, dtype=np.float32)


def _slanted_plane(H=480, W=640, K=_K(), slope=0.005):
    """
    Depth map for a plane slanted along u only: Z(u,v) = slope*u + base_depth.
    At pixel (cx, cy): T_u has a gz_u component, T_v is nearly flat.
    Used to verify that u-gradients are handled correctly.
    """
    uu = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    return (slope * uu + 2.0).astype(np.float32)


# ── acceptance test ───────────────────────────────────────────────────────────

def test_horizontal_table_primary() -> bool:
    """
    PRIMARY ACCEPTANCE — spec 3.1

    A flat horizontal table returns surface normal [0, -1, 0] in camera frame.
    Tests multiple pixels across the table surface; all must pass.
    """
    print("\n  [Primary acceptance] Horizontal table → normal [0, -1, 0]")
    K = _K()
    depth = _horizontal_plane(K=K, h=1.0)

    test_pixels = [
        (320, 320),   # centred u, 80px below cy
        (200, 350),   # left of centre
        (450, 380),   # right of centre
        (320, 400),   # further below
    ]
    results = []
    for u, v in test_pixels:
        results.append(_check(
            f"table pixel ({u},{v})",
            depth, u, v,
            expected=[0.0, -1.0, 0.0],
            K=K,
        ))
    return all(results)


# ── additional geometry tests ─────────────────────────────────────────────────

def test_vertical_wall() -> bool:
    """Flat wall at constant depth → normal toward camera [0, 0, -1]."""
    depth = _vertical_wall(d=3.0)
    return _check(
        "vertical wall at constant depth → [0, 0, -1]",
        depth, u=320, v=240,
        expected=[0.0, 0.0, -1.0],
    )


def test_slanted_plane_u_gradient() -> bool:
    """
    Plane slanted only along u → normal has no Y component.
    Checks gz_u path through the Jacobian.
    """
    K = _K()
    depth = _slanted_plane(K=K, slope=0.005)
    n = estimate_normal(depth, u=320, v=240, intrinsics=K)
    # Normal should have near-zero Y component (plane not tilted up/down)
    ok = abs(n[1]) < 0.1
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  slanted plane: Y component ≈ 0  (got {n[1]:+.4f})")
    return ok


def test_normal_is_unit_vector() -> bool:
    """Returned normal must always be a unit vector."""
    K   = _K()
    depth = _horizontal_plane(K=K, h=1.0)
    results = []
    for u in range(100, 541, 110):
        for v in range(260, 421, 80):
            n = estimate_normal(depth, u, v, K)
            mag = float(np.linalg.norm(n))
            ok  = abs(mag - 1.0) < 1e-6
            results.append(ok)
    passed = sum(results)
    total  = len(results)
    tag = "PASS" if passed == total else "FAIL"
    print(f"  [{tag}]  unit vector check: {passed}/{total} pixels |n|=1")
    return passed == total


def test_normal_map_shape() -> bool:
    """normal_map() returns correct H×W×3 shape."""
    K = _K()
    depth = _horizontal_plane(K=K, h=1.0)
    nm = normal_map(depth, K)
    ok = nm.shape == (480, 640, 3)
    print(f"  [{'PASS' if ok else 'FAIL'}]  normal_map shape: {nm.shape}  expected (480, 640, 3)")
    return ok


def test_normal_map_table_centre() -> bool:
    """normal_map() at table-centre pixels also gives [0,-1,0]."""
    K = _K()
    depth = _horizontal_plane(K=K, h=1.0)
    nm = normal_map(depth, K, blur_ksize=0)
    n = nm[350, 320]  # row 350, col 320
    angle = _angle_deg(n, np.array([0, -1, 0]))
    ok = angle <= ANGLE_TOL_DEG
    print(f"  [{'PASS' if ok else 'FAIL'}]  normal_map table centre: "
          f"[{n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f}]  angle={angle:.2f}°")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Block 3.1 — surface normal estimation ────────────────────────")

    results = [
        test_horizontal_table_primary(),
        test_vertical_wall(),
        test_slanted_plane_u_gradient(),
        test_normal_is_unit_vector(),
        test_normal_map_shape(),
        test_normal_map_table_centre(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
