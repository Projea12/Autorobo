"""
tests/test_back_project.py — Acceptance test for Block 2.2 back-projection.

Acceptance criterion
--------------------
    A mug at pixel (u=180, v=400) with metric depth d=0.8 m maps to the
    correct 3D point in the camera frame using pin-hole back-projection:

        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

    Test uses a standard VGA camera (640×480, fx=fy=500, cx=320, cy=240).
    Tolerance: 1e-6 m (numerical precision only — no model involved).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ar.localiser import Localiser

TOLERANCE_M = 1e-6   # pure math — should be exact


# ── minimal intrinsics stub ───────────────────────────────────────────────────

@dataclass
class _K:
    fx: float
    fy: float
    cx: float
    cy: float


# ── test cases ────────────────────────────────────────────────────────────────

def _run(label: str, u, v, d, K, expected_X, expected_Y, expected_Z) -> bool:
    X, Y, Z = Localiser.back_project(u, v, d, K)
    ok = (
        abs(X - expected_X) < TOLERANCE_M and
        abs(Y - expected_Y) < TOLERANCE_M and
        abs(Z - expected_Z) < TOLERANCE_M
    )
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}]  {label}")
    print(f"           pixel=({u},{v})  d={d}m")
    print(f"           expected  X={expected_X:+.6f}  Y={expected_Y:+.6f}  Z={expected_Z:.6f}")
    print(f"           got       X={X:+.6f}  Y={Y:+.6f}  Z={Z:.6f}")
    return ok


def test_back_project() -> None:
    print("\n── Block 2.2 — back-projection acceptance tests ─────────────────")

    # Standard VGA intrinsics used throughout these tests
    K = _K(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    results = []

    # ── Primary acceptance case: mug at (u=180, v=400), d=0.8m ───────────────
    #   X = (180 - 320) * 0.8 / 500 = -140 * 0.8 / 500 = -0.224
    #   Y = (400 - 240) * 0.8 / 500 =  160 * 0.8 / 500 =  0.256
    #   Z = 0.8
    results.append(_run(
        "mug at (180, 400) d=0.8m",
        u=180, v=400, d=0.8,
        K=K,
        expected_X=-0.224,
        expected_Y= 0.256,
        expected_Z= 0.8,
    ))

    # ── Optical-axis centre: pixel on cx,cy → X=Y=0 ───────────────────────────
    #   X = (320 - 320) * 1.0 / 500 = 0
    #   Y = (240 - 240) * 1.0 / 500 = 0
    #   Z = 1.0
    results.append(_run(
        "optical axis (cx, cy) d=1.0m → X=Y=0",
        u=320, v=240, d=1.0,
        K=K,
        expected_X=0.0,
        expected_Y=0.0,
        expected_Z=1.0,
    ))

    # ── Right of centre: pixel right of cx → positive X ───────────────────────
    #   X = (420 - 320) * 0.5 / 500 = 100 * 0.5 / 500 = 0.1
    #   Y = (240 - 240) * 0.5 / 500 = 0
    #   Z = 0.5
    results.append(_run(
        "right of centre (420, 240) d=0.5m → X=+0.1",
        u=420, v=240, d=0.5,
        K=K,
        expected_X=0.1,
        expected_Y=0.0,
        expected_Z=0.5,
    ))

    # ── Asymmetric focal lengths (fx ≠ fy) ────────────────────────────────────
    K2 = _K(fx=600.0, fy=400.0, cx=320.0, cy=240.0)
    #   X = (200 - 320) * 1.5 / 600 = -120 * 1.5 / 600 = -0.3
    #   Y = (360 - 240) * 1.5 / 400 =  120 * 1.5 / 400 =  0.45
    #   Z = 1.5
    results.append(_run(
        "asymmetric K (fx=600, fy=400) pixel=(200,360) d=1.5m",
        u=200, v=360, d=1.5,
        K=K2,
        expected_X=-0.3,
        expected_Y= 0.45,
        expected_Z= 1.5,
    ))

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) failed"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    test_back_project()
