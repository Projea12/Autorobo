"""
tests/test_approach_vector.py — Acceptance tests for Block 3.2 approach vector.

Acceptance criterion (from spec)
---------------------------------
    Correctly classifies table-top vs vertical-surface approach from depth map.

    Table-top object  → approach [0, +1, 0]  (TOP_DOWN,   downward in cam frame)
    Shelf/wall object → approach [0,  0, +1]  (HORIZONTAL, forward  in cam frame)

Approach vector = -surface_normal (arm arrives perpendicular to surface).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ar.grasp_pose import (
    ApproachType, GraspApproach,
    approach_vector, classify_approach, estimate_approach,
)
from ar.surface_normal import estimate_normal

ANGLE_TOL_DEG = 5.0


@dataclass
class _K:
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0


def _angle_deg(a, b):
    dot = float(np.clip(np.dot(
        np.array(a) / np.linalg.norm(a),
        np.array(b) / np.linalg.norm(b),
    ), -1.0, 1.0))
    return np.degrees(np.arccos(abs(dot)))


# ── synthetic depth helpers ───────────────────────────────────────────────────

def _table_depth(H=480, W=640, K=_K(), h=1.0):
    """Horizontal plane Z(u,v) = h*fy/(v-cy) — surface normal [0,-1,0]."""
    depth = np.zeros((H, W), dtype=np.float32)
    for v in range(int(K.cy) + 1, H):
        depth[v, :] = h * K.fy / (v - K.cy)
    return depth


def _shelf_depth(H=480, W=640, d=2.0):
    """Vertical wall at constant depth — surface normal [0,0,-1]."""
    return np.full((H, W), d, dtype=np.float32)


# ── approach_vector() unit tests ──────────────────────────────────────────────

def test_approach_is_negative_normal() -> bool:
    """approach_vector(n) must equal -n for any unit normal."""
    normals = [
        np.array([0.0, -1.0,  0.0]),
        np.array([0.0,  0.0, -1.0]),
        np.array([0.577, -0.577, 0.577]),
    ]
    ok = all(np.allclose(approach_vector(n), -n) for n in normals)
    print(f"  [{'PASS' if ok else 'FAIL'}]  approach_vector(n) == -n  for 3 normals")
    return ok


# ── classify_approach() unit tests ────────────────────────────────────────────

def test_classify_top_down() -> bool:
    """[0,+1,0] → TOP_DOWN with confidence 1.0."""
    t, conf = classify_approach(np.array([0.0, 1.0, 0.0]))
    ok = (t == ApproachType.TOP_DOWN and abs(conf - 1.0) < 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}]  [0,1,0] → {t.value}  conf={conf:.3f}  (expect TOP_DOWN 1.0)")
    return ok


def test_classify_horizontal() -> bool:
    """[0,0,+1] → HORIZONTAL with confidence 1.0."""
    t, conf = classify_approach(np.array([0.0, 0.0, 1.0]))
    ok = (t == ApproachType.HORIZONTAL and abs(conf - 1.0) < 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}]  [0,0,1] → {t.value}  conf={conf:.3f}  (expect HORIZONTAL 1.0)")
    return ok


def test_classify_lateral() -> bool:
    """[+1,0,0] → LATERAL."""
    t, _ = classify_approach(np.array([1.0, 0.0, 0.0]))
    ok = (t == ApproachType.LATERAL)
    print(f"  [{'PASS' if ok else 'FAIL'}]  [1,0,0] → {t.value}  (expect LATERAL)")
    return ok


def test_classify_confidence_45deg() -> bool:
    """45° between top-down and horizontal → confidence ≈ 0.5."""
    v = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
    _, conf = classify_approach(v)
    ok = abs(conf - 0.5) < 0.01
    print(f"  [{'PASS' if ok else 'FAIL'}]  45° approach → conf={conf:.3f}  (expect ~0.5)")
    return ok


# ── PRIMARY ACCEPTANCE — full depth-map pipeline ──────────────────────────────

def test_table_top_approach() -> bool:
    """
    PRIMARY ACCEPTANCE (table-top)

    Horizontal table depth map → estimate_approach() classifies as TOP_DOWN.
    Approach vector must be within 5° of [0, +1, 0].
    """
    K     = _K()
    depth = _table_depth(K=K, h=1.0)
    ga    = estimate_approach(depth, u=320, v=350, intrinsics=K)

    angle = _angle_deg(ga.approach_vec, [0.0, 1.0, 0.0])
    type_ok  = (ga.approach_type == ApproachType.TOP_DOWN)
    angle_ok = (angle <= ANGLE_TOL_DEG)
    ok = type_ok and angle_ok

    a = ga.approach_vec
    print(f"  [{'PASS' if ok else 'FAIL'}]  TABLE-TOP approach")
    print(f"         approach = ({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f})  "
          f"type={ga.approach_type.value}  conf={ga.confidence:.2f}  angle={angle:.2f}°")
    return ok


def test_shelf_approach() -> bool:
    """
    PRIMARY ACCEPTANCE (shelf/wall)

    Vertical shelf depth map → estimate_approach() classifies as HORIZONTAL.
    Approach vector must be within 5° of [0, 0, +1].
    """
    K     = _K()
    depth = _shelf_depth(d=2.0)
    ga    = estimate_approach(depth, u=320, v=240, intrinsics=K)

    angle    = _angle_deg(ga.approach_vec, [0.0, 0.0, 1.0])
    type_ok  = (ga.approach_type == ApproachType.HORIZONTAL)
    angle_ok = (angle <= ANGLE_TOL_DEG)
    ok = type_ok and angle_ok

    a = ga.approach_vec
    print(f"  [{'PASS' if ok else 'FAIL'}]  SHELF/WALL approach")
    print(f"         approach = ({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f})  "
          f"type={ga.approach_type.value}  conf={ga.confidence:.2f}  angle={angle:.2f}°")
    return ok


def test_grasp_approach_str() -> bool:
    """GraspApproach.__str__() renders without error."""
    K     = _K()
    depth = _table_depth(K=K)
    ga    = estimate_approach(depth, u=320, v=350, intrinsics=K)
    s     = str(ga)
    ok    = "top_down" in s and "conf=" in s
    print(f"  [{'PASS' if ok else 'FAIL'}]  GraspApproach str: {s}")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n── Block 3.2 — approach vector ──────────────────────────────────")

    results = [
        test_approach_is_negative_normal(),
        test_classify_top_down(),
        test_classify_horizontal(),
        test_classify_lateral(),
        test_classify_confidence_45deg(),
        test_table_top_approach(),
        test_shelf_approach(),
        test_grasp_approach_str(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} passed")
    assert passed == total, f"{total - passed} test(s) FAILED"
    print("Acceptance: PASS\n")


if __name__ == "__main__":
    main()
