"""
tests/test_depth_calibration.py — Acceptance test for Block 2.1 depth accuracy.

Acceptance criterion
--------------------
    Depth estimate within ±15% of a ruler measurement at 0.5m, 1.0m, 1.5m.

Two modes
---------
  1. Synthetic (automatic, no hardware):
     Verifies that the sampling and back-projection logic is correct by
     injecting a known constant depth map and checking the pipeline returns
     the right value.

  2. Physical (requires camera + printed target + ruler):
     Run:
         python tests/test_depth_calibration.py --physical --distances 0.5 1.0 1.5

     The script opens your webcam, captures one frame per distance as you
     press SPACE, runs the metric model, and reports % error against ground truth.

Usage
-----
    # Synthetic only (fast, no hardware):
    python tests/test_depth_calibration.py

    # Physical calibration (camera + ruler required):
    python tests/test_depth_calibration.py --physical --distances 0.5 1.0 1.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOLERANCE = 0.15   # ±15 %


# ── helpers ───────────────────────────────────────────────────────────────────

def _pct_error(estimate: float, truth: float) -> float:
    return abs(estimate - truth) / truth


def _make_bbox(cx: int, cy: int, half: int = 40) -> tuple:
    return (cx - half, cy - half, cx + half, cy + half)


# ── synthetic tests ───────────────────────────────────────────────────────────

def run_synthetic() -> bool:
    """
    Inject known metric depth maps and verify the sampling pipeline.
    No camera or model required.
    """
    from ar.localiser import _inner_region, INNER_FRAC

    print("\n── Synthetic calibration tests ──────────────────────────────────")
    passed = 0
    total  = 0

    test_depths = [0.5, 1.0, 1.5, 2.5, 4.0]

    for true_depth in test_depths:
        total += 1
        # Build a 480×640 depth map filled with the true value
        H, W      = 480, 640
        depth_map = np.full((H, W), true_depth, dtype=np.float32)

        # Add mild Gaussian noise to simulate real sensor noise
        noise = np.random.normal(0, true_depth * 0.02, (H, W)).astype(np.float32)
        depth_map = (depth_map + noise).clip(0.01, None)

        # Place a bbox in the centre
        bbox      = _make_bbox(W // 2, H // 2, half=60)
        patch     = _inner_region(bbox, depth_map)
        valid     = patch[patch > 0.01]
        estimated = float(np.median(valid))

        pct = _pct_error(estimated, true_depth)
        ok  = pct <= TOLERANCE
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        print(f"  [{status}]  true={true_depth:.2f}m  estimated={estimated:.3f}m  "
              f"error={pct*100:.1f}%")

    print(f"\nSynthetic: {passed}/{total} passed\n")
    return passed == total


# ── physical calibration ──────────────────────────────────────────────────────

def run_physical(distances: list[float]) -> bool:
    """
    Open webcam, capture a frame at each ground-truth distance (user presses SPACE),
    run DepthAnything V2 Metric-Indoor, report % error.
    """
    from ar.depth_estimator import DepthEstimator, DepthConfig
    from ar.localiser import _inner_region

    print("\n── Physical calibration ─────────────────────────────────────────")
    print("   Place a flat target (box / book) at the given distance.")
    print("   Centre it in the camera view, then press SPACE to capture.\n")

    cfg       = DepthConfig(metric=True)
    estimator = DepthEstimator(cfg)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam — is it connected?")
        return False

    results = []
    for true_d in distances:
        print(f"  → Place target at {true_d:.2f} m, press SPACE to capture …")
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            display = frame.copy()
            H, W    = frame.shape[:2]

            # Guide rectangle — inner 50% of a centred 200×200 box
            bx1, by1 = W // 2 - 100, H // 2 - 100
            bx2, by2 = W // 2 + 100, H // 2 + 100
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            cv2.putText(display,
                        f"Target: {true_d:.2f}m  |  SPACE=capture  Q=skip",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Depth Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                depth_map = estimator.estimate(frame)
                bbox  = (bx1, by1, bx2, by2)
                patch = _inner_region(bbox, depth_map)
                valid = patch[patch > 0.01]
                if valid.size == 0:
                    print("    No valid depth pixels — try again.")
                    continue
                estimated = float(np.median(valid))
                pct  = _pct_error(estimated, true_d)
                ok_  = pct <= TOLERANCE
                results.append((true_d, estimated, pct, ok_))
                print(f"    Captured: true={true_d:.2f}m  est={estimated:.3f}m  "
                      f"error={pct*100:.1f}%  {'PASS' if ok_ else 'FAIL'}")
                break
            elif key == ord("q"):
                print(f"    Skipped {true_d:.2f}m")
                break

    cap.release()
    cv2.destroyAllWindows()

    if not results:
        print("No results captured.")
        return False

    print("\n── Summary ──────────────────────────────────────────────────────")
    passed = 0
    for true_d, est, pct, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {true_d:.2f}m  →  {est:.3f}m  ({pct*100:.1f}% error)")
        if ok:
            passed += 1
    print(f"\nPhysical: {passed}/{len(results)} within ±{TOLERANCE*100:.0f}%\n")
    return passed == len(results)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth calibration acceptance test (Block 2.1)"
    )
    parser.add_argument("--physical", action="store_true",
                        help="Run physical camera test (requires webcam)")
    parser.add_argument("--distances", nargs="+", type=float,
                        default=[0.5, 1.0, 1.5],
                        help="Ground-truth distances in metres (physical mode)")
    args = parser.parse_args()

    synth_ok = run_synthetic()

    if args.physical:
        phys_ok = run_physical(args.distances)
        overall = synth_ok and phys_ok
    else:
        print("[info] Run with --physical to test against a real ruler measurement.")
        overall = synth_ok

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
