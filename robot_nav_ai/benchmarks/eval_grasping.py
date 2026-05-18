"""
eval_grasping.py — Grasping Evaluation Benchmark (Phase 9)

Measures grasp policy performance on YCB objects with per-object breakdown.

Metrics:
  - Overall grasp success rate
  - Per-object grasp success rate (breakdown by YCB class)
  - Average grasp attempt time (seconds)
  - Grasp stability rate (object held for > 3 seconds after lift)
  - Failure mode distribution (miss / slip / collision / timeout)

YCB Object Set (subset used for evaluation):
  003_cracker_box, 005_tomato_soup_can, 006_mustard_bottle,
  011_banana, 021_bleach_cleanser, 024_bowl, 025_mug, 035_power_drill

Usage:
    python benchmarks/eval_grasping.py
    python benchmarks/eval_grasping.py --model models/grasping/sac/sac_grasp_final.zip
    python benchmarks/eval_grasping.py --objects mug,banana --n-per-object 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from benchmarks.metrics import compute_success_rate

log = logging.getLogger(__name__)

# YCB objects used in evaluation
YCB_EVAL_OBJECTS = [
    "003_cracker_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "011_banana",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
]

# Failure mode constants
FAILURE_MISS = "miss"           # gripper did not contact object
FAILURE_SLIP = "slip"           # object slipped out after grasp
FAILURE_COLLISION = "collision" # arm hit table or other object
FAILURE_TIMEOUT = "timeout"     # episode exceeded max steps


def run_grasp_episode(model, env, object_id: str) -> dict[str, Any]:
    """
    Run a single grasp attempt on a specific object.

    Args:
        model: Loaded SAC model.
        env: Grasping gymnasium environment configured for object_id.
        object_id: YCB object identifier string.

    Returns:
        Episode result dict:
        {
            "object_id": str,
            "success": bool,
            "failure_mode": str | None,
            "n_steps": int,
            "time_seconds": float,
            "grasp_stable": bool,   # held > 3s after lift
        }

    TODO: Phase 9 — roll out episode with model.predict(deterministic=True),
    detect success via env.unwrapped.grasp_success, classify failure mode.
    """
    raise NotImplementedError(
        f"TODO: Phase 9 — implement grasp episode rollout for {object_id}. "
        "Track whether lift was achieved and maintained for stability check."
    )


def evaluate_grasping(
    model_path: Path,
    object_ids: list[str] | None = None,
    n_per_object: int = 20,
    deterministic: bool = True,
) -> dict[str, Any]:
    """
    Run grasping evaluation across all YCB objects.

    Args:
        model_path: Path to trained SAC .zip checkpoint.
        object_ids: List of YCB object IDs. Defaults to YCB_EVAL_OBJECTS.
        n_per_object: Episodes per object class.
        deterministic: Use deterministic policy.

    Returns:
        Nested metrics dict:
        {
            "overall_success_rate": float,
            "overall_stability_rate": float,
            "avg_grasp_time": float,
            "failure_mode_distribution": {"miss": float, "slip": float, ...},
            "per_object": {
                "003_cracker_box": {"success_rate": float, "n_episodes": int},
                ...
            }
        }

    TODO: Phase 9 — iterate object_ids, spawn each in env, run n_per_object episodes,
    aggregate results with compute_success_rate() from metrics.py.
    """
    if object_ids is None:
        object_ids = YCB_EVAL_OBJECTS

    raise NotImplementedError(
        f"TODO: Phase 9 — evaluate {len(object_ids)} objects × {n_per_object} episodes "
        f"using model from {model_path}. Return nested metrics dict."
    )


def print_grasping_report(metrics: dict[str, Any]) -> None:
    """
    Print a formatted grasping benchmark report.

    Args:
        metrics: Dict from evaluate_grasping().
    """
    print("\n" + "=" * 60)
    print("  Grasping Benchmark Results")
    print("=" * 60)
    print(f"  Overall Success Rate: {metrics['overall_success_rate']:.1%}  (target: > 70%)")
    print(f"  Grasp Stability Rate: {metrics['overall_stability_rate']:.1%}")
    print(f"  Avg Grasp Time:       {metrics['avg_grasp_time']:.1f}s")
    print("\n  Failure Mode Distribution:")
    for mode, rate in metrics.get("failure_mode_distribution", {}).items():
        print(f"    {mode:<12}: {rate:.1%}")
    print("\n  Per-Object Success Rates:")
    for obj_id, obj_metrics in metrics.get("per_object", {}).items():
        bar = "#" * int(obj_metrics["success_rate"] * 20)
        print(f"    {obj_id:<30} {obj_metrics['success_rate']:.1%} [{bar:<20}]")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grasping evaluation benchmark")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/grasping/sac/sac_grasp_final.zip"),
    )
    parser.add_argument("--objects", type=str, default=None,
                        help="Comma-separated YCB object IDs")
    parser.add_argument("--n-per-object", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    object_ids = args.objects.split(",") if args.objects else None
    metrics = evaluate_grasping(
        model_path=args.model,
        object_ids=object_ids,
        n_per_object=args.n_per_object,
    )
    print_grasping_report(metrics)


if __name__ == "__main__":
    main()
