"""
eval_navigation.py — Navigation Evaluation Benchmark (Phase 4)

Measures the performance of the trained navigation policy using standard
robot navigation metrics:

  - SPL (Success weighted by Path Length): primary metric
  - Success Rate: fraction of episodes reaching goal within threshold
  - Collision Rate: fraction of episodes with at least one collision
  - Average Steps: mean steps to reach goal (successful episodes only)
  - Average Path Length: mean path length vs optimal path length

Reference:
  Anderson et al. (2018) "On Evaluation of Embodied Navigation Agents"
  https://arxiv.org/abs/1807.06757

Usage:
    python benchmarks/eval_navigation.py
    python benchmarks/eval_navigation.py --model models/navigation/ppo/best_model.zip
    python benchmarks/eval_navigation.py --n-episodes 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from benchmarks.metrics import compute_spl, compute_success_rate, compute_collision_rate

log = logging.getLogger(__name__)


def run_episode(model, env) -> dict[str, Any]:
    """
    Run a single navigation evaluation episode.

    Args:
        model: Loaded PPO model (deterministic=True).
        env: Navigation gymnasium environment.

    Returns:
        Episode result dict:
        {
            "success": bool,
            "n_steps": int,
            "path_length": float,        # actual path length in metres
            "optimal_path_length": float, # shortest possible path
            "collision": bool,
            "final_distance": float       # distance to goal at episode end
        }

    TODO: Phase 4 — roll out one episode with model.predict(obs, deterministic=True),
    track position at each step, compute path length from position history.
    """
    raise NotImplementedError(
        "TODO: Phase 4 — implement episode rollout with position tracking. "
        "Use env.unwrapped.get_optimal_path_length() for optimal path."
    )


def evaluate_navigation(
    model_path: Path,
    n_episodes: int = 50,
    deterministic: bool = True,
) -> dict[str, float]:
    """
    Run the full navigation benchmark over multiple episodes.

    Args:
        model_path: Path to trained PPO .zip checkpoint.
        n_episodes: Number of evaluation episodes.
        deterministic: Use deterministic policy (no exploration noise).

    Returns:
        Dict of aggregated metrics:
        {
            "spl": float,              # 0.0 to 1.0
            "success_rate": float,     # 0.0 to 1.0
            "collision_rate": float,   # 0.0 to 1.0
            "avg_steps": float,
            "avg_path_length": float,
            "n_episodes": int,
        }

    TODO: Phase 4 — load PPO model, create eval env, run n_episodes calls to
    run_episode(), aggregate with metrics.py functions.
    """
    raise NotImplementedError(
        f"TODO: Phase 4 — run {n_episodes} evaluation episodes with model from "
        f"{model_path}, aggregate SPL, success_rate, collision_rate."
    )


def print_navigation_report(metrics: dict[str, float]) -> None:
    """
    Print a formatted navigation benchmark report to stdout.

    Args:
        metrics: Dict from evaluate_navigation().
    """
    print("\n" + "=" * 50)
    print("  Navigation Benchmark Results")
    print("=" * 50)
    print(f"  Episodes:        {metrics['n_episodes']}")
    print(f"  SPL:             {metrics['spl']:.3f}  (target: > 0.60)")
    print(f"  Success Rate:    {metrics['success_rate']:.1%}  (target: > 80%)")
    print(f"  Collision Rate:  {metrics['collision_rate']:.1%}  (target: < 5%)")
    print(f"  Avg Steps:       {metrics['avg_steps']:.1f}")
    print(f"  Avg Path Length: {metrics['avg_path_length']:.2f}m")
    print("=" * 50 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Navigation evaluation benchmark")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/navigation/ppo/best_model.zip"),
        help="Path to trained PPO checkpoint",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    metrics = evaluate_navigation(
        model_path=args.model,
        n_episodes=args.n_episodes,
        deterministic=not args.stochastic,
    )
    print_navigation_report(metrics)


if __name__ == "__main__":
    main()
