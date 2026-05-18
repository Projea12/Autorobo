"""
eval_full_pipeline.py — End-to-End Pick-and-Place Benchmark (Phase 10)

Evaluates the complete pipeline: language instruction → navigate → perceive → grasp → place.

Metrics:
  - Task completion rate (full pick-and-place success)
  - Phase-level success rates (nav success, grasp success, place success)
  - Average task completion time
  - Failure phase distribution (where in the pipeline tasks fail)
  - Recovery system activation rate

Task Scenarios:
  1. Single object pick-and-place (easy)
  2. Pick specific object among distractors (medium)
  3. Multi-step rearrangement ("clear the table") (hard)
  4. Language-conditioned selection ("the heavy one") (hard)

Usage:
    python benchmarks/eval_full_pipeline.py
    python benchmarks/eval_full_pipeline.py --scenario single_object --n-episodes 50
    python benchmarks/eval_full_pipeline.py --all-scenarios
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from benchmarks.metrics import compute_success_rate

log = logging.getLogger(__name__)

# Task scenario definitions
SCENARIOS = {
    "single_object": {
        "description": "Pick up a single object and place it at goal",
        "difficulty": "easy",
        "n_objects": 1,
        "n_distractors": 0,
        "language_template": "Pick up the {object} and put it on the {target}.",
    },
    "with_distractors": {
        "description": "Pick specific object among distractors",
        "difficulty": "medium",
        "n_objects": 1,
        "n_distractors": 3,
        "language_template": "Find the {object} and move it to the {target}.",
    },
    "multi_step": {
        "description": "Multi-step rearrangement task",
        "difficulty": "hard",
        "n_objects": 3,
        "n_distractors": 2,
        "language_template": "Clear the table — put all the {object_class} objects in the {target}.",
    },
}

# Pipeline phases for failure tracking
PHASE_PLANNING = "task_planning"
PHASE_NAVIGATION = "navigation"
PHASE_PERCEPTION = "perception"
PHASE_GRASPING = "grasping"
PHASE_PLACING = "placing"


def run_pipeline_episode(
    nav_model,
    grasp_model,
    task_planner,
    env,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """
    Run a single end-to-end pick-and-place episode.

    Args:
        nav_model: Loaded PPO navigation model.
        grasp_model: Loaded SAC grasp model.
        task_planner: TaskPlanner instance (Claude API).
        env: Full pipeline gymnasium environment.
        scenario: Scenario config dict.

    Returns:
        Episode result dict:
        {
            "success": bool,
            "failure_phase": str | None,
            "nav_success": bool,
            "grasp_success": bool,
            "place_success": bool,
            "n_recovery_activations": int,
            "total_time_seconds": float,
            "task_plan": TaskGraph,
        }

    TODO: Phase 10 — orchestrate full pipeline:
    1. task_planner.plan(instruction, world_state) → TaskGraph
    2. task_executor.execute(task_graph) → step through nodes
    3. Record success/failure at each phase
    4. Count recovery system activations
    """
    raise NotImplementedError(
        "TODO: Phase 10 — implement full pipeline episode. "
        "Chain: plan → navigate → perceive → grasp → place. "
        "Track which phase fails and recovery activations."
    )


def evaluate_full_pipeline(
    nav_model_path: Path,
    grasp_model_path: Path,
    scenario_name: str = "single_object",
    n_episodes: int = 30,
) -> dict[str, Any]:
    """
    Run the full pipeline benchmark for a given scenario.

    Args:
        nav_model_path: Path to PPO navigation model.
        grasp_model_path: Path to SAC grasp model.
        scenario_name: Key into SCENARIOS dict.
        n_episodes: Number of evaluation episodes.

    Returns:
        Aggregated metrics dict:
        {
            "scenario": str,
            "task_completion_rate": float,
            "nav_success_rate": float,
            "grasp_success_rate": float,
            "place_success_rate": float,
            "avg_task_time": float,
            "recovery_activation_rate": float,
            "failure_phase_distribution": {phase: rate, ...},
            "n_episodes": int,
        }

    TODO: Phase 10 — load models and task planner, create pipeline env,
    run n_episodes calls to run_pipeline_episode(), aggregate metrics.
    """
    scenario = SCENARIOS.get(scenario_name)
    if scenario is None:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {list(SCENARIOS.keys())}"
        )

    raise NotImplementedError(
        f"TODO: Phase 10 — evaluate full pipeline on scenario '{scenario_name}' "
        f"for {n_episodes} episodes. Load nav from {nav_model_path}, "
        f"grasp from {grasp_model_path}."
    )


def print_pipeline_report(metrics: dict[str, Any]) -> None:
    """
    Print a formatted full pipeline benchmark report.

    Args:
        metrics: Dict from evaluate_full_pipeline().
    """
    print("\n" + "=" * 60)
    print(f"  Full Pipeline Benchmark — Scenario: {metrics['scenario']}")
    print("=" * 60)
    print(f"  Episodes:               {metrics['n_episodes']}")
    print(f"  Task Completion Rate:   {metrics['task_completion_rate']:.1%}  (target: > 60%)")
    print(f"  Navigation Success:     {metrics['nav_success_rate']:.1%}")
    print(f"  Grasp Success:          {metrics['grasp_success_rate']:.1%}")
    print(f"  Place Success:          {metrics['place_success_rate']:.1%}")
    print(f"  Avg Task Time:          {metrics['avg_task_time']:.1f}s")
    print(f"  Recovery Activations:   {metrics['recovery_activation_rate']:.1%} of episodes")
    print("\n  Failure Phase Distribution:")
    for phase, rate in metrics.get("failure_phase_distribution", {}).items():
        print(f"    {phase:<20}: {rate:.1%}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline evaluation benchmark")
    parser.add_argument(
        "--nav-model",
        type=Path,
        default=Path("models/navigation/ppo/best_model.zip"),
    )
    parser.add_argument(
        "--grasp-model",
        type=Path,
        default=Path("models/grasping/sac/sac_grasp_final.zip"),
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="single_object",
    )
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all scenarios sequentially",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    scenarios = list(SCENARIOS.keys()) if args.all_scenarios else [args.scenario]
    for scenario in scenarios:
        metrics = evaluate_full_pipeline(
            nav_model_path=args.nav_model,
            grasp_model_path=args.grasp_model,
            scenario_name=scenario,
            n_episodes=args.n_episodes,
        )
        print_pipeline_report(metrics)


if __name__ == "__main__":
    main()
