"""
evaluate.py — Full Evaluation Suite (Phase 10 onward)

Runs the complete evaluation pipeline across all sub-systems:
  1. Navigation benchmark (SPL, success rate, collision rate)
  2. Grasping benchmark (grasp success per YCB object)
  3. Full end-to-end pick-and-place pipeline

Results are logged to W&B and written as JSON to the eval output dir.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py eval.nav_model=models/navigation/ppo/final_model.zip
    python scripts/evaluate.py eval.n_episodes=100
    python scripts/evaluate.py eval.tasks=navigation,grasping
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def load_navigation_model(model_path: Path):
    """
    Load a trained PPO navigation model from disk.

    Args:
        model_path: Path to the .zip checkpoint.

    Returns:
        Loaded PPO model in evaluation mode.

    TODO: Phase 10 — use stable_baselines3.PPO.load(model_path).
    """
    raise NotImplementedError(
        f"TODO: Phase 10 — load PPO model from {model_path} using "
        "stable_baselines3.PPO.load()."
    )


def load_grasp_model(model_path: Path):
    """
    Load a trained SAC grasp model from disk.

    Args:
        model_path: Path to the .zip checkpoint.

    Returns:
        Loaded SAC model in evaluation mode.

    TODO: Phase 10 — use stable_baselines3.SAC.load(model_path).
    """
    raise NotImplementedError(
        f"TODO: Phase 10 — load SAC model from {model_path} using "
        "stable_baselines3.SAC.load()."
    )


def run_navigation_eval(cfg: DictConfig, model) -> dict[str, Any]:
    """
    Run the navigation evaluation benchmark.

    Args:
        cfg: Hydra config.
        model: Loaded PPO model.

    Returns:
        Dict of metrics: {spl, success_rate, collision_rate, avg_steps}.

    TODO: Phase 10 — call benchmarks/eval_navigation.py logic,
    aggregate over n_eval_episodes, return metric dict.
    """
    raise NotImplementedError(
        "TODO: Phase 10 — run nav eval across cfg.eval.n_episodes episodes, "
        "compute SPL, success_rate, collision_rate."
    )


def run_grasping_eval(cfg: DictConfig, model) -> dict[str, Any]:
    """
    Run the grasping evaluation benchmark.

    Args:
        cfg: Hydra config.
        model: Loaded SAC model.

    Returns:
        Dict of metrics: {grasp_success_rate, per_object_success, avg_time}.

    TODO: Phase 10 — run grasp eval on all YCB objects, log per-object breakdown.
    """
    raise NotImplementedError(
        "TODO: Phase 10 — run grasping eval across all YCB object classes, "
        "return per-object and aggregate success rates."
    )


def run_full_pipeline_eval(cfg: DictConfig, nav_model, grasp_model) -> dict[str, Any]:
    """
    Run end-to-end pick-and-place evaluation.

    Args:
        cfg: Hydra config.
        nav_model: Loaded navigation policy.
        grasp_model: Loaded grasp policy.

    Returns:
        Dict of metrics: {task_success_rate, avg_completion_time, failure_modes}.

    TODO: Phase 10 — chain nav → grasp → place, record failures by phase,
    compute overall task success rate.
    """
    raise NotImplementedError(
        "TODO: Phase 10 — implement full pipeline eval: navigate to object, "
        "grasp it, navigate to goal, place it. Log each failure mode."
    )


def save_results(results: dict[str, Any], output_dir: Path) -> None:
    """
    Save evaluation results as JSON and log summary to console.

    Args:
        results: Nested dict of all benchmark metrics.
        output_dir: Directory to write eval_results.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_path}")
    log.info("=== Evaluation Summary ===")
    for task, metrics in results.items():
        log.info(f"  [{task}]")
        for k, v in metrics.items():
            log.info(f"    {k}: {v}")


@hydra.main(config_path="../configs/hydra", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Orchestrates the full evaluation suite.

    Args:
        cfg: Composed Hydra config.
    """
    log.info("=== AutoRobo Full Evaluation Suite ===")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path(cfg.project.log_dir) / "eval"
    results: dict[str, Any] = {}

    # Load models
    nav_model_path = Path(cfg.project.model_dir) / "navigation/ppo/final_model.zip"
    grasp_model_path = Path(cfg.project.model_dir) / "grasping/sac/sac_grasp_final.zip"

    raise NotImplementedError(
        "TODO: Phase 10 — load both models, run all three benchmarks, "
        "aggregate results, call save_results(results, output_dir), "
        "and optionally push metrics to W&B with wandb.log(results)."
    )

    # nav_model = load_navigation_model(nav_model_path)
    # grasp_model = load_grasp_model(grasp_model_path)

    # results["navigation"] = run_navigation_eval(cfg, nav_model)
    # results["grasping"] = run_grasping_eval(cfg, grasp_model)
    # results["full_pipeline"] = run_full_pipeline_eval(cfg, nav_model, grasp_model)

    # save_results(results, output_dir)


if __name__ == "__main__":
    main()
