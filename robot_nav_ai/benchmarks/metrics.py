"""
metrics.py — Shared Metric Computation Utilities

Provides standardised metric computation functions used across all
evaluation benchmarks. All functions are pure (no side effects) and
operate on lists of episode result dicts.

Reference metrics:
  - SPL: Anderson et al. (2018) https://arxiv.org/abs/1807.06757
  - Grasp success: standard pick-and-place benchmark definition
"""

from __future__ import annotations

import math
from typing import Any


def compute_spl(episodes: list[dict[str, Any]]) -> float:
    """
    Compute SPL (Success weighted by Path Length).

    SPL = (1/N) * sum_i [ S_i * L_i* / max(P_i, L_i*) ]

    Where:
        S_i     = 1 if episode i succeeded, 0 otherwise
        L_i*    = optimal (shortest) path length for episode i
        P_i     = actual path length taken by the agent in episode i
        N       = total number of episodes

    Args:
        episodes: List of episode dicts, each containing:
            - "success": bool
            - "path_length": float (actual path in metres)
            - "optimal_path_length": float (shortest possible path in metres)

    Returns:
        SPL score in [0.0, 1.0]. 1.0 = perfect (all successes via optimal path).

    Raises:
        ValueError: If any episode has non-positive optimal_path_length.
    """
    if not episodes:
        return 0.0

    total = 0.0
    for ep in episodes:
        success = float(ep["success"])
        actual = ep["path_length"]
        optimal = ep["optimal_path_length"]

        if optimal <= 0:
            raise ValueError(
                f"optimal_path_length must be > 0, got {optimal}. "
                "Check that goal position differs from start position."
            )

        # SPL contribution: S_i * L* / max(P, L*)
        total += success * optimal / max(actual, optimal)

    return total / len(episodes)


def compute_success_rate(episodes: list[dict[str, Any]]) -> float:
    """
    Compute the fraction of episodes that succeeded.

    Args:
        episodes: List of episode dicts with "success": bool field.

    Returns:
        Success rate in [0.0, 1.0].
    """
    if not episodes:
        return 0.0
    return sum(1 for ep in episodes if ep["success"]) / len(episodes)


def compute_collision_rate(episodes: list[dict[str, Any]]) -> float:
    """
    Compute the fraction of episodes with at least one collision.

    Args:
        episodes: List of episode dicts with "collision": bool field.

    Returns:
        Collision rate in [0.0, 1.0].
    """
    if not episodes:
        return 0.0
    return sum(1 for ep in episodes if ep.get("collision", False)) / len(episodes)


def compute_mean_steps(
    episodes: list[dict[str, Any]],
    successful_only: bool = True,
) -> float:
    """
    Compute mean number of steps to task completion.

    Args:
        episodes: List of episode dicts with "n_steps": int field.
        successful_only: If True, only count successful episodes.

    Returns:
        Mean steps, or 0.0 if no qualifying episodes.
    """
    qualifying = [ep for ep in episodes if not successful_only or ep["success"]]
    if not qualifying:
        return 0.0
    return sum(ep["n_steps"] for ep in qualifying) / len(qualifying)


def compute_mean_time(
    episodes: list[dict[str, Any]],
    successful_only: bool = True,
) -> float:
    """
    Compute mean task completion time in seconds.

    Args:
        episodes: List of episode dicts with "time_seconds": float field.
        successful_only: If True, only count successful episodes.

    Returns:
        Mean time in seconds, or 0.0 if no qualifying episodes.
    """
    qualifying = [ep for ep in episodes if not successful_only or ep["success"]]
    if not qualifying:
        return 0.0
    return sum(ep["time_seconds"] for ep in qualifying) / len(qualifying)


def compute_failure_mode_distribution(
    episodes: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Compute the distribution of failure modes across failed episodes.

    Args:
        episodes: List of episode dicts with "failure_mode": str | None field.
            failure_mode is None for successful episodes.

    Returns:
        Dict mapping failure mode name → fraction of failed episodes.
        E.g. {"miss": 0.4, "slip": 0.3, "timeout": 0.3}
    """
    failed = [ep for ep in episodes if not ep["success"]]
    if not failed:
        return {}

    counts: dict[str, int] = {}
    for ep in failed:
        mode = ep.get("failure_mode", "unknown") or "unknown"
        counts[mode] = counts.get(mode, 0) + 1

    return {mode: count / len(failed) for mode, count in counts.items()}


def compute_per_class_success(
    episodes: list[dict[str, Any]],
    class_key: str = "object_id",
) -> dict[str, dict[str, Any]]:
    """
    Compute success rates broken down by object class.

    Args:
        episodes: List of episode dicts with "success": bool and
            the field specified by class_key.
        class_key: Key in episode dict to group by (e.g. "object_id").

    Returns:
        Dict: {class_name: {"success_rate": float, "n_episodes": int}}
    """
    classes: dict[str, list[bool]] = {}
    for ep in episodes:
        cls = ep.get(class_key, "unknown")
        if cls not in classes:
            classes[cls] = []
        classes[cls].append(ep["success"])

    return {
        cls: {
            "success_rate": sum(successes) / len(successes),
            "n_episodes": len(successes),
        }
        for cls, successes in classes.items()
    }


def compute_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute a confidence interval for a list of values using normal approximation.

    Args:
        values: List of scalar values (e.g. episode rewards or binary success).
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        (lower_bound, upper_bound) of the confidence interval.

    Note: Uses normal approximation — valid for n > 30.
    For small n, use bootstrapping instead.
    """
    if not values:
        return 0.0, 0.0

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std_err = math.sqrt(variance / n)

    # Z-score for common confidence levels
    z_scores = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_scores.get(confidence, 1.960)

    margin = z * std_err
    return mean - margin, mean + margin


def aggregate_metrics(episodes: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute all standard metrics from a list of episodes in one call.

    Args:
        episodes: List of episode result dicts (from any benchmark).

    Returns:
        Dict with all computed metrics.
    """
    metrics = {
        "n_episodes": len(episodes),
        "success_rate": compute_success_rate(episodes),
        "collision_rate": compute_collision_rate(episodes),
        "mean_steps": compute_mean_steps(episodes, successful_only=True),
    }

    # SPL only if path lengths are available
    if episodes and "path_length" in episodes[0]:
        metrics["spl"] = compute_spl(episodes)

    # Time if available
    if episodes and "time_seconds" in episodes[0]:
        metrics["mean_time"] = compute_mean_time(episodes, successful_only=True)

    # Failure modes if available
    if episodes and "failure_mode" in episodes[0]:
        metrics["failure_modes"] = compute_failure_mode_distribution(episodes)

    # Confidence interval on success rate
    successes = [float(ep["success"]) for ep in episodes]
    ci_low, ci_high = compute_confidence_interval(successes)
    metrics["success_rate_ci_low"] = ci_low
    metrics["success_rate_ci_high"] = ci_high

    return metrics
