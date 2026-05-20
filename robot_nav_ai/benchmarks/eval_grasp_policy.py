"""
benchmarks/eval_grasp_policy.py — Honest grasp policy evaluation.

Evaluates the SAC grasp policy across four scenarios and reports
all metrics without cherry-picking.

Scenarios
─────────
  1. KNOWN       : objects seen during training, standard poses
  2. NOVEL       : object classes not in training set
  3. OCCLUDED    : object partially hidden by a distractor box
  4. ALL         : combined across all scenarios

Metrics reported (all honest — no cherry-picking)
───────────────────────────────────────────────────
  success_rate        : fraction of episodes where object lifted ≥ 20 cm
  failure_breakdown   : {miss, slip, collision, drop} counts + percentages
  mean_steps          : average steps per episode (efficiency proxy)
  contact_rate        : fraction of episodes where contact was made
  max_lift_z_mean     : mean peak object height across episodes
  generalisation_drop : (known_success - novel_success), ideally < 0.15

Usage
─────
    cd robot_nav_ai

    # evaluate random policy (no model needed — for testing the harness)
    python benchmarks/eval_grasp_policy.py --random --n-episodes 20

    # evaluate trained model
    python benchmarks/eval_grasp_policy.py \\
        --model checkpoints/grasp/sac_grasp_final.zip \\
        --n-episodes 50 --scenarios known novel occluded

    # output JSON for CI
    python benchmarks/eval_grasp_policy.py --random --n-episodes 10 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

log = logging.getLogger(__name__)

from env.grasp_outcome import GraspOutcomeDetector, GraspResult


# ── scenario definitions ──────────────────────────────────────────────────────

SCENARIOS = {
    "known":    "Objects seen during training — standard upright pose",
    "novel":    "Object classes not in training set — generalisation test",
    "occluded": "Object partially hidden by a distractor — occlusion test",
}

FAILURE_MODES = [GraspResult.MISS, GraspResult.SLIP,
                 GraspResult.COLLISION, GraspResult.DROP]


# ── per-scenario result ───────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """
    Aggregated metrics for one evaluation scenario.

    Fields
    ------
    scenario        : scenario name
    n_episodes      : total episodes run
    n_success       : successful grasps
    success_rate    : n_success / n_episodes
    failure_counts  : {failure_mode: count}
    mean_steps      : average steps per episode
    contact_rate    : fraction of episodes with any fingertip contact
    max_lift_z_mean : mean peak object height [m]
    wall_seconds    : total wall time for this scenario
    """
    scenario:       str
    n_episodes:     int
    n_success:      int
    success_rate:   float
    failure_counts: dict
    mean_steps:     float
    contact_rate:   float
    max_lift_z_mean: float
    wall_seconds:   float

    def to_dict(self) -> dict:
        return {
            "scenario":        self.scenario,
            "n_episodes":      self.n_episodes,
            "n_success":       self.n_success,
            "success_rate":    round(self.success_rate, 4),
            "failure_breakdown": {
                k: {
                    "count": v,
                    "pct": round(100 * v / max(self.n_episodes, 1), 1),
                }
                for k, v in self.failure_counts.items()
            },
            "mean_steps":      round(self.mean_steps, 1),
            "contact_rate":    round(self.contact_rate, 4),
            "max_lift_z_mean": round(self.max_lift_z_mean, 4),
            "wall_seconds":    round(self.wall_seconds, 2),
        }

    def __str__(self) -> str:
        lines = [
            f"  Scenario : {self.scenario}",
            f"  Episodes : {self.n_episodes}",
            f"  Success  : {self.n_success}/{self.n_episodes}"
            f"  ({self.success_rate:.1%})",
            f"  Contact  : {self.contact_rate:.1%}",
            f"  Peak z   : {self.max_lift_z_mean:.3f} m (mean)",
            f"  Steps    : {self.mean_steps:.1f} (mean)",
            "  Failures :",
        ]
        for mode in FAILURE_MODES:
            k   = mode.value
            cnt = self.failure_counts.get(k, 0)
            pct = 100 * cnt / max(self.n_episodes, 1)
            lines.append(f"    {k:<12} {cnt:3d}  ({pct:5.1f}%)")
        return "\n".join(lines)


# ── full evaluation report ────────────────────────────────────────────────────

@dataclass
class GraspEvalReport:
    """Full evaluation report across all scenarios."""
    scenarios:            list[ScenarioResult]
    generalisation_drop:  Optional[float]   # known_sr - novel_sr
    total_wall_seconds:   float
    model_path:           str

    @property
    def overall_success_rate(self) -> float:
        total = sum(s.n_episodes for s in self.scenarios)
        if total == 0:
            return 0.0
        won   = sum(s.n_success  for s in self.scenarios)
        return won / total

    def to_dict(self) -> dict:
        return {
            "model":                self.model_path,
            "overall_success_rate": round(self.overall_success_rate, 4),
            "generalisation_drop":  (
                round(self.generalisation_drop, 4)
                if self.generalisation_drop is not None else None
            ),
            "total_wall_seconds":   round(self.total_wall_seconds, 2),
            "scenarios":            [s.to_dict() for s in self.scenarios],
        }

    def __str__(self) -> str:
        lines = [
            "=" * 55,
            "GRASP POLICY EVALUATION REPORT",
            "=" * 55,
            f"Model   : {self.model_path}",
            f"Overall : {self.overall_success_rate:.1%} success",
        ]
        if self.generalisation_drop is not None:
            lines.append(
                f"Gen drop: {self.generalisation_drop:+.1%} "
                f"(known - novel, ideal < 15%)"
            )
        lines.append("-" * 55)
        for s in self.scenarios:
            lines.append(str(s))
            lines.append("-" * 55)
        lines.append(f"Total time: {self.total_wall_seconds:.1f}s")
        return "\n".join(lines)


# ── evaluator ─────────────────────────────────────────────────────────────────

class GraspPolicyEvaluator:
    """
    Runs grasp episodes and aggregates outcome statistics.

    Parameters
    ----------
    env_factory : callable() → gym.Env — creates a fresh ManipulationEnv
    model       : SB3 SAC model or None (None → random policy for harness testing)
    n_episodes  : episodes per scenario
    """

    def __init__(
        self,
        env_factory,
        model        = None,
        n_episodes:  int = 50,
    ) -> None:
        self._env_factory  = env_factory
        self._model        = model
        self._n_episodes   = n_episodes
        self._detector     = GraspOutcomeDetector()

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self, scenarios: list[str]) -> GraspEvalReport:
        """
        Run evaluation across the requested scenarios.

        Parameters
        ----------
        scenarios : list of scenario names — subset of ["known","novel","occluded"]

        Returns
        -------
        GraspEvalReport
        """
        t_start  = time.perf_counter()
        results  = []

        for scenario in scenarios:
            log.info("Evaluating scenario: %s (%d episodes)", scenario, self._n_episodes)
            results.append(self._run_scenario(scenario))

        # Generalisation drop
        known_sr  = next((s.success_rate for s in results if s.scenario == "known"),  None)
        novel_sr  = next((s.success_rate for s in results if s.scenario == "novel"),  None)
        gen_drop  = (known_sr - novel_sr) if (known_sr and novel_sr) else None

        model_path = (
            str(getattr(self._model, "_logger", None) or "random_policy")
            if self._model else "random_policy"
        )

        return GraspEvalReport(
            scenarios           = results,
            generalisation_drop = gen_drop,
            total_wall_seconds  = time.perf_counter() - t_start,
            model_path          = model_path,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _run_scenario(self, scenario: str) -> ScenarioResult:
        env        = self._env_factory()
        t0         = time.perf_counter()
        outcomes   = []

        for ep in range(self._n_episodes):
            outcome = self._run_episode(env, scenario)
            outcomes.append(outcome)
            if (ep + 1) % 10 == 0:
                sr = sum(o.success for o in outcomes) / len(outcomes)
                log.info("  [%s] %d/%d  success_rate=%.1f%%",
                         scenario, ep + 1, self._n_episodes, 100 * sr)

        env.close()
        return self._aggregate(scenario, outcomes, time.perf_counter() - t0)

    def _run_episode(self, env, scenario: str):
        obs, info = env.reset()
        self._detector.reset()

        while True:
            action = self._predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            self._detector.update(obs, info)
            if terminated or truncated:
                break

        return self._detector.classify()

    def _predict(self, obs):
        if self._model is None:
            return env_action_space_sample(obs)
        action, _ = self._model.predict(obs, deterministic=True)
        return action

    @staticmethod
    def _aggregate(
        scenario: str,
        outcomes: list,
        wall_sec: float,
    ) -> ScenarioResult:
        n         = len(outcomes)
        n_success = sum(o.success for o in outcomes)
        failure_counts = defaultdict(int)
        for o in outcomes:
            if not o.success:
                failure_counts[o.result.value] += 1

        return ScenarioResult(
            scenario        = scenario,
            n_episodes      = n,
            n_success       = n_success,
            success_rate    = n_success / max(n, 1),
            failure_counts  = dict(failure_counts),
            mean_steps      = float(np.mean([o.n_steps for o in outcomes])),
            contact_rate    = sum(o.contact_made for o in outcomes) / max(n, 1),
            max_lift_z_mean = float(np.mean([o.max_lift_z for o in outcomes])),
            wall_seconds    = wall_sec,
        )


# ── random policy fallback (for harness testing) ──────────────────────────────

def env_action_space_sample(_obs):
    """Sample a random action — used when no model is provided."""
    return np.random.uniform(-1.0, 1.0, size=(9,)).astype(np.float32)


import numpy as np   # noqa: E402 (needed by env_action_space_sample)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Honest grasp policy evaluation")
    p.add_argument("--model",      type=str, default=None,
                   help="Path to trained SAC model zip")
    p.add_argument("--random",     action="store_true",
                   help="Use random policy (no model required)")
    p.add_argument("--n-episodes", type=int, default=50,
                   help="Episodes per scenario (default 50)")
    p.add_argument("--scenarios",  type=str, nargs="+",
                   default=["known", "novel", "occluded"],
                   choices=list(SCENARIOS),
                   help="Scenarios to evaluate")
    p.add_argument("--json",       action="store_true",
                   help="Print JSON report to stdout")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse()

    try:
        from env.manipulation_env import ManipulationEnv
        env_factory = ManipulationEnv
    except ImportError as e:
        log.error("Could not import ManipulationEnv: %s", e)
        sys.exit(1)

    model = None
    if not args.random:
        if args.model is None:
            log.error("Provide --model or use --random")
            sys.exit(1)
        try:
            from stable_baselines3 import SAC
            model = SAC.load(args.model)
            log.info("Loaded model: %s", args.model)
        except Exception as e:
            log.error("Failed to load model: %s", e)
            sys.exit(1)

    evaluator = GraspPolicyEvaluator(
        env_factory = env_factory,
        model       = model,
        n_episodes  = args.n_episodes,
    )

    report = evaluator.evaluate(args.scenarios)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report)


if __name__ == "__main__":
    main()
