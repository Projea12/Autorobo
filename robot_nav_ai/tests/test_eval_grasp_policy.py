"""
tests/test_eval_grasp_policy.py — Unit tests for GraspPolicyEvaluator and report types.

No real env or model needed — MagicMock env and random policy are used.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from benchmarks.eval_grasp_policy import (
    GraspEvalReport,
    GraspPolicyEvaluator,
    ScenarioResult,
    SCENARIOS,
    env_action_space_sample,
)
from env.grasp_outcome import GraspResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_env(n_steps: int = 3, success: bool = False) -> MagicMock:
    """Build a mock gym env that terminates after n_steps."""
    env = MagicMock()
    obs = np.zeros(45, dtype=np.float32)

    # reset returns (obs, info)
    env.reset.return_value = (obs, {})

    # step returns (obs, reward, terminated, truncated, info)
    # terminates on the last step
    step_returns = [
        (obs, 0.0, False, False, {})
    ] * (n_steps - 1) + [
        (obs, 1.0, True, False, {"success": success})
    ]
    env.step.side_effect = step_returns * 1000   # repeat to handle multiple episodes
    return env


def _evaluator(n_episodes: int = 2, model=None) -> GraspPolicyEvaluator:
    env = _make_env(n_steps=3)

    def factory():
        # Reset side_effect on each call so episodes don't run out
        e = _make_env(n_steps=3)
        return e

    return GraspPolicyEvaluator(
        env_factory = factory,
        model       = model,
        n_episodes  = n_episodes,
    )


# ── ScenarioResult ────────────────────────────────────────────────────────────

class TestScenarioResult:
    def _result(self, success_rate: float = 0.5) -> ScenarioResult:
        n = 10
        return ScenarioResult(
            scenario        = "known",
            n_episodes      = n,
            n_success       = int(n * success_rate),
            success_rate    = success_rate,
            failure_counts  = {"miss": 3, "slip": 1, "collision": 1},
            mean_steps      = 12.5,
            contact_rate    = 0.6,
            max_lift_z_mean = 0.15,
            wall_seconds    = 2.0,
        )

    def test_to_dict_keys(self):
        d = self._result().to_dict()
        assert "scenario"          in d
        assert "success_rate"      in d
        assert "failure_breakdown" in d
        assert "mean_steps"        in d
        assert "contact_rate"      in d
        assert "max_lift_z_mean"   in d
        assert "wall_seconds"      in d

    def test_failure_breakdown_has_pct(self):
        d = self._result().to_dict()
        for v in d["failure_breakdown"].values():
            assert "count" in v
            assert "pct"   in v

    def test_str_contains_scenario_name(self):
        s = str(self._result())
        assert "known" in s

    def test_str_contains_success_rate(self):
        s = str(self._result(0.7))
        assert "70.0%" in s or "0.7" in s


# ── GraspEvalReport ───────────────────────────────────────────────────────────

class TestGraspEvalReport:
    def _sr(self, name, n, won, sr) -> ScenarioResult:
        return ScenarioResult(
            scenario=name, n_episodes=n, n_success=won, success_rate=sr,
            failure_counts={}, mean_steps=10.0,
            contact_rate=0.5, max_lift_z_mean=0.1, wall_seconds=1.0,
        )

    def test_overall_success_rate_single_scenario(self):
        sr     = self._sr("known", 10, 7, 0.7)
        report = GraspEvalReport(
            scenarios=[sr], generalisation_drop=None,
            total_wall_seconds=1.0, model_path="random",
        )
        assert report.overall_success_rate == pytest.approx(0.7)

    def test_overall_success_rate_combined(self):
        s1     = self._sr("known",  10, 8, 0.8)
        s2     = self._sr("novel",  10, 4, 0.4)
        report = GraspEvalReport(
            scenarios=[s1, s2], generalisation_drop=0.4,
            total_wall_seconds=2.0, model_path="random",
        )
        assert report.overall_success_rate == pytest.approx(12 / 20)

    def test_overall_success_rate_empty(self):
        report = GraspEvalReport(
            scenarios=[], generalisation_drop=None,
            total_wall_seconds=0.0, model_path="random",
        )
        assert report.overall_success_rate == pytest.approx(0.0)

    def test_to_dict_keys(self):
        sr     = self._sr("known", 10, 5, 0.5)
        report = GraspEvalReport(
            scenarios=[sr], generalisation_drop=None,
            total_wall_seconds=1.0, model_path="test",
        )
        d = report.to_dict()
        assert "model"                in d
        assert "overall_success_rate" in d
        assert "generalisation_drop"  in d
        assert "scenarios"            in d

    def test_str_contains_overall(self):
        sr     = self._sr("known", 10, 5, 0.5)
        report = GraspEvalReport(
            scenarios=[sr], generalisation_drop=None,
            total_wall_seconds=1.0, model_path="test",
        )
        s = str(report)
        assert "GRASP POLICY EVALUATION REPORT" in s
        assert "50.0%" in s

    def test_gen_drop_in_str(self):
        s1 = self._sr("known", 10, 8, 0.8)
        s2 = self._sr("novel", 10, 5, 0.5)
        report = GraspEvalReport(
            scenarios=[s1, s2], generalisation_drop=0.3,
            total_wall_seconds=2.0, model_path="test",
        )
        s = str(report)
        assert "Gen drop" in s or "gen" in s.lower()


# ── env_action_space_sample ───────────────────────────────────────────────────

class TestActionSample:
    def test_returns_array(self):
        obs = np.zeros(45)
        a   = env_action_space_sample(obs)
        assert isinstance(a, np.ndarray)

    def test_shape_9(self):
        a = env_action_space_sample(np.zeros(45))
        assert a.shape == (9,)

    def test_values_in_range(self):
        for _ in range(20):
            a = env_action_space_sample(np.zeros(45))
            assert np.all(a >= -1.0)
            assert np.all(a <=  1.0)

    def test_dtype_float32(self):
        a = env_action_space_sample(np.zeros(45))
        assert a.dtype == np.float32


# ── SCENARIOS constant ────────────────────────────────────────────────────────

class TestScenariosConstant:
    def test_three_scenarios(self):
        assert len(SCENARIOS) == 3

    def test_known_in_scenarios(self):
        assert "known" in SCENARIOS

    def test_novel_in_scenarios(self):
        assert "novel" in SCENARIOS

    def test_occluded_in_scenarios(self):
        assert "occluded" in SCENARIOS


# ── GraspPolicyEvaluator — _aggregate ─────────────────────────────────────────

class TestAggregate:
    def _outcome(self, success: bool, n_steps: int = 5, contact: bool = True,
                 max_z: float = 0.1):
        from env.grasp_outcome import GraspOutcome
        result = GraspResult.SUCCESS if success else GraspResult.MISS
        return GraspOutcome(
            result       = result,
            reason       = "test",
            contact_made = contact,
            max_lift_z   = max_z,
            n_steps      = n_steps,
        )

    def test_success_rate_correct(self):
        outcomes = [self._outcome(True)] * 3 + [self._outcome(False)] * 7
        sr = GraspPolicyEvaluator._aggregate("known", outcomes, 1.0)
        assert sr.success_rate == pytest.approx(0.3)

    def test_n_episodes(self):
        outcomes = [self._outcome(True)] * 5
        sr = GraspPolicyEvaluator._aggregate("known", outcomes, 1.0)
        assert sr.n_episodes == 5

    def test_mean_steps(self):
        outcomes = [self._outcome(True, n_steps=10), self._outcome(False, n_steps=6)]
        sr = GraspPolicyEvaluator._aggregate("known", outcomes, 1.0)
        assert sr.mean_steps == pytest.approx(8.0)

    def test_contact_rate(self):
        outcomes = [
            self._outcome(False, contact=True),
            self._outcome(False, contact=True),
            self._outcome(False, contact=False),
            self._outcome(False, contact=False),
        ]
        sr = GraspPolicyEvaluator._aggregate("known", outcomes, 1.0)
        assert sr.contact_rate == pytest.approx(0.5)

    def test_max_lift_z_mean(self):
        outcomes = [
            self._outcome(True, max_z=0.1),
            self._outcome(True, max_z=0.3),
        ]
        sr = GraspPolicyEvaluator._aggregate("known", outcomes, 1.0)
        assert sr.max_lift_z_mean == pytest.approx(0.2)

    def test_failure_counts_populated(self):
        from env.grasp_outcome import GraspOutcome
        o1 = GraspOutcome(GraspResult.MISS,  "test", n_steps=5)
        o2 = GraspOutcome(GraspResult.MISS,  "test", n_steps=5)
        o3 = GraspOutcome(GraspResult.SUCCESS,"test", n_steps=5)
        sr = GraspPolicyEvaluator._aggregate("known", [o1, o2, o3], 1.0)
        assert sr.failure_counts.get("miss", 0) == 2
        assert sr.failure_counts.get("success", 0) == 0


# ── GraspPolicyEvaluator — evaluate ──────────────────────────────────────────

class TestEvaluate:
    def test_returns_report(self):
        ev     = _evaluator(n_episodes=2)
        report = ev.evaluate(["known"])
        assert isinstance(report, GraspEvalReport)

    def test_report_has_requested_scenarios(self):
        ev     = _evaluator(n_episodes=2)
        report = ev.evaluate(["known", "novel"])
        names  = [s.scenario for s in report.scenarios]
        assert "known" in names
        assert "novel" in names

    def test_n_episodes_correct(self):
        ev     = _evaluator(n_episodes=3)
        report = ev.evaluate(["known"])
        assert report.scenarios[0].n_episodes == 3

    def test_gen_drop_computed_when_both_scenarios(self):
        ev     = _evaluator(n_episodes=2)
        report = ev.evaluate(["known", "novel"])
        assert report.generalisation_drop is not None

    def test_gen_drop_none_with_only_known(self):
        ev     = _evaluator(n_episodes=2)
        report = ev.evaluate(["known"])
        assert report.generalisation_drop is None

    def test_model_path_random_when_no_model(self):
        ev     = _evaluator(n_episodes=2)
        report = ev.evaluate(["known"])
        assert "random" in report.model_path.lower()

    def test_total_wall_seconds_positive(self):
        ev     = _evaluator(n_episodes=2)
        report = ev.evaluate(["known"])
        assert report.total_wall_seconds > 0

    def test_to_dict_serialisable(self):
        import json
        ev     = _evaluator(n_episodes=2)
        report = ev.evaluate(["known"])
        d      = report.to_dict()
        # should not raise
        json.dumps(d)
