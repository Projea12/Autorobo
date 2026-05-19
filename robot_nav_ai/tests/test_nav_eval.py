"""
tests/test_nav_eval.py — Unit + integration tests for NavigationEvaluator.

All tests use a randomly-initialised policy (no training) to stay fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.nav_obs import NAV_OBS_DIM
from agent.ppo import ActorCritic, PPOAgent, PPOConfig, make_ppo_agent
from agent.nav_eval import (
    EpisodeResult, EvalReport, NavigationEvaluator, _aggregate,
)


OBS_DIM = NAV_OBS_DIM
ACT_DIM = 2


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def evaluator():
    net = ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    return NavigationEvaluator(agent=net, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def report(evaluator):
    return evaluator.evaluate(n_episodes=10, seed=0, max_steps=30)


# ── EpisodeResult ─────────────────────────────────────────────────────────────

class TestEpisodeResult:
    def test_repr(self):
        r = EpisodeResult(success=True, n_steps=50, total_return=8.5,
                          failure_mode="success")
        assert "50" in repr(r)
        assert "success" in repr(r)

    def test_failure_mode_string(self):
        r = EpisodeResult(success=False, n_steps=300, total_return=-3.0,
                          failure_mode="timeout")
        assert r.failure_mode == "timeout"


# ── EvalReport ────────────────────────────────────────────────────────────────

class TestEvalReport:
    def _make_results(self):
        return [
            EpisodeResult(True,  50, 8.0,  "success"),
            EpisodeResult(False, 300, -2.0, "timeout"),
            EpisodeResult(False, 10,  -5.0, "collision"),
            EpisodeResult(False, 200, -1.5, "retreating"),
        ]

    def test_aggregate_success_rate(self):
        r = _aggregate(self._make_results(), "test")
        assert r.success_rate == pytest.approx(0.25)

    def test_aggregate_n_episodes(self):
        r = _aggregate(self._make_results(), "test")
        assert r.n_episodes == 4

    def test_aggregate_mean_return(self):
        r = _aggregate(self._make_results(), "test")
        expected = (8.0 - 2.0 - 5.0 - 1.5) / 4
        assert r.mean_return == pytest.approx(expected, rel=1e-4)

    def test_aggregate_mean_steps_success_only(self):
        r = _aggregate(self._make_results(), "test")
        assert r.mean_steps == pytest.approx(50.0)

    def test_aggregate_mean_steps_all(self):
        r = _aggregate(self._make_results(), "test")
        assert r.mean_steps_all == pytest.approx((50 + 300 + 10 + 200) / 4)

    def test_aggregate_failure_timeout(self):
        r = _aggregate(self._make_results(), "test")
        assert r.failure_timeout == pytest.approx(0.25)

    def test_aggregate_failure_collision(self):
        r = _aggregate(self._make_results(), "test")
        assert r.failure_collision == pytest.approx(0.25)

    def test_aggregate_failure_retreating(self):
        r = _aggregate(self._make_results(), "test")
        assert r.failure_retreating == pytest.approx(0.25)

    def test_aggregate_empty(self):
        r = _aggregate([], "empty")
        assert r.n_episodes == 0
        assert r.success_rate == 0.0

    def test_str_contains_label(self):
        r = _aggregate(self._make_results(), "my_label")
        assert "my_label" in str(r)

    def test_str_contains_success_rate(self):
        r = _aggregate(self._make_results(), "test")
        s = str(r)
        assert "success_rate" in s or "25" in s

    def test_to_dict_keys(self):
        r = _aggregate(self._make_results(), "test")
        d = r.to_dict()
        for key in ("success_rate", "mean_steps", "mean_return",
                    "failure_timeout", "failure_collision", "failure_retreating"):
            assert key in d

    def test_to_dict_values_match(self):
        r = _aggregate(self._make_results(), "test")
        d = r.to_dict()
        assert d["success_rate"] == pytest.approx(r.success_rate)
        assert d["mean_return"]   == pytest.approx(r.mean_return)

    def test_all_failure_fractions_sum_le_one(self):
        r = _aggregate(self._make_results(), "test")
        total = r.failure_timeout + r.failure_collision + r.failure_retreating
        assert total <= 1.0 + 1e-9

    def test_label_stored(self):
        r = _aggregate(self._make_results(), "xyz")
        assert r.label == "xyz"


# ── NavigationEvaluator construction ─────────────────────────────────────────

class TestEvaluatorConstruction:
    def test_from_net(self):
        net = ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        ev  = NavigationEvaluator(agent=net)
        assert ev is not None

    def test_from_ppo_agent(self):
        agent = make_ppo_agent(OBS_DIM, ACT_DIM)
        ev    = NavigationEvaluator(agent=agent)
        assert ev is not None

    def test_from_checkpoint(self, tmp_path):
        agent = make_ppo_agent(OBS_DIM, ACT_DIM)
        path  = tmp_path / "ckpt.pt"
        torch.save({"agent": agent.state_dict()}, path)
        ev = NavigationEvaluator.from_checkpoint(
            path, obs_dim=OBS_DIM, act_dim=ACT_DIM
        )
        assert isinstance(ev, NavigationEvaluator)


# ── evaluate() — smoke tests ──────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_eval_report(self, report):
        assert isinstance(report, EvalReport)

    def test_n_episodes_matches_request(self, report):
        assert report.n_episodes == 10

    def test_success_rate_in_0_1(self, report):
        assert 0.0 <= report.success_rate <= 1.0

    def test_mean_steps_positive(self, report):
        assert report.mean_steps_all > 0.0

    def test_episodes_list_length(self, report):
        assert len(report.episodes) == 10

    def test_all_episodes_are_results(self, report):
        for ep in report.episodes:
            assert isinstance(ep, EpisodeResult)

    def test_failure_fractions_sum_le_n(self, report):
        n = report.n_episodes
        total_failures = (report.failure_timeout + report.failure_collision
                          + report.failure_retreating) * n
        assert total_failures <= n + 1e-6

    def test_different_seeds_can_differ(self, evaluator):
        r1 = evaluator.evaluate(n_episodes=5, seed=0,    max_steps=20)
        r2 = evaluator.evaluate(n_episodes=5, seed=9999, max_steps=20)
        # Results need not differ, but we check the call doesn't crash
        assert r1.n_episodes == r2.n_episodes == 5

    def test_label_propagated(self, evaluator):
        r = evaluator.evaluate(n_episodes=3, seed=1, max_steps=20, label="my_label")
        assert r.label == "my_label"


# ── evaluate_novel_layout() ───────────────────────────────────────────────────

class TestNovelLayout:
    def test_returns_eval_report(self, evaluator):
        r = evaluator.evaluate_novel_layout(n_episodes=5, seed=0, max_steps=20)
        assert isinstance(r, EvalReport)

    def test_n_episodes(self, evaluator):
        r = evaluator.evaluate_novel_layout(n_episodes=5, seed=0, max_steps=20)
        assert r.n_episodes == 5

    def test_label_novel(self, evaluator):
        r = evaluator.evaluate_novel_layout(n_episodes=3, seed=0, max_steps=20)
        assert "novel" in r.label
