"""
tests/test_grasp_latency.py — Unit tests for GraspLatencyProfiler and GraspLatencyConfig.

No real planner/policy/controller needed — MagicMock callables control timing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from benchmarks.grasp_latency import (
    BUDGET_CONTROLLER_MS,
    BUDGET_PERCEPTION_MS,
    BUDGET_PLANNER_MS,
    BUDGET_POLICY_MS,
    BUDGET_TOTAL_MS,
    GraspLatencyConfig,
    GraspLatencyProfiler,
)
from perception.latency_profiler import LatencyReport, StageResult


# ── GraspLatencyConfig ────────────────────────────────────────────────────────

class TestGraspLatencyConfig:
    def test_defaults(self):
        cfg = GraspLatencyConfig()
        assert cfg.budget_ms == pytest.approx(BUDGET_TOTAL_MS)
        assert cfg.n_warmup  == 3
        assert cfg.n_runs    == 20

    def test_frozen(self):
        with pytest.raises(Exception):
            GraspLatencyConfig().n_runs = 5

    def test_custom(self):
        cfg = GraspLatencyConfig(budget_ms=50.0, n_warmup=1, n_runs=5)
        assert cfg.budget_ms == pytest.approx(50.0)
        assert cfg.n_warmup  == 1
        assert cfg.n_runs    == 5


# ── Budget constants ──────────────────────────────────────────────────────────

class TestBudgetConstants:
    def test_total_budget(self):
        assert BUDGET_TOTAL_MS == pytest.approx(100.0)

    def test_perception_budget(self):
        assert BUDGET_PERCEPTION_MS == pytest.approx(50.0)

    def test_planner_budget(self):
        assert BUDGET_PLANNER_MS == pytest.approx(10.0)

    def test_policy_budget(self):
        assert BUDGET_POLICY_MS == pytest.approx(5.0)

    def test_controller_budget(self):
        assert BUDGET_CONTROLLER_MS == pytest.approx(5.0)

    def test_component_budgets_fit_in_total(self):
        component_sum = BUDGET_PLANNER_MS + BUDGET_POLICY_MS + BUDGET_CONTROLLER_MS
        assert component_sum < BUDGET_TOTAL_MS


# ── helpers ───────────────────────────────────────────────────────────────────

def _fast_profiler() -> GraspLatencyProfiler:
    cfg = GraspLatencyConfig(n_warmup=0, n_runs=3, budget_ms=BUDGET_TOTAL_MS)
    return GraspLatencyProfiler(cfg=cfg)


def _noop(*args, **kwargs):
    return None


# ── GraspLatencyProfiler construction ────────────────────────────────────────

class TestProfilerConstruction:
    def test_default_construction(self):
        prof = GraspLatencyProfiler()
        assert prof.cfg.budget_ms == pytest.approx(BUDGET_TOTAL_MS)

    def test_custom_cfg(self):
        cfg  = GraspLatencyConfig(n_runs=5)
        prof = GraspLatencyProfiler(cfg=cfg)
        assert prof.cfg.n_runs == 5

    def test_repr(self):
        prof = GraspLatencyProfiler()
        r = repr(prof)
        assert "GraspLatencyProfiler" in r
        assert "budget=" in r


# ── profile_policy ────────────────────────────────────────────────────────────

class TestProfilePolicy:
    def test_returns_stage_result(self):
        prof   = _fast_profiler()
        obs    = np.zeros(31, dtype=np.float32)
        result = prof.profile_policy(_noop, obs)
        assert isinstance(result, StageResult)

    def test_stage_name_contains_policy(self):
        prof   = _fast_profiler()
        result = prof.profile_policy(_noop, np.zeros(10))
        assert "policy" in result.name.lower() or "sac" in result.name.lower()

    def test_budget_is_policy_budget(self):
        prof   = _fast_profiler()
        result = prof.profile_policy(_noop, np.zeros(10))
        assert result.budget_ms == pytest.approx(BUDGET_POLICY_MS)

    def test_mean_ms_positive(self):
        prof   = _fast_profiler()
        result = prof.profile_policy(_noop, np.zeros(10))
        assert result.mean_ms >= 0.0

    def test_noop_passes_budget(self):
        prof   = _fast_profiler()
        result = prof.profile_policy(_noop, np.zeros(10))
        assert result.passed


# ── profile_planner ───────────────────────────────────────────────────────────

class TestProfilePlanner:
    def _planner(self):
        m = MagicMock()
        m.plan.return_value = []
        return m

    def test_returns_stage_result(self):
        prof    = _fast_profiler()
        planner = self._planner()
        result  = prof.profile_planner(
            planner,
            obj_pos    = np.zeros(3),
            robot_pos  = np.zeros(3),
            robot_quat = np.array([1.0, 0, 0, 0]),
        )
        assert isinstance(result, StageResult)

    def test_planner_called(self):
        prof    = _fast_profiler()
        planner = self._planner()
        prof.profile_planner(
            planner,
            obj_pos    = np.zeros(3),
            robot_pos  = np.zeros(3),
            robot_quat = np.array([1.0, 0, 0, 0]),
        )
        assert planner.plan.call_count >= 1

    def test_budget_is_planner_budget(self):
        prof   = _fast_profiler()
        result = prof.profile_planner(
            self._planner(), np.zeros(3), np.zeros(3), np.array([1, 0, 0, 0])
        )
        assert result.budget_ms == pytest.approx(BUDGET_PLANNER_MS)

    def test_passes_point_cloud_kwarg(self):
        prof    = _fast_profiler()
        planner = self._planner()
        pc      = np.random.randn(10, 3).astype(np.float32)
        prof.profile_planner(
            planner, np.zeros(3), np.zeros(3), np.array([1, 0, 0, 0]),
            point_cloud=pc,
        )
        # plan should have been called with point_cloud= kwarg
        _, kwargs = planner.plan.call_args
        assert "point_cloud" in kwargs


# ── profile_controller ────────────────────────────────────────────────────────

class TestProfileController:
    def _controller(self):
        m = MagicMock()
        m.step.return_value = MagicMock()
        return m

    def test_returns_stage_result(self):
        prof   = _fast_profiler()
        ctrl   = self._controller()
        action = MagicMock()
        result = prof.profile_controller(ctrl, action)
        assert isinstance(result, StageResult)

    def test_controller_called(self):
        prof   = _fast_profiler()
        ctrl   = self._controller()
        prof.profile_controller(ctrl, MagicMock())
        assert ctrl.step.call_count >= 1

    def test_budget_is_controller_budget(self):
        prof   = _fast_profiler()
        result = prof.profile_controller(self._controller(), MagicMock())
        assert result.budget_ms == pytest.approx(BUDGET_CONTROLLER_MS)


# ── profile_all ───────────────────────────────────────────────────────────────

class TestProfileAll:
    def _setup(self):
        planner = MagicMock()
        planner.plan.return_value = []
        controller = MagicMock()
        controller.step.return_value = MagicMock()
        obs    = np.zeros(31, dtype=np.float32)
        action = MagicMock()
        return planner, controller, obs, action

    def test_returns_latency_report(self):
        prof = _fast_profiler()
        planner, ctrl, obs, action = self._setup()
        report = prof.profile_all(
            planner     = planner,
            predict_fn  = _noop,
            controller  = ctrl,
            action      = action,
            obj_pos     = np.zeros(3),
            robot_pos   = np.zeros(3),
            robot_quat  = np.array([1.0, 0, 0, 0]),
            obs         = obs,
        )
        assert isinstance(report, LatencyReport)

    def test_report_has_three_stages(self):
        prof = _fast_profiler()
        planner, ctrl, obs, action = self._setup()
        report = prof.profile_all(
            planner, _noop, ctrl, action,
            np.zeros(3), np.zeros(3), np.array([1, 0, 0, 0]), obs,
        )
        assert len(report.stages) == 3

    def test_total_ms_is_sum_of_stages(self):
        prof = _fast_profiler()
        planner, ctrl, obs, action = self._setup()
        report = prof.profile_all(
            planner, _noop, ctrl, action,
            np.zeros(3), np.zeros(3), np.array([1, 0, 0, 0]), obs,
        )
        expected = sum(s.mean_ms for s in report.stages)
        assert report.total_ms == pytest.approx(expected)


# ── run_standalone ────────────────────────────────────────────────────────────

class TestRunStandalone:
    def test_returns_latency_report(self):
        prof    = _fast_profiler()
        planner = MagicMock()
        planner.plan.return_value = []
        report  = prof.run_standalone(
            planner,
            obj_pos    = np.zeros(3),
            robot_pos  = np.zeros(3),
            robot_quat = np.array([1.0, 0, 0, 0]),
        )
        assert isinstance(report, LatencyReport)

    def test_standalone_has_one_stage(self):
        prof    = _fast_profiler()
        planner = MagicMock()
        planner.plan.return_value = []
        report  = prof.run_standalone(
            planner, np.zeros(3), np.zeros(3), np.array([1, 0, 0, 0])
        )
        assert len(report.stages) == 1


# ── hints ─────────────────────────────────────────────────────────────────────

class TestHints:
    def _stage(self, name, mean_ms, budget_ms):
        return StageResult(
            name=name, mean_ms=mean_ms, min_ms=mean_ms * 0.9,
            max_ms=mean_ms * 1.1, n_runs=5, budget_ms=budget_ms,
        )

    def test_no_hints_when_all_pass(self):
        prof   = _fast_profiler()
        stages = [
            self._stage("grasp_planner",  5.0,  BUDGET_PLANNER_MS),
            self._stage("sac_policy",     2.0,  BUDGET_POLICY_MS),
            self._stage("arm_controller", 2.0,  BUDGET_CONTROLLER_MS),
        ]
        report = LatencyReport(stages=stages, total_ms=9.0, budget_ms=BUDGET_TOTAL_MS)
        hints  = prof.hints(report)
        assert hints == []

    def test_hint_for_slow_planner(self):
        prof   = _fast_profiler()
        stages = [self._stage("grasp_planner", 50.0, BUDGET_PLANNER_MS)]
        report = LatencyReport(stages=stages, total_ms=50.0, budget_ms=BUDGET_TOTAL_MS)
        hints  = prof.hints(report)
        assert len(hints) > 0
        assert any("planner" in h.lower() or "top_k" in h.lower() or "n_side" in h.lower()
                   for h in hints)

    def test_hint_for_slow_policy(self):
        prof   = _fast_profiler()
        stages = [self._stage("sac_policy", 50.0, BUDGET_POLICY_MS)]
        report = LatencyReport(stages=stages, total_ms=50.0, budget_ms=BUDGET_TOTAL_MS)
        hints  = prof.hints(report)
        assert len(hints) > 0
        assert any("onnx" in h.lower() or "torchscript" in h.lower() or "policy" in h.lower()
                   for h in hints)

    def test_hint_for_slow_controller(self):
        prof   = _fast_profiler()
        stages = [self._stage("arm_controller", 50.0, BUDGET_CONTROLLER_MS)]
        report = LatencyReport(stages=stages, total_ms=50.0, budget_ms=BUDGET_TOTAL_MS)
        hints  = prof.hints(report)
        assert len(hints) > 0
        assert any("dls" in h.lower() or "jacobian" in h.lower() or "controller" in h.lower()
                   for h in hints)

    def test_total_hint_when_no_stage_fails_but_total_does(self):
        prof   = _fast_profiler()
        # each stage passes budget individually, but total exceeds 100ms
        stages = [
            self._stage("grasp_planner",  9.0, BUDGET_PLANNER_MS),
            self._stage("sac_policy",     4.0, BUDGET_POLICY_MS),
            self._stage("arm_controller", 4.0, BUDGET_CONTROLLER_MS),
        ]
        report = LatencyReport(stages=stages, total_ms=120.0, budget_ms=BUDGET_TOTAL_MS)
        hints  = prof.hints(report)
        assert len(hints) > 0
