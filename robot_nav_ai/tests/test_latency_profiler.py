"""
tests/test_latency_profiler.py — Unit tests for PerceptionProfiler, ProfilerConfig,
StageResult, and LatencyReport.

No real YOLO/SAM inference is run — MagicMock callables are used to control
call counts and timing.  A few tests use a minimal real sleep to verify that
elapsed times are measured correctly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from perception.latency_profiler import (
    LatencyReport, PerceptionProfiler, ProfilerConfig, StageResult,
)
from perception.confidence import AggregatorConfig, SceneAggregator, SceneConfidence
from perception.uncertainty_gate import GateConfig, UncertaintyGate
from perception.depth_projector import DepthProjector, ProjectionResult
from perception.detector import Detection
from perception.rgbd_camera import RGBDFrame, _build_K


# ── helpers ───────────────────────────────────────────────────────────────────

def _stage(name="test", mean=5.0, mn=4.0, mx=6.0, n=10, budget=50.0):
    return StageResult(name=name, mean_ms=mean, min_ms=mn, max_ms=mx,
                       n_runs=n, budget_ms=budget)


def _report(stages=None, total=5.0, budget=50.0):
    return LatencyReport(stages=stages or [_stage()], total_ms=total,
                         budget_ms=budget)


def _make_det(conf=0.9):
    return Detection(
        class_id=0, class_name="mug", confidence=conf,
        bbox_xyxy=np.array([10, 10, 60, 50], dtype=np.float32),
        bbox_xywh=np.array([35, 30, 50, 40], dtype=np.float32),
    )


def _make_frame(z=2.0, H=60, W=80):
    K     = _build_K(60.0, W, H)
    depth = np.full((H, W), z, dtype=np.float32)
    rgb   = np.zeros((H, W, 3), dtype=np.uint8)
    return RGBDFrame(rgb=rgb, depth=depth, K=K, step=0)


# ── ProfilerConfig ────────────────────────────────────────────────────────────

class TestProfilerConfig:
    def test_defaults(self):
        cfg = ProfilerConfig()
        assert cfg.budget_ms == pytest.approx(50.0)
        assert cfg.n_warmup  == 2
        assert cfg.n_runs    == 10

    def test_frozen(self):
        with pytest.raises(Exception):
            ProfilerConfig().budget_ms = 30.0

    def test_custom(self):
        cfg = ProfilerConfig(budget_ms=30.0, n_warmup=5, n_runs=20)
        assert cfg.budget_ms == pytest.approx(30.0)
        assert cfg.n_warmup  == 5
        assert cfg.n_runs    == 20


# ── StageResult ───────────────────────────────────────────────────────────────

class TestStageResult:
    def test_passed_when_fast(self):
        s = _stage(mean=1.0, budget=50.0)
        assert s.passed is True

    def test_failed_when_slow(self):
        s = _stage(mean=60.0, budget=50.0)
        assert s.passed is False

    def test_not_passed_at_exact_budget(self):
        s = _stage(mean=50.0, budget=50.0)
        assert s.passed is False   # requires strictly < budget

    def test_repr_contains_name(self):
        assert "mystagename" in repr(_stage(name="mystagename"))

    def test_repr_contains_mean(self):
        assert "5.00" in repr(_stage(mean=5.0))

    def test_repr_contains_pass(self):
        assert "PASS" in repr(_stage(mean=1.0, budget=50.0))

    def test_repr_contains_fail(self):
        assert "FAIL" in repr(_stage(mean=60.0, budget=50.0))

    def test_n_runs_stored(self):
        s = _stage(n=7)
        assert s.n_runs == 7

    def test_min_le_mean_le_max(self):
        s = _stage(mean=5.0, mn=4.0, mx=6.0)
        assert s.min_ms <= s.mean_ms <= s.max_ms


# ── LatencyReport ─────────────────────────────────────────────────────────────

class TestLatencyReport:
    def test_passed_under_budget(self):
        r = _report(total=30.0, budget=50.0)
        assert r.passed is True

    def test_failed_over_budget(self):
        r = _report(total=60.0, budget=50.0)
        assert r.passed is False

    def test_failed_at_exact_budget(self):
        r = _report(total=50.0, budget=50.0)
        assert r.passed is False

    def test_slowest_is_max_mean_stage(self):
        r = LatencyReport(
            stages=[_stage("fast", mean=1.0), _stage("slow", mean=20.0)],
            total_ms=21.0, budget_ms=50.0,
        )
        assert r.slowest == "slow"

    def test_slowest_none_when_no_stages(self):
        r = LatencyReport(stages=[], total_ms=0.0, budget_ms=50.0)
        assert r.slowest is None

    def test_total_ms_stored(self):
        r = _report(total=12.5)
        assert r.total_ms == pytest.approx(12.5)

    def test_to_dict_has_stages(self):
        d = _report().to_dict()
        assert "stages" in d
        assert isinstance(d["stages"], list)

    def test_to_dict_has_total(self):
        assert "total_ms" in _report().to_dict()

    def test_to_dict_has_passed(self):
        d = _report(total=10.0, budget=50.0).to_dict()
        assert d["passed"] is True

    def test_to_dict_has_slowest(self):
        assert "slowest" in _report().to_dict()

    def test_to_dict_stage_fields(self):
        d = _report().to_dict()
        s = d["stages"][0]
        for key in ("name", "mean_ms", "min_ms", "max_ms", "n_runs", "passed"):
            assert key in s

    def test_str_contains_total(self):
        s = str(_report(total=7.5))
        assert "7.5" in s or "7.50" in s

    def test_str_contains_stage_name(self):
        r = LatencyReport(stages=[_stage("my_stage")], total_ms=5.0, budget_ms=50.0)
        assert "my_stage" in str(r)

    def test_str_contains_pass_or_fail(self):
        s = str(_report(total=1.0, budget=50.0))
        assert "PASS" in s or "FAIL" in s

    def test_str_is_multiline(self):
        assert "\n" in str(_report())


# ── PerceptionProfiler.time_stage ─────────────────────────────────────────────

class TestTimeStage:
    def test_calls_fn_warmup_plus_n_runs(self):
        fn = MagicMock()
        prof = PerceptionProfiler(ProfilerConfig(n_warmup=3, n_runs=5))
        prof.time_stage("t", fn)
        assert fn.call_count == 3 + 5

    def test_calls_fn_with_args(self):
        fn = MagicMock()
        prof = PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=1))
        prof.time_stage("t", fn, 1, 2, key="val")
        fn.assert_called_with(1, 2, key="val")

    def test_returns_stage_result(self):
        fn = MagicMock()
        r  = PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=1)).time_stage("t", fn)
        assert isinstance(r, StageResult)

    def test_name_stored(self):
        fn = MagicMock()
        r  = PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=1)).time_stage("mysub", fn)
        assert r.name == "mysub"

    def test_n_runs_stored(self):
        fn = MagicMock()
        r  = PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=7)).time_stage("t", fn)
        assert r.n_runs == 7

    def test_elapsed_nonnegative(self):
        r = PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=3)).time_stage(
            "t", lambda: None)
        assert r.mean_ms >= 0.0

    def test_min_le_max(self):
        r = PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=5)).time_stage(
            "t", lambda: None)
        assert r.min_ms <= r.max_ms

    def test_custom_budget_used(self):
        fn = MagicMock()
        r  = PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=1)).time_stage(
            "t", fn, budget_ms=25.0)
        assert r.budget_ms == pytest.approx(25.0)

    def test_custom_n_runs_overrides_config(self):
        fn = MagicMock()
        PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=10)).time_stage(
            "t", fn, n_runs=3)
        assert fn.call_count == 3

    def test_custom_warmup_overrides_config(self):
        fn = MagicMock()
        PerceptionProfiler(ProfilerConfig(n_warmup=5, n_runs=1)).time_stage(
            "t", fn, n_warmup=0)
        assert fn.call_count == 1   # only the 1 timed run, 0 warmup

    def test_measurable_sleep(self):
        # 1 ms sleep → mean_ms should be ≥ 0.5 ms
        r = PerceptionProfiler(ProfilerConfig(n_warmup=1, n_runs=3)).time_stage(
            "sleep", lambda: time.sleep(0.001))
        assert r.mean_ms >= 0.5


# ── PerceptionProfiler.profile_stages ─────────────────────────────────────────

class TestProfileStages:
    def test_empty_stages_total_zero(self):
        r = PerceptionProfiler().profile_stages([])
        assert r.total_ms == pytest.approx(0.0)

    def test_empty_stages_passes(self):
        r = PerceptionProfiler().profile_stages([])
        assert r.passed is True

    def test_single_stage_total_equals_mean(self):
        fn = MagicMock()
        r  = PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=1)).profile_stages(
            [("s", fn)])
        assert r.total_ms == pytest.approx(r.stages[0].mean_ms)

    def test_multiple_stages_total_is_sum(self):
        # Use zero-sleep lambdas; total ≈ sum of all means (all ~0 ms)
        prof = PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=2))
        r    = prof.profile_stages([("a", lambda: None), ("b", lambda: None)])
        assert r.total_ms == pytest.approx(r.stages[0].mean_ms + r.stages[1].mean_ms,
                                           abs=0.1)

    def test_returns_latency_report(self):
        r = PerceptionProfiler().profile_stages([])
        assert isinstance(r, LatencyReport)

    def test_stage_count_matches_input(self):
        fn = MagicMock()
        r  = PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=1)).profile_stages(
            [("a", fn), ("b", fn), ("c", fn)])
        assert len(r.stages) == 3

    def test_accepts_args_tuple(self):
        fn = MagicMock()
        PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=1)).profile_stages(
            [("t", fn, (42,))])
        fn.assert_called_with(42)

    def test_accepts_kwargs_dict(self):
        fn = MagicMock()
        PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=1)).profile_stages(
            [("t", fn, (), {"k": "v"})])
        fn.assert_called_with(k="v")

    def test_budget_from_config(self):
        fn = MagicMock()
        r  = PerceptionProfiler(ProfilerConfig(budget_ms=30.0, n_warmup=0, n_runs=1)
                                 ).profile_stages([("t", fn)])
        assert r.budget_ms == pytest.approx(30.0)


# ── convenience methods ───────────────────────────────────────────────────────

class TestConvenienceMethods:
    def _prof(self):
        return PerceptionProfiler(ProfilerConfig(n_warmup=0, n_runs=2))

    def test_profile_aggregator_returns_report(self):
        agg   = SceneAggregator()
        dets  = [_make_det()]
        r     = self._prof().profile_aggregator(agg, dets)
        assert isinstance(r, LatencyReport)

    def test_profile_aggregator_stage_named_aggregator(self):
        agg   = SceneAggregator()
        r     = self._prof().profile_aggregator(agg, [_make_det()])
        assert r.stages[0].name == "aggregator"

    def test_profile_gate_returns_report(self):
        gate  = UncertaintyGate(GateConfig(require_depth=False))
        scene = SceneConfidence(objects=[], global_score=0.0, n_objects=0)
        r     = self._prof().profile_gate(gate, scene)
        assert isinstance(r, LatencyReport)

    def test_profile_gate_stage_named_gate(self):
        gate  = UncertaintyGate(GateConfig(require_depth=False))
        scene = SceneConfidence(objects=[], global_score=0.0, n_objects=0)
        r     = self._prof().profile_gate(gate, scene)
        assert r.stages[0].name == "gate"

    def test_profile_projector_returns_report(self):
        proj  = DepthProjector()
        frame = _make_frame()
        r     = self._prof().profile_projector(proj, [_make_det()], frame)
        assert isinstance(r, LatencyReport)

    def test_profile_projector_stage_named(self):
        proj  = DepthProjector()
        frame = _make_frame()
        r     = self._prof().profile_projector(proj, [_make_det()], frame)
        assert "projector" in r.stages[0].name

    def test_aggregator_fast_enough_for_budget(self):
        agg   = SceneAggregator()
        dets  = [_make_det() for _ in range(10)]
        r     = PerceptionProfiler(ProfilerConfig(budget_ms=50.0, n_warmup=2, n_runs=20)
                                   ).profile_aggregator(agg, dets)
        assert r.passed, f"aggregator too slow: {r.total_ms:.2f}ms"

    def test_gate_fast_enough_for_budget(self):
        gate  = UncertaintyGate(GateConfig(require_depth=False))
        scene = SceneConfidence(objects=[], global_score=0.0, n_objects=0)
        r     = PerceptionProfiler(ProfilerConfig(budget_ms=50.0, n_warmup=2, n_runs=20)
                                   ).profile_gate(gate, scene)
        assert r.passed, f"gate too slow: {r.total_ms:.2f}ms"

    def test_projector_fast_enough_for_budget(self):
        proj  = DepthProjector()
        frame = _make_frame(z=2.0, H=480, W=640)
        dets  = [_make_det()]
        r     = PerceptionProfiler(ProfilerConfig(budget_ms=50.0, n_warmup=2, n_runs=20)
                                   ).profile_projector(proj, dets, frame)
        assert r.passed, f"projector too slow: {r.total_ms:.2f}ms"


# ── optimization_hints ────────────────────────────────────────────────────────

class TestOptimizationHints:
    def _prof(self):
        return PerceptionProfiler()

    def _slow_stage(self, name):
        return _stage(name=name, mean=60.0, budget=50.0)

    def test_returns_list(self):
        r = _report(total=5.0, budget=50.0)
        assert isinstance(self._prof().optimization_hints(r), list)

    def test_empty_when_all_pass(self):
        r = LatencyReport(
            stages=[_stage("a", mean=1.0, budget=50.0)],
            total_ms=1.0, budget_ms=50.0,
        )
        assert self._prof().optimization_hints(r) == []

    def test_hint_for_slow_detector(self):
        r = LatencyReport(stages=[self._slow_stage("yolo_detect")],
                          total_ms=60.0, budget_ms=50.0)
        hints = self._prof().optimization_hints(r)
        assert len(hints) > 0
        assert any("yolo" in h.lower() or "detect" in h.lower() or "imgsz" in h.lower()
                   for h in hints)

    def test_hint_for_slow_sam(self):
        r = LatencyReport(stages=[self._slow_stage("sam_segmentor")],
                          total_ms=60.0, budget_ms=50.0)
        hints = self._prof().optimization_hints(r)
        assert any("sam" in h.lower() or "vit" in h.lower() for h in hints)

    def test_hint_for_slow_projector(self):
        r = LatencyReport(stages=[self._slow_stage("depth_projector")],
                          total_ms=60.0, budget_ms=50.0)
        hints = self._prof().optimization_hints(r)
        assert len(hints) > 0

    def test_fallback_hint_when_no_named_stage(self):
        r = LatencyReport(stages=[self._slow_stage("custom_stage")],
                          total_ms=60.0, budget_ms=50.0)
        hints = self._prof().optimization_hints(r)
        assert len(hints) > 0

    def test_repr(self):
        r = repr(PerceptionProfiler())
        assert "50" in r or "budget" in r.lower()
