"""
tests/test_conservative_policy_latency.py

Tests for:
  planning/conservative_policy.py  — ConservativePolicy
  planning/decision_latency.py     — DecisionLatencyProfiler
"""

from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from planning.decision_gate import (
    Decision, DecisionGate, DecisionGateConfig, DecisionResult,
)
from planning.uncertainty_pipeline import UncertaintyPipeline, LayeredConfidence
from planning.conservative_policy import (
    ConservativeActionType, ConservativePolicyConfig, ConservativePolicy,
    ConservativeAction,
)
from planning.decision_latency import (
    LatencyBudgetConfig, PhaseLatency, LatencyReport,
    DecisionLatencyProfiler,
)
from planning.nav_confidence   import NavSignals
from planning.grasp_confidence import GraspSignals


# ── helpers ───────────────────────────────────────────────────────────────────

def _layered(propagated: float, bottleneck: str = "navigation",
             bn_score: float = 0.5) -> LayeredConfidence:
    """Build a minimal LayeredConfidence for gate input."""
    return LayeredConfidence(
        perception_score = 0.9,
        nav_score        = 0.9,
        grasp_score      = 0.9,
        propagated       = propagated,
        eff_perception   = 0.9,
        eff_nav          = 0.9,
        eff_grasp        = 0.9,
        bottleneck_layer = bottleneck,
        bottleneck_score = bn_score,
    )


def _gate_result(
    decision: Decision,
    bottleneck: str = "navigation",
    bn_score: float = 0.5,
    score: float = 0.75,
    safer_action: str = "halt and request human review",
) -> DecisionResult:
    return DecisionResult(
        decision         = decision,
        score            = score,
        reason           = "test",
        safer_action     = safer_action,
        bottleneck_layer = bottleneck,
        bottleneck_score = bn_score,
    )


def _policy() -> ConservativePolicy:
    return ConservativePolicy()


def _ideal_nav() -> NavSignals:
    return NavSignals(
        min_clearance_m=1.5, path_length_m=3.0, n_waypoints=8,
        localisation_std_m=0.02, goal_distance_m=2.0,
    )


def _ideal_grasp() -> GraspSignals:
    return GraspSignals(
        best_candidate_score=0.9, n_candidates=5,
        depth_std_m=0.01, reachability=0.95, n_cloud_points=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ConservativePolicyConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestConservativePolicyConfig:

    def test_defaults(self):
        cfg = ConservativePolicyConfig()
        assert cfg.slow_velocity_scale       == pytest.approx(0.30)
        assert cfg.reposition_velocity_scale == pytest.approx(0.15)
        assert cfg.rescan_extra_frames       == 3
        assert cfg.halt_requires_human       is True

    def test_frozen(self):
        cfg = ConservativePolicyConfig()
        with pytest.raises((FrozenInstanceError, TypeError)):
            cfg.slow_velocity_scale = 0.5  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# ACT decision → EXECUTE
# ══════════════════════════════════════════════════════════════════════════════

class TestPolicyActDecision:

    def test_act_gives_execute(self):
        r = _gate_result(Decision.ACT, score=0.95)
        a = _policy().apply(r)
        assert a.action_type == ConservativeActionType.EXECUTE

    def test_execute_velocity_scale_is_one(self):
        r = _gate_result(Decision.ACT, score=0.95)
        a = _policy().apply(r)
        assert a.velocity_scale == pytest.approx(1.0)

    def test_execute_does_not_require_human(self):
        r = _gate_result(Decision.ACT, score=0.95)
        a = _policy().apply(r)
        assert a.requires_human is False

    def test_execute_instruction_non_empty(self):
        r = _gate_result(Decision.ACT, score=0.95)
        a = _policy().apply(r)
        assert len(a.instruction) > 0

    def test_execute_bottleneck_propagated(self):
        r = _gate_result(Decision.ACT, bottleneck="grasp", bn_score=0.88, score=0.95)
        a = _policy().apply(r)
        assert a.bottleneck_layer == "grasp"
        assert a.bottleneck_score == pytest.approx(0.88)

    def test_execute_score_propagated(self):
        r = _gate_result(Decision.ACT, score=0.93)
        a = _policy().apply(r)
        assert a.decision_score == pytest.approx(0.93)


# ══════════════════════════════════════════════════════════════════════════════
# GATHER / perception → RESCAN
# ══════════════════════════════════════════════════════════════════════════════

class TestPolicyGatherPerception:

    def test_perception_bottleneck_gives_rescan(self):
        r = _gate_result(Decision.GATHER, bottleneck="perception")
        a = _policy().apply(r)
        assert a.action_type == ConservativeActionType.RESCAN

    def test_rescan_velocity_scale_is_zero(self):
        r = _gate_result(Decision.GATHER, bottleneck="perception")
        a = _policy().apply(r)
        assert a.velocity_scale == pytest.approx(0.0)

    def test_rescan_does_not_require_human(self):
        r = _gate_result(Decision.GATHER, bottleneck="perception")
        a = _policy().apply(r)
        assert a.requires_human is False

    def test_rescan_extra_frames_from_config(self):
        cfg = ConservativePolicyConfig(rescan_extra_frames=5)
        policy = ConservativePolicy(cfg)
        r = _gate_result(Decision.GATHER, bottleneck="perception")
        a = policy.apply(r)
        assert a.extra_frames == 5

    def test_rescan_default_extra_frames(self):
        r = _gate_result(Decision.GATHER, bottleneck="perception")
        a = _policy().apply(r)
        assert a.extra_frames == 3

    def test_rescan_instruction_mentions_perception(self):
        r = _gate_result(Decision.GATHER, bottleneck="perception")
        a = _policy().apply(r)
        assert "perception" in a.instruction.lower() or "sensor" in a.instruction.lower()


# ══════════════════════════════════════════════════════════════════════════════
# GATHER / navigation → SLOW_DOWN
# ══════════════════════════════════════════════════════════════════════════════

class TestPolicyGatherNavigation:

    def test_nav_bottleneck_gives_slow_down(self):
        r = _gate_result(Decision.GATHER, bottleneck="navigation")
        a = _policy().apply(r)
        assert a.action_type == ConservativeActionType.SLOW_DOWN

    def test_slow_down_velocity_scale_from_config(self):
        cfg = ConservativePolicyConfig(slow_velocity_scale=0.40)
        policy = ConservativePolicy(cfg)
        r = _gate_result(Decision.GATHER, bottleneck="navigation")
        a = policy.apply(r)
        assert a.velocity_scale == pytest.approx(0.40)

    def test_slow_down_default_velocity_scale(self):
        r = _gate_result(Decision.GATHER, bottleneck="navigation")
        a = _policy().apply(r)
        assert a.velocity_scale == pytest.approx(0.30)

    def test_slow_down_does_not_require_human(self):
        r = _gate_result(Decision.GATHER, bottleneck="navigation")
        a = _policy().apply(r)
        assert a.requires_human is False

    def test_slow_down_instruction_mentions_nav(self):
        r = _gate_result(Decision.GATHER, bottleneck="navigation")
        a = _policy().apply(r)
        assert "navigation" in a.instruction.lower() or "localisation" in a.instruction.lower()

    def test_slow_down_velocity_for_helper(self):
        policy = _policy()
        r = _gate_result(Decision.GATHER, bottleneck="navigation")
        v = policy.velocity_for(r, max_velocity=1.0)
        assert v == pytest.approx(0.30)

    def test_velocity_for_act_is_full_speed(self):
        policy = _policy()
        r = _gate_result(Decision.ACT, score=0.95)
        v = policy.velocity_for(r, max_velocity=2.0)
        assert v == pytest.approx(2.0)

    def test_velocity_for_safer_is_zero(self):
        policy = _policy()
        r = _gate_result(Decision.SAFER, score=0.40)
        v = policy.velocity_for(r, max_velocity=1.5)
        assert v == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# GATHER / grasp → REPOSITION
# ══════════════════════════════════════════════════════════════════════════════

class TestPolicyGatherGrasp:

    def test_grasp_bottleneck_gives_reposition(self):
        r = _gate_result(Decision.GATHER, bottleneck="grasp")
        a = _policy().apply(r)
        assert a.action_type == ConservativeActionType.REPOSITION

    def test_reposition_velocity_scale_from_config(self):
        cfg = ConservativePolicyConfig(reposition_velocity_scale=0.20)
        policy = ConservativePolicy(cfg)
        r = _gate_result(Decision.GATHER, bottleneck="grasp")
        a = policy.apply(r)
        assert a.velocity_scale == pytest.approx(0.20)

    def test_reposition_default_velocity_scale(self):
        r = _gate_result(Decision.GATHER, bottleneck="grasp")
        a = _policy().apply(r)
        assert a.velocity_scale == pytest.approx(0.15)

    def test_reposition_does_not_require_human(self):
        r = _gate_result(Decision.GATHER, bottleneck="grasp")
        a = _policy().apply(r)
        assert a.requires_human is False

    def test_reposition_instruction_mentions_arm_or_cloud(self):
        r = _gate_result(Decision.GATHER, bottleneck="grasp")
        a = _policy().apply(r)
        assert any(kw in a.instruction.lower()
                   for kw in ["reposition", "arm", "point", "grasp", "cloud"])


# ══════════════════════════════════════════════════════════════════════════════
# SAFER → HALT
# ══════════════════════════════════════════════════════════════════════════════

class TestPolicySaferDecision:

    def test_safer_gives_halt(self):
        r = _gate_result(Decision.SAFER, score=0.30)
        a = _policy().apply(r)
        assert a.action_type == ConservativeActionType.HALT

    def test_halt_velocity_scale_is_zero(self):
        r = _gate_result(Decision.SAFER, score=0.30)
        a = _policy().apply(r)
        assert a.velocity_scale == pytest.approx(0.0)

    def test_halt_requires_human_by_default(self):
        r = _gate_result(Decision.SAFER, score=0.30)
        a = _policy().apply(r)
        assert a.requires_human is True

    def test_halt_requires_human_can_be_disabled(self):
        cfg = ConservativePolicyConfig(halt_requires_human=False)
        policy = ConservativePolicy(cfg)
        r = _gate_result(Decision.SAFER, score=0.30)
        a = policy.apply(r)
        assert a.requires_human is False

    def test_halt_instruction_matches_safer_action(self):
        safer = "retreat to last known safe position"
        r = _gate_result(Decision.SAFER, score=0.30, safer_action=safer)
        a = _policy().apply(r)
        assert safer in a.instruction

    def test_safer_with_perception_bottleneck_still_halts(self):
        r = _gate_result(Decision.SAFER, bottleneck="perception", score=0.20)
        a = _policy().apply(r)
        assert a.action_type == ConservativeActionType.HALT

    def test_safer_with_grasp_bottleneck_still_halts(self):
        r = _gate_result(Decision.SAFER, bottleneck="grasp", score=0.25)
        a = _policy().apply(r)
        assert a.action_type == ConservativeActionType.HALT


# ══════════════════════════════════════════════════════════════════════════════
# Unknown bottleneck fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestPolicyUnknownBottleneck:

    def test_unknown_gather_bottleneck_falls_back_to_halt(self):
        r = _gate_result(Decision.GATHER, bottleneck="unknown_sensor")
        a = _policy().apply(r)
        assert a.action_type == ConservativeActionType.HALT

    def test_unknown_bottleneck_velocity_zero(self):
        r = _gate_result(Decision.GATHER, bottleneck="unknown_sensor")
        a = _policy().apply(r)
        assert a.velocity_scale == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# ConservativeAction dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestConservativeActionDataclass:

    def test_repr(self):
        r = _gate_result(Decision.ACT, bottleneck="grasp", bn_score=0.91, score=0.95)
        a = _policy().apply(r)
        text = repr(a)
        assert "execute" in text
        assert "grasp" in text

    def test_extra_frames_zero_for_non_rescan(self):
        r = _gate_result(Decision.GATHER, bottleneck="navigation")
        a = _policy().apply(r)
        assert a.extra_frames == 0

    def test_decision_score_preserved(self):
        r = _gate_result(Decision.GATHER, bottleneck="grasp", score=0.72)
        a = _policy().apply(r)
        assert a.decision_score == pytest.approx(0.72)


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: full pipeline → policy
# ══════════════════════════════════════════════════════════════════════════════

class TestPolicyEndToEnd:

    def _pipeline_result(self, p, n, g) -> DecisionResult:
        lc = UncertaintyPipeline().propagate(p, n, g)
        return DecisionGate().evaluate(lc)

    def test_high_confidence_gives_execute(self):
        dr = self._pipeline_result(0.95, 0.95, 0.95)
        a  = _policy().apply(dr)
        assert a.action_type == ConservativeActionType.EXECUTE

    def test_low_nav_gives_slow_or_halt(self):
        dr = self._pipeline_result(0.90, 0.40, 0.90)
        a  = _policy().apply(dr)
        assert a.action_type in (
            ConservativeActionType.SLOW_DOWN,
            ConservativeActionType.HALT,
        )

    def test_low_perception_gives_rescan_or_halt(self):
        dr = self._pipeline_result(0.30, 0.90, 0.90)
        a  = _policy().apply(dr)
        assert a.action_type in (
            ConservativeActionType.RESCAN,
            ConservativeActionType.HALT,
        )

    def test_low_grasp_gives_reposition_or_halt(self):
        dr = self._pipeline_result(0.90, 0.90, 0.40)
        a  = _policy().apply(dr)
        assert a.action_type in (
            ConservativeActionType.REPOSITION,
            ConservativeActionType.HALT,
        )

    def test_very_low_all_gives_halt(self):
        dr = self._pipeline_result(0.20, 0.20, 0.20)
        a  = _policy().apply(dr)
        assert a.action_type == ConservativeActionType.HALT
        assert a.velocity_scale == pytest.approx(0.0)

    def test_halt_velocity_is_full_stop(self):
        dr = self._pipeline_result(0.10, 0.10, 0.10)
        a  = _policy().apply(dr)
        assert a.velocity_scale == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# LatencyBudgetConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestLatencyBudgetConfig:

    def test_defaults(self):
        b = LatencyBudgetConfig()
        assert b.nav_score_budget_ms   == pytest.approx(3.0)
        assert b.grasp_score_budget_ms == pytest.approx(3.0)
        assert b.propagate_budget_ms   == pytest.approx(5.0)
        assert b.gate_budget_ms        == pytest.approx(3.0)
        assert b.policy_budget_ms      == pytest.approx(3.0)
        assert b.total_budget_ms       == pytest.approx(20.0)

    def test_frozen(self):
        b = LatencyBudgetConfig()
        with pytest.raises((FrozenInstanceError, TypeError)):
            b.total_budget_ms = 50.0  # type: ignore[misc]

    def test_total_budget_is_20ms(self):
        assert LatencyBudgetConfig().total_budget_ms == pytest.approx(20.0)


# ══════════════════════════════════════════════════════════════════════════════
# PhaseLatency
# ══════════════════════════════════════════════════════════════════════════════

class TestPhaseLatency:

    def test_passed_when_under_budget(self):
        p = PhaseLatency("nav_score", elapsed_ms=1.0, budget_ms=3.0)
        assert p.passed is True

    def test_passed_at_exact_budget(self):
        p = PhaseLatency("nav_score", elapsed_ms=3.0, budget_ms=3.0)
        assert p.passed is True

    def test_failed_when_over_budget(self):
        p = PhaseLatency("nav_score", elapsed_ms=3.1, budget_ms=3.0)
        assert p.passed is False

    def test_repr_ok_label(self):
        p = PhaseLatency("gate", elapsed_ms=1.0, budget_ms=3.0)
        assert "OK" in repr(p)

    def test_repr_slow_label(self):
        p = PhaseLatency("gate", elapsed_ms=5.0, budget_ms=3.0)
        assert "SLOW" in repr(p)


# ══════════════════════════════════════════════════════════════════════════════
# LatencyReport
# ══════════════════════════════════════════════════════════════════════════════

class TestLatencyReport:

    def _make_report(self, total_ms=5.0, budget_ms=20.0, phase_ms=1.0):
        phases = [
            PhaseLatency("nav_score",   phase_ms, 3.0),
            PhaseLatency("grasp_score", phase_ms, 3.0),
            PhaseLatency("propagate",   phase_ms, 5.0),
            PhaseLatency("gate",        phase_ms, 3.0),
            PhaseLatency("policy",      phase_ms, 3.0),
        ]
        lc = _layered(0.85)
        dr = DecisionGate().evaluate(lc)
        ca = ConservativePolicy().apply(dr)
        return LatencyReport(
            phases=phases, total_ms=total_ms, budget_ms=budget_ms,
            decision_result=dr, conservative_action=ca,
        )

    def test_passed_when_under_budget(self):
        r = self._make_report(total_ms=10.0, budget_ms=20.0)
        assert r.passed is True

    def test_failed_when_over_budget(self):
        r = self._make_report(total_ms=25.0, budget_ms=20.0)
        assert r.passed is False

    def test_failed_when_phase_over_budget(self):
        phases = [
            PhaseLatency("nav_score",   10.0, 3.0),  # over budget
            PhaseLatency("grasp_score",  1.0, 3.0),
            PhaseLatency("propagate",    1.0, 5.0),
            PhaseLatency("gate",         1.0, 3.0),
            PhaseLatency("policy",       1.0, 3.0),
        ]
        lc = _layered(0.85)
        dr = DecisionGate().evaluate(lc)
        ca = ConservativePolicy().apply(dr)
        r = LatencyReport(phases=phases, total_ms=14.0, budget_ms=20.0,
                          decision_result=dr, conservative_action=ca)
        assert r.passed is False

    def test_slowest_phase(self):
        r = self._make_report()
        # All phases have equal elapsed — slowest should return one of them
        assert r.slowest_phase.name in {"nav_score", "grasp_score",
                                         "propagate", "gate", "policy"}

    def test_summary_contains_pass(self):
        r = self._make_report(total_ms=5.0)
        assert "PASS" in r.summary()

    def test_summary_contains_fail(self):
        r = self._make_report(total_ms=25.0)
        assert "FAIL" in r.summary()

    def test_summary_lists_all_phases(self):
        r = self._make_report()
        s = r.summary()
        for name in ("nav_score", "grasp_score", "propagate", "gate", "policy"):
            assert name in s

    def test_repr_contains_total(self):
        r = self._make_report(total_ms=7.5)
        assert "7.5" in repr(r) or "7.50" in repr(r)


# ══════════════════════════════════════════════════════════════════════════════
# DecisionLatencyProfiler — structure
# ══════════════════════════════════════════════════════════════════════════════

class TestDecisionLatencyProfilerStructure:

    def test_profile_returns_report(self):
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.85, n_warmup_runs=1)
        assert isinstance(report, LatencyReport)

    def test_report_has_five_phases(self):
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.85, n_warmup_runs=1)
        assert len(report.phases) == 5

    def test_phase_names_correct(self):
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.85, n_warmup_runs=1)
        names = {p.name for p in report.phases}
        assert names == {"nav_score", "grasp_score", "propagate", "gate", "policy"}

    def test_report_contains_decision_result(self):
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.85, n_warmup_runs=1)
        assert isinstance(report.decision_result, DecisionResult)

    def test_report_contains_conservative_action(self):
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.85, n_warmup_runs=1)
        assert isinstance(report.conservative_action, ConservativeAction)

    def test_total_ms_is_sum_of_phases(self):
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.85, n_warmup_runs=1)
        assert report.total_ms == pytest.approx(
            sum(p.elapsed_ms for p in report.phases), rel=1e-6
        )

    def test_budget_ms_matches_config(self):
        cfg = LatencyBudgetConfig(total_budget_ms=50.0)
        profiler = DecisionLatencyProfiler(budget=cfg)
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.85, n_warmup_runs=1)
        assert report.budget_ms == pytest.approx(50.0)


# ══════════════════════════════════════════════════════════════════════════════
# DecisionLatencyProfiler — latency requirement (20 ms)
# ══════════════════════════════════════════════════════════════════════════════

class TestDecisionLatencyUnder20ms:
    """
    Hard latency requirement: the complete uncertainty gate must finish in < 20 ms.

    These tests use n_warmup_runs=5 to eliminate import / JIT cold-start overhead
    and measure steady-state performance.
    """

    def test_full_pipeline_under_20ms(self):
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.85,
                                    n_warmup_runs=5)
        assert report.total_ms < 20.0, (
            f"Gate exceeded 20ms budget: {report.total_ms:.3f} ms\n"
            + report.summary()
        )

    def test_full_pipeline_passes_report(self):
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.85,
                                    n_warmup_runs=5)
        assert report.passed, (
            f"LatencyReport.passed is False:\n{report.summary()}"
        )

    def test_each_phase_under_its_budget(self):
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.85,
                                    n_warmup_runs=5)
        for phase in report.phases:
            assert phase.passed, (
                f"Phase '{phase.name}' exceeded budget: "
                f"{phase.elapsed_ms:.3f} ms > {phase.budget_ms} ms"
            )

    def test_repeated_runs_p95_under_20ms(self):
        profiler = DecisionLatencyProfiler()
        stats    = profiler.profile_repeated(_ideal_nav(), _ideal_grasp(), 0.85,
                                             n_runs=50, n_warmup_runs=5)
        assert stats["p95_ms"] < 20.0, (
            f"p95 latency {stats['p95_ms']:.3f} ms exceeds 20ms budget"
        )

    def test_repeated_runs_mean_well_under_budget(self):
        profiler = DecisionLatencyProfiler()
        stats    = profiler.profile_repeated(_ideal_nav(), _ideal_grasp(), 0.85,
                                             n_runs=50, n_warmup_runs=5)
        assert stats["mean_ms"] < 10.0, (
            f"Mean latency {stats['mean_ms']:.3f} ms — expected well under 10ms"
        )

    def test_near_zero_perception_still_under_20ms(self):
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(_ideal_nav(), _ideal_grasp(), 0.05,
                                    n_warmup_runs=5)
        assert report.total_ms < 20.0

    def test_all_worst_case_signals_still_under_20ms(self):
        worst_nav = NavSignals(
            min_clearance_m=0.0, path_length_m=20.0, n_waypoints=0,
            localisation_std_m=1.0, goal_distance_m=25.0,
        )
        worst_grasp = GraspSignals(
            best_candidate_score=0.0, n_candidates=0,
            depth_std_m=0.5, reachability=0.0, n_cloud_points=0,
        )
        profiler = DecisionLatencyProfiler()
        report   = profiler.profile(worst_nav, worst_grasp, 0.0, n_warmup_runs=5)
        assert report.total_ms < 20.0


# ══════════════════════════════════════════════════════════════════════════════
# DecisionLatencyProfiler — profile_repeated statistics
# ══════════════════════════════════════════════════════════════════════════════

class TestDecisionLatencyRepeated:

    def test_returns_required_keys(self):
        profiler = DecisionLatencyProfiler()
        stats    = profiler.profile_repeated(_ideal_nav(), _ideal_grasp(), 0.85,
                                             n_runs=10, n_warmup_runs=2)
        for key in ("min_ms", "max_ms", "mean_ms", "p95_ms", "p99_ms",
                    "budget_ms", "pass_rate"):
            assert key in stats, f"Missing key: {key}"

    def test_min_le_mean_le_max(self):
        profiler = DecisionLatencyProfiler()
        stats    = profiler.profile_repeated(_ideal_nav(), _ideal_grasp(), 0.85,
                                             n_runs=20, n_warmup_runs=2)
        assert stats["min_ms"] <= stats["mean_ms"] <= stats["max_ms"]

    def test_p95_le_max(self):
        profiler = DecisionLatencyProfiler()
        stats    = profiler.profile_repeated(_ideal_nav(), _ideal_grasp(), 0.85,
                                             n_runs=20, n_warmup_runs=2)
        assert stats["p95_ms"] <= stats["max_ms"]

    def test_pass_rate_is_fraction(self):
        profiler = DecisionLatencyProfiler()
        stats    = profiler.profile_repeated(_ideal_nav(), _ideal_grasp(), 0.85,
                                             n_runs=20, n_warmup_runs=2)
        assert 0.0 <= stats["pass_rate"] <= 1.0

    def test_budget_ms_in_stats(self):
        profiler = DecisionLatencyProfiler()
        stats    = profiler.profile_repeated(_ideal_nav(), _ideal_grasp(), 0.85,
                                             n_runs=10, n_warmup_runs=2)
        assert stats["budget_ms"] == pytest.approx(20.0)

    def test_pass_rate_is_one_for_easy_workload(self):
        profiler = DecisionLatencyProfiler()
        stats    = profiler.profile_repeated(_ideal_nav(), _ideal_grasp(), 0.85,
                                             n_runs=30, n_warmup_runs=5)
        assert stats["pass_rate"] == pytest.approx(1.0), (
            f"Expected 100% pass rate, got {stats['pass_rate']:.2%} "
            f"(p99={stats['p99_ms']:.3f}ms)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# DecisionLatencyProfiler — repr
# ══════════════════════════════════════════════════════════════════════════════

class TestDecisionLatencyProfilerRepr:

    def test_repr(self):
        profiler = DecisionLatencyProfiler()
        text = repr(profiler)
        assert "20" in text   # budget
