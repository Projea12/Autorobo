"""
tests/test_uncertainty_layer.py — Phase 7 uncertainty & decision layer tests.

Covers all four modules:
  NavigationConfidenceScorer  — per-signal sub-scores + weighted combination
  GraspConfidenceScorer       — per-signal sub-scores + weighted combination
  UncertaintyPipeline         — geometric-mean propagation + bottleneck detection
  DecisionGate                — 0.90/0.60 thresholds + all three decisions
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from planning.nav_confidence import (
    NavConfidence, NavConfidenceConfig, NavSignals, NavigationConfidenceScorer,
)
from planning.grasp_confidence import (
    GraspConfidence, GraspConfidenceConfig, GraspSignals, GraspConfidenceScorer,
)
from planning.uncertainty_pipeline import (
    LayeredConfidence, PropagationConfig, UncertaintyPipeline,
)
from planning.decision_gate import (
    Decision, DecisionGate, DecisionGateConfig, DecisionResult,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _nav_scorer(**kw) -> NavigationConfidenceScorer:
    return NavigationConfidenceScorer(NavConfidenceConfig(**kw))


def _grasp_scorer(**kw) -> GraspConfidenceScorer:
    return GraspConfidenceScorer(GraspConfidenceConfig(**kw))


def _pipeline(**kw) -> UncertaintyPipeline:
    return UncertaintyPipeline(PropagationConfig(**kw))


def _gate(**kw) -> DecisionGate:
    return DecisionGate(DecisionGateConfig(**kw))


def _layered(p=0.9, n=0.9, g=0.9) -> LayeredConfidence:
    return _pipeline().propagate(p, n, g)


# ══════════════════════════════════════════════════════════════════════════════
# NavigationConfidenceScorer — config
# ══════════════════════════════════════════════════════════════════════════════

class TestNavConfidenceConfig:
    def test_defaults(self):
        cfg = NavConfidenceConfig()
        assert cfg.clearance_min_m   == pytest.approx(0.10)
        assert cfg.clearance_sat_m   == pytest.approx(1.00)
        assert cfg.w_clearance       == pytest.approx(0.35)
        assert cfg.w_path            == pytest.approx(0.25)
        assert cfg.w_localisation    == pytest.approx(0.25)
        assert cfg.w_goal_dist       == pytest.approx(0.15)

    def test_frozen(self):
        with pytest.raises(Exception):
            NavConfidenceConfig().w_clearance = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# NavigationConfidenceScorer — clearance sub-score
# ══════════════════════════════════════════════════════════════════════════════

class TestNavClearanceScore:
    def _scorer(self):
        return NavigationConfidenceScorer(
            NavConfidenceConfig(clearance_min_m=0.1, clearance_sat_m=1.0)
        )

    def test_blocked_path_zero(self):
        s = self._scorer()
        assert s._clearance_score(0.0)  == pytest.approx(0.0)

    def test_below_min_zero(self):
        s = self._scorer()
        assert s._clearance_score(0.05) == pytest.approx(0.0)

    def test_at_min_zero(self):
        s = self._scorer()
        assert s._clearance_score(0.1)  == pytest.approx(0.0)

    def test_at_sat_one(self):
        s = self._scorer()
        assert s._clearance_score(1.0)  == pytest.approx(1.0)

    def test_above_sat_one(self):
        s = self._scorer()
        assert s._clearance_score(5.0)  == pytest.approx(1.0)

    def test_midpoint(self):
        s = self._scorer()
        # midpoint of [0.1, 1.0] = 0.55 → score = 0.5
        assert s._clearance_score(0.55) == pytest.approx(0.5, abs=1e-6)

    def test_monotone_increasing(self):
        s = self._scorer()
        dists = [0.0, 0.2, 0.5, 0.8, 1.0, 2.0]
        scores = [s._clearance_score(d) for d in dists]
        assert scores == sorted(scores)


# ══════════════════════════════════════════════════════════════════════════════
# NavigationConfidenceScorer — path sub-score
# ══════════════════════════════════════════════════════════════════════════════

class TestNavPathScore:
    def _scorer(self):
        return NavigationConfidenceScorer(
            NavConfidenceConfig(path_length_max_m=10.0, n_waypoints_min=1)
        )

    def test_no_waypoints_zero(self):
        s = self._scorer()
        assert s._path_score(1.0, 0) == pytest.approx(0.0)

    def test_short_path_high_score(self):
        s = self._scorer()
        assert s._path_score(1.0, 5) == pytest.approx(0.9)

    def test_at_max_length_zero(self):
        s = self._scorer()
        assert s._path_score(10.0, 5) == pytest.approx(0.0)

    def test_beyond_max_clipped_zero(self):
        s = self._scorer()
        assert s._path_score(20.0, 5) == pytest.approx(0.0)

    def test_half_length_half_score(self):
        s = self._scorer()
        assert s._path_score(5.0, 5) == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# NavigationConfidenceScorer — localisation sub-score
# ══════════════════════════════════════════════════════════════════════════════

class TestNavLocalisationScore:
    def _scorer(self):
        return NavigationConfidenceScorer(
            NavConfidenceConfig(localisation_std_max=0.5)
        )

    def test_perfect_localisation_one(self):
        s = self._scorer()
        assert s._localisation_score(0.0) == pytest.approx(1.0)

    def test_at_max_std_zero(self):
        s = self._scorer()
        assert s._localisation_score(0.5) == pytest.approx(0.0)

    def test_above_max_clipped_zero(self):
        s = self._scorer()
        assert s._localisation_score(1.0) == pytest.approx(0.0)

    def test_half_std_half_score(self):
        s = self._scorer()
        assert s._localisation_score(0.25) == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# NavigationConfidenceScorer — goal distance sub-score
# ══════════════════════════════════════════════════════════════════════════════

class TestNavGoalDistScore:
    def _scorer(self):
        return NavigationConfidenceScorer(
            NavConfidenceConfig(goal_dist_near_m=3.0, goal_dist_far_m=20.0)
        )

    def test_nearby_goal_one(self):
        s = self._scorer()
        assert s._goal_dist_score(1.0) == pytest.approx(1.0)

    def test_at_near_boundary_one(self):
        s = self._scorer()
        assert s._goal_dist_score(3.0) == pytest.approx(1.0)

    def test_at_far_boundary_zero(self):
        s = self._scorer()
        assert s._goal_dist_score(20.0) == pytest.approx(0.0)

    def test_beyond_far_clipped_zero(self):
        s = self._scorer()
        assert s._goal_dist_score(50.0) == pytest.approx(0.0)

    def test_midpoint(self):
        s = self._scorer()
        # midpoint between 3 and 20 = 11.5 → score = 0.5
        assert s._goal_dist_score(11.5) == pytest.approx(0.5, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# NavigationConfidenceScorer — combined score
# ══════════════════════════════════════════════════════════════════════════════

class TestNavCombinedScore:
    def test_ideal_signals_near_one(self):
        scorer  = NavigationConfidenceScorer()
        signals = NavSignals(
            min_clearance_m    = 2.0,
            path_length_m      = 0.5,
            n_waypoints        = 10,
            localisation_std_m = 0.0,
            goal_distance_m    = 1.0,
        )
        result = scorer.score(signals)
        assert result.combined >= 0.95

    def test_blocked_path_combined_low(self):
        scorer  = NavigationConfidenceScorer()
        signals = NavSignals(
            min_clearance_m    = 0.0,
            path_length_m      = 5.0,
            n_waypoints        = 0,
            localisation_std_m = 0.5,
            goal_distance_m    = 25.0,
        )
        result = scorer.score(signals)
        assert result.combined < 0.15

    def test_combined_in_unit_interval(self):
        scorer = NavigationConfidenceScorer()
        for _ in range(50):
            signals = NavSignals(
                min_clearance_m    = float(np.random.uniform(0, 5)),
                path_length_m      = float(np.random.uniform(0, 20)),
                n_waypoints        = int(np.random.randint(0, 20)),
                localisation_std_m = float(np.random.uniform(0, 1)),
                goal_distance_m    = float(np.random.uniform(0, 30)),
            )
            result = scorer.score(signals)
            assert 0.0 <= result.combined <= 1.0

    def test_result_type(self):
        result = NavigationConfidenceScorer().score(NavSignals())
        assert isinstance(result, NavConfidence)

    def test_repr(self):
        assert "NavigationConfidenceScorer" in repr(NavigationConfidenceScorer())


# ══════════════════════════════════════════════════════════════════════════════
# GraspConfidenceScorer — config
# ══════════════════════════════════════════════════════════════════════════════

class TestGraspConfidenceConfig:
    def test_defaults(self):
        cfg = GraspConfidenceConfig()
        assert cfg.n_candidates_sat == 5
        assert cfg.depth_std_max_m  == pytest.approx(0.10)
        assert cfg.w_candidate      == pytest.approx(0.35)
        assert cfg.w_depth          == pytest.approx(0.20)
        assert cfg.w_reachability   == pytest.approx(0.20)

    def test_frozen(self):
        with pytest.raises(Exception):
            GraspConfidenceConfig().w_depth = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# GraspConfidenceScorer — sub-scores
# ══════════════════════════════════════════════════════════════════════════════

class TestGraspSubScores:
    def _scorer(self):
        return GraspConfidenceScorer(GraspConfidenceConfig(
            n_candidates_sat=4, depth_std_max_m=0.1,
            cloud_pts_min=10, cloud_pts_sat=100,
        ))

    def test_zero_candidates_zero(self):
        s = self._scorer()
        assert s._n_candidates_score(0) == pytest.approx(0.0)

    def test_at_sat_candidates_one(self):
        s = self._scorer()
        assert s._n_candidates_score(4) == pytest.approx(1.0)

    def test_above_sat_candidates_one(self):
        s = self._scorer()
        assert s._n_candidates_score(10) == pytest.approx(1.0)

    def test_half_candidates_half(self):
        s = self._scorer()
        assert s._n_candidates_score(2) == pytest.approx(0.5)

    def test_zero_depth_std_one(self):
        s = self._scorer()
        assert s._depth_score(0.0) == pytest.approx(1.0)

    def test_at_max_depth_std_zero(self):
        s = self._scorer()
        assert s._depth_score(0.1) == pytest.approx(0.0)

    def test_above_max_depth_std_zero(self):
        s = self._scorer()
        assert s._depth_score(0.5) == pytest.approx(0.0)

    def test_sparse_cloud_zero(self):
        s = self._scorer()
        assert s._cloud_score(5) == pytest.approx(0.0)

    def test_at_cloud_min_zero(self):
        s = self._scorer()
        assert s._cloud_score(10) == pytest.approx(0.0)

    def test_at_cloud_sat_one(self):
        s = self._scorer()
        assert s._cloud_score(100) == pytest.approx(1.0)

    def test_midpoint_cloud(self):
        s = self._scorer()
        # midpoint of [10, 100] = 55 → score = 0.5
        assert s._cloud_score(55) == pytest.approx(0.5, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# GraspConfidenceScorer — combined score
# ══════════════════════════════════════════════════════════════════════════════

class TestGraspCombinedScore:
    def test_ideal_signals_near_one(self):
        scorer  = GraspConfidenceScorer()
        signals = GraspSignals(
            best_candidate_score = 1.0,
            n_candidates         = 10,
            depth_std_m          = 0.0,
            reachability         = 1.0,
            n_cloud_points       = 1000,
        )
        result = scorer.score(signals)
        assert result.combined >= 0.95

    def test_failed_plan_zero(self):
        scorer  = GraspConfidenceScorer()
        signals = GraspSignals(
            best_candidate_score = 0.0,
            n_candidates         = 0,
            depth_std_m          = 0.2,
            reachability         = 0.0,
            n_cloud_points       = 0,
        )
        result = scorer.score(signals)
        assert result.combined < 0.10

    def test_combined_in_unit_interval(self):
        scorer = GraspConfidenceScorer()
        for _ in range(50):
            signals = GraspSignals(
                best_candidate_score = float(np.random.uniform(0, 1)),
                n_candidates         = int(np.random.randint(0, 10)),
                depth_std_m          = float(np.random.uniform(0, 0.5)),
                reachability         = float(np.random.uniform(0, 1)),
                n_cloud_points       = int(np.random.randint(0, 1000)),
            )
            result = scorer.score(signals)
            assert 0.0 <= result.combined <= 1.0

    def test_result_type(self):
        result = GraspConfidenceScorer().score(GraspSignals())
        assert isinstance(result, GraspConfidence)

    def test_repr(self):
        assert "GraspConfidenceScorer" in repr(GraspConfidenceScorer())


# ══════════════════════════════════════════════════════════════════════════════
# UncertaintyPipeline — propagation
# ══════════════════════════════════════════════════════════════════════════════

class TestPropagationConfig:
    def test_defaults(self):
        cfg = PropagationConfig()
        assert cfg.perception_weight == pytest.approx(0.40)
        assert cfg.nav_weight        == pytest.approx(0.30)
        assert cfg.grasp_weight      == pytest.approx(0.30)

    def test_frozen(self):
        with pytest.raises(Exception):
            PropagationConfig().nav_weight = 0.5


class TestUncertaintyPipelineGeometricMean:
    def test_all_perfect_gives_one(self):
        p = _pipeline()
        r = p.propagate(1.0, 1.0, 1.0)
        assert r.propagated == pytest.approx(1.0)

    def test_all_zero_gives_floor(self):
        p = _pipeline(floor=0.0)
        r = p.propagate(0.0, 0.0, 0.0)
        assert r.propagated == pytest.approx(0.0)

    def test_one_zero_layer_drives_to_floor(self):
        p = _pipeline(floor=0.0)
        r = p.propagate(1.0, 0.0, 1.0)
        assert r.propagated == pytest.approx(0.0)

    def test_symmetric_equal_weights(self):
        """Swapping nav and grasp scores with equal weights should give same result."""
        p = _pipeline(nav_weight=0.30, grasp_weight=0.30)
        r1 = p.propagate(0.8, 0.6, 0.7)
        r2 = p.propagate(0.8, 0.7, 0.6)
        assert r1.propagated == pytest.approx(r2.propagated, abs=1e-6)

    def test_propagated_between_zero_and_one(self):
        p = _pipeline()
        for _ in range(100):
            scores = np.random.uniform(0, 1, 3)
            r = p.propagate(*scores)
            assert 0.0 <= r.propagated <= 1.0

    def test_geometric_mean_correct_formula(self):
        """Verify the propagated score matches the hand-computed weighted geometric mean."""
        α, β, γ = 0.4, 0.3, 0.3
        p_s, n_s, g_s = 0.8, 0.7, 0.6
        expected = math.exp(
            (α * math.log(p_s) + β * math.log(n_s) + γ * math.log(g_s))
            / (α + β + γ)
        )
        p = _pipeline(perception_weight=α, nav_weight=β, grasp_weight=γ)
        r = p.propagate(p_s, n_s, g_s)
        assert r.propagated == pytest.approx(expected, rel=1e-6)

    def test_low_perception_degrades_propagated(self):
        """Low perception score must drag propagated well below nav and grasp scores."""
        p = _pipeline()
        r_low  = p.propagate(0.1, 0.9, 0.9)
        r_high = p.propagate(0.9, 0.9, 0.9)
        assert r_low.propagated < r_high.propagated

    def test_floor_respected(self):
        p = _pipeline(floor=0.05)
        r = p.propagate(0.0, 0.5, 0.5)
        assert r.propagated >= 0.05

    def test_scores_clipped_to_unit_interval(self):
        p = _pipeline()
        r = p.propagate(1.5, -0.2, 0.8)   # out-of-range inputs
        assert 0.0 <= r.propagated <= 1.0


class TestUncertaintyPipelineBottleneck:
    def test_bottleneck_is_lowest_layer(self):
        p = _pipeline()
        r = p.propagate(0.9, 0.3, 0.8)
        assert r.bottleneck_layer == "navigation"
        assert r.bottleneck_score == pytest.approx(0.3)

    def test_bottleneck_perception(self):
        r = _pipeline().propagate(0.1, 0.9, 0.9)
        assert r.bottleneck_layer == "perception"

    def test_bottleneck_grasp(self):
        r = _pipeline().propagate(0.9, 0.9, 0.2)
        assert r.bottleneck_layer == "grasp"

    def test_all_equal_bottleneck_any(self):
        r = _pipeline().propagate(0.7, 0.7, 0.7)
        assert r.bottleneck_layer in ("perception", "navigation", "grasp")


class TestUncertaintyPipelineEffectiveScores:
    def test_eff_perception_equals_raw(self):
        r = _pipeline().propagate(0.8, 0.7, 0.6)
        assert r.eff_perception == pytest.approx(0.8)

    def test_eff_nav_degraded_by_perception(self):
        r = _pipeline(perception_weight=0.4, nav_weight=0.3).propagate(0.5, 0.9, 0.9)
        assert r.eff_nav < r.nav_score

    def test_eff_grasp_most_degraded(self):
        # Equal raw scores: grasp receives most upstream degradation
        r = _pipeline().propagate(0.5, 0.5, 0.5)
        assert r.eff_grasp <= r.eff_nav <= r.eff_perception

    def test_all_perfect_effective_all_one(self):
        r = _pipeline().propagate(1.0, 1.0, 1.0)
        assert r.eff_perception == pytest.approx(1.0)
        assert r.eff_nav        == pytest.approx(1.0)
        assert r.eff_grasp      == pytest.approx(1.0)


class TestUncertaintyPipelinePartial:
    def test_partial_two_layers(self):
        p = _pipeline()
        score = p.propagate_partial({"perception": 0.8, "navigation": 0.6})
        assert 0.0 < score < 1.0

    def test_partial_single_layer(self):
        p = _pipeline()
        score = p.propagate_partial({"perception": 0.7})
        assert score == pytest.approx(0.7)

    def test_partial_empty_is_zero(self):
        p = _pipeline()
        score = p.propagate_partial({})
        assert score == pytest.approx(0.0)

    def test_partial_unknown_key_ignored(self):
        p = _pipeline()
        score = p.propagate_partial({"perception": 0.8, "alien": 0.1})
        assert score == pytest.approx(0.8)

    def test_partial_zero_drives_to_floor(self):
        p = _pipeline(floor=0.0)
        score = p.propagate_partial({"perception": 0.0, "navigation": 0.9})
        assert score == pytest.approx(0.0)


class TestUncertaintyPipelineRepr:
    def test_repr(self):
        assert "UncertaintyPipeline" in repr(_pipeline())


# ══════════════════════════════════════════════════════════════════════════════
# DecisionGate — thresholds
# ══════════════════════════════════════════════════════════════════════════════

class TestDecisionGateConfig:
    def test_defaults(self):
        cfg = DecisionGateConfig()
        assert cfg.act_threshold    == pytest.approx(0.90)
        assert cfg.gather_threshold == pytest.approx(0.60)

    def test_frozen(self):
        with pytest.raises(Exception):
            DecisionGateConfig().act_threshold = 0.8


class TestDecisionGateZones:
    def test_score_above_act_threshold_is_act(self):
        g = _gate()
        assert g.evaluate_score(0.95) == Decision.ACT

    def test_score_exactly_at_act_threshold_is_act(self):
        g = _gate()
        assert g.evaluate_score(0.90) == Decision.ACT

    def test_score_just_below_act_is_gather(self):
        g = _gate()
        assert g.evaluate_score(0.89) == Decision.GATHER

    def test_score_at_gather_threshold_is_gather(self):
        g = _gate()
        assert g.evaluate_score(0.60) == Decision.GATHER

    def test_score_just_below_gather_is_safer(self):
        g = _gate()
        assert g.evaluate_score(0.59) == Decision.SAFER

    def test_score_zero_is_safer(self):
        g = _gate()
        assert g.evaluate_score(0.0) == Decision.SAFER

    def test_score_one_is_act(self):
        g = _gate()
        assert g.evaluate_score(1.0) == Decision.ACT

    def test_mid_gather_is_gather(self):
        g = _gate()
        assert g.evaluate_score(0.75) == Decision.GATHER


class TestDecisionGateCustomThresholds:
    def test_custom_thresholds_respected(self):
        g = _gate(act_threshold=0.80, gather_threshold=0.50)
        assert g.evaluate_score(0.85) == Decision.ACT
        assert g.evaluate_score(0.70) == Decision.GATHER
        assert g.evaluate_score(0.40) == Decision.SAFER

    def test_boundary_at_custom_act(self):
        g = _gate(act_threshold=0.75)
        assert g.evaluate_score(0.75) == Decision.ACT
        assert g.evaluate_score(0.74) == Decision.GATHER


# ══════════════════════════════════════════════════════════════════════════════
# DecisionGate — full evaluate() with LayeredConfidence
# ══════════════════════════════════════════════════════════════════════════════

class TestDecisionGateEvaluate:
    def test_returns_decision_result(self):
        result = _gate().evaluate(_layered(0.95, 0.95, 0.95))
        assert isinstance(result, DecisionResult)

    def test_act_decision(self):
        result = _gate().evaluate(_layered(0.95, 0.95, 0.95))
        assert result.decision == Decision.ACT

    def test_gather_decision(self):
        # propagated should land in [0.60, 0.90)
        result = _gate().evaluate(_layered(0.85, 0.80, 0.70))
        assert result.decision == Decision.GATHER

    def test_safer_decision(self):
        result = _gate().evaluate(_layered(0.3, 0.5, 0.4))
        assert result.decision == Decision.SAFER

    def test_score_matches_propagated(self):
        layered = _layered(0.85, 0.80, 0.70)
        result  = _gate().evaluate(layered)
        assert result.score == pytest.approx(layered.propagated)

    def test_safer_action_propagated(self):
        g      = _gate(safer_action="retreat to safe zone")
        result = g.evaluate(_layered(0.2, 0.2, 0.2))
        assert result.safer_action == "retreat to safe zone"

    def test_bottleneck_reported(self):
        # nav is clearly the weakest
        result = _gate().evaluate(_layered(0.95, 0.3, 0.95))
        assert result.bottleneck_layer == "navigation"
        assert result.bottleneck_score == pytest.approx(0.3)

    def test_reason_contains_score(self):
        result = _gate().evaluate(_layered(0.95, 0.95, 0.95))
        assert "propagated=" in result.reason

    def test_gather_reason_mentions_bottleneck(self):
        result = _gate().evaluate(_layered(0.85, 0.50, 0.85))
        assert "bottleneck" in result.reason.lower() \
            or "navigation" in result.reason

    def test_safer_reason_mentions_safer_action(self):
        g      = _gate(safer_action="lower arm")
        result = g.evaluate(_layered(0.2, 0.2, 0.2))
        assert "lower arm" in result.reason

    def test_no_bottleneck_annotation_when_disabled(self):
        g      = _gate(annotate_bottleneck=False)
        result = g.evaluate(_layered(0.75, 0.55, 0.75))
        assert "bottleneck" not in result.reason


# ══════════════════════════════════════════════════════════════════════════════
# DecisionGate — threshold_summary
# ══════════════════════════════════════════════════════════════════════════════

class TestDecisionGateThresholdSummary:
    def test_summary_contains_thresholds(self):
        g = _gate()
        s = g.threshold_summary()
        assert "0.90" in s
        assert "0.60" in s

    def test_summary_contains_all_decisions(self):
        g = _gate()
        s = g.threshold_summary().upper()
        assert "ACT" in s
        assert "GATHER" in s
        assert "SAFER" in s


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: perception → nav → grasp → propagate → decide
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def _full_pipeline(self, perc, nav_sig, grasp_sig):
        from planning.nav_confidence import NavigationConfidenceScorer, NavSignals
        from planning.grasp_confidence import GraspConfidenceScorer, GraspSignals

        nav_r   = NavigationConfidenceScorer().score(nav_sig)
        grasp_r = GraspConfidenceScorer().score(grasp_sig)
        layered = _pipeline().propagate(perc, nav_r.combined, grasp_r.combined)
        return _gate().evaluate(layered), layered

    def test_all_confident_gives_act(self):
        nav_sig   = NavSignals(
            min_clearance_m=2.0, path_length_m=1.0,
            n_waypoints=8, localisation_std_m=0.02, goal_distance_m=1.5,
        )
        grasp_sig = GraspSignals(
            best_candidate_score=0.95, n_candidates=5,
            depth_std_m=0.01, reachability=0.98, n_cloud_points=400,
        )
        result, _ = self._full_pipeline(0.95, nav_sig, grasp_sig)
        assert result.decision == Decision.ACT

    def test_poor_perception_gives_safer(self):
        nav_sig   = NavSignals(
            min_clearance_m=1.0, path_length_m=2.0,
            n_waypoints=5, localisation_std_m=0.05, goal_distance_m=2.0,
        )
        grasp_sig = GraspSignals(
            best_candidate_score=0.9, n_candidates=4,
            depth_std_m=0.02, reachability=0.95, n_cloud_points=300,
        )
        result, _ = self._full_pipeline(0.15, nav_sig, grasp_sig)
        assert result.decision == Decision.SAFER

    def test_poor_nav_gives_gather(self):
        nav_sig   = NavSignals(
            min_clearance_m=0.15, path_length_m=14.0,
            n_waypoints=1, localisation_std_m=0.4, goal_distance_m=18.0,
        )
        grasp_sig = GraspSignals(
            best_candidate_score=0.9, n_candidates=5,
            depth_std_m=0.01, reachability=0.98, n_cloud_points=400,
        )
        result, layered = self._full_pipeline(0.92, nav_sig, grasp_sig)
        assert result.decision in (Decision.GATHER, Decision.SAFER)

    def test_bottleneck_matches_weakest_layer(self):
        nav_sig   = NavSignals(
            min_clearance_m=0.12, path_length_m=12.0,
            n_waypoints=1, localisation_std_m=0.45, goal_distance_m=19.0,
        )
        grasp_sig = GraspSignals(
            best_candidate_score=0.95, n_candidates=5,
            depth_std_m=0.01, reachability=1.0, n_cloud_points=500,
        )
        _, layered = self._full_pipeline(0.95, nav_sig, grasp_sig)
        assert layered.bottleneck_layer == "navigation"

    def test_propagated_below_individual_when_any_low(self):
        nav_sig   = NavSignals(
            min_clearance_m=0.0, path_length_m=20.0,
            n_waypoints=0, localisation_std_m=0.5, goal_distance_m=25.0,
        )
        grasp_sig = GraspSignals(
            best_candidate_score=1.0, n_candidates=10,
            depth_std_m=0.0, reachability=1.0, n_cloud_points=1000,
        )
        nav_r   = NavigationConfidenceScorer().score(nav_sig)
        grasp_r = GraspConfidenceScorer().score(grasp_sig)
        layered = _pipeline().propagate(0.95, nav_r.combined, grasp_r.combined)
        assert layered.propagated < 0.95
        assert layered.propagated < grasp_r.combined
