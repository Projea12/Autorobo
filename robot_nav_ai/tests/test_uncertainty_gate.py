"""
tests/test_uncertainty_gate.py — Unit tests for UncertaintyGate, GateConfig,
GateDecision, and GateResult.

All tests use synthetic ObjectConfidence / SceneConfidence objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from perception.detector import Detection
from perception.confidence import AggregatorConfig, ObjectConfidence, SceneConfidence
from perception.uncertainty_gate import (
    GateConfig, GateDecision, GateResult, UncertaintyGate,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_det(conf: float = 0.9) -> Detection:
    return Detection(
        class_id   = 0,
        class_name = "mug",
        confidence = conf,
        bbox_xyxy  = np.array([10, 10, 60, 50], dtype=np.float32),
        bbox_xywh  = np.array([35, 30, 50, 40], dtype=np.float32),
    )


def _make_obj(combined: float = 0.8, depth_score: float = 0.8,
              seg_score: float = 0.8, det_score: float = 0.8) -> ObjectConfidence:
    return ObjectConfidence(
        detection_score = det_score,
        depth_score     = depth_score,
        seg_score       = seg_score,
        combined        = combined,
        detection       = _make_det(conf=det_score),
    )


def _make_scene(global_score: float = 0.8, n: int = 1,
                obj_combined: float = None) -> SceneConfidence:
    c    = obj_combined if obj_combined is not None else global_score
    objs = [_make_obj(combined=c) for _ in range(n)]
    return SceneConfidence(objects=objs, global_score=global_score, n_objects=len(objs))


# ── GateConfig ────────────────────────────────────────────────────────────────

class TestGateConfig:
    def test_defaults(self):
        cfg = GateConfig()
        assert cfg.act_threshold    == pytest.approx(0.70)
        assert cfg.gather_threshold == pytest.approx(0.40)
        assert cfg.require_depth    is True
        assert cfg.require_seg      is False
        assert cfg.min_objects      == 1

    def test_frozen(self):
        with pytest.raises(Exception):
            GateConfig().act_threshold = 0.5

    def test_custom(self):
        cfg = GateConfig(act_threshold=0.8, gather_threshold=0.5,
                         require_seg=True, min_objects=2)
        assert cfg.act_threshold    == pytest.approx(0.8)
        assert cfg.gather_threshold == pytest.approx(0.5)
        assert cfg.require_seg      is True
        assert cfg.min_objects      == 2


# ── GateDecision ──────────────────────────────────────────────────────────────

class TestGateDecision:
    def test_act_value(self):
        assert GateDecision.ACT.value == "act"

    def test_gather_value(self):
        assert GateDecision.GATHER.value == "gather"

    def test_flag_value(self):
        assert GateDecision.FLAG.value == "flag"

    def test_skip_value(self):
        assert GateDecision.SKIP.value == "skip"

    def test_distinct_members(self):
        members = list(GateDecision)
        assert len(set(members)) == 4


# ── GateResult ────────────────────────────────────────────────────────────────

class TestGateResult:
    def _make(self, decision=GateDecision.ACT, score=0.8) -> GateResult:
        det = _make_det()
        return GateResult(
            decision         = decision,
            reason           = "test reason",
            score            = score,
            object_decisions = [(det, decision)],
        )

    def test_decision_stored(self):
        r = self._make(GateDecision.GATHER)
        assert r.decision == GateDecision.GATHER

    def test_score_stored(self):
        r = self._make(score=0.65)
        assert r.score == pytest.approx(0.65)

    def test_reason_stored(self):
        r = self._make()
        assert r.reason == "test reason"

    def test_repr_contains_decision_value(self):
        r = self._make(GateDecision.FLAG)
        assert "flag" in repr(r)

    def test_repr_contains_score(self):
        r = self._make(score=0.8)
        assert "0.800" in repr(r)

    def test_object_decisions_length(self):
        r = self._make()
        assert len(r.object_decisions) == 1


# ── UncertaintyGate._score_to_decision ───────────────────────────────────────

class TestScoreToDecision:
    def _gate(self, act=0.7, gather=0.4):
        return UncertaintyGate(GateConfig(act_threshold=act,
                                          gather_threshold=gather,
                                          require_depth=False))

    def test_above_act_is_act(self):
        assert self._gate()._score_to_decision(0.9) == GateDecision.ACT

    def test_at_act_is_act(self):
        assert self._gate()._score_to_decision(0.7) == GateDecision.ACT

    def test_just_below_act_is_gather(self):
        assert self._gate()._score_to_decision(0.699) == GateDecision.GATHER

    def test_at_gather_is_gather(self):
        assert self._gate()._score_to_decision(0.4) == GateDecision.GATHER

    def test_just_below_gather_is_flag(self):
        assert self._gate()._score_to_decision(0.399) == GateDecision.FLAG

    def test_zero_is_flag(self):
        assert self._gate()._score_to_decision(0.0) == GateDecision.FLAG

    def test_one_is_act(self):
        assert self._gate()._score_to_decision(1.0) == GateDecision.ACT


# ── UncertaintyGate.evaluate_object ──────────────────────────────────────────

class TestEvaluateObject:
    def test_high_combined_gives_act(self):
        gate = UncertaintyGate(GateConfig(act_threshold=0.7, require_depth=False))
        assert gate.evaluate_object(_make_obj(combined=0.9)) == GateDecision.ACT

    def test_mid_combined_gives_gather(self):
        gate = UncertaintyGate(GateConfig(act_threshold=0.7, gather_threshold=0.4,
                                          require_depth=False))
        assert gate.evaluate_object(_make_obj(combined=0.5)) == GateDecision.GATHER

    def test_low_combined_gives_flag(self):
        gate = UncertaintyGate(GateConfig(gather_threshold=0.4, require_depth=False))
        assert gate.evaluate_object(_make_obj(combined=0.2)) == GateDecision.FLAG

    def test_require_depth_zero_depth_gives_flag(self):
        gate = UncertaintyGate(GateConfig(require_depth=True))
        obj  = _make_obj(combined=0.95, depth_score=0.0)
        assert gate.evaluate_object(obj) == GateDecision.FLAG

    def test_require_depth_false_zero_depth_ok(self):
        gate = UncertaintyGate(GateConfig(require_depth=False, act_threshold=0.7))
        obj  = _make_obj(combined=0.95, depth_score=0.0)
        assert gate.evaluate_object(obj) == GateDecision.ACT

    def test_require_seg_zero_seg_gives_flag(self):
        gate = UncertaintyGate(GateConfig(require_seg=True))
        obj  = _make_obj(combined=0.95, seg_score=0.0)
        assert gate.evaluate_object(obj) == GateDecision.FLAG

    def test_require_seg_false_zero_seg_ok(self):
        gate = UncertaintyGate(GateConfig(require_seg=False, act_threshold=0.7))
        obj  = _make_obj(combined=0.95, seg_score=0.0)
        assert gate.evaluate_object(obj) == GateDecision.ACT

    def test_both_require_depth_and_seg_zero_depth_flags(self):
        gate = UncertaintyGate(GateConfig(require_depth=True, require_seg=True))
        obj  = _make_obj(combined=0.95, depth_score=0.0, seg_score=0.8)
        assert gate.evaluate_object(obj) == GateDecision.FLAG

    def test_both_require_depth_and_seg_zero_seg_flags(self):
        gate = UncertaintyGate(GateConfig(require_depth=True, require_seg=True))
        obj  = _make_obj(combined=0.95, depth_score=0.8, seg_score=0.0)
        assert gate.evaluate_object(obj) == GateDecision.FLAG

    def test_at_act_threshold_gives_act(self):
        gate = UncertaintyGate(GateConfig(act_threshold=0.7, require_depth=False))
        assert gate.evaluate_object(_make_obj(combined=0.7)) == GateDecision.ACT

    def test_at_gather_threshold_gives_gather(self):
        gate = UncertaintyGate(GateConfig(gather_threshold=0.4, require_depth=False))
        assert gate.evaluate_object(_make_obj(combined=0.4)) == GateDecision.GATHER

    def test_nonzero_depth_with_require_depth_not_flagged(self):
        gate = UncertaintyGate(GateConfig(require_depth=True, act_threshold=0.7))
        obj  = _make_obj(combined=0.9, depth_score=0.1)   # nonzero → not FLAG
        assert gate.evaluate_object(obj) == GateDecision.ACT


# ── UncertaintyGate.evaluate ──────────────────────────────────────────────────

class TestEvaluate:
    def test_skip_when_zero_objects(self):
        gate  = UncertaintyGate()
        scene = SceneConfidence(objects=[], global_score=0.0, n_objects=0)
        r     = gate.evaluate(scene)
        assert r.decision == GateDecision.SKIP

    def test_skip_when_below_min_objects(self):
        gate  = UncertaintyGate(GateConfig(min_objects=3))
        scene = _make_scene(global_score=0.95, n=2, obj_combined=0.95)
        r     = gate.evaluate(scene)
        assert r.decision == GateDecision.SKIP

    def test_skip_object_decisions_empty(self):
        gate  = UncertaintyGate()
        scene = SceneConfidence(objects=[], global_score=0.0, n_objects=0)
        r     = gate.evaluate(scene)
        assert r.object_decisions == []

    def test_act_when_global_high_and_objects_ok(self):
        gate  = UncertaintyGate(GateConfig(act_threshold=0.7, require_depth=False))
        scene = _make_scene(global_score=0.9, n=1, obj_combined=0.9)
        assert gate.evaluate(scene).decision == GateDecision.ACT

    def test_gather_when_global_mid(self):
        gate  = UncertaintyGate(GateConfig(act_threshold=0.7, gather_threshold=0.4,
                                           require_depth=False))
        scene = _make_scene(global_score=0.55, n=1, obj_combined=0.55)
        assert gate.evaluate(scene).decision == GateDecision.GATHER

    def test_flag_when_global_low(self):
        gate  = UncertaintyGate(GateConfig(gather_threshold=0.4, require_depth=False))
        scene = _make_scene(global_score=0.2, n=1, obj_combined=0.2)
        assert gate.evaluate(scene).decision == GateDecision.FLAG

    def test_object_flag_overrides_act(self):
        gate = UncertaintyGate(GateConfig(act_threshold=0.7, require_depth=True))
        obj  = _make_obj(combined=0.95, depth_score=0.0)    # FLAG: no depth
        scene = SceneConfidence(objects=[obj], global_score=0.95, n_objects=1)
        assert gate.evaluate(scene).decision == GateDecision.FLAG

    def test_object_gather_overrides_act(self):
        gate = UncertaintyGate(GateConfig(act_threshold=0.7, gather_threshold=0.4,
                                          require_depth=False))
        obj  = _make_obj(combined=0.5)   # GATHER
        scene = SceneConfidence(objects=[obj], global_score=0.95, n_objects=1)
        assert gate.evaluate(scene).decision == GateDecision.GATHER

    def test_multi_object_worst_wins(self):
        gate = UncertaintyGate(GateConfig(act_threshold=0.7, require_depth=True))
        objs = [
            _make_obj(combined=0.9, depth_score=0.8),   # ACT
            _make_obj(combined=0.9, depth_score=0.0),   # FLAG
        ]
        scene = SceneConfidence(objects=objs, global_score=0.9, n_objects=2)
        assert gate.evaluate(scene).decision == GateDecision.FLAG

    def test_all_objects_act_scene_acts(self):
        gate  = UncertaintyGate(GateConfig(act_threshold=0.7, require_depth=False))
        objs  = [_make_obj(combined=0.9) for _ in range(3)]
        scene = SceneConfidence(objects=objs, global_score=0.9, n_objects=3)
        assert gate.evaluate(scene).decision == GateDecision.ACT

    def test_returns_gate_result(self):
        gate  = UncertaintyGate(GateConfig(require_depth=False))
        scene = _make_scene(global_score=0.8)
        assert isinstance(gate.evaluate(scene), GateResult)

    def test_score_equals_global(self):
        gate  = UncertaintyGate(GateConfig(require_depth=False))
        scene = _make_scene(global_score=0.73)
        r     = gate.evaluate(scene)
        assert r.score == pytest.approx(0.73)

    def test_object_decisions_count(self):
        gate  = UncertaintyGate(GateConfig(require_depth=False))
        scene = _make_scene(global_score=0.8, n=4)
        r     = gate.evaluate(scene)
        assert len(r.object_decisions) == 4

    def test_reason_contains_global_score(self):
        gate  = UncertaintyGate(GateConfig(require_depth=False))
        scene = _make_scene(global_score=0.85)
        r     = gate.evaluate(scene)
        assert "0.850" in r.reason or "global_score" in r.reason

    def test_reason_mentions_worst_object_on_override(self):
        gate = UncertaintyGate(GateConfig(act_threshold=0.7, require_depth=True))
        obj  = _make_obj(combined=0.9, depth_score=0.0)
        scene = SceneConfidence(objects=[obj], global_score=0.9, n_objects=1)
        r     = gate.evaluate(scene)
        assert "flag" in r.reason or "worst_object" in r.reason

    def test_object_decisions_are_tuples(self):
        gate  = UncertaintyGate(GateConfig(require_depth=False))
        scene = _make_scene(global_score=0.8)
        r     = gate.evaluate(scene)
        dec, gd = r.object_decisions[0]
        assert isinstance(dec, Detection)
        assert isinstance(gd, GateDecision)

    def test_repr(self):
        gate = UncertaintyGate()
        assert "0.7" in repr(gate) or "act" in repr(gate).lower()

    def test_skip_score_is_zero(self):
        gate  = UncertaintyGate()
        scene = SceneConfidence(objects=[], global_score=0.0, n_objects=0)
        r     = gate.evaluate(scene)
        assert r.score == pytest.approx(0.0)

    def test_min_objects_one_with_one_object_not_skip(self):
        gate  = UncertaintyGate(GateConfig(min_objects=1, require_depth=False))
        scene = _make_scene(global_score=0.8, n=1)
        assert gate.evaluate(scene).decision != GateDecision.SKIP
