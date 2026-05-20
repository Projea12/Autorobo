"""
tests/test_confidence.py — Unit tests for SceneAggregator, AggregatorConfig,
ObjectConfidence, and SceneConfidence.

All tests use synthetic Detection and ProjectionResult objects — no MuJoCo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from perception.detector import Detection
from perception.depth_projector import ProjectionResult
from perception.confidence import (
    AggregatorConfig, ObjectConfidence, SceneAggregator, SceneConfidence,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_det(conf: float = 0.9, mask: np.ndarray = None) -> Detection:
    det = Detection(
        class_id   = 0,
        class_name = "mug",
        confidence = conf,
        bbox_xyxy  = np.array([10, 10, 60, 50], dtype=np.float32),
        # bbox_xywh → cx=35, cy=30, w=50, h=40 → area=2000
        bbox_xywh  = np.array([35, 30, 50, 40], dtype=np.float32),
    )
    if mask is not None:
        det.mask = mask
    return det


def _make_proj(n_points: int = 200, std_z: float = 0.05) -> ProjectionResult:
    return ProjectionResult(
        xyz      = np.array([0.0, 0.0, 2.0], dtype=np.float32),
        std      = np.array([0.01, 0.01, std_z], dtype=np.float32),
        n_points = n_points,
        method   = "bbox_median",
    )


def _make_obj(det_score: float = 0.8, depth_score: float = 0.7,
              seg_score: float = 0.6, combined: float = 0.72) -> ObjectConfidence:
    return ObjectConfidence(
        detection_score = det_score,
        depth_score     = depth_score,
        seg_score       = seg_score,
        combined        = combined,
        detection       = _make_det(conf=det_score),
    )


# ── AggregatorConfig ──────────────────────────────────────────────────────────

class TestAggregatorConfig:
    def test_defaults(self):
        cfg = AggregatorConfig()
        assert cfg.w_detection     == pytest.approx(0.4)
        assert cfg.w_depth         == pytest.approx(0.4)
        assert cfg.w_seg           == pytest.approx(0.2)
        assert cfg.depth_n_pts_max == 500
        assert cfg.depth_std_max   == pytest.approx(0.5)

    def test_frozen(self):
        cfg = AggregatorConfig()
        with pytest.raises(Exception):
            cfg.w_detection = 0.5

    def test_custom(self):
        cfg = AggregatorConfig(w_detection=0.5, w_depth=0.3, w_seg=0.2,
                               depth_n_pts_max=1000, depth_std_max=1.0)
        assert cfg.w_detection     == pytest.approx(0.5)
        assert cfg.depth_n_pts_max == 1000
        assert cfg.depth_std_max   == pytest.approx(1.0)


# ── ObjectConfidence ──────────────────────────────────────────────────────────

class TestObjectConfidence:
    def test_repr_contains_class_name(self):
        obj = _make_obj()
        assert "mug" in repr(obj)

    def test_repr_contains_combined(self):
        obj = _make_obj(combined=0.72)
        assert "0.720" in repr(obj)

    def test_fields_stored(self):
        obj = _make_obj(det_score=0.9, depth_score=0.8, seg_score=0.5, combined=0.77)
        assert obj.detection_score == pytest.approx(0.9)
        assert obj.depth_score     == pytest.approx(0.8)
        assert obj.seg_score       == pytest.approx(0.5)
        assert obj.combined        == pytest.approx(0.77)

    def test_projection_none_by_default(self):
        obj = _make_obj()
        assert obj.projection is None

    def test_projection_stored(self):
        proj = _make_proj()
        obj  = ObjectConfidence(
            detection_score=0.8, depth_score=0.7, seg_score=0.6, combined=0.73,
            detection=_make_det(), projection=proj,
        )
        assert obj.projection is proj


# ── SceneConfidence ───────────────────────────────────────────────────────────

class TestSceneConfidence:
    def test_repr_contains_global_score(self):
        sc = SceneConfidence(objects=[], global_score=0.72, n_objects=0)
        assert "0.720" in repr(sc)

    def test_repr_contains_n_objects(self):
        sc = SceneConfidence(objects=[], global_score=0.5, n_objects=3)
        assert "3" in repr(sc)

    def test_fields_stored(self):
        obj = _make_obj()
        sc  = SceneConfidence(objects=[obj], global_score=0.72, n_objects=1)
        assert sc.n_objects    == 1
        assert sc.global_score == pytest.approx(0.72)
        assert sc.objects[0] is obj


# ── _depth_score ──────────────────────────────────────────────────────────────

class TestDepthScore:
    def _agg(self, **kw):
        return SceneAggregator(AggregatorConfig(depth_n_pts_max=100,
                                                depth_std_max=0.5, **kw))

    def test_none_proj_returns_zero(self):
        assert self._agg()._depth_score(None) == pytest.approx(0.0)

    def test_zero_points_returns_zero(self):
        proj = _make_proj(n_points=0, std_z=0.0)
        assert self._agg()._depth_score(proj) == pytest.approx(0.0)

    def test_full_coverage_zero_std_is_one(self):
        # coverage=1.0, reliability=1.0 → 0.5+0.5=1.0
        proj = _make_proj(n_points=100, std_z=0.0)
        assert self._agg()._depth_score(proj) == pytest.approx(1.0)

    def test_half_coverage_zero_std(self):
        # coverage=0.5, reliability=1.0 → 0.25+0.50=0.75
        proj = _make_proj(n_points=50, std_z=0.0)
        assert self._agg()._depth_score(proj) == pytest.approx(0.75)

    def test_full_coverage_max_std(self):
        # coverage=1.0, reliability=0.0 → 0.50+0.0=0.50
        proj = _make_proj(n_points=100, std_z=0.5)
        assert self._agg()._depth_score(proj) == pytest.approx(0.50)

    def test_half_coverage_max_std(self):
        # coverage=0.5, reliability=0.0 → 0.25+0.0=0.25
        proj = _make_proj(n_points=50, std_z=0.5)
        assert self._agg()._depth_score(proj) == pytest.approx(0.25)

    def test_capped_at_one(self):
        proj = _make_proj(n_points=10_000, std_z=0.0)
        assert self._agg()._depth_score(proj) <= 1.0

    def test_nonnegative(self):
        proj = _make_proj(n_points=1, std_z=10.0)
        assert self._agg()._depth_score(proj) >= 0.0

    def test_std_above_max_capped(self):
        # std_z >> std_max → reliability = 0.0
        proj = _make_proj(n_points=100, std_z=100.0)
        # coverage=1.0, reliability=0.0 → 0.5
        assert self._agg()._depth_score(proj) == pytest.approx(0.5)

    def test_quarter_coverage_half_std(self):
        # coverage=0.25, std_z=0.25 → reliability=0.5 → 0.125+0.25=0.375
        proj = _make_proj(n_points=25, std_z=0.25)
        assert self._agg()._depth_score(proj) == pytest.approx(0.375)


# ── _seg_score ────────────────────────────────────────────────────────────────

class TestSegScore:
    # det has bbox area = 50 × 40 = 2000 pixels
    _H, _W = 100, 100

    def test_no_mask_returns_zero(self):
        assert SceneAggregator()._seg_score(_make_det(mask=None)) == pytest.approx(0.0)

    def test_full_mask_equals_one(self):
        mask = np.zeros((self._H, self._W), dtype=bool)
        mask[10:50, 10:60] = True   # 40×50 = 2000 = bbox area
        assert SceneAggregator()._seg_score(_make_det(mask=mask)) == pytest.approx(1.0)

    def test_half_mask(self):
        mask = np.zeros((self._H, self._W), dtype=bool)
        mask[10:50, 10:35] = True   # 40×25 = 1000 = half
        assert SceneAggregator()._seg_score(_make_det(mask=mask)) == pytest.approx(0.5)

    def test_oversized_mask_capped_at_one(self):
        mask = np.ones((self._H, self._W), dtype=bool)   # 10 000 >> 2000
        assert SceneAggregator()._seg_score(_make_det(mask=mask)) <= 1.0

    def test_empty_mask_returns_zero(self):
        mask = np.zeros((self._H, self._W), dtype=bool)
        assert SceneAggregator()._seg_score(_make_det(mask=mask)) == pytest.approx(0.0)

    def test_score_in_range(self):
        mask = np.zeros((self._H, self._W), dtype=bool)
        mask[10:30, 10:30] = True
        s = SceneAggregator()._seg_score(_make_det(mask=mask))
        assert 0.0 <= s <= 1.0

    def test_quarter_mask(self):
        mask = np.zeros((self._H, self._W), dtype=bool)
        mask[10:30, 10:22] = True    # 20×12 = 240 → 240/2000 = 0.12
        s = SceneAggregator()._seg_score(_make_det(mask=mask))
        assert s == pytest.approx(240.0 / 2000.0, rel=1e-3)


# ── _combined ─────────────────────────────────────────────────────────────────

class TestCombined:
    def test_all_ones(self):
        assert SceneAggregator()._combined(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_all_zeros(self):
        assert SceneAggregator()._combined(0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_default_weights(self):
        # 0.4×0.5 + 0.4×0.8 + 0.2×1.0 = 0.20 + 0.32 + 0.20 = 0.72
        assert SceneAggregator()._combined(0.5, 0.8, 1.0) == pytest.approx(0.72)

    def test_det_only_weight(self):
        cfg = AggregatorConfig(w_detection=1.0, w_depth=0.0, w_seg=0.0)
        assert SceneAggregator(cfg)._combined(0.7, 0.0, 0.0) == pytest.approx(0.7)

    def test_depth_only_weight(self):
        cfg = AggregatorConfig(w_detection=0.0, w_depth=1.0, w_seg=0.0)
        assert SceneAggregator(cfg)._combined(0.0, 0.6, 0.0) == pytest.approx(0.6)

    def test_seg_only_weight(self):
        cfg = AggregatorConfig(w_detection=0.0, w_depth=0.0, w_seg=1.0)
        assert SceneAggregator(cfg)._combined(0.0, 0.0, 0.9) == pytest.approx(0.9)

    def test_normalised_when_weights_not_unit(self):
        cfg = AggregatorConfig(w_detection=2.0, w_depth=2.0, w_seg=1.0)
        # (2×0.6 + 2×0.6 + 1×0.6) / 5 = 3.0/5 = 0.6
        assert SceneAggregator(cfg)._combined(0.6, 0.6, 0.6) == pytest.approx(0.6)

    def test_clamped_above_one(self):
        assert SceneAggregator()._combined(1.5, 1.5, 1.5) == pytest.approx(1.0)

    def test_clamped_below_zero(self):
        assert SceneAggregator()._combined(-1.0, -1.0, -1.0) == pytest.approx(0.0)

    def test_asymmetric_weights(self):
        cfg = AggregatorConfig(w_detection=0.6, w_depth=0.3, w_seg=0.1)
        # (0.6×1.0 + 0.3×0.0 + 0.1×0.0)/1.0 = 0.6
        assert SceneAggregator(cfg)._combined(1.0, 0.0, 0.0) == pytest.approx(0.6)


# ── SceneAggregator.aggregate ─────────────────────────────────────────────────

class TestAggregate:
    def test_returns_scene_confidence(self):
        scene = SceneAggregator().aggregate([_make_det()])
        assert isinstance(scene, SceneConfidence)

    def test_empty_detections_zero_score(self):
        scene = SceneAggregator().aggregate([])
        assert scene.global_score == pytest.approx(0.0)
        assert scene.n_objects    == 0

    def test_empty_detections_empty_objects(self):
        scene = SceneAggregator().aggregate([])
        assert scene.objects == []

    def test_n_objects_matches(self):
        dets  = [_make_det(), _make_det(conf=0.6)]
        scene = SceneAggregator().aggregate(dets)
        assert scene.n_objects == 2

    def test_global_score_mean_of_combined(self):
        # Use detection-only weights so combined == det confidence
        cfg  = AggregatorConfig(w_detection=1.0, w_depth=0.0, w_seg=0.0)
        agg  = SceneAggregator(cfg)
        dets = [_make_det(conf=0.8), _make_det(conf=0.6)]
        scene = agg.aggregate(dets)
        # global = mean(0.8, 0.6) = 0.7
        assert scene.global_score == pytest.approx(0.7)

    def test_single_object_global_equals_combined(self):
        cfg  = AggregatorConfig(w_detection=1.0, w_depth=0.0, w_seg=0.0)
        agg  = SceneAggregator(cfg)
        scene = agg.aggregate([_make_det(conf=0.85)])
        assert scene.global_score == pytest.approx(0.85)

    def test_without_projections_depth_score_zero(self):
        scene = SceneAggregator().aggregate([_make_det()])
        assert scene.objects[0].depth_score == pytest.approx(0.0)

    def test_with_projection_depth_score_nonzero(self):
        proj  = _make_proj(n_points=500, std_z=0.0)
        scene = SceneAggregator().aggregate([_make_det()], [proj])
        assert scene.objects[0].depth_score > 0.0

    def test_projection_stored_on_object(self):
        proj  = _make_proj()
        scene = SceneAggregator().aggregate([_make_det()], [proj])
        assert scene.objects[0].projection is proj

    def test_no_mask_seg_score_zero(self):
        scene = SceneAggregator().aggregate([_make_det(mask=None)])
        assert scene.objects[0].seg_score == pytest.approx(0.0)

    def test_mask_seg_score_nonzero(self):
        mask = np.ones((100, 100), dtype=bool)
        scene = SceneAggregator().aggregate([_make_det(mask=mask)])
        assert scene.objects[0].seg_score > 0.0

    def test_mismatched_projections_raises(self):
        dets  = [_make_det(), _make_det()]
        projs = [_make_proj()]
        with pytest.raises(ValueError, match="len"):
            SceneAggregator().aggregate(dets, projs)

    def test_combined_in_unit_range(self):
        mask  = np.zeros((100, 100), dtype=bool)
        mask[10:50, 10:60] = True
        scene = SceneAggregator().aggregate(
            [_make_det(conf=0.9, mask=mask)], [_make_proj(n_points=300)]
        )
        c = scene.objects[0].combined
        assert 0.0 <= c <= 1.0

    def test_detection_score_clipped(self):
        det   = _make_det(conf=1.5)   # confidence > 1
        scene = SceneAggregator().aggregate([det])
        assert scene.objects[0].detection_score <= 1.0

    def test_repr_contains_global_score(self):
        scene = SceneAggregator().aggregate([_make_det(conf=0.8)])
        assert "global=" in repr(scene)
