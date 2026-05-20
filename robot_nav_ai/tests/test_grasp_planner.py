"""
tests/test_grasp_planner.py — Unit tests for GraspPlanner, PlannerConfig,
and GraspCandidate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.grasp_planner import GraspCandidate, GraspPlanner, PlannerConfig


# ── fixtures ──────────────────────────────────────────────────────────────────

ROBOT_POS  = np.array([0.0, 0.0, 0.0],  dtype=np.float64)
ROBOT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)   # identity
OBJ_POS    = np.array([0.5, 0.0, 0.05], dtype=np.float64)        # on table in front


def _planner(cfg: PlannerConfig = PlannerConfig()) -> GraspPlanner:
    return GraspPlanner(cfg=cfg)


def _cloud(n: int = 50, center=OBJ_POS, spread: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (center + rng.normal(0, spread, size=(n, 3))).astype(np.float64)


# ── PlannerConfig ─────────────────────────────────────────────────────────────

class TestPlannerConfig:
    def test_defaults(self):
        cfg = PlannerConfig()
        assert cfg.approach_dist == pytest.approx(0.12)
        assert cfg.top_k         == 5
        assert cfg.min_points    == 10
        assert cfg.n_side_rotations == 4

    def test_frozen(self):
        with pytest.raises(Exception):
            PlannerConfig().top_k = 3

    def test_custom(self):
        cfg = PlannerConfig(top_k=3, approach_dist=0.08)
        assert cfg.top_k == 3
        assert cfg.approach_dist == pytest.approx(0.08)


# ── GraspCandidate ────────────────────────────────────────────────────────────

class TestGraspCandidate:
    def _make(self) -> GraspCandidate:
        return GraspCandidate(
            ee_pos       = np.array([0.5, 0.0, 0.2], dtype=np.float32),
            approach_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32),
            score        = 0.85,
            method       = "top_down",
        )

    def test_fields(self):
        c = self._make()
        assert c.score   == pytest.approx(0.85)
        assert c.method  == "top_down"
        assert c.reachable

    def test_repr_contains_method(self):
        assert "top_down" in repr(self._make())

    def test_repr_contains_score(self):
        assert "0.850" in repr(self._make())

    def test_ee_pos_shape(self):
        assert self._make().ee_pos.shape == (3,)

    def test_approach_vec_shape(self):
        assert self._make().approach_vec.shape == (3,)


# ── GraspPlanner — construction ───────────────────────────────────────────────

class TestConstruction:
    def test_repr(self):
        assert "GraspPlanner" in repr(_planner())

    def test_default_limits(self):
        p = _planner()
        assert p.limits is not None


# ── GraspPlanner.plan — basic output ─────────────────────────────────────────

class TestPlanBasic:
    def test_returns_list(self):
        result = _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT)
        assert isinstance(result, list)

    def test_all_reachable(self):
        result = _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT)
        assert all(c.reachable for c in result)

    def test_sorted_descending(self):
        result = _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_at_most_top_k(self):
        cfg    = PlannerConfig(top_k=3)
        result = _planner(cfg).plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT)
        assert len(result) <= 3

    def test_candidates_have_correct_types(self):
        for c in _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT):
            assert isinstance(c, GraspCandidate)
            assert c.ee_pos.dtype       == np.float32
            assert c.approach_vec.dtype == np.float32

    def test_methods_are_valid(self):
        valid = {"top_down", "side_axis", "diagonal"}
        for c in _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT):
            assert c.method in valid

    def test_scores_in_unit_range(self):
        for c in _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT):
            assert 0.0 <= c.score <= 1.0 + 1e-6

    def test_with_point_cloud(self):
        cloud  = _cloud()
        result = _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT, point_cloud=cloud)
        assert len(result) > 0

    def test_without_point_cloud(self):
        result = _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT, point_cloud=None)
        assert len(result) > 0

    def test_unreachable_object_returns_empty(self):
        far_obj = np.array([10.0, 10.0, 0.05])
        result  = _planner().plan(far_obj, ROBOT_POS, ROBOT_QUAT)
        assert len(result) == 0


# ── GraspPlanner.plan — candidate families ────────────────────────────────────

class TestCandidateFamilies:
    def test_top_down_present(self):
        methods = {c.method for c in _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT)}
        assert "top_down" in methods

    def test_side_axis_present(self):
        methods = {c.method for c in _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT)}
        assert "side_axis" in methods

    def test_diagonal_present(self):
        methods = {c.method for c in _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT)}
        assert "diagonal" in methods

    def test_top_down_approach_vec_points_down(self):
        for c in _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT):
            if c.method == "top_down":
                assert c.approach_vec[2] < 0.0   # Z component negative → downward

    def test_top_down_ee_above_object(self):
        for c in _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT):
            if c.method == "top_down":
                assert c.ee_pos[2] > OBJ_POS[2]

    def test_side_axis_count(self):
        cfg    = PlannerConfig(top_k=20, n_side_rotations=4)
        result = _planner(cfg).plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT)
        side   = [c for c in result if c.method == "side_axis"]
        assert len(side) <= 4   # at most n_side_rotations (some may be unreachable)


# ── GraspPlanner — top-down geometry ─────────────────────────────────────────

class TestTopDownGeometry:
    def test_ee_xy_matches_object_xy(self):
        for c in _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT):
            if c.method == "top_down":
                assert c.ee_pos[0] == pytest.approx(OBJ_POS[0], abs=1e-5)
                assert c.ee_pos[1] == pytest.approx(OBJ_POS[1], abs=1e-5)

    def test_ee_z_offset_equals_approach_dist(self):
        cfg = PlannerConfig(approach_dist=0.10)
        for c in _planner(cfg).plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT):
            if c.method == "top_down":
                z_offset = float(c.ee_pos[2]) - float(OBJ_POS[2])
                assert z_offset == pytest.approx(0.10, abs=1e-5)

    def test_top_down_approach_vec_is_unit(self):
        for c in _planner().plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT):
            if c.method == "top_down":
                assert np.linalg.norm(c.approach_vec) == pytest.approx(1.0, abs=1e-5)


# ── GraspPlanner.best ─────────────────────────────────────────────────────────

class TestBest:
    def test_returns_candidate_or_none(self):
        b = _planner().best(OBJ_POS, ROBOT_POS, ROBOT_QUAT)
        assert b is None or isinstance(b, GraspCandidate)

    def test_returns_highest_score(self):
        planner = _planner()
        b       = planner.best(OBJ_POS, ROBOT_POS, ROBOT_QUAT)
        all_c   = planner.plan(OBJ_POS, ROBOT_POS, ROBOT_QUAT)
        if b is not None and all_c:
            assert b.score == pytest.approx(all_c[0].score)

    def test_unreachable_returns_none(self):
        b = _planner().best(np.array([99.0, 99.0, 0.0]), ROBOT_POS, ROBOT_QUAT)
        assert b is None


# ── GraspPlanner.plan_from_detection ─────────────────────────────────────────

class TestPlanFromDetection:
    def test_no_detection_no_error(self):
        result = _planner().plan_from_detection(OBJ_POS, ROBOT_POS, ROBOT_QUAT)
        assert isinstance(result, list)

    def test_with_mock_projection(self):
        class MockProj:
            xyz = OBJ_POS.astype(np.float32)

        result = _planner().plan_from_detection(
            OBJ_POS, ROBOT_POS, ROBOT_QUAT, projection=MockProj()
        )
        assert isinstance(result, list)

    def test_with_mock_detection_mask(self):
        class MockDet:
            mask = np.ones((100, 100), dtype=bool)

        result = _planner().plan_from_detection(
            OBJ_POS, ROBOT_POS, ROBOT_QUAT, detection=MockDet()
        )
        assert isinstance(result, list)


# ── principal axes ────────────────────────────────────────────────────────────

class TestPrincipalAxes:
    def test_identity_when_no_cloud(self):
        p    = _planner()
        axes = p._principal_axes(None, OBJ_POS)
        assert axes.shape == (3, 3)
        assert np.allclose(axes, np.eye(3), atol=1e-6)

    def test_identity_when_too_few_points(self):
        p    = _planner()
        axes = p._principal_axes(np.zeros((3, 3)), OBJ_POS)
        assert np.allclose(axes, np.eye(3), atol=1e-6)

    def test_pca_axes_orthogonal(self):
        cloud = _cloud(n=100)
        axes  = _planner()._principal_axes(cloud, OBJ_POS)
        assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-6)

    def test_pca_axes_shape(self):
        cloud = _cloud(n=50)
        axes  = _planner()._principal_axes(cloud, OBJ_POS)
        assert axes.shape == (3, 3)


# ── object radius ─────────────────────────────────────────────────────────────

class TestObjectRadius:
    def test_default_when_no_cloud(self):
        r = _planner()._object_radius(None, OBJ_POS)
        assert r == pytest.approx(0.05)

    def test_radius_positive(self):
        cloud = _cloud(n=50)
        r     = _planner()._object_radius(cloud, OBJ_POS)
        assert r > 0.0

    def test_larger_cloud_larger_radius(self):
        small = _cloud(n=20, spread=0.01)
        big   = _cloud(n=20, spread=0.1)
        r_small = _planner()._object_radius(small, OBJ_POS)
        r_big   = _planner()._object_radius(big,   OBJ_POS)
        assert r_big > r_small
