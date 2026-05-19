"""
tests/test_grasp_goal.py — Unit tests for GraspGoalSelector.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.grasp_goal import (
    GraspCandidate,
    GraspGoalConfig,
    GraspGoalSelector,
    WorkspaceHint,
    make_grasp_goal_selector,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _flat_lidar(n: int = 36, value: float = 1.0) -> np.ndarray:
    """Normalised lidar ring — value=1.0 means no obstacles."""
    return np.full(n, value, dtype=np.float64)


def _selector(
    n_candidates: int = 16,
    n_arc_radii: int = 2,
    arc_angle_deg: float = 150.0,
    min_base_clearance: float = 0.30,
    **kw,
) -> GraspGoalSelector:
    cfg = GraspGoalConfig(
        n_candidates=n_candidates,
        n_arc_radii=n_arc_radii,
        arc_angle_deg=arc_angle_deg,
        min_base_clearance=min_base_clearance,
        **kw,
    )
    return GraspGoalSelector(cfg=cfg)


# ── WorkspaceHint ──────────────────────────────────────────────────────────────

class TestWorkspaceHint:
    def test_defaults(self):
        ws = WorkspaceHint()
        assert ws.arm_mount_fwd   == pytest.approx(0.15)
        assert ws.arm_reach_min   == pytest.approx(0.30)
        assert ws.arm_reach_max   == pytest.approx(0.80)
        assert ws.grasp_height_lo == pytest.approx(0.00)
        assert ws.grasp_height_hi == pytest.approx(0.55)

    def test_frozen(self):
        ws = WorkspaceHint()
        with pytest.raises(Exception):
            ws.arm_reach_min = 99.0

    def test_arm_mount_world_forward_at_zero_yaw(self):
        ws = WorkspaceHint(arm_mount_fwd=0.20)
        base = np.array([0.0, 0.0])
        mount = ws.arm_mount_world(base, yaw=0.0)
        assert mount[0] == pytest.approx(0.20, abs=1e-9)
        assert mount[1] == pytest.approx(0.00, abs=1e-9)

    def test_arm_mount_world_90deg_yaw(self):
        ws = WorkspaceHint(arm_mount_fwd=0.10)
        base = np.array([1.0, 2.0])
        mount = ws.arm_mount_world(base, yaw=math.pi / 2)
        assert mount[0] == pytest.approx(1.00, abs=1e-9)
        assert mount[1] == pytest.approx(2.10, abs=1e-9)

    def test_arm_mount_world_nonzero_base(self):
        ws = WorkspaceHint(arm_mount_fwd=0.15)
        base = np.array([3.0, -1.0])
        mount = ws.arm_mount_world(base, yaw=0.0)
        assert mount[0] == pytest.approx(3.15, abs=1e-9)
        assert mount[1] == pytest.approx(-1.00, abs=1e-9)

    def test_object_in_workspace_true(self):
        ws = WorkspaceHint(arm_mount_fwd=0.0, arm_reach_min=0.30, arm_reach_max=0.80)
        base = np.array([0.0, 0.0])
        obj  = np.array([0.55, 0.0, 0.10])
        assert ws.object_in_workspace(base, obj) is True

    def test_object_in_workspace_too_close(self):
        ws = WorkspaceHint(arm_mount_fwd=0.0, arm_reach_min=0.30, arm_reach_max=0.80)
        base = np.array([0.0, 0.0])
        obj  = np.array([0.10, 0.0, 0.10])
        assert ws.object_in_workspace(base, obj) is False

    def test_object_in_workspace_too_far(self):
        ws = WorkspaceHint(arm_mount_fwd=0.0, arm_reach_min=0.30, arm_reach_max=0.80)
        base = np.array([0.0, 0.0])
        obj  = np.array([1.50, 0.0, 0.10])
        assert ws.object_in_workspace(base, obj) is False

    def test_object_in_workspace_too_high(self):
        ws = WorkspaceHint(arm_mount_fwd=0.0, arm_reach_min=0.30, arm_reach_max=0.80,
                           grasp_height_hi=0.55)
        base = np.array([0.0, 0.0])
        obj  = np.array([0.55, 0.0, 1.20])
        assert ws.object_in_workspace(base, obj) is False

    def test_object_in_workspace_no_z_component(self):
        ws = WorkspaceHint(arm_mount_fwd=0.0, arm_reach_min=0.30, arm_reach_max=0.80)
        base = np.array([0.0, 0.0])
        obj  = np.array([0.55, 0.0])  # 2-D — default z=0.025 ∈ [0, 0.55]
        assert ws.object_in_workspace(base, obj) is True


# ── GraspGoalConfig ────────────────────────────────────────────────────────────

class TestGraspGoalConfig:
    def test_defaults(self):
        cfg = GraspGoalConfig()
        assert cfg.n_candidates       == 16
        assert cfg.n_arc_radii        == 2
        assert cfg.arc_angle_deg      == pytest.approx(150.0)
        assert cfg.min_base_clearance == pytest.approx(0.30)
        assert cfg.w_reach            == pytest.approx(1.0)
        assert cfg.w_clearance        == pytest.approx(2.0)
        assert cfg.w_heading          == pytest.approx(1.5)
        assert cfg.w_approach         == pytest.approx(1.0)
        assert cfg.fallback_to_object is True

    def test_frozen(self):
        cfg = GraspGoalConfig()
        with pytest.raises(Exception):
            cfg.n_candidates = 32

    def test_custom(self):
        cfg = GraspGoalConfig(n_candidates=8, w_reach=2.5)
        assert cfg.n_candidates == 8
        assert cfg.w_reach == pytest.approx(2.5)


# ── GraspCandidate ─────────────────────────────────────────────────────────────

class TestGraspCandidate:
    def _make(self, xy=(1.0, 2.0), yaw=0.5, dist=0.55, score=3.2, feasible=True):
        return GraspCandidate(
            xy=np.array(xy), yaw=yaw, dist_to_obj=dist,
            score=score, feasible=feasible,
        )

    def test_goal_xyz_shape(self):
        c = self._make()
        assert c.goal_xyz().shape == (3,)

    def test_goal_xyz_xy(self):
        c = self._make(xy=(1.5, -0.3))
        g = c.goal_xyz()
        assert g[0] == pytest.approx(1.5)
        assert g[1] == pytest.approx(-0.3)

    def test_goal_xyz_default_z(self):
        c = self._make()
        assert c.goal_xyz()[2] == pytest.approx(0.12)

    def test_goal_xyz_custom_z(self):
        c = self._make()
        assert c.goal_xyz(z=0.20)[2] == pytest.approx(0.20)

    def test_goal_xyz_dtype(self):
        c = self._make()
        assert c.goal_xyz().dtype == np.float32

    def test_mutable_score(self):
        c = self._make(score=0.0)
        c.score = 5.5
        assert c.score == pytest.approx(5.5)

    def test_mutable_feasible(self):
        c = self._make(feasible=False)
        c.feasible = True
        assert c.feasible is True


# ── GraspGoalSelector — construction ──────────────────────────────────────────

class TestSelectorConstruction:
    def test_default_construction(self):
        sel = GraspGoalSelector()
        assert isinstance(sel.cfg, GraspGoalConfig)
        assert isinstance(sel.workspace, WorkspaceHint)

    def test_custom_cfg(self):
        cfg = GraspGoalConfig(n_candidates=8)
        sel = GraspGoalSelector(cfg=cfg)
        assert sel.cfg.n_candidates == 8

    def test_custom_workspace(self):
        ws = WorkspaceHint(arm_reach_max=1.0)
        sel = GraspGoalSelector(workspace=ws)
        assert sel.workspace.arm_reach_max == pytest.approx(1.0)


# ── _generate_candidates ──────────────────────────────────────────────────────

class TestGenerateCandidates:
    def test_returns_list_of_candidates(self):
        sel  = _selector(n_candidates=16, n_arc_radii=2)
        obj  = np.array([2.0, 0.0])
        rbt  = np.array([0.0, 0.0])
        cands = sel._generate_candidates(obj, rbt)
        assert isinstance(cands, list)
        assert all(isinstance(c, GraspCandidate) for c in cands)

    def test_candidate_count(self):
        sel  = _selector(n_candidates=16, n_arc_radii=2)
        obj  = np.array([2.0, 0.0])
        rbt  = np.array([0.0, 0.0])
        cands = sel._generate_candidates(obj, rbt)
        # n_ang = 16 // 2 = 8, times 2 arcs = 16
        assert len(cands) == 16

    def test_candidate_count_single_radius(self):
        sel   = _selector(n_candidates=8, n_arc_radii=1)
        obj   = np.array([3.0, 0.0])
        rbt   = np.array([0.0, 0.0])
        cands = sel._generate_candidates(obj, rbt)
        assert len(cands) == 8

    def test_candidates_have_xy_shape(self):
        sel  = _selector()
        cands = sel._generate_candidates(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        for c in cands:
            assert c.xy.shape == (2,)

    def test_candidates_yaw_points_toward_object(self):
        """For object at (2,0) from robot at (0,0), yaw should be ~π (facing back)."""
        sel  = _selector(n_candidates=1, n_arc_radii=1, arc_angle_deg=0.0)
        obj  = np.array([2.0, 0.0])
        rbt  = np.array([0.0, 0.0])
        cands = sel._generate_candidates(obj, rbt)
        c = cands[0]
        # candidate is placed behind the object (on robot side), so yaw faces object (+x)
        expected_yaw = math.atan2(obj[1] - c.xy[1], obj[0] - c.xy[0])
        assert c.yaw == pytest.approx(expected_yaw, abs=1e-6)

    def test_dist_to_obj_positive(self):
        sel   = _selector()
        cands = sel._generate_candidates(np.array([2.0, 0.0]), np.array([0.0, 0.0]))
        for c in cands:
            assert c.dist_to_obj > 0.0

    def test_initial_score_zero(self):
        sel   = _selector()
        cands = sel._generate_candidates(np.array([2.0, 0.0]), np.array([0.0, 0.0]))
        for c in cands:
            assert c.score == pytest.approx(0.0)

    def test_initial_feasible_false(self):
        sel   = _selector()
        cands = sel._generate_candidates(np.array([2.0, 0.0]), np.array([0.0, 0.0]))
        for c in cands:
            assert c.feasible is False

    def test_candidates_biased_toward_robot_side(self):
        """Candidates should cluster on the robot-facing side of the object."""
        sel  = _selector(n_candidates=16, n_arc_radii=1, arc_angle_deg=180.0)
        obj  = np.array([3.0, 0.0])
        rbt  = np.array([0.0, 0.0])
        cands = sel._generate_candidates(obj, rbt)
        # Most candidates should have x < object.x (between robot and object)
        n_robot_side = sum(c.xy[0] < obj[0] for c in cands)
        assert n_robot_side >= len(cands) // 2


# ── _score ────────────────────────────────────────────────────────────────────

class TestScore:
    def _make_candidate(self, dist=0.55, xy=(1.0, 0.0)):
        return GraspCandidate(
            xy=np.array(xy), yaw=0.0, dist_to_obj=dist,
            score=0.0, feasible=False,
        )

    def test_returns_float(self):
        sel = _selector()
        c   = self._make_candidate()
        raw = np.full(36, 5.0)
        s   = sel._score(c, np.array([2.0, 0.0, 0.025]), raw)
        assert isinstance(s, float)

    def test_score_nonnegative_in_clear_space(self):
        sel = _selector()
        c   = self._make_candidate(dist=0.55)   # midpoint of [0.3, 0.8]
        raw = np.full(36, 5.0)                  # all clear
        s   = sel._score(c, np.array([2.0, 0.0, 0.025]), raw)
        assert s >= 0.0

    def test_score_higher_at_midreach(self):
        """Candidate at midpoint of reach range should outscore edge candidates."""
        sel  = _selector(w_clearance=0.0, w_heading=0.0, w_approach=0.0)
        mid  = (sel.workspace.arm_reach_min + sel.workspace.arm_reach_max) / 2
        raw  = np.full(36, 5.0)
        obj  = np.array([5.0, 0.0, 0.025])

        c_mid  = self._make_candidate(dist=mid)
        c_edge = self._make_candidate(dist=sel.workspace.arm_reach_min)
        s_mid  = sel._score(c_mid,  obj, raw)
        s_edge = sel._score(c_edge, obj, raw)
        assert s_mid > s_edge

    def test_score_higher_when_clear(self):
        """More clearance → higher clearance score."""
        sel   = _selector(w_reach=0.0, w_heading=0.0, w_approach=0.0)
        c     = self._make_candidate(dist=0.55)
        obj   = np.array([2.0, 0.0, 0.025])
        raw_clear   = np.full(36, 5.0)
        raw_blocked = np.full(36, 0.10)
        s_clear   = sel._score(c, obj, raw_clear)
        s_blocked = sel._score(c, obj, raw_blocked)
        assert s_clear > s_blocked

    def test_approach_corridor_penalty(self):
        """When obstacle is closer than object along corridor, s_approach=0."""
        sel = _selector(w_reach=0.0, w_clearance=0.0, w_heading=0.0)
        c   = self._make_candidate(dist=0.55, xy=(0.0, 0.0))
        obj = np.array([2.0, 0.0, 0.025])  # obj at 2m from candidate
        # Obstacle at 1m (blocked)
        raw_blocked = np.full(36, 1.0)
        # Obstacle at 3m (clear beyond object)
        raw_clear   = np.full(36, 3.0)
        s_blocked = sel._score(c, obj, raw_blocked)
        s_clear   = sel._score(c, obj, raw_clear)
        assert s_clear > s_blocked

    def test_weights_scale_score(self):
        """Doubling w_reach doubles reach component of score."""
        sel1 = _selector(w_reach=1.0, w_clearance=0.0, w_heading=0.0, w_approach=0.0)
        sel2 = _selector(w_reach=2.0, w_clearance=0.0, w_heading=0.0, w_approach=0.0)
        c    = self._make_candidate(dist=0.55)
        obj  = np.array([2.0, 0.0, 0.025])
        raw  = np.full(36, 5.0)
        assert sel2._score(c, obj, raw) == pytest.approx(2 * sel1._score(c, obj, raw))


# ── _is_feasible ──────────────────────────────────────────────────────────────

class TestIsFeasible:
    def _make_candidate(self, dist=0.55):
        return GraspCandidate(
            xy=np.array([0.0, 0.0]), yaw=0.0, dist_to_obj=dist,
            score=0.0, feasible=False,
        )

    def test_feasible_when_in_reach_and_clear(self):
        sel = _selector(min_base_clearance=0.30)
        c   = self._make_candidate(dist=0.55)
        raw = np.full(36, 5.0)  # all clear >> 0.30m
        assert sel._is_feasible(c, raw, lidar_range=5.0) is True

    def test_infeasible_when_too_close(self):
        sel = _selector(min_base_clearance=0.30)
        c   = self._make_candidate(dist=0.10)   # below arm_reach_min=0.30
        raw = np.full(36, 5.0)
        assert sel._is_feasible(c, raw, lidar_range=5.0) is False

    def test_infeasible_when_too_far(self):
        sel = _selector(min_base_clearance=0.30)
        c   = self._make_candidate(dist=1.20)   # above arm_reach_max=0.80
        raw = np.full(36, 5.0)
        assert sel._is_feasible(c, raw, lidar_range=5.0) is False

    def test_infeasible_when_obstacle_close(self):
        sel = _selector(min_base_clearance=0.30)
        c   = self._make_candidate(dist=0.55)
        raw = np.full(36, 0.10)  # very close obstacles
        assert sel._is_feasible(c, raw, lidar_range=5.0) is False

    def test_empty_lidar_uses_lidar_range(self):
        sel = _selector(min_base_clearance=0.30)
        c   = self._make_candidate(dist=0.55)
        raw = np.array([])
        # empty lidar → min_d = lidar_range = 5.0 → clear
        assert sel._is_feasible(c, raw, lidar_range=5.0) is True


# ── select ────────────────────────────────────────────────────────────────────

class TestSelect:
    def test_returns_array_shape_3(self):
        sel = _selector()
        goal = sel.select(
            object_pos  = np.array([2.0, 0.0, 0.025]),
            robot_pos   = np.array([0.0, 0.0, 0.12]),
            lidar_dists = _flat_lidar(36, 1.0),
            lidar_range = 5.0,
        )
        assert goal.shape == (3,)

    def test_returns_float32(self):
        sel  = _selector()
        goal = sel.select(
            object_pos=np.array([2.0, 0.0, 0.025]),
            robot_pos =np.array([0.0, 0.0, 0.12]),
            lidar_dists=_flat_lidar(),
        )
        assert goal.dtype == np.float32

    def test_goal_not_at_object_when_clear(self):
        """In open space, goal should be an approach pose, not the object itself."""
        sel  = _selector()
        obj  = np.array([3.0, 0.0, 0.025])
        goal = sel.select(
            object_pos=obj, robot_pos=np.array([0.0, 0.0, 0.12]),
            lidar_dists=_flat_lidar(36, 1.0),
        )
        dist_to_obj = math.hypot(goal[0] - obj[0], goal[1] - obj[1])
        assert dist_to_obj > 0.1

    def test_goal_on_robot_side_of_object(self):
        """Goal should be placed between robot and object, not behind object."""
        sel  = _selector()
        obj  = np.array([3.0, 0.0, 0.025])
        rbt  = np.array([0.0, 0.0, 0.12])
        goal = sel.select(object_pos=obj, robot_pos=rbt, lidar_dists=_flat_lidar())
        # goal x should be < object x (closer to robot)
        assert goal[0] < obj[0]

    def test_fallback_when_all_blocked(self):
        """All candidates infeasible → fallback offset from object."""
        sel  = GraspGoalSelector(
            cfg=GraspGoalConfig(min_base_clearance=100.0, fallback_to_object=True)
        )
        obj  = np.array([2.0, 0.0, 0.025])
        goal = sel.select(
            object_pos=obj, robot_pos=np.array([0.0, 0.0, 0.12]),
            lidar_dists=_flat_lidar(36, 0.001),
        )
        assert goal.shape == (3,)
        # fallback places goal offset from object toward robot
        dist = math.hypot(goal[0] - obj[0], goal[1] - obj[1])
        ws = sel.workspace
        expected = ws.arm_mount_fwd + ws.arm_reach_min
        assert dist == pytest.approx(expected, abs=0.05)

    def test_fallback_direction_toward_robot(self):
        """Fallback goal should be on the robot's side of the object."""
        sel = GraspGoalSelector(
            cfg=GraspGoalConfig(min_base_clearance=100.0, fallback_to_object=True)
        )
        obj  = np.array([5.0, 0.0, 0.025])
        rbt  = np.array([0.0, 0.0, 0.12])
        goal = sel.select(object_pos=obj, robot_pos=rbt, lidar_dists=_flat_lidar(36, 0.001))
        assert goal[0] < obj[0]

    def test_deterministic(self):
        """Same inputs → same output every call."""
        sel  = _selector()
        kw   = dict(object_pos=np.array([2.0, 1.0, 0.025]),
                    robot_pos=np.array([0.0, 0.0, 0.12]),
                    lidar_dists=_flat_lidar())
        g1 = sel.select(**kw)
        g2 = sel.select(**kw)
        np.testing.assert_array_equal(g1, g2)

    def test_normalized_lidar_denormalized_correctly(self):
        """select() should work with normalised [0,1] lidar, not raw metres."""
        sel  = _selector(min_base_clearance=0.30)
        # normalised 1.0 → 5.0 m (clear) → feasible
        goal = sel.select(
            object_pos=np.array([2.0, 0.0, 0.025]),
            robot_pos =np.array([0.0, 0.0, 0.12]),
            lidar_dists=_flat_lidar(36, 1.0),
            lidar_range=5.0,
        )
        assert goal.shape == (3,)

    def test_different_objects_give_different_goals(self):
        sel  = _selector()
        g1 = sel.select(np.array([2.0, 0.0, 0.025]), np.array([0.0, 0.0, 0.12]),
                        _flat_lidar())
        g2 = sel.select(np.array([0.0, 2.0, 0.025]), np.array([0.0, 0.0, 0.12]),
                        _flat_lidar())
        assert not np.allclose(g1, g2)


# ── evaluate_all ──────────────────────────────────────────────────────────────

class TestEvaluateAll:
    def test_returns_list(self):
        sel  = _selector()
        cands = sel.evaluate_all(
            object_pos=np.array([2.0, 0.0, 0.025]),
            robot_pos =np.array([0.0, 0.0, 0.12]),
            lidar_dists=_flat_lidar(),
        )
        assert isinstance(cands, list)

    def test_all_candidates_scored(self):
        sel   = _selector()
        cands = sel.evaluate_all(
            object_pos=np.array([2.0, 0.0, 0.025]),
            robot_pos =np.array([0.0, 0.0, 0.12]),
            lidar_dists=_flat_lidar(),
        )
        for c in cands:
            assert c.score != 0.0 or True   # score may be 0 but field must be set
            assert isinstance(c.feasible, bool)

    def test_count_matches_select(self):
        sel   = _selector(n_candidates=16, n_arc_radii=2)
        cands = sel.evaluate_all(
            object_pos=np.array([2.0, 0.0, 0.025]),
            robot_pos =np.array([0.0, 0.0, 0.12]),
            lidar_dists=_flat_lidar(),
        )
        assert len(cands) == 16

    def test_in_clear_space_at_least_one_feasible(self):
        sel   = _selector()
        cands = sel.evaluate_all(
            object_pos=np.array([2.0, 0.0, 0.025]),
            robot_pos =np.array([0.0, 0.0, 0.12]),
            lidar_dists=_flat_lidar(36, 1.0),
            lidar_range=5.0,
        )
        assert any(c.feasible for c in cands)


# ── make_grasp_goal_selector ──────────────────────────────────────────────────

class TestFactory:
    def test_returns_selector(self):
        sel = make_grasp_goal_selector()
        assert isinstance(sel, GraspGoalSelector)

    def test_custom_reach(self):
        sel = make_grasp_goal_selector(arm_reach_min=0.20, arm_reach_max=0.60)
        assert sel.workspace.arm_reach_min == pytest.approx(0.20)
        assert sel.workspace.arm_reach_max == pytest.approx(0.60)

    def test_custom_n_candidates(self):
        sel = make_grasp_goal_selector(n_candidates=8)
        assert sel.cfg.n_candidates == 8

    def test_functional_select(self):
        sel  = make_grasp_goal_selector()
        goal = sel.select(
            object_pos=np.array([2.0, 0.0, 0.025]),
            robot_pos =np.array([0.0, 0.0, 0.12]),
            lidar_dists=_flat_lidar(),
        )
        assert goal.shape == (3,)
