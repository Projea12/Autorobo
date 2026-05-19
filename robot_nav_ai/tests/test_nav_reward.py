"""
tests/test_nav_reward.py — Unit tests for env/nav_reward.py.

Covers: RewardConfig, RewardInfo, NavRewardFunction (all 6 components),
exploration cell tracking, uncertainty penalty, make_reward_function.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.nav_reward import (
    RewardConfig, RewardInfo, NavRewardFunction, make_reward_function,
)
from env.nav_obs import PerceptionInput


# ── helpers ───────────────────────────────────────────────────────────────────

def _fn(**kwargs) -> NavRewardFunction:
    """Build a NavRewardFunction with default config overridden by kwargs."""
    defaults = {
        "approach": 2.0, "goal": 10.0, "collision": 5.0,
        "obstacle": 0.5, "explore": 0.1, "uncertainty": 0.05,
        "time_step": 0.01, "goal_radius": 0.25, "collision_r": 0.12,
        "danger_r": 0.25, "explore_cell_m": 0.5,
        "conf_thresh": 0.30, "occ_unknown_scale": 0.20,
    }
    defaults.update(kwargs)
    return NavRewardFunction(RewardConfig(**defaults))


_CLEAR_OCC  = np.zeros(64, dtype=np.float32)   # all free
_HALF_OCC   = np.full(64, 0.5, dtype=np.float32)  # all unknown
_ORIGIN     = np.array([0.0, 0.0])
_FAR_LIDAR  = 5.0   # no obstacles in range
_NEAR_LIDAR = 0.05  # very close — below collision_r


def _step(fn, robot_xy=_ORIGIN, d_prev=2.0, d_curr=2.0,
          d_lidar=_FAR_LIDAR, occ=None, perc=None):
    return fn.step(
        robot_xy    = robot_xy,
        d_prev      = d_prev,
        d_curr      = d_curr,
        d_lidar_min = d_lidar,
        occ_grid    = occ if occ is not None else _CLEAR_OCC,
        perception  = perc,
    )


# ── RewardConfig ──────────────────────────────────────────────────────────────

class TestRewardConfig:
    def test_defaults(self):
        cfg = RewardConfig()
        assert cfg.approach    == pytest.approx(2.0)
        assert cfg.goal        == pytest.approx(10.0)
        assert cfg.collision   == pytest.approx(5.0)
        assert cfg.obstacle    == pytest.approx(0.5)
        assert cfg.explore     == pytest.approx(0.10)
        assert cfg.uncertainty == pytest.approx(0.05)
        assert cfg.time_step   == pytest.approx(0.01)
        assert cfg.goal_radius == pytest.approx(0.25)
        assert cfg.collision_r == pytest.approx(0.12)
        assert cfg.danger_r    == pytest.approx(0.25)

    def test_frozen(self):
        cfg = RewardConfig()
        with pytest.raises(Exception):
            cfg.approach = 99.0

    def test_custom(self):
        cfg = RewardConfig(approach=5.0, goal=20.0)
        assert cfg.approach == pytest.approx(5.0)
        assert cfg.goal     == pytest.approx(20.0)


# ── RewardInfo ────────────────────────────────────────────────────────────────

class TestRewardInfo:
    def _make(self, **kw):
        defaults = dict(total=0.0, approach=0.0, goal=0.0, collision=0.0,
                        obstacle=0.0, explore=0.0, uncertainty=0.0, time=-0.01,
                        terminated=False, success=False, collision_flag=False,
                        new_cell=False, n_visited=1)
        defaults.update(kw)
        return RewardInfo(**defaults)

    def test_str_contains_total(self):
        ri = self._make(total=3.14)
        assert "r=" in str(ri)

    def test_str_success_tag(self):
        ri = self._make(success=True, goal=10.0, total=10.0)
        assert "SUCCESS" in str(ri)

    def test_str_crash_tag(self):
        ri = self._make(collision_flag=True, collision=-5.0, total=-5.0)
        assert "CRASH" in str(ri)

    def test_str_no_tag_normal(self):
        ri = self._make()
        assert "SUCCESS" not in str(ri)
        assert "CRASH"   not in str(ri)

    def test_fields_accessible(self):
        ri = self._make(n_visited=7, new_cell=True)
        assert ri.n_visited == 7
        assert ri.new_cell  is True


# ── NavRewardFunction — lifecycle ─────────────────────────────────────────────

class TestNavRewardFunctionLifecycle:
    def test_initial_n_visited_zero(self):
        fn = _fn()
        assert fn.n_visited_cells == 0

    def test_reset_marks_spawn_cell(self):
        fn = _fn(explore_cell_m=1.0)
        fn.reset(np.array([0.5, 0.5]))
        assert fn.n_visited_cells == 1

    def test_reset_clears_previous_episode(self):
        fn = _fn(explore_cell_m=1.0)
        fn.reset(_ORIGIN)
        _step(fn, robot_xy=np.array([5.0, 5.0]))
        fn.reset(_ORIGIN)
        assert fn.n_visited_cells == 1   # only spawn cell

    def test_cfg_accessible(self):
        cfg = RewardConfig(approach=3.0)
        fn  = NavRewardFunction(cfg)
        assert fn.cfg is cfg


# ── Component 1: approach ─────────────────────────────────────────────────────

class TestApproachComponent:
    def test_positive_when_closing(self):
        fn = _fn(approach=2.0, explore=0.0, uncertainty=0.0, time_step=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_prev=2.0, d_curr=1.5)
        assert ri.approach == pytest.approx(2.0 * 0.5)

    def test_negative_when_retreating(self):
        fn = _fn(approach=2.0, explore=0.0, uncertainty=0.0, time_step=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_prev=1.0, d_curr=1.5)
        assert ri.approach == pytest.approx(2.0 * (-0.5))

    def test_zero_when_stationary(self):
        fn = _fn(approach=2.0, explore=0.0, uncertainty=0.0, time_step=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_prev=1.5, d_curr=1.5)
        assert ri.approach == pytest.approx(0.0, abs=1e-9)

    def test_proportional_to_weight(self):
        fn1 = _fn(approach=1.0, explore=0.0, uncertainty=0.0, time_step=0.0)
        fn2 = _fn(approach=4.0, explore=0.0, uncertainty=0.0, time_step=0.0)
        fn1.reset(_ORIGIN);  fn2.reset(_ORIGIN)
        ri1 = _step(fn1, d_prev=2.0, d_curr=1.0)
        ri2 = _step(fn2, d_prev=2.0, d_curr=1.0)
        assert ri2.approach == pytest.approx(4.0 * ri1.approach)


# ── Component 2: goal_reached ─────────────────────────────────────────────────

class TestGoalReachedComponent:
    def test_bonus_when_inside_radius(self):
        fn = _fn(goal=10.0, goal_radius=0.5, explore=0.0, uncertainty=0.0,
                 time_step=0.0, approach=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_curr=0.3)
        assert ri.goal        == pytest.approx(10.0)
        assert ri.success     is True
        assert ri.terminated  is True

    def test_no_bonus_outside_radius(self):
        fn = _fn(goal=10.0, goal_radius=0.25, explore=0.0, uncertainty=0.0,
                 time_step=0.0, approach=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_curr=0.30)
        assert ri.goal    == pytest.approx(0.0)
        assert ri.success is False

    def test_goal_terminates_episode(self):
        fn = _fn(goal_radius=1.0, explore=0.0, uncertainty=0.0,
                 time_step=0.0, approach=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_curr=0.5)
        assert ri.terminated is True


# ── Component 3: collision ────────────────────────────────────────────────────

class TestCollisionComponent:
    def test_penalty_when_below_collision_r(self):
        fn = _fn(collision=5.0, collision_r=0.20, explore=0.0,
                 uncertainty=0.0, time_step=0.0, approach=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_lidar=0.10)
        assert ri.collision      == pytest.approx(-5.0)
        assert ri.collision_flag is True
        assert ri.terminated     is True

    def test_no_penalty_above_collision_r(self):
        fn = _fn(collision=5.0, collision_r=0.12, explore=0.0,
                 uncertainty=0.0, time_step=0.0, approach=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_lidar=0.15)
        assert ri.collision      == pytest.approx(0.0)
        assert ri.collision_flag is False

    def test_collision_and_goal_same_step(self):
        """If both conditions hold simultaneously, both bonuses/penalties apply."""
        fn = _fn(goal=10.0, collision=5.0, goal_radius=1.0, collision_r=0.5,
                 explore=0.0, uncertainty=0.0, time_step=0.0, approach=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_curr=0.3, d_lidar=0.1)
        assert ri.success        is True
        assert ri.collision_flag is True
        assert ri.terminated     is True


# ── Component 4: obstacle proximity ──────────────────────────────────────────

class TestObstacleProximity:
    def test_zero_penalty_far(self):
        fn = _fn(obstacle=0.5, danger_r=0.25, explore=0.0,
                 uncertainty=0.0, time_step=0.0, approach=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_lidar=0.5)
        assert ri.obstacle == pytest.approx(0.0)

    def test_max_penalty_at_zero_range(self):
        fn = _fn(obstacle=0.5, danger_r=0.25, explore=0.0,
                 uncertainty=0.0, time_step=0.0, approach=0.0,
                 collision_r=0.001)   # avoid collision termination
        fn.reset(_ORIGIN)
        ri = _step(fn, d_lidar=0.0)
        assert ri.obstacle == pytest.approx(-0.5, abs=1e-6)

    def test_half_penalty_at_half_range(self):
        fn = _fn(obstacle=1.0, danger_r=0.20, explore=0.0,
                 uncertainty=0.0, time_step=0.0, approach=0.0,
                 collision_r=0.001)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_lidar=0.10)   # proximity = 0.5
        assert ri.obstacle == pytest.approx(-0.5, abs=1e-4)

    def test_penalty_increases_as_range_decreases(self):
        fn = _fn(obstacle=1.0, danger_r=0.3, explore=0.0,
                 uncertainty=0.0, time_step=0.0, approach=0.0,
                 collision_r=0.001)
        fn.reset(_ORIGIN)
        ri_far  = _step(fn, d_lidar=0.25)
        ri_near = _step(fn, d_lidar=0.15)
        assert ri_near.obstacle < ri_far.obstacle


# ── Component 5: exploration ──────────────────────────────────────────────────

class TestExploration:
    def test_bonus_on_new_cell(self):
        fn = _fn(explore=0.1, explore_cell_m=1.0,
                 approach=0.0, uncertainty=0.0, time_step=0.0)
        fn.reset(np.array([0.0, 0.0]))
        # Move to a new cell
        ri = _step(fn, robot_xy=np.array([1.5, 0.0]))
        assert ri.explore   == pytest.approx(0.1)
        assert ri.new_cell  is True

    def test_no_bonus_revisiting_cell(self):
        fn = _fn(explore=0.1, explore_cell_m=1.0,
                 approach=0.0, uncertainty=0.0, time_step=0.0)
        fn.reset(np.array([0.5, 0.5]))
        # Stay in the same cell
        ri = _step(fn, robot_xy=np.array([0.7, 0.3]))
        assert ri.explore  == pytest.approx(0.0)
        assert ri.new_cell is False

    def test_n_visited_increments(self):
        fn = _fn(explore=0.1, explore_cell_m=1.0,
                 approach=0.0, uncertainty=0.0, time_step=0.0)
        fn.reset(np.array([0.0, 0.0]))
        assert fn.n_visited_cells == 1
        _step(fn, robot_xy=np.array([1.5, 0.0]))
        assert fn.n_visited_cells == 2
        _step(fn, robot_xy=np.array([3.0, 0.0]))
        assert fn.n_visited_cells == 3

    def test_revisit_after_leaving(self):
        """Cell visited twice should only award bonus once."""
        fn = _fn(explore=0.1, explore_cell_m=1.0,
                 approach=0.0, uncertainty=0.0, time_step=0.0)
        fn.reset(np.array([0.0, 0.0]))
        _step(fn, robot_xy=np.array([2.0, 0.0]))   # new cell
        _step(fn, robot_xy=np.array([0.0, 0.0]))   # back to origin cell (already visited)
        ri = _step(fn, robot_xy=np.array([0.5, 0.0]))
        assert ri.new_cell is False

    def test_n_visited_in_reward_info(self):
        fn = _fn(explore_cell_m=1.0, approach=0.0,
                 uncertainty=0.0, time_step=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, robot_xy=np.array([5.0, 5.0]))
        assert ri.n_visited == fn.n_visited_cells

    def test_cell_size_affects_granularity(self):
        """Smaller cell → more cells for same trajectory."""
        fn_fine   = _fn(explore=0.1, explore_cell_m=0.25,
                        approach=0.0, uncertainty=0.0, time_step=0.0)
        fn_coarse = _fn(explore=0.1, explore_cell_m=2.0,
                        approach=0.0, uncertainty=0.0, time_step=0.0)
        fn_fine.reset(_ORIGIN);  fn_coarse.reset(_ORIGIN)
        for i in range(10):
            xy = np.array([i * 0.5, 0.0])
            _step(fn_fine,   robot_xy=xy)
            _step(fn_coarse, robot_xy=xy)
        assert fn_fine.n_visited_cells > fn_coarse.n_visited_cells


# ── Component 6: uncertainty ──────────────────────────────────────────────────

class TestUncertaintyComponent:
    def test_zero_uncertainty_no_perception_clear_occ(self):
        fn = _fn(uncertainty=0.05, approach=0.0, time_step=0.0, explore=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, occ=_CLEAR_OCC, perc=None)
        # occ is all 0.0 → unknown_frac = 0; no perception → no shortfall
        assert ri.uncertainty == pytest.approx(0.0, abs=1e-9)

    def test_penalty_with_all_unknown_occ(self):
        fn = _fn(uncertainty=0.1, occ_unknown_scale=1.0,
                 approach=0.0, time_step=0.0, explore=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, occ=_HALF_OCC, perc=None)
        # unknown_frac = 1.0 → penalty = −0.1 × 1.0 × 1.0
        assert ri.uncertainty == pytest.approx(-0.1, abs=1e-6)

    def test_penalty_scales_with_occ_unknown_scale(self):
        fn1 = _fn(uncertainty=0.1, occ_unknown_scale=0.5,
                  approach=0.0, time_step=0.0, explore=0.0)
        fn2 = _fn(uncertainty=0.1, occ_unknown_scale=1.0,
                  approach=0.0, time_step=0.0, explore=0.0)
        fn1.reset(_ORIGIN);  fn2.reset(_ORIGIN)
        ri1 = _step(fn1, occ=_HALF_OCC)
        ri2 = _step(fn2, occ=_HALF_OCC)
        assert abs(ri2.uncertainty) == pytest.approx(2.0 * abs(ri1.uncertainty),
                                                     rel=1e-5)

    def test_perception_low_confidence_penalty(self):
        fn = _fn(uncertainty=1.0, conf_thresh=0.5,
                 approach=0.0, time_step=0.0, explore=0.0, occ_unknown_scale=0.0)
        fn.reset(_ORIGIN)
        perc = PerceptionInput(confidence=0.2, bearing_rad=0.0, dist_est_m=1.0)
        ri = _step(fn, occ=_CLEAR_OCC, perc=perc)
        # shortfall = 0.5 - 0.2 = 0.3 → penalty = -1.0 × 0.3
        assert ri.uncertainty == pytest.approx(-0.3, abs=1e-6)

    def test_no_perception_penalty_above_threshold(self):
        fn = _fn(uncertainty=1.0, conf_thresh=0.3,
                 approach=0.0, time_step=0.0, explore=0.0, occ_unknown_scale=0.0)
        fn.reset(_ORIGIN)
        perc = PerceptionInput(confidence=0.9)
        ri = _step(fn, occ=_CLEAR_OCC, perc=perc)
        assert ri.uncertainty == pytest.approx(0.0, abs=1e-9)

    def test_perception_exactly_at_threshold(self):
        fn = _fn(uncertainty=1.0, conf_thresh=0.5,
                 approach=0.0, time_step=0.0, explore=0.0, occ_unknown_scale=0.0)
        fn.reset(_ORIGIN)
        perc = PerceptionInput(confidence=0.5)
        ri = _step(fn, occ=_CLEAR_OCC, perc=perc)
        assert ri.uncertainty == pytest.approx(0.0, abs=1e-9)

    def test_both_sub_signals_sum(self):
        fn = _fn(uncertainty=1.0, conf_thresh=0.5, occ_unknown_scale=1.0,
                 approach=0.0, time_step=0.0, explore=0.0)
        fn.reset(_ORIGIN)
        perc = PerceptionInput(confidence=0.0)
        ri = _step(fn, occ=_HALF_OCC, perc=perc)
        # shortfall = 0.5 → -0.5; occ = -1.0×1.0×1.0 = -1.0
        assert ri.uncertainty == pytest.approx(-0.5 + -1.0, abs=1e-6)


# ── Time component ────────────────────────────────────────────────────────────

class TestTimeComponent:
    def test_time_penalty_per_step(self):
        fn = _fn(time_step=0.02, approach=0.0, explore=0.0, uncertainty=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn)
        assert ri.time == pytest.approx(-0.02, abs=1e-9)

    def test_time_penalty_in_total(self):
        fn = _fn(time_step=0.05, approach=0.0, explore=0.0, uncertainty=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn)
        assert ri.total == pytest.approx(-0.05, abs=1e-9)


# ── Total reward ──────────────────────────────────────────────────────────────

class TestTotalReward:
    def test_total_is_sum_of_components(self):
        fn = _fn()
        fn.reset(_ORIGIN)
        ri = _step(fn, d_prev=2.0, d_curr=1.8, d_lidar=3.0,
                   occ=_CLEAR_OCC, perc=None)
        expected = (ri.approach + ri.goal + ri.collision
                    + ri.obstacle + ri.explore + ri.uncertainty + ri.time)
        assert ri.total == pytest.approx(expected, abs=1e-9)

    def test_total_with_goal_bonus(self):
        fn = _fn(goal=10.0, goal_radius=1.0, approach=0.0,
                 explore=0.0, uncertainty=0.0, time_step=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_curr=0.5)
        assert ri.total == pytest.approx(10.0, abs=1e-6)

    def test_total_with_collision_penalty(self):
        fn = _fn(collision=5.0, collision_r=0.5, approach=0.0,
                 explore=0.0, uncertainty=0.0, time_step=0.0)
        fn.reset(_ORIGIN)
        ri = _step(fn, d_lidar=0.1)
        assert ri.total == pytest.approx(-5.0, abs=1e-6)

    def test_reward_is_finite(self):
        fn = _fn()
        fn.reset(_ORIGIN)
        for _ in range(10):
            ri = _step(fn, d_prev=1.5, d_curr=1.4, d_lidar=2.0,
                       occ=_CLEAR_OCC)
            assert math.isfinite(ri.total)


# ── make_reward_function ──────────────────────────────────────────────────────

class TestMakeRewardFunction:
    def test_returns_nav_reward_function(self):
        fn = make_reward_function()
        assert isinstance(fn, NavRewardFunction)

    def test_custom_weights_propagate(self):
        fn = make_reward_function(approach=5.0, goal=20.0)
        assert fn.cfg.approach == pytest.approx(5.0)
        assert fn.cfg.goal     == pytest.approx(20.0)

    def test_default_weights_match_config(self):
        fn  = make_reward_function()
        cfg = RewardConfig()
        assert fn.cfg.approach    == pytest.approx(cfg.approach)
        assert fn.cfg.time_step   == pytest.approx(cfg.time_step)
        assert fn.cfg.explore     == pytest.approx(cfg.explore)
        assert fn.cfg.uncertainty == pytest.approx(cfg.uncertainty)
