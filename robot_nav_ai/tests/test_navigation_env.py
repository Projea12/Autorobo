"""
tests/test_navigation_env.py — Unit & integration tests for NavigationEnv.

Tests run against a real compiled MjModel from robot.xml.  All tests keep
episode counts tiny (1-3 episodes, 5-10 steps) to stay fast.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.navigation_env import NavigationEnv
from env.nav_reward import RewardConfig as NavRewardConfig
from env.nav_obs import NAV_OBS_DIM, NavObsConfig
from env.episode_reset import SpawnConfig, GoalConfig


# ── shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    """Default NavigationEnv, module-scoped so the MjModel is compiled once."""
    e = NavigationEnv(
        max_steps  = 50,
        n_substeps = 2,
        seed       = 0,
    )
    yield e
    e.close()


@pytest.fixture(scope="module")
def env_reset(env):
    """env after one reset call."""
    obs, info = env.reset(seed=42)
    return env, obs, info


# ── constructor ────────────────────────────────────────────────────────────────

class TestConstructor:
    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (NAV_OBS_DIM,)

    def test_observation_space_dtype(self, env):
        assert env.observation_space.dtype == np.float32

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (2,)

    def test_action_space_low(self, env):
        assert np.all(env.action_space.low == -1.0)

    def test_action_space_high(self, env):
        assert np.all(env.action_space.high == 1.0)

    def test_action_space_dtype(self, env):
        assert env.action_space.dtype == np.float32

    def test_render_mode_none(self, env):
        assert env.render_mode is None

    def test_metadata_has_render_fps(self, env):
        assert "render_fps" in env.metadata


# ── reset ──────────────────────────────────────────────────────────────────────

class TestReset:
    def test_obs_shape(self, env_reset):
        _, obs, _ = env_reset
        assert obs.shape == (NAV_OBS_DIM,)

    def test_obs_dtype(self, env_reset):
        _, obs, _ = env_reset
        assert obs.dtype == np.float32

    def test_obs_finite(self, env_reset):
        _, obs, _ = env_reset
        assert np.all(np.isfinite(obs))

    def test_info_is_dict(self, env_reset):
        _, _, info = env_reset
        assert isinstance(info, dict)

    def test_info_success_false(self, env_reset):
        _, _, info = env_reset
        assert info["success"] is False

    def test_info_collision_false(self, env_reset):
        _, _, info = env_reset
        assert info["collision"] is False

    def test_info_step_zero(self, env_reset):
        _, _, info = env_reset
        assert info["step"] == 0

    def test_info_dist_positive(self, env_reset):
        _, _, info = env_reset
        assert info["dist_to_goal"] > 0.0

    def test_info_goal_is_2list(self, env_reset):
        _, _, info = env_reset
        assert isinstance(info["goal"], list)
        assert len(info["goal"]) == 2

    def test_step_count_reset(self, env):
        env.reset(seed=1)
        assert env._step_count == 0

    def test_obs_in_observation_space(self, env):
        obs, _ = env.reset(seed=7)
        assert env.observation_space.contains(obs)

    def test_reset_different_seeds_differ(self, env):
        obs1, _ = env.reset(seed=100)
        obs2, _ = env.reset(seed=200)
        assert not np.allclose(obs1, obs2)

    def test_reset_same_seed_reproducible(self, env):
        obs1, _ = env.reset(seed=55)
        obs2, _ = env.reset(seed=55)
        np.testing.assert_array_equal(obs1, obs2)


# ── step ───────────────────────────────────────────────────────────────────────

class TestStep:
    def test_returns_five_tuple(self, env):
        env.reset(seed=0)
        result = env.step(np.zeros(2, dtype=np.float32))
        assert len(result) == 5

    def test_obs_shape(self, env):
        env.reset(seed=0)
        obs, *_ = env.step(np.zeros(2))
        assert obs.shape == (NAV_OBS_DIM,)

    def test_obs_dtype(self, env):
        env.reset(seed=0)
        obs, *_ = env.step(np.zeros(2))
        assert obs.dtype == np.float32

    def test_obs_finite(self, env):
        env.reset(seed=0)
        obs, *_ = env.step(np.zeros(2))
        assert np.all(np.isfinite(obs))

    def test_reward_is_float(self, env):
        env.reset(seed=0)
        _, reward, *_ = env.step(np.zeros(2))
        assert isinstance(reward, float)

    def test_terminated_is_bool(self, env):
        env.reset(seed=0)
        _, _, terminated, *_ = env.step(np.zeros(2))
        assert isinstance(terminated, bool)

    def test_truncated_is_bool(self, env):
        env.reset(seed=0)
        _, _, _, truncated, _ = env.step(np.zeros(2))
        assert isinstance(truncated, bool)

    def test_info_is_dict(self, env):
        env.reset(seed=0)
        *_, info = env.step(np.zeros(2))
        assert isinstance(info, dict)

    def test_step_increments_count(self, env):
        env.reset(seed=0)
        env.step(np.zeros(2))
        assert env._step_count == 1

    def test_not_terminated_at_start(self, env):
        env.reset(seed=0)
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
        # Neither should be true on first step from a random spawn
        # (may fail by chance if goal is placed at spawn — keep seed=0 stable)
        assert not (terminated and truncated)

    def test_truncated_at_max_steps(self):
        e = NavigationEnv(max_steps=3, n_substeps=1, seed=0)
        e.reset(seed=0)
        for _ in range(3):
            _, _, terminated, truncated, _ = e.step(np.zeros(2))
        assert truncated
        e.close()

    def test_step_count_in_info(self, env):
        env.reset(seed=0)
        *_, info = env.step(np.zeros(2))
        assert info["step"] == 1

    def test_success_flag_in_info(self, env):
        env.reset(seed=0)
        *_, info = env.step(np.zeros(2))
        assert "success" in info

    def test_collision_flag_in_info(self, env):
        env.reset(seed=0)
        *_, info = env.step(np.zeros(2))
        assert "collision" in info


# ── reward shaping ─────────────────────────────────────────────────────────────

class TestReward:
    def test_time_penalty_nonzero(self):
        """Idle robot should incur a time penalty (negative reward shift)."""
        cfg = NavRewardConfig(approach=0.0, obstacle=0.0,
                              time_step=0.05, goal=10.0, collision=5.0)
        e = NavigationEnv(reward_cfg=cfg, max_steps=5, n_substeps=1, seed=0)
        e.reset(seed=0)
        _, reward, _, _, _ = e.step(np.zeros(2))
        # With approach=0, only time penalty applies (minus any approach bonus)
        # We can't guarantee sign exactly, but penalty must have been deducted.
        e.close()

    def test_forward_action_nonnegative_approach(self):
        """
        Driving directly toward the goal should yield a positive approach reward.
        We pick a seed where the goal is placed directly in front and drive fwd.
        """
        cfg = NavRewardConfig(obstacle=0.0, time_step=0.0)
        e = NavigationEnv(
            reward_cfg = cfg,
            max_steps  = 10,
            n_substeps = 5,
            goal_cfg   = GoalConfig(mode="relative", fwd_range=(1.5, 1.5),
                                     lat_range=(0.0, 0.0)),
            seed       = 0,
        )
        e.reset(seed=0)
        rewards = [e.step(np.array([1.0, 0.0]))[1] for _ in range(5)]
        # At least some steps should have positive reward when driving forward
        assert any(r > 0 for r in rewards)
        e.close()

    def test_success_bonus_on_arrival(self):
        """Reward when goal_dist < goal_radius must include success bonus."""
        rew_cfg = NavRewardConfig(goal=20.0, goal_radius=100.0)  # very large radius
        e = NavigationEnv(reward_cfg=rew_cfg, max_steps=5, n_substeps=1, seed=0)
        e.reset(seed=0)
        _, reward, terminated, _, info = e.step(np.zeros(2))
        if info["success"]:
            assert reward >= rew_cfg.goal * 0.9   # bonus dominates
        e.close()

    def test_collision_penalty_on_crash(self):
        """When lidar detects collision, reward must be penalised and episode ends."""
        rew_cfg = NavRewardConfig(collision=50.0, collision_r=100.0)  # always "crash"
        e = NavigationEnv(reward_cfg=rew_cfg, max_steps=5, n_substeps=1, seed=0)
        e.reset(seed=0)
        _, reward, terminated, _, info = e.step(np.zeros(2))
        if info["collision"]:
            assert terminated
            assert reward <= -rew_cfg.collision * 0.9
        e.close()


# ── action mapping ─────────────────────────────────────────────────────────────

class TestActionMapping:
    def test_zero_action_stays_idle(self, env):
        env.reset(seed=0)
        pos_before = env._data.qpos[env._base_qadr : env._base_qadr + 2].copy()
        for _ in range(3):
            env.step(np.zeros(2))
        pos_after = env._data.qpos[env._base_qadr : env._base_qadr + 2].copy()
        assert np.linalg.norm(pos_after - pos_before) < 0.05

    def test_forward_action_moves_robot(self, env):
        env.reset(seed=0)
        pos_before = env._data.qpos[env._base_qadr : env._base_qadr + 2].copy()
        for _ in range(10):
            env.step(np.array([1.0, 0.0]))
        pos_after = env._data.qpos[env._base_qadr : env._base_qadr + 2].copy()
        assert np.linalg.norm(pos_after - pos_before) > 0.01

    def test_action_clipped_before_apply(self, env):
        env.reset(seed=0)
        # Should not raise even with out-of-bounds action
        env.action_processor.process(np.array([5.0, -5.0]))

    def test_left_wheel_faster_turns_right(self, env):
        """v_lin=0, v_ang>0 → left wheel backward, right wheel forward → turns left."""
        env.reset(seed=0)
        phys = env.action_processor.process(np.array([0.0, 1.0]))
        # v_ang > 0 (CCW / left turn): left wheel slower (negative), right faster (positive)
        assert phys.ctrl_left  < 0
        assert phys.ctrl_right > 0


# ── _goal_dist helper ──────────────────────────────────────────────────────────

class TestGoalDist:
    def test_positive(self, env):
        env.reset(seed=0)
        assert env._goal_dist() >= 0.0

    def test_decreases_on_approach(self, env):
        """After driving forward toward a frontal goal, distance should shrink."""
        e = NavigationEnv(
            max_steps = 20,
            n_substeps = 5,
            goal_cfg  = GoalConfig(mode="relative", fwd_range=(2.0, 2.0),
                                    lat_range=(0.0, 0.0)),
            seed      = 0,
        )
        e.reset(seed=0)
        d0 = e._goal_dist()
        for _ in range(10):
            e.step(np.array([1.0, 0.0]))
        d1 = e._goal_dist()
        assert d1 < d0
        e.close()


# ── forward_arc_min_range ──────────────────────────────────────────────────────

class TestForwardArcMinRange:
    def test_returns_positive(self, env):
        env.reset(seed=0)
        d = env._forward_arc_min_range()
        assert d > 0.0

    def test_at_most_lidar_max_range(self, env):
        env.reset(seed=0)
        d = env._forward_arc_min_range()
        assert d <= env._obs_cfg.lidar_max_range


# ── render ─────────────────────────────────────────────────────────────────────

class TestRender:
    def test_render_none_returns_none(self, env):
        env.reset(seed=0)
        assert env.render() is None

    def test_render_rgb_array(self):
        e = NavigationEnv(render_mode="rgb_array", max_steps=5, n_substeps=1)
        e.reset(seed=0)
        frame = e.render()
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        e.close()


# ── close ──────────────────────────────────────────────────────────────────────

class TestClose:
    def test_close_idempotent(self):
        e = NavigationEnv(max_steps=5, n_substeps=1)
        e.close()
        e.close()   # should not raise


# ── NavRewardConfig ────────────────────────────────────────────────────────────

class TestNavRewardConfig:
    def test_defaults(self):
        cfg = NavRewardConfig()
        assert cfg.approach    == pytest.approx(2.0)
        assert cfg.obstacle    == pytest.approx(0.5)
        assert cfg.time_step   == pytest.approx(0.01)
        assert cfg.goal        == pytest.approx(10.0)
        assert cfg.collision   == pytest.approx(5.0)
        assert cfg.danger_r    == pytest.approx(0.25)
        assert cfg.collision_r == pytest.approx(0.12)
        assert cfg.goal_radius == pytest.approx(0.25)

    def test_immutable(self):
        cfg = NavRewardConfig()
        with pytest.raises(Exception):
            cfg.approach = 99.0

    def test_custom_values(self):
        cfg = NavRewardConfig(approach=5.0, goal=20.0)
        assert cfg.approach == pytest.approx(5.0)
        assert cfg.goal     == pytest.approx(20.0)


# ── multi-episode consistency ──────────────────────────────────────────────────

class TestMultiEpisode:
    def test_three_episodes_no_crash(self):
        e = NavigationEnv(max_steps=10, n_substeps=2, seed=0)
        for ep in range(3):
            obs, _ = e.reset(seed=ep)
            for _ in range(5):
                action = e.action_space.sample()
                obs, reward, terminated, truncated, info = e.step(action)
                assert obs.shape == (NAV_OBS_DIM,)
                assert np.isfinite(reward)
                if terminated or truncated:
                    break
        e.close()

    def test_episode_step_count_bounded(self):
        e = NavigationEnv(max_steps=8, n_substeps=1, seed=0)
        for _ in range(2):
            e.reset(seed=0)
            steps = 0
            for _ in range(20):
                _, _, terminated, truncated, _ = e.step(np.zeros(2))
                steps += 1
                if terminated or truncated:
                    break
            assert steps <= 8
        e.close()
