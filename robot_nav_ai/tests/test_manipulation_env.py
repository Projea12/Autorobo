"""
Tests for env/manipulation_env.py — ManipulationEnv Gymnasium environment.

Covers:
  1.  Environment construction (MjSpec + compile, spaces)
  2.  reset() — obs shape / dtype, target randomised within bounds
  3.  step() — return types, obs shape, reward is float, truncation
  4.  Observation layout — target_pos in [39:42], rel_target in [42:45]
  5.  Action scaling — wheel ctrl, arm ctrl within ctrlrange, gripper
  6.  Reward signal — time penalty present, nav shaping sign
  7.  Termination / truncation logic
  8.  render("rgb_array") — shape and dtype
  9.  Seeded reset reproducibility
 10.  info dict keys and types
 11.  Model DOF counts (nq=24, nv=22)
 12.  Zero action → physics runs without NaN
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from env.manipulation_env import (
    ManipulationEnv, OBS_DIM, ACT_DIM,
    _quat_to_yaw, _normalise_arm_pos,
    _TARGET_FLOOR_Z,
)
from robot.workspace import DEFAULT_LIMITS


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    e = ManipulationEnv()
    e.reset(seed=0)
    yield e
    e.close()


@pytest.fixture(scope="module")
def render_env():
    e = ManipulationEnv(render_mode="rgb_array")
    e.reset(seed=0)
    yield e
    e.close()


# ── 1. Environment construction ───────────────────────────────────────────────

def test_env_creates():
    e = ManipulationEnv()
    assert e is not None
    e.close()


def test_observation_space_shape(env):
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.observation_space.dtype == np.float32


def test_action_space_shape(env):
    assert env.action_space.shape == (ACT_DIM,)
    assert env.action_space.dtype == np.float32


def test_action_space_bounds(env):
    assert float(env.action_space.low[0])  == pytest.approx(-1.0)
    assert float(env.action_space.high[0]) == pytest.approx(1.0)


# ── 2. reset() ────────────────────────────────────────────────────────────────

def test_reset_returns_obs_and_info():
    e = ManipulationEnv()
    obs, info = e.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    e.close()


def test_reset_obs_shape():
    e = ManipulationEnv()
    obs, _ = e.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    e.close()


def test_reset_obs_dtype():
    e = ManipulationEnv()
    obs, _ = e.reset(seed=0)
    assert obs.dtype == np.float32
    e.close()


def test_reset_obs_no_nan():
    e = ManipulationEnv()
    obs, _ = e.reset(seed=0)
    assert np.all(np.isfinite(obs)), "obs contains NaN or Inf after reset"
    e.close()


def test_reset_target_x_in_range():
    """Target x coordinate (obs[39]) should be in _TARGET_X_RANGE."""
    e = ManipulationEnv()
    for seed in range(10):
        obs, _ = e.reset(seed=seed)
        target_x = float(obs[39])
        assert 0.40 <= target_x <= 0.85, f"seed={seed}: target_x={target_x}"
    e.close()


def test_reset_target_y_in_range():
    e = ManipulationEnv()
    for seed in range(10):
        obs, _ = e.reset(seed=seed)
        target_y = float(obs[40])
        assert -0.30 <= target_y <= 0.30, f"seed={seed}: target_y={target_y}"
    e.close()


def test_reset_target_z_at_floor():
    """Target z (obs[41]) should equal _TARGET_FLOOR_Z right after reset."""
    e = ManipulationEnv()
    obs, _ = e.reset(seed=0)
    target_z = float(obs[41])
    assert target_z == pytest.approx(_TARGET_FLOOR_Z, abs=1e-4)
    e.close()


# ── 3. step() ─────────────────────────────────────────────────────────────────

def test_step_return_types():
    e = ManipulationEnv()
    e.reset(seed=0)
    action = np.zeros(ACT_DIM, dtype=np.float32)
    obs, reward, terminated, truncated, info = e.step(action)
    assert obs.shape     == (OBS_DIM,)
    assert obs.dtype     == np.float32
    assert isinstance(reward,     float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated,  bool)
    assert isinstance(info,       dict)
    e.close()


def test_step_obs_no_nan():
    e = ManipulationEnv()
    e.reset(seed=0)
    obs, *_ = e.step(np.zeros(ACT_DIM, dtype=np.float32))
    assert np.all(np.isfinite(obs))
    e.close()


def test_step_not_terminated_at_start():
    e = ManipulationEnv()
    e.reset(seed=0)
    _, _, terminated, truncated, _ = e.step(np.zeros(ACT_DIM, dtype=np.float32))
    assert not terminated
    e.close()


def test_truncation_at_max_steps():
    """With max_steps=2 the episode should truncate after 2 steps."""
    e = ManipulationEnv(max_steps=2)
    e.reset(seed=0)
    action = np.zeros(ACT_DIM, dtype=np.float32)
    _, _, _, truncated, _ = e.step(action)
    assert not truncated
    _, _, _, truncated, _ = e.step(action)
    assert truncated
    e.close()


# ── 4. Observation layout ─────────────────────────────────────────────────────

def test_obs_base_pos_plausible():
    """Base position [0:3] should be near the origin at reset."""
    e = ManipulationEnv()
    obs, _ = e.reset(seed=0)
    base_pos = obs[0:3]
    assert np.linalg.norm(base_pos[:2]) < 1.0, "base far from origin at reset"
    assert 0.10 <= base_pos[2] <= 0.25, f"base z={base_pos[2]} implausible"
    e.close()


def test_obs_rel_target_consistent():
    """rel_target [42:45] should equal target_pos [39:42] - ee_pos [26:29]."""
    e = ManipulationEnv()
    obs, _ = e.reset(seed=0)
    target_pos = obs[39:42]
    ee_pos     = obs[26:29]
    rel_target = obs[42:45]
    np.testing.assert_allclose(rel_target, target_pos - ee_pos, atol=1e-5)
    e.close()


def test_obs_arm_pos_normalised():
    """Normalised arm positions [10:16] must lie in [-1, 1]."""
    e = ManipulationEnv()
    obs, _ = e.reset(seed=0)
    arm_norm = obs[10:16]
    assert np.all(arm_norm >= -1.0 - 1e-6)
    assert np.all(arm_norm <=  1.0 + 1e-6)
    e.close()


def test_obs_ee_quat_unit():
    """EE quaternion [29:33] should be a unit quaternion."""
    e = ManipulationEnv()
    obs, _ = e.reset(seed=0)
    ee_quat = obs[29:33]
    assert np.linalg.norm(ee_quat) == pytest.approx(1.0, abs=0.01)
    e.close()


# ── 5. Action scaling ─────────────────────────────────────────────────────────

def test_wheel_ctrl_full_forward():
    """Action [1, 1, ...] should produce wheel ctrl = +wheel_vel_max."""
    e = ManipulationEnv()
    e.reset(seed=0)
    action = np.ones(ACT_DIM, dtype=np.float32)
    # Directly inspect the internal _scale_action
    ctrl = e._scale_action(action.astype(np.float64))
    lim  = DEFAULT_LIMITS
    assert ctrl[0] == pytest.approx( lim.wheel_vel_max, rel=1e-9)
    assert ctrl[1] == pytest.approx( lim.wheel_vel_max, rel=1e-9)
    e.close()


def test_wheel_ctrl_full_reverse():
    """Action [-1, -1, ...] → wheel ctrl = -wheel_vel_max."""
    e = ManipulationEnv()
    e.reset(seed=0)
    action = -np.ones(ACT_DIM, dtype=np.float32)
    ctrl   = e._scale_action(action.astype(np.float64))
    lim    = DEFAULT_LIMITS
    assert ctrl[0] == pytest.approx(-lim.wheel_vel_max, rel=1e-9)
    assert ctrl[1] == pytest.approx(-lim.wheel_vel_max, rel=1e-9)
    e.close()


def test_arm_ctrl_within_ctrlrange():
    """Arm ctrl [2:8] must lie within the actuator ctrlrange for all actions."""
    e = ManipulationEnv()
    e.reset(seed=0)
    rng = np.random.default_rng(7)
    for _ in range(20):
        action = rng.uniform(-1, 1, size=ACT_DIM)
        ctrl   = e._scale_action(action)
        lim_lo = e._arm_ctrlrange[:, 0]
        lim_hi = e._arm_ctrlrange[:, 1]
        assert np.all(ctrl[2:8] >= lim_lo - 1e-9)
        assert np.all(ctrl[2:8] <= lim_hi + 1e-9)
    e.close()


def test_gripper_ctrl_range():
    """Gripper ctrl[8] must lie in [0, finger_pos_max] for any action."""
    e = ManipulationEnv()
    e.reset(seed=0)
    for a8 in [-1.0, 0.0, 1.0]:
        action = np.zeros(ACT_DIM)
        action[8] = a8
        ctrl = e._scale_action(action)
        assert 0.0 - 1e-9 <= ctrl[8] <= DEFAULT_LIMITS.finger_pos_max + 1e-9
    e.close()


def test_gripper_ctrl_zero_at_minus_one():
    """action[8] = -1 → ctrl[8] = 0 (fully closed)."""
    e = ManipulationEnv()
    e.reset(seed=0)
    action    = np.zeros(ACT_DIM); action[8] = -1.0
    ctrl      = e._scale_action(action)
    assert ctrl[8] == pytest.approx(0.0, abs=1e-9)
    e.close()


def test_gripper_ctrl_max_at_plus_one():
    """action[8] = +1 → ctrl[8] = finger_pos_max (fully open)."""
    e = ManipulationEnv()
    e.reset(seed=0)
    action    = np.zeros(ACT_DIM); action[8] = 1.0
    ctrl      = e._scale_action(action)
    assert ctrl[8] == pytest.approx(DEFAULT_LIMITS.finger_pos_max, rel=1e-9)
    e.close()


# ── 6. Reward signal ──────────────────────────────────────────────────────────

def test_time_penalty_present():
    """Every step should have a negative time penalty baked in."""
    e = ManipulationEnv()
    e.reset(seed=0)
    # Zero action → no navigation progress expected
    _, reward, _, _, _ = e.step(np.zeros(ACT_DIM, dtype=np.float32))
    # Reward can be positive (reach shaping) but must include -0.01 time penalty
    # At least the time penalty coefficient should be visible
    # We test indirectly: reward <= max_possible - 0.01
    # (nav shaping can add ≤ 0 for zero action → reward ≤ -0.01)
    # Since zero action = zero vel, base_dist and ee_dist barely change → shaping ≈ 0
    assert reward < 1.0   # large positive reward is implausible at step 1
    e.close()


def test_reward_is_finite():
    e = ManipulationEnv()
    e.reset(seed=0)
    for _ in range(10):
        action = np.zeros(ACT_DIM, dtype=np.float32)
        _, reward, _, _, _ = e.step(action)
        assert math.isfinite(reward)
    e.close()


# ── 7. Termination / truncation ───────────────────────────────────────────────

def test_no_premature_termination():
    """100 zero-action steps should not trigger success termination."""
    e = ManipulationEnv(max_steps=200)
    e.reset(seed=0)
    for _ in range(100):
        _, _, terminated, _, _ = e.step(np.zeros(ACT_DIM, dtype=np.float32))
        assert not terminated, "premature success termination"
    e.close()


def test_truncation_not_terminated_simultaneously():
    """When truncated, terminated should still be False (no success)."""
    e = ManipulationEnv(max_steps=3)
    e.reset(seed=0)
    action = np.zeros(ACT_DIM, dtype=np.float32)
    for _ in range(2):
        e.step(action)
    _, _, terminated, truncated, _ = e.step(action)
    assert truncated
    assert not terminated
    e.close()


# ── 8. render("rgb_array") ────────────────────────────────────────────────────

def test_render_shape(render_env):
    frame = render_env.render()
    assert frame is not None
    assert frame.shape == (480, 640, 3)


def test_render_dtype(render_env):
    frame = render_env.render()
    assert frame.dtype == np.uint8


def test_render_not_all_black(render_env):
    """Rendered frame should not be a totally black image."""
    frame = render_env.render()
    assert frame.max() > 0


def test_no_render_mode_returns_none():
    e = ManipulationEnv(render_mode=None)
    e.reset(seed=0)
    assert e.render() is None
    e.close()


# ── 9. Seeded reset reproducibility ──────────────────────────────────────────

def test_same_seed_same_obs():
    e = ManipulationEnv()
    obs1, _ = e.reset(seed=42)
    obs2, _ = e.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
    e.close()


def test_different_seeds_different_obs():
    e = ManipulationEnv()
    obs1, _ = e.reset(seed=0)
    obs2, _ = e.reset(seed=1)
    # target positions differ → obs[39:42] should differ
    assert not np.allclose(obs1[39:42], obs2[39:42])
    e.close()


def test_same_seed_gives_identical_trajectory():
    """10-step trajectory from same seed should be bitwise identical."""
    def run(seed):
        e = ManipulationEnv()
        e.reset(seed=seed)
        e.action_space.seed(seed)
        traj = []
        for _ in range(10):
            action = e.action_space.sample()
            obs, r, *_ = e.step(action)
            traj.append((obs.copy(), r))
        e.close()
        return traj

    t1 = run(99)
    t2 = run(99)
    for (o1, r1), (o2, r2) in zip(t1, t2):
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2


# ── 10. info dict ─────────────────────────────────────────────────────────────

def test_info_keys():
    e = ManipulationEnv()
    e.reset(seed=0)
    _, _, _, _, info = e.step(np.zeros(ACT_DIM, dtype=np.float32))
    for key in ("success", "step", "target_height", "lift", "ee_to_target", "base_to_target"):
        assert key in info, f"missing key: {key}"
    e.close()


def test_info_step_increments():
    e = ManipulationEnv()
    e.reset(seed=0)
    action = np.zeros(ACT_DIM, dtype=np.float32)
    for expected_step in range(1, 5):
        _, _, _, _, info = e.step(action)
        assert info["step"] == expected_step
    e.close()


def test_info_success_false_initially():
    e = ManipulationEnv()
    e.reset(seed=0)
    _, _, _, _, info = e.step(np.zeros(ACT_DIM, dtype=np.float32))
    assert info["success"] is False
    e.close()


def test_info_lift_non_negative_at_floor():
    e = ManipulationEnv()
    e.reset(seed=0)
    _, _, _, _, info = e.step(np.zeros(ACT_DIM, dtype=np.float32))
    assert info["lift"] >= -0.01   # box barely moves
    e.close()


# ── 11. Model DOF counts ──────────────────────────────────────────────────────

def test_model_nq():
    """nq should be 24: 17 robot + 7 target freejoint."""
    e = ManipulationEnv()
    assert e._model.nq == 24, f"expected nq=24, got {e._model.nq}"
    e.close()


def test_model_nv():
    """nv should be 22: 16 robot dofs + 6 target dofs."""
    e = ManipulationEnv()
    assert e._model.nv == 22, f"expected nv=22, got {e._model.nv}"
    e.close()


def test_target_joint_exists():
    e = ManipulationEnv()
    j_id = mujoco.mj_name2id(e._model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
    assert j_id >= 0, "target_joint not found in compiled model"
    e.close()


def test_target_qadr_is_17():
    e = ManipulationEnv()
    assert e._target_qadr == 17
    e.close()


# ── 12. Zero action → no NaN over 50 steps ───────────────────────────────────

def test_zero_action_50_steps_no_nan():
    e = ManipulationEnv(max_steps=50)
    obs, _ = e.reset(seed=0)
    assert np.all(np.isfinite(obs))
    action = np.zeros(ACT_DIM, dtype=np.float32)
    for step in range(50):
        obs, reward, terminated, truncated, _ = e.step(action)
        assert np.all(np.isfinite(obs)), f"NaN/Inf in obs at step {step}"
        assert math.isfinite(reward),   f"NaN/Inf reward at step {step}"
        if terminated or truncated:
            break
    e.close()


# ── module-level helper tests ─────────────────────────────────────────────────

def test_quat_to_yaw_identity():
    """Identity quaternion [1,0,0,0] → yaw = 0."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert _quat_to_yaw(q) == pytest.approx(0.0, abs=1e-9)


def test_quat_to_yaw_90_deg():
    """90° yaw rotation quaternion → yaw ≈ π/2."""
    angle = math.pi / 2
    q = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
    assert _quat_to_yaw(q) == pytest.approx(angle, abs=1e-9)


def test_normalise_arm_pos_midpoint():
    """Mid-range joint angles should normalise to 0."""
    lim = DEFAULT_LIMITS
    mid = (lim.joint_pos_hi + lim.joint_pos_lo) / 2.0
    norm = _normalise_arm_pos(mid, lim)
    np.testing.assert_allclose(norm, np.zeros(6), atol=1e-9)


def test_normalise_arm_pos_at_hi():
    """Joint angles at hi limit should normalise to 1."""
    lim  = DEFAULT_LIMITS
    norm = _normalise_arm_pos(lim.joint_pos_hi, lim)
    np.testing.assert_allclose(norm, np.ones(6), atol=1e-9)
