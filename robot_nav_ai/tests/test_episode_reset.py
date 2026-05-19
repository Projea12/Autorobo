"""
Tests for env/episode_reset.py — EpisodeResetter.

Covers:
  1.  make_resetter() — constructs from model, resolves home_kf_id
  2.  reset() — returns EpisodeInfo with correct fields
  3.  reset() — arm joints still in home pose after reset
  4.  randomise_robot_spawn() — xy in configured range
  5.  randomise_robot_spawn() — z unchanged from keyframe
  6.  randomise_robot_spawn() — yaw in configured range
  7.  randomise_robot_spawn() — quaternion is unit norm
  8.  randomise_robot_spawn() — base velocity zeroed
  9.  randomise_robot_spawn() — clearance from obstacles
 10.  randomise_obstacles() — active obstacles not underground
 11.  randomise_obstacles() — no-op when world_state is None
 12.  randomise_goal(mode="relative") — goal in front of robot
 13.  randomise_goal(mode="relative") — rotates with robot yaw
 14.  randomise_goal(mode="world") — in configured world range
 15.  randomise_goal() — pickable qpos updated when world_state present
 16.  randomise_goal() — returns (3,) array
 17.  Same seed → identical EpisodeInfo
 18.  Different seeds → different robot_xy
 19.  Different seeds → different goal_xyz
 20.  Domain rand applied when provided
 21.  reset() with no world_state and no domain_rand still works
 22.  mj_forward runs without NaN after reset()
 23.  50 physics steps without NaN after reset()
 24.  EpisodeInfo.n_active_obstacles matches configured count
 25.  Yaw-0 relative goal: goal x > robot x (in front)
 26.  make_resetter() raises RuntimeError on model without home keyframe
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from robot.constants import ROBOT_XML_PATH
from world.world import WorldBuilder, WorldConfig
from env.domain_rand import DomainRandomizer
from env.episode_reset import (
    EpisodeResetter, EpisodeInfo, SpawnConfig, GoalConfig, make_resetter,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def world():
    cfg   = WorldConfig(n_obstacles=4, max_obstacles=6, n_pickable=1,
                        max_pickable=2, build_seed=0)
    model, state = WorldBuilder(cfg).build(ROBOT_XML_PATH)
    rand  = DomainRandomizer(model)
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    return model, state, rand, kf_id


@pytest.fixture(scope="module")
def model(world):
    return world[0]


@pytest.fixture(scope="module")
def state(world):
    return world[1]


@pytest.fixture(scope="module")
def rand(world):
    return world[2]


@pytest.fixture(scope="module")
def kf_id(world):
    return world[3]


def fresh_data(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    return d


def make_resetter_full(world):
    model, state, rand, kf_id = world
    return EpisodeResetter(
        home_kf_id  = kf_id,
        world_state = state,
        domain_rand = rand,
    )


# ── 1. make_resetter() ────────────────────────────────────────────────────────

def test_make_resetter_constructs(model, state):
    r = make_resetter(model, world_state=state)
    assert isinstance(r, EpisodeResetter)


def test_make_resetter_resolves_kf_id(model):
    r = make_resetter(model)
    assert r._kf_id >= 0


# ── 2. reset() return type ────────────────────────────────────────────────────

def test_reset_returns_episode_info(world, model):
    r   = make_resetter_full(world)
    d   = fresh_data(model)
    info = r.reset(model, d, np.random.default_rng(0))
    assert isinstance(info, EpisodeInfo)
    assert isinstance(info.robot_xy,   np.ndarray)
    assert isinstance(info.robot_yaw,  float)
    assert isinstance(info.goal_xyz,   np.ndarray)
    assert isinstance(info.n_active_obstacles, int)


def test_episode_info_shapes(world, model):
    r    = make_resetter_full(world)
    d    = fresh_data(model)
    info = r.reset(model, d, np.random.default_rng(1))
    assert info.robot_xy.shape  == (2,)
    assert info.goal_xyz.shape  == (3,)


# ── 3. Arm pose preserved ────────────────────────────────────────────────────

def test_arm_joints_home_pose_after_reset(world, model):
    """j2 should be ~π/2, all others ~0 — home keyframe values."""
    r    = make_resetter_full(world)
    d    = fresh_data(model)
    r.reset(model, d, np.random.default_rng(2))
    # arm joints start at qpos[9] (0-indexed: base 0-6, wheels 7-8, arm 9-14)
    arm = d.qpos[9:15]
    assert arm[1] == pytest.approx(math.pi / 2, abs=1e-4), "j2 should be π/2"
    for i in [0, 2, 3, 4, 5]:
        assert arm[i] == pytest.approx(0.0, abs=1e-4), f"arm[{i}] should be 0"


# ── 4–8. randomise_robot_spawn() ─────────────────────────────────────────────

def test_spawn_xy_in_range(world, model):
    r    = make_resetter_full(world)
    cfg  = r._spawn_cfg
    for s in range(10):
        d   = fresh_data(model)
        xy, _ = r.randomise_robot_spawn(model, d, np.random.default_rng(s))
        assert cfg.x_range[0] - 0.01 <= xy[0] <= cfg.x_range[1] + 0.01
        assert cfg.y_range[0] - 0.01 <= xy[1] <= cfg.y_range[1] + 0.01


def test_spawn_z_from_keyframe(world, model, kf_id):
    r = make_resetter_full(world)
    d = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d, kf_id)
    expected_z = float(d.qpos[2])

    d2  = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d2, kf_id)
    r.randomise_robot_spawn(model, d2, np.random.default_rng(0))
    assert d2.qpos[2] == pytest.approx(expected_z, abs=1e-6)


def test_spawn_yaw_in_range(world, model):
    r   = make_resetter_full(world)
    cfg = r._spawn_cfg
    for s in range(10):
        d   = fresh_data(model)
        mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
        _, yaw = r.randomise_robot_spawn(model, d, np.random.default_rng(s))
        assert cfg.yaw_range[0] - 0.01 <= yaw <= cfg.yaw_range[1] + 0.01


def test_spawn_quaternion_unit_norm(world, model):
    r = make_resetter_full(world)
    d = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
    r.randomise_robot_spawn(model, d, np.random.default_rng(5))
    quat = d.qpos[3:7]
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-6)


def test_spawn_base_velocity_zeroed(world, model):
    r = make_resetter_full(world)
    d = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
    d.qvel[0:6] = 99.0   # dirty velocities
    r.randomise_robot_spawn(model, d, np.random.default_rng(0))
    np.testing.assert_array_equal(d.qvel[0:6], np.zeros(6))


def test_spawn_clears_obstacles(world, model, state):
    """Robot spawn must be ≥ clear_r_robot from every active obstacle."""
    r    = make_resetter_full(world)
    cfg  = r._spawn_cfg
    rng  = np.random.default_rng(7)
    d    = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
    r.randomise_obstacles(d, rng)
    xy, _ = r.randomise_robot_spawn(model, d, rng)
    for i in range(state._cfg.n_obstacles):
        obs_pos = state.obstacle_pos(d, i)
        if obs_pos[2] > -1.0:   # active
            dist = math.hypot(xy[0] - obs_pos[0], xy[1] - obs_pos[1])
            assert dist >= cfg.clear_r_robot - 0.01, \
                f"robot spawned too close to obstacle {i}: dist={dist:.3f}"


# ── 10–11. randomise_obstacles() ─────────────────────────────────────────────

def test_randomise_obstacles_active_not_underground(world, model, state):
    r   = make_resetter_full(world)
    d   = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
    r.randomise_obstacles(d, np.random.default_rng(10))
    for i in range(state._cfg.n_obstacles):
        z = float(state.obstacle_pos(d, i)[2])
        assert z > -1.0, f"obstacle {i} still underground"


def test_randomise_obstacles_noop_without_world_state(model, kf_id):
    r   = EpisodeResetter(home_kf_id=kf_id)   # no world_state
    d   = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d, kf_id)
    qpos_before = d.qpos.copy()
    r.randomise_obstacles(d, np.random.default_rng(0))
    # qpos unchanged — no obstacles to move
    np.testing.assert_array_equal(d.qpos, qpos_before)


# ── 12–16. randomise_goal() ───────────────────────────────────────────────────

def test_goal_relative_returns_3d(world, model):
    r    = make_resetter_full(world)
    d    = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
    xyz  = r.randomise_goal(d, np.random.default_rng(0),
                             robot_xy=np.array([0., 0.]), robot_yaw=0.0)
    assert xyz.shape == (3,)


def test_goal_relative_in_fwd_range(world, model):
    """At yaw=0 the goal x - robot x should equal the fwd sample."""
    r    = make_resetter_full(world)
    cfg  = r._goal_cfg
    robot_xy = np.array([0.0, 0.0])
    for s in range(10):
        d   = fresh_data(model)
        mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
        xyz = r.randomise_goal(d, np.random.default_rng(s),
                                robot_xy=robot_xy, robot_yaw=0.0)
        # At yaw=0: gx = rx + fwd, so fwd = gx - rx
        fwd = xyz[0] - robot_xy[0]
        assert cfg.fwd_range[0] - 0.01 <= fwd <= cfg.fwd_range[1] + 0.01


def test_goal_relative_rotates_with_yaw(world, model):
    """At yaw=π/2 (facing +Y), the goal should be ahead in +Y direction."""
    r    = make_resetter_full(world)
    robot_xy  = np.array([0.0, 0.0])
    robot_yaw = math.pi / 2   # facing +Y
    d   = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
    xyz = r.randomise_goal(d, np.random.default_rng(42),
                            robot_xy=robot_xy, robot_yaw=robot_yaw)
    # At yaw=π/2, forward = +Y
    # gx = rx - lat·sin(π/2) = -lat  → roughly 0
    # gy = ry + fwd·sin(π/2) = fwd   → positive
    assert xyz[1] > 0.30, f"goal y={xyz[1]} not in front at yaw=π/2"


def test_goal_world_mode_in_range(world, model):
    cfg  = GoalConfig(mode="world", x_range=(1.0, 2.0), y_range=(-0.5, 0.5))
    r    = EpisodeResetter(home_kf_id=world[3], world_state=world[1], goal_cfg=cfg)
    for s in range(10):
        d   = fresh_data(model)
        mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
        xyz = r.randomise_goal(d, np.random.default_rng(s))
        assert 1.0 - 0.01 <= xyz[0] <= 2.0 + 0.01
        assert -0.5 - 0.01 <= xyz[1] <= 0.5 + 0.01


def test_goal_updates_pickable_qpos(world, model, state):
    """First pickable object's qpos[qadr:qadr+3] should match goal xyz."""
    r   = make_resetter_full(world)
    d   = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
    xyz = r.randomise_goal(d, np.random.default_rng(0),
                            robot_xy=np.array([0., 0.]), robot_yaw=0.0)
    entry    = state._pick[0]
    pick_pos = d.qpos[entry.qadr : entry.qadr + 3]
    np.testing.assert_allclose(pick_pos, xyz, atol=1e-9)


def test_goal_z_at_floor(world, model):
    r   = make_resetter_full(world)
    d   = fresh_data(model)
    mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
    xyz = r.randomise_goal(d, np.random.default_rng(0),
                            robot_xy=np.array([0., 0.]), robot_yaw=0.0)
    assert xyz[2] == pytest.approx(r._goal_cfg.z, abs=1e-9)


# ── 17–19. Reproducibility ────────────────────────────────────────────────────

def test_same_seed_same_episode_info(world, model):
    r = make_resetter_full(world)

    d1   = fresh_data(model)
    info1 = r.reset(model, d1, np.random.default_rng(99))

    d2   = fresh_data(model)
    info2 = r.reset(model, d2, np.random.default_rng(99))

    np.testing.assert_array_equal(info1.robot_xy,  info2.robot_xy)
    assert info1.robot_yaw == info2.robot_yaw
    np.testing.assert_array_equal(info1.goal_xyz,  info2.goal_xyz)


def test_different_seeds_different_robot_xy(world, model):
    r = make_resetter_full(world)

    d1   = fresh_data(model)
    info1 = r.reset(model, d1, np.random.default_rng(0))

    d2   = fresh_data(model)
    info2 = r.reset(model, d2, np.random.default_rng(1))

    assert not np.allclose(info1.robot_xy, info2.robot_xy), \
        "robot_xy identical for different seeds"


def test_different_seeds_different_goal(world, model):
    r = make_resetter_full(world)
    goals = set()
    for s in range(10):
        d    = fresh_data(model)
        info = r.reset(model, d, np.random.default_rng(s))
        goals.add(tuple(np.round(info.goal_xyz, 4)))
    assert len(goals) > 1, "goal always the same across seeds"


# ── 20. Domain rand applied ───────────────────────────────────────────────────

def test_domain_rand_applied_during_reset(world, model):
    """Floor colour should differ between resets (domain rand active)."""
    r = make_resetter_full(world)
    floor_gid = r._domain_rand._floor_gid

    d1 = fresh_data(model)
    r.reset(model, d1, np.random.default_rng(0))
    rgba0 = model.geom_rgba[floor_gid, :3].copy()

    d2 = fresh_data(model)
    r.reset(model, d2, np.random.default_rng(1))
    rgba1 = model.geom_rgba[floor_gid, :3].copy()

    assert not np.allclose(rgba0, rgba1), "floor colour never changed"


# ── 21. Works without world_state and domain_rand ────────────────────────────

def test_reset_minimal(model, kf_id):
    r    = EpisodeResetter(home_kf_id=kf_id)
    d    = fresh_data(model)
    info = r.reset(model, d, np.random.default_rng(0))
    assert isinstance(info, EpisodeInfo)
    assert info.n_active_obstacles == 0


# ── 22–23. Physics stability ─────────────────────────────────────────────────

def test_mj_forward_no_nan_after_reset(world, model):
    r = make_resetter_full(world)
    d = fresh_data(model)
    r.reset(model, d, np.random.default_rng(5))
    # mj_forward already called inside reset(); check qpos is finite
    assert np.all(np.isfinite(d.qpos)), "NaN in qpos after reset"
    assert np.all(np.isfinite(d.qvel)), "NaN in qvel after reset"


def test_50_physics_steps_no_nan(world, model):
    r = make_resetter_full(world)
    d = fresh_data(model)
    r.reset(model, d, np.random.default_rng(6))
    for step in range(50):
        mujoco.mj_step(model, d)
        assert np.all(np.isfinite(d.qpos)), f"NaN in qpos at step {step}"


# ── 24. n_active_obstacles ────────────────────────────────────────────────────

def test_n_active_obstacles_matches_config(world, model, state):
    r    = make_resetter_full(world)
    d    = fresh_data(model)
    info = r.reset(model, d, np.random.default_rng(0))
    assert info.n_active_obstacles == state._cfg.n_obstacles


def test_n_active_obstacles_override(world, model):
    r = EpisodeResetter(
        home_kf_id  = world[3],
        world_state = world[1],
        n_obstacles = 2,
    )
    d    = fresh_data(model)
    info = r.reset(model, d, np.random.default_rng(0))
    assert info.n_active_obstacles == 2


# ── 25. Yaw=0 goal is in front ────────────────────────────────────────────────

def test_goal_in_front_at_yaw_zero(world, model):
    """At yaw=0 (facing +X), goal x should be > robot x."""
    r     = make_resetter_full(world)
    rx    = 0.0
    robot_xy = np.array([rx, 0.0])
    for s in range(5):
        d   = fresh_data(model)
        mujoco.mj_resetDataKeyframe(model, d, r._kf_id)
        xyz = r.randomise_goal(d, np.random.default_rng(s),
                                robot_xy=robot_xy, robot_yaw=0.0)
        assert xyz[0] > rx + 0.35, f"goal not in front: gx={xyz[0]}, rx={rx}"


# ── 26. make_resetter raises on missing keyframe ──────────────────────────────

def test_make_resetter_raises_without_home_keyframe():
    bare = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body name="b"><freejoint/><geom size=".1"/></body>
          </worldbody>
        </mujoco>
    """)
    with pytest.raises(RuntimeError, match="home"):
        make_resetter(bare)
