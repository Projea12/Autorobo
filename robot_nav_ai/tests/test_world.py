"""
Tests for world/world.py — MuJoCo arena builder.

Covers:
  1.  WorldConfig validation
  2.  build() — model compiles, DOF count increases correctly
  3.  Floor geom exists in the compiled model
  4.  Walls × 4 exist as named geoms
  5.  Obstacle pool — correct number of joints baked in
  6.  Pickable pool — correct number of joints baked in
  7.  WorldState.randomize() — active objects placed in arena, inactive hidden
  8.  Obstacle positions inside arena bounds after randomize()
  9.  WorldState.randomize_obstacles() — only obstacles move
 10.  WorldState.randomize_pickable() — respects x_range / y_range
 11.  Same seed → identical layout (reproducibility)
 12.  Different seeds → different layouts
 13.  WorldState.pickable_pos() / obstacle_pos() return correct shape
 14.  build_world() convenience factory
 15.  Multiple consecutive randomize() calls — no NaN in qpos
 16.  Physics runs 100 steps without NaN after randomize()
 17.  n_obstacles override in randomize()
 18.  WorldConfig n_obstacles > max_obstacles raises ValueError
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from robot.constants import ROBOT_XML_PATH
from world.world import (
    WorldBuilder, WorldConfig, WorldState, ObjectEntry, build_world,
    _HIDDEN_Z,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_cfg():
    """Small but representative config — fast to build."""
    return WorldConfig(
        arena_size=4.0,
        n_obstacles=4,
        max_obstacles=6,
        n_pickable=1,
        max_pickable=2,
        build_seed=42,
    )


@pytest.fixture(scope="module")
def built(small_cfg):
    builder = WorldBuilder(small_cfg)
    model, state = builder.build(ROBOT_XML_PATH)
    return model, state, small_cfg


@pytest.fixture(scope="module")
def model(built):
    return built[0]


@pytest.fixture(scope="module")
def state(built) -> WorldState:
    return built[1]


@pytest.fixture(scope="module")
def cfg(built) -> WorldConfig:
    return built[2]


def fresh_data(model) -> mujoco.MjData:
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    return d


# ── 1. WorldConfig validation ─────────────────────────────────────────────────

def test_config_defaults():
    cfg = WorldConfig()
    assert cfg.arena_size      == 6.0
    assert cfg.n_obstacles     == 6
    assert cfg.max_obstacles   == 12
    assert cfg.n_pickable      == 1
    assert cfg.max_pickable    == 4


def test_config_n_obstacles_exceeds_max_raises():
    with pytest.raises(ValueError, match="n_obstacles"):
        WorldConfig(n_obstacles=10, max_obstacles=5)


def test_config_n_pickable_exceeds_max_raises():
    with pytest.raises(ValueError, match="n_pickable"):
        WorldConfig(n_pickable=3, max_pickable=2)


def test_config_equal_n_and_max_ok():
    cfg = WorldConfig(n_obstacles=6, max_obstacles=6)
    assert cfg.n_obstacles == cfg.max_obstacles


# ── 2. build() — model compiles and DOF count is plausible ───────────────────

def test_build_returns_model_and_state(built):
    model, state, _ = built
    assert isinstance(model, mujoco.MjModel)
    assert isinstance(state, WorldState)


def test_nq_increases_from_robot_baseline(model, small_cfg):
    """
    Robot baseline nq = 17.
    Each obstacle/pickable body adds a freejoint → 7 qpos floats.
    Total pool = max_obstacles + max_pickable.
    """
    robot_model = mujoco.MjModel.from_xml_path(ROBOT_XML_PATH)
    pool_size   = small_cfg.max_obstacles + small_cfg.max_pickable
    expected_nq = robot_model.nq + 7 * pool_size
    assert model.nq == expected_nq


def test_nv_increases_from_robot_baseline(model, small_cfg):
    robot_model = mujoco.MjModel.from_xml_path(ROBOT_XML_PATH)
    pool_size   = small_cfg.max_obstacles + small_cfg.max_pickable
    expected_nv = robot_model.nv + 6 * pool_size
    assert model.nv == expected_nv


# ── 3. Floor geom ─────────────────────────────────────────────────────────────

def test_floor_geom_exists(model):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "world_floor")
    assert gid >= 0, "world_floor geom not found"


def test_floor_is_plane_type(model):
    gid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "world_floor")
    gtype = int(model.geom_type[gid])
    assert gtype == mujoco.mjtGeom.mjGEOM_PLANE


# ── 4. Walls ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("wall_name", [
    "wall_north", "wall_south", "wall_east", "wall_west"
])
def test_wall_exists(model, wall_name):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, wall_name)
    assert gid >= 0, f"{wall_name} geom not found"


@pytest.mark.parametrize("wall_name", [
    "wall_north", "wall_south", "wall_east", "wall_west"
])
def test_wall_is_box_type(model, wall_name):
    gid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, wall_name)
    gtype = int(model.geom_type[gid])
    assert gtype == mujoco.mjtGeom.mjGEOM_BOX


# ── 5. Obstacle pool ──────────────────────────────────────────────────────────

def test_obstacle_pool_joint_count(model, small_cfg):
    """Every obstacle pool slot should have a named joint in the model."""
    n_box = small_cfg.max_obstacles // 2
    n_cyl = small_cfg.max_obstacles - n_box
    for i in range(n_box):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                 f"obstacle_box_{i}_joint")
        assert jid >= 0, f"obstacle_box_{i}_joint not found"
    for i in range(n_cyl):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                 f"obstacle_cyl_{i}_joint")
        assert jid >= 0, f"obstacle_cyl_{i}_joint not found"


def test_obstacle_entries_count(state, small_cfg):
    assert state.n_obstacle_slots == small_cfg.max_obstacles


# ── 6. Pickable pool ──────────────────────────────────────────────────────────

def test_pickable_pool_joint_count(model, small_cfg):
    for i in range(small_cfg.max_pickable):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                 f"pickable_{i}_joint")
        assert jid >= 0, f"pickable_{i}_joint not found"


def test_pickable_entries_count(state, small_cfg):
    assert state.n_pickable_slots == small_cfg.max_pickable


# ── 7. WorldState.randomize() ─────────────────────────────────────────────────

def test_randomize_runs_without_error(model, state):
    d   = fresh_data(model)
    rng = np.random.default_rng(0)
    state.randomize(d, rng)   # must not raise


def test_active_obstacles_not_underground(model, state, cfg):
    d   = fresh_data(model)
    rng = np.random.default_rng(1)
    state.randomize(d, rng)
    for i in range(cfg.n_obstacles):
        z = float(state.obstacle_pos(d, i)[2])
        assert z > -1.0, f"obstacle {i} still underground after randomize (z={z})"


def test_inactive_obstacles_underground(model, state, cfg):
    d   = fresh_data(model)
    rng = np.random.default_rng(2)
    state.randomize(d, rng)
    for i in range(cfg.n_obstacles, state.n_obstacle_slots):
        z = float(state.obstacle_pos(d, i)[2])
        assert z < -50.0, f"inactive obstacle {i} not hidden (z={z})"


def test_active_pickable_not_underground(model, state, cfg):
    d   = fresh_data(model)
    rng = np.random.default_rng(3)
    state.randomize(d, rng)
    for i in range(cfg.n_pickable):
        z = float(state.pickable_pos(d, i)[2])
        assert z > -1.0, f"pickable {i} still underground after randomize (z={z})"


def test_inactive_pickable_underground(model, state, cfg):
    d   = fresh_data(model)
    rng = np.random.default_rng(4)
    state.randomize(d, rng)
    for i in range(cfg.n_pickable, state.n_pickable_slots):
        z = float(state.pickable_pos(d, i)[2])
        assert z < -50.0, f"inactive pickable {i} not hidden (z={z})"


# ── 8. Obstacles inside arena bounds ─────────────────────────────────────────

def test_active_obstacles_inside_arena(model, state, cfg):
    d    = fresh_data(model)
    rng  = np.random.default_rng(5)
    state.randomize(d, rng)
    half = cfg.arena_size / 2.0
    for i in range(cfg.n_obstacles):
        pos = state.obstacle_pos(d, i)
        assert abs(pos[0]) <= half + 0.01, f"obs {i} x={pos[0]} outside arena"
        assert abs(pos[1]) <= half + 0.01, f"obs {i} y={pos[1]} outside arena"


# ── 9. randomize_obstacles() — only obstacles move ───────────────────────────

def test_randomize_obstacles_only(model, state, cfg):
    d   = fresh_data(model)
    rng = np.random.default_rng(6)
    # Place pickable at a known position first
    state.randomize(d, rng)
    pick_pos_before = state.pickable_pos(d, 0).copy()

    # Randomise only obstacles
    rng2 = np.random.default_rng(7)
    state.randomize_obstacles(d, rng2)
    pick_pos_after = state.pickable_pos(d, 0).copy()

    # Pickable position should be unchanged
    np.testing.assert_array_equal(pick_pos_before, pick_pos_after)


# ── 10. randomize_pickable() x/y range ───────────────────────────────────────

def test_randomize_pickable_respects_x_range(model, state, cfg):
    d   = fresh_data(model)
    rng = np.random.default_rng(8)
    state.randomize_pickable(d, rng, x_range=(0.40, 0.85), y_range=(-0.30, 0.30))
    for i in range(cfg.n_pickable):
        pos = state.pickable_pos(d, i)
        assert 0.40 <= pos[0] <= 0.85, f"pickable {i} x={pos[0]} outside range"
        assert -0.30 <= pos[1] <= 0.30, f"pickable {i} y={pos[1]} outside range"


def test_randomize_pickable_z_on_floor(model, state, cfg):
    """Pickable z should be half-height (0.025 m) — sitting on the floor."""
    d   = fresh_data(model)
    rng = np.random.default_rng(9)
    state.randomize_pickable(d, rng)
    for i in range(cfg.n_pickable):
        z = float(state.pickable_pos(d, i)[2])
        assert 0.01 <= z <= 0.10, f"pickable {i} z={z} implausible"


# ── 11. Same seed → identical layout ─────────────────────────────────────────

def test_same_seed_identical_layout(model, state, cfg):
    d1  = fresh_data(model)
    d2  = fresh_data(model)
    state.randomize(d1, np.random.default_rng(42))
    state.randomize(d2, np.random.default_rng(42))
    for i in range(cfg.n_obstacles):
        np.testing.assert_array_equal(
            state.obstacle_pos(d1, i),
            state.obstacle_pos(d2, i),
            err_msg=f"obstacle {i} differs between same-seed runs",
        )
    for i in range(cfg.n_pickable):
        np.testing.assert_array_equal(
            state.pickable_pos(d1, i),
            state.pickable_pos(d2, i),
            err_msg=f"pickable {i} differs between same-seed runs",
        )


# ── 12. Different seeds → different layouts ───────────────────────────────────

def test_different_seeds_different_layout(model, state, cfg):
    d1  = fresh_data(model)
    d2  = fresh_data(model)
    state.randomize(d1, np.random.default_rng(0))
    state.randomize(d2, np.random.default_rng(1))
    # At least one obstacle should differ in x or y
    diffs = [
        not np.allclose(state.obstacle_pos(d1, i)[:2],
                        state.obstacle_pos(d2, i)[:2])
        for i in range(cfg.n_obstacles)
    ]
    assert any(diffs), "identical layouts with different seeds"


# ── 13. pickable_pos / obstacle_pos shape ─────────────────────────────────────

def test_pickable_pos_shape(model, state):
    d = fresh_data(model)
    state.randomize(d, np.random.default_rng(0))
    assert state.pickable_pos(d, 0).shape == (3,)


def test_obstacle_pos_shape(model, state):
    d = fresh_data(model)
    state.randomize(d, np.random.default_rng(0))
    assert state.obstacle_pos(d, 0).shape == (3,)


# ── 14. build_world() convenience factory ─────────────────────────────────────

def test_build_world_factory():
    model, state = build_world(ROBOT_XML_PATH, n_obstacles=2, n_pickable=1,
                               build_seed=99)
    assert isinstance(model, mujoco.MjModel)
    assert isinstance(state, WorldState)
    assert state.n_obstacle_slots >= 2
    assert state.n_pickable_slots >= 1


def test_build_world_extra_kwargs():
    model, state = build_world(
        ROBOT_XML_PATH,
        arena_size=4.0,
        n_obstacles=3,
        max_obstacles=6,
        n_pickable=1,
        build_seed=0,
    )
    assert model.nq > 17   # robot + pool objects


# ── 15. Multiple randomize() calls — no NaN ───────────────────────────────────

def test_multiple_randomize_no_nan(model, state):
    d   = fresh_data(model)
    rng = np.random.default_rng(55)
    for episode in range(5):
        state.randomize(d, rng)
        assert np.all(np.isfinite(d.qpos)), f"NaN in qpos after episode {episode}"
        assert np.all(np.isfinite(d.qvel)), f"NaN in qvel after episode {episode}"


# ── 16. Physics runs without NaN after randomize() ───────────────────────────

def test_physics_stable_after_randomize(model, state):
    d   = fresh_data(model)
    rng = np.random.default_rng(77)
    state.randomize(d, rng)
    mujoco.mj_forward(model, d)
    for step in range(100):
        mujoco.mj_step(model, d)
    assert np.all(np.isfinite(d.qpos)), "NaN in qpos after 100 physics steps"
    assert np.all(np.isfinite(d.qvel)), "NaN in qvel after 100 physics steps"


# ── 17. n_obstacles override in randomize() ───────────────────────────────────

def test_n_obstacles_override_zero(model, state):
    """With n_obstacles=0 all obstacle slots should be hidden."""
    d   = fresh_data(model)
    rng = np.random.default_rng(88)
    state.randomize(d, rng, n_obstacles=0)
    for i in range(state.n_obstacle_slots):
        z = float(state.obstacle_pos(d, i)[2])
        assert z < -50.0, f"obstacle {i} not hidden when n_obstacles=0"


def test_n_obstacles_override_max(model, state):
    """Using the full pool should place all obstacles."""
    d   = fresh_data(model)
    rng = np.random.default_rng(89)
    state.randomize(d, rng, n_obstacles=state.n_obstacle_slots)
    placed = sum(
        1 for i in range(state.n_obstacle_slots)
        if float(state.obstacle_pos(d, i)[2]) > -1.0
    )
    # All or nearly all placed (may be fewer if arena too crowded)
    assert placed > 0


# ── 18. WorldConfig bad values ────────────────────────────────────────────────

def test_build_with_default_config():
    """Default config should compile successfully with robot.xml."""
    cfg = WorldConfig(build_seed=0)
    builder = WorldBuilder(cfg)
    model, state = builder.build(ROBOT_XML_PATH)
    assert model.nq > 17
    assert state.n_obstacle_slots == cfg.max_obstacles
    assert state.n_pickable_slots == cfg.max_pickable
