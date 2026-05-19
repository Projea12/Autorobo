"""
Tests for env/domain_rand.py — DomainRandomizer.

Covers:
  1.  DomainRandomizer constructs without error
  2.  snapshot() returns correct keys
  3.  Floor color changes after randomize()
  4.  Floor friction changes after randomize()
  5.  Wall color changes
  6.  Obstacle color changes
  7.  Obstacle friction changes
  8.  Pickable color changes
  9.  Pickable friction changes
 10.  Pickable mass changes and stays in configured range
 11.  Pickable size changes and stays in nominal ± frac range
 12.  Pickable inertia is recomputed consistently with new size/mass
 13.  Wheel friction changes
 14.  Joint damping changes within ± frac of nominal
 15.  Headlight ambient changes
 16.  Sun light diffuse changes
 17.  Fill light active toggled (both states seen over many calls)
 18.  All values finite after randomize()
 19.  Same seed → identical result
 20.  Different seeds → different results
 21.  Physics stable (no NaN) after 50 steps following randomize()
 22.  Disabled flags — verify nothing changes when flag=False
 23.  _box_inertia() correctness
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from robot.constants import ROBOT_XML_PATH
from world.world import WorldBuilder, WorldConfig
from env.domain_rand import DomainRandomizer, DomainRandConfig, DEFAULT_CONFIG, _box_inertia


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_state():
    cfg   = WorldConfig(build_seed=7)
    model, state = WorldBuilder(cfg).build(ROBOT_XML_PATH)
    return model, state


@pytest.fixture(scope="module")
def model(model_state):
    return model_state[0]


@pytest.fixture(scope="module")
def rand(model):
    return DomainRandomizer(model)


def fresh_data(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    return d


def do_randomize(model, rand, seed=0):
    d   = fresh_data(model)
    rng = np.random.default_rng(seed)
    rand.randomize(model, d, rng)
    return d


# ── 1. construction ───────────────────────────────────────────────────────────

def test_constructs(model):
    r = DomainRandomizer(model)
    assert r is not None


def test_constructs_custom_config(model):
    cfg = DomainRandConfig(randomize_lighting=False, randomize_floor_color=False)
    r   = DomainRandomizer(model, cfg)
    assert r is not None


# ── 2. snapshot() ─────────────────────────────────────────────────────────────

def test_snapshot_keys(model, rand):
    snap = rand.snapshot(model)
    for key in ("floor_rgba", "floor_fric", "obs_rgba", "obs_fric",
                "pick_rgba", "pick_fric", "pick_size", "pick_mass",
                "dof_damping", "hl_ambient", "hl_diffuse"):
        assert key in snap, f"snapshot missing key: {key}"


def test_snapshot_shapes(model, rand):
    snap = rand.snapshot(model)
    assert snap["hl_ambient"].shape == (3,)
    assert snap["hl_diffuse"].shape == (3,)
    assert snap["dof_damping"].shape == (6,)


# ── 3–4. Floor ────────────────────────────────────────────────────────────────

def test_floor_color_changes(model, rand):
    snap_before = rand.snapshot(model)["floor_rgba"].copy()
    do_randomize(model, rand, seed=10)
    snap_after  = rand.snapshot(model)["floor_rgba"]
    assert not np.allclose(snap_before[:3], snap_after[:3]), "floor color unchanged"


def test_floor_color_in_range(model, rand):
    cfg = rand._cfg
    do_randomize(model, rand, seed=11)
    grey = float(model.geom_rgba[rand._floor_gid, 0])
    assert cfg.floor_grey_lo - 0.01 <= grey <= cfg.floor_grey_hi + 0.01


def test_floor_friction_changes(model, rand):
    before = float(model.geom_friction[rand._floor_gid, 0])
    do_randomize(model, rand, seed=12)
    after  = float(model.geom_friction[rand._floor_gid, 0])
    # With many seeds, friction should eventually differ
    changed = False
    for s in range(20):
        do_randomize(model, rand, seed=s)
        if float(model.geom_friction[rand._floor_gid, 0]) != before:
            changed = True
            break
    assert changed, "floor friction never changed over 20 seeds"


def test_floor_friction_in_range(model, rand):
    cfg = rand._cfg
    for s in range(5):
        do_randomize(model, rand, seed=100 + s)
        fric = float(model.geom_friction[rand._floor_gid, 0])
        assert cfg.floor_friction_lo - 0.01 <= fric <= cfg.floor_friction_hi + 0.01


# ── 5. Walls ──────────────────────────────────────────────────────────────────

def test_wall_color_changes(model, rand):
    if not rand._wall_gids:
        pytest.skip("no wall geoms in model")
    snap_before = model.geom_rgba[rand._wall_gids].copy()
    do_randomize(model, rand, seed=20)
    snap_after  = model.geom_rgba[rand._wall_gids].copy()
    assert not np.allclose(snap_before[:, :3], snap_after[:, :3])


def test_wall_color_in_range(model, rand):
    cfg = rand._cfg
    do_randomize(model, rand, seed=21)
    for gid in rand._wall_gids:
        grey = float(model.geom_rgba[gid, 0])
        assert cfg.wall_grey_lo - 0.01 <= grey <= cfg.wall_grey_hi + 0.01


# ── 6–7. Obstacles ────────────────────────────────────────────────────────────

def test_obstacle_color_changes(model, rand):
    if not rand._obs_gids:
        pytest.skip("no obstacle geoms")
    snap_before = model.geom_rgba[rand._obs_gids].copy()
    do_randomize(model, rand, seed=30)
    snap_after  = model.geom_rgba[rand._obs_gids].copy()
    assert not np.allclose(snap_before[:, :3], snap_after[:, :3])


def test_obstacle_friction_in_range(model, rand):
    cfg = rand._cfg
    if not rand._obs_gids:
        pytest.skip("no obstacle geoms")
    do_randomize(model, rand, seed=31)
    for gid in rand._obs_gids:
        fric = float(model.geom_friction[gid, 0])
        assert cfg.obs_friction_lo - 0.01 <= fric <= cfg.obs_friction_hi + 0.01


# ── 8–12. Pickable objects ────────────────────────────────────────────────────

def test_pickable_color_changes(model, rand):
    if not rand._pick_gids:
        pytest.skip("no pickable geoms")
    snap_before = model.geom_rgba[rand._pick_gids].copy()
    do_randomize(model, rand, seed=40)
    snap_after  = model.geom_rgba[rand._pick_gids].copy()
    assert not np.allclose(snap_before[:, :3], snap_after[:, :3])


def test_pickable_friction_in_range(model, rand):
    cfg = rand._cfg
    if not rand._pick_gids:
        pytest.skip("no pickable geoms")
    do_randomize(model, rand, seed=41)
    for gid in rand._pick_gids:
        fric = float(model.geom_friction[gid, 0])
        assert cfg.pick_friction_lo - 0.01 <= fric <= cfg.pick_friction_hi + 0.01


def test_pickable_mass_in_range(model, rand):
    cfg = rand._cfg
    if not rand._pick_bids:
        pytest.skip("no pickable bodies")
    do_randomize(model, rand, seed=42)
    for bid in rand._pick_bids:
        mass = float(model.body_mass[bid])
        assert cfg.pick_mass_lo - 0.01 <= mass <= cfg.pick_mass_hi + 0.01


def test_pickable_mass_changes(model, rand):
    if not rand._pick_bids:
        pytest.skip("no pickable bodies")
    mass_before = float(model.body_mass[rand._pick_bids[0]])
    changed = False
    for s in range(20):
        do_randomize(model, rand, seed=200 + s)
        if float(model.body_mass[rand._pick_bids[0]]) != mass_before:
            changed = True
            break
    assert changed, "pickable mass never changed"


def test_pickable_size_in_range(model, rand):
    cfg = rand._cfg
    if not rand._pick_gids:
        pytest.skip("no pickable geoms")
    nom = rand._nom_pick_size[0].copy()
    lo  = nom * (1.0 - cfg.pick_size_frac) - 1e-6
    hi  = nom * (1.0 + cfg.pick_size_frac) + 1e-6
    do_randomize(model, rand, seed=43)
    size = model.geom_size[rand._pick_gids[0]]
    assert np.all(size >= lo), f"size {size} below lo {lo}"
    assert np.all(size <= hi), f"size {size} above hi {hi}"


def test_pickable_inertia_consistent_with_size_and_mass(model, rand):
    """After randomize, inertia should match _box_inertia(mass, size)."""
    if not rand._pick_gids:
        pytest.skip("no pickable geoms")
    do_randomize(model, rand, seed=44)
    gid  = rand._pick_gids[0]
    bid  = rand._pick_bids[0]
    mass = float(model.body_mass[bid])
    half = model.geom_size[gid].copy()
    expected = _box_inertia(mass, half)
    actual   = model.body_inertia[bid].copy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


# ── 13. Wheel friction ────────────────────────────────────────────────────────

def test_wheel_friction_in_range(model, rand):
    cfg = rand._cfg
    if not rand._wheel_gids:
        pytest.skip("no wheel geoms")
    do_randomize(model, rand, seed=50)
    for gid in rand._wheel_gids:
        fric = float(model.geom_friction[gid, 0])
        assert cfg.wheel_friction_lo - 0.01 <= fric <= cfg.wheel_friction_hi + 0.01


def test_wheel_friction_changes(model, rand):
    if not rand._wheel_gids:
        pytest.skip("no wheel geoms")
    gid    = rand._wheel_gids[0]
    before = float(model.geom_friction[gid, 0])
    changed = False
    for s in range(20):
        do_randomize(model, rand, seed=300 + s)
        if float(model.geom_friction[gid, 0]) != before:
            changed = True
            break
    assert changed, "wheel friction never changed"


# ── 14. Joint damping ─────────────────────────────────────────────────────────

def test_joint_damping_in_range(model, rand):
    cfg = rand._cfg
    nom = rand._nom_dof_damping.copy()
    do_randomize(model, rand, seed=60)
    sl  = rand._ARM_DOF_SLICE
    for i, n in enumerate(nom):
        if n == 0.0:
            continue
        actual = float(model.dof_damping[sl.start + i])
        assert n * (1 - cfg.joint_damping_frac) - 1e-9 <= actual
        assert actual <= n * (1 + cfg.joint_damping_frac) + 1e-9


def test_joint_damping_changes(model, rand):
    sl     = rand._ARM_DOF_SLICE
    before = float(model.dof_damping[sl.start])
    changed = False
    for s in range(20):
        do_randomize(model, rand, seed=400 + s)
        if float(model.dof_damping[sl.start]) != before:
            changed = True
            break
    assert changed, "joint damping never changed"


# ── 15–17. Lighting ───────────────────────────────────────────────────────────

def test_headlight_ambient_changes(model, rand):
    snap_before = rand.snapshot(model)["hl_ambient"].copy()
    do_randomize(model, rand, seed=70)
    snap_after  = rand.snapshot(model)["hl_ambient"]
    assert not np.allclose(snap_before, snap_after)


def test_headlight_ambient_in_range(model, rand):
    cfg = rand._cfg
    do_randomize(model, rand, seed=71)
    for i in range(3):
        v = float(model.vis.headlight.ambient[i])
        assert cfg.headlight_ambient_lo - 0.01 <= v <= cfg.headlight_ambient_hi + 0.01


def test_sun_diffuse_changes(model, rand):
    if rand._sun_lid < 0:
        pytest.skip("no sun light")
    before = model.light_diffuse[rand._sun_lid].copy()
    changed = False
    for s in range(20):
        do_randomize(model, rand, seed=500 + s)
        if not np.allclose(model.light_diffuse[rand._sun_lid], before):
            changed = True
            break
    assert changed


def test_fill_light_toggled_both_states(model, rand):
    if rand._fill_lid < 0:
        pytest.skip("no fill light")
    active_states = set()
    for s in range(50):
        do_randomize(model, rand, seed=s)
        active_states.add(int(model.light_active[rand._fill_lid]))
        if len(active_states) == 2:
            break
    assert 0 in active_states, "fill light never off"
    assert 1 in active_states, "fill light never on"


# ── 18. All values finite ─────────────────────────────────────────────────────

def test_all_rgba_finite_after_randomize(model, rand):
    do_randomize(model, rand, seed=80)
    assert np.all(np.isfinite(model.geom_rgba)), "NaN/Inf in geom_rgba"


def test_all_friction_finite_after_randomize(model, rand):
    do_randomize(model, rand, seed=81)
    assert np.all(np.isfinite(model.geom_friction)), "NaN/Inf in geom_friction"


def test_all_damping_finite_after_randomize(model, rand):
    do_randomize(model, rand, seed=82)
    assert np.all(np.isfinite(model.dof_damping)), "NaN/Inf in dof_damping"


# ── 19–20. Reproducibility ────────────────────────────────────────────────────

def test_same_seed_same_floor_color(model, rand):
    do_randomize(model, rand, seed=99)
    rgba1 = model.geom_rgba[rand._floor_gid].copy()
    do_randomize(model, rand, seed=99)
    rgba2 = model.geom_rgba[rand._floor_gid].copy()
    np.testing.assert_array_equal(rgba1, rgba2)


def test_same_seed_same_damping(model, rand):
    do_randomize(model, rand, seed=99)
    damp1 = model.dof_damping[rand._ARM_DOF_SLICE].copy()
    do_randomize(model, rand, seed=99)
    damp2 = model.dof_damping[rand._ARM_DOF_SLICE].copy()
    np.testing.assert_array_equal(damp1, damp2)


def test_different_seeds_different_floor_color(model, rand):
    do_randomize(model, rand, seed=0)
    rgba0 = model.geom_rgba[rand._floor_gid].copy()
    do_randomize(model, rand, seed=1)
    rgba1 = model.geom_rgba[rand._floor_gid].copy()
    assert not np.allclose(rgba0[:3], rgba1[:3])


# ── 21. Physics stable after randomize ───────────────────────────────────────

def test_physics_stable_50_steps(model, rand):
    d = fresh_data(model)
    do_randomize(model, rand, seed=77)
    mujoco.mj_forward(model, d)
    for step in range(50):
        mujoco.mj_step(model, d)
    assert np.all(np.isfinite(d.qpos)), "NaN in qpos after 50 steps"
    assert np.all(np.isfinite(d.qvel)), "NaN in qvel after 50 steps"


# ── 22. Disabled flags ────────────────────────────────────────────────────────

def test_disabled_floor_color_unchanged(model):
    cfg  = DomainRandConfig(randomize_floor_color=False, randomize_floor_friction=False)
    rand = DomainRandomizer(model, cfg)
    gid  = rand._floor_gid
    if gid < 0:
        pytest.skip("no world_floor")
    rgba_before = model.geom_rgba[gid].copy()
    fric_before = model.geom_friction[gid].copy()
    d   = fresh_data(model)
    rng = np.random.default_rng(0)
    rand.randomize(model, d, rng)
    np.testing.assert_array_equal(model.geom_rgba[gid], rgba_before)
    np.testing.assert_array_equal(model.geom_friction[gid], fric_before)


def test_disabled_lighting_unchanged(model):
    cfg  = DomainRandConfig(randomize_lighting=False)
    rand = DomainRandomizer(model, cfg)
    amb_before = np.array(model.vis.headlight.ambient[:])
    d   = fresh_data(model)
    rng = np.random.default_rng(0)
    rand.randomize(model, d, rng)
    np.testing.assert_array_equal(np.array(model.vis.headlight.ambient[:]), amb_before)


def test_disabled_joint_damping_unchanged(model):
    cfg  = DomainRandConfig(randomize_joint_damping=False)
    rand = DomainRandomizer(model, cfg)
    sl   = rand._ARM_DOF_SLICE
    damp_before = model.dof_damping[sl].copy()
    d   = fresh_data(model)
    rng = np.random.default_rng(0)
    rand.randomize(model, d, rng)
    np.testing.assert_array_equal(model.dof_damping[sl], damp_before)


# ── 23. _box_inertia() ────────────────────────────────────────────────────────

def test_box_inertia_unit_cube():
    """1 kg unit cube: I = 1*(0.5²+0.5²)/3 = 1/6 per axis."""
    half = np.array([0.5, 0.5, 0.5])
    I    = _box_inertia(1.0, half)
    expected = 1.0 * (0.5**2 + 0.5**2) / 3.0
    np.testing.assert_allclose(I, [expected, expected, expected], rtol=1e-9)


def test_box_inertia_scales_with_mass():
    half = np.array([0.025, 0.025, 0.025])
    I1   = _box_inertia(0.2, half)
    I2   = _box_inertia(0.4, half)
    np.testing.assert_allclose(I2, 2.0 * I1, rtol=1e-9)


def test_box_inertia_positive():
    half = np.array([0.01, 0.03, 0.05])
    I    = _box_inertia(0.3, half)
    assert np.all(I > 0)
