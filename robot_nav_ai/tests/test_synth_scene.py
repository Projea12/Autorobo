"""tests/test_synth_scene.py — SynthScene construction and reset tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.synth.scene import SynthScene, SceneConfig, ObjectSlot
from data.ycb.registry import REGISTRY, YCBCategory

import mujoco


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_scene():
    """Scene with 3 objects to keep tests fast."""
    names = ("002_master_chef_can", "003_cracker_box", "005_tomato_soup_can")
    cfg   = SceneConfig(object_names=names, min_objects=1, max_objects=2,
                        image_w=64, image_h=64)
    return SynthScene(cfg=cfg)


# ── construction ──────────────────────────────────────────────────────────────

def test_scene_builds(small_scene):
    assert small_scene.model is not None


def test_scene_slot_count(small_scene):
    assert len(small_scene.slots) == 3


def test_slots_are_objectslot(small_scene):
    for s in small_scene.slots:
        assert isinstance(s, ObjectSlot)


def test_slot_names(small_scene):
    names = {s.name for s in small_scene.slots}
    assert "002_master_chef_can" in names
    assert "003_cracker_box" in names


def test_slot_half_extents_positive(small_scene):
    for s in small_scene.slots:
        assert all(v > 0 for v in s.half_extents)


def test_slot_qadr_nonnegative(small_scene):
    for s in small_scene.slots:
        assert s.qadr >= 0


def test_slot_qadr_unique(small_scene):
    qadrs = [s.qadr for s in small_scene.slots]
    assert len(qadrs) == len(set(qadrs))


def test_model_nq_includes_freejoints(small_scene):
    # Each object has a freejoint = 7 qpos
    # nq ≥ 3 * 7 = 21
    assert small_scene.model.nq >= 3 * 7


def test_model_has_floor(small_scene):
    floor_id = mujoco.mj_name2id(
        small_scene.model, mujoco.mjtObj.mjOBJ_GEOM, "synth_floor"
    )
    assert floor_id >= 0


def test_make_data_returns_mjdata(small_scene):
    d = small_scene.make_data()
    assert isinstance(d, mujoco.MjData)


def test_make_data_fresh_each_call(small_scene):
    d1 = small_scene.make_data()
    d2 = small_scene.make_data()
    assert d1 is not d2


# ── reset ─────────────────────────────────────────────────────────────────────

def test_reset_returns_active_list(small_scene):
    rng = np.random.default_rng(0)
    d   = small_scene.make_data()
    active = small_scene.reset(d, rng)
    assert isinstance(active, list)
    assert len(active) >= 1


def test_reset_active_count_within_range(small_scene):
    cfg = small_scene.cfg
    rng = np.random.default_rng(1)
    d   = small_scene.make_data()
    for _ in range(10):
        active = small_scene.reset(d, rng)
        assert cfg.min_objects <= len(active) <= cfg.max_objects


def test_reset_active_objects_above_floor(small_scene):
    rng = np.random.default_rng(2)
    d   = small_scene.make_data()
    active = small_scene.reset(d, rng)
    for slot in active:
        z = d.qpos[slot.qadr + 2]
        assert z > -0.01, f"Active object {slot.name} is underground: z={z}"


def test_reset_inactive_objects_hidden(small_scene):
    rng    = np.random.default_rng(3)
    d      = small_scene.make_data()
    active = small_scene.reset(d, rng)
    active_names = {s.name for s in active}
    for slot in small_scene.slots:
        if slot.name not in active_names:
            z = d.qpos[slot.qadr + 2]
            assert z < -50.0, f"Inactive {slot.name} not hidden: z={z}"


def test_reset_active_xy_within_spread(small_scene):
    rng    = np.random.default_rng(4)
    d      = small_scene.make_data()
    spread = small_scene.cfg.object_spread
    active = small_scene.reset(d, rng)
    for slot in active:
        x = d.qpos[slot.qadr + 0]
        y = d.qpos[slot.qadr + 1]
        assert abs(x) <= spread + 0.01
        assert abs(y) <= spread + 0.01


def test_reset_quaternion_unit_length(small_scene):
    rng    = np.random.default_rng(5)
    d      = small_scene.make_data()
    active = small_scene.reset(d, rng)
    for slot in active:
        q = d.qpos[slot.qadr + 3: slot.qadr + 7]
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6


def test_reset_zero_base_velocities(small_scene):
    rng    = np.random.default_rng(6)
    d      = small_scene.make_data()
    active = small_scene.reset(d, rng)
    for slot in active:
        vel = d.qvel[slot.vadr: slot.vadr + 6]
        assert np.allclose(vel, 0.0)


def test_reset_qpos_no_nan(small_scene):
    rng = np.random.default_rng(7)
    d   = small_scene.make_data()
    small_scene.reset(d, rng)
    assert np.isfinite(d.qpos).all()


def test_reset_different_seeds_different_layout(small_scene):
    d1 = small_scene.make_data()
    d2 = small_scene.make_data()
    small_scene.reset(d1, np.random.default_rng(0))
    small_scene.reset(d2, np.random.default_rng(999))
    # At least one qpos should differ
    assert not np.allclose(d1.qpos, d2.qpos)


def test_reset_same_seed_reproducible(small_scene):
    d1 = small_scene.make_data()
    d2 = small_scene.make_data()
    small_scene.reset(d1, np.random.default_rng(42))
    small_scene.reset(d2, np.random.default_rng(42))
    assert np.allclose(d1.qpos, d2.qpos)


# ── object_pos / object_quat ──────────────────────────────────────────────────

def test_object_pos_shape(small_scene):
    rng    = np.random.default_rng(8)
    d      = small_scene.make_data()
    active = small_scene.reset(d, rng)
    for slot in active:
        pos = small_scene.object_pos(d, slot)
        assert pos.shape == (3,)


def test_object_pos_finite(small_scene):
    rng    = np.random.default_rng(9)
    d      = small_scene.make_data()
    active = small_scene.reset(d, rng)
    for slot in active:
        assert np.isfinite(small_scene.object_pos(d, slot)).all()


def test_object_quat_shape(small_scene):
    rng    = np.random.default_rng(10)
    d      = small_scene.make_data()
    active = small_scene.reset(d, rng)
    for slot in active:
        assert small_scene.object_quat(d, slot).shape == (4,)


# ── can geometry ─────────────────────────────────────────────────────────────

def test_can_object_cylinder_geom():
    """Cylinder objects should have mjGEOM_CYLINDER geom type."""
    names = ("002_master_chef_can",)
    cfg   = SceneConfig(object_names=names, min_objects=1, max_objects=1)
    scene = SynthScene(cfg=cfg)
    geom_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM, "pool_002_master_chef_can_col"
    )
    assert geom_id >= 0
    assert scene.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER


def test_box_object_box_geom():
    """Box objects should have mjGEOM_BOX geom type."""
    names = ("003_cracker_box",)
    cfg   = SceneConfig(object_names=names, min_objects=1, max_objects=1)
    scene = SynthScene(cfg=cfg)
    geom_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM, "pool_003_cracker_box_col"
    )
    assert geom_id >= 0
    assert scene.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX


# ── full 21-object pool (smoke) ───────────────────────────────────────────────

def test_full_pool_compiles():
    scene = SynthScene()
    assert len(scene.slots) == 21


def test_full_pool_reset(  ):
    scene  = SynthScene()
    d      = scene.make_data()
    rng    = np.random.default_rng(0)
    active = scene.reset(d, rng)
    assert len(active) >= 1
    assert np.isfinite(d.qpos).all()
