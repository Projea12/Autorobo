"""tests/test_ycb_registry.py — Unit tests for data/ycb/registry.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.ycb.registry import (
    YCBObject, YCBCategory, YCBRegistry, REGISTRY, _CATALOG
)


# ── catalog completeness ───────────────────────────────────────────────────────

def test_catalog_has_21_objects():
    assert len(_CATALOG) == 21


def test_registry_len_matches_catalog():
    assert len(REGISTRY) == len(_CATALOG)


def test_no_duplicate_names():
    names = [o.name for o in _CATALOG]
    assert len(names) == len(set(names))


def test_no_duplicate_labels():
    labels = [o.label for o in _CATALOG]
    assert len(labels) == len(set(labels))


def test_all_names_have_number_prefix():
    for o in REGISTRY:
        parts = o.name.split("_", 1)
        assert len(parts) == 2
        assert parts[0].isdigit(), f"{o.name} missing numeric prefix"


def test_all_categories_are_ycbcategory():
    for o in REGISTRY:
        assert isinstance(o.category, YCBCategory)


# ── physical properties ────────────────────────────────────────────────────────

def test_all_masses_positive():
    for o in REGISTRY:
        assert o.mass_kg > 0, f"{o.name} mass={o.mass_kg}"


def test_all_half_extents_positive():
    for o in REGISTRY:
        assert all(v > 0 for v in o.half_extents), \
            f"{o.name} half_extents={o.half_extents}"


def test_all_frictions_in_valid_range():
    for o in REGISTRY:
        assert 0.0 < o.friction <= 3.0, \
            f"{o.name} friction={o.friction} out of [0, 3]"


def test_all_masses_plausible():
    for o in REGISTRY:
        assert 0.010 <= o.mass_kg <= 2.0, \
            f"{o.name} mass={o.mass_kg} kg outside plausible range"


def test_all_dims_plausible_cm():
    for o in REGISTRY:
        dims_cm = [v * 200 for v in o.half_extents]   # full size
        for d in dims_cm:
            assert 1.0 <= d <= 30.0, \
                f"{o.name} dim={d:.1f} cm outside plausible range"


# ── derived properties ─────────────────────────────────────────────────────────

def test_size_is_twice_half_extents():
    obj = REGISTRY["002_master_chef_can"]
    for s, h in zip(obj.size, obj.half_extents):
        assert abs(s - 2.0 * h) < 1e-9


def test_resting_height_equals_half_z():
    obj = REGISTRY["005_tomato_soup_can"]
    assert abs(obj.resting_height - obj.half_extents[2]) < 1e-9


def test_download_name_suffix():
    obj = REGISTRY["002_master_chef_can"]
    assert obj.download_name == "002_master_chef_can_google_16k"


def test_str_representation():
    obj = REGISTRY["011_banana"]
    s   = str(obj)
    assert "banana" in s
    assert "mass=" in s


# ── lookup ────────────────────────────────────────────────────────────────────

def test_lookup_by_canonical_name():
    obj = REGISTRY["002_master_chef_can"]
    assert obj.name == "002_master_chef_can"


def test_lookup_by_short_label():
    obj = REGISTRY["master_chef_can"]
    assert obj.name == "002_master_chef_can"


def test_lookup_missing_raises_keyerror():
    with pytest.raises(KeyError, match="not found"):
        _ = REGISTRY["999_nonexistent"]


def test_contains_canonical():
    assert "002_master_chef_can" in REGISTRY


def test_contains_label():
    assert "master_chef_can" in REGISTRY


def test_not_contains_garbage():
    assert "not_a_real_object" not in REGISTRY


def test_get_returns_none_for_missing():
    assert REGISTRY.get("999_missing") is None


def test_get_returns_default():
    sentinel = object()
    result = REGISTRY.get("999_missing", default=sentinel)
    assert result is sentinel


# ── iteration ─────────────────────────────────────────────────────────────────

def test_iteration_yields_all():
    items = list(REGISTRY)
    assert len(items) == 21


def test_iteration_yields_ycbobject():
    for obj in REGISTRY:
        assert isinstance(obj, YCBObject)


# ── filtering ─────────────────────────────────────────────────────────────────

def test_by_category_can():
    cans = REGISTRY.by_category("can")
    assert len(cans) == 4
    assert all(o.category == YCBCategory.CAN for o in cans)


def test_by_category_box():
    boxes = REGISTRY.by_category("box")
    assert all(o.category == YCBCategory.BOX for o in boxes)


def test_by_category_enum():
    cans = REGISTRY.by_category(YCBCategory.CAN)
    assert len(cans) == 4


def test_by_category_invalid_raises():
    with pytest.raises(ValueError):
        REGISTRY.by_category("furniture")


def test_graspable_all_graspable():
    for obj in REGISTRY.graspable():
        assert obj.graspable is True


def test_graspable_excludes_non_graspable():
    non_g = {o.name for o in REGISTRY if not o.graspable}
    grasp = {o.name for o in REGISTRY.graspable()}
    assert non_g.isdisjoint(grasp)


def test_graspable_at_least_half():
    assert len(REGISTRY.graspable()) >= 11


def test_by_mass_range():
    light = REGISTRY.by_mass_range(0.0, 0.2)
    assert all(o.mass_kg <= 0.2 for o in light)


def test_by_mass_range_empty():
    huge = REGISTRY.by_mass_range(10.0, 100.0)
    assert huge == []


def test_names_sorted():
    names = REGISTRY.names()
    assert names == sorted(names)


def test_labels_sorted():
    labels = REGISTRY.labels()
    assert labels == sorted(labels)


def test_categories_returns_list():
    cats = REGISTRY.categories()
    assert len(cats) > 0
    assert all(isinstance(c, YCBCategory) for c in cats)


# ── sampling ──────────────────────────────────────────────────────────────────

def test_sample_returns_correct_count():
    rng  = np.random.default_rng(0)
    objs = REGISTRY.sample(rng, n=3)
    assert len(objs) == 3


def test_sample_no_duplicates():
    rng  = np.random.default_rng(42)
    objs = REGISTRY.sample(rng, n=10)
    names = [o.name for o in objs]
    assert len(names) == len(set(names))


def test_sample_graspable_filter():
    rng  = np.random.default_rng(1)
    objs = REGISTRY.sample(rng, n=5, graspable=True)
    assert all(o.graspable for o in objs)


def test_sample_category_filter():
    rng  = np.random.default_rng(2)
    objs = REGISTRY.sample(rng, n=2, category="can")
    assert all(o.category == YCBCategory.CAN for o in objs)


def test_sample_too_many_raises():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="Requested"):
        REGISTRY.sample(rng, n=100)


def test_sample_reproducible():
    rng1 = np.random.default_rng(77)
    rng2 = np.random.default_rng(77)
    s1   = [o.name for o in REGISTRY.sample(rng1, n=5)]
    s2   = [o.name for o in REGISTRY.sample(rng2, n=5)]
    assert s1 == s2


def test_sample_different_seeds_differ():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(999)
    s1   = [o.name for o in REGISTRY.sample(rng1, n=5)]
    s2   = [o.name for o in REGISTRY.sample(rng2, n=5)]
    # With 5 draws from 21 objects, different seeds very likely give different results
    assert s1 != s2


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_contains_all_names():
    s = REGISTRY.summary()
    for obj in REGISTRY:
        assert obj.name in s


def test_summary_is_string():
    assert isinstance(REGISTRY.summary(), str)


# ── custom registry ───────────────────────────────────────────────────────────

def test_custom_registry_subset():
    objs = [_CATALOG[0], _CATALOG[1]]
    reg  = YCBRegistry(objects=objs)
    assert len(reg) == 2


def test_custom_registry_lookup():
    obj = _CATALOG[0]
    reg = YCBRegistry(objects=[obj])
    assert reg[obj.name] is obj
