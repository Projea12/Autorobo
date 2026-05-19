"""
tests/test_ycb_preprocessor.py — Unit tests for data/ycb/preprocessor.py.

All tests use tmp_path; no real YCB data is needed.
Tests build synthetic STL/OBJ fixtures to exercise the full pipeline.
"""

from __future__ import annotations

import io
import json
import math
import struct
import tarfile
from pathlib import Path

import numpy as np
import pytest

import sys
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.ycb.preprocessor import YCBPreprocessor, ProcessedObject

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── STL fixture builders ──────────────────────────────────────────────────────

def _box_verts(lo=(-0.05, -0.04, 0.0), hi=(0.05, 0.04, 0.07)):
    x0,y0,z0 = lo; x1,y1,z1 = hi
    return np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=np.float64)


def _write_binary_stl(path: Path, verts: np.ndarray) -> None:
    """Write a minimal binary STL (no real triangles, just vertices repeated)."""
    n_tri = len(verts) // 3
    if n_tri == 0:
        n_tri = 1
        verts = np.vstack([verts[:3]] if len(verts) >= 3 else [verts[0]] * 3)
    buf = bytearray(80 + 4 + n_tri * 50)
    struct.pack_into("<I", buf, 80, n_tri)
    off = 84
    for i in range(n_tri):
        v0 = verts[i * 3 % len(verts)]
        v1 = verts[(i * 3 + 1) % len(verts)]
        v2 = verts[(i * 3 + 2) % len(verts)]
        struct.pack_into("<fff", buf, off, 0.0, 0.0, 1.0); off += 12
        struct.pack_into("<fff", buf, off, *v0.tolist()); off += 12
        struct.pack_into("<fff", buf, off, *v1.tolist()); off += 12
        struct.pack_into("<fff", buf, off, *v2.tolist()); off += 12
        struct.pack_into("<H",   buf, off, 0);             off += 2
    path.write_bytes(bytes(buf))


def _write_ascii_stl(path: Path, verts: np.ndarray) -> None:
    lines = ["solid test"]
    for i in range(0, len(verts) - 2, 3):
        v0,v1,v2 = verts[i], verts[i+1], verts[i+2]
        lines += [
            "  facet normal 0 0 1",
            "    outer loop",
            f"      vertex {v0[0]} {v0[1]} {v0[2]}",
            f"      vertex {v1[0]} {v1[1]} {v1[2]}",
            f"      vertex {v2[0]} {v2[1]} {v2[2]}",
            "    endloop",
            "  endfacet",
        ]
    lines.append("endsolid test")
    path.write_text("\n".join(lines))


def _make_raw_object(raw_dir: Path, name: str, ascii_stl: bool = False) -> Path:
    """Create a fake google_16k directory for `name`."""
    obj_dir = raw_dir / name / "google_16k"
    obj_dir.mkdir(parents=True, exist_ok=True)
    verts = _box_verts()
    if ascii_stl:
        _write_ascii_stl(obj_dir / "nontextured.stl", verts)
    else:
        _write_binary_stl(obj_dir / "nontextured.stl", verts)
    # minimal OBJ
    obj_path = obj_dir / "textured.obj"
    obj_path.write_text(
        "# minimal OBJ\nmtllib textured.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "f 1 2 3\n"
    )
    (obj_dir / "textured.mtl").write_text("# mtl\n")
    return obj_dir


# ── construction ──────────────────────────────────────────────────────────────

def test_preprocessor_creates_out_dir(tmp_path):
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    assert (tmp_path / "out").exists()


# ── missing raw files ─────────────────────────────────────────────────────────

def test_process_missing_stl_returns_failure(tmp_path):
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    assert not r.success
    assert "nontextured.stl not found" in r.error


def test_process_failure_str(tmp_path):
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    assert "fail" in str(r)


# ── binary STL pipeline ───────────────────────────────────────────────────────

def test_process_success_binary_stl(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    assert r.success, r.error


def test_process_creates_collision_stl(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    pre.process("002_master_chef_can")
    assert (tmp_path / "out" / "002_master_chef_can" / "collision.stl").exists()


def test_process_creates_visual_obj(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    pre.process("002_master_chef_can")
    assert (tmp_path / "out" / "002_master_chef_can" / "visual.obj").exists()


def test_process_creates_mjcf(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    pre.process("002_master_chef_can")
    assert (tmp_path / "out" / "002_master_chef_can" / "object.xml").exists()


def test_process_creates_meta_json(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    pre.process("002_master_chef_can")
    assert (tmp_path / "out" / "002_master_chef_can" / "meta.json").exists()


# ── ASCII STL ─────────────────────────────────────────────────────────────────

def test_process_ascii_stl(tmp_path):
    _make_raw_object(tmp_path / "raw", "005_tomato_soup_can", ascii_stl=True)
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("005_tomato_soup_can")
    assert r.success, r.error


# ── STL parsing ───────────────────────────────────────────────────────────────

def test_parse_binary_stl_returns_array(tmp_path):
    verts = _box_verts()
    p     = tmp_path / "test.stl"
    _write_binary_stl(p, verts)
    parsed = YCBPreprocessor._read_stl_vertices(p)
    assert isinstance(parsed, np.ndarray)
    assert parsed.shape[1] == 3


def test_parse_ascii_stl_returns_array(tmp_path):
    verts = _box_verts()
    p     = tmp_path / "test.stl"
    _write_ascii_stl(p, verts)
    parsed = YCBPreprocessor._read_stl_vertices(p)
    assert isinstance(parsed, np.ndarray)
    assert parsed.shape[1] == 3


def test_parse_binary_stl_values_finite(tmp_path):
    verts = _box_verts()
    p     = tmp_path / "test.stl"
    _write_binary_stl(p, verts)
    parsed = YCBPreprocessor._read_stl_vertices(p)
    assert np.isfinite(parsed).all()


def test_parse_too_short_stl_raises(tmp_path):
    p = tmp_path / "bad.stl"
    p.write_bytes(b"\x00" * 10)
    with pytest.raises((ValueError, Exception)):
        YCBPreprocessor._read_stl_vertices(p)


# ── convex hull ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_convex_hull_returns_verts_and_faces():
    verts = _box_verts()
    hv, hf, n = YCBPreprocessor._convex_hull(verts)
    assert hv is not None
    assert hf is not None
    assert n == len(hv)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_convex_hull_verts_subset_of_original():
    verts = _box_verts()
    hv, _, _ = YCBPreprocessor._convex_hull(verts)
    for v in hv:
        dists = np.linalg.norm(verts - v, axis=1)
        assert dists.min() < 1e-9


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_convex_hull_faces_valid_indices():
    verts = _box_verts()
    hv, hf, _ = YCBPreprocessor._convex_hull(verts)
    assert hf.min() >= 0
    assert hf.max() < len(hv)


def test_convex_hull_too_few_verts():
    verts = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
    hv, hf, n = YCBPreprocessor._convex_hull(verts)
    assert hv is None


# ── STL writing ───────────────────────────────────────────────────────────────

def test_write_stl_valid_binary_format(tmp_path):
    verts = _box_verts()
    faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
    path  = tmp_path / "out.stl"
    YCBPreprocessor._write_stl(path, verts, faces)
    raw = path.read_bytes()
    assert len(raw) >= 84
    n_tri = struct.unpack_from("<I", raw, 80)[0]
    assert n_tri == 2


def test_write_box_stl_creates_file(tmp_path):
    path = tmp_path / "box.stl"
    YCBPreprocessor._write_box_stl(
        path,
        centre = np.array([0.0, 0.0, 0.035]),
        half   = np.array([0.05, 0.04, 0.035]),
    )
    assert path.exists()
    raw = path.read_bytes()
    n_tri = struct.unpack_from("<I", raw, 80)[0]
    assert n_tri == 12   # a box has 12 triangles


def test_write_box_stl_file_size(tmp_path):
    path = tmp_path / "box.stl"
    YCBPreprocessor._write_box_stl(
        path, np.zeros(3), np.array([0.05, 0.05, 0.05])
    )
    expected = 80 + 4 + 12 * 50
    assert path.stat().st_size == expected


# ── MJCF generation ───────────────────────────────────────────────────────────

def test_mjcf_is_valid_xml(tmp_path):
    import xml.etree.ElementTree as ET
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    ET.parse(r.mjcf_path)   # raises if invalid XML


def test_mjcf_contains_object_name(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    xml = r.mjcf_path.read_text()
    assert "002_master_chef_can" in xml


def test_mjcf_has_freejoint(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    assert "<freejoint" in r.mjcf_path.read_text()


def test_mjcf_has_inertial(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    assert "<inertial" in r.mjcf_path.read_text()


def test_mjcf_mass_from_registry(tmp_path):
    """MJCF mass should match registry value for known objects."""
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    from data.ycb.registry import REGISTRY
    expected_mass = REGISTRY["002_master_chef_can"].mass_kg
    xml = r.mjcf_path.read_text()
    assert f'mass="{expected_mass:.6f}"' in xml


def test_mjcf_has_collision_and_visual_geoms(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    xml = r.mjcf_path.read_text()
    assert 'contype="1"' in xml    # collision geom
    assert 'contype="0"' in xml    # visual geom


# ── meta.json ─────────────────────────────────────────────────────────────────

def test_meta_has_required_keys(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    required = {"name", "mass_kg", "friction", "half_extents",
                "size_cm", "hull_vertices", "collision_stl", "visual_obj", "mjcf"}
    assert required.issubset(r.meta.keys())


def test_meta_name_matches(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    assert r.meta["name"] == "002_master_chef_can"


def test_meta_half_extents_positive(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    assert all(v > 0 for v in r.meta["half_extents"])


def test_meta_mass_matches_registry(tmp_path):
    from data.ycb.registry import REGISTRY
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    assert abs(r.meta["mass_kg"] - REGISTRY["002_master_chef_can"].mass_kg) < 1e-6


# ── is_processed / load_meta ──────────────────────────────────────────────────

def test_is_processed_false_before_process(tmp_path):
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    assert not pre.is_processed("002_master_chef_can")


def test_is_processed_true_after_process(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    pre.process("002_master_chef_can")
    assert pre.is_processed("002_master_chef_can")


def test_load_meta_roundtrip(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre  = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r    = pre.process("002_master_chef_can")
    meta = pre.load_meta("002_master_chef_can")
    assert meta["name"] == "002_master_chef_can"
    assert meta["mass_kg"] == r.meta["mass_kg"]


def test_load_meta_unprocessed_raises(tmp_path):
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        pre.load_meta("002_master_chef_can")


# ── process_all ───────────────────────────────────────────────────────────────

def test_process_all_returns_one_per_object(tmp_path):
    for name in ["002_master_chef_can", "005_tomato_soup_can"]:
        _make_raw_object(tmp_path / "raw", name)
    pre     = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    results = pre.process_all()
    assert len(results) == 2


def test_process_all_explicit_names(tmp_path):
    for name in ["002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can"]:
        _make_raw_object(tmp_path / "raw", name)
    pre     = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    results = pre.process_all(names=["002_master_chef_can", "007_tuna_fish_can"])
    assert len(results) == 2


def test_process_all_all_succeed(tmp_path):
    for name in ["002_master_chef_can", "005_tomato_soup_can"]:
        _make_raw_object(tmp_path / "raw", name)
    pre     = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    results = pre.process_all()
    assert all(r.success for r in results), [r.error for r in results if not r.success]


# ── elapsed time ─────────────────────────────────────────────────────────────

def test_elapsed_nonnegative(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    assert r.elapsed_s >= 0.0


# ── ProcessedObject str ───────────────────────────────────────────────────────

def test_processed_object_str_success(tmp_path):
    _make_raw_object(tmp_path / "raw", "002_master_chef_can")
    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process("002_master_chef_can")
    s   = str(r)
    assert "ok" in s and "002_master_chef_can" in s


def test_processed_object_str_failure():
    r = ProcessedObject(name="bad_obj", success=False, error="missing")
    assert "fail" in str(r)


# ── no visual OBJ fallback ────────────────────────────────────────────────────

def test_box_obj_fallback_when_no_textured_obj(tmp_path):
    """If textured.obj is missing, a box OBJ should be generated."""
    name    = "002_master_chef_can"
    obj_dir = tmp_path / "raw" / name / "google_16k"
    obj_dir.mkdir(parents=True, exist_ok=True)
    _write_binary_stl(obj_dir / "nontextured.stl", _box_verts())
    # deliberately omit textured.obj

    pre = YCBPreprocessor(tmp_path / "raw", tmp_path / "out")
    r   = pre.process(name)
    assert r.success
    vis = tmp_path / "out" / name / "visual.obj"
    assert vis.exists()
    content = vis.read_text()
    assert "v " in content   # has vertices
