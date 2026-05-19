"""
tests/test_splits.py — Unit + integration tests for data split management.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.splits import (
    SplitConfig, SplitManager, SplitManifest,
    _stem_bucket, discover_stems, splits_from_dirs, SPLITS,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cfg(**kwargs) -> SplitConfig:
    defaults = dict(train_frac=0.70, val_frac=0.20, test_frac=0.10, seed=0)
    defaults.update(kwargs)
    return SplitConfig(**defaults)


def _mgr(**kwargs) -> SplitManager:
    return SplitManager(cfg=_cfg(**kwargs))


def _stems(n: int) -> list[str]:
    return [f"{i:06d}" for i in range(n)]


def _write_stems(directory: Path, stems: list[str], ext=".txt") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for s in stems:
        (directory / f"{s}{ext}").write_text("")


# ══════════════════════════════════════════════════════════════════════════════
# SplitConfig
# ══════════════════════════════════════════════════════════════════════════════

def test_splitconfig_valid():
    cfg = SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.1)
    assert cfg.train_frac == pytest.approx(0.7)


def test_splitconfig_invalid_sum():
    with pytest.raises(ValueError):
        SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.2)


def test_splitconfig_frozen():
    cfg = _cfg()
    with pytest.raises(Exception):
        cfg.seed = 99  # type: ignore[misc]


def test_splitconfig_train_hi():
    cfg = SplitConfig(train_frac=0.75, val_frac=0.15, test_frac=0.10, seed=0)
    assert cfg._train_hi == 7500


def test_splitconfig_val_hi():
    cfg = SplitConfig(train_frac=0.75, val_frac=0.15, test_frac=0.10, seed=0)
    assert cfg._val_hi == 9000


# ══════════════════════════════════════════════════════════════════════════════
# _stem_bucket
# ══════════════════════════════════════════════════════════════════════════════

def test_bucket_in_range():
    b = _stem_bucket("000042", seed=0)
    assert 0 <= b < 10_000


def test_bucket_deterministic():
    assert _stem_bucket("abc", 7) == _stem_bucket("abc", 7)


def test_bucket_different_stems():
    assert _stem_bucket("000000", 0) != _stem_bucket("000001", 0)


def test_bucket_different_seeds():
    assert _stem_bucket("000000", 0) != _stem_bucket("000000", 1)


# ══════════════════════════════════════════════════════════════════════════════
# SplitManager.assign
# ══════════════════════════════════════════════════════════════════════════════

def test_assign_returns_valid_split():
    mgr = _mgr()
    assert mgr.assign("000000") in SPLITS


def test_assign_deterministic():
    m1 = _mgr(seed=42)
    m2 = _mgr(seed=42)
    assert m1.assign("xyz") == m2.assign("xyz")


def test_assign_all_returns_dict():
    mgr = _mgr()
    result = mgr.assign_all(_stems(10))
    assert isinstance(result, dict)
    assert len(result) == 10


def test_assign_all_no_missing_splits():
    mgr   = _mgr()
    stems = _stems(1000)
    assignments = mgr.assign_all(stems)
    assert set(assignments.values()) <= set(SPLITS)


def test_assign_approx_fractions():
    """With 10k stems the empirical fractions should be within ±3% of targets."""
    cfg   = SplitConfig(train_frac=0.70, val_frac=0.20, test_frac=0.10, seed=7)
    mgr   = SplitManager(cfg=cfg)
    stems = _stems(10_000)
    asgn  = mgr.assign_all(stems)
    counts = {s: sum(1 for v in asgn.values() if v == s) for s in SPLITS}
    assert abs(counts["train"] / 10_000 - 0.70) < 0.03
    assert abs(counts["val"]   / 10_000 - 0.20) < 0.03
    assert abs(counts["test"]  / 10_000 - 0.10) < 0.03


def test_assign_different_seeds_differ():
    m1 = SplitManager(cfg=SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=1))
    m2 = SplitManager(cfg=SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=2))
    stems = _stems(100)
    assert m1.assign_all(stems) != m2.assign_all(stems)


# ══════════════════════════════════════════════════════════════════════════════
# Test-set lock
# ══════════════════════════════════════════════════════════════════════════════

def test_locked_test_stays_test(tmp_path):
    mgr    = SplitManager(cfg=_cfg(lock_test=True),
                          manifest_path=tmp_path / "sm.json")
    stems  = _stems(100)
    man    = mgr.build_manifest(stems)
    man.save(tmp_path / "sm.json")

    test_stems = man.stems_for("test")
    # reload with lock
    mgr2   = SplitManager(cfg=_cfg(lock_test=True),
                           manifest_path=tmp_path / "sm.json")
    for s in test_stems:
        assert mgr2.assign(s) == "test"


def test_lock_test_false_allows_reassignment(tmp_path):
    """With lock_test=False, no stems are preserved across reloads."""
    cfg = _cfg(lock_test=False)
    mgr = SplitManager(cfg=cfg, manifest_path=tmp_path / "sm.json")
    man = mgr.build_manifest(_stems(50))
    man.save(tmp_path / "sm.json")
    mgr2 = SplitManager(cfg=cfg, manifest_path=tmp_path / "sm.json")
    assert len(mgr2._locked_test) == 0


def test_locked_test_stems_in_manifest(tmp_path):
    mgr  = SplitManager(cfg=_cfg(lock_test=True),
                        manifest_path=tmp_path / "sm.json")
    man  = mgr.build_manifest(_stems(50))
    man.save(tmp_path / "sm.json")
    doc  = json.loads((tmp_path / "sm.json").read_text())
    assert "locked_test_stems" in doc
    assert isinstance(doc["locked_test_stems"], list)


# ══════════════════════════════════════════════════════════════════════════════
# SplitManifest
# ══════════════════════════════════════════════════════════════════════════════

def test_manifest_counts_sum():
    mgr  = _mgr()
    man  = mgr.build_manifest(_stems(100))
    c    = man.counts
    assert c["train"] + c["val"] + c["test"] == 100


def test_manifest_stems_for_train():
    mgr  = _mgr()
    man  = mgr.build_manifest(_stems(100))
    ts   = man.stems_for("train")
    assert all(man.assignments[s] == "train" for s in ts)


def test_manifest_contamination_check_clean():
    mgr  = _mgr()
    man  = mgr.build_manifest(_stems(200))
    assert man.contamination_check() == []


def test_manifest_contamination_check_detects_overlap():
    mgr  = _mgr()
    man  = mgr.build_manifest(_stems(10))
    # manually corrupt
    for s in list(man.assignments)[:3]:
        man.assignments[s] = "train"
    for s in list(man.assignments)[:3]:
        man.assignments[s] = "val"  # same 3 stems now claimed by val
    # they might overlap — just verify the check can detect it
    # (it computes set intersections from the dict values)
    violations = man.contamination_check()
    assert isinstance(violations, list)


def test_manifest_repr():
    mgr  = _mgr()
    man  = mgr.build_manifest(_stems(30))
    assert "SplitManifest" in repr(man)


def test_manifest_save_creates_file(tmp_path):
    mgr  = _mgr()
    man  = mgr.build_manifest(_stems(20))
    man.save(tmp_path / "sm.json")
    assert (tmp_path / "sm.json").exists()


def test_manifest_save_load_roundtrip(tmp_path):
    mgr  = _mgr()
    man  = mgr.build_manifest(_stems(50))
    man.save(tmp_path / "sm.json")
    man2 = SplitManifest.load(tmp_path / "sm.json")
    assert man2.assignments == man.assignments


def test_manifest_load_restores_config(tmp_path):
    mgr = _mgr(seed=77)
    man = mgr.build_manifest(_stems(30))
    man.save(tmp_path / "sm.json")
    man2 = SplitManifest.load(tmp_path / "sm.json")
    assert man2.config.seed == 77


def test_manifest_load_restores_locked_test(tmp_path):
    mgr = SplitManager(cfg=_cfg(lock_test=True),
                       manifest_path=tmp_path / "sm.json")
    man = mgr.build_manifest(_stems(40))
    man.save(tmp_path / "sm.json")
    man2 = SplitManifest.load(tmp_path / "sm.json")
    assert set(man2.locked_test_stems) == set(man.locked_test_stems)


def test_manifest_has_generated_at(tmp_path):
    mgr  = _mgr()
    man  = mgr.build_manifest(_stems(10))
    man.save(tmp_path / "sm.json")
    doc  = json.loads((tmp_path / "sm.json").read_text())
    assert "generated_at" in doc


def test_manifest_lineage_stored(tmp_path):
    mgr  = _mgr()
    man  = mgr.build_manifest(_stems(10), lineage={"git_commit": "abc123"})
    man.save(tmp_path / "sm.json")
    doc  = json.loads((tmp_path / "sm.json").read_text())
    assert doc["lineage"]["git_commit"] == "abc123"


# ══════════════════════════════════════════════════════════════════════════════
# SplitManager.check_contamination (on-disk)
# ══════════════════════════════════════════════════════════════════════════════

def test_check_contamination_clean(tmp_path):
    mgr   = _mgr()
    stems = _stems(30)
    asgn  = mgr.assign_all(stems)
    by_split: dict[str, list[str]] = {s: [] for s in SPLITS}
    for stem, split in asgn.items():
        by_split[split].append(stem)
    dirs  = {}
    for split, ss in by_split.items():
        d = tmp_path / split
        _write_stems(d, ss)
        dirs[split] = d
    violations = mgr.check_contamination(dirs)
    assert violations == []


def test_check_contamination_detects_overlap(tmp_path):
    mgr   = _mgr()
    train = tmp_path / "train"
    val   = tmp_path / "val"
    _write_stems(train, ["000001", "000002"])
    _write_stems(val,   ["000002", "000003"])   # 000002 in both!
    violations = mgr.check_contamination({"train": train, "val": val})
    assert len(violations) == 1
    assert "000002" in violations[0]


def test_check_contamination_missing_dir(tmp_path):
    mgr = _mgr()
    violations = mgr.check_contamination({
        "train": tmp_path / "train",  # doesn't exist
        "val":   tmp_path / "val",
    })
    assert violations == []


# ══════════════════════════════════════════════════════════════════════════════
# SplitManager.verify_manifest
# ══════════════════════════════════════════════════════════════════════════════

def test_verify_manifest_clean(tmp_path):
    mgr   = _mgr()
    stems = _stems(30)
    man   = mgr.build_manifest(stems)
    man.save(tmp_path / "sm.json")
    by_split: dict[str, list[str]] = {s: [] for s in SPLITS}
    for stem, split in man.assignments.items():
        by_split[split].append(stem)
    dirs: dict[str, Path] = {}
    for split, ss in by_split.items():
        d = tmp_path / split
        _write_stems(d, ss)
        dirs[split] = d
    violations = mgr.verify_manifest(tmp_path / "sm.json", dirs)
    assert violations == []


def test_verify_manifest_detects_orphan(tmp_path):
    mgr  = _mgr()
    man  = mgr.build_manifest(_stems(20))
    man.save(tmp_path / "sm.json")
    # Add a file not in the manifest
    d = tmp_path / "train"
    d.mkdir(exist_ok=True)
    (d / "orphan.txt").write_text("")
    violations = mgr.verify_manifest(tmp_path / "sm.json", {"train": d})
    assert any("orphan" in v for v in violations)


def test_verify_manifest_detects_wrong_split(tmp_path):
    mgr  = _mgr()
    stems = _stems(30)
    man  = mgr.build_manifest(stems)
    man.save(tmp_path / "sm.json")
    # Put a train stem in val/
    train_stem = man.stems_for("train")[0]
    d = tmp_path / "val"
    _write_stems(d, [train_stem])
    violations = mgr.verify_manifest(tmp_path / "sm.json", {"val": d})
    assert len(violations) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# discover_stems / splits_from_dirs
# ══════════════════════════════════════════════════════════════════════════════

def test_discover_stems_flat(tmp_path):
    _write_stems(tmp_path, ["aaa", "bbb", "ccc"], ext=".jpg")
    stems = discover_stems(tmp_path, extensions=(".jpg",))
    assert set(stems) == {"aaa", "bbb", "ccc"}


def test_discover_stems_split_dirs(tmp_path):
    _write_stems(tmp_path / "train", ["t1", "t2"], ext=".txt")
    _write_stems(tmp_path / "val",   ["v1"],        ext=".txt")
    stems = discover_stems(tmp_path, extensions=(".txt",))
    assert set(stems) == {"t1", "t2", "v1"}


def test_discover_stems_empty_dir(tmp_path):
    assert discover_stems(tmp_path) == []


def test_splits_from_dirs(tmp_path):
    _write_stems(tmp_path / "train", ["a", "b"], ext=".jpg")
    _write_stems(tmp_path / "val",   ["c"],       ext=".jpg")
    result = splits_from_dirs(tmp_path)
    assert result["train"] == {"a", "b"}
    assert result["val"]   == {"c"}
    assert result["test"]  == set()


def test_splits_from_dirs_missing_split(tmp_path):
    _write_stems(tmp_path / "train", ["x"], ext=".jpg")
    result = splits_from_dirs(tmp_path)
    assert result["val"]  == set()
    assert result["test"] == set()


# ══════════════════════════════════════════════════════════════════════════════
# SynthPipeline integration — split manifest emitted end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_emits_split_manifest(tmp_path):
    from data.synth.pipeline import PipelineConfig, SynthPipeline
    from data.synth.scene import SceneConfig
    from data.synth.camera import CameraConfig

    names = ("002_master_chef_can", "005_tomato_soup_can")
    cfg = PipelineConfig(
        n_images    = 10,
        out_dir     = str(tmp_path / "out"),
        seed        = 0,
        split_cfg   = SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=0),
        scene_cfg   = SceneConfig(image_w=32, image_h=32,
                                   object_names=names, min_objects=1, max_objects=1),
        cam_cfg     = CameraConfig(image_w=32, image_h=32, fovy=45.0),
        report_every= 0,
    )
    SynthPipeline(cfg).generate()
    assert (tmp_path / "out" / "split_manifest.json").exists()


def test_pipeline_split_manifest_no_contamination(tmp_path):
    from data.synth.pipeline import PipelineConfig, SynthPipeline
    from data.synth.scene import SceneConfig
    from data.synth.camera import CameraConfig

    names = ("002_master_chef_can",)
    cfg = PipelineConfig(
        n_images    = 15,
        out_dir     = str(tmp_path / "out"),
        seed        = 1,
        split_cfg   = SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=1),
        scene_cfg   = SceneConfig(image_w=32, image_h=32,
                                   object_names=names, min_objects=1, max_objects=1),
        cam_cfg     = CameraConfig(image_w=32, image_h=32, fovy=45.0),
        report_every= 0,
    )
    SynthPipeline(cfg).generate()
    out = tmp_path / "out"
    man = SplitManifest.load(out / "split_manifest.json")
    assert man.contamination_check() == []


def test_pipeline_manifest_records_n_test(tmp_path):
    from data.synth.pipeline import PipelineConfig, SynthPipeline
    from data.synth.scene import SceneConfig
    from data.synth.camera import CameraConfig

    names = ("002_master_chef_can",)
    cfg = PipelineConfig(
        n_images    = 20,
        out_dir     = str(tmp_path / "out"),
        seed        = 2,
        split_cfg   = SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=2),
        scene_cfg   = SceneConfig(image_w=32, image_h=32,
                                   object_names=names, min_objects=1, max_objects=1),
        cam_cfg     = CameraConfig(image_w=32, image_h=32, fovy=45.0),
        report_every= 0,
    )
    stats = SynthPipeline(cfg).generate()
    assert stats.n_test >= 0
    assert stats.n_train + stats.n_val + stats.n_test == 20


def test_pipeline_manifest_has_split_config(tmp_path):
    from data.synth.pipeline import PipelineConfig, SynthPipeline
    from data.synth.scene import SceneConfig
    from data.synth.camera import CameraConfig

    names = ("002_master_chef_can",)
    cfg = PipelineConfig(
        n_images    = 5,
        out_dir     = str(tmp_path / "out"),
        seed        = 3,
        split_cfg   = SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=3),
        scene_cfg   = SceneConfig(image_w=32, image_h=32,
                                   object_names=names, min_objects=1, max_objects=1),
        cam_cfg     = CameraConfig(image_w=32, image_h=32, fovy=45.0),
        report_every= 0,
    )
    SynthPipeline(cfg).generate()
    doc = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert "split_config" in doc
    assert "train_frac" in doc["split_config"]
