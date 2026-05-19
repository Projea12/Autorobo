"""
tests/test_dvc_integration.py — DVC utilities and pipeline config tests.

These tests do NOT run `dvc repro` (that would download YCB and render 10k
images).  They verify:
  1. params.yaml is valid YAML and contains required keys
  2. dvc.yaml is valid YAML and each stage references params that exist
  3. dvc_utils lineage_stamp produces correct structure
  4. lineage is embedded in manifest.json produced by SynthPipeline
  5. datasets_match logic works correctly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.dvc_utils import (
    lineage_stamp, read_lineage, datasets_match,
    load_params, load_dvc_lock, stage_output_hashes,
    git_commit, git_dirty,
)

PARAMS_PATH   = _ROOT / "params.yaml"
DVC_YAML_PATH = _ROOT / "dvc.yaml"


# ── params.yaml ───────────────────────────────────────────────────────────────

def test_params_yaml_exists():
    assert PARAMS_PATH.exists()


def test_params_yaml_parseable():
    params = load_params(PARAMS_PATH)
    assert isinstance(params, dict)


def test_params_has_ycb_download():
    params = load_params(PARAMS_PATH)
    assert "ycb_download" in params


def test_params_has_ycb_preprocess():
    params = load_params(PARAMS_PATH)
    assert "ycb_preprocess" in params


def test_params_has_synth_generate():
    params = load_params(PARAMS_PATH)
    assert "synth_generate" in params


def test_params_has_validate_labels():
    params = load_params(PARAMS_PATH)
    assert "validate_labels" in params


def test_params_synth_seed_is_int():
    params = load_params(PARAMS_PATH)
    assert isinstance(params["synth_generate"]["seed"], int)


def test_params_synth_n_images_positive():
    params = load_params(PARAMS_PATH)
    assert params["synth_generate"]["n_images"] > 0


def test_params_synth_train_frac_in_unit():
    params = load_params(PARAMS_PATH)
    f = params["synth_generate"]["train_frac"]
    assert 0.0 < f < 1.0


def test_params_validate_n_classes_21():
    params = load_params(PARAMS_PATH)
    assert params["validate_labels"]["n_classes"] == 21


def test_params_validate_min_area_positive():
    params = load_params(PARAMS_PATH)
    assert params["validate_labels"]["min_area"] > 0


def test_params_ycb_n_workers_positive():
    params = load_params(PARAMS_PATH)
    assert params["ycb_download"]["n_workers"] >= 1


def test_params_missing_file_returns_empty():
    assert load_params("/nonexistent/path/params.yaml") == {}


# ── dvc.yaml ─────────────────────────────────────────────────────────────────

def test_dvc_yaml_exists():
    assert DVC_YAML_PATH.exists()


def test_dvc_yaml_parseable():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    d = yaml.safe_load(DVC_YAML_PATH.read_text())
    assert isinstance(d, dict)


def test_dvc_yaml_has_stages():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    d = yaml.safe_load(DVC_YAML_PATH.read_text())
    assert "stages" in d


def test_dvc_yaml_stage_download_ycb():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    stages = yaml.safe_load(DVC_YAML_PATH.read_text())["stages"]
    assert "download_ycb" in stages


def test_dvc_yaml_stage_preprocess_ycb():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    stages = yaml.safe_load(DVC_YAML_PATH.read_text())["stages"]
    assert "preprocess_ycb" in stages


def test_dvc_yaml_stage_generate_synth():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    stages = yaml.safe_load(DVC_YAML_PATH.read_text())["stages"]
    assert "generate_synth" in stages


def test_dvc_yaml_stage_validate_labels():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    stages = yaml.safe_load(DVC_YAML_PATH.read_text())["stages"]
    assert "validate_labels" in stages


def test_dvc_yaml_each_stage_has_cmd():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    stages = yaml.safe_load(DVC_YAML_PATH.read_text())["stages"]
    for name, stage in stages.items():
        assert "cmd" in stage, f"Stage '{name}' missing 'cmd'"


def test_dvc_yaml_each_stage_has_deps():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    stages = yaml.safe_load(DVC_YAML_PATH.read_text())["stages"]
    for name, stage in stages.items():
        assert "deps" in stage, f"Stage '{name}' missing 'deps'"


def test_dvc_yaml_each_stage_has_outs():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    stages = yaml.safe_load(DVC_YAML_PATH.read_text())["stages"]
    for name, stage in stages.items():
        assert "outs" in stage or "metrics" in stage, \
            f"Stage '{name}' has no 'outs' or 'metrics'"


def test_dvc_yaml_generate_synth_refs_seed_param():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    stages = yaml.safe_load(DVC_YAML_PATH.read_text())["stages"]
    cmd = stages["generate_synth"]["cmd"]
    assert "seed" in cmd


def test_dvc_yaml_validate_labels_refs_n_classes():
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    stages = yaml.safe_load(DVC_YAML_PATH.read_text())["stages"]
    cmd = stages["validate_labels"]["cmd"]
    assert "n-classes" in cmd or "n_classes" in cmd


# ── lineage_stamp ─────────────────────────────────────────────────────────────

def test_lineage_stamp_returns_dict():
    stamp = lineage_stamp(params_path=PARAMS_PATH, repo_root=_ROOT)
    assert isinstance(stamp, dict)


def test_lineage_stamp_has_git_commit():
    stamp = lineage_stamp(params_path=PARAMS_PATH, repo_root=_ROOT)
    assert "git_commit" in stamp


def test_lineage_stamp_has_git_dirty():
    stamp = lineage_stamp(params_path=PARAMS_PATH, repo_root=_ROOT)
    assert "git_dirty" in stamp


def test_lineage_stamp_has_dvc_status():
    stamp = lineage_stamp(params_path=PARAMS_PATH, repo_root=_ROOT)
    assert "dvc_status" in stamp


def test_lineage_stamp_has_params():
    stamp = lineage_stamp(params_path=PARAMS_PATH, repo_root=_ROOT)
    assert "params" in stamp
    assert isinstance(stamp["params"], dict)


def test_lineage_stamp_has_generated_at():
    stamp = lineage_stamp(params_path=PARAMS_PATH, repo_root=_ROOT)
    assert "generated_at" in stamp


def test_lineage_stamp_params_not_empty():
    stamp = lineage_stamp(params_path=PARAMS_PATH, repo_root=_ROOT)
    assert len(stamp["params"]) > 0


def test_lineage_stamp_git_commit_is_string():
    stamp = lineage_stamp(params_path=PARAMS_PATH, repo_root=_ROOT)
    assert isinstance(stamp["git_commit"], str)


def test_lineage_stamp_git_dirty_is_bool():
    stamp = lineage_stamp(params_path=PARAMS_PATH, repo_root=_ROOT)
    assert isinstance(stamp["git_dirty"], bool)


def test_lineage_stamp_missing_params():
    stamp = lineage_stamp(params_path="/nonexistent.yaml", repo_root=_ROOT)
    assert stamp["params"] == {}


def test_lineage_stamp_stage_hashes_empty_without_lock():
    stamp = lineage_stamp(
        params_path=PARAMS_PATH,
        dvc_lock_path="/nonexistent.lock",
        repo_root=_ROOT,
        stage="generate_synth",
    )
    assert stamp["stage_hashes"] == {}


# ── dvc.lock helpers ──────────────────────────────────────────────────────────

def test_load_dvc_lock_missing_returns_empty():
    assert load_dvc_lock("/nonexistent/dvc.lock") == {}


def test_stage_output_hashes_missing_lock():
    assert stage_output_hashes("generate_synth", "/nonexistent.lock") == {}


def test_stage_output_hashes_stage_not_in_lock(tmp_path):
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    lock = tmp_path / "dvc.lock"
    lock.write_text(yaml.dump({"schema": "2.0", "stages": {}}))
    assert stage_output_hashes("generate_synth", lock) == {}


def test_stage_output_hashes_found(tmp_path):
    try:
        import yaml
    except ImportError:
        pytest.skip("pyyaml not installed")
    lock_content = {
        "schema": "2.0",
        "stages": {
            "generate_synth": {
                "outs": [{"data/synthetic": {"md5": "abc123def456"}}]
            }
        }
    }
    lock = tmp_path / "dvc.lock"
    lock.write_text(yaml.dump(lock_content))
    hashes = stage_output_hashes("generate_synth", lock)
    assert "data/synthetic" in hashes
    assert hashes["data/synthetic"] == "abc123def456"


# ── read_lineage / datasets_match ─────────────────────────────────────────────

def test_read_lineage_missing_file():
    assert read_lineage("/nonexistent/manifest.json") == {}


def test_read_lineage_no_lineage_key(tmp_path):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"n_images": 100}))
    assert read_lineage(m) == {}


def test_read_lineage_returns_block(tmp_path):
    stamp = {"git_commit": "abc1234", "params": {"seed": 42}}
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"n_images": 100, "lineage": stamp}))
    lineage = read_lineage(m)
    assert lineage["git_commit"] == "abc1234"


def test_datasets_match_identical(tmp_path):
    stamp = {"git_commit": "abc1234", "git_dirty": False,
             "params": {"seed": 42}, "generated_at": "T1"}
    for name in ("m1.json", "m2.json"):
        (tmp_path / name).write_text(json.dumps({"lineage": stamp}))
    assert datasets_match(tmp_path / "m1.json", tmp_path / "m2.json")


def test_datasets_match_different_commit(tmp_path):
    s1 = {"git_commit": "aaa", "params": {"seed": 42}}
    s2 = {"git_commit": "bbb", "params": {"seed": 42}}
    (tmp_path / "m1.json").write_text(json.dumps({"lineage": s1}))
    (tmp_path / "m2.json").write_text(json.dumps({"lineage": s2}))
    assert not datasets_match(tmp_path / "m1.json", tmp_path / "m2.json")


def test_datasets_match_different_params(tmp_path):
    s1 = {"git_commit": "aaa", "params": {"seed": 42}}
    s2 = {"git_commit": "aaa", "params": {"seed": 99}}
    (tmp_path / "m1.json").write_text(json.dumps({"lineage": s1}))
    (tmp_path / "m2.json").write_text(json.dumps({"lineage": s2}))
    assert not datasets_match(tmp_path / "m1.json", tmp_path / "m2.json")


def test_datasets_match_missing_file(tmp_path):
    stamp = {"git_commit": "abc", "params": {}}
    (tmp_path / "m1.json").write_text(json.dumps({"lineage": stamp}))
    assert not datasets_match(tmp_path / "m1.json", tmp_path / "missing.json")


# ── SynthPipeline embeds lineage in manifest ──────────────────────────────────

def test_synth_pipeline_manifest_has_lineage(tmp_path):
    """End-to-end: run a tiny pipeline and check manifest.json contains lineage."""
    from data.synth.pipeline import PipelineConfig, SynthPipeline
    from data.synth.scene import SceneConfig
    from data.synth.camera import CameraConfig

    names = ("002_master_chef_can", "005_tomato_soup_can")
    cfg = PipelineConfig(
        n_images   = 3,
        out_dir    = str(tmp_path / "out"),
        seed       = 0,
        train_frac = 1.0,
        scene_cfg  = SceneConfig(image_w=32, image_h=32,
                                  object_names=names,
                                  min_objects=1, max_objects=1),
        cam_cfg    = CameraConfig(image_w=32, image_h=32, fovy=45.0),
        report_every = 0,
    )
    SynthPipeline(cfg).generate()
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert "lineage" in manifest


def test_synth_pipeline_manifest_lineage_has_params(tmp_path):
    from data.synth.pipeline import PipelineConfig, SynthPipeline
    from data.synth.scene import SceneConfig
    from data.synth.camera import CameraConfig

    names = ("002_master_chef_can",)
    cfg = PipelineConfig(
        n_images   = 2,
        out_dir    = str(tmp_path / "out"),
        seed       = 1,
        train_frac = 1.0,
        scene_cfg  = SceneConfig(image_w=32, image_h=32,
                                  object_names=names,
                                  min_objects=1, max_objects=1),
        cam_cfg    = CameraConfig(image_w=32, image_h=32, fovy=45.0),
        report_every = 0,
    )
    SynthPipeline(cfg).generate()
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert "params" in manifest["lineage"]


def test_synth_pipeline_manifest_lineage_has_git_commit(tmp_path):
    from data.synth.pipeline import PipelineConfig, SynthPipeline
    from data.synth.scene import SceneConfig
    from data.synth.camera import CameraConfig

    names = ("002_master_chef_can",)
    cfg = PipelineConfig(
        n_images   = 2,
        out_dir    = str(tmp_path / "out"),
        seed       = 2,
        train_frac = 1.0,
        scene_cfg  = SceneConfig(image_w=32, image_h=32,
                                  object_names=names,
                                  min_objects=1, max_objects=1),
        cam_cfg    = CameraConfig(image_w=32, image_h=32, fovy=45.0),
        report_every = 0,
    )
    SynthPipeline(cfg).generate()
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert "git_commit" in manifest["lineage"]
