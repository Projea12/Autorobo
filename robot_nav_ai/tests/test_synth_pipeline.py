"""
tests/test_synth_pipeline.py — Integration tests for the synthetic pipeline.

All tests use small image counts (5-20) and tiny resolution (64×64) to stay
fast.  The pipeline is exercised end-to-end: scene → render → annotate → write.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.synth.annotator import Annotator, Detection, CLASS_NAMES, _NAME_TO_CLASS
from data.synth.camera import CameraConfig, camera_pose_from_spherical
from data.synth.pipeline import PipelineConfig, SynthPipeline, GenerationStats
from data.synth.scene import SceneConfig, SynthScene


# ── small pipeline config factory ─────────────────────────────────────────────

def _small_cfg(tmp_path, n=10, seed=42) -> PipelineConfig:
    W, H  = 64, 64
    names = ("002_master_chef_can", "005_tomato_soup_can", "003_cracker_box")
    return PipelineConfig(
        n_images     = n,
        out_dir      = str(tmp_path / "out"),
        train_frac   = 0.80,
        seed         = seed,
        scene_cfg    = SceneConfig(image_w=W, image_h=H,
                                   object_names=names,
                                   min_objects=1, max_objects=2),
        cam_cfg      = CameraConfig(image_w=W, image_h=H, fovy=45.0),
        jpeg_quality = 85,
        report_every = 0,
    )


# ── Annotator unit tests ──────────────────────────────────────────────────────

def test_annotator_class_ids_unique():
    ids = list(_NAME_TO_CLASS.values())
    assert len(ids) == len(set(ids))


def test_annotator_class_names_count():
    assert len(CLASS_NAMES) == 21


def test_annotator_class_id_lookup():
    assert Annotator.class_id("002_master_chef_can") == _NAME_TO_CLASS["002_master_chef_can"]


def test_annotator_class_name_roundtrip():
    cid  = Annotator.class_id("005_tomato_soup_can")
    name = Annotator.class_name(cid)
    assert name == "005_tomato_soup_can"


def test_annotator_n_classes():
    assert Annotator.n_classes() == 21


def test_annotator_produces_detections():
    names = ("002_master_chef_can", "005_tomato_soup_can")
    cfg   = SceneConfig(object_names=names, min_objects=2, max_objects=2,
                        image_w=64, image_h=64)
    scene = SynthScene(cfg=cfg)
    data  = scene.make_data()
    rng   = np.random.default_rng(0)
    active = scene.reset(data, rng)

    cam_cfg = CameraConfig(image_w=64, image_h=64, fovy=45.0)
    pose    = {"lookat": np.zeros(3), "distance": 1.5,
                "azimuth": 0.0, "elevation": -45.0}
    pos, R  = camera_pose_from_spherical(**pose)

    ann  = Annotator(cam_cfg=cam_cfg, min_area_frac=0.0)
    dets = ann.annotate(active, data, pos, R, scene)
    # At least some objects should be visible
    assert isinstance(dets, list)


def test_detection_yolo_line_format():
    det = Detection(
        name      = "002_master_chef_can",
        class_id  = 0,
        bbox_xyxy = np.array([10., 20., 50., 80.]),
        bbox_yolo = np.array([0.3, 0.5, 0.2, 0.3]),
        depth_m   = 1.2,
    )
    line = det.yolo_line()
    parts = line.split()
    assert len(parts) == 5
    assert parts[0] == "0"
    assert all("." in p for p in parts[1:])


def test_detection_repr():
    det = Detection(
        name      = "002_master_chef_can",
        class_id  = 0,
        bbox_xyxy = np.array([0., 0., 10., 10.]),
        bbox_yolo = np.array([0.5, 0.5, 0.1, 0.1]),
        depth_m   = 1.5,
    )
    assert "002_master_chef_can" in repr(det)


def test_annotator_filters_tiny_objects():
    """Objects with very small projected area should be filtered."""
    names  = ("040_large_marker",)   # very thin object; may project tiny
    cfg    = SceneConfig(object_names=names, min_objects=1, max_objects=1,
                         image_w=64, image_h=64)
    scene  = SynthScene(cfg=cfg)
    data   = scene.make_data()
    rng    = np.random.default_rng(0)
    active = scene.reset(data, rng)

    cam_cfg = CameraConfig(image_w=64, image_h=64)
    pos, R  = camera_pose_from_spherical(np.zeros(3), 5.0, 0.0, -80.0)
    ann     = Annotator(cam_cfg=cam_cfg, min_area_frac=0.9)  # very strict
    dets    = ann.annotate(active, data, pos, R, scene)
    # With 90% area requirement, nothing should survive
    assert dets == []


def test_annotator_bbox_yolo_in_unit_interval():
    names  = ("002_master_chef_can",)
    cfg    = SceneConfig(object_names=names, min_objects=1, max_objects=1,
                         image_w=64, image_h=64)
    scene  = SynthScene(cfg=cfg)
    data   = scene.make_data()
    active = scene.reset(data, np.random.default_rng(0))

    cam_cfg = CameraConfig(image_w=64, image_h=64, fovy=45.0)
    pos, R  = camera_pose_from_spherical(np.zeros(3), 1.2, 0.0, -60.0)
    ann     = Annotator(cam_cfg=cam_cfg, min_area_frac=0.0)
    dets    = ann.annotate(active, data, pos, R, scene)
    for d in dets:
        assert np.all(d.bbox_yolo >= 0.0)
        assert np.all(d.bbox_yolo <= 1.0)


# ── SynthPipeline end-to-end ──────────────────────────────────────────────────

def test_pipeline_runs(tmp_path):
    cfg   = _small_cfg(tmp_path, n=5)
    pipe  = SynthPipeline(cfg)
    stats = pipe.generate()
    assert stats.n_total == 5


def test_pipeline_creates_train_images(tmp_path):
    cfg   = _small_cfg(tmp_path, n=10)
    stats = SynthPipeline(cfg).generate()
    out   = Path(cfg.out_dir)
    images = list((out / "images" / "train").iterdir())
    assert len(images) == stats.n_train


def test_pipeline_creates_val_images(tmp_path):
    cfg   = _small_cfg(tmp_path, n=10)
    stats = SynthPipeline(cfg).generate()
    out   = Path(cfg.out_dir)
    images = list((out / "images" / "val").iterdir())
    assert len(images) == stats.n_val


def test_pipeline_creates_label_files(tmp_path):
    cfg  = _small_cfg(tmp_path, n=6)
    SynthPipeline(cfg).generate()
    out    = Path(cfg.out_dir)
    n_img  = len(list((out / "images" / "train").iterdir()))
    n_lbl  = len(list((out / "labels" / "train").iterdir()))
    assert n_img == n_lbl


def test_pipeline_label_and_image_stems_match(tmp_path):
    cfg  = _small_cfg(tmp_path, n=6)
    SynthPipeline(cfg).generate()
    out   = Path(cfg.out_dir)
    for split in ("train", "val"):
        img_stems = {p.stem for p in (out / "images" / split).iterdir()}
        lbl_stems = {p.stem for p in (out / "labels" / split).iterdir()}
        assert img_stems == lbl_stems


def test_pipeline_creates_dataset_yaml(tmp_path):
    cfg  = _small_cfg(tmp_path, n=5)
    SynthPipeline(cfg).generate()
    yaml = Path(cfg.out_dir) / "dataset.yaml"
    assert yaml.exists()


def test_pipeline_dataset_yaml_has_nc(tmp_path):
    cfg  = _small_cfg(tmp_path, n=5)
    SynthPipeline(cfg).generate()
    content = (Path(cfg.out_dir) / "dataset.yaml").read_text()
    assert "nc: 21" in content


def test_pipeline_dataset_yaml_has_class_names(tmp_path):
    cfg  = _small_cfg(tmp_path, n=5)
    SynthPipeline(cfg).generate()
    content = (Path(cfg.out_dir) / "dataset.yaml").read_text()
    assert "002_master_chef_can" in content


def test_pipeline_creates_manifest_json(tmp_path):
    cfg  = _small_cfg(tmp_path, n=5)
    SynthPipeline(cfg).generate()
    manifest = Path(cfg.out_dir) / "manifest.json"
    assert manifest.exists()


def test_pipeline_manifest_fields(tmp_path):
    cfg      = _small_cfg(tmp_path, n=5)
    SynthPipeline(cfg).generate()
    data     = json.loads((Path(cfg.out_dir) / "manifest.json").read_text())
    required = {"n_images", "n_train", "n_val", "seed", "class_names", "image_size"}
    assert required <= data.keys()


def test_pipeline_manifest_n_images(tmp_path):
    cfg  = _small_cfg(tmp_path, n=7)
    SynthPipeline(cfg).generate()
    data = json.loads((Path(cfg.out_dir) / "manifest.json").read_text())
    assert data["n_images"] == 7


def test_pipeline_stats_counts(tmp_path):
    cfg   = _small_cfg(tmp_path, n=8)
    stats = SynthPipeline(cfg).generate()
    assert stats.n_train + stats.n_val + stats.n_test == 8


def test_pipeline_stats_elapsed_positive(tmp_path):
    cfg   = _small_cfg(tmp_path, n=3)
    stats = SynthPipeline(cfg).generate()
    assert stats.elapsed_s > 0


def test_pipeline_stats_str(tmp_path):
    cfg   = _small_cfg(tmp_path, n=3)
    stats = SynthPipeline(cfg).generate()
    s     = str(stats)
    assert "Generated 3 images" in s


def test_pipeline_label_format(tmp_path):
    """Every label line must be '<int> <float> <float> <float> <float>'."""
    cfg  = _small_cfg(tmp_path, n=5)
    SynthPipeline(cfg).generate()
    out  = Path(cfg.out_dir)
    for split in ("train", "val"):
        for p in (out / "labels" / split).iterdir():
            for line in p.read_text().strip().splitlines():
                if not line:
                    continue
                parts = line.split()
                assert len(parts) == 5, f"Bad label line in {p}: {line!r}"
                assert parts[0].isdigit()
                assert all("." in x or x.isdigit() for x in parts[1:])


def test_pipeline_yolo_bbox_in_unit_interval(tmp_path):
    """All normalised bbox values must be in [0, 1]."""
    cfg  = _small_cfg(tmp_path, n=5)
    SynthPipeline(cfg).generate()
    out  = Path(cfg.out_dir)
    for split in ("train", "val"):
        for p in (out / "labels" / split).iterdir():
            for line in p.read_text().strip().splitlines():
                if not line:
                    continue
                _, cx, cy, w, h = map(float, line.split())
                for v in (cx, cy, w, h):
                    assert 0.0 <= v <= 1.0, f"{v} out of range in {p}"


def test_pipeline_reproducible(tmp_path):
    """Same seed → identical label files for the first image."""
    cfg1 = _small_cfg(tmp_path / "r1", n=5, seed=0)
    cfg2 = _small_cfg(tmp_path / "r2", n=5, seed=0)
    SynthPipeline(cfg1).generate()
    SynthPipeline(cfg2).generate()

    lbl1 = sorted((Path(cfg1.out_dir) / "labels" / "train").iterdir())
    lbl2 = sorted((Path(cfg2.out_dir) / "labels" / "train").iterdir())
    assert len(lbl1) == len(lbl2)
    for p1, p2 in zip(lbl1, lbl2):
        assert p1.read_text() == p2.read_text()


def test_pipeline_different_seeds_differ(tmp_path):
    cfg1 = _small_cfg(tmp_path / "d1", n=10, seed=1)
    cfg2 = _small_cfg(tmp_path / "d2", n=10, seed=9999)
    SynthPipeline(cfg1).generate()
    SynthPipeline(cfg2).generate()
    lbls1 = sorted((Path(cfg1.out_dir) / "labels" / "train").iterdir())
    lbls2 = sorted((Path(cfg2.out_dir) / "labels" / "train").iterdir())
    texts1 = [p.read_text() for p in lbls1]
    texts2 = [p.read_text() for p in lbls2]
    assert texts1 != texts2


# ── GenerationStats ───────────────────────────────────────────────────────────

def test_generation_stats_images_per_second():
    s = GenerationStats(100, 80, 20, 300, 5, 10.0, "/tmp")
    assert s.images_per_second == pytest.approx(10.0)


def test_generation_stats_mean_dets():
    s = GenerationStats(10, 8, 2, 50, 0, 1.0, "/tmp")
    assert s.mean_dets_per_image == pytest.approx(5.0)
