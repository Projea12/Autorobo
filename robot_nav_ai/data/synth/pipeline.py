"""
data/synth/pipeline.py — Orchestrates synthetic YOLO dataset generation.

Output layout (YOLO-compatible)
────────────────────────────────
  <out_dir>/
    images/
      train/  000000.jpg  000001.jpg  …
      val/    …
    labels/
      train/  000000.txt  000001.txt  …
      val/    …
    dataset.yaml     ← YOLOv8-compatible dataset descriptor

Label file format (.txt)
────────────────────────
  One line per detected object:
    <class_id> <cx> <cy> <w> <h>
  All bbox values normalised to [0, 1].  Empty file = no objects visible.

Randomisation per image
───────────────────────
  • Camera:    random azimuth, elevation, distance (within CameraConfig ranges)
  • Objects:   random subset of YCB pool, random (x,y,yaw) on floor
  • Lighting:  random headlight ambient, directional light intensity + direction
  • Floor:     random grey shade

Usage
─────
    from data.synth import SynthPipeline, PipelineConfig

    pipe = SynthPipeline(PipelineConfig(n_images=5000, out_dir="data/synthetic"))
    stats = pipe.generate()
    print(stats)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from .annotator import Annotator, CLASS_NAMES
from .camera import CameraConfig, camera_pose_from_spherical
from .scene import SceneConfig, SynthScene
from data.dvc_utils import lineage_stamp


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Top-level parameters for the synthetic data generation run.

    n_images      : total images to generate (train + val)
    out_dir       : root output directory
    train_frac    : fraction going to train split (remainder → val)
    seed          : master RNG seed (None = random)
    scene_cfg     : SceneConfig (image resolution, object pool, etc.)
    cam_cfg       : CameraConfig (FOV, orbit ranges)
    jpeg_quality  : JPEG compression quality [0, 100]
    min_area_frac : minimum bbox area as fraction of image (Annotator filter)
    max_depth     : max object depth in metres (Annotator filter)
    report_every  : print progress every N images (0 = silent)
    """
    n_images:      int         = 1000
    out_dir:       str         = "data/synthetic"
    train_frac:    float       = 0.80
    seed:          Optional[int] = 42
    scene_cfg:     SceneConfig = field(default_factory=SceneConfig)
    cam_cfg:       CameraConfig = field(default_factory=CameraConfig)
    jpeg_quality:  int         = 92
    min_area_frac: float       = 0.002
    max_depth:     float       = 5.0
    report_every:  int         = 100


# ── statistics ────────────────────────────────────────────────────────────────

@dataclass
class GenerationStats:
    """Summary returned by SynthPipeline.generate()."""
    n_total:       int
    n_train:       int
    n_val:         int
    n_detections:  int    # total across all images
    n_empty:       int    # images with 0 detections
    elapsed_s:     float
    out_dir:       str

    @property
    def images_per_second(self) -> float:
        return self.n_total / max(self.elapsed_s, 1e-6)

    @property
    def mean_dets_per_image(self) -> float:
        return self.n_detections / max(self.n_total, 1)

    def __str__(self) -> str:
        return (
            f"Generated {self.n_total} images in {self.elapsed_s:.1f}s "
            f"({self.images_per_second:.1f} img/s)\n"
            f"  train={self.n_train}  val={self.n_val}\n"
            f"  detections={self.n_detections}  "
            f"mean_per_img={self.mean_dets_per_image:.2f}  "
            f"empty_imgs={self.n_empty}\n"
            f"  output → {self.out_dir}"
        )


# ── pipeline ──────────────────────────────────────────────────────────────────

class SynthPipeline:
    """
    Generates a YOLO-format synthetic image dataset from MuJoCo renders.

    Parameters
    ----------
    cfg : PipelineConfig
    """

    def __init__(self, cfg: PipelineConfig = PipelineConfig()) -> None:
        self.cfg = cfg

    def generate(self) -> GenerationStats:
        """
        Run the full generation loop.

        Returns GenerationStats with counts and timing.
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)

        out      = Path(cfg.out_dir)
        n_train  = int(cfg.n_images * cfg.train_frac)
        n_val    = cfg.n_images - n_train

        # Directory layout
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Build scene + renderer once
        scene    = SynthScene(cfg=cfg.scene_cfg)
        data     = scene.make_data()
        renderer = mujoco.Renderer(
            scene.model, cfg.scene_cfg.image_h, cfg.scene_cfg.image_w
        )
        annotator = Annotator(
            cam_cfg       = cfg.cam_cfg,
            min_area_frac = cfg.min_area_frac,
            max_depth     = cfg.max_depth,
        )

        n_dets   = 0
        n_empty  = 0
        t0       = time.perf_counter()
        img_idx  = 0

        for split, n_split in (("train", n_train), ("val", n_val)):
            for _ in range(n_split):
                # ── 1. randomise scene ────────────────────────────────────────
                active = scene.reset(data, rng)

                # ── 2. randomise lighting ─────────────────────────────────────
                self._randomise_lighting(scene.model, rng)

                # ── 3. camera pose ────────────────────────────────────────────
                pose     = cfg.cam_cfg.sample_pose(rng)
                cam_pos, R = camera_pose_from_spherical(**pose)

                mjvcam = mujoco.MjvCamera()
                mjvcam.type        = mujoco.mjtCamera.mjCAMERA_FREE
                mjvcam.lookat[:]   = pose["lookat"]
                mjvcam.distance    = pose["distance"]
                mjvcam.azimuth     = pose["azimuth"]
                mjvcam.elevation   = pose["elevation"]

                # ── 4. render ─────────────────────────────────────────────────
                renderer.update_scene(data, camera=mjvcam)
                rgb = renderer.render()   # (H, W, 3) uint8

                # ── 5. annotate ───────────────────────────────────────────────
                dets = annotator.annotate(active, data, cam_pos, R, scene)

                # ── 6. write image ────────────────────────────────────────────
                stem = f"{img_idx:06d}"
                img_path = out / "images" / split / f"{stem}.jpg"
                _write_jpeg(rgb, img_path, cfg.jpeg_quality)

                # ── 7. write label ────────────────────────────────────────────
                lbl_path = out / "labels" / split / f"{stem}.txt"
                lbl_path.write_text(
                    "\n".join(d.yolo_line() for d in dets)
                )

                n_dets  += len(dets)
                n_empty += (len(dets) == 0)

                if cfg.report_every and (img_idx + 1) % cfg.report_every == 0:
                    elapsed = time.perf_counter() - t0
                    rate    = (img_idx + 1) / elapsed
                    print(f"  [{img_idx+1:6d}/{cfg.n_images}]  "
                          f"{rate:.1f} img/s  "
                          f"split={split}  dets={len(dets)}")

                img_idx += 1

        renderer.close()

        # ── 8. write dataset.yaml ─────────────────────────────────────────────
        yaml_path = out / "dataset.yaml"
        _write_dataset_yaml(yaml_path, out, CLASS_NAMES)

        # ── 9. write generation manifest ─────────────────────────────────────
        elapsed = time.perf_counter() - t0
        stats   = GenerationStats(
            n_total      = cfg.n_images,
            n_train      = n_train,
            n_val        = n_val,
            n_detections = n_dets,
            n_empty      = n_empty,
            elapsed_s    = elapsed,
            out_dir      = str(out),
        )
        _write_manifest(out / "manifest.json", cfg, stats)

        return stats

    # ── per-image lighting randomisation ─────────────────────────────────────

    @staticmethod
    def _randomise_lighting(model: mujoco.MjModel, rng: np.random.Generator) -> None:
        """Randomise headlight ambient + directional light intensity."""
        ambient = float(rng.uniform(0.20, 0.55))
        model.vis.headlight.ambient[:] = ambient
        diffuse = float(rng.uniform(0.50, 1.00))
        model.vis.headlight.diffuse[:] = diffuse

        # Floor grey shade
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "synth_floor")
        if floor_id >= 0:
            g = float(rng.uniform(0.35, 0.80))
            model.geom_rgba[floor_id, :3] = g

        # Sun light intensity
        sun_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
        if sun_id >= 0:
            sd = float(rng.uniform(0.40, 0.90))
            model.light_diffuse[sun_id] = sd
            # Slight direction jitter
            base_dir = np.array([0.0, -0.3, -1.0])
            noise    = rng.normal(0, 0.15, size=3) * np.array([1, 1, 0])
            d        = base_dir + noise
            d       /= np.linalg.norm(d)
            model.light_dir[sun_id] = d


# ── file writers ──────────────────────────────────────────────────────────────

def _write_jpeg(rgb: np.ndarray, path: Path, quality: int) -> None:
    """Write (H,W,3) uint8 array as JPEG without requiring OpenCV."""
    try:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        return
    except ImportError:
        pass

    # Fallback: PIL / Pillow
    try:
        from PIL import Image
        Image.fromarray(rgb).save(str(path), "JPEG", quality=quality)
        return
    except ImportError:
        pass

    # Last resort: save as PNG (lossless, larger)
    import struct, zlib
    _write_png_fallback(rgb, path.with_suffix(".png"))


def _write_png_fallback(rgb: np.ndarray, path: Path) -> None:
    """Minimal PNG writer (no compression) — only used when PIL/cv2 absent."""
    H, W, _ = rgb.shape
    raw_rows = []
    for row in rgb:
        raw_rows.append(b"\x00" + row.tobytes())
    raw = b"".join(raw_rows)
    import zlib, struct
    compressed = zlib.compress(raw, 1)
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return c + struct.pack(">I", crc)
    sig   = b"\x89PNG\r\n\x1a\n"
    ihdr  = chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    idat  = chunk(b"IDAT", compressed)
    iend  = chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


def _write_dataset_yaml(
    path:        Path,
    out_dir:     Path,
    class_names: list[str],
) -> None:
    """Write a YOLOv8-compatible dataset.yaml."""
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    path.write_text("\n".join(lines) + "\n")


def _write_manifest(
    path:  Path,
    cfg:   PipelineConfig,
    stats: GenerationStats,
) -> None:
    manifest = {
        "n_images":           stats.n_total,
        "n_train":            stats.n_train,
        "n_val":              stats.n_val,
        "n_detections":       stats.n_detections,
        "n_empty":            stats.n_empty,
        "mean_dets_per_img":  stats.mean_dets_per_image,
        "elapsed_s":          round(stats.elapsed_s, 2),
        "images_per_second":  round(stats.images_per_second, 2),
        "seed":               cfg.seed,
        "n_classes":          len(CLASS_NAMES),
        "class_names":        CLASS_NAMES,
        "image_size":         [cfg.scene_cfg.image_w, cfg.scene_cfg.image_h],
        "jpeg_quality":       cfg.jpeg_quality,
        "lineage":            lineage_stamp(stage="generate_synth"),
    }
    path.write_text(json.dumps(manifest, indent=2))
