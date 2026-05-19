"""
scripts/generate_synth_data.py — CLI for synthetic YOLO dataset generation.

Examples
────────
    # Prototype: 50 tiny images to verify the pipeline
    python scripts/generate_synth_data.py --n 50 --size 320 240 --out data/synthetic_test

    # Production: 10 000 full-res images, 8 workers (sequential per-image, but
    # multiple calls can be run in parallel across GPUs / nodes)
    python scripts/generate_synth_data.py --n 10000 --out data/synthetic

    # Use only graspable objects and cap objects per image at 3
    python scripts/generate_synth_data.py --n 200 --graspable --max-obj 3

    # Specific YCB subset
    python scripts/generate_synth_data.py --n 500 \\
        --objects 002_master_chef_can 005_tomato_soup_can 006_mustard_bottle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.synth.pipeline import PipelineConfig, SynthPipeline
from data.synth.scene import SceneConfig
from data.synth.camera import CameraConfig
from data.ycb.registry import REGISTRY


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="generate_synth_data",
        description="Generate synthetic YOLO training images from MuJoCo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── volume ────────────────────────────────────────────────────────────────
    p.add_argument("--n",      type=int, default=1000, metavar="N",
                   help="Total images to generate (train + val, default: 1000)")
    p.add_argument("--out",    default="data/synthetic", metavar="DIR",
                   help="Output directory (default: data/synthetic)")
    p.add_argument("--seed",   type=int, default=42,
                   help="RNG seed for reproducibility (default: 42)")
    p.add_argument("--train-frac", type=float, default=0.80,
                   help="Fraction of images for train split (default: 0.80)")

    # ── resolution / quality ──────────────────────────────────────────────────
    p.add_argument("--size",   type=int, nargs=2, default=[640, 480],
                   metavar=("W", "H"),
                   help="Image size width height in pixels (default: 640 480)")
    p.add_argument("--fovy",   type=float, default=45.0,
                   help="Camera vertical FOV in degrees (default: 45)")
    p.add_argument("--jpeg-quality", type=int, default=92,
                   help="JPEG compression quality 0-100 (default: 92)")

    # ── object selection ──────────────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--all",      action="store_true",
                     help="Use all 21 YCB objects (default)")
    grp.add_argument("--graspable", action="store_true",
                     help="Use only graspable objects")
    grp.add_argument("--objects",  nargs="+", metavar="NAME",
                     help="Explicit list of canonical YCB names")

    p.add_argument("--min-obj", type=int, default=2,
                   help="Min objects per image (default: 2)")
    p.add_argument("--max-obj", type=int, default=5,
                   help="Max objects per image (default: 5)")

    # ── camera ────────────────────────────────────────────────────────────────
    p.add_argument("--dist",   type=float, nargs=2, default=[0.9, 2.2],
                   metavar=("MIN", "MAX"),
                   help="Camera distance range in metres (default: 0.9 2.2)")
    p.add_argument("--elev",   type=float, nargs=2, default=[-55.0, -15.0],
                   metavar=("MIN", "MAX"),
                   help="Camera elevation range in degrees (default: -55 -15)")

    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--report", type=int, default=100,
                   help="Print progress every N images (0=silent, default: 100)")
    p.add_argument("--list",   action="store_true",
                   help="List all available YCB objects and exit")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.list:
        print(REGISTRY.summary())
        return 0

    # ── resolve object names ──────────────────────────────────────────────────
    if args.graspable:
        names = tuple(o.name for o in REGISTRY.graspable())
    elif args.objects:
        invalid = [n for n in args.objects if n not in REGISTRY]
        if invalid:
            print(f"Unknown YCB objects: {invalid}")
            print("Run with --list to see valid names.")
            return 1
        names = tuple(args.objects)
    else:
        names = None   # all 21

    W, H = args.size

    scene_cfg = SceneConfig(
        image_w      = W,
        image_h      = H,
        fovy         = args.fovy,
        object_names = names,
        min_objects  = args.min_obj,
        max_objects  = args.max_obj,
    )
    cam_cfg = CameraConfig(
        image_w        = W,
        image_h        = H,
        fovy           = args.fovy,
        distance_range = tuple(args.dist),
        elevation_range= tuple(args.elev),
    )
    pipe_cfg = PipelineConfig(
        n_images     = args.n,
        out_dir      = args.out,
        train_frac   = args.train_frac,
        seed         = args.seed,
        scene_cfg    = scene_cfg,
        cam_cfg      = cam_cfg,
        jpeg_quality = args.jpeg_quality,
        report_every = args.report,
    )

    n_obj_str = f"{len(names)} objects" if names else "all 21 objects"
    print(f"\nSynthetic data generation")
    print(f"  images:     {args.n}  (train={int(args.n*args.train_frac)} val={args.n - int(args.n*args.train_frac)})")
    print(f"  resolution: {W}×{H}  fovy={args.fovy}°")
    print(f"  objects:    {n_obj_str}  ({args.min_obj}–{args.max_obj} per image)")
    print(f"  output:     {args.out}")
    print(f"  seed:       {args.seed}")
    print()

    pipe  = SynthPipeline(pipe_cfg)
    stats = pipe.generate()
    print(f"\n{stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
