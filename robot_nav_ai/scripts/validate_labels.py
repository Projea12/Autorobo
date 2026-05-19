"""
scripts/validate_labels.py — Validate and repair YOLO label files.

Examples
────────
    # Validate synthetic dataset, print report
    python scripts/validate_labels.py --labels data/synthetic/labels

    # Validate + write fixed labels and JSON report
    python scripts/validate_labels.py \\
        --labels data/synthetic/labels \\
        --images data/synthetic/images \\
        --output data/synthetic_fixed \\
        --n-classes 21

    # Strict: reject border-overflow boxes instead of clamping
    python scripts/validate_labels.py --labels data/synthetic/labels --no-clamp

    # Deduplicate overlapping boxes (IoU > 0.7)
    python scripts/validate_labels.py --labels data/synthetic/labels --max-iou 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.annotation import AnnotationPipeline, PipelineConfig, ValidatorConfig
from data.synth.annotator import CLASS_NAMES


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="validate_labels",
        description="Validate and repair YOLO .txt label files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--labels",    required=True, metavar="DIR",
                   help="Root directory of YOLO .txt label files")
    p.add_argument("--images",    default=None, metavar="DIR",
                   help="Root directory of images (optional, for pairing check)")
    p.add_argument("--output",    default=None, metavar="DIR",
                   help="Output directory for fixed labels + report (omit to validate only)")
    p.add_argument("--n-classes", type=int, default=21,
                   help="Number of valid class IDs [0, n_classes) (default: 21)")
    p.add_argument("--min-area",  type=float, default=1e-4,
                   help="Minimum normalised bbox area (default: 1e-4)")
    p.add_argument("--max-area",  type=float, default=0.99,
                   help="Maximum normalised bbox area (default: 0.99)")
    p.add_argument("--min-ar",    type=float, default=0.02,
                   help="Minimum aspect ratio w/h (default: 0.02)")
    p.add_argument("--max-ar",    type=float, default=50.0,
                   help="Maximum aspect ratio w/h (default: 50.0)")
    p.add_argument("--max-iou",   type=float, default=1.0,
                   help="IoU threshold for duplicate suppression (1.0=off, default: 1.0)")
    p.add_argument("--conf",      type=float, default=0.0,
                   help="Minimum confidence for pseudo-labels (default: 0.0)")
    p.add_argument("--no-clamp",  action="store_true",
                   help="Treat border-overflow as error instead of clamping")
    p.add_argument("--no-fix",    action="store_true",
                   help="Do not write fixed label files (report only)")
    p.add_argument("--no-report", action="store_true",
                   help="Do not write report files")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    vcfg = ValidatorConfig(
        n_classes            = args.n_classes,
        min_area             = args.min_area,
        max_area             = args.max_area,
        min_aspect_ratio     = args.min_ar,
        max_aspect_ratio     = args.max_ar,
        confidence_threshold = args.conf,
        clamp_border         = not args.no_clamp,
        max_overlap_iou      = args.max_iou,
    )
    cfg = PipelineConfig(
        label_dir    = args.labels,
        image_dir    = args.images,
        output_dir   = args.output,
        validator_cfg= vcfg,
        class_names  = CLASS_NAMES,
        write_fixed  = (args.output is not None) and not args.no_fix,
        write_report = (args.output is not None) and not args.no_report,
    )

    print(f"\nValidating labels in: {args.labels}")
    if args.output:
        print(f"Output directory    : {args.output}")
    print()

    pipe   = AnnotationPipeline(cfg)
    report = pipe.run()
    print(report)

    if args.output:
        print(f"\nFixed labels written to: {args.output}/labels/")
        print(f"Report written to      : {args.output}/report.json")

    return 0 if report.n_files_with_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
