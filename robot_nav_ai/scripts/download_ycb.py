"""
scripts/download_ycb.py — CLI for downloading and preprocessing YCB objects.

Quick start
───────────
    # List all 21 standard objects
    python scripts/download_ycb.py --list

    # Download everything (parallel, 4 workers)
    python scripts/download_ycb.py --all

    # Download and preprocess a specific subset
    python scripts/download_ycb.py 002_master_chef_can 005_tomato_soup_can --preprocess

    # Check what's already on disk
    python scripts/download_ycb.py --status

    # Download graspable objects only, then preprocess
    python scripts/download_ycb.py --graspable --preprocess --workers 8

    # Force re-download even if already present
    python scripts/download_ycb.py --all --force

Directories
───────────
    data/ycb/raw/         ← raw extracted archives from S3
    data/ycb/processed/   ← collision STL + visual OBJ + MJCF per object
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.ycb import REGISTRY, YCBDownloader, YCBPreprocessor
from data.ycb.registry import YCBCategory

_RAW_DIR       = _ROOT / "data" / "ycb" / "raw"
_PROCESSED_DIR = _ROOT / "data" / "ycb" / "processed"


# ── progress callback ──────────────────────────────────────────────────────────

def _make_progress(verbose: bool):
    if not verbose:
        return None
    last: dict[str, float] = {}
    def cb(name: str, done: int, total: int) -> None:
        pct = (done / total * 100) if total else 0
        if pct - last.get(name, -5) >= 5:
            last[name] = pct
            bar_len = 20
            filled  = int(bar_len * done / total) if total else 0
            bar     = "█" * filled + "░" * (bar_len - filled)
            mb_done = done / 1e6
            mb_tot  = total / 1e6 if total else 0
            print(f"  {name:<30} [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_tot:.1f} MB",
                  flush=True)
    return cb


# ── subcommands ────────────────────────────────────────────────────────────────

def cmd_list(_args) -> int:
    print(REGISTRY.summary())
    return 0


def cmd_status(_args) -> int:
    dl = YCBDownloader(_RAW_DIR)
    pre_dir = _PROCESSED_DIR
    print(f"\n{'Object':<32} {'Downloaded':>12} {'Processed':>10}")
    print("─" * 58)
    n_dl = n_pre = 0
    for name in REGISTRY.names():
        ok_dl  = dl.status().get(name, False)
        ok_pre = (pre_dir / name / "meta.json").exists()
        n_dl  += ok_dl
        n_pre += ok_pre
        mark = lambda b: "✓" if b else "✗"
        print(f"  {name:<30}  {mark(ok_dl):>10}   {mark(ok_pre):>9}")
    print(f"\n{n_dl}/{len(REGISTRY)} downloaded   {n_pre}/{len(REGISTRY)} preprocessed")
    return 0


def cmd_download(args) -> int:
    dl = YCBDownloader(
        dest_dir    = _RAW_DIR,
        force       = args.force,
        progress_cb = _make_progress(args.verbose),
    )

    # Resolve target names
    if args.all:
        names = REGISTRY.names()
    elif args.graspable:
        names = [o.name for o in REGISTRY.graspable()]
    elif args.category:
        try:
            cat   = YCBCategory(args.category)
            names = [o.name for o in REGISTRY.by_category(cat)]
        except ValueError:
            valid = [c.value for c in YCBCategory]
            print(f"Unknown category {args.category!r}. Valid: {valid}")
            return 1
    else:
        names = args.objects

    if not names:
        print("No objects specified. Use --all, --graspable, --category, or list object names.")
        return 1

    print(f"\nDownloading {len(names)} YCB object(s) → {_RAW_DIR}")
    print("─" * 60)
    t0      = time.perf_counter()
    results = dl.download_all(names=names, n_workers=args.workers)
    elapsed = time.perf_counter() - t0

    n_ok   = sum(r.success for r in results)
    n_skip = sum(r.skipped for r in results)
    total_mb = sum(r.bytes_dl for r in results) / 1e6
    print(f"\n{'─'*60}")
    print(f"{n_ok}/{len(results)} succeeded  ({n_skip} already present)  "
          f"{total_mb:.1f} MB  {elapsed:.1f}s")

    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n{len(failed)} failed:")
        for r in failed:
            print(f"  {r.name}: {r.error}")

    # Optional preprocessing step
    if args.preprocess:
        ok_names = [r.name for r in results if r.success]
        return _run_preprocess(ok_names)

    return 0 if not failed else 1


def cmd_preprocess(args) -> int:
    pre = YCBPreprocessor(raw_dir=_RAW_DIR, out_dir=_PROCESSED_DIR)

    if args.all:
        names = None   # discover from raw_dir
    elif args.objects:
        names = args.objects
    else:
        # preprocess everything that's downloaded but not yet processed
        dl    = YCBDownloader(_RAW_DIR)
        names = [n for n, ok in dl.status().items()
                 if ok and not pre.is_processed(n)]

    return _run_preprocess(names, pre)


def _run_preprocess(names: list[str] | None, pre: YCBPreprocessor | None = None) -> int:
    if pre is None:
        pre = YCBPreprocessor(raw_dir=_RAW_DIR, out_dir=_PROCESSED_DIR)

    if names is not None and not names:
        print("Nothing to preprocess.")
        return 0

    label = "all discovered" if names is None else str(len(names))
    print(f"\nPreprocessing {label} object(s) → {_PROCESSED_DIR}")
    print("─" * 60)
    t0      = time.perf_counter()
    results = pre.process_all(names=names)
    elapsed = time.perf_counter() - t0

    n_ok = sum(r.success for r in results)
    print(f"\n{n_ok}/{len(results)} preprocessed  {elapsed:.1f}s")

    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n{len(failed)} failed:")
        for r in failed:
            print(f"  {r.name}: {r.error}")
    return 0 if not failed else 1


# ── argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "download_ycb",
        description = "Download and preprocess YCB objects for AutoRobo v1.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = __doc__,
    )
    sub = p.add_subparsers(dest="command")

    # ── list ──────────────────────────────────────────────────────────────────
    sub.add_parser("list", help="Print all 21 objects in the registry")

    # ── status ────────────────────────────────────────────────────────────────
    sub.add_parser("status", help="Show download and preprocessing status for every object")

    # ── download ──────────────────────────────────────────────────────────────
    dl = sub.add_parser("download", help="Download raw YCB mesh archives from S3")
    grp = dl.add_mutually_exclusive_group()
    grp.add_argument("--all",       action="store_true", help="Download all 21 objects")
    grp.add_argument("--graspable", action="store_true", help="Download only graspable objects")
    grp.add_argument("--category",  metavar="CAT",
                     help="Download objects in one category (can, box, bottle, ...)")
    dl.add_argument("objects", nargs="*", metavar="NAME",
                    help="Canonical YCB name(s) to download")
    dl.add_argument("--workers",    type=int, default=4, metavar="N",
                    help="Parallel download threads (default: 4)")
    dl.add_argument("--force",      action="store_true",
                    help="Re-download even if already present")
    dl.add_argument("--preprocess", action="store_true",
                    help="Run preprocessing immediately after download")
    dl.add_argument("--verbose", "-v", action="store_true")

    # ── preprocess ────────────────────────────────────────────────────────────
    pp = sub.add_parser("preprocess",
                        help="Convert raw archives to MuJoCo assets (STL + OBJ + MJCF)")
    pp.add_argument("objects", nargs="*", metavar="NAME",
                    help="Names to process; omit → process everything downloaded but not yet done")
    pp.add_argument("--all", action="store_true",
                    help="Process all discovered raw objects")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.command == "list":
        return cmd_list(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "download":
        return cmd_download(args)
    if args.command == "preprocess":
        return cmd_preprocess(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
