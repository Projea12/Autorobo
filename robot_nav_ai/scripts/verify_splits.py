"""
scripts/verify_splits.py — Verify dataset splits are contamination-free.

Checks
──────
  1. No image stem appears in more than one of train / val / test
  2. Every on-disk file has a matching entry in split_manifest.json
  3. Every manifest entry matches its on-disk location
  4. Test set stems match the locked_test_stems in the manifest

Exit codes
──────────
  0  — all checks passed (clean)
  1  — one or more violations found

Examples
────────
    # Check a freshly generated synthetic dataset
    python scripts/verify_splits.py --dataset data/synthetic

    # Check labels directory only (no images)
    python scripts/verify_splits.py --dataset data/synthetic --labels-only

    # Print the split manifest summary
    python scripts/verify_splits.py --dataset data/synthetic --summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.splits import SplitManifest, SplitManager, splits_from_dirs, SPLITS


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="verify_splits",
        description="Verify dataset splits are contamination-free.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, metavar="DIR",
                   help="Root dataset directory (must contain split_manifest.json)")
    p.add_argument("--labels-only", action="store_true",
                   help="Only check labels/ directory (skip images/)")
    p.add_argument("--summary", action="store_true",
                   help="Print split manifest summary and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args   = _parse_args(argv)
    root   = Path(args.dataset)
    mpath  = root / "split_manifest.json"

    if not root.exists():
        print(f"ERROR: dataset directory not found: {root}")
        return 1

    # ── summary mode ──────────────────────────────────────────────────────────
    if args.summary:
        if not mpath.exists():
            print(f"ERROR: split_manifest.json not found in {root}")
            return 1
        m = SplitManifest.load(mpath)
        c = m.counts
        print(f"\nSplit manifest: {mpath}")
        print(f"  train : {c['train']:6d} images")
        print(f"  val   : {c['val']:6d} images")
        print(f"  test  : {c['test']:6d} images")
        print(f"  total : {sum(c.values()):6d} images")
        print(f"  locked test stems: {len(m.locked_test_stems)}")
        cfg = m.config
        print(f"\nSplit config:")
        print(f"  train_frac={cfg.train_frac}  val_frac={cfg.val_frac}  "
              f"test_frac={cfg.test_frac}  seed={cfg.seed}  "
              f"lock_test={cfg.lock_test}")
        if m.lineage:
            print(f"\nLineage:")
            print(f"  git_commit : {m.lineage.get('git_commit', 'n/a')}")
            print(f"  git_dirty  : {m.lineage.get('git_dirty', 'n/a')}")
            print(f"  generated  : {m.lineage.get('generated_at', 'n/a')}")
        return 0

    # ── contamination checks ──────────────────────────────────────────────────
    violations: list[str] = []
    print(f"\nVerifying splits in: {root}")

    # 1. in-memory manifest check
    if mpath.exists():
        m = SplitManifest.load(mpath)
        mem_violations = m.contamination_check()
        if mem_violations:
            violations.extend(
                f"[MANIFEST] {v}" for v in mem_violations
            )
        else:
            print("  ✓ Manifest: no overlapping assignments")

        # 2. locked test integrity
        locked = set(m.locked_test_stems)
        actual_test = set(m.stems_for("test"))
        missing_from_test = locked - actual_test
        if missing_from_test:
            violations.append(
                f"[LOCK] {len(missing_from_test)} locked test stems are not "
                f"assigned to test in manifest (e.g. "
                f"{sorted(missing_from_test)[:3]})"
            )
        else:
            print("  ✓ Lock: all locked test stems correctly assigned to test")
    else:
        print(f"  ! split_manifest.json not found — skipping manifest checks")

    # 3. on-disk contamination
    for subdir in (["labels"] if args.labels_only else ["images", "labels"]):
        d = root / subdir
        if not d.exists():
            continue
        split_dirs = {split: d / split for split in SPLITS}
        mgr = SplitManager()
        disk_violations = mgr.check_contamination(split_dirs)
        if disk_violations:
            violations.extend(f"[DISK/{subdir}] {v}" for v in disk_violations)
        else:
            present = [s for s in SPLITS if (d / s).exists()]
            print(f"  ✓ {subdir}/: no overlap across {present}")

    # 4. manifest vs disk consistency
    if mpath.exists():
        for subdir in (["labels"] if args.labels_only else ["images", "labels"]):
            d = root / subdir
            if not d.exists():
                continue
            on_disk = splits_from_dirs(d)
            for split, stems in on_disk.items():
                for stem in stems:
                    expected = m.assignments.get(stem)
                    if expected is None:
                        violations.append(
                            f"[ORPHAN] '{stem}' in {subdir}/{split}/ "
                            f"has no manifest entry"
                        )
                    elif expected != split:
                        violations.append(
                            f"[MISMATCH] '{stem}' in {subdir}/{split}/ "
                            f"but manifest says '{expected}'"
                        )
        if not violations:
            print("  ✓ Disk vs manifest: all files match their manifest assignments")

    # ── result ────────────────────────────────────────────────────────────────
    if violations:
        print(f"\nFAIL — {len(violations)} violation(s) found:")
        for v in violations:
            print(f"  ✗ {v}")
        return 1

    print(f"\nPASS — dataset splits are clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
