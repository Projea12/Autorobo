"""
data/annotation/pipeline.py — End-to-end annotation validation and repair.

AnnotationPipeline
──────────────────
  1. Discover all .txt label files in label_dir (train + val splits if present)
  2. Validate every file via LabelValidator
  3. Optionally write fixed label files to output_dir (clamped boxes, dupes removed)
  4. Produce a DatasetReport with per-class counts, issue summary, quality metrics

Output directory layout (mirrors YOLO dataset structure)
─────────────────────────────────────────────────────────
  <output_dir>/
      labels/train/*.txt   — fixed labels (split preserved)
      labels/val/*.txt
      report.json          — machine-readable DatasetReport
      report.txt           — human-readable summary
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .validator import (
    Annotation, LabelFile, ValidationResult,
    ValidatorConfig, LabelValidator,
)


# ── dataset report ────────────────────────────────────────────────────────────

@dataclass
class DatasetReport:
    """Aggregated quality statistics for an annotated dataset."""
    n_label_files:       int
    n_annotations_raw:   int
    n_annotations_valid: int
    n_empty_labels:      int
    n_files_with_errors: int
    n_files_with_warnings: int
    n_filtered:          int
    per_class_counts:    dict[int, int]    # class_id → count of valid annotations
    class_names:         list[str]         # index → name (may be empty)
    issue_counts:        dict[str, int]    # issue code → occurrence count
    splits_found:        list[str]         # ["train", "val"] or [""]
    generated_at:        str
    elapsed_s:           float

    # ── derived ───────────────────────────────────────────────────────────────

    @property
    def filter_rate(self) -> float:
        if self.n_annotations_raw == 0:
            return 0.0
        return self.n_filtered / self.n_annotations_raw

    @property
    def mean_annotations_per_image(self) -> float:
        if self.n_label_files == 0:
            return 0.0
        return self.n_annotations_valid / self.n_label_files

    @property
    def classes_present(self) -> list[int]:
        return sorted(k for k, v in self.per_class_counts.items() if v > 0)

    @property
    def classes_missing(self) -> list[int]:
        n = len(self.class_names) or max((self.per_class_counts or {0: 0}).keys(), default=-1) + 1
        return [i for i in range(n) if self.per_class_counts.get(i, 0) == 0]

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["per_class_counts"] = {str(k): v for k, v in d["per_class_counts"].items()}
        path.write_text(json.dumps(d, indent=2))

    def save_text(self, path: str | Path) -> None:
        Path(path).write_text(str(self))

    def __str__(self) -> str:
        lines = [
            "Dataset Annotation Report",
            "─" * 40,
            f"  Label files      : {self.n_label_files}",
            f"  Annotations raw  : {self.n_annotations_raw}",
            f"  Annotations valid: {self.n_annotations_valid}  "
            f"(filtered {self.n_filtered}, {self.filter_rate*100:.1f}%)",
            f"  Empty labels     : {self.n_empty_labels}",
            f"  Files with errors: {self.n_files_with_errors}",
            f"  Files with warns : {self.n_files_with_warnings}",
            f"  Mean ann/image   : {self.mean_annotations_per_image:.2f}",
            f"  Splits           : {', '.join(self.splits_found) or 'flat'}",
            "",
            "Per-class annotation counts:",
        ]
        for cid in sorted(self.per_class_counts):
            name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            lines.append(f"  [{cid:2d}] {name:<32s} {self.per_class_counts[cid]:6d}")

        if self.classes_missing:
            lines.append(f"\n  WARNING: {len(self.classes_missing)} class(es) have zero annotations: "
                         f"{self.classes_missing[:10]}")

        if self.issue_counts:
            lines.append("\nIssue summary:")
            for code, cnt in sorted(self.issue_counts.items()):
                lines.append(f"  {code}: {cnt}")

        lines.append(f"\nGenerated: {self.generated_at}  ({self.elapsed_s:.1f}s)")
        return "\n".join(lines)


# ── pipeline config ───────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Configuration for AnnotationPipeline.

    label_dir       : root containing either *.txt files directly or
                      labels/train/ + labels/val/ subdirectories
    image_dir       : optional image root (same structure as label_dir);
                      used to verify image↔label pairing
    output_dir      : where to write fixed labels and the report;
                      None → validate-only (no writes)
    validator_cfg   : validation thresholds
    class_names     : list of class names indexed by class_id
    write_fixed     : write repaired label files to output_dir
    write_report    : write report.json + report.txt to output_dir
    """
    label_dir:     str
    image_dir:     Optional[str]          = None
    output_dir:    Optional[str]          = None
    validator_cfg: ValidatorConfig        = field(default_factory=ValidatorConfig)
    class_names:   list[str]              = field(default_factory=list)
    write_fixed:   bool                   = True
    write_report:  bool                   = True


# ── pipeline ──────────────────────────────────────────────────────────────────

class AnnotationPipeline:
    """
    End-to-end annotation validation and repair pipeline.

    Parameters
    ----------
    cfg : PipelineConfig
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg       = cfg
        self.validator = LabelValidator(cfg.validator_cfg)

    def run(self) -> DatasetReport:
        t0 = time.perf_counter()

        label_root = Path(self.cfg.label_dir)
        image_root = Path(self.cfg.image_dir) if self.cfg.image_dir else None
        out_root   = Path(self.cfg.output_dir) if self.cfg.output_dir else None

        # discover splits
        splits = self._find_splits(label_root)

        all_results: list[tuple[str, ValidationResult]] = []
        for split, txt_files in splits:
            img_dir = (image_root / split).resolve() if (image_root and split) else image_root
            for txt in txt_files:
                img_path = self._find_image(txt, img_dir)
                lf = LabelFile.from_file(txt, img_path)
                result = self.validator.validate_file(lf)
                all_results.append((split, result))

                if out_root and self.cfg.write_fixed:
                    self._write_fixed(result, txt, split, out_root)

        report = self._build_report(
            all_results,
            splits=[s for s, _ in splits],
            elapsed_s=time.perf_counter() - t0,
        )

        if out_root and self.cfg.write_report:
            out_root.mkdir(parents=True, exist_ok=True)
            report.save(out_root / "report.json")
            report.save_text(out_root / "report.txt")

        return report

    # ── internal ──────────────────────────────────────────────────────────────

    def _find_splits(
        self, label_root: Path
    ) -> list[tuple[str, list[Path]]]:
        """
        Returns list of (split_name, [txt_paths]).
        Supports two layouts:
          • flat:   <label_root>/*.txt
          • split:  <label_root>/labels/train/*.txt  +  labels/val/*.txt
                    or <label_root>/train/*.txt  +  val/*.txt
        """
        # check split subdirs
        for sub in ("labels", ""):
            base = label_root / sub if sub else label_root
            candidates = []
            for split in ("train", "val", "test"):
                d = base / split
                if d.is_dir():
                    candidates.append((split, sorted(d.glob("*.txt"))))
            if candidates:
                return candidates

        # flat
        txts = sorted(label_root.glob("*.txt"))
        return [("", txts)]

    def _find_image(
        self, label_path: Path, image_dir: Optional[Path]
    ) -> Optional[Path]:
        if image_dir is None:
            return None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = image_dir / (label_path.stem + ext)
            if p.exists():
                return p
        return None

    def _write_fixed(
        self,
        result: ValidationResult,
        original_txt: Path,
        split: str,
        out_root: Path,
    ) -> None:
        if split:
            dest = out_root / "labels" / split / original_txt.name
        else:
            dest = out_root / "labels" / original_txt.name
        lf_out = LabelFile(
            label_path  = dest,
            annotations = result.valid_annotations,
        )
        lf_out.save()

    def _build_report(
        self,
        all_results: list[tuple[str, ValidationResult]],
        splits: list[str],
        elapsed_s: float,
    ) -> DatasetReport:
        n_raw         = 0
        n_valid       = 0
        n_empty       = 0
        n_errors      = 0
        n_warnings    = 0
        n_filtered    = 0
        class_counts: dict[int, int]  = defaultdict(int)
        issue_counts: dict[str, int]  = defaultdict(int)

        for _, result in all_results:
            n_raw      += result.original_count
            n_valid    += len(result.valid_annotations)
            n_filtered += result.n_filtered

            has_error   = any(i.is_error   for i in result.issues)
            has_warning = any(i.is_warning for i in result.issues)
            is_empty    = (result.original_count == 0 and not result.issues) or \
                          any(i.code == "F1" for i in result.issues)

            if has_error:   n_errors   += 1
            if has_warning: n_warnings += 1
            if is_empty:    n_empty    += 1

            for ann in result.valid_annotations:
                class_counts[ann.class_id] += 1

            for issue in result.issues:
                issue_counts[issue.code] += 1

        # fill zeros for all known classes
        n_cls = self.cfg.validator_cfg.n_classes
        for i in range(n_cls):
            if i not in class_counts:
                class_counts[i] = 0

        unique_splits = sorted(set(s for s in splits if s))

        return DatasetReport(
            n_label_files         = len(all_results),
            n_annotations_raw     = n_raw,
            n_annotations_valid   = n_valid,
            n_empty_labels        = n_empty,
            n_files_with_errors   = n_errors,
            n_files_with_warnings = n_warnings,
            n_filtered            = n_filtered,
            per_class_counts      = dict(class_counts),
            class_names           = list(self.cfg.class_names),
            issue_counts          = dict(issue_counts),
            splits_found          = unique_splits or ["flat"],
            generated_at          = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            elapsed_s             = elapsed_s,
        )
