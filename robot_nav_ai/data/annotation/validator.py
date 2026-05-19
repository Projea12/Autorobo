"""
data/annotation/validator.py — YOLO label parsing, validation, and filtering.

Annotation pipeline
───────────────────
  LabelFile  — parses / serialises a single YOLO .txt label file
  LabelValidator — applies a configurable rule set to each annotation and
                   returns a ValidationResult with per-annotation issues and
                   the filtered list of annotations that passed.

Validation rules (per annotation)
──────────────────────────────────
  E1  class_id out of range [0, n_classes)
  E2  bbox coordinate not finite
  E3  cx or cy outside (0, 1)
  E4  w or h ≤ 0
  E5  box extends outside image (x0 < 0 or x1 > 1, same for y)
  W1  area < min_area (tiny / far-away object)
  W2  area > max_area (implausibly large)
  W3  aspect ratio outside [min_ar, max_ar] (degenerate shape)
  W4  confidence below threshold  (pseudo-labels only)
  W5  duplicate detection — IoU with another annotation > max_overlap_iou

File-level rules
────────────────
  F1  label file is empty (no annotations) — recorded as a warning
  F2  parse error — malformed line in label file
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


# ── annotation ────────────────────────────────────────────────────────────────

@dataclass
class Annotation:
    """One YOLO detection: normalised cx/cy/w/h in [0, 1]."""
    class_id:   int
    cx:         float
    cy:         float
    w:          float
    h:          float
    confidence: float = 1.0   # for pseudo-labels; always 1.0 for ground-truth

    # ── geometry helpers ──────────────────────────────────────────────────────

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return self.w / max(self.h, 1e-9)

    @property
    def x0(self) -> float:
        return self.cx - self.w / 2.0

    @property
    def y0(self) -> float:
        return self.cy - self.h / 2.0

    @property
    def x1(self) -> float:
        return self.cx + self.w / 2.0

    @property
    def y1(self) -> float:
        return self.cy + self.h / 2.0

    def iou(self, other: "Annotation") -> float:
        """Intersection-over-Union with another annotation."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter == 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / max(union, 1e-9)

    def clamp(self) -> "Annotation":
        """Return a copy with coordinates clamped to [0, 1]."""
        x0 = max(0.0, self.x0)
        y0 = max(0.0, self.y0)
        x1 = min(1.0, self.x1)
        y1 = min(1.0, self.y1)
        w  = max(0.0, x1 - x0)
        h  = max(0.0, y1 - y0)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        return Annotation(self.class_id, cx, cy, w, h, self.confidence)

    def yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    def __repr__(self) -> str:
        return (f"Annotation(cls={self.class_id}, "
                f"cx={self.cx:.3f}, cy={self.cy:.3f}, "
                f"w={self.w:.3f}, h={self.h:.3f}, "
                f"conf={self.confidence:.3f})")

    @classmethod
    def from_yolo_line(cls, line: str) -> "Annotation":
        """Parse a single YOLO label line."""
        parts = line.strip().split()
        if len(parts) not in (5, 6):
            raise ValueError(f"Expected 5 or 6 fields, got {len(parts)}: {line!r}")
        class_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        confidence = float(parts[5]) if len(parts) == 6 else 1.0
        return cls(class_id, cx, cy, w, h, confidence)


# ── label file ────────────────────────────────────────────────────────────────

@dataclass
class LabelFile:
    """Parsed YOLO label file (one image's annotations)."""
    label_path:   Path
    annotations:  list[Annotation]
    image_path:   Optional[Path] = None
    parse_errors: list[str] = field(default_factory=list)

    @classmethod
    def from_file(
        cls,
        label_path: str | Path,
        image_path: Optional[str | Path] = None,
    ) -> "LabelFile":
        label_path = Path(label_path)
        annotations: list[Annotation] = []
        errors: list[str] = []

        if label_path.exists():
            for lineno, line in enumerate(label_path.read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    annotations.append(Annotation.from_yolo_line(line))
                except (ValueError, IndexError) as exc:
                    errors.append(f"line {lineno}: {exc}")
        return cls(
            label_path   = label_path,
            annotations  = annotations,
            image_path   = Path(image_path) if image_path else None,
            parse_errors = errors,
        )

    def save(self, path: Optional[str | Path] = None) -> None:
        dest = Path(path) if path else self.label_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(a.yolo_line() for a in self.annotations) + "\n"
                        if self.annotations else "")

    @property
    def stem(self) -> str:
        return self.label_path.stem

    def __len__(self) -> int:
        return len(self.annotations)

    def __repr__(self) -> str:
        return f"LabelFile({self.label_path.name}, n={len(self.annotations)})"


# ── validation issue ──────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """One problem found during label validation."""
    code:           str             # E1, W1, F1, …
    severity:       str             # "error" | "warning"
    message:        str
    annotation_idx: Optional[int]  = None  # None → file-level issue

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    @property
    def is_warning(self) -> bool:
        return self.severity == "warning"

    def __str__(self) -> str:
        loc = f"ann[{self.annotation_idx}] " if self.annotation_idx is not None else ""
        return f"[{self.severity.upper()} {self.code}] {loc}{self.message}"


# ── validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of validating one LabelFile."""
    label_path:        Path
    issues:            list[ValidationIssue]
    valid_annotations: list[Annotation]
    original_count:    int

    @property
    def is_valid(self) -> bool:
        return not any(i.is_error for i in self.issues)

    @property
    def n_filtered(self) -> int:
        return self.original_count - len(self.valid_annotations)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.is_error]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.is_warning]

    def __repr__(self) -> str:
        status = "OK" if self.is_valid else "INVALID"
        return (f"ValidationResult({self.label_path.name}, {status}, "
                f"valid={len(self.valid_annotations)}/{self.original_count}, "
                f"issues={len(self.issues)})")


# ── validator config ──────────────────────────────────────────────────────────

@dataclass
class ValidatorConfig:
    """
    Validation rule thresholds.

    n_classes           : number of valid class IDs [0, n_classes)
    min_area            : minimum normalised bbox area (w*h); below → W1
    max_area            : maximum normalised bbox area; above → W2
    min_aspect_ratio    : minimum w/h; below → W3
    max_aspect_ratio    : maximum w/h; above → W3
    confidence_threshold: minimum confidence for pseudo-labels; below → W4
    clamp_border        : clamp boxes that slightly overflow image boundary
                          instead of marking them as E5 errors
    max_overlap_iou     : IoU threshold for duplicate suppression (W5);
                          1.0 = no deduplication
    """
    n_classes:           int   = 21
    min_area:            float = 1e-4
    max_area:            float = 0.99
    min_aspect_ratio:    float = 0.02
    max_aspect_ratio:    float = 50.0
    confidence_threshold: float = 0.0
    clamp_border:        bool  = True
    max_overlap_iou:     float = 0.85


# ── validator ─────────────────────────────────────────────────────────────────

class LabelValidator:
    """
    Validates and filters YOLO label files according to a ValidatorConfig.

    Usage
    -----
        validator = LabelValidator(ValidatorConfig(n_classes=21))
        result    = validator.validate_file(LabelFile.from_file("path/to/label.txt"))
        print(result)
    """

    def __init__(self, cfg: ValidatorConfig = ValidatorConfig()) -> None:
        self.cfg = cfg

    # ── public ────────────────────────────────────────────────────────────────

    def validate_file(self, lf: LabelFile) -> ValidationResult:
        issues:  list[ValidationIssue] = []
        passing: list[Annotation]      = []

        # file-level: parse errors
        for err in lf.parse_errors:
            issues.append(ValidationIssue("F2", "error",
                                          f"Parse error: {err}"))

        # file-level: empty label
        if not lf.annotations and not lf.parse_errors:
            issues.append(ValidationIssue("F1", "warning",
                                          "No annotations (empty label file)"))

        # per-annotation checks
        survived: list[tuple[int, Annotation]] = []
        for idx, ann in enumerate(lf.annotations):
            ann_issues, ann_out = self._check_annotation(idx, ann)
            issues.extend(ann_issues)
            has_error = any(i.is_error for i in ann_issues)
            if not has_error and ann_out is not None:
                survived.append((idx, ann_out))

        # deduplication (W5)
        passing = self._deduplicate(survived, issues)

        return ValidationResult(
            label_path        = lf.label_path,
            issues            = issues,
            valid_annotations = passing,
            original_count    = len(lf.annotations),
        )

    def validate_dataset(
        self,
        label_dir:  str | Path,
        image_dir:  Optional[str | Path] = None,
    ) -> list[ValidationResult]:
        """Validate every .txt file in label_dir."""
        label_dir = Path(label_dir)
        results: list[ValidationResult] = []
        for txt in sorted(label_dir.glob("*.txt")):
            img_path = None
            if image_dir:
                for ext in (".jpg", ".jpeg", ".png"):
                    candidate = Path(image_dir) / (txt.stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
            lf = LabelFile.from_file(txt, img_path)
            results.append(self.validate_file(lf))
        return results

    # ── internal ──────────────────────────────────────────────────────────────

    def _check_annotation(
        self, idx: int, ann: Annotation
    ) -> tuple[list[ValidationIssue], Optional[Annotation]]:
        issues: list[ValidationIssue] = []
        out = ann

        # E1 — class id
        if ann.class_id < 0 or ann.class_id >= self.cfg.n_classes:
            issues.append(ValidationIssue(
                "E1", "error",
                f"class_id={ann.class_id} not in [0, {self.cfg.n_classes})",
                idx,
            ))
            return issues, None

        # E2 — finite coordinates
        for val, name in [(ann.cx, "cx"), (ann.cy, "cy"),
                           (ann.w,  "w"),  (ann.h,  "h")]:
            if not math.isfinite(val):
                issues.append(ValidationIssue(
                    "E2", "error", f"{name}={val} is not finite", idx))
                return issues, None

        # E4 — non-positive dimensions
        if ann.w <= 0 or ann.h <= 0:
            issues.append(ValidationIssue(
                "E4", "error",
                f"degenerate bbox: w={ann.w:.6f}, h={ann.h:.6f}", idx))
            return issues, None

        # E3 — centre inside image
        if not (0.0 < ann.cx < 1.0) or not (0.0 < ann.cy < 1.0):
            issues.append(ValidationIssue(
                "E3", "error",
                f"centre outside image: cx={ann.cx:.4f}, cy={ann.cy:.4f}", idx))
            return issues, None

        # E5 / clamp — box boundary
        overflows = ann.x0 < 0.0 or ann.x1 > 1.0 or ann.y0 < 0.0 or ann.y1 > 1.0
        if overflows:
            if self.cfg.clamp_border:
                out = ann.clamp()
                if out.w <= 0 or out.h <= 0:
                    issues.append(ValidationIssue(
                        "E5", "error",
                        "bbox entirely outside image after clamping", idx))
                    return issues, None
            else:
                issues.append(ValidationIssue(
                    "E5", "error",
                    f"bbox overflows image: x=[{ann.x0:.4f},{ann.x1:.4f}] "
                    f"y=[{ann.y0:.4f},{ann.y1:.4f}]", idx))
                return issues, None

        # W1 — area too small
        if out.area < self.cfg.min_area:
            issues.append(ValidationIssue(
                "W1", "warning",
                f"area={out.area:.6f} < min_area={self.cfg.min_area}", idx))

        # W2 — area too large
        if out.area > self.cfg.max_area:
            issues.append(ValidationIssue(
                "W2", "warning",
                f"area={out.area:.4f} > max_area={self.cfg.max_area}", idx))

        # W3 — aspect ratio
        ar = out.aspect_ratio
        if ar < self.cfg.min_aspect_ratio or ar > self.cfg.max_aspect_ratio:
            issues.append(ValidationIssue(
                "W3", "warning",
                f"aspect_ratio={ar:.3f} outside "
                f"[{self.cfg.min_aspect_ratio}, {self.cfg.max_aspect_ratio}]", idx))

        # W4 — confidence threshold (pseudo-labels)
        if ann.confidence < self.cfg.confidence_threshold:
            issues.append(ValidationIssue(
                "W4", "warning",
                f"confidence={ann.confidence:.3f} < "
                f"threshold={self.cfg.confidence_threshold}", idx))

        return issues, out

    def _deduplicate(
        self,
        survived: list[tuple[int, Annotation]],
        issues:   list[ValidationIssue],
    ) -> list[Annotation]:
        if self.cfg.max_overlap_iou >= 1.0:
            return [a for _, a in survived]

        kept: list[tuple[int, Annotation]] = []
        for idx, ann in survived:
            is_dup = False
            for _, other in kept:
                if ann.class_id == other.class_id and ann.iou(other) > self.cfg.max_overlap_iou:
                    issues.append(ValidationIssue(
                        "W5", "warning",
                        f"duplicate annotation (iou > {self.cfg.max_overlap_iou:.2f}) — dropped",
                        idx,
                    ))
                    is_dup = True
                    break
            if not is_dup:
                kept.append((idx, ann))
        return [a for _, a in kept]
