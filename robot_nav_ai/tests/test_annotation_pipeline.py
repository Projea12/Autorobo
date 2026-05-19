"""
tests/test_annotation_pipeline.py — Annotation validator and pipeline tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.annotation.validator import (
    Annotation, LabelFile, ValidationIssue, ValidationResult,
    ValidatorConfig, LabelValidator,
)
from data.annotation.pipeline import (
    PipelineConfig, DatasetReport, AnnotationPipeline,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ann(class_id=0, cx=0.5, cy=0.5, w=0.2, h=0.3, conf=1.0) -> Annotation:
    return Annotation(class_id, cx, cy, w, h, conf)


def _lf(annotations, label_path=Path("test.txt"), errors=None) -> LabelFile:
    return LabelFile(
        label_path   = label_path,
        annotations  = annotations,
        parse_errors = errors or [],
    )


def _validator(**kwargs) -> LabelValidator:
    defaults = {"n_classes": 21}
    defaults.update(kwargs)
    return LabelValidator(ValidatorConfig(**defaults))


def _write_labels(tmp_path: Path, stem: str, lines: list[str]) -> Path:
    p = tmp_path / f"{stem}.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Annotation
# ══════════════════════════════════════════════════════════════════════════════

def test_annotation_area():
    assert _ann(w=0.4, h=0.5).area == pytest.approx(0.2)


def test_annotation_aspect_ratio():
    assert _ann(w=0.4, h=0.2).aspect_ratio == pytest.approx(2.0)


def test_annotation_x0_x1():
    a = _ann(cx=0.5, w=0.4)
    assert a.x0 == pytest.approx(0.3)
    assert a.x1 == pytest.approx(0.7)


def test_annotation_y0_y1():
    a = _ann(cy=0.6, h=0.2)
    assert a.y0 == pytest.approx(0.5)
    assert a.y1 == pytest.approx(0.7)


def test_annotation_iou_identical():
    a = _ann()
    assert a.iou(a) == pytest.approx(1.0)


def test_annotation_iou_no_overlap():
    a = _ann(cx=0.1, cy=0.1, w=0.1, h=0.1)
    b = _ann(cx=0.9, cy=0.9, w=0.1, h=0.1)
    assert a.iou(b) == pytest.approx(0.0)


def test_annotation_iou_partial():
    a = _ann(cx=0.4, cy=0.5, w=0.4, h=0.4)
    b = _ann(cx=0.6, cy=0.5, w=0.4, h=0.4)
    assert 0.0 < a.iou(b) < 1.0


def test_annotation_clamp_overflow():
    a = Annotation(0, cx=0.9, cy=0.5, w=0.4, h=0.2)  # x1 = 1.1
    c = a.clamp()
    assert c.x1 <= 1.0


def test_annotation_clamp_preserves_valid():
    a = _ann(cx=0.5, cy=0.5, w=0.2, h=0.2)
    c = a.clamp()
    assert c.cx == pytest.approx(a.cx)
    assert c.w  == pytest.approx(a.w)


def test_annotation_yolo_line_format():
    line = _ann(class_id=3).yolo_line()
    parts = line.split()
    assert len(parts) == 5
    assert parts[0] == "3"


def test_annotation_from_yolo_line_roundtrip():
    a    = _ann(class_id=5, cx=0.3, cy=0.7, w=0.1, h=0.2)
    line = a.yolo_line()
    a2   = Annotation.from_yolo_line(line)
    assert a2.class_id == 5
    assert a2.cx == pytest.approx(0.3, rel=1e-5)
    assert a2.w  == pytest.approx(0.1, rel=1e-5)


def test_annotation_from_yolo_line_with_confidence():
    a = Annotation.from_yolo_line("2 0.5 0.5 0.2 0.3 0.87")
    assert a.confidence == pytest.approx(0.87)


def test_annotation_from_yolo_line_bad_fields():
    with pytest.raises(ValueError):
        Annotation.from_yolo_line("1 0.5 0.5")  # only 3 fields


def test_annotation_repr():
    assert "cls=0" in repr(_ann())


# ══════════════════════════════════════════════════════════════════════════════
# LabelFile
# ══════════════════════════════════════════════════════════════════════════════

def test_labelfile_from_file_parses(tmp_path):
    p = _write_labels(tmp_path, "img0", ["0 0.5 0.5 0.2 0.3", "1 0.3 0.4 0.1 0.2"])
    lf = LabelFile.from_file(p)
    assert len(lf.annotations) == 2


def test_labelfile_from_file_empty(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    lf = LabelFile.from_file(p)
    assert len(lf.annotations) == 0


def test_labelfile_from_file_nonexistent(tmp_path):
    lf = LabelFile.from_file(tmp_path / "missing.txt")
    assert lf.annotations == []
    assert lf.parse_errors == []


def test_labelfile_from_file_bad_line(tmp_path):
    p = _write_labels(tmp_path, "bad", ["0 0.5 0.5 0.2 0.3", "not a valid line"])
    lf = LabelFile.from_file(p)
    assert len(lf.parse_errors) == 1


def test_labelfile_save_roundtrip(tmp_path):
    anns = [_ann(class_id=0), _ann(class_id=3, cx=0.7)]
    lf   = LabelFile(label_path=tmp_path / "out.txt", annotations=anns)
    lf.save()
    lf2 = LabelFile.from_file(tmp_path / "out.txt")
    assert len(lf2.annotations) == 2
    assert lf2.annotations[0].class_id == 0
    assert lf2.annotations[1].class_id == 3


def test_labelfile_save_empty(tmp_path):
    lf = LabelFile(label_path=tmp_path / "empty.txt", annotations=[])
    lf.save()
    assert (tmp_path / "empty.txt").read_text() == ""


def test_labelfile_stem():
    lf = LabelFile(label_path=Path("foo/bar.txt"), annotations=[])
    assert lf.stem == "bar"


def test_labelfile_len():
    lf = _lf([_ann(), _ann()])
    assert len(lf) == 2


# ══════════════════════════════════════════════════════════════════════════════
# LabelValidator — per-annotation rules
# ══════════════════════════════════════════════════════════════════════════════

def test_valid_annotation_no_issues():
    v   = _validator()
    r   = v.validate_file(_lf([_ann()]))
    ann_issues = [i for i in r.issues if i.annotation_idx is not None]
    assert ann_issues == []


def test_e1_class_out_of_range_high():
    v = _validator(n_classes=5)
    r = v.validate_file(_lf([_ann(class_id=5)]))
    codes = [i.code for i in r.issues]
    assert "E1" in codes


def test_e1_class_negative():
    v = _validator()
    r = v.validate_file(_lf([_ann(class_id=-1)]))
    assert any(i.code == "E1" for i in r.issues)


def test_e2_non_finite_cx():
    v = _validator()
    r = v.validate_file(_lf([_ann(cx=float("nan"))]))
    assert any(i.code == "E2" for i in r.issues)


def test_e2_inf_h():
    v = _validator()
    r = v.validate_file(_lf([_ann(h=float("inf"))]))
    assert any(i.code == "E2" for i in r.issues)


def test_e3_centre_outside_image():
    v = _validator()
    r = v.validate_file(_lf([_ann(cx=1.5)]))
    assert any(i.code == "E3" for i in r.issues)


def test_e3_centre_at_zero():
    v = _validator()
    r = v.validate_file(_lf([_ann(cx=0.0, cy=0.5)]))
    assert any(i.code == "E3" for i in r.issues)


def test_e4_zero_width():
    v = _validator()
    r = v.validate_file(_lf([_ann(w=0.0)]))
    assert any(i.code == "E4" for i in r.issues)


def test_e4_negative_height():
    v = _validator()
    r = v.validate_file(_lf([_ann(h=-0.1)]))
    assert any(i.code == "E4" for i in r.issues)


def test_e5_clamp_mode_no_error():
    # Box overflows slightly — should be clamped, not rejected
    v = _validator(clamp_border=True)
    a = Annotation(0, cx=0.95, cy=0.5, w=0.2, h=0.2)  # x1 = 1.05
    r = v.validate_file(_lf([a]))
    assert not any(i.code == "E5" for i in r.issues)
    assert len(r.valid_annotations) == 1
    assert r.valid_annotations[0].x1 <= 1.0


def test_e5_no_clamp_mode_error():
    v = _validator(clamp_border=False)
    a = Annotation(0, cx=0.95, cy=0.5, w=0.2, h=0.2)  # x1 = 1.05
    r = v.validate_file(_lf([a]))
    assert any(i.code == "E5" for i in r.issues)
    assert len(r.valid_annotations) == 0


def test_w1_small_area():
    v = _validator(min_area=0.01)
    a = _ann(w=0.05, h=0.05)   # area = 0.0025 < 0.01
    r = v.validate_file(_lf([a]))
    assert any(i.code == "W1" for i in r.issues)


def test_w1_small_area_still_kept():
    v = _validator(min_area=0.01)
    a = _ann(w=0.05, h=0.05)
    r = v.validate_file(_lf([a]))
    # W1 is a warning, not an error — annotation survives
    assert len(r.valid_annotations) == 1


def test_w2_large_area():
    v = _validator(max_area=0.5)
    a = _ann(w=0.9, h=0.9)   # area = 0.81 > 0.5
    r = v.validate_file(_lf([a]))
    assert any(i.code == "W2" for i in r.issues)


def test_w3_extreme_aspect_ratio():
    v = _validator(max_aspect_ratio=10.0)
    a = _ann(w=0.5, h=0.01)  # ar = 50
    r = v.validate_file(_lf([a]))
    assert any(i.code == "W3" for i in r.issues)


def test_w4_low_confidence():
    v = _validator(confidence_threshold=0.5)
    a = _ann(conf=0.3)
    r = v.validate_file(_lf([a]))
    assert any(i.code == "W4" for i in r.issues)


def test_w4_sufficient_confidence():
    v = _validator(confidence_threshold=0.5)
    a = _ann(conf=0.8)
    r = v.validate_file(_lf([a]))
    assert not any(i.code == "W4" for i in r.issues)


def test_w5_duplicate_removed():
    v = _validator(max_overlap_iou=0.8)
    a = _ann(cx=0.5, cy=0.5, w=0.3, h=0.3)
    b = _ann(cx=0.5, cy=0.5, w=0.3, h=0.3)   # identical → iou = 1.0
    r = v.validate_file(_lf([a, b]))
    assert any(i.code == "W5" for i in r.issues)
    assert len(r.valid_annotations) == 1


def test_w5_no_dedup_when_disabled():
    v = _validator(max_overlap_iou=1.0)   # 1.0 = off
    a = _ann(cx=0.5, cy=0.5, w=0.3, h=0.3)
    b = _ann(cx=0.5, cy=0.5, w=0.3, h=0.3)
    r = v.validate_file(_lf([a, b]))
    assert len(r.valid_annotations) == 2


# ══════════════════════════════════════════════════════════════════════════════
# LabelValidator — file-level rules
# ══════════════════════════════════════════════════════════════════════════════

def test_f1_empty_label_warning():
    v = _validator()
    r = v.validate_file(_lf([]))
    assert any(i.code == "F1" for i in r.issues)


def test_f2_parse_error():
    v  = _validator()
    lf = _lf([], errors=["line 3: invalid value"])
    r  = v.validate_file(lf)
    assert any(i.code == "F2" for i in r.issues)


def test_f2_is_error_not_warning():
    v  = _validator()
    lf = _lf([], errors=["bad"])
    r  = v.validate_file(lf)
    f2 = next(i for i in r.issues if i.code == "F2")
    assert f2.is_error


# ══════════════════════════════════════════════════════════════════════════════
# ValidationResult
# ══════════════════════════════════════════════════════════════════════════════

def test_result_is_valid_no_errors():
    v = _validator()
    r = v.validate_file(_lf([_ann()]))
    assert r.is_valid


def test_result_is_invalid_with_errors():
    v = _validator()
    r = v.validate_file(_lf([_ann(class_id=999)]))
    assert not r.is_valid


def test_result_n_filtered():
    v  = _validator()
    lf = _lf([_ann(), _ann(class_id=999)])  # one bad
    r  = v.validate_file(lf)
    assert r.n_filtered == 1


def test_result_errors_property():
    v  = _validator()
    lf = _lf([_ann(class_id=999)])
    r  = v.validate_file(lf)
    assert len(r.errors) >= 1


def test_result_warnings_property():
    v  = _validator(min_area=0.5)
    lf = _lf([_ann(w=0.1, h=0.1)])   # area=0.01 < 0.5 → W1
    r  = v.validate_file(lf)
    assert len(r.warnings) >= 1


def test_result_repr():
    v = _validator()
    r = v.validate_file(_lf([_ann()]))
    assert "ValidationResult" in repr(r)


# ══════════════════════════════════════════════════════════════════════════════
# LabelValidator.validate_dataset
# ══════════════════════════════════════════════════════════════════════════════

def test_validate_dataset_returns_one_per_file(tmp_path):
    for i in range(4):
        _write_labels(tmp_path, f"img{i}", [f"{i % 3} 0.5 0.5 0.2 0.3"])
    v = _validator()
    results = v.validate_dataset(tmp_path)
    assert len(results) == 4


def test_validate_dataset_empty_dir(tmp_path):
    results = _validator().validate_dataset(tmp_path)
    assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# AnnotationPipeline — flat layout
# ══════════════════════════════════════════════════════════════════════════════

def _make_flat_dataset(tmp_path: Path, n_files=6) -> Path:
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    for i in range(n_files):
        lines = [f"{i % 5} 0.5 0.5 {0.1 + i*0.05:.2f} 0.2"]
        _write_labels(label_dir, f"img{i:04d}", lines)
    return label_dir


def test_pipeline_runs(tmp_path):
    label_dir = _make_flat_dataset(tmp_path)
    cfg    = PipelineConfig(label_dir=str(label_dir))
    report = AnnotationPipeline(cfg).run()
    assert isinstance(report, DatasetReport)


def test_pipeline_n_label_files(tmp_path):
    label_dir = _make_flat_dataset(tmp_path, n_files=8)
    cfg    = PipelineConfig(label_dir=str(label_dir))
    report = AnnotationPipeline(cfg).run()
    assert report.n_label_files == 8


def test_pipeline_n_annotations_raw(tmp_path):
    label_dir = _make_flat_dataset(tmp_path, n_files=5)
    cfg    = PipelineConfig(label_dir=str(label_dir))
    report = AnnotationPipeline(cfg).run()
    assert report.n_annotations_raw == 5


def test_pipeline_zero_errors_clean_data(tmp_path):
    label_dir = _make_flat_dataset(tmp_path)
    cfg    = PipelineConfig(label_dir=str(label_dir))
    report = AnnotationPipeline(cfg).run()
    assert report.n_files_with_errors == 0


def test_pipeline_detects_error_files(tmp_path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _write_labels(label_dir, "good",    ["0 0.5 0.5 0.2 0.3"])
    _write_labels(label_dir, "bad_cls", ["999 0.5 0.5 0.2 0.3"])
    cfg    = PipelineConfig(label_dir=str(label_dir))
    report = AnnotationPipeline(cfg).run()
    assert report.n_files_with_errors == 1


def test_pipeline_counts_empty_labels(tmp_path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _write_labels(label_dir, "nonempty", ["0 0.5 0.5 0.2 0.3"])
    (label_dir / "empty.txt").write_text("")
    cfg    = PipelineConfig(label_dir=str(label_dir))
    report = AnnotationPipeline(cfg).run()
    assert report.n_empty_labels == 1


def test_pipeline_per_class_counts(tmp_path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _write_labels(label_dir, "a", ["0 0.5 0.5 0.2 0.3", "0 0.3 0.3 0.1 0.1"])
    _write_labels(label_dir, "b", ["1 0.5 0.5 0.2 0.3"])
    cfg    = PipelineConfig(label_dir=str(label_dir))
    report = AnnotationPipeline(cfg).run()
    assert report.per_class_counts[0] == 2
    assert report.per_class_counts[1] == 1


def test_pipeline_classes_missing(tmp_path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _write_labels(label_dir, "a", ["0 0.5 0.5 0.2 0.3"])
    cfg    = PipelineConfig(label_dir=str(label_dir),
                             validator_cfg=ValidatorConfig(n_classes=3))
    report = AnnotationPipeline(cfg).run()
    assert 1 in report.classes_missing
    assert 2 in report.classes_missing


def test_pipeline_classes_present(tmp_path):
    label_dir = _make_flat_dataset(tmp_path, n_files=5)
    cfg    = PipelineConfig(label_dir=str(label_dir))
    report = AnnotationPipeline(cfg).run()
    assert len(report.classes_present) > 0


# ══════════════════════════════════════════════════════════════════════════════
# AnnotationPipeline — split layout
# ══════════════════════════════════════════════════════════════════════════════

def _make_split_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split, n in [("train", 8), ("val", 2)]:
        d = root / "labels" / split
        d.mkdir(parents=True)
        for i in range(n):
            _write_labels(d, f"img{i:04d}", [f"{i % 4} 0.5 0.5 0.2 0.3"])
    return root / "labels"


def test_pipeline_split_layout(tmp_path):
    label_dir = _make_split_dataset(tmp_path)
    cfg    = PipelineConfig(label_dir=str(label_dir))
    report = AnnotationPipeline(cfg).run()
    assert report.n_label_files == 10


def test_pipeline_splits_found(tmp_path):
    label_dir = _make_split_dataset(tmp_path)
    cfg    = PipelineConfig(label_dir=str(label_dir))
    report = AnnotationPipeline(cfg).run()
    assert "train" in report.splits_found
    assert "val"   in report.splits_found


# ══════════════════════════════════════════════════════════════════════════════
# AnnotationPipeline — write fixed labels
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_writes_fixed_labels(tmp_path):
    label_dir = _make_flat_dataset(tmp_path, n_files=4)
    out_dir   = tmp_path / "out"
    cfg = PipelineConfig(label_dir=str(label_dir), output_dir=str(out_dir),
                          write_fixed=True)
    AnnotationPipeline(cfg).run()
    fixed = list((out_dir / "labels").rglob("*.txt"))
    assert len(fixed) == 4


def test_pipeline_fixed_labels_valid_content(tmp_path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    # one overflow box that will be clamped
    _write_labels(label_dir, "img0", ["0 0.95 0.5 0.2 0.2"])
    out_dir = tmp_path / "out"
    cfg = PipelineConfig(label_dir=str(label_dir), output_dir=str(out_dir),
                          write_fixed=True)
    AnnotationPipeline(cfg).run()
    content = (out_dir / "labels" / "img0.txt").read_text()
    parts = content.strip().split()
    assert len(parts) == 5
    # clamped x1 ≤ 1.0
    cx, w = float(parts[1]), float(parts[3])
    assert cx + w / 2 <= 1.0 + 1e-6


def test_pipeline_bad_annotation_not_in_fixed(tmp_path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    _write_labels(label_dir, "img0", ["999 0.5 0.5 0.2 0.3"])  # bad class
    out_dir = tmp_path / "out"
    cfg = PipelineConfig(label_dir=str(label_dir), output_dir=str(out_dir),
                          write_fixed=True)
    AnnotationPipeline(cfg).run()
    content = (out_dir / "labels" / "img0.txt").read_text()
    assert content.strip() == ""


# ══════════════════════════════════════════════════════════════════════════════
# AnnotationPipeline — report files
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_writes_report_json(tmp_path):
    label_dir = _make_flat_dataset(tmp_path)
    out_dir   = tmp_path / "out"
    cfg = PipelineConfig(label_dir=str(label_dir), output_dir=str(out_dir))
    AnnotationPipeline(cfg).run()
    assert (out_dir / "report.json").exists()


def test_pipeline_writes_report_txt(tmp_path):
    label_dir = _make_flat_dataset(tmp_path)
    out_dir   = tmp_path / "out"
    cfg = PipelineConfig(label_dir=str(label_dir), output_dir=str(out_dir))
    AnnotationPipeline(cfg).run()
    assert (out_dir / "report.txt").exists()


def test_pipeline_report_json_has_n_label_files(tmp_path):
    label_dir = _make_flat_dataset(tmp_path, n_files=5)
    out_dir   = tmp_path / "out"
    cfg = PipelineConfig(label_dir=str(label_dir), output_dir=str(out_dir))
    AnnotationPipeline(cfg).run()
    data = json.loads((out_dir / "report.json").read_text())
    assert data["n_label_files"] == 5


# ══════════════════════════════════════════════════════════════════════════════
# DatasetReport
# ══════════════════════════════════════════════════════════════════════════════

def _dummy_report(**kwargs) -> DatasetReport:
    defaults = dict(
        n_label_files=10, n_annotations_raw=50, n_annotations_valid=48,
        n_empty_labels=1, n_files_with_errors=1, n_files_with_warnings=3,
        n_filtered=2, per_class_counts={0: 30, 1: 18}, class_names=["cls0", "cls1"],
        issue_counts={"W1": 2, "E1": 1}, splits_found=["train", "val"],
        generated_at="2026-01-01T00:00:00Z", elapsed_s=0.5,
    )
    defaults.update(kwargs)
    return DatasetReport(**defaults)


def test_report_filter_rate():
    r = _dummy_report(n_annotations_raw=100, n_filtered=10)
    assert r.filter_rate == pytest.approx(0.1)


def test_report_mean_annotations_per_image():
    r = _dummy_report(n_label_files=10, n_annotations_valid=30)
    assert r.mean_annotations_per_image == pytest.approx(3.0)


def test_report_classes_present():
    r = _dummy_report(per_class_counts={0: 5, 1: 0, 2: 3})
    assert 0 in r.classes_present
    assert 1 not in r.classes_present
    assert 2 in r.classes_present


def test_report_classes_missing():
    r = _dummy_report(per_class_counts={0: 5}, class_names=["a", "b", "c"])
    assert 1 in r.classes_missing
    assert 2 in r.classes_missing


def test_report_str_contains_label_files():
    s = str(_dummy_report())
    assert "Label files" in s


def test_report_str_contains_filter_rate():
    s = str(_dummy_report())
    assert "filter" in s.lower()


def test_report_save_json(tmp_path):
    r = _dummy_report()
    r.save(tmp_path / "rep.json")
    loaded = json.loads((tmp_path / "rep.json").read_text())
    assert loaded["n_label_files"] == 10


def test_report_save_text(tmp_path):
    r = _dummy_report()
    r.save_text(tmp_path / "rep.txt")
    assert (tmp_path / "rep.txt").exists()


def test_report_elapsed_positive():
    r = _dummy_report(elapsed_s=1.23)
    assert r.elapsed_s == pytest.approx(1.23)
