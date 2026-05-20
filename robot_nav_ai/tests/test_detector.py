"""
tests/test_detector.py — Unit tests for ObjectDetector and Detection.

ultralytics is mocked throughout.  Tests verify the parsing logic, config
defaults, graceful degradation, and the Detection dataclass API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import perception.detector as _det_mod
from perception.detector import Detection, DetectorConfig, ObjectDetector


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_ultralytics(class_names=None):
    """Build a minimal mock ultralytics module."""
    names = class_names or [f"class_{i}" for i in range(21)]
    mock_model = MagicMock()
    mock_model.names = {i: n for i, n in enumerate(names)}

    mock_ult = MagicMock()
    mock_ult.YOLO.return_value = mock_model
    return mock_ult, mock_model


def _make_boxes(n: int, class_names: list[str]):
    """Build a mock ultralytics Boxes object with n detections."""
    import torch
    xyxy = torch.zeros(n, 4)
    xywh = torch.zeros(n, 4)
    for i in range(n):
        xyxy[i] = torch.tensor([10.0 * i, 5.0, 10.0 * i + 40, 45.0])
        xywh[i] = torch.tensor([10.0 * i + 20, 25.0, 40.0, 40.0])
    conf = torch.linspace(0.9, 0.5, n)
    cls  = torch.arange(n, dtype=torch.float32) % len(class_names)

    boxes = MagicMock()
    boxes.__len__ = MagicMock(return_value=n)
    boxes.xyxy = xyxy
    boxes.xywh = xywh
    boxes.conf = conf
    boxes.cls  = cls
    return boxes


def _make_result(n: int, class_names: list[str]):
    """Build a mock ultralytics Results object."""
    result = MagicMock()
    if n == 0:
        result.boxes = None
    else:
        result.boxes = _make_boxes(n, class_names)
    return result


def _detector_with_mock(mock_ult, cfg=None):
    cfg = cfg or DetectorConfig()
    with patch.object(_det_mod, "_ultralytics", mock_ult):
        det = ObjectDetector(cfg)
    det._class_names = list(mock_ult.YOLO.return_value.names.values())
    return det, mock_ult.YOLO.return_value


# ── Detection dataclass ───────────────────────────────────────────────────────

class TestDetection:
    def _make(self, cid=0, name="mug", conf=0.85,
              xyxy=(10, 20, 110, 220), xywh=(60, 120, 100, 200)):
        return Detection(
            class_id   = cid,
            class_name = name,
            confidence = conf,
            bbox_xyxy  = np.array(xyxy, dtype=np.float32),
            bbox_xywh  = np.array(xywh, dtype=np.float32),
        )

    def test_repr_contains_class_name(self):
        d = self._make(name="025_mug")
        assert "025_mug" in repr(d)

    def test_repr_contains_confidence(self):
        d = self._make(conf=0.85)
        assert "0.85" in repr(d)

    def test_area(self):
        d = self._make(xywh=(60, 120, 100, 200))
        assert d.area == pytest.approx(100.0 * 200.0)

    def test_centre(self):
        d = self._make(xywh=(60, 120, 100, 200))
        np.testing.assert_array_almost_equal(d.centre, [60, 120])

    def test_mask_none_by_default(self):
        assert self._make().mask is None

    def test_position_3d_none_by_default(self):
        assert self._make().position_3d is None

    def test_track_id_none_by_default(self):
        assert self._make().track_id is None

    def test_repr_with_position(self):
        d = self._make()
        d.position_3d = np.array([1.0, 0.5, 0.3])
        assert "pos=" in repr(d)

    def test_bbox_xyxy_dtype(self):
        d = self._make()
        assert d.bbox_xyxy.dtype == np.float32

    def test_bbox_xywh_dtype(self):
        d = self._make()
        assert d.bbox_xywh.dtype == np.float32


# ── DetectorConfig ────────────────────────────────────────────────────────────

class TestDetectorConfig:
    def test_defaults(self):
        cfg = DetectorConfig()
        assert cfg.weights_path == "yolov8n.pt"
        assert cfg.conf_thresh  == pytest.approx(0.25)
        assert cfg.iou_thresh   == pytest.approx(0.45)
        assert cfg.device       == ""
        assert cfg.half         is False
        assert cfg.imgsz        == 640
        assert cfg.max_det      == 100

    def test_frozen(self):
        cfg = DetectorConfig()
        with pytest.raises(Exception):
            cfg.conf_thresh = 0.9

    def test_custom(self):
        cfg = DetectorConfig(conf_thresh=0.5, device="cuda")
        assert cfg.conf_thresh == pytest.approx(0.5)
        assert cfg.device == "cuda"


# ── ObjectDetector — ultralytics unavailable ──────────────────────────────────

class TestDetectorUnavailable:
    def test_construction_succeeds_without_ultralytics(self):
        with patch.object(_det_mod, "_ultralytics", None):
            det = ObjectDetector()
        assert det is not None

    def test_is_loaded_false_without_ultralytics(self):
        with patch.object(_det_mod, "_ultralytics", None):
            det = ObjectDetector()
        assert det.is_loaded is False

    def test_detect_raises_import_error(self):
        with patch.object(_det_mod, "_ultralytics", None):
            det = ObjectDetector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(ImportError, match="ultralytics"):
            det.detect(img)

    def test_detect_batch_raises_import_error(self):
        with patch.object(_det_mod, "_ultralytics", None):
            det = ObjectDetector()
        with pytest.raises(ImportError):
            det.detect_batch([np.zeros((480, 640, 3), dtype=np.uint8)])

    def test_class_names_empty_without_ultralytics(self):
        with patch.object(_det_mod, "_ultralytics", None):
            det = ObjectDetector()
        assert det.class_names == []


# ── ObjectDetector — happy path (mocked) ────────────────────────────────────

class TestDetectorHappyPath:
    def test_is_loaded_true_with_mock(self):
        mock_ult, _ = _mock_ultralytics()
        det, _ = _detector_with_mock(mock_ult)
        assert det.is_loaded is True

    def test_class_names_set(self):
        mock_ult, _ = _mock_ultralytics(["can", "box", "mug"])
        det, _ = _detector_with_mock(mock_ult)
        assert "can" in det.class_names

    def test_n_classes(self):
        names = [f"obj_{i}" for i in range(21)]
        mock_ult, _ = _mock_ultralytics(names)
        det, _ = _detector_with_mock(mock_ult)
        assert det.n_classes == 21

    def test_detect_returns_list(self):
        mock_ult, mock_model = _mock_ultralytics()
        result = _make_result(3, list(mock_ult.YOLO.return_value.names.values()))
        mock_model.predict.return_value = [result]
        det, _ = _detector_with_mock(mock_ult)
        dets = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert isinstance(dets, list)

    def test_detect_returns_correct_count(self):
        mock_ult, mock_model = _mock_ultralytics()
        result = _make_result(3, list(mock_ult.YOLO.return_value.names.values()))
        mock_model.predict.return_value = [result]
        det, _ = _detector_with_mock(mock_ult)
        dets = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(dets) == 3

    def test_detect_empty_image_returns_empty(self):
        mock_ult, mock_model = _mock_ultralytics()
        result = _make_result(0, [])
        mock_model.predict.return_value = [result]
        det, _ = _detector_with_mock(mock_ult)
        dets = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert dets == []

    def test_detect_sorted_by_confidence(self):
        mock_ult, mock_model = _mock_ultralytics()
        result = _make_result(4, list(mock_ult.YOLO.return_value.names.values()))
        mock_model.predict.return_value = [result]
        det, _ = _detector_with_mock(mock_ult)
        dets = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        confs = [d.confidence for d in dets]
        assert confs == sorted(confs, reverse=True)

    def test_detection_instances_are_Detection(self):
        mock_ult, mock_model = _mock_ultralytics()
        result = _make_result(2, list(mock_ult.YOLO.return_value.names.values()))
        mock_model.predict.return_value = [result]
        det, _ = _detector_with_mock(mock_ult)
        dets = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        for d in dets:
            assert isinstance(d, Detection)

    def test_detection_bbox_shape(self):
        mock_ult, mock_model = _mock_ultralytics()
        result = _make_result(2, list(mock_ult.YOLO.return_value.names.values()))
        mock_model.predict.return_value = [result]
        det, _ = _detector_with_mock(mock_ult)
        dets = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        for d in dets:
            assert d.bbox_xyxy.shape == (4,)
            assert d.bbox_xywh.shape == (4,)

    def test_detection_confidence_in_range(self):
        mock_ult, mock_model = _mock_ultralytics()
        result = _make_result(3, list(mock_ult.YOLO.return_value.names.values()))
        mock_model.predict.return_value = [result]
        det, _ = _detector_with_mock(mock_ult)
        dets = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        for d in dets:
            assert 0.0 <= d.confidence <= 1.0

    def test_detect_batch_empty_list(self):
        mock_ult, _ = _mock_ultralytics()
        det, _ = _detector_with_mock(mock_ult)
        assert det.detect_batch([]) == []

    def test_detect_batch_correct_count(self):
        mock_ult, mock_model = _mock_ultralytics()
        n_images = 3
        mock_model.predict.return_value = [
            _make_result(2, list(mock_ult.YOLO.return_value.names.values()))
            for _ in range(n_images)
        ]
        det, _ = _detector_with_mock(mock_ult)
        images = [np.zeros((480, 640, 3), dtype=np.uint8)] * n_images
        batched = det.detect_batch(images)
        assert len(batched) == n_images

    def test_detect_uses_conf_thresh(self):
        mock_ult, mock_model = _mock_ultralytics()
        mock_model.predict.return_value = [_make_result(0, [])]
        cfg = DetectorConfig(conf_thresh=0.6)
        det, _ = _detector_with_mock(mock_ult, cfg)
        det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        _, kw = mock_model.predict.call_args
        assert kw["conf"] == pytest.approx(0.6)

    def test_detect_uses_iou_thresh(self):
        mock_ult, mock_model = _mock_ultralytics()
        mock_model.predict.return_value = [_make_result(0, [])]
        cfg = DetectorConfig(iou_thresh=0.7)
        det, _ = _detector_with_mock(mock_ult, cfg)
        det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        _, kw = mock_model.predict.call_args
        assert kw["iou"] == pytest.approx(0.7)

    def test_detect_verbose_false(self):
        mock_ult, mock_model = _mock_ultralytics()
        mock_model.predict.return_value = [_make_result(0, [])]
        det, _ = _detector_with_mock(mock_ult)
        det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        _, kw = mock_model.predict.call_args
        assert kw["verbose"] is False
