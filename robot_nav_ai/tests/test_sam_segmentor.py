"""
tests/test_sam_segmentor.py — Unit tests for SAMSegmentor and SAMConfig.

segment_anything is mocked throughout; no actual SAM inference runs.
Tests cover: stub mode, construction failure, happy-path inference,
score thresholding, set_image/predict call counts, and return shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import perception.sam_segmentor as _sam_mod
from perception.sam_segmentor import SAMConfig, SAMSegmentor
from perception.detector import Detection


# ── helpers ───────────────────────────────────────────────────────────────────

_H, _W = 60, 80


def _make_masks(best_score: float = 0.95):
    """Three candidate masks with descending scores; best is index 0."""
    masks         = np.zeros((3, _H, _W), dtype=bool)
    masks[0, 10:40, 10:50] = True          # best mask
    masks[1, 5:35, 5:45]   = True
    masks[2, 0:20, 0:30]   = True
    scores = np.array([best_score, 0.70, 0.50])
    return masks, scores


def _mock_sam_lib(best_score: float = 0.95):
    """Return a mock segment_anything module + predictor."""
    masks, scores       = _make_masks(best_score)
    mock_predictor      = MagicMock()
    mock_predictor.predict.return_value = (masks, scores, None)

    mock_sam_model      = MagicMock()
    mock_lib            = MagicMock()
    mock_lib.sam_model_registry = {
        "vit_b": MagicMock(return_value=mock_sam_model),
    }
    mock_lib.SamPredictor.return_value = mock_predictor
    return mock_lib, mock_predictor, masks, scores


def _segmentor(cfg: SAMConfig = None, best_score: float = 0.95) -> SAMSegmentor:
    """Build a SAMSegmentor with mocked SAM library."""
    cfg                 = cfg or SAMConfig()
    mock_lib, mock_pred, masks, scores = _mock_sam_lib(best_score)
    with patch.object(_sam_mod, "_sam_lib", mock_lib):
        seg             = SAMSegmentor(cfg)
    seg._mock_pred  = mock_pred
    seg._mock_masks = masks
    seg._mock_scores = scores
    return seg


def _make_det(x1=10, y1=10, x2=70, y2=50) -> Detection:
    return Detection(
        class_id   = 0,
        class_name = "mug",
        confidence = 0.9,
        bbox_xyxy  = np.array([x1, y1, x2, y2], dtype=np.float32),
        bbox_xywh  = np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1],
                               dtype=np.float32),
    )


# ── SAMConfig ─────────────────────────────────────────────────────────────────

class TestSAMConfig:
    def test_defaults(self):
        cfg = SAMConfig()
        assert cfg.model_type   == "vit_b"
        assert cfg.weights_path == "sam_vit_b.pth"
        assert cfg.device       == ""
        assert cfg.score_thresh == pytest.approx(0.5)

    def test_frozen(self):
        cfg = SAMConfig()
        with pytest.raises(Exception):
            cfg.model_type = "vit_h"

    def test_custom(self):
        cfg = SAMConfig(model_type="vit_h", device="cuda", score_thresh=0.7)
        assert cfg.model_type   == "vit_h"
        assert cfg.score_thresh == pytest.approx(0.7)


# ── stub mode (segment_anything unavailable) ──────────────────────────────────

class TestStubMode:
    def test_construction_succeeds_without_sam(self):
        with patch.object(_sam_mod, "_sam_lib", None):
            seg = SAMSegmentor()
        assert seg is not None

    def test_is_loaded_false_without_sam(self):
        with patch.object(_sam_mod, "_sam_lib", None):
            seg = SAMSegmentor()
        assert seg.is_loaded is False

    def test_repr_contains_stub(self):
        with patch.object(_sam_mod, "_sam_lib", None):
            seg = SAMSegmentor()
        assert "stub" in repr(seg)

    def test_segment_from_bbox_raises_import_error(self):
        with patch.object(_sam_mod, "_sam_lib", None):
            seg = SAMSegmentor()
        img  = np.zeros((_H, _W, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 60, 50], dtype=np.float32)
        with pytest.raises(ImportError, match="segment_anything"):
            seg.segment_from_bbox(img, bbox)

    def test_segment_detections_raises_import_error(self):
        with patch.object(_sam_mod, "_sam_lib", None):
            seg = SAMSegmentor()
        img = np.zeros((_H, _W, 3), dtype=np.uint8)
        with pytest.raises(ImportError):
            seg.segment_detections(img, [_make_det()])


# ── construction error handling ───────────────────────────────────────────────

class TestConstructionError:
    def test_failed_load_yields_stub(self):
        mock_lib = MagicMock()
        mock_lib.sam_model_registry = {
            "vit_b": MagicMock(side_effect=RuntimeError("bad checkpoint")),
        }
        with patch.object(_sam_mod, "_sam_lib", mock_lib):
            seg = SAMSegmentor()
        assert seg.is_loaded is False

    def test_failed_load_repr_contains_stub(self):
        mock_lib = MagicMock()
        mock_lib.sam_model_registry = {
            "vit_b": MagicMock(side_effect=RuntimeError("err")),
        }
        with patch.object(_sam_mod, "_sam_lib", mock_lib):
            seg = SAMSegmentor()
        assert "stub" in repr(seg)


# ── happy path (mocked SAM) ───────────────────────────────────────────────────

class TestHappyPath:
    def test_is_loaded_true_with_mock(self):
        seg = _segmentor()
        assert seg.is_loaded is True

    def test_repr_contains_loaded(self):
        assert "loaded" in repr(_segmentor())

    def test_repr_contains_model_type(self):
        assert "vit_b" in repr(_segmentor())


# ── segment_from_bbox ─────────────────────────────────────────────────────────

class TestSegmentFromBbox:
    def _call(self, seg, score_thresh=0.5):
        img  = np.zeros((_H, _W, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 60, 50], dtype=np.float32)
        return seg.segment_from_bbox(img, bbox)

    def test_returns_ndarray(self):
        assert isinstance(self._call(_segmentor()), np.ndarray)

    def test_shape(self):
        assert self._call(_segmentor()).shape == (_H, _W)

    def test_dtype_bool(self):
        assert self._call(_segmentor()).dtype == bool

    def test_returns_best_mask(self):
        seg    = _segmentor()
        result = self._call(seg)
        # Best mask (index 0) has True at [10:40, 10:50]
        assert bool(result[15, 15]) is True
        assert bool(result[0, 0])   is False

    def test_below_thresh_returns_none(self):
        # All scores ≤ 0.95, threshold set above that
        seg    = _segmentor(SAMConfig(score_thresh=0.99), best_score=0.95)
        result = self._call(seg)
        assert result is None

    def test_calls_set_image(self):
        seg = _segmentor()
        self._call(seg)
        seg._mock_pred.set_image.assert_called_once()

    def test_calls_predict_with_multimask(self):
        seg = _segmentor()
        self._call(seg)
        _, kw = seg._mock_pred.predict.call_args
        assert kw["multimask_output"] is True

    def test_box_passed_as_2d_array(self):
        seg = _segmentor()
        self._call(seg)
        _, kw = seg._mock_pred.predict.call_args
        assert kw["box"].ndim == 2
        assert kw["box"].shape[1] == 4


# ── segment_detections ────────────────────────────────────────────────────────

class TestSegmentDetections:
    def _run(self, seg, n_dets=1):
        img  = np.zeros((_H, _W, 3), dtype=np.uint8)
        dets = [_make_det() for _ in range(n_dets)]
        seg.segment_detections(img, dets)
        return dets

    def test_mask_filled_on_detection(self):
        seg  = _segmentor()
        dets = self._run(seg)
        assert dets[0].mask is not None

    def test_mask_shape(self):
        seg  = _segmentor()
        dets = self._run(seg)
        assert dets[0].mask.shape == (_H, _W)

    def test_mask_dtype_bool(self):
        seg  = _segmentor()
        dets = self._run(seg)
        assert dets[0].mask.dtype == bool

    def test_set_image_called_once_for_multiple_detections(self):
        seg = _segmentor()
        img = np.zeros((_H, _W, 3), dtype=np.uint8)
        seg.segment_detections(img, [_make_det(), _make_det()])
        seg._mock_pred.set_image.assert_called_once()

    def test_predict_called_once_per_detection(self):
        seg = _segmentor()
        img = np.zeros((_H, _W, 3), dtype=np.uint8)
        seg.segment_detections(img, [_make_det(), _make_det(), _make_det()])
        assert seg._mock_pred.predict.call_count == 3

    def test_returns_same_list_object(self):
        seg  = _segmentor()
        img  = np.zeros((_H, _W, 3), dtype=np.uint8)
        dets = [_make_det()]
        out  = seg.segment_detections(img, dets)
        assert out is dets

    def test_empty_list_returns_empty(self):
        seg = _segmentor()
        img = np.zeros((_H, _W, 3), dtype=np.uint8)
        assert seg.segment_detections(img, []) == []

    def test_empty_list_skips_set_image(self):
        seg = _segmentor()
        img = np.zeros((_H, _W, 3), dtype=np.uint8)
        seg.segment_detections(img, [])
        seg._mock_pred.set_image.assert_not_called()

    def test_below_thresh_mask_remains_none(self):
        seg  = _segmentor(SAMConfig(score_thresh=0.99), best_score=0.95)
        dets = self._run(seg)
        assert dets[0].mask is None

    def test_above_thresh_mask_set(self):
        seg  = _segmentor(SAMConfig(score_thresh=0.5), best_score=0.95)
        dets = self._run(seg)
        assert dets[0].mask is not None
