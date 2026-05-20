"""
tests/test_yolo_trainer.py — Unit tests for YOLOTrainer, YOLOTrainConfig, TrainResult.

ultralytics is mocked; no actual training runs are executed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import agent.yolo_trainer as _trainer_mod
from agent.yolo_trainer import YOLOTrainConfig, YOLOTrainer, TrainResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_ultralytics(tmp_path=None):
    """Return a mock ultralytics module with sensible training behaviour."""
    results_dict = {
        "metrics/mAP50(B)":    0.72,
        "metrics/mAP50-95(B)": 0.48,
        "metrics/precision(B)": 0.80,
        "metrics/recall(B)":    0.65,
    }
    mock_results = MagicMock()
    mock_results.results_dict = results_dict

    val_metrics = MagicMock()
    val_metrics.box.map50 = 0.72
    val_metrics.box.map   = 0.48
    val_metrics.box.mp    = 0.80
    val_metrics.box.mr    = 0.65

    mock_model = MagicMock()
    mock_model.train.return_value  = mock_results
    mock_model.val.return_value    = val_metrics
    mock_model.export.return_value = str(tmp_path / "best.onnx") if tmp_path else "best.onnx"

    mock_ult = MagicMock()
    mock_ult.YOLO.return_value = mock_model
    return mock_ult, mock_model, mock_results


def _trainer(tmp_path, **kw) -> YOLOTrainer:
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text("nc: 21\nnames: []")
    cfg = YOLOTrainConfig(
        data_yaml   = str(data_yaml),
        project     = str(tmp_path / "runs"),
        run_name    = "test_run",
        epochs      = kw.pop("epochs", 2),
        **kw,
    )
    mock_ult, _, _ = _mock_ultralytics(tmp_path)
    with patch.object(_trainer_mod, "_ultralytics", mock_ult):
        trainer = YOLOTrainer(cfg)
    trainer._mock_ult = mock_ult   # keep reference for assertions
    return trainer


# ── YOLOTrainConfig ───────────────────────────────────────────────────────────

class TestYOLOTrainConfig:
    def test_defaults(self):
        cfg = YOLOTrainConfig()
        assert cfg.base_weights  == "yolov8n.pt"
        assert cfg.epochs        == 100
        assert cfg.imgsz         == 640
        assert cfg.batch         == 16
        assert cfg.device        == ""
        assert cfg.patience      == 20
        assert cfg.lr0           == pytest.approx(0.01)
        assert cfg.lrf           == pytest.approx(0.01)
        assert cfg.mosaic        == pytest.approx(1.0)
        assert cfg.auto_generate is False

    def test_custom_values(self):
        cfg = YOLOTrainConfig(epochs=50, batch=32, device="cuda")
        assert cfg.epochs == 50
        assert cfg.batch  == 32
        assert cfg.device == "cuda"

    def test_run_name(self):
        cfg = YOLOTrainConfig(run_name="my_run")
        assert cfg.run_name == "my_run"


# ── TrainResult ───────────────────────────────────────────────────────────────

class TestTrainResult:
    def _make(self, tmp_path):
        return TrainResult(
            best_model_path = tmp_path / "best.pt",
            last_model_path = tmp_path / "last.pt",
            results_dir     = tmp_path,
            map50           = 0.72,
            map50_95        = 0.48,
            precision       = 0.80,
            recall          = 0.65,
            elapsed_s       = 120.5,
        )

    def test_str_contains_map50(self, tmp_path):
        r = self._make(tmp_path)
        assert "0.7200" in str(r)

    def test_str_contains_elapsed(self, tmp_path):
        r = self._make(tmp_path)
        assert "120" in str(r)

    def test_map50_value(self, tmp_path):
        r = self._make(tmp_path)
        assert r.map50 == pytest.approx(0.72)

    def test_map50_95_value(self, tmp_path):
        r = self._make(tmp_path)
        assert r.map50_95 == pytest.approx(0.48)

    def test_precision_value(self, tmp_path):
        r = self._make(tmp_path)
        assert r.precision == pytest.approx(0.80)

    def test_recall_value(self, tmp_path):
        r = self._make(tmp_path)
        assert r.recall == pytest.approx(0.65)


# ── YOLOTrainer — construction ────────────────────────────────────────────────

class TestTrainerConstruction:
    def test_raises_without_ultralytics(self, tmp_path):
        cfg = YOLOTrainConfig(data_yaml=str(tmp_path / "d.yaml"))
        with patch.object(_trainer_mod, "_ultralytics", None):
            with pytest.raises(ImportError, match="ultralytics"):
                YOLOTrainer(cfg)

    def test_construction_succeeds_with_mock(self, tmp_path):
        t = _trainer(tmp_path)
        assert isinstance(t, YOLOTrainer)

    def test_repr(self, tmp_path):
        t = _trainer(tmp_path)
        assert "YOLOTrainer" in repr(t)


# ── YOLOTrainer.train() ───────────────────────────────────────────────────────

class TestTrainerTrain:
    def test_returns_train_result(self, tmp_path):
        t = _trainer(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            result = t.train()
        assert isinstance(result, TrainResult)

    def test_map50_extracted(self, tmp_path):
        t = _trainer(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            result = t.train()
        assert result.map50 == pytest.approx(0.72)

    def test_map50_95_extracted(self, tmp_path):
        t = _trainer(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            result = t.train()
        assert result.map50_95 == pytest.approx(0.48)

    def test_precision_extracted(self, tmp_path):
        t = _trainer(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            result = t.train()
        assert result.precision == pytest.approx(0.80)

    def test_elapsed_positive(self, tmp_path):
        t = _trainer(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            result = t.train()
        assert result.elapsed_s >= 0.0

    def test_train_called_once(self, tmp_path):
        t = _trainer(tmp_path)
        mock_model = t._mock_ult.YOLO.return_value
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            t.train()
        mock_model.train.assert_called_once()

    def test_train_kwargs_data_yaml(self, tmp_path):
        t = _trainer(tmp_path)
        mock_model = t._mock_ult.YOLO.return_value
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            t.train()
        _, kw = mock_model.train.call_args
        assert kw["data"] == t.cfg.data_yaml

    def test_train_kwargs_epochs(self, tmp_path):
        t = _trainer(tmp_path, epochs=3)
        mock_model = t._mock_ult.YOLO.return_value
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            t.train()
        _, kw = mock_model.train.call_args
        assert kw["epochs"] == 3

    def test_train_kwargs_imgsz(self, tmp_path):
        t = _trainer(tmp_path)
        mock_model = t._mock_ult.YOLO.return_value
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            t.train()
        _, kw = mock_model.train.call_args
        assert kw["imgsz"] == t.cfg.imgsz

    def test_train_kwargs_project(self, tmp_path):
        t = _trainer(tmp_path)
        mock_model = t._mock_ult.YOLO.return_value
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            t.train()
        _, kw = mock_model.train.call_args
        assert kw["project"] == t.cfg.project

    def test_missing_data_yaml_raises_without_auto_generate(self, tmp_path):
        cfg = YOLOTrainConfig(
            data_yaml     = str(tmp_path / "nonexistent.yaml"),
            auto_generate = False,
        )
        mock_ult, _, _ = _mock_ultralytics(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", mock_ult):
            t = YOLOTrainer(cfg)
        with patch.object(_trainer_mod, "_ultralytics", mock_ult):
            with pytest.raises(FileNotFoundError, match="auto_generate"):
                t.train()


# ── YOLOTrainer.validate() ────────────────────────────────────────────────────

class TestTrainerValidate:
    def test_returns_dict(self, tmp_path):
        t = _trainer(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            metrics = t.validate(weights_path=tmp_path / "best.pt")
        assert isinstance(metrics, dict)

    def test_dict_contains_map50(self, tmp_path):
        t = _trainer(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            m = t.validate(weights_path=tmp_path / "best.pt")
        assert "map50" in m

    def test_dict_contains_precision(self, tmp_path):
        t = _trainer(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            m = t.validate(weights_path=tmp_path / "best.pt")
        assert "precision" in m

    def test_map50_value(self, tmp_path):
        t = _trainer(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            m = t.validate(weights_path=tmp_path / "best.pt")
        assert m["map50"] == pytest.approx(0.72)


# ── YOLOTrainer.export() ──────────────────────────────────────────────────────

class TestTrainerExport:
    def test_returns_path(self, tmp_path):
        t = _trainer(tmp_path)
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            out = t.export(weights_path=tmp_path / "best.pt", fmt="onnx")
        assert isinstance(out, Path)

    def test_export_called_with_format(self, tmp_path):
        t = _trainer(tmp_path)
        mock_model = t._mock_ult.YOLO.return_value
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            t.export(weights_path=tmp_path / "best.pt", fmt="onnx")
        _, kw = mock_model.export.call_args
        assert kw["format"] == "onnx"

    def test_export_uses_imgsz(self, tmp_path):
        t = _trainer(tmp_path)
        mock_model = t._mock_ult.YOLO.return_value
        with patch.object(_trainer_mod, "_ultralytics", t._mock_ult):
            t.export(weights_path=tmp_path / "best.pt")
        _, kw = mock_model.export.call_args
        assert kw["imgsz"] == t.cfg.imgsz
