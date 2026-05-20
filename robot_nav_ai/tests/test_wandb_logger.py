"""
tests/test_wandb_logger.py — Unit tests for WandbLogger and build_hparams.

wandb is mocked throughout so these tests run without a network connection
and without wandb being installed.  The mock is applied at the module level
inside wandb_logger so every import path is covered.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agent.wandb_logger import WandbConfig, WandbLogger, build_hparams
from agent.nav_train import (
    NavTrainConfig,
    _reward_cfg_stage0,
    _reward_cfg_stage1,
)
from agent.ppo import PPOConfig


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_wandb(run_id: str = "abc123", url: str = "https://wandb.ai/test/run/abc123"):
    """Return a mock wandb module whose init() returns a mock run."""
    mock_run = MagicMock()
    mock_run.id  = run_id
    mock_run.url = url

    mock_wb = MagicMock()
    mock_wb.init.return_value = mock_run
    return mock_wb, mock_run


def _logger(mode: str = "online", **kw) -> WandbLogger:
    cfg = WandbConfig(mode=mode, **kw)
    return WandbLogger(cfg)


# ── WandbConfig ───────────────────────────────────────────────────────────────

class TestWandbConfig:
    def test_defaults(self):
        cfg = WandbConfig()
        assert cfg.project == "autorobo-nav"
        assert cfg.entity  is None
        assert cfg.group   is None
        assert cfg.tags    == ()
        assert cfg.mode    == "online"
        assert cfg.resume_run_id is None

    def test_custom_project(self):
        cfg = WandbConfig(project="my-project")
        assert cfg.project == "my-project"

    def test_tags_tuple(self):
        cfg = WandbConfig(tags=("ppo", "nav"))
        assert cfg.tags == ("ppo", "nav")

    def test_disabled_mode(self):
        cfg = WandbConfig(mode="disabled")
        assert cfg.mode == "disabled"


# ── WandbLogger — disabled mode ───────────────────────────────────────────────

class TestDisabledMode:
    def test_init_returns_none_when_disabled(self):
        lb = _logger(mode="disabled")
        result = lb.init("run", {})
        assert result is None

    def test_enabled_false_when_disabled(self):
        lb = _logger(mode="disabled")
        lb.init("run", {})
        assert lb.enabled is False

    def test_log_is_noop_when_disabled(self):
        lb = _logger(mode="disabled")
        lb.init("run", {})
        lb.log({"x": 1}, step=0)   # should not raise

    def test_finish_is_noop_when_disabled(self):
        lb = _logger(mode="disabled")
        lb.init("run", {})
        lb.finish()   # should not raise

    def test_run_id_none_when_disabled(self):
        lb = _logger(mode="disabled")
        lb.init("run", {})
        assert lb.run_id is None


# ── WandbLogger — wandb unavailable ───────────────────────────────────────────

class TestWandbUnavailable:
    def test_init_returns_none_when_import_fails(self):
        lb = _logger()
        with patch("agent.wandb_logger.wandb", None):
            result = lb.init("run", {})
        assert result is None

    def test_enabled_false_when_import_fails(self):
        lb = _logger()
        with patch("agent.wandb_logger.wandb", None):
            lb.init("run", {})
        assert lb.enabled is False


# ── WandbLogger — happy path ──────────────────────────────────────────────────

class TestWandbLoggerHappyPath:
    def test_init_calls_wandb_init(self):
        mock_wb, mock_run = _mock_wandb()
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("my_run", {"lr": 3e-4})
        mock_wb.init.assert_called_once()

    def test_init_passes_project(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger(project="test-project")
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
        _, call_kw = mock_wb.init.call_args
        assert call_kw["project"] == "test-project"

    def test_init_passes_name(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("cool_run", {})
        _, call_kw = mock_wb.init.call_args
        assert call_kw["name"] == "cool_run"

    def test_init_passes_entity(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger(entity="my-team")
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
        _, call_kw = mock_wb.init.call_args
        assert call_kw["entity"] == "my-team"

    def test_init_passes_tags(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger(tags=("ppo", "nav"))
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
        _, call_kw = mock_wb.init.call_args
        assert "ppo" in call_kw["tags"]
        assert "nav" in call_kw["tags"]

    def test_init_passes_group(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger(group="sweep-01")
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
        _, call_kw = mock_wb.init.call_args
        assert call_kw["group"] == "sweep-01"

    def test_init_passes_config(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger()
        hparams = {"lr": 3e-4, "gamma": 0.99}
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", hparams)
        _, call_kw = mock_wb.init.call_args
        assert call_kw["config"] == hparams

    def test_init_returns_run_id(self):
        mock_wb, _ = _mock_wandb(run_id="xyz789")
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            run_id = lb.init("run", {})
        assert run_id == "xyz789"

    def test_enabled_true_after_successful_init(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
        assert lb.enabled is True

    def test_run_id_property(self):
        mock_wb, _ = _mock_wandb(run_id="run99")
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
        assert lb.run_id == "run99"

    def test_run_url_property(self):
        mock_wb, _ = _mock_wandb(url="https://wandb.ai/u/p/r/run99")
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
        assert "run99" in lb.run_url

    def test_log_calls_run_log(self):
        mock_wb, mock_run = _mock_wandb()
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
            lb.log({"train/mean_reward": -2.5}, step=1000)
        mock_run.log.assert_called_once_with(
            {"train/mean_reward": -2.5}, step=1000
        )

    def test_log_multiple_metrics(self):
        mock_wb, mock_run = _mock_wandb()
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
            lb.log({"train/mean_reward": -1.0,
                    "train/success_rate": 0.3,
                    "loss/policy": 0.05}, step=500)
        args, kw = mock_run.log.call_args
        assert len(args[0]) == 3

    def test_log_before_init_is_noop(self):
        lb = _logger()
        lb.log({"x": 1}, step=0)   # no init — should not raise

    def test_finish_calls_run_finish(self):
        mock_wb, mock_run = _mock_wandb()
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
            lb.finish()
        mock_run.finish.assert_called_once()

    def test_finish_before_init_is_noop(self):
        lb = _logger()
        lb.finish()   # should not raise


# ── resume run ID ─────────────────────────────────────────────────────────────

class TestResumeRunId:
    def test_resume_run_id_passed_to_init(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {}, resume_run_id="prev123")
        _, call_kw = mock_wb.init.call_args
        assert call_kw["id"] == "prev123"
        assert call_kw["resume"] == "allow"

    def test_no_resume_id_means_none_id(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
        _, call_kw = mock_wb.init.call_args
        assert call_kw["id"] is None
        assert call_kw["resume"] is None

    def test_config_resume_run_id_used_as_fallback(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger(resume_run_id="cfg_run_id")
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
        _, call_kw = mock_wb.init.call_args
        assert call_kw["id"] == "cfg_run_id"

    def test_explicit_resume_overrides_config(self):
        mock_wb, _ = _mock_wandb()
        lb = _logger(resume_run_id="cfg_id")
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {}, resume_run_id="explicit_id")
        _, call_kw = mock_wb.init.call_args
        assert call_kw["id"] == "explicit_id"


# ── init exception handling ───────────────────────────────────────────────────

class TestInitException:
    def test_init_exception_returns_none(self):
        mock_wb = MagicMock()
        mock_wb.init.side_effect = RuntimeError("network error")
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            result = lb.init("run", {})
        assert result is None

    def test_init_exception_leaves_disabled(self):
        mock_wb = MagicMock()
        mock_wb.init.side_effect = Exception("boom")
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
        assert lb.enabled is False

    def test_log_after_failed_init_is_noop(self):
        mock_wb = MagicMock()
        mock_wb.init.side_effect = Exception("boom")
        lb = _logger()
        with patch("agent.wandb_logger.wandb", mock_wb):
            lb.init("run", {})
            lb.log({"x": 1}, step=0)   # should not raise


# ── build_hparams ─────────────────────────────────────────────────────────────

class TestBuildHparams:
    def _hparams(self, **kw):
        cfg = NavTrainConfig(**kw)
        return build_hparams(cfg, _reward_cfg_stage0(), _reward_cfg_stage1(),
                             obs_dim=46, act_dim=2)

    def test_returns_dict(self):
        assert isinstance(self._hparams(), dict)

    def test_contains_lr(self):
        h = self._hparams()
        assert "ppo/lr" in h

    def test_lr_value(self):
        cfg = NavTrainConfig(ppo=PPOConfig(lr=5e-4))
        h = build_hparams(cfg, _reward_cfg_stage0(), _reward_cfg_stage1(), 46, 2)
        assert h["ppo/lr"] == pytest.approx(5e-4)

    def test_contains_gamma(self):
        assert "ppo/gamma" in self._hparams()

    def test_contains_total_steps(self):
        h = self._hparams(total_steps=1_000_000)
        assert h["budget/total_steps"] == 1_000_000

    def test_contains_obs_dim(self):
        cfg = NavTrainConfig()
        h = build_hparams(cfg, _reward_cfg_stage0(), _reward_cfg_stage1(), 46, 2)
        assert h["env/obs_dim"] == 46

    def test_contains_act_dim(self):
        cfg = NavTrainConfig()
        h = build_hparams(cfg, _reward_cfg_stage0(), _reward_cfg_stage1(), 46, 2)
        assert h["env/act_dim"] == 2

    def test_contains_stage0_reward_fields(self):
        h = self._hparams()
        assert "reward_s0/goal" in h
        assert "reward_s0/approach" in h
        assert "reward_s0/collision" in h

    def test_contains_stage1_reward_fields(self):
        h = self._hparams()
        assert "reward_s1/goal" in h
        assert "reward_s1/explore" in h

    def test_stage0_reward_values_match(self):
        h    = self._hparams()
        rew0 = _reward_cfg_stage0()
        assert h["reward_s0/goal"]     == pytest.approx(rew0.goal)
        assert h["reward_s0/approach"] == pytest.approx(rew0.approach)

    def test_contains_seed(self):
        h = self._hparams(seed=42)
        assert h["env/seed"] == 42

    def test_contains_n_steps(self):
        cfg = NavTrainConfig(ppo=PPOConfig(n_steps=512))
        h = build_hparams(cfg, _reward_cfg_stage0(), _reward_cfg_stage1(), 46, 2)
        assert h["ppo/n_steps"] == 512

    def test_contains_stage0_steps(self):
        h = self._hparams(stage0_steps=250_000)
        assert h["budget/stage0_steps"] == 250_000

    def test_all_values_are_scalars_or_strings(self):
        """Every hparam value must be JSON-serialisable for wandb.config."""
        import json
        h = self._hparams()
        json.dumps(h)   # raises if any value is not serialisable


# ── NavTrainConfig W&B integration ───────────────────────────────────────────

class TestNavTrainConfigWandb:
    def test_wandb_field_exists(self):
        cfg = NavTrainConfig()
        assert isinstance(cfg.wandb, WandbConfig)

    def test_wandb_default_project(self):
        assert NavTrainConfig().wandb.project == "autorobo-nav"

    def test_custom_wandb_config(self):
        wb = WandbConfig(project="my-proj", mode="offline")
        cfg = NavTrainConfig(wandb=wb)
        assert cfg.wandb.project == "my-proj"
        assert cfg.wandb.mode == "offline"
