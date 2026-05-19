"""
tests/test_nav_train.py — Unit tests for nav_train components (no full training run).

We test the helper objects (config, checkpointer, reward/goal cfg factories)
without running the actual training loop (too slow for CI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agent.ppo import ActorCritic, PPOConfig, PPOAgent, make_ppo_agent
from agent.nav_train import (
    NavTrainConfig,
    _reward_cfg_stage0,
    _reward_cfg_stage1,
    _goal_cfg_stage0,
    _goal_cfg_stage1,
    _SimpleCheckpointer,
)
from env.nav_reward import RewardConfig
from env.episode_reset import GoalConfig


# ── NavTrainConfig ────────────────────────────────────────────────────────────

class TestNavTrainConfig:
    def test_defaults(self):
        cfg = NavTrainConfig()
        assert cfg.total_steps    > 0
        assert cfg.stage0_steps   > 0
        assert cfg.stage0_steps   < cfg.total_steps
        assert cfg.max_episode_steps > 0
        assert cfg.n_substeps     >= 1

    def test_ppo_field_is_ppo_config(self):
        cfg = NavTrainConfig()
        assert isinstance(cfg.ppo, PPOConfig)

    def test_custom_total_steps(self):
        cfg = NavTrainConfig(total_steps=100_000)
        assert cfg.total_steps == 100_000

    def test_custom_ppo_lr(self):
        ppo = PPOConfig(lr=1e-3)
        cfg = NavTrainConfig(ppo=ppo)
        assert cfg.ppo.lr == pytest.approx(1e-3)

    def test_run_name_is_string(self):
        assert isinstance(NavTrainConfig().run_name, str)

    def test_ckpt_dir_is_string(self):
        assert isinstance(NavTrainConfig().ckpt_dir, str)


# ── curriculum reward / goal configs ─────────────────────────────────────────

class TestCurriculumConfigs:
    def test_stage0_reward_is_reward_config(self):
        assert isinstance(_reward_cfg_stage0(), RewardConfig)

    def test_stage1_reward_is_reward_config(self):
        assert isinstance(_reward_cfg_stage1(), RewardConfig)

    def test_stage0_goal_is_goal_config(self):
        assert isinstance(_goal_cfg_stage0(), GoalConfig)

    def test_stage1_goal_is_goal_config(self):
        assert isinstance(_goal_cfg_stage1(), GoalConfig)

    def test_stage0_goal_mode_relative(self):
        assert _goal_cfg_stage0().mode == "relative"

    def test_stage1_goal_mode_random(self):
        assert _goal_cfg_stage1().mode == "random"

    def test_stage0_reward_no_exploration(self):
        # Stage 0 should have zero exploration bonus (goals are close)
        assert _reward_cfg_stage0().explore == pytest.approx(0.0)

    def test_stage0_goal_bonus_larger(self):
        # Stage 0 goal bonus >= stage 1 (easier goals need bigger incentive)
        assert _reward_cfg_stage0().goal >= _reward_cfg_stage1().goal


# ── _SimpleCheckpointer ───────────────────────────────────────────────────────

class TestSimpleCheckpointer:
    def _agent(self) -> PPOAgent:
        return make_ppo_agent(obs_dim=10, act_dim=2)

    def test_creates_dir(self, tmp_path):
        ckpt = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        assert (tmp_path / "ckpts").exists()

    def test_save_creates_file(self, tmp_path):
        ckpt  = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        agent = self._agent()
        ckpt.save(agent, global_step=1000, mean_reward=-5.0)
        files = list((tmp_path / "ckpts").glob("step_*.pt"))
        assert len(files) == 1

    def test_save_creates_meta_json(self, tmp_path):
        ckpt  = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        agent = self._agent()
        ckpt.save(agent, global_step=500, mean_reward=-2.0)
        assert (tmp_path / "ckpts" / "meta.json").exists()

    def test_meta_json_content(self, tmp_path):
        ckpt  = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        agent = self._agent()
        ckpt.save(agent, global_step=200, mean_reward=-3.5)
        meta = json.loads((tmp_path / "ckpts" / "meta.json").read_text())
        assert meta["global_step"] == 200
        assert meta["mean_reward"] == pytest.approx(-3.5)

    def test_best_pt_created_on_first_save(self, tmp_path):
        ckpt  = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        ckpt.save(self._agent(), global_step=100, mean_reward=-10.0)
        assert (tmp_path / "ckpts" / "best.pt").exists()

    def test_best_pt_updated_on_improvement(self, tmp_path):
        ckpt  = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        ckpt.save(self._agent(), global_step=100, mean_reward=-10.0)
        mtime1 = (tmp_path / "ckpts" / "best.pt").stat().st_mtime
        ckpt.save(self._agent(), global_step=200, mean_reward=-1.0)
        mtime2 = (tmp_path / "ckpts" / "best.pt").stat().st_mtime
        assert mtime2 >= mtime1

    def test_best_pt_not_updated_on_regression(self, tmp_path):
        ckpt  = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        ckpt.save(self._agent(), global_step=100, mean_reward=-1.0)
        mtime1 = (tmp_path / "ckpts" / "best.pt").stat().st_mtime
        ckpt.save(self._agent(), global_step=200, mean_reward=-10.0)
        mtime2 = (tmp_path / "ckpts" / "best.pt").stat().st_mtime
        # best.pt not rewritten on regression
        assert mtime2 == mtime1

    def test_rolling_eviction(self, tmp_path):
        ckpt = _SimpleCheckpointer(str(tmp_path / "ckpts"), keep_last=3)
        for i in range(6):
            ckpt.save(self._agent(), global_step=(i + 1) * 100, mean_reward=-float(i))
        files = list((tmp_path / "ckpts").glob("step_*.pt"))
        assert len(files) <= 3

    def test_latest_path_none_when_empty(self, tmp_path):
        ckpt = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        assert ckpt.latest_path() is None

    def test_latest_path_returns_last(self, tmp_path):
        ckpt  = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        ckpt.save(self._agent(), global_step=100, mean_reward=-5.0)
        ckpt.save(self._agent(), global_step=200, mean_reward=-4.0)
        p = ckpt.latest_path()
        assert p is not None
        assert "0000000200" in p.name

    def test_best_path_none_when_empty(self, tmp_path):
        ckpt = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        assert ckpt.best_path() is None

    def test_best_path_returns_file(self, tmp_path):
        ckpt  = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        ckpt.save(self._agent(), global_step=100, mean_reward=-5.0)
        p = ckpt.best_path()
        assert p is not None and p.exists()

    def test_checkpoint_loadable(self, tmp_path):
        ckpt   = _SimpleCheckpointer(str(tmp_path / "ckpts"))
        agent1 = make_ppo_agent(obs_dim=10, act_dim=2)
        ckpt.save(agent1, global_step=100, mean_reward=-5.0)

        path = ckpt.latest_path()
        sd   = torch.load(path, map_location="cpu")
        assert "agent" in sd
        assert "global_step" in sd

        agent2 = make_ppo_agent(obs_dim=10, act_dim=2)
        agent2.load_state_dict(sd["agent"])   # should not raise


# ── PPOConfig integration with NavTrainConfig ─────────────────────────────────

class TestPPOConfigIntegration:
    def test_ppo_config_n_steps(self):
        cfg = NavTrainConfig(ppo=PPOConfig(n_steps=512))
        assert cfg.ppo.n_steps == 512

    def test_ppo_config_lr_affects_optimizer(self):
        lr  = 5e-4
        cfg = PPOConfig(lr=lr)
        net = ActorCritic(obs_dim=8, act_dim=2)
        from agent.ppo import PPOAgent
        agent = PPOAgent(net, cfg)
        # Retrieve LR from optimizer param group
        actual_lr = agent.optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(lr)
