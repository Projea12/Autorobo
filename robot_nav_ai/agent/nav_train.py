"""
agent/nav_train.py — PPO training entry-point for the NavigationEnv.

Usage
─────
    python -m agent.nav_train                       # fresh run
    python -m agent.nav_train --resume latest       # resume from last checkpoint
    python -m agent.nav_train --resume step_0100000 # resume specific step

Training curriculum
───────────────────
  Stage 0  (0 → stage0_steps): goal placed 0.5–1.5 m in front (easy)
  Stage 1  (stage0_steps → end): goal placed 1.0–4.0 m anywhere (full)

Checkpoints
───────────
  Saved every save_interval environment steps.
  Uses the CheckpointManager from agent/checkpoint.py (rolling 5 + best slot).

Logging
───────
  Progress printed to stdout every log_interval steps.
  TensorBoard SummaryWriter written to runs/<run_name>/ if available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.navigation_env import NavigationEnv
from env.episode_reset import GoalConfig
from env.nav_reward import RewardConfig
from agent.ppo import PPOConfig, make_ppo_agent


# ── training configuration ────────────────────────────────────────────────────

@dataclass
class NavTrainConfig:
    """All knobs for a navigation training run."""
    # --- budget ---
    total_steps:     int   = 5_000_000
    stage0_steps:    int   = 500_000    # easy-goal curriculum stage

    # --- env ---
    max_episode_steps: int  = 300
    n_substeps:        int  = 4
    seed:              int  = 0

    # --- PPO ---
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # --- logging / checkpoints ---
    run_name:      str   = "nav_ppo"
    log_dir:       str   = "runs"
    ckpt_dir:      str   = "checkpoints/nav"
    log_interval:  int   = 10_000    # print every N env steps
    save_interval: int   = 100_000   # checkpoint every N env steps
    keep_last:     int   = 5


# ── reward config for each curriculum stage ───────────────────────────────────

def _reward_cfg_stage0() -> RewardConfig:
    return RewardConfig(
        approach    = 2.0,
        goal        = 15.0,    # larger bonus — close goals are achievable fast
        collision   = 5.0,
        obstacle    = 0.3,
        explore     = 0.0,     # no exploration bonus in stage 0 (goal is close)
        uncertainty = 0.0,
        time_step   = 0.005,
        goal_radius = 0.30,
        collision_r = 0.12,
        danger_r    = 0.25,
    )


def _reward_cfg_stage1() -> RewardConfig:
    return RewardConfig()      # library defaults


# ── goal config for each curriculum stage ────────────────────────────────────

def _goal_cfg_stage0() -> GoalConfig:
    return GoalConfig(mode="relative", fwd_range=(0.5, 1.5), lat_range=(-0.3, 0.3))


def _goal_cfg_stage1() -> GoalConfig:
    return GoalConfig(mode="random")


# ── checkpoint helpers ────────────────────────────────────────────────────────

class _SimpleCheckpointer:
    """Minimal rolling checkpointer (no external dependency on CheckpointManager)."""

    def __init__(self, ckpt_dir: str, keep_last: int = 5) -> None:
        self.dir       = Path(ckpt_dir)
        self.keep_last = keep_last
        self.dir.mkdir(parents=True, exist_ok=True)
        self._history: list[Path] = []
        self._best_reward: float  = -float("inf")

    def save(self, agent, global_step: int, mean_reward: float) -> None:
        tag  = f"step_{global_step:010d}"
        path = self.dir / f"{tag}.pt"
        torch.save({
            "agent":       agent.state_dict(),
            "global_step": global_step,
            "mean_reward": mean_reward,
        }, path)
        self._history.append(path)

        # rolling eviction
        while len(self._history) > self.keep_last:
            old = self._history.pop(0)
            if old.exists() and "best" not in old.name:
                old.unlink(missing_ok=True)

        # best slot
        if mean_reward > self._best_reward:
            self._best_reward = mean_reward
            best_path = self.dir / "best.pt"
            torch.save({
                "agent":       agent.state_dict(),
                "global_step": global_step,
                "mean_reward": mean_reward,
            }, best_path)

        # metadata
        meta = {
            "global_step":   global_step,
            "mean_reward":   mean_reward,
            "best_reward":   self._best_reward,
            "latest_ckpt":   str(path),
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def latest_path(self) -> Optional[Path]:
        candidates = sorted(self.dir.glob("step_*.pt"))
        return candidates[-1] if candidates else None

    def best_path(self) -> Optional[Path]:
        p = self.dir / "best.pt"
        return p if p.exists() else None


# ── logging ───────────────────────────────────────────────────────────────────

def _try_tensorboard(log_dir: str, run_name: str):
    """Return a SummaryWriter or None if TensorBoard is unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    except ImportError:
        return None


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg: NavTrainConfig, resume: Optional[str] = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[nav_train] device={device}  total_steps={cfg.total_steps:,}")

    # --- build initial env (stage 0) ---
    env = NavigationEnv(
        max_steps   = cfg.max_episode_steps,
        n_substeps  = cfg.n_substeps,
        reward_cfg  = _reward_cfg_stage0(),
        goal_cfg    = _goal_cfg_stage0(),
        seed        = cfg.seed,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # --- PPO agent ---
    agent = make_ppo_agent(obs_dim, act_dim, cfg.ppo, device=device)

    # --- checkpointer & tensorboard ---
    ckpt  = _SimpleCheckpointer(cfg.ckpt_dir, keep_last=cfg.keep_last)
    writer = _try_tensorboard(cfg.log_dir, cfg.run_name)

    global_step  = 0
    episode_step = 0
    ep_rewards: list[float] = []
    ep_lengths: list[int]   = []
    ep_successes: list[bool] = []

    # --- resume ---
    if resume is not None:
        if resume == "latest":
            load_path = ckpt.latest_path()
        elif resume == "best":
            load_path = ckpt.best_path()
        else:
            load_path = Path(cfg.ckpt_dir) / f"{resume}.pt"
        if load_path and load_path.exists():
            sd = torch.load(load_path, map_location=device)
            agent.load_state_dict(sd["agent"])
            global_step = sd.get("global_step", 0)
            print(f"[nav_train] resumed from {load_path} (step {global_step:,})")
        else:
            print(f"[nav_train] WARNING: checkpoint not found: {resume}")

    obs_np, _ = env.reset(seed=cfg.seed)
    obs_t     = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    ep_reward = 0.0
    ep_len    = 0
    done      = False
    stage     = 0

    t_start = time.time()

    while global_step < cfg.total_steps:
        # ── curriculum switch ──
        if stage == 0 and global_step >= cfg.stage0_steps:
            stage = 1
            env.close()
            env = NavigationEnv(
                max_steps  = cfg.max_episode_steps,
                n_substeps = cfg.n_substeps,
                reward_cfg = _reward_cfg_stage1(),
                goal_cfg   = _goal_cfg_stage1(),
                seed       = cfg.seed + 1,
            )
            obs_np, _ = env.reset()
            obs_t     = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            ep_reward = 0.0
            ep_len    = 0
            done      = False
            # rebuild buffer in agent with same dims (same obs/act dim)
            from agent.ppo import RolloutBuffer
            agent.rollout_buffer = RolloutBuffer(
                n_steps    = cfg.ppo.n_steps,
                obs_dim    = obs_dim,
                act_dim    = act_dim,
                gamma      = cfg.ppo.gamma,
                gae_lambda = cfg.ppo.gae_lambda,
                device     = device,
            )
            print(f"[nav_train] stage 1 active at step {global_step:,}")

        # ── collect one rollout ──
        buf = agent.rollout_buffer
        buf.reset()

        for _ in range(cfg.ppo.n_steps):
            with torch.no_grad():
                act, lp, val = agent.net.act(obs_t)
            act_np = act.cpu().numpy()
            obs_np2, rew, term, trunc, info = env.step(act_np)
            done = term or trunc

            buf.add(obs_t, act, lp, val, float(rew), done)

            obs_t     = torch.as_tensor(obs_np2, dtype=torch.float32, device=device)
            ep_reward += float(rew)
            ep_len    += 1
            global_step += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                ep_successes.append(bool(info.get("success", False)))
                ep_reward = 0.0
                ep_len    = 0
                obs_np, _ = env.reset()
                obs_t     = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                done      = False

            if global_step >= cfg.total_steps:
                break

        # ── finish rollout ──
        with torch.no_grad():
            _, _, last_val = agent.net.act(obs_t)
        buf.finish_rollout(last_val, done)

        # ── PPO update ──
        metrics = agent.update()

        # ── logging ──
        if ep_rewards and global_step % cfg.log_interval < cfg.ppo.n_steps:
            mean_rew = float(np.mean(ep_rewards[-20:]))
            mean_len = float(np.mean(ep_lengths[-20:]))
            succ_rate = float(np.mean(ep_successes[-20:])) if ep_successes else 0.0
            fps  = global_step / max(time.time() - t_start, 1)
            print(
                f"step={global_step:>9,}  stage={stage}"
                f"  rew={mean_rew:+.2f}  len={mean_len:.0f}"
                f"  succ={succ_rate:.2%}  fps={fps:.0f}"
                f"  pg={metrics['loss/policy']:.4f}"
                f"  vf={metrics['loss/value']:.4f}"
            )
            if writer:
                writer.add_scalar("train/mean_reward",   mean_rew,   global_step)
                writer.add_scalar("train/mean_ep_len",   mean_len,   global_step)
                writer.add_scalar("train/success_rate",  succ_rate,  global_step)
                writer.add_scalar("train/fps",           fps,        global_step)
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)

        # ── checkpoint ──
        if global_step % cfg.save_interval < cfg.ppo.n_steps:
            mean_rew = float(np.mean(ep_rewards[-20:])) if ep_rewards else 0.0
            ckpt.save(agent, global_step, mean_rew)
            print(f"[nav_train] checkpoint saved at step {global_step:,}"
                  f"  mean_rew={mean_rew:.2f}")

    env.close()
    if writer:
        writer.close()
    print(f"[nav_train] training complete at step {global_step:,}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO navigation policy")
    p.add_argument("--resume",          type=str, default=None,
                   help="latest | best | step_XXXXXXXXXX")
    p.add_argument("--total-steps",     type=int, default=NavTrainConfig.total_steps)
    p.add_argument("--stage0-steps",    type=int, default=NavTrainConfig.stage0_steps)
    p.add_argument("--seed",            type=int, default=NavTrainConfig.seed)
    p.add_argument("--run-name",        type=str, default=NavTrainConfig.run_name)
    p.add_argument("--ckpt-dir",        type=str, default=NavTrainConfig.ckpt_dir)
    p.add_argument("--lr",              type=float, default=PPOConfig.lr)
    p.add_argument("--n-steps",         type=int, default=PPOConfig.n_steps)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ppo_cfg = PPOConfig(lr=args.lr, n_steps=args.n_steps)
    cfg = NavTrainConfig(
        total_steps  = args.total_steps,
        stage0_steps = args.stage0_steps,
        seed         = args.seed,
        run_name     = args.run_name,
        ckpt_dir     = args.ckpt_dir,
        ppo          = ppo_cfg,
    )
    train(cfg, resume=args.resume)
