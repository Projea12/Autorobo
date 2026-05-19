"""
agent/nav_eval.py — Honest evaluation of the navigation policy.

Metrics
───────
  success_rate      : fraction of episodes where the agent reached the goal
  mean_steps        : average steps per episode (successes only)
  mean_steps_all    : average steps per episode (all episodes)
  mean_return       : average undiscounted episodic return
  failure_breakdown : {timeout, collision, retreating} fractions
  novel_layout_sr   : success rate on unseen goal configs (generalisation)

Usage
─────
    evaluator = NavigationEvaluator(agent=ppo_agent, device=device)
    report = evaluator.evaluate(n_episodes=100, seed=999)
    print(report)

    # novel-layout test
    novel = evaluator.evaluate_novel_layout(n_episodes=50, seed=888)
    print(novel)
"""

from __future__ import annotations

import sys
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
from agent.ppo import ActorCritic, PPOAgent


# ── episode result ────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Outcome of one evaluation episode."""
    success:    bool
    n_steps:    int
    total_return: float
    failure_mode: str   # "success" | "timeout" | "collision" | "retreating"

    def __repr__(self) -> str:
        return (f"EpisodeResult(success={self.success}, steps={self.n_steps}, "
                f"R={self.total_return:.2f}, mode={self.failure_mode!r})")


# ── evaluation report ─────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    """Aggregated evaluation metrics from a batch of episodes."""
    n_episodes:       int
    success_rate:     float
    mean_steps:       float       # mean steps of successful episodes only
    mean_steps_all:   float       # mean steps of all episodes
    mean_return:      float
    failure_timeout:  float       # fraction of episodes
    failure_collision: float
    failure_retreating: float
    label:            str = ""    # e.g. "train_layout" / "novel_layout"
    episodes:         list[EpisodeResult] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        tag = f"[{self.label}] " if self.label else ""
        lines = [
            f"{tag}EvalReport over {self.n_episodes} episodes",
            f"  success_rate      : {self.success_rate:.1%}",
            f"  mean_steps (succ) : {self.mean_steps:.1f}",
            f"  mean_steps (all)  : {self.mean_steps_all:.1f}",
            f"  mean_return       : {self.mean_return:.3f}",
            f"  failure breakdown :",
            f"    timeout         : {self.failure_timeout:.1%}",
            f"    collision       : {self.failure_collision:.1%}",
            f"    retreating      : {self.failure_retreating:.1%}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "label":              self.label,
            "n_episodes":         self.n_episodes,
            "success_rate":       self.success_rate,
            "mean_steps":         self.mean_steps,
            "mean_steps_all":     self.mean_steps_all,
            "mean_return":        self.mean_return,
            "failure_timeout":    self.failure_timeout,
            "failure_collision":  self.failure_collision,
            "failure_retreating": self.failure_retreating,
        }


# ── retreating heuristic ─────────────────────────────────────────────────────

def _detect_retreating(episode: EpisodeResult, dist_history: list[float]) -> bool:
    """
    True if the agent's final distance to goal exceeds its minimum distance
    by more than a threshold — i.e. it approached then backed away.
    """
    if len(dist_history) < 2:
        return False
    min_d = min(dist_history)
    final_d = dist_history[-1]
    return (final_d - min_d) > 0.5   # retreated > 0.5 m from closest approach


# ── evaluator ─────────────────────────────────────────────────────────────────

class NavigationEvaluator:
    """
    Runs deterministic rollouts of a PPO navigation policy and computes
    honest metrics including a novel-layout generalisation test.

    Parameters
    ----------
    agent  : PPOAgent (or ActorCritic) to evaluate
    device : torch device for inference
    """

    def __init__(
        self,
        agent:  PPOAgent | ActorCritic,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if isinstance(agent, PPOAgent):
            self._net = agent.net
        else:
            self._net = agent
        self._net.eval()
        self._device = device

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        n_episodes: int = 100,
        seed:       int = 0,
        max_steps:  int = 300,
        n_substeps: int = 4,
        goal_cfg:   Optional[GoalConfig] = None,
        label:      str = "eval",
    ) -> EvalReport:
        """
        Evaluate on the standard training layout.

        goal_cfg defaults to random placement (same as stage-1 training).
        """
        if goal_cfg is None:
            goal_cfg = GoalConfig(mode="random")
        return self._run(
            n_episodes = n_episodes,
            seed       = seed,
            max_steps  = max_steps,
            n_substeps = n_substeps,
            goal_cfg   = goal_cfg,
            label      = label,
        )

    def evaluate_novel_layout(
        self,
        n_episodes: int = 50,
        seed:       int = 10000,
        max_steps:  int = 300,
        n_substeps: int = 4,
        label:      str = "novel_layout",
    ) -> EvalReport:
        """
        Generalisation test: goals placed exclusively to the side (lat_range)
        — a layout regime not seen during stage-0 curriculum training.
        """
        goal_cfg = GoalConfig(
            mode      = "relative",
            fwd_range = (0.5, 0.5),    # fixed forward distance
            lat_range = (1.5, 2.5),    # large lateral offset — novel regime
        )
        return self._run(
            n_episodes = n_episodes,
            seed       = seed,
            max_steps  = max_steps,
            n_substeps = n_substeps,
            goal_cfg   = goal_cfg,
            label      = label,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _run(
        self,
        n_episodes: int,
        seed:       int,
        max_steps:  int,
        n_substeps: int,
        goal_cfg:   GoalConfig,
        label:      str,
    ) -> EvalReport:
        env = NavigationEnv(
            max_steps  = max_steps,
            n_substeps = n_substeps,
            goal_cfg   = goal_cfg,
            seed       = seed,
        )

        results: list[EpisodeResult] = []

        for ep_idx in range(n_episodes):
            obs_np, _ = env.reset(seed=seed + ep_idx)
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32,
                                    device=self._device)
            ep_return = 0.0
            dist_history: list[float] = []

            for step in range(max_steps):
                with torch.no_grad():
                    act, _, _ = self._net.act(obs_t, deterministic=True)
                act_np = act.cpu().numpy()
                obs_np, rew, term, trunc, info = env.step(act_np)
                obs_t    = torch.as_tensor(obs_np, dtype=torch.float32,
                                           device=self._device)
                ep_return += float(rew)
                dist_history.append(float(info.get("dist_to_goal", 0.0)))

                if term or trunc:
                    success   = bool(info.get("success",   False))
                    collision = bool(info.get("collision", False))
                    retreating = (not success and not collision and
                                  _detect_retreating(
                                      EpisodeResult("", 0, 0.0, ""), dist_history
                                  ))
                    if success:
                        mode = "success"
                    elif collision:
                        mode = "collision"
                    elif retreating:
                        mode = "retreating"
                    else:
                        mode = "timeout"

                    results.append(EpisodeResult(
                        success      = success,
                        n_steps      = step + 1,
                        total_return = ep_return,
                        failure_mode = mode,
                    ))
                    break

        env.close()
        return _aggregate(results, label)

    # ── checkpoint loading ────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path:   str | Path,
        obs_dim:     int,
        act_dim:     int,
        hidden_sizes: tuple = (256, 256),
        device:      torch.device = torch.device("cpu"),
    ) -> "NavigationEvaluator":
        """Load a saved checkpoint and wrap it in a NavigationEvaluator."""
        sd  = torch.load(ckpt_path, map_location=device)
        net = ActorCritic(obs_dim=obs_dim, act_dim=act_dim,
                          hidden_sizes=hidden_sizes)
        net.load_state_dict(sd["agent"]["net"])
        return cls(agent=net, device=device)


# ── aggregation helper ────────────────────────────────────────────────────────

def _aggregate(results: list[EpisodeResult], label: str) -> EvalReport:
    n = len(results)
    if n == 0:
        return EvalReport(
            n_episodes=0, success_rate=0.0, mean_steps=0.0,
            mean_steps_all=0.0, mean_return=0.0,
            failure_timeout=0.0, failure_collision=0.0,
            failure_retreating=0.0, label=label, episodes=results,
        )

    successes  = [r for r in results if r.success]
    sr         = len(successes) / n
    mean_steps = float(np.mean([r.n_steps for r in successes])) if successes else 0.0
    mean_steps_all = float(np.mean([r.n_steps for r in results]))
    mean_return    = float(np.mean([r.total_return for r in results]))

    n_timeout    = sum(1 for r in results if r.failure_mode == "timeout")
    n_collision  = sum(1 for r in results if r.failure_mode == "collision")
    n_retreating = sum(1 for r in results if r.failure_mode == "retreating")

    return EvalReport(
        n_episodes        = n,
        success_rate      = sr,
        mean_steps        = mean_steps,
        mean_steps_all    = mean_steps_all,
        mean_return       = mean_return,
        failure_timeout   = n_timeout   / n,
        failure_collision = n_collision / n,
        failure_retreating= n_retreating/ n,
        label             = label,
        episodes          = results,
    )
