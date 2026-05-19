"""
agent/ppo.py — Self-contained PPO implementation (PyTorch, no SB3 dependency).

Components
──────────
  ActorCritic   : shared MLP trunk, separate policy head (Gaussian) and value head
  RolloutBuffer : fixed-size on-policy storage with GAE advantage computation
  PPOAgent      : full clip-objective PPO update with optional entropy bonus

Usage
─────
    from agent.ppo import ActorCritic, PPOConfig, PPOAgent

    net   = ActorCritic(obs_dim=env.observation_space.shape[0],
                        act_dim=env.action_space.shape[0])
    cfg   = PPOConfig()
    agent = PPOAgent(net, cfg)

    # collect rollout
    buf = agent.rollout_buffer
    obs, info = env.reset()
    for _ in range(cfg.n_steps):
        act, logp, val = agent.act(obs_tensor)
        next_obs, rew, term, trunc, info = env.step(act.numpy())
        buf.add(obs_tensor, act, logp, val, rew, term or trunc)
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    agent.finish_rollout(last_obs_tensor, last_done)

    metrics = agent.update()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    """Hyper-parameters for PPOAgent."""
    # rollout
    n_steps:         int   = 2048    # steps per rollout per env
    gamma:           float = 0.99
    gae_lambda:      float = 0.95

    # optimisation
    n_epochs:        int   = 10
    batch_size:      int   = 64
    lr:              float = 3e-4
    clip_eps:        float = 0.2
    vf_coef:         float = 0.5
    ent_coef:        float = 0.01
    max_grad_norm:   float = 0.5

    # network
    hidden_sizes:    Tuple[int, ...]  = (256, 256)
    log_std_init:    float = -0.5
    log_std_min:     float = -4.0
    log_std_max:     float = 2.0

    # misc
    normalize_adv:   bool  = True
    target_kl:       Optional[float] = None   # early-stop epoch if KL > target_kl


# ── network ───────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-trunk MLP with a diagonal Gaussian policy head and a scalar value head.

    obs → trunk → [mu (act_dim), log_std (act_dim), value (1)]
    """

    def __init__(
        self,
        obs_dim:      int,
        act_dim:      int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # shared trunk
        layers: list[nn.Module] = []
        in_size = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.Tanh()]
            in_size = h
        self.trunk = nn.Sequential(*layers)

        # policy head
        self.policy_mu  = nn.Linear(in_size, act_dim)
        self.log_std    = nn.Parameter(torch.full((act_dim,), log_std_init))

        # value head
        self.value_head = nn.Linear(in_size, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_mu.weight, gain=0.01)
        nn.init.zeros_(self.policy_mu.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu, log_std_clamped, value)."""
        h     = self.trunk(obs)
        mu    = self.policy_mu(h)
        log_s = self.log_std.expand_as(mu)
        val   = self.value_head(h).squeeze(-1)
        return mu, log_s, val

    def distribution(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """Return (Normal distribution, value)."""
        mu, log_s, val = self(obs)
        std = log_s.exp()
        return Normal(mu, std), val

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and return (action, log_prob, value).
        Actions are NOT clipped here — the env or action processor handles that.
        """
        dist, val = self.distribution(obs)
        action    = dist.mean if deterministic else dist.rsample()
        log_prob  = dist.log_prob(action).sum(-1)
        return action, log_prob, val

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate stored actions under current policy.
        Returns (log_prob, entropy, value).
        """
        dist, val = self.distribution(obs)
        log_prob  = dist.log_prob(actions).sum(-1)
        entropy   = dist.entropy().sum(-1)
        return log_prob, entropy, val


# ── rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Fixed-capacity on-policy rollout storage.

    Call add() for each environment step, then finish_rollout() once to compute
    GAE advantages, then iterate mini-batches via batches().
    """

    def __init__(
        self,
        n_steps:   int,
        obs_dim:   int,
        act_dim:   int,
        gamma:     float = 0.99,
        gae_lambda: float = 0.95,
        device:    torch.device = torch.device("cpu"),
    ) -> None:
        self.n_steps    = n_steps
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device
        self._ptr       = 0
        self._full      = False
        self._advantages_ready = False

        self.observations = torch.zeros(n_steps, obs_dim,  device=device)
        self.actions      = torch.zeros(n_steps, act_dim,  device=device)
        self.log_probs    = torch.zeros(n_steps,            device=device)
        self.values       = torch.zeros(n_steps,            device=device)
        self.rewards      = torch.zeros(n_steps,            device=device)
        self.dones        = torch.zeros(n_steps,            device=device)
        self.advantages   = torch.zeros(n_steps,            device=device)
        self.returns      = torch.zeros(n_steps,            device=device)

    @property
    def size(self) -> int:
        return self.n_steps if self._full else self._ptr

    def reset(self) -> None:
        self._ptr = 0
        self._full = False
        self._advantages_ready = False

    def add(
        self,
        obs:      torch.Tensor,   # (obs_dim,)
        action:   torch.Tensor,   # (act_dim,)
        log_prob: torch.Tensor,   # scalar
        value:    torch.Tensor,   # scalar
        reward:   float,
        done:     bool,
    ) -> None:
        assert self._ptr < self.n_steps, "Buffer full — call reset() first."
        i = self._ptr
        self.observations[i] = obs.detach()
        self.actions[i]      = action.detach()
        self.log_probs[i]    = log_prob.detach()
        self.values[i]       = value.detach()
        self.rewards[i]      = reward
        self.dones[i]        = float(done)
        self._ptr           += 1
        if self._ptr == self.n_steps:
            self._full = True

    def finish_rollout(
        self,
        last_value: torch.Tensor,  # scalar — V(s_{T})
        last_done:  bool,
    ) -> None:
        """Compute GAE advantages and discounted returns in-place."""
        gae     = 0.0
        next_v  = last_value.item() * (1.0 - float(last_done))
        n       = self.size

        for t in reversed(range(n)):
            mask   = 1.0 - self.dones[t].item()
            delta  = (self.rewards[t].item()
                      + self.gamma * next_v * mask
                      - self.values[t].item())
            gae    = delta + self.gamma * self.gae_lambda * mask * gae
            self.advantages[t] = gae
            next_v             = self.values[t].item()
        self.returns[:n] = self.advantages[:n] + self.values[:n]
        self._advantages_ready = True

    def batches(
        self, batch_size: int
    ):
        """Yield shuffled mini-batches as tuples of tensors."""
        assert self._advantages_ready, "Call finish_rollout() before iterating."
        n       = self.size
        indices = torch.randperm(n, device=self.device)
        start   = 0
        while start < n:
            idx   = indices[start : start + batch_size]
            start += batch_size
            yield (
                self.observations[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.returns[idx],
                self.advantages[idx],
            )


# ── PPO agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Wraps ActorCritic + RolloutBuffer with a full PPO update step.

    Typical loop
    ────────────
        agent = PPOAgent(net, PPOConfig())
        for iteration in range(total_iterations):
            # --- collect ---
            agent.rollout_buffer.reset()
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            for _ in range(cfg.n_steps):
                with torch.no_grad():
                    act, lp, val = agent.net.act(obs_t)
                obs2, rew, done, trunc, _ = env.step(act.numpy())
                agent.rollout_buffer.add(obs_t, act, lp, val, rew, done or trunc)
                obs_t = torch.as_tensor(obs2, dtype=torch.float32)
                if done or trunc:
                    obs_t = torch.as_tensor(env.reset()[0], dtype=torch.float32)
            with torch.no_grad():
                _, _, last_val = agent.net.act(obs_t)
            agent.rollout_buffer.finish_rollout(last_val, done or trunc)
            # --- update ---
            metrics = agent.update()
    """

    def __init__(
        self,
        net:    ActorCritic,
        cfg:    PPOConfig,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.net    = net.to(device)
        self.cfg    = cfg
        self.device = device

        self.optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
        self.rollout_buffer = RolloutBuffer(
            n_steps    = cfg.n_steps,
            obs_dim    = net.obs_dim,
            act_dim    = net.act_dim,
            gamma      = cfg.gamma,
            gae_lambda = cfg.gae_lambda,
            device     = device,
        )

    # ── convenience act wrapper ───────────────────────────────────────────────

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, value) — no gradient tracking."""
        return self.net.act(obs.to(self.device), deterministic=deterministic)

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """
        Run n_epochs of mini-batch PPO updates.

        Returns a dict of mean training metrics for logging.
        """
        cfg = self.cfg
        buf = self.rollout_buffer

        adv = buf.advantages[: buf.size]
        if cfg.normalize_adv and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            buf.advantages[: buf.size] = adv

        total_pg   = 0.0
        total_vf   = 0.0
        total_ent  = 0.0
        total_loss = 0.0
        n_batches  = 0
        early_stop = False

        for _ in range(cfg.n_epochs):
            if early_stop:
                break
            for obs_b, act_b, old_lp_b, ret_b, adv_b in buf.batches(cfg.batch_size):
                log_prob, entropy, value = self.net.evaluate_actions(obs_b, act_b)

                # policy gradient loss (clipped)
                ratio    = (log_prob - old_lp_b).exp()
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # value loss (clipped for stability)
                vf_loss  = F.mse_loss(value, ret_b)

                # entropy bonus
                ent_loss = -entropy.mean()

                loss = pg_loss + cfg.vf_coef * vf_loss + cfg.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_pg   += pg_loss.item()
                total_vf   += vf_loss.item()
                total_ent  += (-ent_loss).item()
                total_loss += loss.item()
                n_batches  += 1

                # optional early stopping on KL divergence
                if cfg.target_kl is not None:
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (log_prob - old_lp_b)).mean().item()
                    if approx_kl > cfg.target_kl:
                        early_stop = True
                        break

        n = max(n_batches, 1)
        return {
            "loss/policy":  total_pg   / n,
            "loss/value":   total_vf   / n,
            "loss/entropy": total_ent  / n,
            "loss/total":   total_loss / n,
        }

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        return {
            "net":       self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.net.load_state_dict(sd["net"])
        self.optimizer.load_state_dict(sd["optimizer"])


# ── factory ───────────────────────────────────────────────────────────────────

def make_ppo_agent(
    obs_dim:      int,
    act_dim:      int,
    cfg:          PPOConfig = PPOConfig(),
    device:       torch.device = torch.device("cpu"),
) -> PPOAgent:
    """Convenience factory that builds ActorCritic + PPOAgent in one call."""
    net = ActorCritic(
        obs_dim      = obs_dim,
        act_dim      = act_dim,
        hidden_sizes = cfg.hidden_sizes,
        log_std_init = cfg.log_std_init,
    )
    return PPOAgent(net, cfg, device=device)
