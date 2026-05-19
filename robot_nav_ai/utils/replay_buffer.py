"""
utils/replay_buffer.py — Episode-level replay buffer with priority sampling.

Design
──────
Stores complete episodes rather than individual transitions so the consumer
(offline fine-tuning, failure analysis, curriculum replay) always gets a
coherent trajectory.  Sampling is biased toward failure / low-return episodes
using proportional Prioritised Experience Replay (PER):

    priority_i  = (max_seen_return − return_i + ε) ^ α
    P(i)        = priority_i / Σ priority_j

Importance-sampling weights correct for the introduced bias:

    w_i = (1 / (N · P(i))) ^ β,  normalised by max(w_j)

When the buffer is full the episode with the *lowest* priority (highest return,
i.e. easiest episodes) is evicted first, so hard failures are preserved.

Persistence
───────────
save(path) writes  <path>/meta.json  +  <path>/episodes/<id>.npz
load(path) restores the full buffer including priorities.

Episode layout (stored per-episode)
────────────────────────────────────
  obs      : (T, obs_dim)  float32
  actions  : (T, act_dim)  float32
  rewards  : (T,)          float32
  dones    : (T,)          bool      — terminated flags (not truncated)
  total_return : float
  success      : bool
  length       : int
  episode_id   : int        — monotonic counter, never reused
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np


# ── episode dataclass ─────────────────────────────────────────────────────────

@dataclass
class Episode:
    """One complete environment episode."""
    obs:          np.ndarray   # (T, obs_dim) float32
    actions:      np.ndarray   # (T, act_dim) float32
    rewards:      np.ndarray   # (T,)         float32
    dones:        np.ndarray   # (T,)         bool
    total_return: float
    success:      bool
    length:       int
    episode_id:   int

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def is_failure(self) -> bool:
        """Episode ended without success (timeout or env termination)."""
        return not self.success

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / f"{self.episode_id}.npz",
            obs=self.obs,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            meta=np.array([self.total_return, float(self.success),
                            float(self.length), float(self.episode_id)]),
        )

    @classmethod
    def load(cls, npz_path: Path) -> "Episode":
        d = np.load(npz_path)
        total_return, success, length, episode_id = d["meta"]
        return cls(
            obs=d["obs"],
            actions=d["actions"],
            rewards=d["rewards"],
            dones=d["dones"].astype(bool),
            total_return=float(total_return),
            success=bool(success),
            length=int(length),
            episode_id=int(episode_id),
        )

    def __repr__(self) -> str:
        tag = "SUCCESS" if self.success else "FAIL"
        return (f"Episode(id={self.episode_id}, {tag}, "
                f"len={self.length}, R={self.total_return:.3f})")


# ── buffer configuration ──────────────────────────────────────────────────────

@dataclass
class ReplayConfig:
    """
    Parameters controlling the replay buffer.

    capacity          : max number of episodes stored (ring buffer)
    alpha             : priority exponent — 0 = uniform, 1 = full priority
    beta_start        : IS-weight exponent at t=0 (anneal toward 1.0)
    beta_end          : final IS-weight exponent
    beta_steps        : number of sample() calls over which beta is annealed
    eps               : small constant added to priorities to ensure non-zero
    success_priority_scale : multiplier applied to success-episode priorities
                        (< 1.0 down-weights successes relative to failures)
    failure_only      : if True, only store failure episodes
    """
    capacity:               int   = 10_000
    alpha:                  float = 0.6
    beta_start:             float = 0.4
    beta_end:               float = 1.0
    beta_steps:             int   = 50_000
    eps:                    float = 1e-6
    success_priority_scale: float = 0.25
    failure_only:           bool  = False


# ── sample batch ─────────────────────────────────────────────────────────────

@dataclass
class SampleBatch:
    """Return type of EpisodeReplayBuffer.sample()."""
    episodes:  list[Episode]
    weights:   np.ndarray    # (n,) float32 IS weights, max-normalised to [0,1]
    indices:   np.ndarray    # (n,) int buffer slot indices (for update_priorities)

    def __len__(self) -> int:
        return len(self.episodes)


# ── replay buffer ─────────────────────────────────────────────────────────────

class EpisodeReplayBuffer:
    """
    Fixed-capacity episode replay buffer with priority sampling.

    Parameters
    ----------
    cfg          : ReplayConfig
    obs_dim      : observation dimension (used for type checking on push)
    act_dim      : action dimension
    seed         : RNG seed for reproducible sampling
    """

    def __init__(
        self,
        cfg:     ReplayConfig = ReplayConfig(),
        obs_dim: Optional[int] = None,
        act_dim: Optional[int] = None,
        seed:    int           = 0,
    ) -> None:
        self.cfg     = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._rng    = np.random.default_rng(seed)

        self._episodes:    list[Optional[Episode]] = [None] * cfg.capacity
        self._priorities:  np.ndarray              = np.zeros(cfg.capacity, dtype=np.float64)
        self._size:        int                     = 0
        self._write_ptr:   int                     = 0    # next slot to write
        self._n_added:     int                     = 0    # monotonic counter
        self._sample_step: int                     = 0    # for beta annealing
        self._max_return:  float                   = 0.0  # tracked for priority

        self._next_episode_id: int = 0

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self.cfg.capacity

    @property
    def is_full(self) -> bool:
        return self._size == self.cfg.capacity

    def push(
        self,
        obs:     np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones:   np.ndarray,
        success: bool,
    ) -> Optional[Episode]:
        """
        Add a complete episode to the buffer.

        Parameters
        ----------
        obs     : (T, obs_dim) observations including the final obs
        actions : (T, act_dim) actions taken
        rewards : (T,)         per-step rewards
        dones   : (T,)         terminated flags
        success : whether the episode ended with task success

        Returns the Episode object if stored, None if filtered (failure_only).
        """
        if self.cfg.failure_only and success:
            return None

        total_return = float(np.sum(rewards))
        if total_return > self._max_return:
            self._max_return = total_return

        ep = Episode(
            obs          = np.asarray(obs,     dtype=np.float32),
            actions      = np.asarray(actions, dtype=np.float32),
            rewards      = np.asarray(rewards, dtype=np.float32),
            dones        = np.asarray(dones,   dtype=bool),
            total_return = total_return,
            success      = success,
            length       = int(len(rewards)),
            episode_id   = self._next_episode_id,
        )
        self._next_episode_id += 1

        priority = self._compute_priority(total_return, success)

        slot = self._write_ptr
        self._episodes[slot]   = ep
        self._priorities[slot] = priority
        self._write_ptr        = (self._write_ptr + 1) % self.cfg.capacity
        self._size             = min(self._size + 1, self.cfg.capacity)
        self._n_added         += 1

        return ep

    def sample(self, n: int) -> SampleBatch:
        """
        Sample n episodes with priority-proportional probability.

        Raises ValueError if the buffer has fewer than n episodes.
        """
        if self._size < n:
            raise ValueError(
                f"Buffer has {self._size} episodes, requested {n}."
            )

        probs = self._sampling_probs()
        indices = self._rng.choice(self._size, size=n, replace=False, p=probs[:self._size])

        beta  = self._current_beta()
        self._sample_step += 1

        raw_w = (self._size * probs[indices]) ** (-beta)
        weights = (raw_w / raw_w.max()).astype(np.float32)

        episodes = [self._episodes[i] for i in indices]
        return SampleBatch(episodes=episodes, weights=weights, indices=indices)

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray) -> None:
        """
        Update priorities for given buffer slot indices (after TD-error update).

        new_priorities should be raw (not yet exponentiated); they will be
        processed through the same pipeline as push().
        """
        for idx, p in zip(indices, new_priorities):
            self._priorities[idx] = max(float(p), self.cfg.eps) ** self.cfg.alpha

    def episodes(self) -> Iterator[Episode]:
        """Iterate over all stored episodes in insertion order."""
        for i in range(self._size):
            ep = self._episodes[i]
            if ep is not None:
                yield ep

    def failure_episodes(self) -> list[Episode]:
        return [e for e in self.episodes() if e.is_failure]

    def success_episodes(self) -> list[Episode]:
        return [e for e in self.episodes() if e.success]

    def stats(self) -> dict:
        """Summary statistics for the current buffer contents."""
        if self._size == 0:
            return {"size": 0}
        eps = list(self.episodes())
        returns  = np.array([e.total_return for e in eps])
        n_fail   = sum(1 for e in eps if e.is_failure)
        n_succ   = self._size - n_fail
        return {
            "size":             self._size,
            "capacity":         self.cfg.capacity,
            "n_failures":       n_fail,
            "n_successes":      n_succ,
            "failure_rate":     n_fail / self._size,
            "mean_return":      float(returns.mean()),
            "min_return":       float(returns.min()),
            "max_return":       float(returns.max()),
            "mean_length":      float(np.mean([e.length for e in eps])),
            "n_added_total":    self._n_added,
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the buffer to disk."""
        root = Path(path)
        ep_dir = root / "episodes"
        ep_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "cfg":              asdict(self.cfg),
            "obs_dim":          self.obs_dim,
            "act_dim":          self.act_dim,
            "size":             self._size,
            "write_ptr":        self._write_ptr,
            "n_added":          self._n_added,
            "sample_step":      self._sample_step,
            "max_return":       self._max_return,
            "next_episode_id":  self._next_episode_id,
            "priorities":       self._priorities[:self._size].tolist(),
            "episode_ids":      [
                self._episodes[i].episode_id if self._episodes[i] is not None else -1
                for i in range(self._size)
            ],
            "saved_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (root / "meta.json").write_text(json.dumps(meta, indent=2))

        for i in range(self._size):
            ep = self._episodes[i]
            if ep is not None:
                ep.save(ep_dir)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "EpisodeReplayBuffer":
        """Restore a buffer from disk."""
        root = Path(path)
        meta = json.loads((root / "meta.json").read_text())

        from dataclasses import fields as dc_fields
        cfg_dict = meta["cfg"]
        valid_keys = {f.name for f in dc_fields(ReplayConfig)}
        cfg = ReplayConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

        buf = cls(cfg=cfg, obs_dim=meta["obs_dim"], act_dim=meta["act_dim"], seed=seed)
        buf._write_ptr        = meta["write_ptr"]
        buf._n_added          = meta["n_added"]
        buf._sample_step      = meta["sample_step"]
        buf._max_return       = meta["max_return"]
        buf._next_episode_id  = meta["next_episode_id"]

        ep_dir   = root / "episodes"
        id_to_ep = {}
        for npz in ep_dir.glob("*.npz"):
            ep = Episode.load(npz)
            id_to_ep[ep.episode_id] = ep

        priorities = meta["priorities"]
        episode_ids = meta["episode_ids"]
        for slot, (eid, pri) in enumerate(zip(episode_ids, priorities)):
            if eid != -1 and eid in id_to_ep:
                buf._episodes[slot]   = id_to_ep[eid]
                buf._priorities[slot] = pri

        buf._size = meta["size"]
        return buf

    # ── internal helpers ──────────────────────────────────────────────────────

    def _compute_priority(self, total_return: float, success: bool) -> float:
        """Map (return, success) → priority scalar (already exponentiated)."""
        raw = (self._max_return - total_return + self.cfg.eps) ** self.cfg.alpha
        if success:
            raw *= self.cfg.success_priority_scale
        return max(raw, self.cfg.eps ** self.cfg.alpha)

    def _sampling_probs(self) -> np.ndarray:
        p = self._priorities[:self._size].copy()
        total = p.sum()
        if total == 0.0:
            return np.ones(self._size, dtype=np.float64) / self._size
        return p / total

    def _current_beta(self) -> float:
        frac = min(self._sample_step / max(self.cfg.beta_steps, 1), 1.0)
        return self.cfg.beta_start + frac * (self.cfg.beta_end - self.cfg.beta_start)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        s = self.stats()
        return (f"EpisodeReplayBuffer(size={self._size}/{self.cfg.capacity}, "
                f"failures={s.get('n_failures', 0)}, "
                f"mean_R={s.get('mean_return', 0):.2f})")


# ── episode collector ─────────────────────────────────────────────────────────

class EpisodeCollector:
    """
    Accumulates per-step data during environment interaction and flushes a
    complete Episode into the buffer on episode end.

    Usage
    -----
        collector = EpisodeCollector(buffer)
        obs, _ = env.reset()
        while True:
            action = policy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            collector.step(obs, action, reward, terminated)
            obs = next_obs
            if terminated or truncated:
                collector.flush(success=info.get("success", False))
                obs, _ = env.reset()
    """

    def __init__(self, buffer: EpisodeReplayBuffer) -> None:
        self._buf     = buffer
        self._obs:     list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float]      = []
        self._dones:   list[bool]       = []

    def step(
        self,
        obs:    np.ndarray,
        action: np.ndarray,
        reward: float,
        done:   bool,
    ) -> None:
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))

    def flush(self, success: bool) -> Optional[Episode]:
        """Finalise the current episode and push it to the buffer."""
        if not self._obs:
            return None
        ep = self._buf.push(
            obs     = np.stack(self._obs),
            actions = np.stack(self._actions),
            rewards = np.array(self._rewards, dtype=np.float32),
            dones   = np.array(self._dones,   dtype=bool),
            success = success,
        )
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        return ep

    def reset(self) -> None:
        """Discard accumulated steps without flushing (e.g., on env crash)."""
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()

    @property
    def current_length(self) -> int:
        return len(self._rewards)
