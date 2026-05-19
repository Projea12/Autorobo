"""
tests/test_replay_buffer.py — Unit + integration tests for the episode replay buffer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils.replay_buffer import (
    Episode, ReplayConfig, SampleBatch,
    EpisodeReplayBuffer, EpisodeCollector,
)

OBS_DIM = 45
ACT_DIM = 9
T       = 20   # episode length used in helpers


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_episode(
    total_return: float = -5.0,
    success: bool = False,
    length: int = T,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    rng     = np.random.default_rng(seed)
    obs     = rng.standard_normal((length, OBS_DIM)).astype(np.float32)
    actions = rng.standard_normal((length, ACT_DIM)).astype(np.float32)
    rewards = np.full(length, total_return / length, dtype=np.float32)
    dones   = np.zeros(length, dtype=bool)
    dones[-1] = True
    return obs, actions, rewards, dones, success


def _buf(capacity=50, **kwargs) -> EpisodeReplayBuffer:
    cfg = ReplayConfig(capacity=capacity, **kwargs)
    return EpisodeReplayBuffer(cfg=cfg, obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)


# ── construction ──────────────────────────────────────────────────────────────

def test_buffer_initial_size():
    assert _buf().size == 0


def test_buffer_initial_not_full():
    assert not _buf().is_full


def test_buffer_capacity():
    assert _buf(capacity=100).capacity == 100


def test_buffer_repr_empty():
    assert "EpisodeReplayBuffer" in repr(_buf())


# ── push ──────────────────────────────────────────────────────────────────────

def test_push_increments_size():
    buf = _buf()
    buf.push(*_make_episode())
    assert buf.size == 1


def test_push_returns_episode():
    buf = _buf()
    ep  = buf.push(*_make_episode())
    assert isinstance(ep, Episode)


def test_push_episode_id_monotonic():
    buf = _buf()
    ep0 = buf.push(*_make_episode(seed=0))
    ep1 = buf.push(*_make_episode(seed=1))
    assert ep1.episode_id == ep0.episode_id + 1


def test_push_success_flag():
    buf = _buf()
    ep  = buf.push(*_make_episode(success=True))
    assert ep.success is True


def test_push_failure_flag():
    buf = _buf()
    ep  = buf.push(*_make_episode(success=False))
    assert ep.is_failure is True


def test_push_total_return():
    buf = _buf()
    ep  = buf.push(*_make_episode(total_return=-10.0))
    assert abs(ep.total_return - (-10.0)) < 1e-4


def test_push_length():
    buf = _buf()
    ep  = buf.push(*_make_episode(length=30))
    assert ep.length == 30


def test_push_obs_dtype():
    buf = _buf()
    ep  = buf.push(*_make_episode())
    assert ep.obs.dtype == np.float32


def test_push_actions_dtype():
    buf = _buf()
    ep  = buf.push(*_make_episode())
    assert ep.actions.dtype == np.float32


def test_push_dones_dtype():
    buf = _buf()
    ep  = buf.push(*_make_episode())
    assert ep.dones.dtype == bool


def test_push_obs_shape():
    buf = _buf()
    ep  = buf.push(*_make_episode(length=T))
    assert ep.obs.shape == (T, OBS_DIM)


def test_push_actions_shape():
    buf = _buf()
    ep  = buf.push(*_make_episode(length=T))
    assert ep.actions.shape == (T, ACT_DIM)


def test_push_rewards_shape():
    buf = _buf()
    ep  = buf.push(*_make_episode(length=T))
    assert ep.rewards.shape == (T,)


# ── failure_only mode ─────────────────────────────────────────────────────────

def test_failure_only_filters_success():
    buf = _buf(failure_only=True)
    ep  = buf.push(*_make_episode(success=True))
    assert ep is None
    assert buf.size == 0


def test_failure_only_keeps_failures():
    buf = _buf(failure_only=True)
    ep  = buf.push(*_make_episode(success=False))
    assert ep is not None
    assert buf.size == 1


# ── ring buffer eviction ──────────────────────────────────────────────────────

def test_ring_buffer_wraps_at_capacity():
    buf = _buf(capacity=5)
    for i in range(7):
        buf.push(*_make_episode(seed=i))
    assert buf.size == 5


def test_ring_buffer_size_never_exceeds_capacity():
    buf = _buf(capacity=3)
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    assert buf.size <= 3


# ── priority ──────────────────────────────────────────────────────────────────

def test_failure_has_higher_priority_than_success():
    buf = _buf(success_priority_scale=0.1)
    buf.push(*_make_episode(total_return=-5.0, success=False, seed=0))
    buf.push(*_make_episode(total_return=-5.0, success=True,  seed=1))
    probs = buf._sampling_probs()
    assert probs[0] > probs[1]


def test_lower_return_has_higher_priority():
    buf = _buf()
    buf.push(*_make_episode(total_return=10.0, success=False, seed=0))
    buf.push(*_make_episode(total_return=-50.0, success=False, seed=1))
    probs = buf._sampling_probs()
    # slot 1 has lower return → higher priority
    assert probs[1] > probs[0]


def test_priorities_sum_to_one():
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(total_return=float(-i * 3), seed=i))
    assert abs(buf._sampling_probs().sum() - 1.0) < 1e-9


def test_priorities_all_nonnegative():
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    assert (buf._sampling_probs() >= 0).all()


# ── sample ────────────────────────────────────────────────────────────────────

def test_sample_returns_samplebatch():
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    batch = buf.sample(3)
    assert isinstance(batch, SampleBatch)


def test_sample_correct_count():
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    batch = buf.sample(5)
    assert len(batch.episodes) == 5


def test_sample_weights_shape():
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    batch = buf.sample(5)
    assert batch.weights.shape == (5,)


def test_sample_weights_max_is_one():
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    batch = buf.sample(5)
    assert abs(batch.weights.max() - 1.0) < 1e-6


def test_sample_weights_nonnegative():
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    assert (buf.sample(5).weights >= 0).all()


def test_sample_indices_in_range():
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    batch = buf.sample(5)
    assert all(0 <= idx < buf.size for idx in batch.indices)


def test_sample_no_duplicates():
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    batch = buf.sample(5)
    assert len(set(batch.indices.tolist())) == 5


def test_sample_raises_if_too_few():
    buf = _buf()
    buf.push(*_make_episode())
    with pytest.raises(ValueError):
        buf.sample(5)


def test_sample_episodes_are_episode_instances():
    buf = _buf()
    for i in range(5):
        buf.push(*_make_episode(seed=i))
    batch = buf.sample(3)
    for ep in batch.episodes:
        assert isinstance(ep, Episode)


# ── IS weight beta annealing ──────────────────────────────────────────────────

def test_beta_increases_over_time():
    cfg = ReplayConfig(beta_start=0.4, beta_end=1.0, beta_steps=10)
    buf = EpisodeReplayBuffer(cfg=cfg, seed=0)
    for i in range(20):
        buf.push(*_make_episode(seed=i))

    beta_early = buf._current_beta()
    for _ in range(5):
        buf.sample(3)
    beta_later = buf._current_beta()
    assert beta_later > beta_early


def test_beta_clamps_at_end():
    cfg = ReplayConfig(beta_start=0.4, beta_end=1.0, beta_steps=3)
    buf = EpisodeReplayBuffer(cfg=cfg, seed=0)
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    # exhaust annealing steps
    for _ in range(20):
        buf.sample(3)
    assert buf._current_beta() == pytest.approx(1.0)


# ── update_priorities ─────────────────────────────────────────────────────────

def test_update_priorities_changes_probs():
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    batch = buf.sample(3)
    probs_before = buf._sampling_probs().copy()
    buf.update_priorities(batch.indices, np.array([100.0, 100.0, 100.0]))
    probs_after = buf._sampling_probs()
    assert not np.allclose(probs_before, probs_after)


# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats_empty():
    buf = _buf()
    assert buf.stats()["size"] == 0


def test_stats_size():
    buf = _buf()
    for i in range(7):
        buf.push(*_make_episode(seed=i))
    assert buf.stats()["size"] == 7


def test_stats_failure_count():
    buf = _buf()
    buf.push(*_make_episode(success=False, seed=0))
    buf.push(*_make_episode(success=False, seed=1))
    buf.push(*_make_episode(success=True,  seed=2))
    assert buf.stats()["n_failures"] == 2


def test_stats_success_count():
    buf = _buf()
    buf.push(*_make_episode(success=True, seed=0))
    buf.push(*_make_episode(success=False, seed=1))
    assert buf.stats()["n_successes"] == 1


def test_stats_failure_rate():
    buf = _buf()
    for _ in range(3):
        buf.push(*_make_episode(success=False, seed=0))
    for _ in range(1):
        buf.push(*_make_episode(success=True, seed=1))
    assert buf.stats()["failure_rate"] == pytest.approx(0.75)


def test_stats_mean_return():
    buf = _buf()
    buf.push(*_make_episode(total_return=-10.0, seed=0))
    buf.push(*_make_episode(total_return=-20.0, seed=1))
    assert buf.stats()["mean_return"] == pytest.approx(-15.0, rel=1e-3)


def test_stats_n_added_total():
    buf = _buf(capacity=3)
    for i in range(7):
        buf.push(*_make_episode(seed=i))
    assert buf.stats()["n_added_total"] == 7


# ── episode iteration ─────────────────────────────────────────────────────────

def test_episodes_iterator_count():
    buf = _buf()
    for i in range(6):
        buf.push(*_make_episode(seed=i))
    assert len(list(buf.episodes())) == 6


def test_failure_episodes_filter():
    buf = _buf()
    buf.push(*_make_episode(success=False, seed=0))
    buf.push(*_make_episode(success=True,  seed=1))
    buf.push(*_make_episode(success=False, seed=2))
    assert len(buf.failure_episodes()) == 2


def test_success_episodes_filter():
    buf = _buf()
    buf.push(*_make_episode(success=True,  seed=0))
    buf.push(*_make_episode(success=False, seed=1))
    assert len(buf.success_episodes()) == 1


# ── persistence ───────────────────────────────────────────────────────────────

def test_save_creates_meta(tmp_path):
    buf = _buf()
    for i in range(5):
        buf.push(*_make_episode(seed=i))
    buf.save(tmp_path / "rbuf")
    assert (tmp_path / "rbuf" / "meta.json").exists()


def test_save_creates_episode_files(tmp_path):
    buf = _buf()
    for i in range(5):
        buf.push(*_make_episode(seed=i))
    buf.save(tmp_path / "rbuf")
    ep_files = list((tmp_path / "rbuf" / "episodes").glob("*.npz"))
    assert len(ep_files) == 5


def test_load_restores_size(tmp_path):
    buf = _buf()
    for i in range(8):
        buf.push(*_make_episode(seed=i))
    buf.save(tmp_path / "rbuf")
    buf2 = EpisodeReplayBuffer.load(tmp_path / "rbuf")
    assert buf2.size == 8


def test_load_restores_obs(tmp_path):
    buf = _buf()
    ep0 = buf.push(*_make_episode(seed=42))
    buf.save(tmp_path / "rbuf")
    buf2 = EpisodeReplayBuffer.load(tmp_path / "rbuf")
    ep0_r = next(buf2.episodes())
    assert np.allclose(ep0.obs, ep0_r.obs)


def test_load_restores_priorities(tmp_path):
    buf = _buf()
    for i in range(5):
        buf.push(*_make_episode(total_return=float(-i * 5), seed=i))
    p_before = buf._priorities[:buf.size].copy()
    buf.save(tmp_path / "rbuf")
    buf2 = EpisodeReplayBuffer.load(tmp_path / "rbuf")
    p_after = buf2._priorities[:buf2.size]
    assert np.allclose(sorted(p_before), sorted(p_after))


def test_load_restores_sample_step(tmp_path):
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    for _ in range(3):
        buf.sample(2)
    buf.save(tmp_path / "rbuf")
    buf2 = EpisodeReplayBuffer.load(tmp_path / "rbuf")
    assert buf2._sample_step == 3


def test_load_restores_next_episode_id(tmp_path):
    buf = _buf()
    for i in range(5):
        buf.push(*_make_episode(seed=i))
    buf.save(tmp_path / "rbuf")
    buf2 = EpisodeReplayBuffer.load(tmp_path / "rbuf")
    assert buf2._next_episode_id == 5


def test_load_can_still_sample(tmp_path):
    buf = _buf()
    for i in range(10):
        buf.push(*_make_episode(seed=i))
    buf.save(tmp_path / "rbuf")
    buf2 = EpisodeReplayBuffer.load(tmp_path / "rbuf")
    batch = buf2.sample(3)
    assert len(batch) == 3


# ── episode save / load round-trip ────────────────────────────────────────────

def test_episode_save_load_obs(tmp_path):
    buf = _buf()
    ep  = buf.push(*_make_episode(seed=7))
    ep.save(tmp_path)
    ep2 = Episode.load(tmp_path / f"{ep.episode_id}.npz")
    assert np.allclose(ep.obs, ep2.obs)


def test_episode_save_load_rewards(tmp_path):
    buf = _buf()
    ep  = buf.push(*_make_episode(total_return=-8.0, seed=3))
    ep.save(tmp_path)
    ep2 = Episode.load(tmp_path / f"{ep.episode_id}.npz")
    assert np.allclose(ep.rewards, ep2.rewards)


def test_episode_save_load_success(tmp_path):
    buf = _buf()
    ep  = buf.push(*_make_episode(success=True, seed=5))
    ep.save(tmp_path)
    ep2 = Episode.load(tmp_path / f"{ep.episode_id}.npz")
    assert ep2.success is True


def test_episode_repr_contains_id():
    buf = _buf()
    ep  = buf.push(*_make_episode())
    assert str(ep.episode_id) in repr(ep)


def test_episode_repr_contains_status():
    buf = _buf()
    ep  = buf.push(*_make_episode(success=False))
    assert "FAIL" in repr(ep)


# ── EpisodeCollector ──────────────────────────────────────────────────────────

def test_collector_flush_pushes_to_buffer():
    buf = _buf()
    col = EpisodeCollector(buf)
    rng = np.random.default_rng(0)
    for _ in range(T):
        obs    = rng.standard_normal(OBS_DIM).astype(np.float32)
        action = rng.standard_normal(ACT_DIM).astype(np.float32)
        col.step(obs, action, reward=-0.1, done=False)
    col.flush(success=False)
    assert buf.size == 1


def test_collector_flush_returns_episode():
    buf = _buf()
    col = EpisodeCollector(buf)
    rng = np.random.default_rng(1)
    for _ in range(T):
        col.step(rng.standard_normal(OBS_DIM).astype(np.float32),
                 rng.standard_normal(ACT_DIM).astype(np.float32),
                 reward=-0.1, done=False)
    ep = col.flush(success=True)
    assert isinstance(ep, Episode)


def test_collector_current_length():
    buf = _buf()
    col = EpisodeCollector(buf)
    rng = np.random.default_rng(2)
    for i in range(5):
        col.step(rng.standard_normal(OBS_DIM).astype(np.float32),
                 rng.standard_normal(ACT_DIM).astype(np.float32),
                 reward=-0.1, done=False)
    assert col.current_length == 5


def test_collector_clears_after_flush():
    buf = _buf()
    col = EpisodeCollector(buf)
    rng = np.random.default_rng(3)
    for _ in range(T):
        col.step(rng.standard_normal(OBS_DIM).astype(np.float32),
                 rng.standard_normal(ACT_DIM).astype(np.float32),
                 reward=-0.1, done=False)
    col.flush(success=False)
    assert col.current_length == 0


def test_collector_reset_clears_without_push():
    buf = _buf()
    col = EpisodeCollector(buf)
    rng = np.random.default_rng(4)
    for _ in range(5):
        col.step(rng.standard_normal(OBS_DIM).astype(np.float32),
                 rng.standard_normal(ACT_DIM).astype(np.float32),
                 reward=-0.1, done=False)
    col.reset()
    assert col.current_length == 0
    assert buf.size == 0


def test_collector_flush_empty_returns_none():
    buf = _buf()
    col = EpisodeCollector(buf)
    assert col.flush(success=False) is None


def test_collector_multiple_episodes():
    buf = _buf()
    col = EpisodeCollector(buf)
    rng = np.random.default_rng(5)
    for ep_i in range(4):
        for _ in range(T):
            col.step(rng.standard_normal(OBS_DIM).astype(np.float32),
                     rng.standard_normal(ACT_DIM).astype(np.float32),
                     reward=-0.1, done=False)
        col.flush(success=(ep_i == 3))
    assert buf.size == 4
    assert buf.stats()["n_successes"] == 1
