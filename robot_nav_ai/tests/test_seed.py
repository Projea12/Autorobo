"""
Proves that set_global_seed guarantees reproducibility:
  - same root seed  → identical observations, rewards, goals every episode
  - different seeds → different outcomes (sanity-check that seeding is active)
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.seed import set_global_seed, SeedConfig
from env.robot_nav_env import RobotNavEnv


# ── helpers ───────────────────────────────────────────────────────────────────

def run_episode(root_seed: int, n_steps: int = 10):
    """Seed globally, reset env with derived env_seed, collect trajectory."""
    cfg = set_global_seed(root_seed)
    env = RobotNavEnv()
    obs, _ = env.reset(seed=cfg.env_seed)

    obs_seq, reward_seq = [obs.copy()], []
    for _ in range(n_steps):
        action = env.action_space.sample()   # np_random drives this
        obs, reward, terminated, truncated, _ = env.step(action)
        obs_seq.append(obs.copy())
        reward_seq.append(float(reward))
        if terminated or truncated:
            break

    env.close()
    return np.array(obs_seq), np.array(reward_seq)


# ── SeedConfig derivation ─────────────────────────────────────────────────────

def test_seedconfig_is_deterministic():
    a = SeedConfig(root=0)
    b = SeedConfig(root=0)
    assert a.numpy_seed == b.numpy_seed
    assert a.torch_seed  == b.torch_seed
    assert a.env_seed    == b.env_seed
    assert a.python_seed == b.python_seed


def test_seedconfig_differs_across_roots():
    a = SeedConfig(root=1)
    b = SeedConfig(root=2)
    assert a.env_seed != b.env_seed


def test_seedconfig_fields_in_valid_range():
    cfg = SeedConfig(root=99)
    for val in (cfg.numpy_seed, cfg.torch_seed, cfg.env_seed, cfg.python_seed):
        assert 0 <= val < 2**31


# ── set_global_seed return value ──────────────────────────────────────────────

def test_set_global_seed_returns_seedconfig():
    cfg = set_global_seed(7)
    assert isinstance(cfg, SeedConfig)
    assert cfg.root == 7


# ── NumPy reproducibility ─────────────────────────────────────────────────────

def test_numpy_reproducible():
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


# ── Python random reproducibility ────────────────────────────────────────────

def test_python_random_reproducible():
    import random
    set_global_seed(42)
    a = [random.random() for _ in range(5)]
    set_global_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


# ── PyTorch reproducibility ───────────────────────────────────────────────────

def test_torch_reproducible():
    torch = pytest.importorskip("torch")
    set_global_seed(42)
    a = torch.randn(5)
    set_global_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


# ── full episode reproducibility ──────────────────────────────────────────────

def test_same_seed_gives_identical_episode():
    obs_a, rew_a = run_episode(seed=42)
    obs_b, rew_b = run_episode(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(rew_a, rew_b)


def test_different_seeds_give_different_episodes():
    obs_a, _ = run_episode(seed=42)
    obs_b, _ = run_episode(seed=99)
    assert not np.array_equal(obs_a, obs_b)


def test_env_seed_matches_three_consecutive_resets():
    """Three resets with the same env_seed must produce the same initial obs."""
    cfg = set_global_seed(123)
    env = RobotNavEnv()

    results = []
    for _ in range(3):
        obs, _ = env.reset(seed=cfg.env_seed)
        results.append(obs.copy())

    env.close()
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[1], results[2])
