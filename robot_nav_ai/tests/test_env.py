import numpy as np
import pytest
from env import RobotNavEnv


@pytest.fixture
def env():
    e = RobotNavEnv()
    yield e
    e.close()


def test_reset_returns_valid_obs(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_step_returns_correct_types(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_terminates_within_max_steps(env):
    env.reset(seed=1)
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps <= env.max_steps
