import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RobotNavEnv(gym.Env):
    """Custom Gymnasium environment for robot navigation."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Observation: [x, y, theta, goal_x, goal_y, lidar_readings x N]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Action: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        self.state = None
        self.goal = None
        self.step_count = 0
        self.max_steps = 500

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = np.zeros(7, dtype=np.float32)
        self.goal = self.np_random.uniform(-5.0, 5.0, size=(2,)).astype(np.float32)
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1
        # Placeholder dynamics — replace with PyBullet/MuJoCo integration
        self.state[:2] += action * 0.1

        obs = self._get_obs()
        dist = np.linalg.norm(self.state[:2] - self.goal)
        reward = -dist
        terminated = bool(dist < 0.2)
        truncated = self.step_count >= self.max_steps
        info = {"distance_to_goal": dist}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.concatenate([self.state[:3], self.goal, [0.0, 0.0]])
        return obs.astype(np.float32)

    def render(self):
        pass

    def close(self):
        pass
