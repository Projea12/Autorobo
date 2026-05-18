"""Evaluate a trained navigation policy and optionally record a video."""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from env import RobotNavEnv

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "..", "videos")


def evaluate(model_path: str, n_episodes: int = 5, record: bool = False):
    env = DummyVecEnv([lambda: RobotNavEnv(render_mode="rgb_array")])

    if record:
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        env = VecVideoRecorder(env, VIDEOS_DIR, record_video_trigger=lambda _: True, video_length=500)

    model = PPO.load(model_path, env=env)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
        print(f"Episode {ep + 1}: total reward = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    evaluate(os.path.join(MODELS_DIR, "ppo_robot_nav_final"))
