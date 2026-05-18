"""Train a navigation policy using Stable-Baselines3."""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from env import RobotNavEnv

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def train(total_timesteps: int = 500_000, n_envs: int = 4):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    env = make_vec_env(RobotNavEnv, n_envs=n_envs)
    eval_env = make_vec_env(RobotNavEnv, n_envs=1)

    callbacks = [
        EvalCallback(eval_env, best_model_save_path=MODELS_DIR, log_path=LOGS_DIR, eval_freq=10_000),
        CheckpointCallback(save_freq=50_000, save_path=MODELS_DIR, name_prefix="ppo_robot_nav"),
    ]

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(MODELS_DIR, "ppo_robot_nav_final"))
    print("Training complete. Model saved.")


if __name__ == "__main__":
    train()
