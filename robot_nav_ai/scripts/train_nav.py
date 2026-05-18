"""
train_nav.py — Navigation Policy Training Script (Phase 3)

Trains a PPO agent to navigate the mobile base to goal positions
in a MuJoCo simulated environment. Uses Stable-Baselines3 and Hydra
for configuration management.

Usage:
    python scripts/train_nav.py
    python scripts/train_nav.py training.ppo.learning_rate=1e-4
    python scripts/train_nav.py training.total_timesteps=10000000
    python scripts/train_nav.py --multirun training.ppo.gamma=0.95,0.99
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def make_env(cfg: DictConfig, rank: int = 0, seed: int = 0):
    """
    Factory function to create a single navigation environment.

    Args:
        cfg: Hydra config containing env and robot settings.
        rank: Environment index (for vectorised envs).
        seed: Random seed offset.

    Returns:
        Callable that returns a gym.Env instance.

    TODO: Phase 3 — instantiate MuJoCoInterface, wrap in gym.Env,
    apply Monitor wrapper for episode logging.
    """
    raise NotImplementedError(
        "TODO: Phase 3 — implement make_env using MuJoCoInterface "
        "and register MobileNavigationEnv-v0 with gymnasium."
    )


def build_model(cfg: DictConfig, env):
    """
    Construct a PPO model from Stable-Baselines3.

    Args:
        cfg: Hydra config with training.ppo section.
        env: Vectorised gymnasium environment.

    Returns:
        stable_baselines3.PPO instance ready to train.

    TODO: Phase 3 — build PPO with MultiInputPolicy, custom CNN
    feature extractor for RGB+depth, configure W&B callback.
    """
    raise NotImplementedError(
        "TODO: Phase 3 — build PPO model with stable_baselines3.PPO, "
        "configure TensorBoard + W&B logging callbacks."
    )


def load_checkpoint(cfg: DictConfig, model):
    """
    Resume training from a saved checkpoint if one exists.

    Args:
        cfg: Hydra config with checkpoint path settings.
        model: PPO model to load weights into.

    Returns:
        model with loaded weights, or original model if no checkpoint found.

    TODO: Phase 3 — scan checkpoint dir, load latest .zip, log step count.
    """
    checkpoint_path = Path(cfg.training.checkpoint.save_path)
    if not checkpoint_path.exists():
        log.info("No checkpoint directory found — training from scratch.")
        return model

    checkpoints = sorted(checkpoint_path.glob("*.zip"))
    if not checkpoints:
        log.info("No checkpoint files found — training from scratch.")
        return model

    latest = checkpoints[-1]
    log.info(f"Resuming from checkpoint: {latest}")
    raise NotImplementedError(
        f"TODO: Phase 3 — load PPO from {latest} using model.load(latest, env=env)."
    )


def setup_callbacks(cfg: DictConfig, eval_env) -> list:
    """
    Build SB3 training callbacks: checkpoint saver, eval callback, W&B.

    Args:
        cfg: Hydra config.
        eval_env: Evaluation environment.

    Returns:
        List of stable_baselines3 callbacks.

    TODO: Phase 3 — assemble EvalCallback, CheckpointCallback, WandbCallback.
    """
    raise NotImplementedError(
        "TODO: Phase 3 — configure CheckpointCallback, EvalCallback, "
        "and optionally WandbCallback from wandb.integration.sb3."
    )


@hydra.main(config_path="../configs/hydra", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main training entry point.

    Args:
        cfg: Composed Hydra config (robot + env + training + perception).

    Returns:
        Final mean reward (for Hydra sweeps / optuna integration).
    """
    log.info("=== AutoRobo Navigation Policy Training (Phase 3) ===")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set global seed
    seed = cfg.project.seed
    log.info(f"Global seed: {seed}")

    # 1. Create vectorised training environments
    log.info(f"Creating {cfg.training.n_envs} parallel environments...")
    # vec_env = make_vec_env(make_env(cfg), n_envs=cfg.training.n_envs, seed=seed)
    raise NotImplementedError(
        "TODO: Phase 3 — wire up make_env, make_vec_env, build_model, "
        "load_checkpoint, setup_callbacks, then call model.learn()."
    )

    # 2. Build model
    # model = build_model(cfg, vec_env)

    # 3. Optionally resume from checkpoint
    # model = load_checkpoint(cfg, model)

    # 4. Setup callbacks
    # callbacks = setup_callbacks(cfg, eval_env)

    # 5. Train
    # model.learn(
    #     total_timesteps=cfg.training.total_timesteps,
    #     callback=callbacks,
    #     log_interval=cfg.training.logging.log_interval,
    #     reset_num_timesteps=False,
    # )

    # 6. Save final model
    # final_path = Path(cfg.training.checkpoint.save_path) / "final_model"
    # model.save(final_path)
    # log.info(f"Final model saved to {final_path}")

    # 7. Return final eval reward for sweep optimisation
    # return eval_callback.best_mean_reward


if __name__ == "__main__":
    main()
