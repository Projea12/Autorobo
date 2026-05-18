"""
train_grasp.py — Grasp Policy Training Script (Phase 8)

Trains a SAC agent to perform pick-and-place manipulation using
the robot arm in a MuJoCo simulated tabletop environment.
Uses Stable-Baselines3, Hydra config, and a demonstration-seeded
replay buffer for more efficient learning.

Usage:
    python scripts/train_grasp.py
    python scripts/train_grasp.py training=sac
    python scripts/train_grasp.py training.sac.learning_rate=1e-4
    python scripts/train_grasp.py training.total_timesteps=5000000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def make_grasp_env(cfg: DictConfig, seed: int = 0):
    """
    Create a single grasp training environment.

    Args:
        cfg: Hydra config containing env and robot settings.
        seed: Random seed.

    Returns:
        Callable that instantiates a GraspingEnv gymnasium wrapper.

    TODO: Phase 8 — instantiate MuJoCoInterface with arm control enabled,
    wrap as GraspingEnv-v0, add Monitor wrapper.
    """
    raise NotImplementedError(
        "TODO: Phase 8 — implement GraspingEnv-v0 using MuJoCoInterface, "
        "wrap with HER (Hindsight Experience Replay) for sparse rewards."
    )


def seed_replay_buffer(model, demo_path: Path) -> None:
    """
    Pre-populate the SAC replay buffer with human demonstrations.

    Args:
        model: SAC model with an attached replay buffer.
        demo_path: Path to collected demonstration data (.npz or HDF5).

    TODO: Phase 8 — load demonstrations from collect_data.py output,
    validate (obs, action, reward, next_obs, done) tuples, add to buffer.
    """
    if not demo_path.exists():
        log.warning(
            f"Demonstration data not found at {demo_path}. "
            "Training without demonstrations — this will be slower."
        )
        return
    raise NotImplementedError(
        "TODO: Phase 8 — implement demonstration seeding using "
        "model.replay_buffer.add() for each (obs, act, rew, next_obs, done) tuple."
    )


def build_sac_model(cfg: DictConfig, env):
    """
    Construct a SAC model from Stable-Baselines3.

    Args:
        cfg: Hydra config with training.sac section.
        env: Gymnasium environment (single, not vectorised — SAC standard).

    Returns:
        stable_baselines3.SAC instance.

    TODO: Phase 8 — configure SAC with auto entropy tuning, CNN feature extractor
    for RGB-D input, optionally wrap with HER for goal-conditioned learning.
    """
    raise NotImplementedError(
        "TODO: Phase 8 — build SAC model with stable_baselines3.SAC, "
        "optionally wrap with stable_baselines3.HerReplayBuffer."
    )


def setup_sac_callbacks(cfg: DictConfig, eval_env) -> list:
    """
    Build SAC training callbacks.

    Args:
        cfg: Hydra config.
        eval_env: Evaluation environment.

    Returns:
        List of SB3 callback instances.

    TODO: Phase 8 — CheckpointCallback, EvalCallback (grasp-specific metrics),
    custom GraspSuccessRateCallback that logs per-object success rates.
    """
    raise NotImplementedError(
        "TODO: Phase 8 — build callbacks including a custom GraspSuccessRateCallback "
        "that logs success rates broken down by YCB object class."
    )


@hydra.main(config_path="../configs/hydra", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main grasp policy training entry point.

    Args:
        cfg: Composed Hydra config.

    Returns:
        Final grasp success rate (for Hydra sweeps).
    """
    log.info("=== AutoRobo Grasp Policy Training (Phase 8) ===")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    seed = cfg.project.seed
    demo_path = Path(cfg.project.data_dir) / "demonstrations"

    log.info("Creating grasp training environment...")
    # env = make_grasp_env(cfg, seed=seed)

    log.info("Building SAC model...")
    # model = build_sac_model(cfg, env)

    log.info(f"Seeding replay buffer from {demo_path}...")
    # seed_replay_buffer(model, demo_path)

    log.info("Setting up training callbacks...")
    # callbacks = setup_sac_callbacks(cfg, eval_env)

    log.info(
        f"Starting SAC training for {cfg.training.total_timesteps:,} timesteps..."
    )
    raise NotImplementedError(
        "TODO: Phase 8 — wire everything together and call model.learn(). "
        "Expected training time: ~6 hours on GPU for 2M timesteps."
    )

    # model.learn(
    #     total_timesteps=cfg.training.total_timesteps,
    #     callback=callbacks,
    #     log_interval=cfg.training.logging.log_interval,
    #     reset_num_timesteps=False,
    # )

    # final_path = Path(cfg.training.checkpoint.save_path) / "sac_grasp_final"
    # model.save(final_path)
    # log.info(f"Final grasp model saved to {final_path}")


if __name__ == "__main__":
    main()
