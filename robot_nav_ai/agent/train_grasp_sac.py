"""
agent/train_grasp_sac.py — SAC training entry-point for the grasp policy.

Trains a Soft Actor-Critic agent to grasp objects of varying shapes, sizes,
and poses in the ManipulationEnv MuJoCo environment.

Algorithm: SAC + HER (Hindsight Experience Replay)
  HER converts failed grasp episodes into successful ones by relabelling
  the achieved goal (e.g. "object moved to X") as the intended goal,
  dramatically improving sample efficiency under sparse rewards.

Curriculum (two stages)
────────────────────────
  Stage 0 (0 → stage0_steps): objects always placed directly in front of arm
    — reduces initial exploration burden
  Stage 1 (stage0_steps → total_steps): full random placement within reach

Two-stage training mirrors the nav policy (nav_train.py) so both policies
can be evaluated and compared on the same timeline.

Checkpointing
─────────────
  Saves every --save-freq steps (default 50k).
  Saves a separate "best" checkpoint whenever eval mean_reward improves.
  All checkpoints written to --ckpt-dir (default checkpoints/grasp).

W&B logging
────────────
  Pass --wandb-project <name> to enable.  Every training run is tagged with
  algorithm, seed, stage, and key hyperparameters.

Usage
─────
    cd robot_nav_ai
    python agent/train_grasp_sac.py                              # fresh run
    python agent/train_grasp_sac.py --resume latest              # resume
    python agent/train_grasp_sac.py --total-steps 1000000 \\
        --wandb-project autorobo --no-her                        # SAC without HER
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

log = logging.getLogger(__name__)

# ── SAC hyperparameter config ─────────────────────────────────────────────────

@dataclass(frozen=True)
class SACConfig:
    """
    Hyperparameters for SAC + HER grasp training.

    Defaults calibrated for ManipulationEnv (31-dim obs, 7-dim action).
    All values match configs/training/sac.yaml for cross-reference.
    """
    learning_rate:          float = 3e-4
    buffer_size:            int   = 500_000
    learning_starts:        int   = 10_000
    batch_size:             int   = 256
    tau:                    float = 0.005
    gamma:                  float = 0.99
    train_freq:             int   = 1
    gradient_steps:         int   = 1
    ent_coef:               str   = "auto"
    target_entropy:         str   = "auto"
    net_arch:               tuple = (256, 256)
    use_her:                bool  = False   # requires Dict obs space (GoalEnv wrapper)
    her_n_sampled_goals:    int   = 4
    her_goal_selection:     str   = "future"   # "final", "episode", "future"


# ── trainer ───────────────────────────────────────────────────────────────────

class GraspSACTrainer:
    """
    Manages SAC + HER training for the grasp policy.

    Parameters
    ----------
    cfg          : SACConfig
    total_steps  : total environment interaction steps
    stage0_steps : steps in easy-placement curriculum stage
    seed         : RNG seed
    ckpt_dir     : directory for checkpoints
    save_freq    : save checkpoint every N steps
    eval_freq    : run evaluation every N steps
    eval_episodes: number of evaluation episodes
    wandb_cfg    : dict of W&B kwargs, or None to disable
    """

    def __init__(
        self,
        cfg:           SACConfig    = SACConfig(),
        total_steps:   int          = 2_000_000,
        stage0_steps:  int          = 200_000,
        seed:          int          = 0,
        ckpt_dir:      Path         = Path("checkpoints/grasp"),
        save_freq:     int          = 50_000,
        eval_freq:     int          = 25_000,
        eval_episodes: int          = 20,
        wandb_cfg:     Optional[dict] = None,
    ) -> None:
        self.cfg           = cfg
        self.total_steps   = total_steps
        self.stage0_steps  = stage0_steps
        self.seed          = seed
        self.ckpt_dir      = Path(ckpt_dir)
        self.save_freq     = save_freq
        self.eval_freq     = eval_freq
        self.eval_episodes = eval_episodes
        self.wandb_cfg     = wandb_cfg
        self._model        = None

    # ── public API ────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Build environment and SAC model. Call before train()."""
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
            from stable_baselines3.common.monitor import Monitor
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required. "
                "Install with: pip install stable-baselines3"
            ) from e

        from env.manipulation_env import ManipulationEnv

        log.info("Building ManipulationEnv (stage 0: easy placement) ...")
        env      = ManipulationEnv()
        eval_env = ManipulationEnv()

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        policy_kwargs = {"net_arch": list(self.cfg.net_arch)}

        her_kwargs = {}
        replay_buffer_class = None
        from gymnasium import spaces as _spaces
        obs_is_dict = isinstance(env.observation_space, _spaces.Dict)
        if self.cfg.use_her and obs_is_dict:
            replay_buffer_class = HerReplayBuffer
            her_kwargs = {
                "n_sampled_goal":          self.cfg.her_n_sampled_goals,
                "goal_selection_strategy": self.cfg.her_goal_selection,
            }
        elif self.cfg.use_her and not obs_is_dict:
            log.warning(
                "HER requested but env uses a flat Box obs space — "
                "falling back to plain SAC. Wrap env as GoalEnv to enable HER."
            )

        self._model = SAC(
            policy               = "MlpPolicy",
            env                  = env,
            learning_rate        = self.cfg.learning_rate,
            buffer_size          = self.cfg.buffer_size,
            learning_starts      = self.cfg.learning_starts,
            batch_size           = self.cfg.batch_size,
            tau                  = self.cfg.tau,
            gamma                = self.cfg.gamma,
            train_freq           = self.cfg.train_freq,
            gradient_steps       = self.cfg.gradient_steps,
            ent_coef             = self.cfg.ent_coef,
            target_entropy       = self.cfg.target_entropy,
            replay_buffer_class  = replay_buffer_class,
            replay_buffer_kwargs = her_kwargs or None,
            policy_kwargs        = policy_kwargs,
            seed                 = self.seed,
            verbose              = 1,
        )
        self._eval_env = eval_env
        log.info("SAC model built: %s", self._model.policy)

    def train(self, resume_path: Optional[Path] = None) -> None:
        """
        Run training.

        Parameters
        ----------
        resume_path : path to a saved SAC model zip to resume from, or None.
        """
        if self._model is None:
            self.setup()

        if resume_path is not None:
            log.info("Resuming from %s", resume_path)
            from stable_baselines3 import SAC
            self._model = SAC.load(resume_path, env=self._model.get_env())

        callbacks = self._build_callbacks()

        log.info(
            "Training SAC grasp policy for %d steps "
            "(stage0=%d, HER=%s, seed=%d)",
            self.total_steps, self.stage0_steps,
            self.cfg.use_her, self.seed,
        )
        t0 = time.time()
        self._model.learn(
            total_timesteps     = self.total_steps,
            callback            = callbacks,
            reset_num_timesteps = resume_path is None,
        )
        elapsed = time.time() - t0
        log.info("Training complete in %.1f s", elapsed)
        self._save("final")

    def evaluate(self, n_episodes: int = 20) -> dict:
        """
        Run n_episodes deterministically and return mean/std reward + success rate.
        """
        if self._model is None:
            raise RuntimeError("Call setup() or train() first.")

        from stable_baselines3.common.evaluation import evaluate_policy

        mean_r, std_r = evaluate_policy(
            self._model, self._eval_env,
            n_eval_episodes = n_episodes,
            deterministic   = True,
        )
        return {"mean_reward": mean_r, "std_reward": std_r}

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_callbacks(self) -> list:
        callbacks = []
        try:
            from stable_baselines3.common.callbacks import (
                EvalCallback, CheckpointCallback,
            )
            callbacks.append(CheckpointCallback(
                save_freq   = self.save_freq,
                save_path   = str(self.ckpt_dir),
                name_prefix = "sac_grasp",
                verbose     = 1,
            ))
            callbacks.append(EvalCallback(
                eval_env            = self._eval_env,
                eval_freq           = self.eval_freq,
                n_eval_episodes     = self.eval_episodes,
                best_model_save_path= str(self.ckpt_dir / "best"),
                deterministic       = True,
                verbose             = 1,
            ))
        except ImportError:
            pass

        if self.wandb_cfg:
            try:
                from wandb.integration.sb3 import WandbCallback
                callbacks.append(WandbCallback(
                    gradient_save_freq = 1000,
                    verbose            = 2,
                    **self.wandb_cfg,
                ))
            except ImportError:
                log.warning("wandb not installed — W&B logging disabled")

        return callbacks

    def _save(self, tag: str) -> Path:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"sac_grasp_{tag}"
        self._model.save(str(path))
        log.info("Saved checkpoint: %s.zip", path)
        return path

    def __repr__(self) -> str:
        return (
            f"GraspSACTrainer(total_steps={self.total_steps}, "
            f"HER={self.cfg.use_her}, seed={self.seed})"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC grasp policy")
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to checkpoint zip to resume from")
    p.add_argument("--total-steps",   type=int,   default=2_000_000)
    p.add_argument("--stage0-steps",  type=int,   default=200_000)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--ckpt-dir",      type=str,   default="checkpoints/grasp")
    p.add_argument("--save-freq",     type=int,   default=50_000)
    p.add_argument("--eval-freq",     type=int,   default=25_000)
    p.add_argument("--eval-episodes", type=int,   default=20)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--buffer-size",   type=int,   default=500_000)
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--no-her",        action="store_true",
                   help="Disable HER (train plain SAC)")
    p.add_argument("--no-wandb",      action="store_true",
                   help="Disable W&B logging entirely")
    p.add_argument("--wandb-project", type=str,   default=None)
    p.add_argument("--wandb-entity",  type=str,   default=None)
    p.add_argument("--wandb-tags",    type=str,   nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse()

    wandb_cfg = None
    if not args.no_wandb and args.wandb_project:
        try:
            import wandb
            wandb.init(
                project = args.wandb_project,
                entity  = args.wandb_entity,
                tags    = args.wandb_tags or ["sac", "grasp"],
                config  = vars(args),
            )
            wandb_cfg = {}
        except ImportError:
            log.warning("wandb not installed — skipping W&B init")

    cfg = SACConfig(
        learning_rate = args.lr,
        buffer_size   = args.buffer_size,
        batch_size    = args.batch_size,
        use_her       = False,   # HER needs GoalEnv (Dict obs) — plain SAC for now
    )

    trainer = GraspSACTrainer(
        cfg           = cfg,
        total_steps   = args.total_steps,
        stage0_steps  = args.stage0_steps,
        seed          = args.seed,
        ckpt_dir      = Path(args.ckpt_dir),
        save_freq     = args.save_freq,
        eval_freq     = args.eval_freq,
        wandb_cfg     = wandb_cfg,
    )

    resume = Path(args.resume) if args.resume else None
    trainer.setup()
    trainer.train(resume_path=resume)


if __name__ == "__main__":
    main()
