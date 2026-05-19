"""
agent/train.py — PPO training entry-point for AutoRobo v1.

Supports full resume from any checkpoint:

    python -m agent.train                          # fresh run
    python -m agent.train --resume latest          # resume most recent
    python -m agent.train --resume best            # resume best eval checkpoint
    python -m agent.train --resume step_0000050000 # resume specific step

Phase 1 stack
─────────────
  Env        : ManipulationEnv (MuJoCo 3, OBS_DIM=45, ACT_DIM=8)
  Reset      : EpisodeResetter — randomise_obstacles + spawn + goal
  Domain rand: DomainRandomizer — lighting / friction / color / mass / size
  Checkpoints: CheckpointManager — rolling keep-last-5, best slot, meta.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from env import ManipulationEnv
from utils.checkpoint import CheckpointManager, SB3CheckpointCallback, make_run_dir

# ── paths ─────────────────────────────────────────────────────────────────────
RUNS_DIR  = _ROOT / "runs"
LOGS_DIR  = _ROOT / "logs"


# ── default hyper-parameters ──────────────────────────────────────────────────

DEFAULTS = dict(
    total_timesteps = 2_000_000,
    n_envs          = 4,
    learning_rate   = 3e-4,
    n_steps         = 2048,
    batch_size      = 64,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    save_freq       = 50_000,
    eval_freq       = 25_000,
    keep_last       = 5,
    run_name        = "ppo_manipulation",
    device          = "auto",
)


# ── training ──────────────────────────────────────────────────────────────────

def train(
    *,
    resume:          str | None  = None,
    total_timesteps: int         = DEFAULTS["total_timesteps"],
    n_envs:          int         = DEFAULTS["n_envs"],
    learning_rate:   float       = DEFAULTS["learning_rate"],
    n_steps:         int         = DEFAULTS["n_steps"],
    batch_size:      int         = DEFAULTS["batch_size"],
    n_epochs:        int         = DEFAULTS["n_epochs"],
    gamma:           float       = DEFAULTS["gamma"],
    gae_lambda:      float       = DEFAULTS["gae_lambda"],
    clip_range:      float       = DEFAULTS["clip_range"],
    ent_coef:        float       = DEFAULTS["ent_coef"],
    save_freq:       int         = DEFAULTS["save_freq"],
    eval_freq:       int         = DEFAULTS["eval_freq"],
    keep_last:       int         = DEFAULTS["keep_last"],
    run_name:        str         = DEFAULTS["run_name"],
    device:          str         = DEFAULTS["device"],
) -> None:
    """
    Run (or resume) a PPO training session on ManipulationEnv.

    Parameters
    ----------
    resume : None → fresh run; "latest"/"best"/"step_XXXXXXXXXX" → resume
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── locate run directory ──────────────────────────────────────────────────
    if resume is not None:
        # Find the most recently modified run that has a checkpoints/ subdir
        run_dir = _find_latest_run(RUNS_DIR, run_name)
        if run_dir is None:
            raise RuntimeError(
                f"No existing run matching '{run_name}' found in {RUNS_DIR}. "
                "Start a fresh run first (omit --resume)."
            )
        print(f"[train] Resuming from run: {run_dir}")
    else:
        run_dir = make_run_dir(RUNS_DIR, run_name)
        print(f"[train] New run: {run_dir}")

    mgr = CheckpointManager(run_dir, keep_last=keep_last)

    # ── environments ─────────────────────────────────────────────────────────
    env      = make_vec_env(ManipulationEnv, n_envs=n_envs)
    eval_env = make_vec_env(ManipulationEnv, n_envs=1)

    # ── model ─────────────────────────────────────────────────────────────────
    reset_num_timesteps = True
    start_meta: dict    = {}

    if resume is not None:
        model, start_meta = mgr.resume_sb3(PPO, env, ckpt_name=resume, device=device)
        reset_num_timesteps = False
        resumed_step = start_meta.get("step", 0)
        print(f"[train] Resumed at step {resumed_step:,}  episode {start_meta.get('episode', '?')}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose              = 1,
            tensorboard_log      = str(LOGS_DIR),
            learning_rate        = learning_rate,
            n_steps              = n_steps,
            batch_size           = batch_size,
            n_epochs             = n_epochs,
            gamma                = gamma,
            gae_lambda           = gae_lambda,
            clip_range           = clip_range,
            ent_coef             = ent_coef,
            device               = device,
        )

    # ── callbacks ─────────────────────────────────────────────────────────────
    ckpt_cb = SB3CheckpointCallback(mgr, save_freq=save_freq, verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(mgr.ckpt_root / "best"),
        log_path             = str(run_dir / "eval_logs"),
        eval_freq            = eval_freq,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    # ── train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = total_timesteps,
        callback            = [ckpt_cb._cb, eval_cb],
        reset_num_timesteps = reset_num_timesteps,
        progress_bar        = True,
    )

    # ── final save ────────────────────────────────────────────────────────────
    final_step = model.num_timesteps
    final_path = mgr.save_sb3(model, step=final_step, extra={"final": True})
    print(f"[train] Training complete. Final checkpoint: {final_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_latest_run(runs_dir: Path, run_name: str) -> Path | None:
    """Return the most recently modified run directory matching run_name."""
    if not runs_dir.exists():
        return None
    candidates = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith(run_name)
        and (d / "checkpoints").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AutoRobo v1 manipulation policy")
    p.add_argument("--resume", default=None,
                   help='Checkpoint to resume: "latest", "best", or "step_XXXXXXXXXX"')
    p.add_argument("--timesteps", type=int,    default=DEFAULTS["total_timesteps"])
    p.add_argument("--n-envs",   type=int,    default=DEFAULTS["n_envs"])
    p.add_argument("--lr",       type=float,  default=DEFAULTS["learning_rate"])
    p.add_argument("--save-freq", type=int,   default=DEFAULTS["save_freq"])
    p.add_argument("--eval-freq", type=int,   default=DEFAULTS["eval_freq"])
    p.add_argument("--keep-last", type=int,   default=DEFAULTS["keep_last"])
    p.add_argument("--run-name", default=DEFAULTS["run_name"])
    p.add_argument("--device",   default=DEFAULTS["device"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        resume          = args.resume,
        total_timesteps = args.timesteps,
        n_envs          = args.n_envs,
        learning_rate   = args.lr,
        save_freq       = args.save_freq,
        eval_freq       = args.eval_freq,
        keep_last       = args.keep_last,
        run_name        = args.run_name,
        device          = args.device,
    )
