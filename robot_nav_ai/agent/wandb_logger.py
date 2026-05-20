"""
agent/wandb_logger.py — Weights & Biases logging wrapper for nav_train.

Design goals
────────────
  • Fully optional  — if wandb is not installed or the run is disabled,
    every method is a silent no-op; training is unaffected.
  • Single init call logs ALL hyperparameters as wandb.config so every
    metric chart is filterable by hparam in the W&B UI.
  • Resumable runs  — the wandb run_id is returned from init() and stored
    in checkpoints; pass it back via resume_run_id to continue the same
    W&B run after a crash or checkpoint restart.
  • Structured metric groups — all keys use a "group/name" convention so
    the W&B dashboard auto-groups them into panels.

Logged hyperparameters (at init)
─────────────────────────────────
  From NavTrainConfig   : total_steps, stage0_steps, max_episode_steps,
                          n_substeps, seed
  From PPOConfig        : n_steps, gamma, gae_lambda, n_epochs, batch_size,
                          lr, clip_eps, vf_coef, ent_coef, max_grad_norm,
                          hidden_sizes, log_std_init, normalize_adv, target_kl
  From RewardConfig (stage 0 & 1): all fields, prefixed reward_s0_ / reward_s1_
  Derived               : obs_dim, act_dim

Metrics logged per log_interval steps
───────────────────────────────────────
  train/mean_reward      rolling-20 episode mean return
  train/mean_ep_len      rolling-20 episode length (all episodes)
  train/success_rate     rolling-20 fraction of successful episodes
  train/fps              environment steps per wall-clock second
  curriculum/stage       0 or 1 (for curriculum switch visibility)
  loss/policy            mean PPO policy-gradient loss
  loss/value             mean value-function MSE loss
  loss/entropy           mean policy entropy
  loss/total             weighted sum loss

Metrics logged per save_interval steps
────────────────────────────────────────
  checkpoint/mean_reward   reward at checkpoint time
  checkpoint/global_step   step index
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

# Module-level import so tests can patch `agent.wandb_logger.wandb`.
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # type: ignore


# ── W&B configuration ─────────────────────────────────────────────────────────

@dataclasses.dataclass
class WandbConfig:
    """
    Controls how the W&B run is initialised.

    project        : W&B project name
    entity         : W&B team / username (None → personal account default)
    group          : group name for grouping related runs in the UI
    tags           : list of string tags
    mode           : "online" | "offline" | "disabled"
    resume_run_id  : existing W&B run ID to resume (set automatically on
                     checkpoint resume — don't set this manually)
    """
    project:       str            = "autorobo-nav"
    entity:        Optional[str]  = None
    group:         Optional[str]  = None
    tags:          tuple[str, ...] = ()
    mode:          str            = "online"   # "disabled" skips W&B entirely
    resume_run_id: Optional[str]  = None


# ── logger ────────────────────────────────────────────────────────────────────

class WandbLogger:
    """
    Thin wrapper around wandb that degrades gracefully when wandb is absent.

    All public methods are safe to call unconditionally — they become no-ops
    when W&B is disabled or not installed.

    Parameters
    ----------
    wandb_cfg : WandbConfig
    """

    def __init__(self, wandb_cfg: WandbConfig) -> None:
        self._cfg      = wandb_cfg
        self._run      = None
        self._enabled  = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def init(
        self,
        run_name:      str,
        hparams:       Dict[str, Any],
        resume_run_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Initialise a W&B run and log all hyperparameters.

        Parameters
        ----------
        run_name      : human-readable run name shown in the W&B UI
        hparams       : flat dict of all hyperparameters (logged as wandb.config)
        resume_run_id : W&B run ID to resume; overrides WandbConfig.resume_run_id

        Returns the wandb run ID (str) if successful, None otherwise.
        """
        if self._cfg.mode == "disabled":
            return None

        if wandb is None:
            print("[wandb_logger] wandb not installed — W&B logging disabled.")
            return None

        run_id = resume_run_id or self._cfg.resume_run_id

        try:
            self._run = wandb.init(
                project  = self._cfg.project,
                entity   = self._cfg.entity,
                name     = run_name,
                group    = self._cfg.group,
                tags     = list(self._cfg.tags),
                mode     = self._cfg.mode,
                id       = run_id,
                resume   = "allow" if run_id else None,
                config   = hparams,
            )
            self._enabled = True
            print(f"[wandb_logger] run started: {self._run.url}")
            return self._run.id
        except Exception as exc:
            print(f"[wandb_logger] W&B init failed ({exc}) — continuing without W&B.")
            return None

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log a dict of scalar metrics at the given global step."""
        if not self._enabled or self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        """Mark the W&B run as finished."""
        if not self._enabled or self._run is None:
            return
        try:
            self._run.finish()
        except Exception:
            pass

    @property
    def run_id(self) -> Optional[str]:
        """The active W&B run ID, or None if W&B is not running."""
        if self._run is None:
            return None
        return getattr(self._run, "id", None)

    @property
    def run_url(self) -> Optional[str]:
        """The W&B run URL for quick browser access, or None."""
        if self._run is None:
            return None
        return getattr(self._run, "url", None)

    @property
    def enabled(self) -> bool:
        return self._enabled


# ── hyperparameter extraction ─────────────────────────────────────────────────

def build_hparams(
    train_cfg,       # NavTrainConfig
    reward_s0,       # RewardConfig
    reward_s1,       # RewardConfig
    obs_dim: int,
    act_dim: int,
) -> Dict[str, Any]:
    """
    Flatten all training knobs into a single dict for wandb.config.

    Keys follow a "group/name" pattern matching the metric names so that
    W&B's parallel-coordinates plot works out of the box.
    """
    ppo = train_cfg.ppo

    hparams: Dict[str, Any] = {
        # budget
        "budget/total_steps":       train_cfg.total_steps,
        "budget/stage0_steps":      train_cfg.stage0_steps,
        # env
        "env/max_episode_steps":    train_cfg.max_episode_steps,
        "env/n_substeps":           train_cfg.n_substeps,
        "env/seed":                 train_cfg.seed,
        "env/obs_dim":              obs_dim,
        "env/act_dim":              act_dim,
        # PPO
        "ppo/n_steps":              ppo.n_steps,
        "ppo/gamma":                ppo.gamma,
        "ppo/gae_lambda":           ppo.gae_lambda,
        "ppo/n_epochs":             ppo.n_epochs,
        "ppo/batch_size":           ppo.batch_size,
        "ppo/lr":                   ppo.lr,
        "ppo/clip_eps":             ppo.clip_eps,
        "ppo/vf_coef":              ppo.vf_coef,
        "ppo/ent_coef":             ppo.ent_coef,
        "ppo/max_grad_norm":        ppo.max_grad_norm,
        "ppo/hidden_sizes":         str(ppo.hidden_sizes),
        "ppo/log_std_init":         ppo.log_std_init,
        "ppo/normalize_adv":        ppo.normalize_adv,
        "ppo/target_kl":            ppo.target_kl,
    }

    # stage-0 reward config
    for f in dataclasses.fields(reward_s0):
        hparams[f"reward_s0/{f.name}"] = getattr(reward_s0, f.name)

    # stage-1 reward config
    for f in dataclasses.fields(reward_s1):
        hparams[f"reward_s1/{f.name}"] = getattr(reward_s1, f.name)

    return hparams
