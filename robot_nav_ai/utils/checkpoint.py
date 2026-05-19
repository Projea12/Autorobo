"""
utils/checkpoint.py — Model checkpoint management for AutoRobo v1.

Provides a CheckpointManager that:
  - Saves model + optimizer state + training metadata every N steps
  - Keeps the K most recent checkpoints (plus a permanent "best" slot)
  - Exposes a resume() helper that loads the latest (or a named) checkpoint
  - Works with Stable-Baselines3 models and raw PyTorch state_dicts alike

Usage (SB3)
───────────
    from utils.checkpoint import CheckpointManager, SB3CheckpointCallback

    mgr = CheckpointManager(run_dir="runs/exp_001", keep_last=5)
    cb  = SB3CheckpointCallback(mgr, save_freq=50_000)

    model = PPO(...)
    model.learn(..., callback=cb)

    # resume
    model, meta = mgr.resume(PPO, env)

Usage (raw PyTorch)
───────────────────
    mgr.save_torch(step=t, actor=actor, critic=critic,
                   actor_opt=opt_a, critic_opt=opt_c,
                   extra={"episode": ep, "best_return": r})
    ckpt = mgr.load_torch("latest")
    actor.load_state_dict(ckpt["actor"])
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _meta_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "meta.json"


def _write_meta(ckpt_dir: Path, meta: dict) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(ckpt_dir).write_text(json.dumps(meta, indent=2))


def _read_meta(ckpt_dir: Path) -> dict:
    p = _meta_path(ckpt_dir)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ── main class ────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages a directory of numbered checkpoints with rotation.

    Directory layout
    ────────────────
    <run_dir>/
      checkpoints/
        step_0000050000/
          model.zip          ← SB3 model (or omitted for raw PyTorch)
          torch.pt           ← raw PyTorch state dicts (or omitted)
          meta.json          ← step, wall_time, episode, extra fields
        step_0000100000/
          ...
        best/
          model.zip
          meta.json
      latest_step.txt        ← symlink target for quick lookup

    Parameters
    ----------
    run_dir   : root directory for this training run
    keep_last : number of rolling checkpoints to retain (best is always kept)
    """

    def __init__(self, run_dir: str | Path, keep_last: int = 5) -> None:
        self.run_dir   = Path(run_dir)
        self.ckpt_root = self.run_dir / "checkpoints"
        self.keep_last = keep_last
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

    # ── SB3 interface ─────────────────────────────────────────────────────────

    def save_sb3(
        self,
        model,
        step:    int,
        episode: int  = 0,
        extra:   dict = {},
        is_best: bool = False,
    ) -> Path:
        """
        Save an SB3 model (PPO, SAC, …).

        SB3's model.save() already serialises the policy network weights AND
        the optimiser state (Adam moments) inside a single .zip archive.
        We add a meta.json alongside it for step/episode/timing bookkeeping.

        Parameters
        ----------
        model   : SB3 BaseAlgorithm
        step    : total environment steps so far
        episode : total episodes completed
        extra   : arbitrary JSON-serialisable fields to store in meta
        is_best : if True also copies to <ckpt_root>/best/

        Returns
        -------
        Path to the checkpoint directory that was written.
        """
        ckpt_dir = self.ckpt_root / f"step_{step:010d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model.save(str(ckpt_dir / "model"))   # SB3 appends .zip

        meta = {"step": step, "episode": episode,
                 "wall_time": time.time(), **extra}
        _write_meta(ckpt_dir, meta)

        (self.run_dir / "latest_step.txt").write_text(str(step))

        if is_best:
            best_dir = self.ckpt_root / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)

        self._rotate()
        return ckpt_dir

    def resume_sb3(
        self,
        algo_cls,
        env,
        ckpt_name: str = "latest",
        device:    str = "auto",
    ):
        """
        Load an SB3 model from a checkpoint for continued training or eval.

        Parameters
        ----------
        algo_cls  : SB3 algorithm class (e.g. PPO, SAC)
        env       : the Gymnasium env to attach
        ckpt_name : "latest", "best", or "step_XXXXXXXXXX"
        device    : torch device string

        Returns
        -------
        (model, meta_dict)  — model ready for .learn() or .predict()
        """
        ckpt_dir = self._resolve(ckpt_name)
        model    = algo_cls.load(str(ckpt_dir / "model"), env=env, device=device)
        meta     = _read_meta(ckpt_dir)
        return model, meta

    # ── raw PyTorch interface ─────────────────────────────────────────────────

    def save_torch(
        self,
        step:    int,
        episode: int  = 0,
        extra:   dict = {},
        is_best: bool = False,
        **state_dicts,
    ) -> Path:
        """
        Save arbitrary PyTorch state_dicts.

        Parameters
        ----------
        step        : total environment steps
        episode     : total episodes
        extra       : JSON-serialisable metadata
        is_best     : copy to best/ slot
        **state_dicts : name → state_dict (e.g. actor=actor.state_dict(),
                         critic_opt=opt.state_dict())

        Returns
        -------
        Path to the checkpoint directory.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for save_torch()") from exc

        ckpt_dir = self.ckpt_root / f"step_{step:010d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(state_dicts, ckpt_dir / "torch.pt")

        meta = {"step": step, "episode": episode,
                 "wall_time": time.time(), **extra}
        _write_meta(ckpt_dir, meta)

        (self.run_dir / "latest_step.txt").write_text(str(step))

        if is_best:
            best_dir = self.ckpt_root / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)

        self._rotate()
        return ckpt_dir

    def load_torch(self, ckpt_name: str = "latest", map_location: str = "cpu") -> dict:
        """
        Load a raw PyTorch checkpoint.

        Returns
        -------
        dict with all saved state_dicts plus a "meta" key.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for load_torch()") from exc

        ckpt_dir = self._resolve(ckpt_name)
        pt_path  = ckpt_dir / "torch.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"No torch.pt in {ckpt_dir}")

        state = torch.load(pt_path, map_location=map_location, weights_only=True)
        state["meta"] = _read_meta(ckpt_dir)
        return state

    # ── listing / info ────────────────────────────────────────────────────────

    def list_checkpoints(self) -> list[dict]:
        """
        Return metadata for every checkpoint sorted by step (ascending).

        Each entry has at least: step, wall_time, path.
        """
        results = []
        for d in sorted(self.ckpt_root.iterdir()):
            if not d.is_dir() or d.name == "best":
                continue
            meta = _read_meta(d)
            meta["path"] = str(d)
            results.append(meta)
        return results

    def latest_step(self) -> Optional[int]:
        """Return the step number of the most recent checkpoint, or None."""
        p = self.run_dir / "latest_step.txt"
        if not p.exists():
            return None
        return int(p.read_text().strip())

    # ── internal ──────────────────────────────────────────────────────────────

    def _resolve(self, ckpt_name: str) -> Path:
        """Resolve "latest", "best", or a directory name to a Path."""
        if ckpt_name == "best":
            d = self.ckpt_root / "best"
            if not d.exists():
                raise FileNotFoundError("No 'best' checkpoint saved yet.")
            return d

        if ckpt_name == "latest":
            candidates = sorted(
                (d for d in self.ckpt_root.iterdir()
                 if d.is_dir() and d.name.startswith("step_")),
                key=lambda d: d.name,
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No step checkpoints found in {self.ckpt_root}."
                )
            return candidates[-1]

        d = self.ckpt_root / ckpt_name
        if not d.exists():
            raise FileNotFoundError(f"Checkpoint not found: {d}")
        return d

    def _rotate(self) -> None:
        """Delete oldest checkpoints beyond keep_last."""
        candidates = sorted(
            d for d in self.ckpt_root.iterdir()
            if d.is_dir() and d.name.startswith("step_")
        )
        excess = len(candidates) - self.keep_last
        for old in candidates[:excess]:
            shutil.rmtree(old, ignore_errors=True)


# ── SB3 callback ─────────────────────────────────────────────────────────────

class SB3CheckpointCallback:
    """
    Stable-Baselines3 BaseCallback that saves via CheckpointManager.

    Compatible with SB3's callback interface — pass to model.learn().

    Parameters
    ----------
    manager       : CheckpointManager instance
    save_freq     : save every this many environment steps
    eval_callback : optional SB3 EvalCallback; if provided, is_best is
                    set when eval_callback.best_mean_reward was just updated
    verbose       : 0 = silent, 1 = log saves
    """

    def __init__(
        self,
        manager:    CheckpointManager,
        save_freq:  int = 50_000,
        verbose:    int = 1,
    ) -> None:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise ImportError("stable-baselines3 is required for SB3CheckpointCallback") from exc

        self._mgr       = manager
        self._save_freq = save_freq
        self._verbose   = verbose
        self._last_save = 0
        self._episode   = 0
        self._BaseCallback = BaseCallback

        # Build the real SB3 callback lazily (needs self to be fully initialised)
        self._cb = self._build()

    def _build(self):
        mgr        = self._mgr
        save_freq  = self._save_freq
        verbose    = self._verbose
        outer      = self

        class _Inner(self._BaseCallback):
            def __init__(self):
                super().__init__(verbose=verbose)

            def _on_step(self) -> bool:
                step = self.num_timesteps
                if step - outer._last_save >= save_freq:
                    outer._last_save = step
                    path = mgr.save_sb3(
                        self.model,
                        step    = step,
                        episode = outer._episode,
                        extra   = {"n_envs": self.training_env.num_envs},
                    )
                    if verbose:
                        print(f"[CheckpointManager] saved step {step:,} → {path}")
                return True

            def _on_rollout_end(self) -> None:
                outer._episode += getattr(self.training_env, "num_envs", 1)

        return _Inner()

    def __getattr__(self, name):
        return getattr(self._cb, name)


# ── convenience factory ───────────────────────────────────────────────────────

def make_run_dir(base: str | Path, run_name: str) -> Path:
    """
    Return <base>/<run_name>_<timestamp>/, creating it.

    Using a timestamp suffix avoids accidental overwrite when re-running with
    the same name.
    """
    ts      = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
