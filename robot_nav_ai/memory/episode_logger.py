"""
episode_logger.py — Episode Logging (Phase 11)

Logs every episode's observations, actions, rewards, and outcomes to disk.
Provides data for:
  - Post-hoc analysis and debugging
  - Self-improvement (SelfImprover identifies failure episodes)
  - Demonstration dataset creation (successful episodes)
  - W&B artifact upload for experiment tracking

Logs are stored as HDF5 files, one per episode, with episode-level
metadata (success, total_reward, failure_mode) in JSON.

Usage:
    from memory.episode_logger import EpisodeLogger

    logger = EpisodeLogger(log_dir="logs/episodes")
    logger.begin_episode(episode_id=42, task_id="pick_mug_001")
    logger.log_step(obs, action, reward, done, info)
    ...
    logger.end_episode(success=True, metadata={"grasp_attempts": 2})
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class EpisodeLogger:
    """
    Logs episode data for offline analysis and self-improvement.

    Each episode produces:
      - <episode_id>.h5: HDF5 file with step-level data
      - <episode_id>_meta.json: Episode metadata

    The HDF5 structure:
      /obs/rgb       (T, H, W, 3) uint8
      /obs/depth     (T, H, W) float32
      /obs/lidar     (T, N) float32
      /obs/proprio   (T, D) float32
      /actions       (T, A) float32
      /rewards       (T,) float32
      /dones         (T,) bool
    """

    def __init__(
        self,
        log_dir: str | Path = "logs/episodes",
        save_images: bool = True,
    ) -> None:
        """
        Initialise the episode logger.

        Args:
            log_dir: Directory for episode log files.
            save_images: If True, save RGB and depth images. Set False to save disk.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images

        self._current_episode_id: int | None = None
        self._current_task_id: str | None = None
        self._episode_start_time: float | None = None
        self._steps: list[dict[str, Any]] = []
        self._episode_open = False

        log.info(f"EpisodeLogger initialised. Log dir: {self.log_dir}")

    def begin_episode(
        self,
        episode_id: int,
        task_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Begin logging a new episode.

        Args:
            episode_id: Unique episode identifier.
            task_id: Task identifier from TaskGraph.
            metadata: Optional initial metadata (instruction, world state, etc.).

        Raises:
            RuntimeError: If an episode is already open (end_episode not called).
        """
        if self._episode_open:
            raise RuntimeError(
                f"Episode {self._current_episode_id} is still open. "
                "Call end_episode() before starting a new one."
            )

        self._current_episode_id = episode_id
        self._current_task_id = task_id
        self._episode_start_time = time.time()
        self._steps = []
        self._episode_open = True

        log.debug(f"Episode {episode_id} logging started (task: {task_id})")

    def log_step(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """
        Log a single environment step.

        Args:
            obs: Observation dict (rgb, depth, lidar, proprioception).
            action: Action array applied this step.
            reward: Reward received this step.
            done: Whether the episode ended after this step.
            info: Info dict from the environment step.

        Raises:
            RuntimeError: If no episode is currently open.
        """
        if not self._episode_open:
            raise RuntimeError(
                "No episode is open. Call begin_episode() first."
            )

        step_data: dict[str, Any] = {
            "action": action,
            "reward": float(reward),
            "done": bool(done),
            "info": info,
        }

        if self.save_images:
            step_data["obs_rgb"] = obs.get("rgb")
            step_data["obs_depth"] = obs.get("depth")

        step_data["obs_lidar"] = obs.get("lidar")
        step_data["obs_proprio"] = obs.get("proprioception")

        self._steps.append(step_data)

    def end_episode(
        self,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Finalise and save the episode to disk.

        Args:
            success: Whether the episode succeeded.
            metadata: Optional metadata to include in the JSON (failure mode, etc.).

        Returns:
            Path to the saved HDF5 episode file.

        Raises:
            RuntimeError: If no episode is open.

        TODO: Phase 11 — implement HDF5 save using h5py:
            import h5py
            with h5py.File(episode_path, "w") as f:
                obs_grp = f.create_group("obs")
                obs_grp.create_dataset("rgb", data=stacked_rgb)
                obs_grp.create_dataset("proprio", data=stacked_proprio)
                f.create_dataset("actions", data=stacked_actions)
                f.create_dataset("rewards", data=stacked_rewards)
                f.create_dataset("dones", data=stacked_dones)
        """
        if not self._episode_open:
            raise RuntimeError("No episode is open. Call begin_episode() first.")

        episode_duration = time.time() - self._episode_start_time
        n_steps = len(self._steps)

        # Build metadata
        meta = {
            "episode_id": self._current_episode_id,
            "task_id": self._current_task_id,
            "success": success,
            "n_steps": n_steps,
            "duration_seconds": episode_duration,
            "total_reward": sum(s["reward"] for s in self._steps),
            "timestamp": self._episode_start_time,
        }
        if metadata:
            meta.update(metadata)

        # Save metadata JSON (always — even if HDF5 save fails)
        meta_path = self.log_dir / f"episode_{self._current_episode_id:06d}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        episode_path = self.log_dir / f"episode_{self._current_episode_id:06d}.h5"

        log.debug(
            f"Episode {self._current_episode_id} ended: "
            f"success={success}, steps={n_steps}, "
            f"reward={meta['total_reward']:.2f}"
        )

        self._episode_open = False
        self._steps = []

        raise NotImplementedError(
            f"TODO: Phase 11 — implement HDF5 save to {episode_path} using h5py. "
            "Stack step arrays and write datasets for obs, actions, rewards, dones."
        )

    def get_episode_count(self) -> int:
        """Return the number of completed episode files in the log directory."""
        return len(list(self.log_dir.glob("episode_*_meta.json")))

    def load_episode_metadata(self, episode_id: int) -> dict[str, Any]:
        """
        Load the metadata for a specific episode.

        Args:
            episode_id: Episode ID to load.

        Returns:
            Metadata dict from the episode's JSON file.

        Raises:
            FileNotFoundError: If the episode doesn't exist.
        """
        meta_path = self.log_dir / f"episode_{episode_id:06d}_meta.json"
        with open(meta_path) as f:
            return json.load(f)

    @property
    def is_open(self) -> bool:
        """Return True if an episode is currently being logged."""
        return self._episode_open
