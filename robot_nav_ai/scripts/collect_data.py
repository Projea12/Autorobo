"""
collect_data.py — Demonstration Data Collection (Phase 7)

Collects human teleoperation demonstrations or scripted oracle demonstrations
for seeding the SAC replay buffer in grasp policy training.

Supported collection modes:
  - keyboard: WASD + gripper keys via pygame
  - spacemouse: 6-DOF SpaceMouse via pyspacemouse
  - oracle: scripted heuristic oracle (for automated data generation)
  - replay: replay existing demonstrations for validation

Demonstrations are saved as HDF5 files containing:
  - observations (rgb, depth, proprioception)
  - actions (joint velocities or end-effector deltas)
  - rewards
  - episode metadata

Usage:
    python scripts/collect_data.py mode=oracle n_demos=1000
    python scripts/collect_data.py mode=keyboard output_dir=data/demos
    python scripts/collect_data.py mode=spacemouse
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@dataclass
class Episode:
    """Container for a single demonstration episode."""
    observations: list[dict[str, Any]] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    n_steps: int = 0
    total_reward: float = 0.0


def collect_keyboard_demo(env, cfg: DictConfig) -> Optional[Episode]:
    """
    Collect a single demonstration via keyboard teleoperation.

    Args:
        env: The gymnasium environment to interact with.
        cfg: Data collection config.

    Returns:
        Episode if completed successfully, None if user aborted.

    TODO: Phase 7 — implement pygame window with key bindings:
      W/S: forward/backward, A/D: rotate, Q/E: arm up/down,
      Space: open/close gripper, R: reset, ESC: abort.
    """
    raise NotImplementedError(
        "TODO: Phase 7 — implement keyboard teleoperation using pygame. "
        "Render the env in a pygame window, capture key events, map to actions."
    )


def collect_oracle_demo(env, cfg: DictConfig) -> Optional[Episode]:
    """
    Collect a single demonstration using a scripted oracle policy.

    The oracle uses ground-truth object poses from the simulator to
    compute deterministic grasp trajectories. Used for fast bulk collection.

    Args:
        env: The gymnasium environment (must expose ground-truth state).
        cfg: Data collection config.

    Returns:
        Episode if oracle succeeded, None if max steps exceeded.

    TODO: Phase 7 — implement oracle:
      1. Query object pose from MuJoCo (mj_data.body_xpos).
      2. Plan straight-line approach to pre-grasp pose.
      3. Move arm along planned trajectory (joint interpolation).
      4. Close gripper.
      5. Lift and move to place position.
    """
    raise NotImplementedError(
        "TODO: Phase 7 — implement scripted oracle using MuJoCo ground-truth state. "
        "Target success rate: >90% for simple tabletop objects."
    )


def save_episode(episode: Episode, output_dir: Path, episode_idx: int) -> Path:
    """
    Save a demonstration episode to HDF5 format.

    Args:
        episode: The collected episode data.
        output_dir: Directory to write the file.
        episode_idx: Episode index (used in filename).

    Returns:
        Path to the saved file.

    TODO: Phase 7 — use h5py to write each field as a dataset,
    add metadata attrs (success, n_steps, total_reward, timestamp).
    """
    raise NotImplementedError(
        "TODO: Phase 7 — save episode to HDF5 using h5py. "
        "Schema: /obs/rgb, /obs/depth, /obs/proprio, /actions, /rewards, /dones."
    )


def load_and_validate_demos(demo_dir: Path) -> dict[str, Any]:
    """
    Load all demos from a directory and compute summary statistics.

    Args:
        demo_dir: Directory containing .h5 episode files.

    Returns:
        Dict with: n_demos, success_rate, avg_steps, avg_reward.

    TODO: Phase 7 — iterate .h5 files, load metadata, aggregate stats.
    """
    raise NotImplementedError(
        "TODO: Phase 7 — implement demo validation and stats computation."
    )


@hydra.main(config_path="../configs/hydra", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main data collection entry point.

    Args:
        cfg: Composed Hydra config. Key overrides:
          - collect.mode: keyboard | oracle | spacemouse
          - collect.n_demos: number of demos to collect
          - collect.output_dir: where to save demos
          - collect.success_only: only save successful episodes
    """
    log.info("=== AutoRobo Demonstration Data Collection (Phase 7) ===")

    mode = cfg.get("collect", {}).get("mode", "oracle")
    n_demos = cfg.get("collect", {}).get("n_demos", 100)
    output_dir = Path(cfg.project.data_dir) / "demonstrations"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Collection mode: {mode}, target: {n_demos} demos")
    log.info(f"Output directory: {output_dir}")

    raise NotImplementedError(
        f"TODO: Phase 7 — create env, run {n_demos} collection loops using "
        f"collect_{mode}_demo(), save with save_episode(), log summary stats."
    )


if __name__ == "__main__":
    main()
