"""
workspace_limits.py — Robot Arm Workspace Boundaries (Phase 2)

Defines the reachable workspace of the robot arm in Cartesian space.
Used by GraspPlanner and ArmController to validate waypoints before execution.

The workspace is defined as an axis-aligned bounding box in the robot base frame,
configurable via configs/robot/base.yaml (robot.workspace section).

Usage:
    from manipulation.workspace_limits import WorkspaceLimits

    limits = WorkspaceLimits(cfg.robot.workspace)
    if limits.contains(np.array([0.4, 0.0, 0.8])):
        print("Position is reachable")
    clipped = limits.clip(target_position)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Default workspace (metres, robot base frame) — matches base.yaml defaults
DEFAULT_WORKSPACE = {
    "x_min": 0.2, "x_max": 0.8,
    "y_min": -0.4, "y_max": 0.4,
    "z_min": 0.0,  "z_max": 1.2,
}


class WorkspaceLimits:
    """
    Defines and validates the robot arm's reachable Cartesian workspace.

    The workspace is modelled as an axis-aligned bounding box (AABB).
    This is an approximation — the actual reachable workspace is more complex
    (it depends on arm kinematics). The AABB is conservative: all positions
    within the box should be reachable by a properly calibrated IK solver.

    For more accurate reachability, use the IK solver's feasibility check
    in GraspPlanner._solve_ik().
    """

    def __init__(self, cfg: Any = None) -> None:
        """
        Initialise workspace limits from config or defaults.

        Args:
            cfg: DictConfig with x_min, x_max, y_min, y_max, z_min, z_max.
                 Falls back to DEFAULT_WORKSPACE if None.
        """
        if cfg is None:
            limits = DEFAULT_WORKSPACE
            log.warning(
                "No workspace config provided — using DEFAULT_WORKSPACE. "
                "Set robot.workspace in configs/robot/base.yaml."
            )
        else:
            limits = {
                "x_min": float(cfg.x_min),
                "x_max": float(cfg.x_max),
                "y_min": float(cfg.y_min),
                "y_max": float(cfg.y_max),
                "z_min": float(cfg.z_min),
                "z_max": float(cfg.z_max),
            }

        self.x_min = limits["x_min"]
        self.x_max = limits["x_max"]
        self.y_min = limits["y_min"]
        self.y_max = limits["y_max"]
        self.z_min = limits["z_min"]
        self.z_max = limits["z_max"]

        log.debug(
            f"WorkspaceLimits: x=[{self.x_min}, {self.x_max}], "
            f"y=[{self.y_min}, {self.y_max}], "
            f"z=[{self.z_min}, {self.z_max}]"
        )

    def contains(self, position: np.ndarray) -> bool:
        """
        Check if a 3D position is within the workspace AABB.

        Args:
            position: 3D position [x, y, z] in robot base frame (metres). Shape (3,).

        Returns:
            True if position is within all workspace bounds.
        """
        x, y, z = float(position[0]), float(position[1]), float(position[2])
        return (
            self.x_min <= x <= self.x_max and
            self.y_min <= y <= self.y_max and
            self.z_min <= z <= self.z_max
        )

    def clip(self, position: np.ndarray) -> np.ndarray:
        """
        Clip a position to the nearest valid workspace point.

        Args:
            position: Target 3D position. Shape (3,).

        Returns:
            Clipped position within workspace bounds. Shape (3,).
        """
        return np.array([
            np.clip(position[0], self.x_min, self.x_max),
            np.clip(position[1], self.y_min, self.y_max),
            np.clip(position[2], self.z_min, self.z_max),
        ], dtype=np.float32)

    def random_position(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Sample a random valid position within the workspace.

        Args:
            rng: Optional numpy random generator. Creates a new one if None.

        Returns:
            Random 3D position within workspace. Shape (3,).
        """
        if rng is None:
            rng = np.random.default_rng()
        return np.array([
            rng.uniform(self.x_min, self.x_max),
            rng.uniform(self.y_min, self.y_max),
            rng.uniform(self.z_min, self.z_max),
        ], dtype=np.float32)

    @property
    def centre(self) -> np.ndarray:
        """Return the centre of the workspace AABB. Shape (3,)."""
        return np.array([
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        ], dtype=np.float32)

    @property
    def volume(self) -> float:
        """Return the workspace volume in cubic metres."""
        return (
            (self.x_max - self.x_min) *
            (self.y_max - self.y_min) *
            (self.z_max - self.z_min)
        )

    def __repr__(self) -> str:
        return (
            f"WorkspaceLimits("
            f"x=[{self.x_min}, {self.x_max}], "
            f"y=[{self.y_min}, {self.y_max}], "
            f"z=[{self.z_min}, {self.z_max}], "
            f"volume={self.volume:.3f}m³)"
        )
