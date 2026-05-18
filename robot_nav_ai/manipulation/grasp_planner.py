"""
grasp_planner.py — Grasp Approach Planner (Phase 8)

Plans the arm trajectory from its current configuration to a target grasp pose.
Takes the 6D grasp pose estimated by GraspEstimator and outputs a sequence of
joint configurations (or Cartesian waypoints) for the arm controller to execute.

Planning includes:
  1. Pre-grasp approach: move to position above the object
  2. Final grasp descent: lower to grasp contact position
  3. Post-grasp lift: raise the object after closing the gripper

Uses inverse kinematics (IK) to convert Cartesian waypoints to joint angles.

Usage:
    from manipulation.grasp_planner import GraspPlanner
    from perception.grasp_estimator import GraspCandidate

    planner = GraspPlanner(cfg.robot)
    trajectory = planner.plan_grasp(current_joints, grasp_candidate)
    for waypoint in trajectory.waypoints:
        arm_controller.move_to_joints(waypoint)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from manipulation.workspace_limits import WorkspaceLimits
from perception.grasp_estimator import GraspCandidate

log = logging.getLogger(__name__)


@dataclass
class GraspTrajectory:
    """
    Planned trajectory for a grasp approach sequence.

    Attributes:
        waypoints: List of joint configurations [j1..j6] in radians.
        cartesian_waypoints: Corresponding end-effector positions. Shape (N, 3).
        approach_phase_end: Index into waypoints where pre-grasp approach ends.
        grasp_phase_end: Index into waypoints where gripper should close.
        is_feasible: False if IK failed for any waypoint.
        failure_reason: Human-readable reason if is_feasible is False.
    """
    waypoints: list[np.ndarray] = field(default_factory=list)  # each (6,)
    cartesian_waypoints: list[np.ndarray] = field(default_factory=list)  # each (3,)
    approach_phase_end: int = 0
    grasp_phase_end: int = 0
    is_feasible: bool = True
    failure_reason: str | None = None


class GraspPlanner:
    """
    Plans arm trajectories for grasping objects.

    Converts grasp candidates (from GraspEstimator) into executable
    joint-space trajectories using inverse kinematics.

    The planner checks:
    - All waypoints are within WorkspaceLimits
    - IK has a valid solution at each waypoint
    - The trajectory is collision-free (simplified: no self-collision)
    - The approach direction is reachable from the current configuration
    """

    def __init__(self, robot_cfg: Any) -> None:
        """
        Initialise the grasp planner with robot configuration.

        Args:
            robot_cfg: DictConfig with robot.arm and robot.workspace settings.

        TODO: Phase 8 — implement:
            self._workspace = WorkspaceLimits(robot_cfg.workspace)
            self._pre_grasp_height_offset = 0.15  # metres above object
            self._ik_solver = self._init_ik_solver(robot_cfg.arm)
        """
        self.robot_cfg = robot_cfg
        self._workspace = WorkspaceLimits(robot_cfg.workspace if hasattr(robot_cfg, "workspace") else None)
        self._pre_grasp_height_offset = 0.15   # metres above object centroid
        self._grasp_speed = 0.05               # m/s during final descent
        log.info("GraspPlanner created (IK solver not yet initialised — TODO: Phase 8)")

    def plan_grasp(
        self,
        current_joints: np.ndarray,
        grasp_candidate: GraspCandidate,
    ) -> GraspTrajectory:
        """
        Plan a complete grasp trajectory from current arm state to grasp pose.

        Planning phases:
        1. Move to pre-grasp position (15cm above object)
        2. Descend slowly to grasp position
        3. (Gripper closes — handled by GripperController)
        4. Lift object (20cm upward)

        Args:
            current_joints: Current arm joint angles [j1..j6], radians. Shape (6,).
            grasp_candidate: Target grasp pose from GraspEstimator.

        Returns:
            GraspTrajectory with joint waypoints.
            Check trajectory.is_feasible before executing.

        TODO: Phase 8 — implement:
            pre_grasp_pose = grasp_candidate.pre_grasp_position
            grasp_pose = grasp_candidate.position_3d
            lift_pose = grasp_pose + np.array([0, 0, 0.2])

            pre_grasp_joints = self._solve_ik(pre_grasp_pose, grasp_candidate.orientation)
            grasp_joints = self._solve_ik(grasp_pose, grasp_candidate.orientation)
            lift_joints = self._solve_ik(lift_pose, grasp_candidate.orientation)

            if any(joints is None for joints in [pre_grasp_joints, grasp_joints, lift_joints]):
                return GraspTrajectory(is_feasible=False, failure_reason="IK failed")

            return self._interpolate_trajectory(
                [current_joints, pre_grasp_joints, grasp_joints, lift_joints]
            )
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement plan_grasp(): "
            "compute pre-grasp and grasp poses, solve IK, interpolate trajectory."
        )

    def _solve_ik(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        seed_joints: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """
        Solve inverse kinematics for a target Cartesian pose.

        Args:
            target_position: Target end-effector position (x, y, z), metres.
            target_orientation: Target orientation as rotation matrix (3, 3).
            seed_joints: Initial joint guess for IK solver. Uses current joints if None.

        Returns:
            Joint configuration [j1..j6] in radians, or None if IK failed.

        TODO: Phase 8 — implement using PyKDL or ikfast:
            from kdl_parser import urdf_to_kdl
            # or: use mujoco's built-in IK via mjtFwdDynCtrl
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement IK solver using PyKDL or MuJoCo IK."
        )

    def _interpolate_trajectory(
        self,
        key_configs: list[np.ndarray],
        n_waypoints_per_segment: int = 20,
    ) -> GraspTrajectory:
        """
        Interpolate between key joint configurations with linear joint interpolation.

        Args:
            key_configs: List of key joint configurations.
            n_waypoints_per_segment: Waypoints between each pair of key configs.

        Returns:
            GraspTrajectory with interpolated waypoints.

        TODO: Phase 8 — implement linear joint interpolation (LERP) between
        consecutive key configs. Check workspace limits at each waypoint.
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement trajectory interpolation (LERP in joint space)."
        )

    def is_reachable(self, position_3d: np.ndarray) -> bool:
        """
        Check if a 3D position is reachable by the arm.

        Args:
            position_3d: Target position in robot base frame, metres.

        Returns:
            True if position is within workspace limits and IK has a solution.
        """
        if not self._workspace.contains(position_3d):
            return False
        # Full IK check deferred to Phase 8
        log.debug(
            f"Position {position_3d} is within workspace — "
            "IK reachability check not yet implemented (TODO: Phase 8)."
        )
        return True
