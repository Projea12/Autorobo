"""
arm_controller.py — Robot Arm Controller (Phase 8)

Sends joint position/velocity commands to the robot arm and monitors
execution. Operates on the robot interface layer (BaseRobotInterface)
so it works identically in simulation and on the real robot.

The arm controller:
  - Accepts joint-space targets (from GraspPlanner trajectories)
  - Converts to interface actions and calls apply_action()
  - Monitors joint positions to detect completion or failure
  - Implements safety checks (torque limits, workspace limits)

Usage:
    from manipulation.arm_controller import ArmController

    controller = ArmController(interface, cfg.robot.arm)
    controller.move_to_joints(target_joints, timeout=10.0)
    controller.move_to_cartesian(target_pose, timeout=10.0)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from interfaces.base_interface import BaseRobotInterface

log = logging.getLogger(__name__)


class ArmController:
    """
    High-level controller for the robot arm.

    Wraps the low-level BaseRobotInterface with arm-specific control
    logic: trajectory following, position monitoring, and timeout handling.

    Motion control modes:
      - Joint position control: move each joint to target angle
      - Cartesian control: move end-effector to target 6D pose (via IK)
      - Velocity control: apply joint velocity commands directly
    """

    def __init__(self, interface: BaseRobotInterface, arm_cfg: Any) -> None:
        """
        Initialise the arm controller.

        Args:
            interface: Robot interface (MuJoCo or ROS2).
            arm_cfg: DictConfig with robot.arm settings (DOF, limits, frequency).

        TODO: Phase 8 — implement:
            self._dof = arm_cfg.dof  # 6
            self._joint_limits = np.array(arm_cfg.joint_limits)  # (6, 2)
            self._max_velocity = arm_cfg.max_joint_velocity
            self._control_freq = arm_cfg.control_frequency
        """
        self.interface = interface
        self.arm_cfg = arm_cfg
        self._dof = getattr(arm_cfg, "dof", 6)
        self._position_tolerance = 0.01  # radians — "at target" threshold
        log.info(f"ArmController created for {self._dof}-DOF arm")

    def move_to_joints(
        self,
        target_joints: np.ndarray,
        timeout: float = 10.0,
        blocking: bool = True,
    ) -> bool:
        """
        Move the arm to a target joint configuration.

        Args:
            target_joints: Target joint angles [j1..j6] in radians. Shape (6,).
            timeout: Maximum time to wait for motion completion, seconds.
            blocking: If True, wait until target reached or timeout. If False,
                send command and return immediately.

        Returns:
            True if target was reached within tolerance and timeout.
            False if timed out or motion failed.

        TODO: Phase 8 — implement:
            start_time = time.time()
            while time.time() - start_time < timeout:
                current_obs = self.interface.get_observation()
                current_joints = current_obs["proprioception"][:6]
                error = np.abs(target_joints - current_joints)
                if np.all(error < self._position_tolerance):
                    return True
                action = self._joint_position_to_action(target_joints)
                self.interface.apply_action(action)
                time.sleep(1.0 / self._control_freq)
            return False
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement move_to_joints() with position feedback loop."
        )

    def move_to_cartesian(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        timeout: float = 15.0,
    ) -> bool:
        """
        Move the end-effector to a target Cartesian pose.

        Internally solves IK (via GraspPlanner) and calls move_to_joints().

        Args:
            target_position: Target EE position (x, y, z) in base frame. Shape (3,).
            target_orientation: Target orientation as rotation matrix. Shape (3, 3).
            timeout: Maximum time for motion completion.

        Returns:
            True if EE reached target within tolerance.

        TODO: Phase 8 — implement via IK + move_to_joints().
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement move_to_cartesian() via IK + move_to_joints()."
        )

    def follow_trajectory(
        self,
        trajectory,
        timeout: float = 60.0,
    ) -> bool:
        """
        Execute a planned GraspTrajectory waypoint-by-waypoint.

        Args:
            trajectory: GraspTrajectory from GraspPlanner.plan_grasp().
            timeout: Maximum total time for full trajectory execution.

        Returns:
            True if all waypoints were reached successfully.

        TODO: Phase 8 — implement:
            for i, waypoint in enumerate(trajectory.waypoints):
                success = self.move_to_joints(waypoint, timeout=timeout/len(trajectory.waypoints))
                if not success:
                    log.warning(f"Trajectory failed at waypoint {i}/{len(trajectory.waypoints)}")
                    return False
            return True
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement follow_trajectory() waypoint execution."
        )

    def get_current_joints(self) -> np.ndarray:
        """
        Get the current arm joint positions from the robot interface.

        Returns:
            Current joint angles [j1..j6] in radians. Shape (6,).

        TODO: Phase 8 — implement:
            obs = self.interface.get_observation()
            return obs["proprioception"][:self._dof]
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement get_current_joints() from obs proprioception."
        )

    def home(self, timeout: float = 15.0) -> bool:
        """
        Move the arm to the home (zero) configuration.

        Args:
            timeout: Maximum time for homing motion.

        Returns:
            True if home position was reached.
        """
        home_joints = np.zeros(self._dof)
        log.info("Homing arm to zero configuration...")
        return self.move_to_joints(home_joints, timeout=timeout)
