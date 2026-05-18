"""
gripper_controller.py — Gripper Controller (Phase 8)

Controls the parallel-jaw gripper: open, close, and position control.
Monitors gripper force to detect grasps (object contact) and slippage.

Usage:
    from manipulation.gripper_controller import GripperController

    gripper = GripperController(interface, cfg.robot.gripper)
    gripper.open()
    gripper.close()
    success = gripper.close_and_verify()  # returns True if object grasped
    current_width = gripper.get_width()
"""

from __future__ import annotations

import logging
import time
from typing import Any

from interfaces.base_interface import BaseRobotInterface

log = logging.getLogger(__name__)


class GripperController:
    """
    Controls the parallel-jaw gripper on the robot arm.

    Provides open/close commands and grasp verification via force sensing.
    Works with both simulation (MuJoCo) and real robot (ROS2).

    Grasp verification strategy:
      - After closing, check if gripper width is > min_object_width
        (if width = 0, fingers fully closed = no object)
      - Monitor gripper force during hold — sudden drop = slip detected
    """

    OPEN_WIDTH: float = 0.085    # metres — fully open
    CLOSED_WIDTH: float = 0.001  # metres — fully closed (no object)
    MIN_GRASP_WIDTH: float = 0.005  # metres — minimum to detect object contact

    def __init__(self, interface: BaseRobotInterface, gripper_cfg: Any) -> None:
        """
        Initialise the gripper controller.

        Args:
            interface: Robot interface (MuJoCo or ROS2).
            gripper_cfg: DictConfig with robot.gripper settings.
        """
        self.interface = interface
        self.gripper_cfg = gripper_cfg
        self._max_opening = getattr(gripper_cfg, "max_opening", self.OPEN_WIDTH)
        self._max_force = getattr(gripper_cfg, "max_force", 20.0)  # Newtons
        self._is_open: bool = True
        log.info(f"GripperController created (max opening: {self._max_opening}m)")

    def open(self, timeout: float = 3.0) -> bool:
        """
        Open the gripper fully.

        Args:
            timeout: Maximum time to wait for gripper to open.

        Returns:
            True if gripper reached open position within timeout.

        TODO: Phase 8 — implement:
            action = self._width_to_action(self._max_opening)
            self.interface.apply_action(action)
            # Wait for width sensor to confirm open
            start = time.time()
            while time.time() - start < timeout:
                if self.get_width() >= self._max_opening - 0.005:
                    self._is_open = True
                    return True
                time.sleep(0.02)
            return False
        """
        log.info("Opening gripper...")
        self._is_open = True
        raise NotImplementedError(
            "TODO: Phase 8 — implement open() with width feedback."
        )

    def close(
        self,
        target_width: float | None = None,
        timeout: float = 3.0,
    ) -> bool:
        """
        Close the gripper to a target width.

        Args:
            target_width: Target width in metres. None = fully close.
            timeout: Maximum time to wait for gripper motion.

        Returns:
            True if target width was reached within timeout.

        TODO: Phase 8 — implement:
            target = target_width if target_width is not None else self.CLOSED_WIDTH
            action = self._width_to_action(target)
            self.interface.apply_action(action)
            # Wait for completion
        """
        target = target_width if target_width is not None else self.CLOSED_WIDTH
        log.info(f"Closing gripper to {target:.3f}m...")
        self._is_open = False
        raise NotImplementedError(
            "TODO: Phase 8 — implement close() with width feedback."
        )

    def close_and_verify(self, timeout: float = 5.0) -> bool:
        """
        Close the gripper and verify that an object was grasped.

        An object is considered grasped if the gripper stops at a width
        greater than MIN_GRASP_WIDTH (fingers were stopped by the object).

        Args:
            timeout: Maximum time for close + verification.

        Returns:
            True if an object was detected in the gripper.

        TODO: Phase 8 — implement:
            self.close(timeout=timeout)
            final_width = self.get_width()
            grasped = final_width > self.MIN_GRASP_WIDTH
            if not grasped:
                log.warning(f"Grasp failed: final width {final_width:.3f}m (too narrow)")
            return grasped
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement close_and_verify() with grasp detection."
        )

    def get_width(self) -> float:
        """
        Get the current gripper jaw separation width.

        Returns:
            Current gripper width in metres. Range: [0, max_opening].

        TODO: Phase 8 — implement:
            obs = self.interface.get_observation()
            # Gripper state is the last element of proprioception
            gripper_state = obs["proprioception"][-1]  # normalised [0, 1]
            return gripper_state * self._max_opening
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement get_width() from proprioception obs."
        )

    def get_force(self) -> float:
        """
        Get the current gripper contact force.

        Returns:
            Contact force in Newtons. 0.0 if no contact.

        TODO: Phase 8 — implement from force/torque sensor reading.
        In simulation: use MuJoCo contact force from self.interface._data.
        On real robot: read from force/torque sensor via ROS2 topic.
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement get_force() from contact sensors."
        )

    def detect_slip(self, force_threshold: float = 2.0) -> bool:
        """
        Detect if the grasped object is slipping.

        Slip is detected when the gripper force drops below force_threshold
        after having been above it (object was being held, then slipped).

        Args:
            force_threshold: Force drop threshold in Newtons.

        Returns:
            True if slip is detected.

        TODO: Phase 8 — implement with a running force history:
            current_force = self.get_force()
            if self._prev_force > force_threshold and current_force < force_threshold:
                return True
            self._prev_force = current_force
            return False
        """
        raise NotImplementedError(
            "TODO: Phase 8 — implement slip detection from force history."
        )

    @property
    def is_open(self) -> bool:
        """Return True if the gripper is in the open state."""
        return self._is_open
