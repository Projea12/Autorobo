"""
fault_detector.py — Fault Detection System (Phase 12)

Detects faults during robot operation and classifies them by type.
Runs as a background monitor that checks sensor readings at each step.

Detected fault types:
  - GRASP_FAILURE: gripper closed but no object detected (miss or slip)
  - NAV_STUCK: robot not moving despite nav commands (wheel spin or obstacle)
  - ARM_COLLISION: unexpected arm contact force spike
  - TIMEOUT: episode or sub-task exceeded time limit
  - PERCEPTION_FAILURE: no objects detected when expected

Usage:
    from recovery.fault_detector import FaultDetector, FaultType

    detector = FaultDetector(interface, cfg)
    fault = detector.check()
    if fault:
        print(f"Fault detected: {fault.fault_type.name} — {fault.message}")
        recovery_manager.handle_failure(current_node, fault.exception)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class FaultType(Enum):
    """Classification of detected fault types."""
    GRASP_FAILURE = auto()       # gripper reports no object
    NAV_STUCK = auto()           # robot not moving
    ARM_COLLISION = auto()       # unexpected contact force
    TIMEOUT = auto()             # exceeded time limit
    PERCEPTION_FAILURE = auto()  # detection confidence too low
    JOINT_LIMIT = auto()         # joint approaching limit
    EMERGENCY_STOP = auto()      # E-stop triggered


@dataclass
class Fault:
    """
    A detected fault event.

    Attributes:
        fault_type: Classification of the fault.
        message: Human-readable description.
        severity: "warning", "error", or "critical".
        timestamp: When the fault was detected.
        sensor_data: Relevant sensor readings at fault time.
        exception: RuntimeError wrapping the fault (for recovery_manager).
    """
    fault_type: FaultType
    message: str
    severity: str = "error"
    timestamp: float = 0.0
    sensor_data: dict[str, Any] | None = None
    exception: RuntimeError | None = None

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.exception is None:
            self.exception = RuntimeError(
                f"[{self.fault_type.name}] {self.message}"
            )


class FaultDetector:
    """
    Background fault detection system.

    Monitors robot state at each control step and raises faults
    when anomalies are detected. Integrates with RecoveryManager.

    Thresholds are configurable via Hydra config (cfg.safety / cfg.robot).
    """

    # Default thresholds (overridden by config)
    NAV_STUCK_VELOCITY_THRESHOLD = 0.01   # m/s — below this = stuck
    NAV_STUCK_DURATION_SECONDS = 3.0      # seconds before declaring stuck
    COLLISION_FORCE_THRESHOLD = 30.0      # N — above this = collision
    GRASP_MIN_WIDTH = 0.005               # metres — below this = missed

    def __init__(self, interface: Any, cfg: Any = None) -> None:
        """
        Initialise the fault detector.

        Args:
            interface: BaseRobotInterface for sensor readings.
            cfg: Hydra config with threshold settings.
        """
        self.interface = interface
        self.cfg = cfg

        self._position_history: list[np.ndarray] = []
        self._last_motion_time = time.time()
        self._step_count = 0

        log.info("FaultDetector initialised")

    def check(self) -> Fault | None:
        """
        Check for faults in the current robot state.

        Should be called at every control step (50 Hz).

        Returns:
            Fault object if a fault is detected, None otherwise.

        TODO: Phase 12 — implement all fault checks:
            obs = self.interface.get_observation()
            for check_fn in [
                self._check_nav_stuck,
                self._check_arm_collision,
                self._check_joint_limits,
            ]:
                fault = check_fn(obs)
                if fault:
                    return fault
            return None
        """
        self._step_count += 1
        raise NotImplementedError(
            "TODO: Phase 12 — implement check() to run all fault detection functions "
            "at each step and return the first Fault detected."
        )

    def check_grasp_failure(self, gripper_width: float) -> Fault | None:
        """
        Check if a grasp attempt failed.

        Called after gripper closes — if width is too small, no object was grasped.

        Args:
            gripper_width: Current gripper jaw separation in metres.

        Returns:
            GRASP_FAILURE fault if width < threshold, None otherwise.
        """
        if gripper_width < self.GRASP_MIN_WIDTH:
            return Fault(
                fault_type=FaultType.GRASP_FAILURE,
                message=(
                    f"Grasp failed: gripper closed to {gripper_width:.4f}m "
                    f"(< {self.GRASP_MIN_WIDTH}m minimum). Object not in gripper."
                ),
                severity="error",
                sensor_data={"gripper_width": gripper_width},
            )
        return None

    def _check_nav_stuck(self, obs: dict[str, Any]) -> Fault | None:
        """
        Detect if the robot is stuck (not moving despite commands).

        Uses a sliding window of robot positions to detect lack of motion.

        Args:
            obs: Current observation dict with proprioception.

        Returns:
            NAV_STUCK fault if robot hasn't moved in NAV_STUCK_DURATION_SECONDS.

        TODO: Phase 12 — implement:
            current_pos = obs["proprioception"][:2]  # x, y base position
            self._position_history.append(current_pos)
            if len(self._position_history) > 150:  # 3s at 50Hz
                self._position_history.pop(0)
            displacement = np.linalg.norm(
                self._position_history[-1] - self._position_history[0]
            )
            if len(self._position_history) == 150 and displacement < 0.05:
                return Fault(FaultType.NAV_STUCK, "Robot not moving")
        """
        raise NotImplementedError(
            "TODO: Phase 12 — implement _check_nav_stuck() using position history."
        )

    def _check_arm_collision(self, obs: dict[str, Any]) -> Fault | None:
        """
        Detect unexpected arm contact forces.

        A force spike above COLLISION_FORCE_THRESHOLD during a non-grasp move
        indicates the arm hit an obstacle.

        Args:
            obs: Current observation dict.

        Returns:
            ARM_COLLISION fault if force exceeds threshold.

        TODO: Phase 12 — implement:
            force = obs.get("contact_force", 0.0)
            if force > self.COLLISION_FORCE_THRESHOLD:
                return Fault(
                    FaultType.ARM_COLLISION,
                    f"Arm collision: {force:.1f}N > {self.COLLISION_FORCE_THRESHOLD}N",
                    severity="critical"
                )
        """
        raise NotImplementedError(
            "TODO: Phase 12 — implement _check_arm_collision() using force sensor data."
        )

    def _check_joint_limits(self, obs: dict[str, Any]) -> Fault | None:
        """
        Detect if any arm joint is approaching its mechanical limit.

        Args:
            obs: Current observation dict with proprioception (joint positions).

        Returns:
            JOINT_LIMIT fault if any joint is within 5° of its limit.

        TODO: Phase 12 — implement using robot_cfg.arm.joint_limits.
        """
        raise NotImplementedError(
            "TODO: Phase 12 — implement _check_joint_limits() from joint positions."
        )

    def check_timeout(
        self,
        start_time: float,
        timeout_seconds: float,
        task_description: str = "",
    ) -> Fault | None:
        """
        Check if a task has exceeded its time limit.

        Args:
            start_time: Unix timestamp when the task/action started.
            timeout_seconds: Maximum allowed duration.
            task_description: Human-readable task name for the fault message.

        Returns:
            TIMEOUT fault if elapsed time > timeout_seconds.
        """
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            return Fault(
                fault_type=FaultType.TIMEOUT,
                message=(
                    f"Task timed out: '{task_description}' "
                    f"took {elapsed:.1f}s > {timeout_seconds}s limit."
                ),
                severity="error",
                sensor_data={"elapsed_seconds": elapsed, "timeout": timeout_seconds},
            )
        return None

    def reset(self) -> None:
        """Reset fault detector state (call at start of each episode)."""
        self._position_history.clear()
        self._last_motion_time = time.time()
        self._step_count = 0
        log.debug("FaultDetector reset for new episode.")
