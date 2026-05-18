"""
force_limits.py — Joint Torque and Force Limit Monitor (Phase 13)

Monitors joint torques and end-effector contact forces to prevent:
  - Over-force on grasped objects (crushing fragile items)
  - Arm self-collision via unexpected force spikes
  - Joint damage from excessive torque commands

Force limits are defined per-joint in configs/robot/base.yaml.
Violations trigger the EmergencyStop system.

Usage:
    from safety.force_limits import ForceLimitMonitor

    monitor = ForceLimitMonitor(interface, emergency_stop, cfg.robot.arm)
    monitor.start_monitoring()  # runs in background thread

    # Or check manually each step:
    violation = monitor.check_limits(obs)
    if violation:
        print(f"Force limit exceeded: {violation}")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ForceViolation:
    """Describes a force/torque limit violation."""
    joint_id: int | None        # None if EE force violation
    joint_name: str
    measured_value: float       # Nm (torque) or N (force)
    limit_value: float          # Nm (torque) or N (force)
    violation_type: str         # "torque" or "contact_force"

    def __repr__(self) -> str:
        return (
            f"ForceViolation({self.joint_name}: "
            f"{self.measured_value:.1f} > {self.limit_value:.1f} {self.violation_type})"
        )


class ForceLimitMonitor:
    """
    Monitors arm joint torques and contact forces for safety limit violations.

    Runs at 50 Hz in a background thread during robot operation.
    Triggers EmergencyStop immediately on any violation.

    Safety limits (configurable via robot.arm config):
      - Per-joint max torque (typically 50 Nm for arm joints)
      - End-effector max contact force (typically 30 N)
      - Gripper max force (typically 20 N)
    """

    DEFAULT_MAX_JOINT_TORQUE = 50.0    # Nm per joint
    DEFAULT_MAX_CONTACT_FORCE = 30.0   # N at end-effector
    DEFAULT_MAX_GRIPPER_FORCE = 20.0   # N gripper contact force

    def __init__(
        self,
        interface: Any,
        emergency_stop: Any,
        arm_cfg: Any = None,
    ) -> None:
        """
        Initialise the force limit monitor.

        Args:
            interface: BaseRobotInterface for sensor readings.
            emergency_stop: EmergencyStop instance to trigger on violation.
            arm_cfg: DictConfig with robot.arm settings.
        """
        self.interface = interface
        self.emergency_stop = emergency_stop
        self.arm_cfg = arm_cfg

        # Load limits from config or use defaults
        self._max_joint_torque = getattr(
            arm_cfg, "max_joint_torque", self.DEFAULT_MAX_JOINT_TORQUE
        ) if arm_cfg else self.DEFAULT_MAX_JOINT_TORQUE
        self._max_contact_force = self.DEFAULT_MAX_CONTACT_FORCE
        self._max_gripper_force = self.DEFAULT_MAX_GRIPPER_FORCE

        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._violation_history: list[ForceViolation] = []

        log.info(
            f"ForceLimitMonitor initialised: "
            f"max_torque={self._max_joint_torque}Nm, "
            f"max_contact={self._max_contact_force}N"
        )

    def check_limits(self, obs: dict[str, Any]) -> ForceViolation | None:
        """
        Check all force/torque limits against current observation.

        Args:
            obs: Observation dict from BaseRobotInterface.get_observation().
                Should include "joint_torques" and "contact_forces" if available.

        Returns:
            ForceViolation if any limit is exceeded, None otherwise.

        TODO: Phase 13 — implement:
            joint_torques = obs.get("joint_torques", np.zeros(6))
            for i, (torque, joint_name) in enumerate(zip(joint_torques, joint_names)):
                if abs(torque) > self._max_joint_torque:
                    return ForceViolation(
                        joint_id=i,
                        joint_name=joint_name,
                        measured_value=abs(torque),
                        limit_value=self._max_joint_torque,
                        violation_type="torque",
                    )
            contact_force = obs.get("contact_force_ee", 0.0)
            if contact_force > self._max_contact_force:
                return ForceViolation(
                    joint_id=None,
                    joint_name="end_effector",
                    measured_value=contact_force,
                    limit_value=self._max_contact_force,
                    violation_type="contact_force",
                )
            return None
        """
        raise NotImplementedError(
            "TODO: Phase 13 — implement check_limits(): "
            "read joint_torques and contact_forces from obs, compare to limits."
        )

    def start_monitoring(self, check_frequency_hz: float = 50.0) -> None:
        """
        Start background force monitoring thread.

        Args:
            check_frequency_hz: Rate at which to check limits (default: 50 Hz).

        TODO: Phase 13 — implement background thread that calls check_limits()
        at check_frequency_hz and triggers emergency_stop.trigger() on violation.
        """
        if self._monitoring:
            log.warning("Force monitoring is already running.")
            return

        self._monitoring = True
        sleep_time = 1.0 / check_frequency_hz

        def _monitor_loop() -> None:
            while self._monitoring:
                try:
                    obs = self.interface.get_observation()
                    violation = self.check_limits(obs)
                    if violation:
                        self._violation_history.append(violation)
                        self.emergency_stop.trigger(
                            reason=f"Force limit exceeded: {violation}",
                            severity="critical",
                        )
                except Exception as e:
                    log.error(f"Force monitor error: {e}")
                time.sleep(sleep_time)

        raise NotImplementedError(
            "TODO: Phase 13 — implement start_monitoring() background thread. "
            "After implementing check_limits(), the loop body above is ready to use."
        )

    def stop_monitoring(self) -> None:
        """Stop the background force monitoring thread."""
        self._monitoring = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        log.info("Force monitoring stopped.")

    def set_max_joint_torque(self, max_torque_nm: float) -> None:
        """
        Dynamically update the maximum joint torque limit.

        Args:
            max_torque_nm: New maximum torque in Newton-metres.

        Note: Use caution — reducing limits may cause false positives.
        Increasing limits above hardware specs risks joint damage.
        """
        old_limit = self._max_joint_torque
        self._max_joint_torque = max_torque_nm
        log.info(
            f"Max joint torque updated: {old_limit}Nm → {max_torque_nm}Nm"
        )

    @property
    def violation_count(self) -> int:
        """Return the total number of force violations detected this session."""
        return len(self._violation_history)

    @property
    def is_monitoring(self) -> bool:
        """Return True if background monitoring is active."""
        return self._monitoring
