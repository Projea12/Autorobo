"""
emergency_stop.py — Emergency Stop System (Phase 13)

Implements immediate robot halt on critical fault detection.
The E-stop is the highest-priority safety system and overrides all other commands.

E-stop is triggered by:
  - Critical arm collision (force > CRITICAL_FORCE_LIMIT)
  - Human proximity violation (distance < STOP_ZONE, 0.2m)
  - Hardware fault signal (ROS2 /emergency_stop topic)
  - Watchdog timeout (main control loop frozen)
  - Manual trigger (keyboard Ctrl+C / physical E-stop button)

After E-stop:
  1. All velocity commands set to zero immediately
  2. Arm commanded to freeze in place (hold current joints)
  3. Gripper holds (does not open — prevents dropping held objects)
  4. EpisodeLogger records the E-stop event
  5. Human must acknowledge before autonomous operation resumes

Usage:
    from safety.emergency_stop import EmergencyStop

    estop = EmergencyStop(interface)
    estop.register_trigger_callback(lambda: print("E-STOP TRIGGERED"))

    # In control loop:
    if estop.is_triggered:
        estop.wait_for_acknowledgement()
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Callable, Any

log = logging.getLogger(__name__)


class EmergencyStop:
    """
    Emergency stop system for safe robot operation.

    Monitors multiple trigger sources and halts the robot immediately
    when any trigger fires. Designed to be thread-safe — can be triggered
    from a control loop, a ROS2 callback, or a signal handler.

    State machine:
        NORMAL → (trigger fires) → STOPPED → (human acknowledges) → NORMAL
    """

    def __init__(self, interface: Any, cfg: Any = None) -> None:
        """
        Initialise the emergency stop system.

        Args:
            interface: BaseRobotInterface — for issuing zero-velocity commands.
            cfg: Optional config with force limits and zone distances.
        """
        self.interface = interface
        self.cfg = cfg

        self._is_triggered = False
        self._trigger_reason = ""
        self._trigger_time: float | None = None
        self._trigger_callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

        # Register SIGINT handler (Ctrl+C → graceful E-stop)
        signal.signal(signal.SIGINT, self._sigint_handler)

        log.info("EmergencyStop initialised — monitoring for critical faults")

    def trigger(self, reason: str = "Unknown", severity: str = "critical") -> None:
        """
        Trigger the emergency stop immediately.

        Thread-safe — can be called from any thread.

        Args:
            reason: Human-readable reason for the E-stop.
            severity: "warning", "error", or "critical".
        """
        with self._lock:
            if self._is_triggered:
                # Already stopped — log but don't re-trigger
                log.warning(f"E-stop already active. Additional trigger: {reason}")
                return

            self._is_triggered = True
            self._trigger_reason = reason
            self._trigger_time = time.time()

        # Halt robot immediately (outside lock to avoid deadlock)
        self._halt_robot()

        log.critical(
            f"EMERGENCY STOP TRIGGERED [{severity.upper()}]: {reason}"
        )

        # Notify registered callbacks
        for callback in self._trigger_callbacks:
            try:
                callback()
            except Exception as e:
                log.error(f"E-stop callback failed: {e}")

    def reset(self, acknowledge_code: str = "") -> bool:
        """
        Reset the emergency stop (requires human acknowledgement).

        Args:
            acknowledge_code: Optional safety code required to reset.
                Set to "" to disable code requirement (development only).

        Returns:
            True if reset was successful.
        """
        if not self._is_triggered:
            log.warning("E-stop reset called but E-stop is not active.")
            return True

        # In production, verify acknowledge_code here
        # For development, allow reset without code
        log.warning(
            "E-stop RESET acknowledged. "
            f"Was triggered at {self._trigger_time}: {self._trigger_reason}"
        )

        with self._lock:
            self._is_triggered = False
            self._trigger_reason = ""
            self._trigger_time = None

        log.info("Emergency stop RESET. Robot may resume autonomous operation.")
        return True

    def wait_for_acknowledgement(self, timeout: float = 300.0) -> bool:
        """
        Block until the E-stop is acknowledged and reset.

        Args:
            timeout: Maximum wait time in seconds (default: 5 minutes).

        Returns:
            True if acknowledged within timeout, False if timeout elapsed.
        """
        log.critical(
            "=== ROBOT HALTED — EMERGENCY STOP ACTIVE ===\n"
            f"Reason: {self._trigger_reason}\n"
            "Resolve the fault and call estop.reset() to resume."
        )

        start_time = time.time()
        while self._is_triggered:
            if time.time() - start_time > timeout:
                log.error(f"E-stop acknowledgement timed out after {timeout}s.")
                return False
            time.sleep(0.5)

        return True

    def test(self) -> None:
        """
        Run a self-test of the E-stop system.

        Triggers E-stop with a test reason, verifies halt commands are sent,
        then resets. Should be called before every real-robot session.
        """
        log.info("Running E-stop self-test...")
        self.trigger(reason="SELF-TEST", severity="warning")
        time.sleep(0.1)  # Allow halt to propagate
        if not self._is_triggered:
            raise RuntimeError("E-stop self-test failed: trigger did not activate.")
        self.reset(acknowledge_code="")
        if self._is_triggered:
            raise RuntimeError("E-stop self-test failed: reset did not clear trigger.")
        log.info("E-stop self-test PASSED.")

    def register_trigger_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be called when E-stop triggers.

        Args:
            callback: Zero-argument callable (e.g., alert sound, push notification).
        """
        self._trigger_callbacks.append(callback)

    def _halt_robot(self) -> None:
        """
        Issue immediate halt commands to all actuators.

        Sends zero velocity to base, holds arm joints at current position,
        and keeps gripper state (does not open to prevent drops).

        TODO: Phase 13 — implement:
            # Zero velocity to mobile base
            zero_nav = np.zeros(2)  # [linear=0, angular=0]
            self.interface.apply_action(zero_nav)

            # Hold arm at current joints (position control → stay in place)
            current_obs = self.interface.get_observation()
            current_joints = current_obs["proprioception"][:6]
            hold_action = np.concatenate([current_joints, [0.0]])  # keep gripper state
            self.interface.apply_action(hold_action)
        """
        log.critical("HALTING ROBOT — sending zero velocity commands...")
        raise NotImplementedError(
            "TODO: Phase 13 — implement _halt_robot(): "
            "zero base velocity, hold arm joints at current position."
        )

    def _sigint_handler(self, signum: int, frame: Any) -> None:
        """Handle Ctrl+C by triggering E-stop gracefully."""
        log.warning("SIGINT received — triggering emergency stop.")
        self.trigger(reason="SIGINT (Ctrl+C)", severity="warning")

    @property
    def is_triggered(self) -> bool:
        """Return True if the emergency stop is currently active."""
        return self._is_triggered

    @property
    def trigger_reason(self) -> str:
        """Return the reason the E-stop was triggered (empty if not triggered)."""
        return self._trigger_reason
