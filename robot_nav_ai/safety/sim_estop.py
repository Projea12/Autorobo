"""
safety/sim_estop.py — Simulation-native Emergency Stop.

The E-stop is the HIGHEST-PRIORITY signal in the system.  Once triggered it
overrides every other output — joint targets, gripper, base velocity — until
explicitly reset.

Priority chain (highest → lowest)
──────────────────────────────────
  1. SimEStop (this module) — vetos ALL other outputs immediately
  2. ForceLimitGuard        — triggers E-stop on Newton-threshold breach
  3. TrajectoryCollisionChecker — triggers E-stop on pre-exec collision
  4. ArmController F/T check — per-step wrist safety (inner loop)
  5. Policy action           — nominal control

State machine
─────────────
  NORMAL → trigger(reason) → STOPPED → reset() → NORMAL

Thread-safety
─────────────
  trigger() and reset() acquire a lock so they can be called from any thread
  (control loop, watchdog, force monitor).  is_active is a plain attribute
  read — safe to poll without locking on CPython.

Usage
─────
    estop = SimEStop()

    # register optional callbacks (logging, alerts)
    estop.on_trigger(lambda r: print(f"E-STOP: {r}"))

    # inside control loop
    result = arm_ctrl.step(action)
    if not result.wrist_safe:
        estop.trigger("wrist force exceeded")

    # before applying any command
    action = estop.gate(action)  # zeros action if E-stop is active

    # human resets after inspecting the robot
    estop.reset()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimEStopConfig:
    """
    Configuration for the simulation E-stop.

    enabled              : if False, trigger() is a no-op (useful for unit tests
                           that intentionally exceed thresholds)
    zero_action_on_stop  : if True, gate() returns a zero-filled array instead
                           of raising — allows the env to keep stepping cleanly
    log_level            : logging level for trigger/reset events
    """
    enabled:             bool = True
    zero_action_on_stop: bool = True
    log_level:           int  = logging.CRITICAL


# ── trigger record ────────────────────────────────────────────────────────────

@dataclass
class EStopEvent:
    """Record of a single E-stop trigger."""
    reason:    str
    timestamp: float
    source:    str = "unknown"   # e.g. "force_limit", "collision", "manual"

    def __str__(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"[{t}] E-STOP ({self.source}): {self.reason}"


# ── E-stop ────────────────────────────────────────────────────────────────────

class SimEStop:
    """
    Highest-priority safety gate for simulation.

    Once triggered, gate() returns a zeroed action array so the env can
    keep calling step() without the arm moving.  The episode is expected
    to terminate on the same step (reward function detects the stop).

    Parameters
    ----------
    cfg : SimEStopConfig
    """

    def __init__(self, cfg: SimEStopConfig = SimEStopConfig()) -> None:
        self.cfg          = cfg
        self._active      = False
        self._event:  Optional[EStopEvent] = None
        self._history:    list[EStopEvent] = []
        self._callbacks:  list[Callable[[str], None]] = []
        self._lock        = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    def trigger(self, reason: str, source: str = "unknown") -> None:
        """
        Activate the E-stop immediately.

        Thread-safe.  If already active, logs the additional trigger but does
        not overwrite the original event.

        Parameters
        ----------
        reason : human-readable description of why E-stop was triggered
        source : which subsystem triggered it ("force_limit", "collision", …)
        """
        if not self.cfg.enabled:
            return

        with self._lock:
            if self._active:
                log.warning("E-stop already active; additional trigger: %s", reason)
                return
            event          = EStopEvent(reason=reason, timestamp=time.time(), source=source)
            self._event    = event
            self._history.append(event)
            self._active   = True

        log.log(self.cfg.log_level, "EMERGENCY STOP — %s", event)

        for cb in self._callbacks:
            try:
                cb(reason)
            except Exception as exc:
                log.error("E-stop callback error: %s", exc)

    def reset(self) -> None:
        """
        Clear the E-stop and return to NORMAL state.

        In simulation, reset() is called automatically at episode start so
        the next episode begins clean.  On a real robot this would require a
        physical key-switch acknowledgement.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._event  = None
        log.info("E-stop RESET — resuming normal operation")

    def gate(self, action: np.ndarray) -> np.ndarray:
        """
        Return action unchanged when safe, or a zero array when E-stop is active.

        Parameters
        ----------
        action : action array (any shape)

        Returns
        -------
        action if not active, np.zeros_like(action) if active
        """
        if not self._active:
            return action
        if self.cfg.zero_action_on_stop:
            return np.zeros_like(action)
        raise RuntimeError(
            f"E-stop is active — cannot execute action. "
            f"Reason: {self._event.reason if self._event else 'unknown'}"
        )

    def on_trigger(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked with the trigger reason on E-stop."""
        self._callbacks.append(callback)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True while E-stop is triggered and not yet reset."""
        return self._active

    @property
    def event(self) -> Optional[EStopEvent]:
        """The current (or most recent) E-stop event, or None if not triggered."""
        return self._event

    @property
    def history(self) -> list[EStopEvent]:
        """All E-stop events recorded this session (not cleared by reset)."""
        return list(self._history)

    @property
    def trigger_count(self) -> int:
        """Total number of E-stop triggers this session."""
        return len(self._history)

    def __repr__(self) -> str:
        state = "ACTIVE" if self._active else "normal"
        return f"SimEStop(state={state}, triggers={self.trigger_count})"
