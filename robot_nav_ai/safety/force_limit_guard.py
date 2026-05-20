"""
safety/force_limit_guard.py — Newton-threshold force limit guard.

Reads wrist force-torque directly from MuJoCo sensordata and triggers the
SimEStop immediately if either magnitude exceeds the configured limit.

Why this is separate from ArmController's F/T check
─────────────────────────────────────────────────────
  ArmController.step() checks wrist safety *inside* the IK solve — it returns
  wrist_safe=False and the env terminates the episode.  That is the inner-loop
  check.

  ForceLimitGuard is the *outer-loop* Phase 6 safety gate.  It can be called:
    • before applying any command (pre-exec guard)
    • after mj_step (post-exec verification)
  and it triggers the E-stop directly — the arm command is then gated to zero
  by SimEStop.gate() before data.ctrl is ever written.

Force units
────────────
  MuJoCo force sensors report in the model's force unit (Newtons if the model
  uses SI).  sensordata[28:31] = wrist 3-axis force [N],
  sensordata[31:34] = wrist 3-axis torque [N·m].
  Both slices match DEFAULT_LIMITS (see robot/workspace.py).

Usage
─────
    guard = ForceLimitGuard(model, data, estop)

    # call every step before writing data.ctrl
    result = guard.check()
    if result.violated:
        # estop already triggered — gate() will zero the action
        pass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ForceLimitConfig:
    """
    Newton thresholds for the force limit guard.

    max_force_n          : maximum wrist resultant force [N].
                           Default 50 N — matches DEFAULT_LIMITS.wrist_force_max.
    max_torque_nm        : maximum wrist resultant torque [N·m].
                           Default 10 N·m — matches DEFAULT_LIMITS.wrist_torque_max.
    sensor_force_slice   : sensordata index range for 3-axis wrist force.
    sensor_torque_slice  : sensordata index range for 3-axis wrist torque.
    """
    max_force_n:         float = 50.0
    max_torque_nm:       float = 10.0
    sensor_force_slice:  tuple = (28, 31)
    sensor_torque_slice: tuple = (31, 34)


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class ForceLimitResult:
    """
    Outcome of a single force-limit check.

    Fields
    ------
    violated        : True if either force or torque exceeded its limit
    force_n         : measured wrist force magnitude [N]
    torque_nm       : measured wrist torque magnitude [N·m]
    force_limit     : configured force limit [N]
    torque_limit    : configured torque limit [N·m]
    reason          : human-readable violation description (empty if ok)
    """
    violated:    bool
    force_n:     float
    torque_nm:   float
    force_limit: float
    torque_limit: float
    reason:      str = ""

    @property
    def safe(self) -> bool:
        return not self.violated

    def __repr__(self) -> str:
        if self.violated:
            return f"ForceLimitResult(VIOLATED: {self.reason})"
        return (f"ForceLimitResult(ok, "
                f"F={self.force_n:.1f}N/{self.force_limit:.0f}N, "
                f"T={self.torque_nm:.2f}Nm/{self.torque_limit:.0f}Nm)")


# ── guard ─────────────────────────────────────────────────────────────────────

class ForceLimitGuard:
    """
    Phase 6 outer-loop force/torque safety gate.

    Reads wrist sensordata from MuJoCo and triggers the E-stop if any limit
    is breached.  Designed to be called every control step (10 ms loop).

    Parameters
    ----------
    model  : mujoco.MjModel
    data   : mujoco.MjData  (sensordata read every call to check())
    estop  : SimEStop — triggered immediately on violation
    cfg    : ForceLimitConfig
    """

    def __init__(
        self,
        model,
        data,
        estop,
        cfg: ForceLimitConfig = ForceLimitConfig(),
    ) -> None:
        self._model  = model
        self._data   = data
        self._estop  = estop
        self.cfg     = cfg

        fs, fe = cfg.sensor_force_slice
        ts, te = cfg.sensor_torque_slice
        self._fslice = slice(fs, fe)
        self._tslice = slice(ts, te)

        self._violation_count = 0

    # ── public API ────────────────────────────────────────────────────────────

    def check(self) -> ForceLimitResult:
        """
        Read current sensordata and check force/torque limits.

        If a limit is exceeded:
          1. Builds a ForceLimitResult with violated=True and a reason string.
          2. Calls estop.trigger() with that reason.
          3. Returns the result (caller can log or use for reward shaping).

        If within limits:
          Returns ForceLimitResult with violated=False.
        """
        wrist_force  = self._data.sensordata[self._fslice]
        wrist_torque = self._data.sensordata[self._tslice]

        f_mag = float(np.linalg.norm(wrist_force))
        t_mag = float(np.linalg.norm(wrist_torque))

        reason = ""
        if f_mag > self.cfg.max_force_n:
            reason = (f"wrist force {f_mag:.1f} N > limit {self.cfg.max_force_n:.1f} N")
        elif t_mag > self.cfg.max_torque_nm:
            reason = (f"wrist torque {t_mag:.2f} N·m > limit {self.cfg.max_torque_nm:.1f} N·m")

        violated = bool(reason)
        if violated:
            self._violation_count += 1
            log.warning("ForceLimitGuard: %s", reason)
            self._estop.trigger(reason=reason, source="force_limit")

        return ForceLimitResult(
            violated     = violated,
            force_n      = f_mag,
            torque_nm    = t_mag,
            force_limit  = self.cfg.max_force_n,
            torque_limit = self.cfg.max_torque_nm,
            reason       = reason,
        )

    def check_raw(
        self,
        force_vec:  np.ndarray,
        torque_vec: np.ndarray,
    ) -> ForceLimitResult:
        """
        Check force/torque limits from raw vectors (without reading sensordata).

        Useful for testing or when the caller has already read the sensor.

        Parameters
        ----------
        force_vec  : (3,) wrist force vector [N]
        torque_vec : (3,) wrist torque vector [N·m]
        """
        f_mag = float(np.linalg.norm(force_vec))
        t_mag = float(np.linalg.norm(torque_vec))

        reason = ""
        if f_mag > self.cfg.max_force_n:
            reason = f"wrist force {f_mag:.1f} N > limit {self.cfg.max_force_n:.1f} N"
        elif t_mag > self.cfg.max_torque_nm:
            reason = f"wrist torque {t_mag:.2f} N·m > limit {self.cfg.max_torque_nm:.1f} N·m"

        violated = bool(reason)
        if violated:
            self._violation_count += 1
            self._estop.trigger(reason=reason, source="force_limit")

        return ForceLimitResult(
            violated     = violated,
            force_n      = f_mag,
            torque_nm    = t_mag,
            force_limit  = self.cfg.max_force_n,
            torque_limit = self.cfg.max_torque_nm,
            reason       = reason,
        )

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def violation_count(self) -> int:
        """Total force/torque violations detected this session."""
        return self._violation_count

    def __repr__(self) -> str:
        return (f"ForceLimitGuard("
                f"max_force={self.cfg.max_force_n}N, "
                f"max_torque={self.cfg.max_torque_nm}Nm, "
                f"violations={self._violation_count})")
