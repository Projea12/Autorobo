"""
safety/fault_detector.py — Robot fault detector with defined safe responses.

Detects three categories of fault and executes a pre-defined safe response
for each.  All faults ultimately route through SimEStop so the arm is halted
with the same mechanism as force/collision violations.

Fault categories
─────────────────
  SENSOR_DROPOUT   : A sensor channel has gone NaN/Inf OR its value has been
                     frozen (identical for N consecutive steps).  For wrist F/T
                     this is a critical fault — arm halts.  For non-critical
                     sensors (e.g. gripper tactile) a warning is issued and the
                     last known-good value is held.

  JOINT_STALL      : A joint was commanded to move (|dq_cmd| > threshold) but
                     its measured position has not changed significantly
                     (|Δqpos| < deadband) for N consecutive steps.  Indicates
                     a jammed joint, slipped belt, or amplifier fault.
                     Safe response: freeze arm at current qpos + E-stop.

  COMM_FAILURE     : A watchdog timer — heartbeat() must be called every
                     watchdog_ms milliseconds.  If it is not called (e.g.
                     control loop hangs, network disconnects), the watchdog
                     fires on the next check_comm() call.
                     Safe response: E-stop immediately.

Safe responses per fault
─────────────────────────
  SENSOR_DROPOUT (critical)  → E-stop + freeze arm
  SENSOR_DROPOUT (warning)   → log + hold last-good value, no E-stop
  JOINT_STALL                → E-stop + freeze arm
  COMM_FAILURE               → E-stop immediately

Usage
─────
    detector = FaultDetector(estop, cfg=FaultConfig())
    detector.reset()

    # in control loop:
    detector.heartbeat()                           # keep watchdog alive
    faults = detector.check_all(data, q_cmd, q_actual)
    for f in faults:
        print(f.fault_type, f.safe_response)       # already handled
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ── fault type ────────────────────────────────────────────────────────────────

class FaultType(Enum):
    NONE           = "none"
    SENSOR_DROPOUT = "sensor_dropout"
    JOINT_STALL    = "joint_stall"
    COMM_FAILURE   = "comm_failure"


class FaultSeverity(Enum):
    WARNING  = "warning"   # log + degrade gracefully
    CRITICAL = "critical"  # E-stop immediately


# ── fault event ───────────────────────────────────────────────────────────────

@dataclass
class FaultEvent:
    """
    Record of a detected fault.

    Fields
    ──────
    fault_type    : FaultType enum
    severity      : WARNING or CRITICAL
    description   : human-readable explanation
    safe_response : what the system is doing in response (past tense)
    timestamp     : wall-clock time of detection
    details       : optional extra data (channel index, stall step count, etc.)
    """
    fault_type:    FaultType
    severity:      FaultSeverity
    description:   str
    safe_response: str
    timestamp:     float = field(default_factory=time.time)
    details:       dict  = field(default_factory=dict)

    def __str__(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return (f"[{t}] {self.severity.name} FAULT — {self.fault_type.value}: "
                f"{self.description}  →  {self.safe_response}")


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FaultConfig:
    """
    Thresholds for the three fault detectors.

    Sensor dropout
    ──────────────
    freeze_window        : steps with identical sensordata before calling dropout
    critical_slices      : sensordata ranges that are CRITICAL (wrist F/T)
                           expressed as list of (start, end) pairs
    warning_slices       : sensordata ranges that are WARNING-only

    Joint stall
    ───────────
    stall_cmd_threshold  : |dq_cmd| must exceed this [rad/s] to be "commanded"
    stall_deadband       : |Δqpos| below this over stall_window steps → stall
    stall_window         : consecutive steps before declaring stall

    Communication / watchdog
    ────────────────────────
    watchdog_ms          : milliseconds between required heartbeat() calls
    """
    # sensor dropout
    freeze_window:       int   = 10
    critical_slices:     tuple = ((28, 34),)   # wrist F/T sensordata[28:34]
    warning_slices:      tuple = ((24, 26),)   # gripper tactile sensordata[24:26]

    # joint stall
    stall_cmd_threshold: float = 0.05    # rad/s — below this = "not commanded"
    stall_deadband:      float = 1e-4    # rad   — below this = "not moving"
    stall_window:        int   = 15      # steps

    # watchdog
    watchdog_ms:         float = 500.0   # 500 ms — fires if loop hangs


# ── detector ──────────────────────────────────────────────────────────────────

class FaultDetector:
    """
    Detects sensor dropout, joint stall, and communication failure faults.

    All detected faults call estop.trigger() (except WARNING-severity sensor
    dropouts which only log).  The caller still receives the FaultEvent so it
    can shape the reward / end the episode.

    Parameters
    ----------
    estop : SimEStop
    cfg   : FaultConfig
    """

    def __init__(self, estop, cfg: FaultConfig = FaultConfig()) -> None:
        self._estop = estop
        self.cfg    = cfg
        self.reset()

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Call once per episode to clear all fault state."""
        # sensor dropout tracking
        self._sensor_prev:   Optional[np.ndarray] = None
        self._freeze_counts: dict[tuple, int]     = {}   # slice → freeze steps
        self._last_good:     dict[tuple, np.ndarray] = {}  # slice → last non-frozen

        # joint stall tracking
        self._stall_counts:  np.ndarray = np.zeros(self.cfg.stall_window + 1)
        self._stall_steps:   np.ndarray = np.zeros(0)   # initialised on first call
        self._qpos_prev:     Optional[np.ndarray] = None

        # watchdog
        self._last_heartbeat: float = time.monotonic()

        # fault history
        self._history: list[FaultEvent] = []

    def heartbeat(self) -> None:
        """
        Reset the communication watchdog timer.

        Must be called at least every cfg.watchdog_ms milliseconds from the
        main control loop.  Missing this call for longer than watchdog_ms
        causes check_comm() to return a COMM_FAILURE fault.
        """
        self._last_heartbeat = time.monotonic()

    def check_sensor_dropout(self, sensordata: np.ndarray) -> list[FaultEvent]:
        """
        Check for NaN/Inf values and frozen sensor channels.

        Parameters
        ----------
        sensordata : full sensordata array (e.g. data.sensordata)

        Returns
        -------
        List of FaultEvent — one per affected channel range.
        """
        faults: list[FaultEvent] = []
        sd = np.asarray(sensordata, dtype=np.float64)

        for slices, is_critical in (
            *[(s, True)  for s in self.cfg.critical_slices],
            *[(s, False) for s in self.cfg.warning_slices],
        ):
            s, e   = slices
            chunk  = sd[s:e]
            key    = (s, e)

            # ── NaN / Inf check ───────────────────────────────────────────────
            if not np.all(np.isfinite(chunk)):
                bad = "NaN" if np.any(np.isnan(chunk)) else "Inf"
                fault = self._make_sensor_fault(
                    severity   = FaultSeverity.CRITICAL if is_critical else FaultSeverity.WARNING,
                    desc       = f"sensordata[{s}:{e}] contains {bad}",
                    response   = "E-stop triggered; arm frozen" if is_critical
                                 else f"holding last-good value for sensordata[{s}:{e}]",
                    details    = {"slice": (s, e), "is_critical": is_critical},
                    is_critical= is_critical,
                )
                faults.append(fault)
                continue   # skip freeze check for this chunk

            # ── freeze (stuck value) check ────────────────────────────────────
            if key in self._last_good:
                prev = self._last_good[key]
                if np.allclose(chunk, prev, atol=1e-9):
                    self._freeze_counts[key] = self._freeze_counts.get(key, 0) + 1
                else:
                    self._freeze_counts[key] = 0
                    self._last_good[key] = chunk.copy()
            else:
                self._last_good[key]    = chunk.copy()
                self._freeze_counts[key] = 0

            if self._freeze_counts[key] >= self.cfg.freeze_window:
                fault = self._make_sensor_fault(
                    severity   = FaultSeverity.CRITICAL if is_critical else FaultSeverity.WARNING,
                    desc       = (f"sensordata[{s}:{e}] frozen for "
                                  f"{self._freeze_counts[key]} steps"),
                    response   = "E-stop triggered; arm frozen" if is_critical
                                 else f"holding last-good value for sensordata[{s}:{e}]",
                    details    = {"slice": (s, e), "freeze_steps": self._freeze_counts[key]},
                    is_critical= is_critical,
                )
                faults.append(fault)

        return faults

    def check_joint_stall(
        self,
        q_cmd:    np.ndarray,
        q_actual: np.ndarray,
    ) -> list[FaultEvent]:
        """
        Detect joints that are commanded to move but not responding.

        Parameters
        ----------
        q_cmd    : (n,) commanded joint velocities [rad/s]
        q_actual : (n,) measured joint positions   [rad]

        Returns
        -------
        List of FaultEvent — one per stalled joint.
        """
        q_cmd    = np.asarray(q_cmd,    dtype=np.float64)
        q_actual = np.asarray(q_actual, dtype=np.float64)
        n        = len(q_cmd)
        faults:  list[FaultEvent] = []

        if self._qpos_prev is None or len(self._qpos_prev) != n:
            self._qpos_prev     = q_actual.copy()
            self._stall_steps   = np.zeros(n, dtype=np.int32)
            return faults

        delta_q = np.abs(q_actual - self._qpos_prev)
        commanded = np.abs(q_cmd) > self.cfg.stall_cmd_threshold
        stuck     = delta_q < self.cfg.stall_deadband

        # Increment stall counter only for joints that are commanded AND stuck
        self._stall_steps[commanded & stuck]  += 1
        # Reset counter for joints that are not commanded or are moving
        self._stall_steps[~(commanded & stuck)] = 0

        for j in np.where(self._stall_steps >= self.cfg.stall_window)[0]:
            fault = FaultEvent(
                fault_type    = FaultType.JOINT_STALL,
                severity      = FaultSeverity.CRITICAL,
                description   = (f"joint {j} stalled — commanded "
                                 f"|dq|={abs(q_cmd[j]):.3f} rad/s "
                                 f"but moved only {delta_q[j]:.2e} rad "
                                 f"over {self._stall_steps[j]} steps"),
                safe_response = "E-stop triggered; arm frozen at current qpos",
                details       = {"joint_index": int(j),
                                 "stall_steps": int(self._stall_steps[j])},
            )
            self._history.append(fault)
            log.error("JOINT STALL: %s", fault)
            self._estop.trigger(reason=fault.description, source="fault_detector")
            faults.append(fault)

        self._qpos_prev = q_actual.copy()
        return faults

    def check_comm(self) -> Optional[FaultEvent]:
        """
        Check the communication watchdog.

        Returns a COMM_FAILURE FaultEvent (and triggers E-stop) if heartbeat()
        has not been called within cfg.watchdog_ms milliseconds.
        Returns None if communication is healthy.
        """
        elapsed_ms = (time.monotonic() - self._last_heartbeat) * 1_000.0
        if elapsed_ms > self.cfg.watchdog_ms:
            fault = FaultEvent(
                fault_type    = FaultType.COMM_FAILURE,
                severity      = FaultSeverity.CRITICAL,
                description   = (f"watchdog fired — no heartbeat for "
                                 f"{elapsed_ms:.0f} ms "
                                 f"(limit {self.cfg.watchdog_ms:.0f} ms)"),
                safe_response = "E-stop triggered immediately",
                details       = {"elapsed_ms": elapsed_ms,
                                 "watchdog_ms": self.cfg.watchdog_ms},
            )
            self._history.append(fault)
            log.critical("COMM FAILURE: %s", fault)
            self._estop.trigger(reason=fault.description, source="fault_detector")
            return fault
        return None

    def check_all(
        self,
        sensordata: np.ndarray,
        q_cmd:      np.ndarray,
        q_actual:   np.ndarray,
    ) -> list[FaultEvent]:
        """
        Run all three fault checks in priority order and return every event.

        Parameters
        ----------
        sensordata : full MuJoCo sensordata array
        q_cmd      : (n,) commanded joint velocities [rad/s]
        q_actual   : (n,) measured joint positions   [rad]

        Returns
        -------
        List of FaultEvent (may be empty if no faults).
        """
        faults: list[FaultEvent] = []

        # 1. Communication — highest priority
        comm_fault = self.check_comm()
        if comm_fault:
            faults.append(comm_fault)

        # 2. Sensor dropout
        faults.extend(self.check_sensor_dropout(sensordata))

        # 3. Joint stall
        faults.extend(self.check_joint_stall(q_cmd, q_actual))

        return faults

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def fault_count(self) -> int:
        """Total faults detected since last reset()."""
        return len(self._history)

    @property
    def history(self) -> list[FaultEvent]:
        """All fault events recorded since last reset() (newest last)."""
        return list(self._history)

    # ── internals ─────────────────────────────────────────────────────────────

    def _make_sensor_fault(
        self,
        severity:    FaultSeverity,
        desc:        str,
        response:    str,
        details:     dict,
        is_critical: bool,
    ) -> FaultEvent:
        fault = FaultEvent(
            fault_type    = FaultType.SENSOR_DROPOUT,
            severity      = severity,
            description   = desc,
            safe_response = response,
            details       = details,
        )
        self._history.append(fault)
        if is_critical:
            log.error("SENSOR FAULT (CRITICAL): %s", fault)
            self._estop.trigger(reason=desc, source="fault_detector")
        else:
            log.warning("SENSOR FAULT (WARNING): %s", fault)
        return fault

    def __repr__(self) -> str:
        return (f"FaultDetector("
                f"faults={self.fault_count}, "
                f"watchdog={self.cfg.watchdog_ms}ms)")
