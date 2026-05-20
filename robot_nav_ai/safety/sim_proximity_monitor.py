"""
safety/sim_proximity_monitor.py — Simulation-native human proximity monitor.

Implements three ISO 10218-2 collaborative robot safety zones using
ground-truth body positions from MuJoCo (no sensor required in sim).

Zone thresholds and velocity scales
─────────────────────────────────────
  SAFE     (> 1.0 m)  : full speed           — scale 1.00
  WARNING  (< 1.0 m)  : slow to 50%          — scale 0.50
  CAUTION  (< 0.5 m)  : slow to 25%          — scale 0.25
  STOP     (< 0.2 m)  : full stop + E-stop   — scale 0.00

How robot–human distance is measured
──────────────────────────────────────
  The 2-D horizontal distance between the robot base XY position and each
  human body XY position is used.  Height is excluded so that a person
  standing directly above/below (e.g. table height) does not cause false
  positives.  Closest human determines the active zone.

MuJoCo integration
───────────────────
  ProximityConfig.human_body_names lists the MuJoCo body names that count
  as "humans" (e.g. ["human", "operator"]).  The monitor resolves their IDs
  once at construction and reads data.body_xpos each step — O(1) per body.

  robot_body_name specifies which body represents the robot's footprint
  (default "base_link").

Usage
─────
    monitor = SimProximityMonitor(model, data, estop)

    # call every control step (10 ms)
    reading = monitor.check()
    velocity_command *= reading.velocity_scale
    if reading.zone == SafetyZone.STOP:
        pass   # estop already triggered inside check()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import mujoco
import numpy as np

log = logging.getLogger(__name__)


# ── safety zones ──────────────────────────────────────────────────────────────

class SafetyZone(Enum):
    SAFE    = "safe"     # > warning_m — full speed
    WARNING = "warning"  # < warning_m — 50% speed
    CAUTION = "caution"  # < caution_m — 25% speed
    STOP    = "stop"     # < stop_m    — full stop + E-stop


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProximityConfig:
    """
    Zone distance thresholds and velocity scales.

    warning_m       : outer zone boundary [m] — robot slows to warning_scale
    caution_m       : middle zone boundary [m] — robot slows to caution_scale
    stop_m          : inner zone boundary [m] — full stop + E-stop
    warning_scale   : velocity multiplier in WARNING zone (default 0.50)
    caution_scale   : velocity multiplier in CAUTION zone (default 0.25)
    robot_body_name : MuJoCo body that represents the robot footprint
    human_body_names: list of MuJoCo body names treated as humans / obstacles
    """
    warning_m:        float      = 1.0
    caution_m:        float      = 0.5
    stop_m:           float      = 0.2
    warning_scale:    float      = 0.50
    caution_scale:    float      = 0.25
    robot_body_name:  str        = "base_link"
    human_body_names: tuple      = ()    # e.g. ("human", "operator")


# ── reading ───────────────────────────────────────────────────────────────────

@dataclass
class ProximityReading:
    """
    Result of one proximity check.

    Fields
    ──────
    zone            : active SafetyZone
    min_distance    : distance to nearest human [m]  (inf if no humans tracked)
    nearest_body    : name of the closest human body  (empty if no humans)
    velocity_scale  : multiply ALL velocity commands by this factor [0.0–1.0]
    estop_triggered : True if this check triggered the E-stop
    """
    zone:            SafetyZone
    min_distance:    float
    nearest_body:    str
    velocity_scale:  float
    estop_triggered: bool = False

    def __repr__(self) -> str:
        return (f"ProximityReading(zone={self.zone.name}, "
                f"dist={self.min_distance:.2f}m, "
                f"scale={self.velocity_scale:.2f})")


# ── monitor ───────────────────────────────────────────────────────────────────

class SimProximityMonitor:
    """
    Ground-truth human proximity monitor for MuJoCo simulation.

    Reads body positions directly from MuJoCo data — no sensor noise,
    no LiDAR clustering needed.  Designed as a drop-in for Phase 6 testing;
    Phase 13 replaces this with the real-sensor ProximityMonitor.

    Parameters
    ----------
    model : mujoco.MjModel
    data  : mujoco.MjData
    estop : SimEStop — triggered if STOP zone is entered
    cfg   : ProximityConfig
    """

    def __init__(self, model, data, estop, cfg: ProximityConfig = ProximityConfig()) -> None:
        self._model = model
        self._data  = data
        self._estop = estop
        self.cfg    = cfg

        # Resolve body IDs once (O(1) per step thereafter)
        self._robot_body_id   = self._resolve_body(cfg.robot_body_name)
        self._human_body_ids  = [
            self._resolve_body(name) for name in cfg.human_body_names
        ]
        self._human_body_ids  = [bid for bid in self._human_body_ids if bid >= 0]

        self._last_reading:   Optional[ProximityReading] = None
        self._stop_count:     int = 0
        self._warning_count:  int = 0

        log.info(
            "SimProximityMonitor: robot=%s (%d), humans=%s, "
            "zones=stop<%.1fm/caution<%.1fm/warning<%.1fm",
            cfg.robot_body_name, self._robot_body_id,
            list(cfg.human_body_names),
            cfg.stop_m, cfg.caution_m, cfg.warning_m,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def check(self) -> ProximityReading:
        """
        Compute human proximity from current MuJoCo state.

        Reads data.body_xpos for all tracked bodies, computes 2-D horizontal
        distances, determines the active zone, and triggers E-stop if needed.

        Returns
        -------
        ProximityReading with zone, distance, and velocity_scale.
        """
        if not self._human_body_ids:
            reading = ProximityReading(
                zone           = SafetyZone.SAFE,
                min_distance   = float("inf"),
                nearest_body   = "",
                velocity_scale = 1.0,
            )
            self._last_reading = reading
            return reading

        robot_xy = self._body_xy(self._robot_body_id)

        min_dist    = float("inf")
        nearest     = ""
        for bid, name in zip(self._human_body_ids, self.cfg.human_body_names):
            human_xy = self._body_xy(bid)
            dist     = float(np.linalg.norm(robot_xy - human_xy))
            if dist < min_dist:
                min_dist = dist
                nearest  = name

        zone  = self._classify(min_dist)
        scale = self._scale(zone)

        estop_now = False
        if zone == SafetyZone.STOP and not self._estop.is_active:
            self._estop.trigger(
                reason=f"human '{nearest}' at {min_dist:.2f}m — STOP zone (<{self.cfg.stop_m}m)",
                source="proximity_monitor",
            )
            estop_now     = True
            self._stop_count += 1
        elif zone == SafetyZone.WARNING:
            self._warning_count += 1
            log.warning(
                "Proximity WARNING: '%s' at %.2fm — slowing to %.0f%%",
                nearest, min_dist, scale * 100,
            )
        elif zone == SafetyZone.CAUTION:
            log.warning(
                "Proximity CAUTION: '%s' at %.2fm — slowing to %.0f%%",
                nearest, min_dist, scale * 100,
            )

        reading = ProximityReading(
            zone           = zone,
            min_distance   = min_dist,
            nearest_body   = nearest,
            velocity_scale = scale,
            estop_triggered= estop_now,
        )
        self._last_reading = reading
        return reading

    def check_positions(
        self,
        robot_xy:         np.ndarray,
        human_positions:  list[np.ndarray],
        human_names:      Optional[list[str]] = None,
    ) -> ProximityReading:
        """
        Check proximity from raw XY positions (no MuJoCo read).

        Useful for unit testing or when positions come from an external source.

        Parameters
        ----------
        robot_xy        : (2,) robot position [m]
        human_positions : list of (2,) human positions [m]
        human_names     : optional list of names (for logging / nearest_body)
        """
        if not human_positions:
            return ProximityReading(
                zone=SafetyZone.SAFE, min_distance=float("inf"),
                nearest_body="", velocity_scale=1.0,
            )

        names = human_names or [f"human_{i}" for i in range(len(human_positions))]
        min_dist = float("inf")
        nearest  = ""
        for pos, name in zip(human_positions, names):
            dist = float(np.linalg.norm(robot_xy - np.asarray(pos)))
            if dist < min_dist:
                min_dist = dist
                nearest  = name

        zone  = self._classify(min_dist)
        scale = self._scale(zone)

        estop_now = False
        if zone == SafetyZone.STOP and not self._estop.is_active:
            self._estop.trigger(
                reason=f"human '{nearest}' at {min_dist:.2f}m",
                source="proximity_monitor",
            )
            estop_now = True

        reading = ProximityReading(
            zone=zone, min_distance=min_dist,
            nearest_body=nearest, velocity_scale=scale,
            estop_triggered=estop_now,
        )
        self._last_reading = reading
        return reading

    def scale_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """
        Apply the current proximity velocity scale to a velocity command.

        Parameters
        ----------
        velocity : any-shape array of velocity commands

        Returns
        -------
        velocity * scale  (or zeros if E-stop is active)
        """
        if self._estop.is_active:
            return np.zeros_like(velocity)
        scale = self._last_reading.velocity_scale if self._last_reading else 1.0
        return velocity * scale

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def last_reading(self) -> Optional[ProximityReading]:
        return self._last_reading

    @property
    def stop_count(self) -> int:
        """Number of STOP-zone events detected this session."""
        return self._stop_count

    @property
    def warning_count(self) -> int:
        """Number of WARNING-zone events detected this session."""
        return self._warning_count

    # ── internals ─────────────────────────────────────────────────────────────

    def _classify(self, dist: float) -> SafetyZone:
        if dist < self.cfg.stop_m:
            return SafetyZone.STOP
        if dist < self.cfg.caution_m:
            return SafetyZone.CAUTION
        if dist < self.cfg.warning_m:
            return SafetyZone.WARNING
        return SafetyZone.SAFE

    def _scale(self, zone: SafetyZone) -> float:
        return {
            SafetyZone.SAFE:    1.00,
            SafetyZone.WARNING: self.cfg.warning_scale,
            SafetyZone.CAUTION: self.cfg.caution_scale,
            SafetyZone.STOP:    0.00,
        }[zone]

    def _body_xy(self, body_id: int) -> np.ndarray:
        """Return XY position of a body from data.body_xpos."""
        return self._data.body_xpos[body_id, :2].copy()

    def _resolve_body(self, name: str) -> int:
        """Return body ID by name, or -1 if not found."""
        try:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                log.warning("ProximityMonitor: body '%s' not found in model", name)
            return bid
        except Exception:
            log.warning("ProximityMonitor: could not resolve body '%s'", name)
            return -1

    def __repr__(self) -> str:
        zone = self._last_reading.zone.name if self._last_reading else "unknown"
        return (f"SimProximityMonitor(zone={zone}, "
                f"humans={len(self._human_body_ids)}, "
                f"stops={self._stop_count})")
