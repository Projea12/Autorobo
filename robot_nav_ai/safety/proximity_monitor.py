"""
proximity_monitor.py — Human Proximity Safety Monitor (Phase 13)

Monitors the distance between the robot and humans (or other obstacles)
and enforces a three-zone safety model:

  Zone 1 — WARNING  (< 1.0m): slow robot to 50% speed, alert
  Zone 2 — CAUTION  (< 0.5m): slow robot to 10% speed, alert loudly
  Zone 3 — STOP     (< 0.2m): trigger emergency stop immediately

Human positions are detected via:
  - Simulation: ground-truth human body positions from MuJoCo
  - Real robot: LiDAR scan clustering + depth camera person detection

This implements ISO 10218-2 collaborative robot safety zones.

Usage:
    from safety.proximity_monitor import ProximityMonitor, SafetyZone

    monitor = ProximityMonitor(interface, emergency_stop, cfg)
    zone, distance = monitor.check_proximity()
    if zone == SafetyZone.STOP:
        emergency_stop.trigger("Human in stop zone")
    elif zone == SafetyZone.CAUTION:
        velocity_scale = 0.1  # slow to 10%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class SafetyZone(Enum):
    """Human proximity safety zones."""
    SAFE = "safe"           # > 1.0m — normal operation
    WARNING = "warning"     # < 1.0m — reduce speed
    CAUTION = "caution"     # < 0.5m — minimal speed
    STOP = "stop"           # < 0.2m — emergency stop


# Zone distance thresholds in metres
ZONE_THRESHOLDS = {
    SafetyZone.WARNING: 1.0,
    SafetyZone.CAUTION: 0.5,
    SafetyZone.STOP: 0.2,
}

# Velocity scaling factors per zone
VELOCITY_SCALE = {
    SafetyZone.SAFE: 1.0,
    SafetyZone.WARNING: 0.5,
    SafetyZone.CAUTION: 0.1,
    SafetyZone.STOP: 0.0,
}


@dataclass
class ProximityReading:
    """
    A single proximity check result.

    Attributes:
        zone: The current safety zone.
        min_distance: Distance to the nearest human/obstacle (metres).
        human_positions: 2D positions of all detected humans (base frame).
        velocity_scale: Recommended velocity scale factor [0.0, 1.0].
        should_stop: True if emergency stop should be triggered.
    """
    zone: SafetyZone
    min_distance: float
    human_positions: list[np.ndarray]
    velocity_scale: float
    should_stop: bool

    def __repr__(self) -> str:
        return (
            f"ProximityReading(zone={self.zone.name}, "
            f"min_dist={self.min_distance:.2f}m, "
            f"vel_scale={self.velocity_scale:.1f})"
        )


class ProximityMonitor:
    """
    Human proximity safety monitor implementing ISO 10218-2 safety zones.

    Detects humans near the robot and enforces velocity limits or E-stop
    based on their distance. Works in both simulation and real robot modes.

    Zone thresholds (configurable):
      WARNING:  < 1.0m — robot slows to 50% speed
      CAUTION:  < 0.5m — robot slows to 10% speed
      STOP:     < 0.2m — emergency stop triggered
    """

    def __init__(
        self,
        interface: Any,
        emergency_stop: Any,
        cfg: Any = None,
    ) -> None:
        """
        Initialise the proximity monitor.

        Args:
            interface: BaseRobotInterface for sensor readings (LiDAR, depth camera).
            emergency_stop: EmergencyStop instance.
            cfg: Optional config with zone distance overrides.
        """
        self.interface = interface
        self.emergency_stop = emergency_stop
        self.cfg = cfg

        # Load zone thresholds (can be overridden by config)
        self._zone_thresholds = dict(ZONE_THRESHOLDS)
        self._velocity_scale = dict(VELOCITY_SCALE)

        self._last_reading: ProximityReading | None = None
        self._stop_triggered_count = 0

        log.info(
            "ProximityMonitor initialised. Safety zones: "
            f"WARNING<{self._zone_thresholds[SafetyZone.WARNING]}m, "
            f"CAUTION<{self._zone_thresholds[SafetyZone.CAUTION]}m, "
            f"STOP<{self._zone_thresholds[SafetyZone.STOP]}m"
        )

    def check_proximity(self) -> ProximityReading:
        """
        Check current human proximity and return safety reading.

        Detects humans from LiDAR scan and/or depth camera.
        Classifies distance into a SafetyZone.

        Returns:
            ProximityReading with zone, distance, and velocity scale.

        TODO: Phase 13 — implement:
            obs = self.interface.get_observation()
            human_positions = self._detect_humans(obs)
            if not human_positions:
                return ProximityReading(SafetyZone.SAFE, float("inf"), [], 1.0, False)
            distances = [np.linalg.norm(pos[:2]) for pos in human_positions]
            min_distance = min(distances)
            zone = self._classify_zone(min_distance)
            if zone == SafetyZone.STOP and not self.emergency_stop.is_triggered:
                self.emergency_stop.trigger(
                    f"Human at {min_distance:.2f}m — STOP zone (<{...}m)"
                )
            return ProximityReading(...)
        """
        raise NotImplementedError(
            "TODO: Phase 13 — implement check_proximity(): "
            "detect humans from LiDAR/depth, classify zone, trigger E-stop if STOP zone."
        )

    def _detect_humans(self, obs: dict[str, Any]) -> list[np.ndarray]:
        """
        Detect human positions from sensor observations.

        In simulation: uses ground-truth human positions from MuJoCo.
        On real robot: LiDAR leg detection + depth camera upper body detection.

        Args:
            obs: Observation dict from BaseRobotInterface.

        Returns:
            List of 2D positions [x, y] (base frame, metres) for each detected human.

        TODO: Phase 13 — implement:
            # Simulation: query MuJoCo for human body position
            # Real robot: cluster LiDAR scan into legs, detect torsos in depth image
        """
        raise NotImplementedError(
            "TODO: Phase 13 — implement _detect_humans() from LiDAR or depth camera."
        )

    def _classify_zone(self, distance: float) -> SafetyZone:
        """
        Classify a distance value into a SafetyZone.

        Args:
            distance: Distance to nearest human in metres.

        Returns:
            SafetyZone for this distance.
        """
        if distance < self._zone_thresholds[SafetyZone.STOP]:
            return SafetyZone.STOP
        elif distance < self._zone_thresholds[SafetyZone.CAUTION]:
            return SafetyZone.CAUTION
        elif distance < self._zone_thresholds[SafetyZone.WARNING]:
            return SafetyZone.WARNING
        else:
            return SafetyZone.SAFE

    def get_velocity_scale(self) -> float:
        """
        Get the recommended velocity scale based on current proximity.

        Returns:
            Scale factor in [0.0, 1.0] to multiply velocity commands by.
            0.0 = robot should stop, 1.0 = full speed allowed.
        """
        if self._last_reading is None:
            return 1.0  # No reading yet — assume safe
        return self._velocity_scale.get(self._last_reading.zone, 1.0)

    def set_zone_thresholds(
        self,
        warning_m: float = 1.0,
        caution_m: float = 0.5,
        stop_m: float = 0.2,
    ) -> None:
        """
        Update the proximity zone distance thresholds.

        Args:
            warning_m: Warning zone distance (metres). Must be > caution_m.
            caution_m: Caution zone distance (metres). Must be > stop_m.
            stop_m: Stop zone distance (metres). Must be > 0.

        Raises:
            ValueError: If thresholds are not strictly decreasing.
        """
        if not (warning_m > caution_m > stop_m > 0):
            raise ValueError(
                f"Zone thresholds must satisfy warning > caution > stop > 0. "
                f"Got: warning={warning_m}, caution={caution_m}, stop={stop_m}"
            )

        self._zone_thresholds[SafetyZone.WARNING] = warning_m
        self._zone_thresholds[SafetyZone.CAUTION] = caution_m
        self._zone_thresholds[SafetyZone.STOP] = stop_m

        log.info(
            f"Proximity thresholds updated: "
            f"WARNING<{warning_m}m, CAUTION<{caution_m}m, STOP<{stop_m}m"
        )

    @property
    def last_reading(self) -> ProximityReading | None:
        """Return the most recent proximity reading."""
        return self._last_reading

    @property
    def stop_triggered_count(self) -> int:
        """Return how many times a STOP event has been triggered this session."""
        return self._stop_triggered_count
