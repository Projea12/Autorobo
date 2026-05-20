"""
tests/test_safety_proximity_fault.py — Tests for SimProximityMonitor and FaultDetector.

No live MuJoCo environment needed:
  - ProximityMonitor is tested via check_positions() (raw XY injection).
  - FaultDetector is tested with synthetic sensordata / qpos arrays.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from safety.sim_estop import SimEStop, SimEStopConfig
from safety.sim_proximity_monitor import (
    ProximityConfig, ProximityReading, SafetyZone, SimProximityMonitor,
)
from safety.fault_detector import (
    FaultConfig, FaultDetector, FaultEvent, FaultSeverity, FaultType,
)


# ── shared helpers ────────────────────────────────────────────────────────────

def _fresh_estop() -> SimEStop:
    return SimEStop()


def _make_monitor(
    warning_m=1.0, caution_m=0.5, stop_m=0.2,
    warning_scale=0.50, caution_scale=0.25,
) -> tuple[SimProximityMonitor, SimEStop]:
    estop = _fresh_estop()
    model = MagicMock()
    data  = MagicMock()
    data.body_xpos = np.zeros((10, 3), dtype=np.float64)
    cfg = ProximityConfig(
        warning_m=warning_m, caution_m=caution_m, stop_m=stop_m,
        warning_scale=warning_scale, caution_scale=caution_scale,
    )
    with patch("mujoco.mj_name2id", return_value=0):
        monitor = SimProximityMonitor(model, data, estop, cfg=cfg)
    return monitor, estop


# ══════════════════════════════════════════════════════════════════════════════
# ProximityConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestProximityConfig:
    def test_defaults(self):
        cfg = ProximityConfig()
        assert cfg.warning_m     == pytest.approx(1.0)
        assert cfg.caution_m     == pytest.approx(0.5)
        assert cfg.stop_m        == pytest.approx(0.2)
        assert cfg.warning_scale == pytest.approx(0.50)
        assert cfg.caution_scale == pytest.approx(0.25)

    def test_frozen(self):
        with pytest.raises(Exception):
            ProximityConfig().warning_m = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# SafetyZone classification
# ══════════════════════════════════════════════════════════════════════════════

class TestZoneClassification:
    def _classify(self, dist: float, monitor: SimProximityMonitor) -> SafetyZone:
        return monitor._classify(dist)

    def test_beyond_warning_is_safe(self):
        m, _ = _make_monitor()
        assert self._classify(2.0, m) == SafetyZone.SAFE

    def test_at_warning_boundary_is_warning(self):
        m, _ = _make_monitor()
        # strictly less than 1.0 → WARNING
        assert self._classify(0.99, m) == SafetyZone.WARNING

    def test_exactly_warning_is_safe(self):
        m, _ = _make_monitor()
        # 1.0 is NOT < 1.0 → SAFE
        assert self._classify(1.0, m) == SafetyZone.SAFE

    def test_between_caution_and_warning_is_warning(self):
        m, _ = _make_monitor()
        assert self._classify(0.7, m) == SafetyZone.WARNING

    def test_below_caution_is_caution(self):
        m, _ = _make_monitor()
        assert self._classify(0.35, m) == SafetyZone.CAUTION

    def test_below_stop_is_stop(self):
        m, _ = _make_monitor()
        assert self._classify(0.10, m) == SafetyZone.STOP

    def test_zero_distance_is_stop(self):
        m, _ = _make_monitor()
        assert self._classify(0.0, m) == SafetyZone.STOP


# ══════════════════════════════════════════════════════════════════════════════
# Velocity scale per zone
# ══════════════════════════════════════════════════════════════════════════════

class TestVelocityScale:
    def test_safe_scale_1(self):
        m, _ = _make_monitor()
        assert m._scale(SafetyZone.SAFE) == pytest.approx(1.0)

    def test_warning_scale_50pct(self):
        m, _ = _make_monitor(warning_scale=0.50)
        assert m._scale(SafetyZone.WARNING) == pytest.approx(0.50)

    def test_caution_scale_25pct(self):
        m, _ = _make_monitor(caution_scale=0.25)
        assert m._scale(SafetyZone.CAUTION) == pytest.approx(0.25)

    def test_stop_scale_zero(self):
        m, _ = _make_monitor()
        assert m._scale(SafetyZone.STOP) == pytest.approx(0.0)

    def test_custom_warning_scale(self):
        m, _ = _make_monitor(warning_scale=0.6)
        assert m._scale(SafetyZone.WARNING) == pytest.approx(0.6)

    def test_custom_caution_scale(self):
        m, _ = _make_monitor(caution_scale=0.15)
        assert m._scale(SafetyZone.CAUTION) == pytest.approx(0.15)


# ══════════════════════════════════════════════════════════════════════════════
# check_positions — raw XY injection (no MuJoCo read)
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckPositions:
    def test_no_humans_returns_safe(self):
        m, estop = _make_monitor()
        reading  = m.check_positions(np.array([0.0, 0.0]), [])
        assert reading.zone == SafetyZone.SAFE
        assert reading.velocity_scale == pytest.approx(1.0)
        assert not estop.is_active

    def test_far_human_safe(self):
        m, estop = _make_monitor()
        reading  = m.check_positions(
            np.array([0.0, 0.0]),
            [np.array([2.0, 0.0])],
        )
        assert reading.zone == SafetyZone.SAFE
        assert reading.velocity_scale == pytest.approx(1.0)
        assert not estop.is_active

    def test_warning_zone_50pct(self):
        m, estop = _make_monitor()
        reading  = m.check_positions(
            np.array([0.0, 0.0]),
            [np.array([0.7, 0.0])],   # 0.7m → WARNING
        )
        assert reading.zone == SafetyZone.WARNING
        assert reading.velocity_scale == pytest.approx(0.50)
        assert not estop.is_active

    def test_caution_zone_25pct(self):
        m, estop = _make_monitor()
        reading  = m.check_positions(
            np.array([0.0, 0.0]),
            [np.array([0.35, 0.0])],   # 0.35m → CAUTION
        )
        assert reading.zone == SafetyZone.CAUTION
        assert reading.velocity_scale == pytest.approx(0.25)
        assert not estop.is_active

    def test_stop_zone_triggers_estop(self):
        m, estop = _make_monitor()
        reading  = m.check_positions(
            np.array([0.0, 0.0]),
            [np.array([0.1, 0.0])],    # 0.1m → STOP
        )
        assert reading.zone == SafetyZone.STOP
        assert reading.velocity_scale == pytest.approx(0.0)
        assert estop.is_active
        assert reading.estop_triggered

    def test_nearest_human_determines_zone(self):
        m, _ = _make_monitor()
        reading = m.check_positions(
            np.array([0.0, 0.0]),
            [np.array([2.0, 0.0]), np.array([0.1, 0.0])],  # second is closest
            human_names=["far", "close"],
        )
        assert reading.zone == SafetyZone.STOP
        assert reading.nearest_body == "close"

    def test_min_distance_correct(self):
        m, _ = _make_monitor()
        reading = m.check_positions(
            np.array([0.0, 0.0]),
            [np.array([3.0, 4.0])],   # distance = 5.0m
        )
        assert reading.min_distance == pytest.approx(5.0)

    def test_stop_zone_no_double_trigger(self):
        m, estop = _make_monitor()
        m.check_positions(np.array([0.0, 0.0]), [np.array([0.1, 0.0])])
        count_before = estop.trigger_count
        m.check_positions(np.array([0.0, 0.0]), [np.array([0.1, 0.0])])
        assert estop.trigger_count == count_before   # already active, no re-trigger

    def test_diagonal_distance_computed_correctly(self):
        m, _ = _make_monitor()
        reading = m.check_positions(
            np.array([1.0, 1.0]),
            [np.array([1.0 + 0.6, 1.0 + 0.8])],   # distance = 1.0m (exactly at boundary)
        )
        assert reading.zone == SafetyZone.SAFE   # 1.0 is NOT < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# scale_velocity
# ══════════════════════════════════════════════════════════════════════════════

class TestScaleVelocity:
    def test_full_speed_when_safe(self):
        m, _  = _make_monitor()
        m.check_positions(np.array([0.0, 0.0]), [np.array([2.0, 0.0])])
        cmd   = np.array([1.0, 0.5])
        result = m.scale_velocity(cmd)
        np.testing.assert_allclose(result, cmd)

    def test_50pct_in_warning(self):
        m, _  = _make_monitor()
        m.check_positions(np.array([0.0, 0.0]), [np.array([0.7, 0.0])])
        result = m.scale_velocity(np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_zeros_when_estop_active(self):
        m, estop = _make_monitor()
        estop.trigger("test")
        result = m.scale_velocity(np.array([1.0, 2.0]))
        np.testing.assert_allclose(result, [0.0, 0.0])


# ══════════════════════════════════════════════════════════════════════════════
# ProximityReading
# ══════════════════════════════════════════════════════════════════════════════

class TestProximityReading:
    def test_repr_contains_zone(self):
        r = ProximityReading(SafetyZone.WARNING, 0.7, "human", 0.5)
        assert "WARNING" in repr(r)

    def test_repr_contains_dist(self):
        r = ProximityReading(SafetyZone.SAFE, 2.0, "", 1.0)
        assert "2.00" in repr(r)

    def test_last_reading_stored(self):
        m, _ = _make_monitor()
        m.check_positions(np.array([0.0, 0.0]), [np.array([0.7, 0.0])])
        assert m.last_reading is not None
        assert m.last_reading.zone == SafetyZone.WARNING


# ══════════════════════════════════════════════════════════════════════════════
# FaultConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestFaultConfig:
    def test_defaults(self):
        cfg = FaultConfig()
        assert cfg.freeze_window       == 10
        assert cfg.stall_cmd_threshold == pytest.approx(0.05)
        assert cfg.stall_deadband      == pytest.approx(1e-4)
        assert cfg.stall_window        == 15
        assert cfg.watchdog_ms         == pytest.approx(500.0)

    def test_frozen(self):
        with pytest.raises(Exception):
            FaultConfig().watchdog_ms = 100.0


# ══════════════════════════════════════════════════════════════════════════════
# FaultEvent
# ══════════════════════════════════════════════════════════════════════════════

class TestFaultEvent:
    def _event(self, ft=FaultType.JOINT_STALL) -> FaultEvent:
        return FaultEvent(
            fault_type=ft, severity=FaultSeverity.CRITICAL,
            description="test fault", safe_response="E-stop",
        )

    def test_str_contains_fault_type(self):
        assert "joint_stall" in str(self._event())

    def test_str_contains_response(self):
        assert "E-stop" in str(self._event())

    def test_timestamp_set(self):
        e = self._event()
        assert e.timestamp > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FaultDetector — sensor dropout
# ══════════════════════════════════════════════════════════════════════════════

def _make_detector(cfg: FaultConfig = None) -> tuple[FaultDetector, SimEStop]:
    estop = _fresh_estop()
    fd    = FaultDetector(estop, cfg=cfg or FaultConfig())
    fd.reset()
    return fd, estop


def _sensordata(size=40, **kwargs) -> np.ndarray:
    """Build a sensordata array, optionally overriding slices."""
    sd = np.zeros(size, dtype=np.float64)
    for (s, e), val in kwargs.items():
        sd[s:e] = val
    return sd


class TestSensorDropout:
    def test_clean_data_no_fault(self):
        fd, estop = _make_detector()
        sd = _sensordata()
        faults = fd.check_sensor_dropout(sd)
        assert faults == []
        assert not estop.is_active

    def test_nan_in_critical_slice_triggers_estop(self):
        cfg = FaultConfig(critical_slices=((28, 34),), warning_slices=())
        fd, estop = _make_detector(cfg)
        sd = _sensordata()
        sd[28] = float("nan")
        faults = fd.check_sensor_dropout(sd)
        assert len(faults) == 1
        assert faults[0].fault_type == FaultType.SENSOR_DROPOUT
        assert faults[0].severity   == FaultSeverity.CRITICAL
        assert estop.is_active

    def test_inf_in_critical_slice_triggers_estop(self):
        cfg = FaultConfig(critical_slices=((28, 34),), warning_slices=())
        fd, estop = _make_detector(cfg)
        sd = _sensordata()
        sd[30] = float("inf")
        faults = fd.check_sensor_dropout(sd)
        assert len(faults) == 1
        assert estop.is_active

    def test_nan_in_warning_slice_no_estop(self):
        cfg = FaultConfig(critical_slices=(), warning_slices=((24, 26),))
        fd, estop = _make_detector(cfg)
        sd = _sensordata()
        sd[24] = float("nan")
        faults = fd.check_sensor_dropout(sd)
        assert len(faults) == 1
        assert faults[0].severity == FaultSeverity.WARNING
        assert not estop.is_active

    def test_frozen_critical_triggers_after_window(self):
        cfg = FaultConfig(freeze_window=3, critical_slices=((28, 34),), warning_slices=())
        fd, estop = _make_detector(cfg)
        sd = _sensordata()
        sd[28:34] = 5.0   # non-zero but frozen
        # first call sets last_good
        fd.check_sensor_dropout(sd)
        # calls 2 and 3 increment freeze count (counts 1, 2)
        fd.check_sensor_dropout(sd)
        fd.check_sensor_dropout(sd)
        # 4th call: freeze_count == 3 >= freeze_window → fault
        faults = fd.check_sensor_dropout(sd)
        assert any(f.fault_type == FaultType.SENSOR_DROPOUT for f in faults)
        assert estop.is_active

    def test_frozen_warning_no_estop_after_window(self):
        cfg = FaultConfig(freeze_window=3, critical_slices=(), warning_slices=((24, 26),))
        fd, estop = _make_detector(cfg)
        sd = _sensordata()
        sd[24:26] = 2.0
        for _ in range(5):
            fd.check_sensor_dropout(sd)
        assert not estop.is_active

    def test_value_change_resets_freeze_count(self):
        cfg = FaultConfig(freeze_window=3, critical_slices=((28, 34),), warning_slices=())
        fd, estop = _make_detector(cfg)
        sd = _sensordata()
        sd[28:34] = 5.0
        fd.check_sensor_dropout(sd)
        fd.check_sensor_dropout(sd)   # freeze_count = 1
        sd[28] = 6.0                  # value changed → reset
        faults = fd.check_sensor_dropout(sd)
        assert faults == []
        assert not estop.is_active

    def test_reset_clears_freeze_state(self):
        cfg = FaultConfig(freeze_window=2, critical_slices=((28, 34),), warning_slices=())
        fd, estop = _make_detector(cfg)
        sd = _sensordata()
        sd[28:34] = 3.0
        fd.check_sensor_dropout(sd)
        fd.check_sensor_dropout(sd)
        fd.reset()                     # clear state
        faults = fd.check_sensor_dropout(sd)
        assert faults == []


# ══════════════════════════════════════════════════════════════════════════════
# FaultDetector — joint stall
# ══════════════════════════════════════════════════════════════════════════════

class TestJointStall:
    def test_no_command_no_stall(self):
        fd, estop = _make_detector()
        q_cmd = np.zeros(6)
        q_pos = np.zeros(6)
        for _ in range(20):
            faults = fd.check_joint_stall(q_cmd, q_pos)
        assert faults == []
        assert not estop.is_active

    def test_moving_joint_no_stall(self):
        cfg = FaultConfig(stall_window=5)
        fd, estop = _make_detector(cfg)
        q_cmd = np.array([0.1, 0, 0, 0, 0, 0])  # commanding joint 0
        q_pos = np.zeros(6)
        for step in range(10):
            q_pos[0] += 0.01   # position IS changing
            faults = fd.check_joint_stall(q_cmd, q_pos)
        assert faults == []
        assert not estop.is_active

    def test_stall_detected_after_window(self):
        cfg = FaultConfig(stall_window=5, stall_cmd_threshold=0.05, stall_deadband=1e-4)
        fd, estop = _make_detector(cfg)
        q_cmd = np.array([0.1, 0, 0, 0, 0, 0])   # commanding joint 0
        q_pos = np.zeros(6)                         # position NOT changing
        fd.check_joint_stall(q_cmd, q_pos)          # first call sets prev
        for _ in range(6):
            faults = fd.check_joint_stall(q_cmd, q_pos)
        assert any(f.fault_type == FaultType.JOINT_STALL for f in faults)
        assert estop.is_active

    def test_stall_fault_identifies_joint(self):
        cfg = FaultConfig(stall_window=3, stall_cmd_threshold=0.05, stall_deadband=1e-4)
        fd, estop = _make_detector(cfg)
        q_cmd = np.array([0, 0, 0.2, 0, 0, 0])   # commanding joint 2
        q_pos = np.zeros(6)
        fd.check_joint_stall(q_cmd, q_pos)
        for _ in range(4):
            faults = fd.check_joint_stall(q_cmd, q_pos)
        assert any(f.details.get("joint_index") == 2 for f in faults)

    def test_reset_clears_stall_counters(self):
        cfg = FaultConfig(stall_window=3)
        fd, estop = _make_detector(cfg)
        q_cmd = np.array([0.1, 0, 0, 0, 0, 0])
        q_pos = np.zeros(6)
        fd.check_joint_stall(q_cmd, q_pos)
        fd.check_joint_stall(q_cmd, q_pos)
        fd.reset()   # clear all stall state
        # after reset, need stall_window steps again before fault
        for _ in range(2):
            faults = fd.check_joint_stall(q_cmd, q_pos)
        assert faults == []

    def test_below_cmd_threshold_no_stall(self):
        cfg = FaultConfig(stall_window=3, stall_cmd_threshold=0.05)
        fd, estop = _make_detector(cfg)
        q_cmd = np.array([0.01, 0, 0, 0, 0, 0])   # below threshold
        q_pos = np.zeros(6)
        fd.check_joint_stall(q_cmd, q_pos)
        for _ in range(5):
            faults = fd.check_joint_stall(q_cmd, q_pos)
        assert faults == []


# ══════════════════════════════════════════════════════════════════════════════
# FaultDetector — communication / watchdog
# ══════════════════════════════════════════════════════════════════════════════

class TestCommFailure:
    def test_heartbeat_keeps_watchdog_alive(self):
        cfg = FaultConfig(watchdog_ms=200.0)
        fd, estop = _make_detector(cfg)
        fd.heartbeat()
        fault = fd.check_comm()
        assert fault is None
        assert not estop.is_active

    def test_watchdog_fires_when_expired(self):
        cfg = FaultConfig(watchdog_ms=10.0)   # 10 ms — easy to expire
        fd, estop = _make_detector(cfg)
        time.sleep(0.02)   # 20 ms > 10 ms
        fault = fd.check_comm()
        assert fault is not None
        assert fault.fault_type == FaultType.COMM_FAILURE
        assert estop.is_active

    def test_watchdog_fault_severity_critical(self):
        cfg = FaultConfig(watchdog_ms=10.0)
        fd, estop = _make_detector(cfg)
        time.sleep(0.02)
        fault = fd.check_comm()
        assert fault.severity == FaultSeverity.CRITICAL

    def test_repeated_heartbeats_prevent_fault(self):
        cfg = FaultConfig(watchdog_ms=50.0)
        fd, estop = _make_detector(cfg)
        for _ in range(5):
            fd.heartbeat()
            time.sleep(0.005)
        fault = fd.check_comm()
        assert fault is None

    def test_watchdog_response_string(self):
        cfg = FaultConfig(watchdog_ms=10.0)
        fd, estop = _make_detector(cfg)
        time.sleep(0.02)
        fault = fd.check_comm()
        assert "E-stop" in fault.safe_response


# ══════════════════════════════════════════════════════════════════════════════
# FaultDetector — check_all
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckAll:
    def test_returns_empty_when_healthy(self):
        fd, estop = _make_detector()
        fd.heartbeat()
        sd    = _sensordata()
        faults = fd.check_all(sd, np.zeros(6), np.zeros(6))
        assert faults == []

    def test_returns_sensor_fault_with_nan(self):
        cfg = FaultConfig(critical_slices=((28, 34),), warning_slices=())
        fd, estop = _make_detector(cfg)
        fd.heartbeat()
        sd = _sensordata()
        sd[28] = float("nan")
        faults = fd.check_all(sd, np.zeros(6), np.zeros(6))
        assert any(f.fault_type == FaultType.SENSOR_DROPOUT for f in faults)

    def test_history_accumulates(self):
        cfg = FaultConfig(critical_slices=((28, 34),), warning_slices=(), watchdog_ms=500)
        fd, estop = _make_detector(cfg)
        fd.heartbeat()
        sd = _sensordata()
        sd[28] = float("nan")
        fd.check_all(sd, np.zeros(6), np.zeros(6))
        assert fd.fault_count >= 1

    def test_reset_clears_history(self):
        cfg = FaultConfig(critical_slices=((28, 34),), warning_slices=())
        fd, estop = _make_detector(cfg)
        fd.heartbeat()
        sd = _sensordata()
        sd[28] = float("nan")
        fd.check_all(sd, np.zeros(6), np.zeros(6))
        fd.reset()
        assert fd.fault_count == 0


# ══════════════════════════════════════════════════════════════════════════════
# FaultDetector — repr
# ══════════════════════════════════════════════════════════════════════════════

class TestFaultDetectorRepr:
    def test_repr(self):
        fd, _ = _make_detector()
        r = repr(fd)
        assert "FaultDetector" in r
        assert "faults=" in r
        assert "watchdog=" in r
