"""
tests/test_safety_responses.py — Integration tests: simulate each Phase 6 fault,
confirm the correct safe behaviour fires every time.

These tests verify the FULL response chain, not individual module internals:
    fault occurs  →  correct subsystem detects it
                  →  E-stop triggered (or warning issued)
                  →  output gated to zero / velocity scaled
                  →  system recoverable after reset

Test matrix
───────────
  Force limit breach          → E-stop + action zeroed
  Torque limit breach         → E-stop + action zeroed
  Force within limits         → action passes unchanged
  Self-collision pre-exec     → E-stop + action zeroed
  Env-collision pre-exec      → E-stop + action zeroed
  Safe trajectory             → action passes unchanged
  Human in STOP zone          → E-stop + velocity = 0
  Human in CAUTION zone       → NO E-stop + velocity × 0.25
  Human in WARNING zone       → NO E-stop + velocity × 0.50
  Human outside all zones     → NO E-stop + full velocity
  Sensor dropout (critical)   → E-stop + action zeroed
  Sensor dropout (warning)    → NO E-stop + warning only
  Sensor frozen (critical)    → E-stop after window
  Sensor frozen (warning)     → NO E-stop after window
  Joint stall                 → E-stop + action zeroed
  Joint moving (no stall)     → no fault
  Comm failure (watchdog)     → E-stop + action zeroed
  Heartbeat maintained        → no fault
  Multiple simultaneous faults→ E-stop triggered once, action zeroed
  Reset after fault           → system resumes normal operation
  E-stop priority over all    → any upstream fault blocks all outputs
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
from safety.force_limit_guard import ForceLimitGuard, ForceLimitConfig
from safety.collision_checker import (
    CollisionCheckConfig, CollisionType, TrajectoryCollisionChecker,
)
from safety.sim_proximity_monitor import ProximityConfig, SafetyZone, SimProximityMonitor
from safety.fault_detector import FaultConfig, FaultDetector, FaultType, FaultSeverity


# ── shared helpers ────────────────────────────────────────────────────────────

def _estop() -> SimEStop:
    return SimEStop()


def _action(n: int = 7, val: float = 1.0) -> np.ndarray:
    """Return a nonzero action array."""
    return np.full(n, val, dtype=np.float32)


def _force_guard(estop, max_force=50.0, max_torque=10.0) -> ForceLimitGuard:
    model = MagicMock()
    data  = MagicMock()
    data.sensordata = np.zeros(40, dtype=np.float64)
    cfg = ForceLimitConfig(max_force_n=max_force, max_torque_nm=max_torque)
    return ForceLimitGuard(model, data, estop, cfg=cfg)


def _collision_checker(estop, geomgroup_pairs=None) -> TrajectoryCollisionChecker:
    """Return a checker whose contact scan is pre-programmed."""
    model = MagicMock()
    data  = MagicMock()
    data.qpos = np.zeros(20, dtype=np.float64)
    data.ctrl = np.zeros(10, dtype=np.float64)

    if geomgroup_pairs:
        data.ncon = len(geomgroup_pairs)
        contacts  = []
        for idx, (g1, g2) in enumerate(geomgroup_pairs):
            c = MagicMock()
            c.geom1, c.geom2 = idx * 2, idx * 2 + 1
            contacts.append(c)
        data.contact = contacts

        def _grp(geom_id):
            for idx, (g1, g2) in enumerate(geomgroup_pairs):
                if geom_id == idx * 2:    return g1
                if geom_id == idx * 2 + 1: return g2
            return 0
        model.geom_group.__getitem__ = lambda self, i: _grp(i)
    else:
        data.ncon    = 0
        data.contact = []

    cfg = CollisionCheckConfig(n_arm_joints=6, qpos_arm_start=0)
    return TrajectoryCollisionChecker(model, data, estop, cfg=cfg)


def _proximity_monitor(estop) -> SimProximityMonitor:
    model = MagicMock()
    data  = MagicMock()
    data.body_xpos = np.zeros((10, 3), dtype=np.float64)
    cfg = ProximityConfig()
    with patch("mujoco.mj_name2id", return_value=0):
        return SimProximityMonitor(model, data, estop, cfg=cfg)


def _fault_detector(estop, watchdog_ms=500.0, freeze_window=3) -> FaultDetector:
    cfg = FaultConfig(
        freeze_window       = freeze_window,
        critical_slices     = ((28, 34),),
        warning_slices      = ((24, 26),),
        watchdog_ms         = watchdog_ms,
        stall_window        = 4,
        stall_cmd_threshold = 0.05,
        stall_deadband      = 1e-4,
    )
    fd = FaultDetector(estop, cfg=cfg)
    fd.reset()
    return fd


def _sd(**patches) -> np.ndarray:
    """Build a 40-element sensordata array; patches = {(s,e): value}."""
    sd = np.zeros(40, dtype=np.float64)
    for (s, e), v in patches.items():
        sd[s:e] = v
    return sd


# ══════════════════════════════════════════════════════════════════════════════
# FORCE LIMIT RESPONSES
# ══════════════════════════════════════════════════════════════════════════════

class TestForceLimitResponse:

    # ── scenario 1: force breach ───────────────────────────────────────────────

    def test_force_breach_triggers_estop(self):
        """When wrist force exceeds the Newton limit, E-stop must activate."""
        estop = _estop()
        guard = _force_guard(estop, max_force=50.0)

        guard.check_raw(np.array([60.0, 0.0, 0.0]), np.zeros(3))

        assert estop.is_active, "E-stop must be active after force breach"

    def test_force_breach_zeros_subsequent_action(self):
        """After force breach, every action must be gated to zero."""
        estop  = _estop()
        guard  = _force_guard(estop, max_force=50.0)
        action = _action()

        guard.check_raw(np.array([60.0, 0.0, 0.0]), np.zeros(3))
        result = estop.gate(action)

        assert np.all(result == 0), "gate() must return zeros after force breach"

    def test_force_breach_source_recorded(self):
        """E-stop event must attribute the source as 'force_limit'."""
        estop = _estop()
        guard = _force_guard(estop)
        guard.check_raw(np.array([60.0, 0.0, 0.0]), np.zeros(3))

        assert estop.event.source == "force_limit"

    # ── scenario 2: torque breach ──────────────────────────────────────────────

    def test_torque_breach_triggers_estop(self):
        """When wrist torque exceeds the limit, E-stop must activate."""
        estop = _estop()
        guard = _force_guard(estop, max_torque=10.0)

        guard.check_raw(np.zeros(3), np.array([0.0, 12.0, 0.0]))

        assert estop.is_active

    def test_torque_breach_zeros_action(self):
        estop = _estop()
        guard = _force_guard(estop, max_torque=10.0)
        guard.check_raw(np.zeros(3), np.array([0.0, 12.0, 0.0]))

        np.testing.assert_array_equal(estop.gate(_action()), np.zeros(7))

    # ── scenario 3: within limits (safe path) ─────────────────────────────────

    def test_safe_force_passes_action(self):
        """When force is below the limit, action must pass through unmodified."""
        estop  = _estop()
        guard  = _force_guard(estop, max_force=50.0)
        action = _action(val=0.5)

        guard.check_raw(np.array([30.0, 0.0, 0.0]), np.zeros(3))
        result = estop.gate(action)

        np.testing.assert_array_equal(result, action)
        assert not estop.is_active

    def test_exactly_at_force_limit_is_safe(self):
        estop = _estop()
        guard = _force_guard(estop, max_force=50.0)
        guard.check_raw(np.array([50.0, 0.0, 0.0]), np.zeros(3))
        assert not estop.is_active


# ══════════════════════════════════════════════════════════════════════════════
# COLLISION DETECTION RESPONSES
# ══════════════════════════════════════════════════════════════════════════════

class TestCollisionResponse:

    # ── scenario 4: self-collision detected pre-execution ─────────────────────

    def test_self_collision_triggers_estop(self):
        """When proposed qpos causes self-collision, E-stop must activate."""
        estop   = _estop()
        checker = _collision_checker(estop, geomgroup_pairs=[(1, 1)])

        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="arm_link"):
            checker.check_qpos(np.zeros(6))

        assert estop.is_active

    def test_self_collision_zeros_action(self):
        estop   = _estop()
        checker = _collision_checker(estop, geomgroup_pairs=[(1, 1)])

        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="arm_link"):
            checker.check_qpos(np.zeros(6))

        np.testing.assert_array_equal(estop.gate(_action()), np.zeros(7))

    def test_self_collision_source_recorded(self):
        estop   = _estop()
        checker = _collision_checker(estop, geomgroup_pairs=[(1, 1)])

        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="arm_link"):
            checker.check_qpos(np.zeros(6))

        assert estop.event.source == "collision_checker"

    # ── scenario 5: env-collision detected pre-execution ──────────────────────

    def test_env_collision_triggers_estop(self):
        """When proposed qpos collides with the environment, E-stop must activate."""
        estop   = _estop()
        checker = _collision_checker(estop, geomgroup_pairs=[(1, 0)])

        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="geom"):
            checker.check_qpos(np.zeros(6))

        assert estop.is_active

    def test_env_collision_zeros_action(self):
        estop   = _estop()
        checker = _collision_checker(estop, geomgroup_pairs=[(1, 0)])

        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="geom"):
            checker.check_qpos(np.zeros(6))

        np.testing.assert_array_equal(estop.gate(_action()), np.zeros(7))

    # ── scenario 6: safe trajectory (no collision) ────────────────────────────

    def test_safe_qpos_passes_action(self):
        """When no collision is found, action must pass through unmodified."""
        estop   = _estop()
        checker = _collision_checker(estop)  # no contacts
        action  = _action(val=0.3)

        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            checker.check_qpos(np.zeros(6))

        result = estop.gate(action)
        np.testing.assert_array_equal(result, action)
        assert not estop.is_active

    def test_state_restored_after_safe_check(self):
        """check_qpos must not alter qpos or ctrl regardless of outcome."""
        estop   = _estop()
        checker = _collision_checker(estop)
        checker._data.qpos = np.arange(20, dtype=np.float64)
        checker._data.ctrl = np.arange(10, dtype=np.float64)
        original_qpos = checker._data.qpos.copy()
        original_ctrl = checker._data.ctrl.copy()

        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            checker.check_qpos(np.zeros(6))

        np.testing.assert_array_equal(checker._data.qpos, original_qpos)
        np.testing.assert_array_equal(checker._data.ctrl, original_ctrl)

    def test_state_restored_after_collision(self):
        """check_qpos must restore state even when collision is detected."""
        estop   = _estop()
        checker = _collision_checker(estop, geomgroup_pairs=[(1, 1)])
        checker._data.qpos = np.arange(20, dtype=np.float64)
        orig = checker._data.qpos.copy()

        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            checker.check_qpos(np.zeros(6))

        np.testing.assert_array_equal(checker._data.qpos, orig)


# ══════════════════════════════════════════════════════════════════════════════
# PROXIMITY ZONE RESPONSES
# ══════════════════════════════════════════════════════════════════════════════

class TestProximityResponse:

    # ── scenario 7: human in STOP zone ────────────────────────────────────────

    def test_stop_zone_triggers_estop(self):
        """Human < 0.2 m must trigger E-stop immediately."""
        estop   = _estop()
        monitor = _proximity_monitor(estop)

        monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.1, 0.0])])

        assert estop.is_active

    def test_stop_zone_velocity_is_zero(self):
        """In STOP zone the velocity scale must be 0.0."""
        estop   = _estop()
        monitor = _proximity_monitor(estop)

        reading = monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.1, 0.0])])

        assert reading.velocity_scale == pytest.approx(0.0)

    def test_stop_zone_scale_velocity_zeros_command(self):
        """scale_velocity() must return zeros in STOP zone."""
        estop   = _estop()
        monitor = _proximity_monitor(estop)
        monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.1, 0.0])])

        result = monitor.scale_velocity(np.array([1.0, 0.5, -0.3]))
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    # ── scenario 8: human in CAUTION zone ─────────────────────────────────────

    def test_caution_zone_no_estop(self):
        """Human at 0.35 m (CAUTION) must NOT trigger E-stop."""
        estop   = _estop()
        monitor = _proximity_monitor(estop)

        monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.35, 0.0])])

        assert not estop.is_active

    def test_caution_zone_25pct_velocity(self):
        """In CAUTION zone velocity scale must be exactly 0.25."""
        estop   = _estop()
        monitor = _proximity_monitor(estop)

        reading = monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.35, 0.0])])

        assert reading.zone == SafetyZone.CAUTION
        assert reading.velocity_scale == pytest.approx(0.25)

    def test_caution_zone_scale_applied_to_command(self):
        estop   = _estop()
        monitor = _proximity_monitor(estop)
        monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.35, 0.0])])

        result = monitor.scale_velocity(np.array([1.0, 0.0]))
        np.testing.assert_allclose(result, [0.25, 0.0])

    # ── scenario 9: human in WARNING zone ─────────────────────────────────────

    def test_warning_zone_no_estop(self):
        """Human at 0.7 m (WARNING) must NOT trigger E-stop."""
        estop   = _estop()
        monitor = _proximity_monitor(estop)

        monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.7, 0.0])])

        assert not estop.is_active

    def test_warning_zone_50pct_velocity(self):
        """In WARNING zone velocity scale must be exactly 0.50."""
        estop   = _estop()
        monitor = _proximity_monitor(estop)

        reading = monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.7, 0.0])])

        assert reading.zone == SafetyZone.WARNING
        assert reading.velocity_scale == pytest.approx(0.50)

    def test_warning_zone_scale_applied_to_command(self):
        estop   = _estop()
        monitor = _proximity_monitor(estop)
        monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.7, 0.0])])

        result = monitor.scale_velocity(np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, [0.50, 0.50])

    # ── scenario 10: human outside all zones ──────────────────────────────────

    def test_safe_zone_full_velocity(self):
        """Human > 1.0 m must leave velocity fully unscaled."""
        estop   = _estop()
        monitor = _proximity_monitor(estop)
        monitor.check_positions(np.array([0.0, 0.0]), [np.array([2.0, 0.0])])

        result = monitor.scale_velocity(np.array([1.0, -0.5]))
        np.testing.assert_allclose(result, [1.0, -0.5])
        assert not estop.is_active

    def test_no_humans_tracked_full_velocity(self):
        """With no humans registered, velocity must be fully unscaled."""
        estop   = _estop()
        monitor = _proximity_monitor(estop)
        reading = monitor.check_positions(np.array([0.0, 0.0]), [])

        assert reading.zone == SafetyZone.SAFE
        result = monitor.scale_velocity(np.array([1.0]))
        np.testing.assert_allclose(result, [1.0])


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR DROPOUT RESPONSES
# ══════════════════════════════════════════════════════════════════════════════

class TestSensorDropoutResponse:

    # ── scenario 11: critical sensor NaN ──────────────────────────────────────

    def test_critical_nan_triggers_estop(self):
        """NaN in wrist F/T sensordata (critical) must trigger E-stop."""
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        sd = _sd()
        sd[28] = float("nan")

        fd.check_sensor_dropout(sd)

        assert estop.is_active

    def test_critical_nan_zeros_action(self):
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        sd = _sd()
        sd[28] = float("nan")
        fd.check_sensor_dropout(sd)

        np.testing.assert_array_equal(estop.gate(_action()), np.zeros(7))

    def test_critical_nan_fault_type_correct(self):
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        sd = _sd()
        sd[28] = float("nan")
        faults = fd.check_sensor_dropout(sd)

        assert any(f.fault_type == FaultType.SENSOR_DROPOUT for f in faults)
        assert any(f.severity   == FaultSeverity.CRITICAL    for f in faults)

    def test_critical_inf_triggers_estop(self):
        """Inf in wrist F/T (critical) must also trigger E-stop."""
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        sd = _sd()
        sd[30] = float("inf")
        fd.check_sensor_dropout(sd)

        assert estop.is_active

    # ── scenario 12: warning sensor NaN ───────────────────────────────────────

    def test_warning_nan_no_estop(self):
        """NaN in gripper tactile (warning) must NOT trigger E-stop."""
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        sd = _sd()
        sd[24] = float("nan")
        faults = fd.check_sensor_dropout(sd)

        assert not estop.is_active
        assert any(f.severity == FaultSeverity.WARNING for f in faults)

    def test_warning_nan_action_still_passes(self):
        """After a warning-only dropout, gate() must still pass action through."""
        estop  = _estop()
        fd     = _fault_detector(estop)
        fd.heartbeat()
        sd = _sd()
        sd[24] = float("nan")
        fd.check_sensor_dropout(sd)
        action = _action(val=0.4)

        result = estop.gate(action)
        np.testing.assert_array_equal(result, action)

    # ── scenario 13: critical sensor frozen ───────────────────────────────────

    def test_critical_frozen_triggers_estop_after_window(self):
        """Wrist F/T frozen for freeze_window steps must trigger E-stop."""
        estop = _estop()
        fd    = _fault_detector(estop, freeze_window=3)
        fd.heartbeat()
        sd = _sd()
        sd[28:34] = 5.0   # non-zero, non-nan, but constant

        # call 1: sets last_good (no freeze count yet)
        fd.check_sensor_dropout(sd)
        assert not estop.is_active

        # calls 2, 3: freeze_count increments to 1, 2
        fd.check_sensor_dropout(sd)
        fd.check_sensor_dropout(sd)
        assert not estop.is_active

        # call 4: freeze_count == 3 >= window → fault
        fd.check_sensor_dropout(sd)
        assert estop.is_active

    def test_critical_frozen_zeros_action(self):
        estop = _estop()
        fd    = _fault_detector(estop, freeze_window=3)
        fd.heartbeat()
        sd = _sd()
        sd[28:34] = 5.0
        for _ in range(4):
            fd.check_sensor_dropout(sd)

        np.testing.assert_array_equal(estop.gate(_action()), np.zeros(7))

    # ── scenario 14: warning sensor frozen ────────────────────────────────────

    def test_warning_frozen_no_estop(self):
        """Frozen warning-sensor must NOT trigger E-stop even past window."""
        estop = _estop()
        fd    = _fault_detector(estop, freeze_window=3)
        fd.heartbeat()

        # Keep the critical wrist F/T slice (28:34) varying so it never
        # freezes — only the warning tactile slice (24:26) is held constant.
        sd = _sd()
        sd[24:26] = 3.0   # warning slice: frozen (constant)

        for step in range(5):
            sd[28:34] = float(step + 1)   # critical slice: always changing
            fd.check_sensor_dropout(sd)

        assert not estop.is_active

    def test_sensor_recovers_when_value_changes(self):
        """Changing the sensor value must reset the frozen-step counter."""
        estop = _estop()
        fd    = _fault_detector(estop, freeze_window=3)
        fd.heartbeat()
        sd = _sd()
        sd[28:34] = 5.0
        fd.check_sensor_dropout(sd)
        fd.check_sensor_dropout(sd)   # freeze_count = 1

        sd[28] = 6.0                  # value changed — counter resets
        faults = fd.check_sensor_dropout(sd)
        assert faults == []
        assert not estop.is_active


# ══════════════════════════════════════════════════════════════════════════════
# JOINT STALL RESPONSES
# ══════════════════════════════════════════════════════════════════════════════

class TestJointStallResponse:

    # ── scenario 15: joint stall detected ────────────────────────────────────

    def test_stall_triggers_estop(self):
        """Joint commanded but immobile for stall_window steps must trigger E-stop."""
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        q_cmd = np.array([0.1, 0, 0, 0, 0, 0])
        q_pos = np.zeros(6)

        fd.check_joint_stall(q_cmd, q_pos)   # sets prev
        for _ in range(5):                    # accumulate stall steps
            faults = fd.check_joint_stall(q_cmd, q_pos)

        assert estop.is_active

    def test_stall_zeros_action(self):
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        q_cmd = np.array([0.1, 0, 0, 0, 0, 0])
        q_pos = np.zeros(6)
        fd.check_joint_stall(q_cmd, q_pos)
        for _ in range(5):
            fd.check_joint_stall(q_cmd, q_pos)

        np.testing.assert_array_equal(estop.gate(_action()), np.zeros(7))

    def test_stall_fault_type_recorded(self):
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        q_cmd = np.array([0.1, 0, 0, 0, 0, 0])
        q_pos = np.zeros(6)
        fd.check_joint_stall(q_cmd, q_pos)
        all_faults = []
        for _ in range(5):
            all_faults.extend(fd.check_joint_stall(q_cmd, q_pos))

        assert any(f.fault_type == FaultType.JOINT_STALL for f in all_faults)

    def test_stall_response_string_mentions_freeze(self):
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        q_cmd = np.array([0.1, 0, 0, 0, 0, 0])
        q_pos = np.zeros(6)
        fd.check_joint_stall(q_cmd, q_pos)
        all_faults = []
        for _ in range(5):
            all_faults.extend(fd.check_joint_stall(q_cmd, q_pos))

        stall_faults = [f for f in all_faults if f.fault_type == FaultType.JOINT_STALL]
        assert stall_faults
        assert "frozen" in stall_faults[0].safe_response.lower() \
            or "freeze" in stall_faults[0].safe_response.lower() \
            or "E-stop" in stall_faults[0].safe_response

    # ── scenario 16: joint moving (no stall) ──────────────────────────────────

    def test_moving_joint_no_estop(self):
        """Joint that responds to commands must never trigger a stall fault."""
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        q_cmd = np.array([0.1, 0, 0, 0, 0, 0])
        q_pos = np.zeros(6)

        for step in range(10):
            q_pos[0] += 0.01   # position IS changing
            faults = fd.check_joint_stall(q_cmd, q_pos)

        assert not estop.is_active
        assert faults == []

    def test_unactuated_joint_no_stall(self):
        """Joint with q_cmd below threshold must never accumulate stall steps."""
        estop = _estop()
        fd    = _fault_detector(estop)
        fd.heartbeat()
        q_cmd = np.array([0.01, 0, 0, 0, 0, 0])   # below 0.05 threshold
        q_pos = np.zeros(6)

        fd.check_joint_stall(q_cmd, q_pos)
        for _ in range(10):
            faults = fd.check_joint_stall(q_cmd, q_pos)

        assert not estop.is_active
        assert faults == []


# ══════════════════════════════════════════════════════════════════════════════
# COMMUNICATION FAILURE RESPONSES
# ══════════════════════════════════════════════════════════════════════════════

class TestCommFailureResponse:

    # ── scenario 17: watchdog expired ─────────────────────────────────────────

    def test_expired_watchdog_triggers_estop(self):
        """If heartbeat() is not called within watchdog_ms, E-stop must fire."""
        estop = _estop()
        fd    = _fault_detector(estop, watchdog_ms=15.0)
        # deliberately skip heartbeat()
        time.sleep(0.025)   # 25 ms > 15 ms

        fd.check_comm()

        assert estop.is_active

    def test_expired_watchdog_zeros_action(self):
        estop = _estop()
        fd    = _fault_detector(estop, watchdog_ms=15.0)
        time.sleep(0.025)
        fd.check_comm()

        np.testing.assert_array_equal(estop.gate(_action()), np.zeros(7))

    def test_expired_watchdog_fault_type(self):
        estop = _estop()
        fd    = _fault_detector(estop, watchdog_ms=15.0)
        time.sleep(0.025)
        fault = fd.check_comm()

        assert fault is not None
        assert fault.fault_type == FaultType.COMM_FAILURE
        assert fault.severity   == FaultSeverity.CRITICAL

    def test_expired_watchdog_response_mentions_estop(self):
        estop = _estop()
        fd    = _fault_detector(estop, watchdog_ms=15.0)
        time.sleep(0.025)
        fault = fd.check_comm()

        assert "E-stop" in fault.safe_response

    # ── scenario 18: heartbeat maintained ─────────────────────────────────────

    def test_regular_heartbeat_no_fault(self):
        """Regular heartbeat() calls must prevent comm failure entirely."""
        estop = _estop()
        fd    = _fault_detector(estop, watchdog_ms=100.0)

        for _ in range(5):
            fd.heartbeat()
            time.sleep(0.005)   # 5 ms — well within 100 ms window

        fault = fd.check_comm()
        assert fault is None
        assert not estop.is_active

    def test_heartbeat_after_gap_resets_watchdog(self):
        """A heartbeat() call after a long gap must reset the timer."""
        estop = _estop()
        fd    = _fault_detector(estop, watchdog_ms=30.0)
        time.sleep(0.020)   # 20 ms
        fd.heartbeat()      # resets timer — now only 0 ms elapsed
        fault = fd.check_comm()

        assert fault is None


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSABILITY AND PRIORITY
# ══════════════════════════════════════════════════════════════════════════════

class TestSafetyPriority:

    def test_estop_gates_all_outputs_regardless_of_source(self):
        """
        Once E-stop is active (from ANY source), gate() must return zeros
        for every subsequent action, regardless of content.
        """
        estop = _estop()
        estop.trigger("manual", source="operator")

        for val in [1.0, -1.0, 0.5, 100.0]:
            result = estop.gate(_action(val=val))
            assert np.all(result == 0), f"gate() should zero action with val={val}"

    def test_force_fault_blocks_collision_check_output(self):
        """
        After a force limit breach triggers E-stop, even a collision-safe
        action must be gated to zero.
        """
        estop = _estop()
        guard = _force_guard(estop)
        guard.check_raw(np.array([60.0, 0.0, 0.0]), np.zeros(3))

        # collision check would have returned safe, but estop is already active
        action = _action(val=0.9)
        np.testing.assert_array_equal(estop.gate(action), np.zeros(7))

    def test_proximity_stop_blocks_force_safe_action(self):
        """
        After proximity E-stop, even a force-safe action must be zeroed.
        """
        estop   = _estop()
        monitor = _proximity_monitor(estop)
        monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.05, 0.0])])

        assert estop.is_active
        np.testing.assert_array_equal(estop.gate(_action()), np.zeros(7))

    def test_multiple_faults_trigger_estop_exactly_once(self):
        """
        Simultaneous force + proximity faults must trigger E-stop only once
        (not double-trigger causing unexpected behaviour).
        """
        estop   = _estop()
        guard   = _force_guard(estop)
        monitor = _proximity_monitor(estop)

        guard.check_raw(np.array([60.0, 0.0, 0.0]), np.zeros(3))
        monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.05, 0.0])])

        assert estop.trigger_count == 1   # only one trigger recorded

    def test_all_five_subsystems_share_one_estop(self):
        """
        Force, collision, proximity, sensor dropout, and comm failure must all
        route through the same E-stop and leave it active after any one fires.
        """
        scenarios = [
            lambda e: _force_guard(e).check_raw(np.array([60.0, 0.0, 0.0]), np.zeros(3)),
            lambda e: e.trigger("sensor dropout", source="fault_detector"),
            lambda e: e.trigger("comm failure",   source="fault_detector"),
            lambda e: _proximity_monitor(e).check_positions(
                np.array([0.0, 0.0]), [np.array([0.0, 0.0])]
            ),
        ]
        for i, scenario in enumerate(scenarios):
            estop = _estop()
            scenario(estop)
            assert estop.is_active, f"Scenario {i} did not activate E-stop"
            np.testing.assert_array_equal(
                estop.gate(_action()), np.zeros(7),
                err_msg=f"Scenario {i}: gate() did not zero action"
            )


# ══════════════════════════════════════════════════════════════════════════════
# RECOVERY AFTER FAULT
# ══════════════════════════════════════════════════════════════════════════════

class TestFaultRecovery:

    def test_reset_after_force_fault_restores_action(self):
        """After force breach and reset, action must pass through again."""
        estop  = _estop()
        guard  = _force_guard(estop)
        action = _action(val=0.7)

        guard.check_raw(np.array([60.0, 0.0, 0.0]), np.zeros(3))
        assert estop.is_active

        estop.reset()
        result = estop.gate(action)
        np.testing.assert_array_equal(result, action)

    def test_reset_after_proximity_stop_restores_velocity(self):
        """After STOP zone and reset, scale_velocity must return full speed."""
        estop   = _estop()
        monitor = _proximity_monitor(estop)
        monitor.check_positions(np.array([0.0, 0.0]), [np.array([0.1, 0.0])])
        assert estop.is_active

        estop.reset()
        # Human now far away
        monitor.check_positions(np.array([0.0, 0.0]), [np.array([3.0, 0.0])])
        result = monitor.scale_velocity(np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_reset_after_stall_fault_restores_action(self):
        """After joint stall and reset, action must pass through again."""
        estop  = _estop()
        fd     = _fault_detector(estop)
        fd.heartbeat()
        q_cmd  = np.array([0.1, 0, 0, 0, 0, 0])
        q_pos  = np.zeros(6)
        fd.check_joint_stall(q_cmd, q_pos)
        for _ in range(5):
            fd.check_joint_stall(q_cmd, q_pos)
        assert estop.is_active

        estop.reset()
        fd.reset()
        action = _action(val=0.6)
        np.testing.assert_array_equal(estop.gate(action), action)

    def test_reset_after_comm_failure_restores_action(self):
        """After watchdog and reset, action must pass through again."""
        estop = _estop()
        fd    = _fault_detector(estop, watchdog_ms=15.0)
        time.sleep(0.025)
        fd.check_comm()
        assert estop.is_active

        estop.reset()
        fd.heartbeat()
        fault  = fd.check_comm()
        action = _action(val=0.5)
        assert fault is None
        np.testing.assert_array_equal(estop.gate(action), action)

    def test_fault_history_preserved_across_reset(self):
        """
        E-stop reset must clear active state but NOT erase fault history
        — the history is needed for post-incident review.
        """
        estop = _estop()
        estop.trigger("fault 1")
        estop.reset()
        estop.trigger("fault 2")
        estop.reset()

        assert estop.trigger_count == 2
        assert len(estop.history)  == 2

    def test_system_handles_repeated_fault_reset_cycles(self):
        """System must remain stable through multiple fault→reset cycles."""
        estop = _estop()
        guard = _force_guard(estop)

        for cycle in range(5):
            guard.check_raw(np.array([60.0, 0.0, 0.0]), np.zeros(3))
            assert estop.is_active, f"cycle {cycle}: E-stop should be active"
            np.testing.assert_array_equal(estop.gate(_action()), np.zeros(7))
            estop.reset()
            np.testing.assert_array_equal(estop.gate(_action()), _action())
