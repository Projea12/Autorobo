"""
tests/test_safety_phase6.py — Tests for Phase 6 safety modules.

Covers:
  SimEStop            — emergency stop state machine
  ForceLimitGuard     — Newton-threshold force checking (via check_raw)
  TrajectoryCollisionChecker — pre-execution collision probe (mocked MuJoCo)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from safety.sim_estop import SimEStop, SimEStopConfig, EStopEvent
from safety.force_limit_guard import ForceLimitGuard, ForceLimitConfig, ForceLimitResult
from safety.collision_checker import (
    CollisionCheckConfig, CollisionResult, CollisionType, TrajectoryCollisionChecker,
)


# ══════════════════════════════════════════════════════════════════════════════
# SimEStop
# ══════════════════════════════════════════════════════════════════════════════

class TestSimEStopDefaults:
    def test_not_active_initially(self):
        assert not SimEStop().is_active

    def test_trigger_count_zero(self):
        assert SimEStop().trigger_count == 0

    def test_event_none_initially(self):
        assert SimEStop().event is None

    def test_history_empty_initially(self):
        assert SimEStop().history == []

    def test_repr_normal(self):
        assert "normal" in repr(SimEStop()).lower()


class TestSimEStopTrigger:
    def test_trigger_activates(self):
        e = SimEStop()
        e.trigger("test")
        assert e.is_active

    def test_trigger_sets_event(self):
        e = SimEStop()
        e.trigger("test reason")
        assert e.event is not None
        assert "test reason" in e.event.reason

    def test_trigger_records_source(self):
        e = SimEStop()
        e.trigger("boom", source="force_limit")
        assert e.event.source == "force_limit"

    def test_trigger_count_increments(self):
        e = SimEStop()
        e.trigger("a")
        e.reset()
        e.trigger("b")
        assert e.trigger_count == 2

    def test_trigger_twice_keeps_first(self):
        e = SimEStop()
        e.trigger("first")
        e.trigger("second")
        assert "first" in e.event.reason

    def test_history_grows(self):
        e = SimEStop()
        e.trigger("a")
        e.reset()
        e.trigger("b")
        assert len(e.history) == 2

    def test_disabled_estop_no_trigger(self):
        cfg = SimEStopConfig(enabled=False)
        e   = SimEStop(cfg=cfg)
        e.trigger("test")
        assert not e.is_active

    def test_repr_active(self):
        e = SimEStop()
        e.trigger("x")
        assert "ACTIVE" in repr(e)


class TestSimEStopReset:
    def test_reset_clears_active(self):
        e = SimEStop()
        e.trigger("x")
        e.reset()
        assert not e.is_active

    def test_reset_clears_event(self):
        e = SimEStop()
        e.trigger("x")
        e.reset()
        assert e.event is None

    def test_reset_when_not_active_is_noop(self):
        e = SimEStop()
        e.reset()   # should not raise
        assert not e.is_active

    def test_history_survives_reset(self):
        e = SimEStop()
        e.trigger("x")
        e.reset()
        assert len(e.history) == 1


class TestSimEStopGate:
    def test_gate_passes_action_when_safe(self):
        e      = SimEStop()
        action = np.array([1.0, 2.0, 3.0])
        result = e.gate(action)
        np.testing.assert_array_equal(result, action)

    def test_gate_zeros_action_when_active(self):
        e      = SimEStop()
        e.trigger("x")
        action = np.array([1.0, 2.0, 3.0])
        result = e.gate(action)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_gate_raises_when_zero_action_false(self):
        cfg    = SimEStopConfig(zero_action_on_stop=False)
        e      = SimEStop(cfg=cfg)
        e.trigger("x")
        with pytest.raises(RuntimeError, match="E-stop is active"):
            e.gate(np.zeros(3))

    def test_gate_preserves_shape(self):
        e      = SimEStop()
        e.trigger("x")
        action = np.ones((3, 4))
        result = e.gate(action)
        assert result.shape == (3, 4)


class TestSimEStopCallback:
    def test_callback_called_on_trigger(self):
        e      = SimEStop()
        called = []
        e.on_trigger(lambda r: called.append(r))
        e.trigger("hi")
        assert len(called) == 1
        assert "hi" in called[0]

    def test_multiple_callbacks(self):
        e   = SimEStop()
        log = []
        e.on_trigger(lambda r: log.append("A"))
        e.on_trigger(lambda r: log.append("B"))
        e.trigger("x")
        assert log == ["A", "B"]

    def test_callback_error_does_not_propagate(self):
        e = SimEStop()
        e.on_trigger(lambda r: (_ for _ in ()).throw(ValueError("bad")))
        e.trigger("x")   # should not raise


class TestSimEStopThreadSafety:
    def test_concurrent_triggers(self):
        e       = SimEStop()
        results = []

        def _trigger(i):
            e.trigger(f"reason-{i}", source="thread")
            results.append(e.trigger_count)

        threads = [threading.Thread(target=_trigger, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one trigger should succeed (others see it already active)
        assert e.trigger_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# ForceLimitGuard
# ══════════════════════════════════════════════════════════════════════════════

def _make_guard(max_force=50.0, max_torque=10.0):
    """Build ForceLimitGuard with a mock MuJoCo model/data and a fresh SimEStop."""
    model = MagicMock()
    data  = MagicMock()
    # sensordata must behave like a numpy array slice
    data.sensordata = np.zeros(40, dtype=np.float64)
    estop = SimEStop()
    cfg   = ForceLimitConfig(max_force_n=max_force, max_torque_nm=max_torque)
    return ForceLimitGuard(model, data, estop, cfg=cfg), estop, data


class TestForceLimitConfig:
    def test_defaults(self):
        cfg = ForceLimitConfig()
        assert cfg.max_force_n   == pytest.approx(50.0)
        assert cfg.max_torque_nm == pytest.approx(10.0)

    def test_frozen(self):
        with pytest.raises(Exception):
            ForceLimitConfig().max_force_n = 1.0

    def test_sensor_slices(self):
        cfg = ForceLimitConfig()
        assert cfg.sensor_force_slice  == (28, 31)
        assert cfg.sensor_torque_slice == (31, 34)


class TestForceLimitResult:
    def test_safe_property(self):
        r = ForceLimitResult(False, 10.0, 1.0, 50.0, 10.0)
        assert r.safe is True

    def test_violated_property(self):
        r = ForceLimitResult(True, 60.0, 1.0, 50.0, 10.0, reason="over")
        assert r.safe is False

    def test_repr_ok(self):
        r = ForceLimitResult(False, 10.0, 1.0, 50.0, 10.0)
        assert "ok" in repr(r)

    def test_repr_violated(self):
        r = ForceLimitResult(True, 60.0, 1.0, 50.0, 10.0, reason="over")
        assert "VIOLATED" in repr(r)


class TestForceLimitGuardCheckRaw:
    def test_within_limits_safe(self):
        guard, estop, _ = _make_guard(max_force=50.0, max_torque=10.0)
        result = guard.check_raw(
            force_vec  = np.array([10.0, 0.0, 0.0]),
            torque_vec = np.array([1.0, 0.0, 0.0]),
        )
        assert result.safe
        assert not estop.is_active

    def test_force_exceeded_triggers_estop(self):
        guard, estop, _ = _make_guard(max_force=50.0)
        result = guard.check_raw(
            force_vec  = np.array([60.0, 0.0, 0.0]),
            torque_vec = np.zeros(3),
        )
        assert result.violated
        assert estop.is_active
        assert "force" in result.reason

    def test_torque_exceeded_triggers_estop(self):
        guard, estop, _ = _make_guard(max_torque=10.0)
        result = guard.check_raw(
            force_vec  = np.zeros(3),
            torque_vec = np.array([0.0, 12.0, 0.0]),
        )
        assert result.violated
        assert estop.is_active
        assert "torque" in result.reason

    def test_exactly_at_limit_is_safe(self):
        guard, estop, _ = _make_guard(max_force=50.0)
        result = guard.check_raw(
            force_vec  = np.array([50.0, 0.0, 0.0]),
            torque_vec = np.zeros(3),
        )
        assert result.safe

    def test_violation_count_increments(self):
        guard, _, _ = _make_guard(max_force=50.0)
        guard.check_raw(np.array([60.0, 0.0, 0.0]), np.zeros(3))
        guard._estop.reset()
        guard.check_raw(np.array([70.0, 0.0, 0.0]), np.zeros(3))
        assert guard.violation_count == 2

    def test_force_magnitude_reported(self):
        guard, _, _ = _make_guard(max_force=50.0)
        result = guard.check_raw(
            force_vec  = np.array([3.0, 4.0, 0.0]),   # magnitude = 5
            torque_vec = np.zeros(3),
        )
        assert result.force_n == pytest.approx(5.0)

    def test_torque_magnitude_reported(self):
        guard, _, _ = _make_guard(max_torque=10.0)
        result = guard.check_raw(
            force_vec  = np.zeros(3),
            torque_vec = np.array([0.0, 3.0, 4.0]),   # magnitude = 5
        )
        assert result.torque_nm == pytest.approx(5.0)


class TestForceLimitGuardCheckSensordata:
    def test_check_reads_sensordata(self):
        guard, estop, data = _make_guard(max_force=50.0)
        # Put a safe force in sensordata[28:31]
        data.sensordata[28] = 10.0
        data.sensordata[29] = 0.0
        data.sensordata[30] = 0.0
        result = guard.check()
        assert result.safe
        assert not estop.is_active

    def test_check_triggers_on_high_force(self):
        guard, estop, data = _make_guard(max_force=50.0)
        data.sensordata[28] = 60.0
        result = guard.check()
        assert result.violated
        assert estop.is_active


class TestForceLimitGuardRepr:
    def test_repr(self):
        guard, _, _ = _make_guard()
        assert "ForceLimitGuard" in repr(guard)
        assert "violations=" in repr(guard)


# ══════════════════════════════════════════════════════════════════════════════
# TrajectoryCollisionChecker
# ══════════════════════════════════════════════════════════════════════════════

def _make_checker(
    check_self=True,
    check_env=True,
    n_contacts=0,
    geomgroup_pairs=None,   # list of (g1, g2) for each contact
):
    """
    Build a TrajectoryCollisionChecker with a fully mocked MuJoCo model/data.

    geomgroup_pairs : if provided, data.contact will have that many contacts
                      with the given geomgroup values.
    """
    import mujoco

    model = MagicMock()
    data  = MagicMock()
    estop = SimEStop()

    # qpos must support copy() and slice assignment
    data.qpos = np.zeros(20, dtype=np.float64)
    data.ctrl = np.zeros(10, dtype=np.float64)

    if geomgroup_pairs is None:
        data.ncon = 0
        data.contact = []
    else:
        data.ncon = len(geomgroup_pairs)
        contacts = []
        for idx, (g1, g2) in enumerate(geomgroup_pairs):
            c = MagicMock()
            c.geom1 = idx * 2
            c.geom2 = idx * 2 + 1
            contacts.append(c)
        data.contact = contacts

        def _geom_group(geom_id):
            for idx, (g1, g2) in enumerate(geomgroup_pairs):
                if geom_id == idx * 2:
                    return g1
                if geom_id == idx * 2 + 1:
                    return g2
            return 0
        model.geom_group.__getitem__ = lambda self, i: _geom_group(i)

    cfg = CollisionCheckConfig(
        check_self=check_self,
        check_env=check_env,
        n_arm_joints=6,
        qpos_arm_start=0,
    )

    with patch("mujoco.mj_fwdPosition"), \
         patch("mujoco.mj_id2name", return_value="test_geom"):
        checker = TrajectoryCollisionChecker(model, data, estop, cfg=cfg)

    return checker, estop, model, data


class TestCollisionCheckConfig:
    def test_defaults(self):
        cfg = CollisionCheckConfig()
        assert cfg.robot_geomgroup == 1
        assert cfg.check_self is True
        assert cfg.check_env  is True
        assert cfg.n_arm_joints == 6

    def test_frozen(self):
        with pytest.raises(Exception):
            CollisionCheckConfig().check_self = False


class TestCollisionResult:
    def test_safe_property(self):
        r = CollisionResult(False, CollisionType.NONE)
        assert r.safe is True

    def test_collision_property(self):
        r = CollisionResult(True, CollisionType.SELF, reason="boom")
        assert r.safe is False

    def test_repr_safe(self):
        assert "safe" in repr(CollisionResult(False, CollisionType.NONE)).lower()

    def test_repr_collision(self):
        r = CollisionResult(True, CollisionType.SELF, "g1", "g2")
        assert "self_collision" in repr(r)


class TestTrajectoryCollisionCheckerNoCollision:
    def test_check_qpos_safe_when_no_contacts(self):
        checker, estop, model, data = _make_checker(n_contacts=0)
        q = np.zeros(6)
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            result = checker.check_qpos(q)
        assert result.safe
        assert not estop.is_active

    def test_check_count_increments(self):
        checker, _, _, _ = _make_checker()
        q = np.zeros(6)
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            checker.check_qpos(q)
            checker.check_qpos(q)
        assert checker.check_count == 2


class TestTrajectoryCollisionCheckerSelfCollision:
    def test_self_collision_detected(self):
        # both geoms in robot group (1)
        checker, estop, model, data = _make_checker(geomgroup_pairs=[(1, 1)])
        q = np.zeros(6)
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="robot_geom"):
            result = checker.check_qpos(q)
        assert result.collision
        assert result.collision_type == CollisionType.SELF
        assert estop.is_active

    def test_self_collision_check_disabled(self):
        checker, estop, model, data = _make_checker(
            check_self=False, geomgroup_pairs=[(1, 1)]
        )
        q = np.zeros(6)
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            result = checker.check_qpos(q)
        assert result.safe


class TestTrajectoryCollisionCheckerEnvCollision:
    def test_env_collision_detected(self):
        # geom1 = robot (group 1), geom2 = env (group 0)
        checker, estop, model, data = _make_checker(geomgroup_pairs=[(1, 0)])
        q = np.zeros(6)
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="some_geom"):
            result = checker.check_qpos(q)
        assert result.collision
        assert result.collision_type == CollisionType.ENV
        assert estop.is_active

    def test_env_collision_check_disabled(self):
        checker, estop, model, data = _make_checker(
            check_env=False, geomgroup_pairs=[(1, 0)]
        )
        q = np.zeros(6)
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            result = checker.check_qpos(q)
        assert result.safe


class TestTrajectoryCollisionCheckerPath:
    def test_safe_path_returns_safe(self):
        checker, estop, _, _ = _make_checker(n_contacts=0)
        q_start = np.zeros(6)
        q_end   = np.ones(6) * 0.1
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            result = checker.check_path(q_start, q_end, n_waypoints=3)
        assert result.safe

    def test_path_collision_has_waypoint_index(self):
        checker, estop, model, data = _make_checker(geomgroup_pairs=[(1, 1)])
        q_start = np.zeros(6)
        q_end   = np.ones(6) * 0.1
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="robot_link"):
            result = checker.check_path(q_start, q_end, n_waypoints=3)
        assert result.collision
        assert result.waypoint_index >= 0
        assert "waypoint" in result.reason

    def test_path_min_two_waypoints(self):
        checker, _, _, _ = _make_checker(n_contacts=0)
        q_start = np.zeros(6)
        q_end   = np.ones(6) * 0.1
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            # n_waypoints=1 is below minimum — should be clamped to 2
            result = checker.check_path(q_start, q_end, n_waypoints=1)
        assert isinstance(result, CollisionResult)


class TestTrajectoryCollisionCheckerRepr:
    def test_repr(self):
        checker, _, _, _ = _make_checker()
        r = repr(checker)
        assert "TrajectoryCollisionChecker" in r
        assert "checks=" in r
        assert "collisions=" in r


class TestCollisionCheckerStateRestored:
    def test_qpos_restored_after_collision(self):
        checker, _, _, data = _make_checker(geomgroup_pairs=[(1, 1)])
        original_qpos = data.qpos.copy()
        q_probe = np.ones(6) * 0.5
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            checker.check_qpos(q_probe)
        np.testing.assert_array_equal(data.qpos, original_qpos)

    def test_ctrl_restored_after_safe_check(self):
        checker, _, _, data = _make_checker(n_contacts=0)
        original_ctrl = data.ctrl.copy()
        with patch("mujoco.mj_fwdPosition"), \
             patch("mujoco.mj_id2name", return_value="g"):
            checker.check_qpos(np.zeros(6))
        np.testing.assert_array_equal(data.ctrl, original_ctrl)
