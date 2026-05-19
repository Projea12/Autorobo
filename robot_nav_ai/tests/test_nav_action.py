"""
tests/test_nav_action.py — Unit tests for env/nav_action.py.

Covers: ActionConfig validation, ActionProcessor pipeline (scaling,
smoothing, rate limiting), differential-drive kinematics, and
inverse kinematics.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.nav_action import (
    ActionConfig, ActionProcessor, PhysicalAction,
    LIN_VEL_MAX, ANG_VEL_MAX, WHEEL_RADIUS, WHEELBASE, WHEEL_VEL_MAX,
    make_action_space, differential_drive, inverse_differential_drive,
)


# ── ActionConfig ──────────────────────────────────────────────────────────────

class TestActionConfig:
    def test_defaults(self):
        cfg = ActionConfig()
        assert cfg.lin_vel_max == pytest.approx(LIN_VEL_MAX)
        assert cfg.ang_vel_max == pytest.approx(ANG_VEL_MAX)
        assert cfg.wheel_radius == pytest.approx(WHEEL_RADIUS)
        assert cfg.wheelbase    == pytest.approx(WHEELBASE)
        assert cfg.smoothing_alpha == pytest.approx(1.0)
        assert cfg.lin_acc_max is None
        assert cfg.ang_acc_max is None

    def test_frozen(self):
        cfg = ActionConfig()
        with pytest.raises(Exception):
            cfg.lin_vel_max = 99.0

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="smoothing_alpha"):
            ActionConfig(smoothing_alpha=0.0)

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError):
            ActionConfig(smoothing_alpha=-0.1)

    def test_valid_alpha_one(self):
        cfg = ActionConfig(smoothing_alpha=1.0)
        assert cfg.smoothing_alpha == 1.0

    def test_invalid_lin_vel_max(self):
        with pytest.raises(ValueError):
            ActionConfig(lin_vel_max=0.0)

    def test_invalid_ang_vel_max(self):
        with pytest.raises(ValueError):
            ActionConfig(ang_vel_max=-1.0)

    def test_invalid_wheel_radius(self):
        with pytest.raises(ValueError):
            ActionConfig(wheel_radius=0.0)

    def test_invalid_wheelbase(self):
        with pytest.raises(ValueError):
            ActionConfig(wheelbase=0.0)

    def test_wheel_vel_max_formula(self):
        cfg  = ActionConfig(lin_vel_max=1.0, ang_vel_max=2.0,
                            wheel_radius=0.1, wheelbase=0.4)
        # (1.0 + 0.2 × 2.0) / 0.1 = 14.0
        assert cfg.wheel_vel_max == pytest.approx(14.0)

    def test_custom_values(self):
        cfg = ActionConfig(lin_vel_max=2.0, ang_vel_max=3.0,
                           smoothing_alpha=0.5,
                           lin_acc_max=4.0, ang_acc_max=8.0)
        assert cfg.lin_vel_max   == pytest.approx(2.0)
        assert cfg.smoothing_alpha == pytest.approx(0.5)
        assert cfg.lin_acc_max   == pytest.approx(4.0)


# ── ActionProcessor construction ──────────────────────────────────────────────

class TestActionProcessorConstruction:
    def test_invalid_dt(self):
        with pytest.raises(ValueError):
            ActionProcessor(dt_env=0.0)

    def test_invalid_dt_negative(self):
        with pytest.raises(ValueError):
            ActionProcessor(dt_env=-0.01)

    def test_cfg_accessible(self):
        cfg  = ActionConfig(lin_vel_max=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        assert proc.cfg is cfg

    def test_initial_smoothed_zeros(self):
        proc = ActionProcessor()
        np.testing.assert_array_equal(proc.smoothed_action, [0.0, 0.0])

    def test_initial_prev_physical_zeros(self):
        proc = ActionProcessor()
        np.testing.assert_array_equal(proc.prev_physical, [0.0, 0.0])


# ── ActionProcessor.reset ─────────────────────────────────────────────────────

class TestActionProcessorReset:
    def test_reset_clears_smoothed(self):
        proc = ActionProcessor(cfg=ActionConfig(smoothing_alpha=0.5))
        proc.process(np.array([1.0, 1.0]))
        proc.reset()
        np.testing.assert_array_equal(proc.smoothed_action, [0.0, 0.0])

    def test_reset_clears_prev_physical(self):
        proc = ActionProcessor(cfg=ActionConfig(lin_acc_max=1.0), dt_env=0.01)
        proc.process(np.array([0.5, 0.0]))
        proc.reset()
        np.testing.assert_array_equal(proc.prev_physical, [0.0, 0.0])

    def test_reset_restores_unclipped_output(self):
        cfg  = ActionConfig(lin_acc_max=0.1, smoothing_alpha=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        # After large step, vel is limited; after reset it's back at zero
        proc.process(np.array([1.0, 0.0]))
        proc.reset()
        phys = proc.process(np.array([0.0, 0.0]))
        assert phys.v_linear == pytest.approx(0.0, abs=1e-6)


# ── ActionProcessor.process — no smoothing, no rate limit ─────────────────────

class TestActionProcessorProcess:
    def test_returns_physical_action(self):
        proc = ActionProcessor()
        proc.reset()
        result = proc.process(np.array([0.0, 0.0]))
        assert isinstance(result, PhysicalAction)

    def test_zero_action_zero_velocity(self):
        proc = ActionProcessor()
        proc.reset()
        p = proc.process(np.array([0.0, 0.0]))
        assert p.v_linear  == pytest.approx(0.0, abs=1e-6)
        assert p.v_angular == pytest.approx(0.0, abs=1e-6)

    def test_full_forward_maps_to_lin_vel_max(self):
        cfg  = ActionConfig(smoothing_alpha=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        p = proc.process(np.array([1.0, 0.0]))
        assert p.v_linear == pytest.approx(cfg.lin_vel_max, rel=1e-5)

    def test_full_reverse_maps_to_neg_lin_vel_max(self):
        cfg  = ActionConfig(smoothing_alpha=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        p = proc.process(np.array([-1.0, 0.0]))
        assert p.v_linear == pytest.approx(-cfg.lin_vel_max, rel=1e-5)

    def test_full_left_maps_to_ang_vel_max(self):
        cfg  = ActionConfig(smoothing_alpha=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        p = proc.process(np.array([0.0, 1.0]))
        assert p.v_angular == pytest.approx(cfg.ang_vel_max, rel=1e-5)

    def test_full_right_maps_to_neg_ang_vel_max(self):
        cfg  = ActionConfig(smoothing_alpha=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        p = proc.process(np.array([0.0, -1.0]))
        assert p.v_angular == pytest.approx(-cfg.ang_vel_max, rel=1e-5)

    def test_action_clipped_outside_pm1(self):
        proc = ActionProcessor()
        proc.reset()
        p = proc.process(np.array([5.0, -5.0]))
        assert p.v_linear  == pytest.approx(proc.cfg.lin_vel_max, rel=1e-5)
        assert p.v_angular == pytest.approx(-proc.cfg.ang_vel_max, rel=1e-5)

    def test_raw_stored_in_physical_action(self):
        proc = ActionProcessor()
        proc.reset()
        raw = np.array([0.3, -0.7], dtype=np.float32)
        p   = proc.process(raw)
        np.testing.assert_allclose(p.raw, np.clip(raw, -1, 1), atol=1e-6)

    def test_wheel_velocities_symmetric_forward(self):
        """v_lin > 0, v_ang = 0 → ctrl_left == ctrl_right."""
        proc = ActionProcessor()
        proc.reset()
        p = proc.process(np.array([1.0, 0.0]))
        assert p.ctrl_left == pytest.approx(p.ctrl_right, rel=1e-5)

    def test_spin_in_place_opposite_wheels(self):
        """v_lin = 0, v_ang > 0 → ctrl_left < 0, ctrl_right > 0."""
        proc = ActionProcessor()
        proc.reset()
        p = proc.process(np.array([0.0, 1.0]))
        assert p.ctrl_left  < 0
        assert p.ctrl_right > 0

    def test_ctrl_within_wheel_vel_max(self):
        proc = ActionProcessor()
        proc.reset()
        p = proc.process(np.array([1.0, 1.0]))
        assert abs(p.ctrl_left)  <= proc.cfg.wheel_vel_max + 1e-9
        assert abs(p.ctrl_right) <= proc.cfg.wheel_vel_max + 1e-9


# ── Smoothing ─────────────────────────────────────────────────────────────────

class TestSmoothing:
    def test_no_smoothing_alpha_1(self):
        """α=1 → smoothed == raw after one step (from zero)."""
        cfg  = ActionConfig(smoothing_alpha=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        p = proc.process(np.array([0.8, 0.0]))
        assert p.smoothed[0] == pytest.approx(0.8, rel=1e-5)

    def test_ema_alpha_half(self):
        """α=0.5 → smoothed = 0.5×0 + 0.5×1.0 = 0.5 on first step."""
        cfg  = ActionConfig(smoothing_alpha=0.5)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        p = proc.process(np.array([1.0, 0.0]))
        assert p.smoothed[0] == pytest.approx(0.5, rel=1e-5)

    def test_ema_converges_to_target(self):
        """Repeated constant action should drive smoothed → raw."""
        cfg  = ActionConfig(smoothing_alpha=0.3)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        for _ in range(100):
            p = proc.process(np.array([1.0, 0.0]))
        assert p.smoothed[0] == pytest.approx(1.0, abs=1e-3)

    def test_smoothed_changes_slower_than_raw(self):
        """Sudden change should be attenuated by EMA."""
        cfg  = ActionConfig(smoothing_alpha=0.2)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        # Warm up at 0
        for _ in range(20):
            proc.process(np.array([0.0, 0.0]))
        # Sudden full-forward
        p = proc.process(np.array([1.0, 0.0]))
        # Smoothed should be less than raw (0.2 × 1.0 = 0.2)
        assert p.smoothed[0] < 1.0
        assert p.smoothed[0] > 0.0


# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimit:
    def test_no_rate_limit_by_default(self):
        """Default ActionConfig has no rate limit; full velocity in one step."""
        cfg  = ActionConfig(smoothing_alpha=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        p = proc.process(np.array([1.0, 0.0]))
        assert p.v_linear == pytest.approx(cfg.lin_vel_max, rel=1e-5)

    def test_lin_acc_limits_first_step(self):
        """lin_acc_max=1 m/s², dt=0.01 → max Δv = 0.01 m/s from zero."""
        cfg  = ActionConfig(smoothing_alpha=1.0, lin_vel_max=2.0,
                            lin_acc_max=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        p = proc.process(np.array([1.0, 0.0]))
        assert p.v_linear == pytest.approx(0.01, abs=1e-6)

    def test_ang_acc_limits_first_step(self):
        """ang_acc_max=2 rad/s², dt=0.01 → max Δω = 0.02 rad/s from zero."""
        cfg  = ActionConfig(smoothing_alpha=1.0, ang_vel_max=3.0,
                            ang_acc_max=2.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        p = proc.process(np.array([0.0, 1.0]))
        assert p.v_angular == pytest.approx(0.02, abs=1e-6)

    def test_velocity_builds_over_steps(self):
        """With acc limit, velocity ramps up over multiple steps."""
        cfg  = ActionConfig(smoothing_alpha=1.0, lin_vel_max=2.0,
                            lin_acc_max=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        velocities = [proc.process(np.array([1.0, 0.0])).v_linear
                      for _ in range(10)]
        # Each step adds 0.01 m/s
        assert velocities[-1] == pytest.approx(0.10, abs=1e-6)
        # Monotonically increasing
        assert all(velocities[i] <= velocities[i + 1]
                   for i in range(len(velocities) - 1))

    def test_deceleration_limited(self):
        """Sudden stop from full speed should be rate-limited."""
        cfg  = ActionConfig(smoothing_alpha=1.0, lin_vel_max=2.0,
                            lin_acc_max=5.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        # Ramp up to max in many steps
        for _ in range(50):
            proc.process(np.array([1.0, 0.0]))
        # Sudden stop
        p = proc.process(np.array([0.0, 0.0]))
        # Should not drop to zero immediately
        assert abs(p.v_linear) >= cfg.lin_acc_max * 0.01 * 0.5

    def test_reset_restarts_ramp(self):
        cfg  = ActionConfig(smoothing_alpha=1.0, lin_acc_max=1.0)
        proc = ActionProcessor(cfg=cfg, dt_env=0.01)
        proc.reset()
        for _ in range(10):
            proc.process(np.array([1.0, 0.0]))
        proc.reset()
        p = proc.process(np.array([1.0, 0.0]))
        assert p.v_linear == pytest.approx(0.01, abs=1e-6)


# ── action_space ──────────────────────────────────────────────────────────────

class TestActionSpace:
    def test_shape(self):
        proc = ActionProcessor()
        sp   = proc.action_space()
        assert sp.shape == (2,)

    def test_bounds(self):
        proc = ActionProcessor()
        sp   = proc.action_space()
        assert np.all(sp.low  == -1.0)
        assert np.all(sp.high ==  1.0)

    def test_make_action_space_helper(self):
        sp = make_action_space()
        assert sp.shape == (2,)

    def test_action_sample_within_bounds(self):
        proc = ActionProcessor()
        sp   = proc.action_space()
        for _ in range(20):
            a = sp.sample()
            assert sp.contains(a)


# ── differential_drive kinematics ─────────────────────────────────────────────

class TestDifferentialDrive:
    def test_forward_equal_wheels(self):
        cl, cr = differential_drive(v_lin=1.0, v_ang=0.0,
                                    radius=0.1, wheelbase=0.4)
        assert cl == pytest.approx(10.0, rel=1e-5)
        assert cr == pytest.approx(10.0, rel=1e-5)

    def test_spin_in_place(self):
        """v_ang>0 (left turn): left wheel slower / backward, right faster."""
        cl, cr = differential_drive(v_lin=0.0, v_ang=1.0,
                                    radius=0.1, wheelbase=0.4)
        assert cl < 0
        assert cr > 0
        assert cr == pytest.approx(-cl, rel=1e-5)

    def test_reverse(self):
        cl, cr = differential_drive(v_lin=-1.0, v_ang=0.0,
                                    radius=0.1, wheelbase=0.4)
        assert cl < 0
        assert cr < 0
        assert cl == pytest.approx(cr, rel=1e-5)

    def test_formula_correctness(self):
        r, wb = 0.08, 0.30
        vl, va = 1.2, 0.5
        cl_exp = (vl - (wb / 2) * va) / r
        cr_exp = (vl + (wb / 2) * va) / r
        cl, cr = differential_drive(vl, va, r, wb)
        assert cl == pytest.approx(cl_exp, rel=1e-6)
        assert cr == pytest.approx(cr_exp, rel=1e-6)


# ── inverse_differential_drive ────────────────────────────────────────────────

class TestInverseDifferentialDrive:
    def test_forward_recovery(self):
        r, wb = 0.08, 0.30
        cl, cr = differential_drive(1.0, 0.0, r, wb)
        v_lin, v_ang = inverse_differential_drive(cl, cr, r, wb)
        assert v_lin == pytest.approx(1.0, rel=1e-5)
        assert v_ang == pytest.approx(0.0, abs=1e-9)

    def test_spin_recovery(self):
        r, wb = 0.08, 0.30
        cl, cr = differential_drive(0.0, 1.0, r, wb)
        v_lin, v_ang = inverse_differential_drive(cl, cr, r, wb)
        assert v_lin == pytest.approx(0.0, abs=1e-9)
        assert v_ang == pytest.approx(1.0, rel=1e-5)

    def test_roundtrip(self):
        r, wb = 0.08, 0.30
        for vl, va in [(0.5, 0.3), (-1.0, 0.8), (0.0, -2.0), (1.5, 0.0)]:
            cl, cr       = differential_drive(vl, va, r, wb)
            vl_r, va_r   = inverse_differential_drive(cl, cr, r, wb)
            assert vl_r  == pytest.approx(vl, rel=1e-5)
            assert va_r  == pytest.approx(va, rel=1e-5)

    def test_zero_wheels_zero_velocity(self):
        vl, va = inverse_differential_drive(0.0, 0.0)
        assert vl == pytest.approx(0.0, abs=1e-9)
        assert va == pytest.approx(0.0, abs=1e-9)


# ── PhysicalAction repr ────────────────────────────────────────────────────────

class TestPhysicalActionRepr:
    def test_repr_contains_m_per_s(self):
        proc = ActionProcessor()
        proc.reset()
        p = proc.process(np.array([0.5, 0.2]))
        assert "m/s" in repr(p)

    def test_repr_contains_rad_per_s(self):
        proc = ActionProcessor()
        proc.reset()
        p = proc.process(np.array([0.5, 0.2]))
        assert "rad/s" in repr(p)
