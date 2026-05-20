"""
tests/test_grasp_reward.py — Unit tests for GraspRewardFunction, GraspRewardConfig,
and GraspRewardInfo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.grasp_reward import GraspRewardConfig, GraspRewardFunction, GraspRewardInfo


# ── helpers ───────────────────────────────────────────────────────────────────

OBJ  = np.array([0.5, 0.0, 0.05], dtype=np.float64)   # object on table
EE_NEAR  = np.array([0.5, 0.0, 0.10], dtype=np.float64)
EE_FAR   = np.array([0.0, 0.0, 0.30], dtype=np.float64)

def _fn(cfg: GraspRewardConfig = GraspRewardConfig()) -> GraspRewardFunction:
    fn = GraspRewardFunction(cfg)
    fn.reset(obj_pos=OBJ, ee_pos=EE_FAR)
    return fn


# ── GraspRewardConfig ─────────────────────────────────────────────────────────

class TestGraspRewardConfig:
    def test_defaults(self):
        cfg = GraspRewardConfig()
        assert cfg.approach     == pytest.approx(3.0)
        assert cfg.contact      == pytest.approx(0.5)
        assert cfg.lift         == pytest.approx(5.0)
        assert cfg.stability    == pytest.approx(1.0)
        assert cfg.symmetry     == pytest.approx(0.2)
        assert cfg.time_step    == pytest.approx(0.01)
        assert cfg.success      == pytest.approx(10.0)
        assert cfg.collision    == pytest.approx(5.0)
        assert cfg.success_height == pytest.approx(0.20)

    def test_frozen(self):
        with pytest.raises(Exception):
            GraspRewardConfig().approach = 1.0

    def test_custom(self):
        cfg = GraspRewardConfig(approach=1.0, success_height=0.15)
        assert cfg.approach == pytest.approx(1.0)
        assert cfg.success_height == pytest.approx(0.15)


# ── GraspRewardInfo ───────────────────────────────────────────────────────────

class TestGraspRewardInfo:
    def test_defaults_zero(self):
        info = GraspRewardInfo()
        assert info.total == pytest.approx(0.0)
        assert not info.success
        assert not info.terminated

    def test_to_dict_keys(self):
        info = GraspRewardInfo()
        d = info.to_dict()
        for key in ("approach", "contact", "lift", "stability",
                    "symmetry", "time", "terminal", "total",
                    "success", "terminated"):
            assert key in d

    def test_repr_contains_total(self):
        info = GraspRewardInfo(total=1.23)
        assert "1.23" in repr(info) or "1.2" in repr(info)


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_contact_made_false_after_reset(self):
        fn = _fn()
        assert not fn.contact_made

    def test_lifted_false_after_reset(self):
        fn = _fn()
        assert not fn.lifted

    def test_reset_without_ee_pos(self):
        fn = GraspRewardFunction()
        fn.reset(obj_pos=OBJ)   # no ee_pos — should not raise
        assert not fn.contact_made


# ── approach reward ───────────────────────────────────────────────────────────

class TestApproachReward:
    def test_approaching_positive(self):
        fn = _fn()
        # EE far from object → step with EE near object
        info = fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)
        assert info.approach > 0.0

    def test_retreating_negative(self):
        fn = GraspRewardFunction()
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        info = fn.step(ee_pos=EE_FAR, obj_pos=OBJ, n_touching=0)
        assert info.approach < 0.0

    def test_no_movement_zero(self):
        fn = GraspRewardFunction()
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        info = fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)
        assert info.approach == pytest.approx(0.0, abs=1e-9)

    def test_approach_scales_with_weight(self):
        cfg1 = GraspRewardConfig(approach=1.0)
        cfg2 = GraspRewardConfig(approach=2.0)
        fn1, fn2 = GraspRewardFunction(cfg1), GraspRewardFunction(cfg2)
        fn1.reset(obj_pos=OBJ, ee_pos=EE_FAR)
        fn2.reset(obj_pos=OBJ, ee_pos=EE_FAR)
        i1 = fn1.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)
        i2 = fn2.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)
        assert i2.approach == pytest.approx(2 * i1.approach, rel=1e-5)


# ── contact reward ────────────────────────────────────────────────────────────

class TestContactReward:
    def test_no_touch_zero(self):
        info = _fn().step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)
        assert info.contact == pytest.approx(0.0)

    def test_one_finger_half(self):
        cfg = GraspRewardConfig(contact=1.0)
        fn = GraspRewardFunction(cfg)
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        info = fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=1)
        assert info.contact == pytest.approx(0.5)

    def test_two_fingers_full(self):
        cfg = GraspRewardConfig(contact=1.0)
        fn = GraspRewardFunction(cfg)
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        info = fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)
        assert info.contact == pytest.approx(1.0)

    def test_contact_sets_flag(self):
        fn = _fn()
        assert not fn.contact_made
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=1)
        assert fn.contact_made

    def test_no_contact_does_not_set_flag(self):
        fn = _fn()
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)
        assert not fn.contact_made


# ── lift reward ───────────────────────────────────────────────────────────────

class TestLiftReward:
    def test_no_contact_no_lift_reward(self):
        fn = _fn()
        lifted_obj = np.array([0.5, 0.0, 0.25])
        info = fn.step(ee_pos=EE_NEAR, obj_pos=lifted_obj, n_touching=0)
        assert info.lift == pytest.approx(0.0)

    def test_contact_then_lift_earns_reward(self):
        fn = _fn()
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)     # make contact
        lifted_obj = np.array([0.5, 0.0, 0.25])
        info = fn.step(ee_pos=EE_NEAR, obj_pos=lifted_obj, n_touching=2)
        assert info.lift > 0.0

    def test_lift_proportional_to_height(self):
        cfg = GraspRewardConfig(lift=1.0, lift_thresh=0.0, table_z=0.0)
        fn = GraspRewardFunction(cfg)
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)   # contact
        obj_h1 = np.array([0.5, 0.0, 0.10])
        obj_h2 = np.array([0.5, 0.0, 0.20])
        i1 = fn.step(ee_pos=EE_NEAR, obj_pos=obj_h1, n_touching=2)
        fn2 = GraspRewardFunction(cfg)
        fn2.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        fn2.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)
        i2 = fn2.step(ee_pos=EE_NEAR, obj_pos=obj_h2, n_touching=2)
        assert i2.lift == pytest.approx(2 * i1.lift, rel=1e-5)

    def test_sets_lifted_flag(self):
        cfg = GraspRewardConfig(lift_thresh=0.0, table_z=0.0)
        fn = GraspRewardFunction(cfg)
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)
        lifted = np.array([0.5, 0.0, 0.10])
        fn.step(ee_pos=EE_NEAR, obj_pos=lifted, n_touching=2)
        assert fn.lifted


# ── stability reward ──────────────────────────────────────────────────────────

class TestStabilityReward:
    def _lifted_fn(self):
        cfg = GraspRewardConfig(lift_thresh=0.0, table_z=0.0)
        fn = GraspRewardFunction(cfg)
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)       # contact
        lifted = np.array([0.5, 0.0, 0.10])
        fn.step(ee_pos=EE_NEAR, obj_pos=lifted, n_touching=2)    # lift
        return fn, cfg

    def test_holding_earns_stability(self):
        fn, cfg = self._lifted_fn()
        lifted = np.array([0.5, 0.0, 0.10])
        info = fn.step(ee_pos=EE_NEAR, obj_pos=lifted, n_touching=2)
        assert info.stability > 0.0

    def test_drop_gives_penalty(self):
        fn, cfg = self._lifted_fn()
        dropped = np.array([0.5, 0.0, 0.0])   # object fell to table
        info = fn.step(ee_pos=EE_NEAR, obj_pos=dropped, n_touching=0)
        assert info.stability < 0.0

    def test_no_lift_no_stability(self):
        fn = _fn()   # not lifted
        info = fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)
        assert info.stability == pytest.approx(0.0)


# ── symmetry reward ───────────────────────────────────────────────────────────

class TestSymmetryReward:
    def test_aligned_approach_earns_bonus(self):
        fn = _fn()
        # EE directly below object → approach vec points up → align with -Z grasp axis
        ee_below = np.array([0.5, 0.0, 0.0])
        obj_above = np.array([0.5, 0.0, 0.1])
        grasp_axis = np.array([0.0, 0.0, 1.0])   # approach from below
        info = fn.step(ee_pos=ee_below, obj_pos=obj_above,
                       n_touching=0, grasp_axis=grasp_axis)
        assert info.symmetry > 0.0

    def test_perpendicular_zero_bonus(self):
        fn = _fn()
        ee_side = np.array([0.4, 0.0, 0.05])
        grasp_axis = np.array([0.0, 0.0, 1.0])   # up
        info = fn.step(ee_pos=ee_side, obj_pos=OBJ,
                       n_touching=0, grasp_axis=grasp_axis)
        assert info.symmetry == pytest.approx(0.0, abs=0.05)

    def test_no_grasp_axis_zero(self):
        info = _fn().step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)
        assert info.symmetry == pytest.approx(0.0)

    def test_symmetry_disabled_after_contact(self):
        fn = _fn()
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)  # make contact
        grasp_axis = np.array([0.0, 0.0, -1.0])
        info = fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2,
                       grasp_axis=grasp_axis)
        assert info.symmetry == pytest.approx(0.0)


# ── time penalty ──────────────────────────────────────────────────────────────

class TestTimePenalty:
    def test_time_negative_every_step(self):
        info = _fn().step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)
        assert info.time < 0.0

    def test_time_equals_neg_weight(self):
        cfg = GraspRewardConfig(time_step=0.05)
        fn = GraspRewardFunction(cfg)
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        info = fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)
        assert info.time == pytest.approx(-0.05)


# ── terminal conditions ───────────────────────────────────────────────────────

class TestTerminal:
    def test_success_on_lift_height(self):
        cfg = GraspRewardConfig(success_height=0.20, table_z=0.0)
        fn = GraspRewardFunction(cfg)
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)    # contact
        high = np.array([0.5, 0.0, 0.21])
        info = fn.step(ee_pos=EE_NEAR, obj_pos=high, n_touching=2)
        assert info.success
        assert info.terminated

    def test_no_success_without_contact(self):
        cfg = GraspRewardConfig(success_height=0.10, table_z=0.0)
        fn = GraspRewardFunction(cfg)
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        high = np.array([0.5, 0.0, 0.15])
        info = fn.step(ee_pos=EE_NEAR, obj_pos=high, n_touching=0)
        assert not info.success

    def test_wrist_unsafe_terminates(self):
        fn = _fn()
        info = fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0,
                       wrist_safe=False)
        assert info.terminated
        assert not info.success
        assert info.terminal == pytest.approx(-fn.cfg.collision)

    def test_success_terminal_equals_weight(self):
        cfg = GraspRewardConfig(success=7.0, success_height=0.10, table_z=0.0)
        fn = GraspRewardFunction(cfg)
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)
        high = np.array([0.5, 0.0, 0.11])
        info = fn.step(ee_pos=EE_NEAR, obj_pos=high, n_touching=2)
        assert info.terminal == pytest.approx(7.0)

    def test_wrist_collision_terminal_equals_neg_weight(self):
        cfg = GraspRewardConfig(collision=3.0)
        fn = GraspRewardFunction(cfg)
        fn.reset(obj_pos=OBJ, ee_pos=EE_NEAR)
        info = fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0,
                       wrist_safe=False)
        assert info.terminal == pytest.approx(-3.0)


# ── total reward consistency ──────────────────────────────────────────────────

class TestTotalReward:
    def test_total_is_sum_of_components(self):
        info = _fn().step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)
        expected = (info.approach + info.contact + info.lift
                    + info.stability + info.symmetry + info.time + info.terminal)
        assert info.total == pytest.approx(expected, rel=1e-6)

    def test_repr_contains_total(self):
        fn = _fn()
        assert "total" in repr(fn).lower() or "Grasp" in repr(fn)


# ── properties ────────────────────────────────────────────────────────────────

class TestProperties:
    def test_contact_made_persists(self):
        fn = _fn()
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=0)   # no contact
        assert fn.contact_made   # flag persists

    def test_reset_clears_contact_made(self):
        fn = _fn()
        fn.step(ee_pos=EE_NEAR, obj_pos=OBJ, n_touching=2)
        fn.reset(obj_pos=OBJ, ee_pos=EE_FAR)
        assert not fn.contact_made
