"""
tests/test_grasp_action.py — Unit tests for GraspActionProcessor, GraspActionConfig,
GraspPhysicalAction, GripperCmd, and make_grasp_action_space.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.grasp_action import (
    ACTION_DIM,
    ACTION_SPACE_SPEC,
    GRIPPER_HYSTERESIS,
    GRIPPER_THRESHOLD,
    MAX_POS_DELTA,
    MAX_ROT_DELTA,
    GraspActionConfig,
    GraspActionProcessor,
    GraspPhysicalAction,
    GripperCmd,
    make_grasp_action_space,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _proc(cfg: GraspActionConfig = GraspActionConfig()) -> GraspActionProcessor:
    p = GraspActionProcessor(cfg=cfg)
    p.reset()
    return p


def _zeros() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float32)


def _action(**kw) -> np.ndarray:
    a = _zeros()
    for k, v in kw.items():
        idx = {"dx": 0, "dy": 1, "dz": 2, "droll": 3, "dpitch": 4, "dyaw": 5, "grip": 6}[k]
        a[idx] = v
    return a


# ── GraspActionConfig ─────────────────────────────────────────────────────────

class TestGraspActionConfig:
    def test_defaults(self):
        cfg = GraspActionConfig()
        assert cfg.max_pos_delta  == pytest.approx(MAX_POS_DELTA)
        assert cfg.max_rot_delta  == pytest.approx(MAX_ROT_DELTA)
        assert cfg.smooth_alpha   == pytest.approx(0.6)
        assert cfg.gripper_thresh == pytest.approx(GRIPPER_THRESHOLD)
        assert cfg.gripper_hyst   == pytest.approx(GRIPPER_HYSTERESIS)

    def test_frozen(self):
        with pytest.raises(Exception):
            GraspActionConfig().max_pos_delta = 0.1

    def test_custom(self):
        cfg = GraspActionConfig(max_pos_delta=0.02, smooth_alpha=1.0)
        assert cfg.max_pos_delta == pytest.approx(0.02)
        assert cfg.smooth_alpha  == pytest.approx(1.0)


# ── GripperCmd ────────────────────────────────────────────────────────────────

class TestGripperCmd:
    def test_open_value(self):
        assert GripperCmd.OPEN.value == "open"

    def test_close_value(self):
        assert GripperCmd.CLOSE.value == "close"

    def test_distinct(self):
        assert GripperCmd.OPEN != GripperCmd.CLOSE


# ── GraspPhysicalAction ───────────────────────────────────────────────────────

class TestGraspPhysicalAction:
    def _make(self) -> GraspPhysicalAction:
        return GraspPhysicalAction(
            delta_pos       = np.array([0.01, 0.02, 0.03], dtype=np.float32),
            delta_euler     = np.array([0.1, 0.2, 0.3], dtype=np.float32),
            gripper_cmd     = GripperCmd.OPEN,
            gripper_changed = False,
            raw             = np.zeros(ACTION_DIM, dtype=np.float32),
        )

    def test_fields_stored(self):
        a = self._make()
        assert a.gripper_cmd == GripperCmd.OPEN
        assert not a.gripper_changed

    def test_delta_pos_shape(self):
        assert self._make().delta_pos.shape == (3,)

    def test_delta_euler_shape(self):
        assert self._make().delta_euler.shape == (3,)

    def test_repr_contains_gripper(self):
        assert "open" in repr(self._make())

    def test_repr_contains_cm(self):
        assert "cm" in repr(self._make())


# ── GraspActionProcessor — construction & reset ───────────────────────────────

class TestProcessorInit:
    def test_default_gripper_open(self):
        p = _proc()
        assert p.gripper_state == GripperCmd.OPEN

    def test_reset_clears_smooth(self):
        p = _proc(GraspActionConfig(smooth_alpha=1.0))
        p.process(_action(dx=1.0))
        p.reset()
        r = p.process(_zeros())
        assert np.allclose(r.delta_pos, 0.0, atol=1e-6)

    def test_reset_clears_gripper(self):
        p = _proc()
        p.process(_action(grip=1.0))
        p.reset()
        assert p.gripper_state == GripperCmd.OPEN

    def test_repr_contains_cm(self):
        assert "cm" in repr(_proc())


# ── GraspActionProcessor — position scaling ───────────────────────────────────

class TestPositionScaling:
    def test_full_positive_x(self):
        cfg = GraspActionConfig(smooth_alpha=1.0)
        r = _proc(cfg).process(_action(dx=1.0))
        assert r.delta_pos[0] == pytest.approx(MAX_POS_DELTA, rel=1e-5)

    def test_full_negative_x(self):
        cfg = GraspActionConfig(smooth_alpha=1.0)
        r = _proc(cfg).process(_action(dx=-1.0))
        assert r.delta_pos[0] == pytest.approx(-MAX_POS_DELTA, rel=1e-5)

    def test_zero_action_zero_delta(self):
        cfg = GraspActionConfig(smooth_alpha=1.0)
        r = _proc(cfg).process(_zeros())
        assert np.allclose(r.delta_pos, 0.0, atol=1e-6)

    def test_half_action_half_delta(self):
        cfg = GraspActionConfig(smooth_alpha=1.0)
        r = _proc(cfg).process(_action(dy=0.5))
        assert r.delta_pos[1] == pytest.approx(0.5 * MAX_POS_DELTA, rel=1e-5)

    def test_clipping_above_one(self):
        cfg = GraspActionConfig(smooth_alpha=1.0)
        r = _proc(cfg).process(np.full(ACTION_DIM, 2.0, dtype=np.float32))
        assert r.delta_pos[0] <= MAX_POS_DELTA + 1e-6

    def test_clipping_below_neg_one(self):
        cfg = GraspActionConfig(smooth_alpha=1.0)
        r = _proc(cfg).process(np.full(ACTION_DIM, -2.0, dtype=np.float32))
        assert r.delta_pos[0] >= -MAX_POS_DELTA - 1e-6

    def test_delta_pos_dtype(self):
        r = _proc().process(_zeros())
        assert r.delta_pos.dtype == np.float32

    def test_all_three_axes_scaled(self):
        cfg = GraspActionConfig(smooth_alpha=1.0)
        r = _proc(cfg).process(_action(dx=1.0, dy=-1.0, dz=0.5))
        assert r.delta_pos[0] == pytest.approx( MAX_POS_DELTA,       rel=1e-5)
        assert r.delta_pos[1] == pytest.approx(-MAX_POS_DELTA,       rel=1e-5)
        assert r.delta_pos[2] == pytest.approx( 0.5 * MAX_POS_DELTA, rel=1e-5)


# ── GraspActionProcessor — rotation scaling ───────────────────────────────────

class TestRotationScaling:
    def test_full_roll(self):
        r = _proc().process(_action(droll=1.0))
        assert r.delta_euler[0] == pytest.approx(MAX_ROT_DELTA, rel=1e-5)

    def test_full_negative_pitch(self):
        r = _proc().process(_action(dpitch=-1.0))
        assert r.delta_euler[1] == pytest.approx(-MAX_ROT_DELTA, rel=1e-5)

    def test_half_yaw(self):
        r = _proc().process(_action(dyaw=0.5))
        assert r.delta_euler[2] == pytest.approx(0.5 * MAX_ROT_DELTA, rel=1e-5)

    def test_zero_rotation(self):
        r = _proc().process(_zeros())
        assert np.allclose(r.delta_euler, 0.0, atol=1e-6)

    def test_delta_euler_dtype(self):
        r = _proc().process(_zeros())
        assert r.delta_euler.dtype == np.float32

    def test_rotation_not_smoothed(self):
        # No EMA on rotation — two steps, second should equal raw scaled value
        cfg = GraspActionConfig(smooth_alpha=0.3)
        p = _proc(cfg)
        p.process(_action(droll=1.0))          # step 1
        r = p.process(_action(droll=-1.0))     # step 2
        assert r.delta_euler[0] == pytest.approx(-MAX_ROT_DELTA, rel=1e-5)


# ── GraspActionProcessor — smoothing ─────────────────────────────────────────

class TestSmoothing:
    def test_alpha_one_no_smoothing(self):
        cfg = GraspActionConfig(smooth_alpha=1.0)
        r = _proc(cfg).process(_action(dx=1.0))
        assert r.delta_pos[0] == pytest.approx(MAX_POS_DELTA, rel=1e-4)

    def test_alpha_low_dampens_first_step(self):
        cfg = GraspActionConfig(smooth_alpha=0.1)
        r = _proc(cfg).process(_action(dx=1.0))
        # smoothed = 0.1 * MAX_POS_DELTA
        assert r.delta_pos[0] == pytest.approx(0.1 * MAX_POS_DELTA, rel=1e-4)

    def test_ema_accumulates_over_steps(self):
        cfg = GraspActionConfig(smooth_alpha=0.5)
        p = _proc(cfg)
        r1 = p.process(_action(dx=1.0))
        r2 = p.process(_action(dx=1.0))
        # second step should be larger than first (EMA building up)
        assert r2.delta_pos[0] > r1.delta_pos[0]

    def test_reset_clears_ema(self):
        cfg = GraspActionConfig(smooth_alpha=0.5)
        p = _proc(cfg)
        for _ in range(10):
            p.process(_action(dx=1.0))
        p.reset()
        r = p.process(_action(dx=1.0))
        expected = 0.5 * MAX_POS_DELTA
        assert r.delta_pos[0] == pytest.approx(expected, rel=1e-4)


# ── GraspActionProcessor — gripper ───────────────────────────────────────────

class TestGripper:
    def test_positive_signal_closes(self):
        p = _proc()
        r = p.process(_action(grip=1.0))
        assert r.gripper_cmd == GripperCmd.CLOSE

    def test_negative_signal_stays_open(self):
        p = _proc()
        r = p.process(_action(grip=-1.0))
        assert r.gripper_cmd == GripperCmd.OPEN

    def test_zero_signal_stays_open(self):
        p = _proc()
        r = p.process(_action(grip=0.0))
        assert r.gripper_cmd == GripperCmd.OPEN

    def test_gripper_changed_on_close(self):
        p = _proc()
        r = p.process(_action(grip=1.0))
        assert r.gripper_changed is True

    def test_gripper_not_changed_on_second_close(self):
        p = _proc()
        p.process(_action(grip=1.0))
        r = p.process(_action(grip=1.0))
        assert r.gripper_changed is False

    def test_gripper_reopens(self):
        p = _proc()
        p.process(_action(grip=1.0))   # close
        r = p.process(_action(grip=-1.0))  # open
        assert r.gripper_cmd == GripperCmd.OPEN
        assert r.gripper_changed is True

    def test_hysteresis_prevents_chatter(self):
        # Signal just above threshold — should close
        # Then signal drops to just above (thresh - hyst) — should stay closed
        p = _proc()
        p.process(_action(grip=GRIPPER_THRESHOLD + GRIPPER_HYSTERESIS + 0.01))
        r = p.process(_action(grip=GRIPPER_THRESHOLD - GRIPPER_HYSTERESIS + 0.01))
        assert r.gripper_cmd == GripperCmd.CLOSE  # hysteresis holds it closed

    def test_hysteresis_opens_below_lower_band(self):
        p = _proc()
        p.process(_action(grip=1.0))   # close
        r = p.process(_action(grip=GRIPPER_THRESHOLD - GRIPPER_HYSTERESIS - 0.01))
        assert r.gripper_cmd == GripperCmd.OPEN

    def test_gripper_state_property(self):
        p = _proc()
        p.process(_action(grip=1.0))
        assert p.gripper_state == GripperCmd.CLOSE


# ── GraspActionProcessor — input validation ───────────────────────────────────

class TestInputValidation:
    def test_wrong_shape_raises(self):
        p = _proc()
        with pytest.raises(ValueError):
            p.process(np.zeros(5, dtype=np.float32))

    def test_raw_stored_in_result(self):
        a = _action(dx=0.5, grip=1.0)
        r = _proc().process(a)
        assert r.raw.shape == (ACTION_DIM,)

    def test_accepts_float64_input(self):
        p = _proc()
        r = p.process(np.zeros(ACTION_DIM, dtype=np.float64))
        assert r.delta_pos.dtype == np.float32


# ── make_grasp_action_space ───────────────────────────────────────────────────

class TestActionSpace:
    def test_shape(self):
        space = make_grasp_action_space()
        assert space.shape == (ACTION_DIM,)

    def test_low_minus_one(self):
        space = make_grasp_action_space()
        assert np.all(space.low == -1.0)

    def test_high_plus_one(self):
        space = make_grasp_action_space()
        assert np.all(space.high == 1.0)

    def test_dtype_float32(self):
        space = make_grasp_action_space()
        assert space.dtype == np.float32

    def test_sample_in_bounds(self):
        space = make_grasp_action_space()
        for _ in range(20):
            s = space.sample()
            assert np.all(s >= -1.0) and np.all(s <= 1.0)


# ── ACTION_SPACE_SPEC ─────────────────────────────────────────────────────────

class TestActionSpaceSpec:
    def test_dims(self):
        assert ACTION_SPACE_SPEC["dims"] == ACTION_DIM

    def test_seven_components(self):
        assert len(ACTION_SPACE_SPEC["components"]) == ACTION_DIM

    def test_indices_are_zero_to_six(self):
        idxs = [c["index"] for c in ACTION_SPACE_SPEC["components"]]
        assert idxs == list(range(ACTION_DIM))

    def test_gripper_last(self):
        last = ACTION_SPACE_SPEC["components"][-1]
        assert last["name"] == "gripper"
