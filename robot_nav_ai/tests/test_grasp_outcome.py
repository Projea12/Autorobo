"""
tests/test_grasp_outcome.py — Unit tests for GraspOutcomeDetector and related types.

No MuJoCo env needed — all tests use synthetic obs vectors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.grasp_outcome import (
    GraspOutcome, GraspOutcomeDetector, GraspResult, OutcomeConfig,
)


# ── helpers ───────────────────────────────────────────────────────────────────

CFG = OutcomeConfig()

def _obs(
    touch: float = 0.0,
    wrist_mag: float = 0.0,
    obj_z: float = 0.02,
) -> np.ndarray:
    """Build a minimal 45-dim obs vector with specific sensor values."""
    obs = np.zeros(45, dtype=np.float32)
    ts, te = CFG.touch_slice      # 24:26
    ws, we = CFG.wrist_slice      # 33:36
    ps, pe = CFG.target_slice     # 39:42
    obs[ts]          = touch
    obs[ws]          = wrist_mag  # put in first component; norm = wrist_mag
    obs[ps + 2]      = obj_z
    return obs


def _info(success: bool = False) -> dict:
    return {"success": success}


def _run_episode(detector, steps) -> GraspOutcome:
    """Feed a list of (obs, info) pairs to detector and classify."""
    detector.reset()
    for obs, info in steps:
        detector.update(obs, info)
    return detector.classify()


# ── OutcomeConfig ─────────────────────────────────────────────────────────────

class TestOutcomeConfig:
    def test_defaults(self):
        cfg = OutcomeConfig()
        assert cfg.contact_thresh   == pytest.approx(0.05)
        assert cfg.lift_thresh_z    == pytest.approx(0.045)
        assert cfg.success_height_z == pytest.approx(0.225)
        assert cfg.wrist_force_max  == pytest.approx(0.90)

    def test_frozen(self):
        with pytest.raises(Exception):
            OutcomeConfig().contact_thresh = 0.1

    def test_slice_indices(self):
        cfg = OutcomeConfig()
        assert cfg.touch_slice  == (24, 26)
        assert cfg.wrist_slice  == (33, 36)
        assert cfg.target_slice == (39, 42)


# ── GraspOutcome ──────────────────────────────────────────────────────────────

class TestGraspOutcome:
    def _outcome(self, result: GraspResult) -> GraspOutcome:
        return GraspOutcome(result=result, reason="test", n_steps=10)

    def test_success_property_true(self):
        assert self._outcome(GraspResult.SUCCESS).success is True

    def test_success_property_false(self):
        for r in [GraspResult.MISS, GraspResult.SLIP,
                  GraspResult.COLLISION, GraspResult.DROP]:
            assert self._outcome(r).success is False

    def test_failure_mode_none_on_success(self):
        assert self._outcome(GraspResult.SUCCESS).failure_mode is None

    def test_failure_mode_string_on_failure(self):
        o = self._outcome(GraspResult.MISS)
        assert o.failure_mode == "miss"

    def test_to_dict_keys(self):
        o = self._outcome(GraspResult.SUCCESS)
        d = o.to_dict()
        assert "result" in d
        assert "success" in d
        assert "contact_made" in d
        assert "max_lift_z" in d

    def test_repr_contains_result(self):
        o = self._outcome(GraspResult.SLIP)
        assert "SLIP" in repr(o)


# ── GraspOutcomeDetector — no-step edge case ──────────────────────────────────

class TestDetectorEdgeCases:
    def test_classify_before_any_steps(self):
        d = GraspOutcomeDetector()
        d.reset()
        outcome = d.classify()
        assert outcome.result == GraspResult.UNKNOWN
        assert outcome.n_steps == 0

    def test_reset_clears_state(self):
        d = GraspOutcomeDetector()
        d.update(_obs(touch=0.9, obj_z=0.3), _info(success=True))
        d.reset()
        outcome = d.classify()
        assert outcome.result == GraspResult.UNKNOWN

    def test_n_steps_increments(self):
        d = GraspOutcomeDetector()
        d.reset()
        for _ in range(5):
            d.update(_obs(), _info())
        assert d._step == 5


# ── MISS detection ────────────────────────────────────────────────────────────

class TestMissDetection:
    def test_no_contact_is_miss(self):
        d = GraspOutcomeDetector()
        steps = [(_obs(touch=0.0, obj_z=0.02), _info()) for _ in range(10)]
        outcome = _run_episode(d, steps)
        assert outcome.result == GraspResult.MISS
        assert outcome.contact_made is False

    def test_sub_threshold_touch_is_miss(self):
        d = GraspOutcomeDetector()
        # touch just below threshold (0.04 < 0.05)
        steps = [(_obs(touch=0.04, obj_z=0.02), _info()) for _ in range(5)]
        outcome = _run_episode(d, steps)
        assert outcome.result == GraspResult.MISS

    def test_contact_but_not_lifted_is_miss(self):
        d = GraspOutcomeDetector()
        # contact made, object stays below lift_thresh
        steps = [(_obs(touch=0.9, obj_z=0.02), _info()) for _ in range(10)]
        outcome = _run_episode(d, steps)
        assert outcome.result == GraspResult.MISS
        assert outcome.contact_made is True


# ── SUCCESS detection ─────────────────────────────────────────────────────────

class TestSuccessDetection:
    def test_info_success_flag(self):
        d = GraspOutcomeDetector()
        steps = [
            (_obs(touch=0.9, obj_z=0.05), _info()),
            (_obs(touch=0.9, obj_z=0.05), _info(success=True)),
        ]
        outcome = _run_episode(d, steps)
        assert outcome.result == GraspResult.SUCCESS
        assert outcome.success is True

    def test_height_threshold_success(self):
        d = GraspOutcomeDetector()
        # obj_z >= success_height_z (0.225) + contact
        steps = [(_obs(touch=0.9, obj_z=0.23), _info()) for _ in range(5)]
        outcome = _run_episode(d, steps)
        assert outcome.result == GraspResult.SUCCESS

    def test_max_lift_z_recorded(self):
        d = GraspOutcomeDetector()
        steps = [
            (_obs(touch=0.9, obj_z=0.1), _info()),
            (_obs(touch=0.9, obj_z=0.25), _info()),
            (_obs(touch=0.9, obj_z=0.23), _info()),
        ]
        outcome = _run_episode(d, steps)
        assert outcome.max_lift_z == pytest.approx(0.25)


# ── COLLISION detection ───────────────────────────────────────────────────────

class TestCollisionDetection:
    def test_high_wrist_force_is_collision(self):
        d = GraspOutcomeDetector()
        # wrist_force_max = 0.90 → norm(0.95, 0, 0) = 0.95 > 0.90
        steps = [(_obs(wrist_mag=0.95), _info())]
        outcome = _run_episode(d, steps)
        assert outcome.result == GraspResult.COLLISION

    def test_collision_step_recorded(self):
        d = GraspOutcomeDetector()
        steps = [
            (_obs(wrist_mag=0.0), _info()),
            (_obs(wrist_mag=0.0), _info()),
            (_obs(wrist_mag=0.95), _info()),
        ]
        outcome = _run_episode(d, steps)
        assert outcome.collision_step == 3

    def test_collision_priority_over_success(self):
        d = GraspOutcomeDetector()
        # collision AND success flag in same episode
        steps = [
            (_obs(touch=0.9, obj_z=0.23, wrist_mag=0.95), _info(success=True)),
        ]
        outcome = _run_episode(d, steps)
        assert outcome.result == GraspResult.COLLISION

    def test_safe_wrist_no_collision(self):
        d = GraspOutcomeDetector()
        steps = [(_obs(wrist_mag=0.5), _info()) for _ in range(5)]
        outcome = _run_episode(d, steps)
        assert outcome.result != GraspResult.COLLISION


# ── SLIP detection ────────────────────────────────────────────────────────────

class TestSlipDetection:
    def test_contact_lost_after_lift_is_slip(self):
        d = GraspOutcomeDetector()
        steps = [
            # contact + lift
            (_obs(touch=0.9, obj_z=0.1), _info()),
            # contact lost while still above lift threshold
            (_obs(touch=0.0, obj_z=0.1), _info()),
        ]
        outcome = _run_episode(d, steps)
        assert outcome.result == GraspResult.SLIP
        assert outcome.contact_lost is True

    def test_contact_lost_without_lift_is_not_slip(self):
        d = GraspOutcomeDetector()
        # contact made but never lifted → MISS (contact made but not lifted)
        steps = [
            (_obs(touch=0.9, obj_z=0.02), _info()),
            (_obs(touch=0.0, obj_z=0.02), _info()),
        ]
        outcome = _run_episode(d, steps)
        assert outcome.result != GraspResult.SLIP


# ── DROP detection ────────────────────────────────────────────────────────────

class TestDropDetection:
    def test_drop_after_lift(self):
        d = GraspOutcomeDetector()
        cfg = OutcomeConfig(lift_thresh_z=0.05)
        d = GraspOutcomeDetector(cfg=cfg)
        steps = [
            # contact + lifted above lift_thresh
            (_obs(touch=0.9, obj_z=0.10), _info()),
            # still in contact, but object descended below lift_thresh
            (_obs(touch=0.9, obj_z=0.02), _info()),
        ]
        outcome = _run_episode(d, steps)
        assert outcome.result == GraspResult.DROP

    def test_drop_without_lift_is_not_drop(self):
        d = GraspOutcomeDetector()
        # object never lifted → MISS
        steps = [
            (_obs(touch=0.9, obj_z=0.02), _info()),
        ]
        outcome = _run_episode(d, steps)
        assert outcome.result != GraspResult.DROP


# ── Priority ordering ─────────────────────────────────────────────────────────

class TestPriority:
    def test_collision_beats_slip(self):
        d = GraspOutcomeDetector()
        steps = [
            (_obs(touch=0.9, obj_z=0.1), _info()),
            (_obs(touch=0.0, obj_z=0.1, wrist_mag=0.95), _info()),
        ]
        outcome = _run_episode(d, steps)
        assert outcome.result == GraspResult.COLLISION

    def test_slip_beats_drop(self):
        d = GraspOutcomeDetector()
        cfg = OutcomeConfig(lift_thresh_z=0.05)
        d = GraspOutcomeDetector(cfg=cfg)
        steps = [
            (_obs(touch=0.9, obj_z=0.15), _info()),
            # contact lost + obj dropped below lift_thresh → both slip & drop
            (_obs(touch=0.0, obj_z=0.02), _info()),
        ]
        outcome = _run_episode(d, steps)
        # slip wins because contact was lost after lift
        assert outcome.result == GraspResult.SLIP


# ── from_episode class method ─────────────────────────────────────────────────

class TestFromEpisode:
    def test_from_episode_success(self):
        obs_list  = [_obs(touch=0.9, obj_z=0.23)] * 5
        info_list = [_info()] * 4 + [_info(success=True)]
        outcome   = GraspOutcomeDetector.from_episode(obs_list, info_list)
        assert outcome.result == GraspResult.SUCCESS

    def test_from_episode_miss(self):
        obs_list  = [_obs(touch=0.0, obj_z=0.02)] * 5
        info_list = [_info()] * 5
        outcome   = GraspOutcomeDetector.from_episode(obs_list, info_list)
        assert outcome.result == GraspResult.MISS

    def test_from_episode_empty(self):
        outcome = GraspOutcomeDetector.from_episode([], [])
        assert outcome.result == GraspResult.UNKNOWN

    def test_from_episode_n_steps(self):
        obs_list  = [_obs()] * 7
        info_list = [_info()] * 7
        outcome   = GraspOutcomeDetector.from_episode(obs_list, info_list)
        assert outcome.n_steps == 7

    def test_from_episode_custom_cfg(self):
        cfg = OutcomeConfig(wrist_force_max=0.5)
        obs_list  = [_obs(wrist_mag=0.6)]
        info_list = [_info()]
        outcome   = GraspOutcomeDetector.from_episode(obs_list, info_list, cfg=cfg)
        assert outcome.result == GraspResult.COLLISION


# ── repr ──────────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_contains_step(self):
        d = GraspOutcomeDetector()
        d.reset()
        d.update(_obs(), _info())
        assert "step=1" in repr(d)

    def test_repr_contains_contact(self):
        d = GraspOutcomeDetector()
        d.reset()
        assert "contact=" in repr(d)
