"""
Tests for robot/workspace.py — workspace limits and utilities.

Covers:
  1. MJCF forcerange / ctrlrange values match DEFAULT_LIMITS
  2. WorkspaceLimits internal consistency
  3. check_joint_positions — within, at boundary, out of range
  4. check_joint_velocities
  5. soft_joint_warnings
  6. cmd_vel_to_wheels / wheels_to_cmd_vel (round-trip)
  7. clamp_wheel_commands — proportional scaling
  8. clamp_cmd_vel — preserves turning radius under clamping
  9. clamp_gripper_pos / clamp_gripper_force
 10. check_wrist_safety
 11. is_ee_reachable — inside sphere, outside, in singularity zone
 12. MJCF model respects forcerange after adding limits
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from robot.constants import ROBOT_XML_PATH
from robot.workspace import (
    DEFAULT_LIMITS, WorkspaceLimits,
    LimitCheck,
    check_joint_positions, check_joint_velocities, soft_joint_warnings,
    cmd_vel_to_wheels, wheels_to_cmd_vel,
    clamp_wheel_commands, clamp_cmd_vel,
    clamp_gripper_pos, clamp_gripper_force,
    check_wrist_safety, is_ee_reachable,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(ROBOT_XML_PATH)


# ── 1. MJCF ↔ DEFAULT_LIMITS consistency ─────────────────────────────────────

def test_wheel_ctrlrange_matches_limits(model):
    lim = DEFAULT_LIMITS
    for name in ("drive_left", "drive_right"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        lo, hi = model.actuator_ctrlrange[aid]
        assert lo == pytest.approx(-lim.wheel_vel_max, abs=0.01)
        assert hi == pytest.approx( lim.wheel_vel_max, abs=0.01)


def test_wheel_forcerange_matches_limits(model):
    lim = DEFAULT_LIMITS
    for name in ("drive_left", "drive_right"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        lo, hi = model.actuator_forcerange[aid]
        assert lo == pytest.approx(-lim.wheel_force_max, abs=0.1)
        assert hi == pytest.approx( lim.wheel_force_max, abs=0.1)


def test_arm_forcerange_matches_limits(model):
    arm_names     = ("arm_j1","arm_j2","arm_j3","arm_j4","arm_j5","arm_j6")
    expected_max  = DEFAULT_LIMITS.joint_force_max
    for name, expected in zip(arm_names, expected_max):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        lo, hi = model.actuator_forcerange[aid]
        assert lo == pytest.approx(-expected, abs=0.1), name
        assert hi == pytest.approx( expected, abs=0.1), name


def test_gripper_forcerange_matches_limits(model):
    lim = DEFAULT_LIMITS
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    lo, hi = model.actuator_forcerange[aid]
    assert lo == pytest.approx(-lim.finger_force_max, abs=0.1)
    assert hi == pytest.approx( lim.finger_force_max, abs=0.1)


def test_gripper_ctrlrange_matches_limits(model):
    lim = DEFAULT_LIMITS
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    lo, hi = model.actuator_ctrlrange[aid]
    assert lo == pytest.approx(0.0,                 abs=1e-4)
    assert hi == pytest.approx(lim.finger_pos_max,  abs=1e-4)


# ── 2. WorkspaceLimits internal consistency ───────────────────────────────────

def test_joint_limits_shape():
    lim = DEFAULT_LIMITS
    assert lim.joint_pos_lo.shape  == (6,)
    assert lim.joint_pos_hi.shape  == (6,)
    assert lim.joint_vel_max.shape == (6,)
    assert lim.joint_force_max.shape == (6,)


def test_joint_limits_lo_lt_hi():
    lim = DEFAULT_LIMITS
    assert np.all(lim.joint_pos_lo < lim.joint_pos_hi)


def test_derived_base_limits():
    lim = DEFAULT_LIMITS
    expected_lin = lim.wheel_vel_max * lim.wheel_radius
    expected_ang = 2.0 * expected_lin / lim.wheelbase
    assert lim.base_lin_vel_max == pytest.approx(expected_lin, rel=1e-6)
    assert lim.base_ang_vel_max == pytest.approx(expected_ang, rel=1e-6)


def test_reach_min_lt_reach_max():
    lim = DEFAULT_LIMITS
    assert lim.reach_min < lim.reach_max


def test_soft_limit_frac_valid():
    lim = DEFAULT_LIMITS
    assert 0.0 < lim.soft_limit_frac < 1.0


# ── 3. check_joint_positions ──────────────────────────────────────────────────

def test_all_zeros_is_valid():
    result = check_joint_positions(np.zeros(6))
    assert result.ok
    assert result.reason == ""


def test_home_pose_is_valid():
    # Home: j2 = π/2, others 0 — well within limits
    q = np.array([0.0, math.pi / 2, 0.0, 0.0, 0.0, 0.0])
    assert check_joint_positions(q).ok


def test_joint1_at_boundary_valid():
    q = np.zeros(6)
    q[0] = math.pi        # exactly at hi limit
    assert check_joint_positions(q).ok
    q[0] = -math.pi       # exactly at lo limit
    assert check_joint_positions(q).ok


def test_joint1_exceeded():
    q = np.zeros(6)
    q[0] = math.pi + 0.01
    result = check_joint_positions(q)
    assert not result.ok
    assert "joint1" in result.reason


def test_joint2_below_min():
    q = np.zeros(6)
    q[1] = -math.pi / 2 - 0.01   # below −90°
    result = check_joint_positions(q)
    assert not result.ok
    assert "joint2" in result.reason


def test_wrong_shape_returns_error():
    result = check_joint_positions(np.zeros(5))
    assert not result.ok


# ── 4. check_joint_velocities ─────────────────────────────────────────────────

def test_zero_velocity_is_valid():
    assert check_joint_velocities(np.zeros(6)).ok


def test_proximal_at_limit():
    dq = np.zeros(6)
    dq[0] = math.pi       # exactly at π rad/s
    assert check_joint_velocities(dq).ok


def test_proximal_exceeds_limit():
    dq = np.zeros(6)
    dq[2] = math.pi + 0.1   # joint3 just over π
    result = check_joint_velocities(dq)
    assert not result.ok
    assert "joint3" in result.reason


def test_distal_at_limit():
    dq = np.zeros(6)
    dq[5] = 2 * math.pi    # exactly at 2π rad/s for joint6
    assert check_joint_velocities(dq).ok


def test_distal_exceeds_limit():
    dq = np.zeros(6)
    dq[3] = 2 * math.pi + 0.1
    result = check_joint_velocities(dq)
    assert not result.ok


# ── 5. soft_joint_warnings ────────────────────────────────────────────────────

def test_no_warnings_at_zero():
    assert soft_joint_warnings(np.zeros(6)) == []


def test_warning_near_joint1_max():
    q = np.zeros(6)
    q[0] = math.pi * 0.96   # 96% of limit → within 10% of edge → warn
    warnings = soft_joint_warnings(q)
    assert any("joint1" in w for w in warnings)


def test_no_warning_at_mid_range():
    q = np.zeros(6)
    q[0] = math.pi * 0.5    # 50% of range — safe
    assert soft_joint_warnings(q) == []


# ── 6. cmd_vel ↔ wheel round-trip ────────────────────────────────────────────

def test_cmd_vel_to_wheels_pure_forward():
    lim = DEFAULT_LIMITS
    v_l, v_r = cmd_vel_to_wheels(0.5, 0.0, lim)
    # Both wheels should spin at the same speed
    assert v_l == pytest.approx(v_r, rel=1e-9)
    # v = ω × r → ω = v / r
    assert v_l == pytest.approx(0.5 / lim.wheel_radius, rel=1e-9)


def test_cmd_vel_to_wheels_pure_rotation():
    lim = DEFAULT_LIMITS
    v_l, v_r = cmd_vel_to_wheels(0.0, 1.0, lim)
    # Left wheel goes backward, right wheel goes forward (turn left)
    assert v_l < 0.0
    assert v_r > 0.0
    assert v_l == pytest.approx(-v_r, rel=1e-9)


def test_wheels_to_cmd_vel_round_trip():
    lim = DEFAULT_LIMITS
    for v_lin, v_ang in [(0.3, 0.0), (0.0, 0.5), (0.4, -0.3)]:
        v_l, v_r   = cmd_vel_to_wheels(v_lin, v_ang, lim)
        v_lin2, v_ang2 = wheels_to_cmd_vel(v_l, v_r, lim)
        assert v_lin2 == pytest.approx(v_lin, abs=1e-9)
        assert v_ang2 == pytest.approx(v_ang, abs=1e-9)


# ── 7. clamp_wheel_commands ───────────────────────────────────────────────────

def test_clamp_within_limits_unchanged():
    v_l, v_r = clamp_wheel_commands(2.0, 3.0)
    assert v_l == pytest.approx(2.0)
    assert v_r == pytest.approx(3.0)


def test_clamp_scales_proportionally():
    lim = DEFAULT_LIMITS
    # Right wheel at 2× limit
    v_l, v_r = clamp_wheel_commands(3.0, lim.wheel_vel_max * 2, lim)
    assert v_r == pytest.approx(lim.wheel_vel_max, rel=1e-9)
    assert v_l == pytest.approx(3.0 / 2.0, rel=1e-9)   # scaled by same factor


def test_clamp_preserves_sign():
    lim = DEFAULT_LIMITS
    v_l, v_r = clamp_wheel_commands(-20.0, -10.0, lim)
    assert v_l < 0.0
    assert v_r < 0.0


# ── 8. clamp_cmd_vel ──────────────────────────────────────────────────────────

def test_clamp_cmd_vel_within_limits():
    v_lin, v_ang = clamp_cmd_vel(0.3, 0.5)
    assert abs(v_lin) <= DEFAULT_LIMITS.base_lin_vel_max + 1e-9
    assert abs(v_ang) <= DEFAULT_LIMITS.base_ang_vel_max + 1e-9


def test_clamp_cmd_vel_preserves_turning_ratio():
    """After clamping, the ratio v_ang / v_lin should be preserved."""
    v_lin_in, v_ang_in = 5.0, 2.0   # way over limits
    v_lin_out, v_ang_out = clamp_cmd_vel(v_lin_in, v_ang_in)
    ratio_in  = v_ang_in  / v_lin_in
    ratio_out = v_ang_out / v_lin_out
    assert ratio_out == pytest.approx(ratio_in, rel=1e-6)


# ── 9. gripper clamping ───────────────────────────────────────────────────────

def test_clamp_gripper_pos_valid():
    assert clamp_gripper_pos(0.02) == pytest.approx(0.02)
    assert clamp_gripper_pos(-0.1) == pytest.approx(0.0)
    assert clamp_gripper_pos(0.10) == pytest.approx(DEFAULT_LIMITS.finger_pos_max)


def test_clamp_gripper_force_valid():
    assert clamp_gripper_force(20.0) == pytest.approx(20.0)
    assert clamp_gripper_force(-5.0) == pytest.approx(0.0)
    assert clamp_gripper_force(500.0) == pytest.approx(DEFAULT_LIMITS.finger_force_max)


# ── 10. check_wrist_safety ────────────────────────────────────────────────────

def test_wrist_safe_at_zero():
    result = check_wrist_safety(np.zeros(3), np.zeros(3))
    assert result.ok


def test_wrist_safe_within_limits():
    result = check_wrist_safety(
        np.array([10.0, 5.0, 3.0]),
        np.array([1.0, 0.5, 0.3]),
    )
    assert result.ok


def test_wrist_unsafe_force():
    lim = DEFAULT_LIMITS
    result = check_wrist_safety(
        np.array([lim.wrist_force_max + 1, 0, 0]),
        np.zeros(3),
    )
    assert not result.ok
    assert "force" in result.reason.lower()


def test_wrist_unsafe_torque():
    lim = DEFAULT_LIMITS
    result = check_wrist_safety(
        np.zeros(3),
        np.array([0, lim.wrist_torque_max + 0.1, 0]),
    )
    assert not result.ok
    assert "torque" in result.reason.lower()


# ── 11. is_ee_reachable ───────────────────────────────────────────────────────

def _base_state():
    base_pos  = np.array([0.0, 0.0, 0.15])
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])   # identity
    return base_pos, base_quat


def test_reachable_point_in_workspace():
    base_pos, base_quat = _base_state()
    # Point 0.5 m in front at shoulder height — well within reach_max
    target = np.array([0.5, 0.0, base_pos[2] + DEFAULT_LIMITS.shoulder_in_base[2]])
    assert is_ee_reachable(target, base_pos, base_quat)


def test_unreachable_point_too_far():
    base_pos, base_quat = _base_state()
    target = np.array([5.0, 0.0, 0.3])   # 5 m away — outside reach_max
    assert not is_ee_reachable(target, base_pos, base_quat)


def test_unreachable_point_in_singularity_zone():
    base_pos, base_quat = _base_state()
    # Target at the shoulder itself — inside reach_min (0.15 m)
    shoulder_world = base_pos + DEFAULT_LIMITS.shoulder_in_base
    target = shoulder_world + np.array([0.05, 0.0, 0.0])   # 0.05 m < reach_min
    assert not is_ee_reachable(target, base_pos, base_quat)


def test_reachable_at_exactly_reach_max():
    base_pos, base_quat = _base_state()
    lim = DEFAULT_LIMITS
    shoulder_world = base_pos + lim.shoulder_in_base
    # Exactly at reach_max along X axis
    target = shoulder_world + np.array([lim.reach_max, 0.0, 0.0])
    assert is_ee_reachable(target, base_pos, base_quat)


# ── 12. Physics: forcerange is enforced in MuJoCo ─────────────────────────────

def test_arm_actuator_output_capped(model):
    """
    Apply a huge ctrl signal and verify the actual generalized force
    produced stays within forcerange.
    """
    d = mujoco.MjData(model)
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, d, kf)

    # Slam ctrl to 10× the arm's ctrlrange — should be clipped
    d.ctrl[:] = 0.0
    d.ctrl[2] = 100.0   # arm_j1 commanded to 100 rad (way outside ctrlrange)
    mujoco.mj_forward(model, d)

    # actuator_force[2] = output of arm_j1 — must not exceed forcerange
    force = float(d.actuator_force[2])
    assert abs(force) <= DEFAULT_LIMITS.joint_force_max[0] + 1.0   # 1 N tolerance
