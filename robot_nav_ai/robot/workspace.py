"""
robot/workspace.py — Workspace limits for AutoRobo v1.

Single source of truth for every numeric limit in the system.
Import DEFAULT_LIMITS everywhere rather than scattering magic numbers.

Physical basis for each limit is documented inline.

Sections
────────
  WorkspaceLimits   — frozen dataclass holding all limits
  DEFAULT_LIMITS    — canonical instance (matches robot.xml forcerange / ctrlrange)
  Arm utilities     — joint position / velocity checking
  Base utilities    — wheel ↔ cmd_vel conversion, clamping
  Gripper utilities — force clamping
  Safety utilities  — wrist F/T threshold checking, EE reachability
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


# ── limit container ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkspaceLimits:
    """
    All robot limits in one immutable object.

    Distances in metres, angles in radians, velocities in rad/s or m/s,
    forces in Newton, torques in Newton·metre.
    """

    # ── arm joint position limits [rad] (6 joints) ───────────────────────────
    # Sourced from joint range attributes in robot.xml.
    # joint2 min = −π/2 prevents the upper arm from driving into the chassis.
    joint_pos_lo: np.ndarray   # shape (6,)
    joint_pos_hi: np.ndarray   # shape (6,)

    # ── arm joint velocity limits [rad/s] ────────────────────────────────────
    # Proximal joints (1–3) are slower — heavier links, larger motors.
    # Distal joints (4–6) are faster — lighter links.
    # Matches UR5e specs: 180°/s proximal, 360°/s distal.
    joint_vel_max: np.ndarray   # shape (6,) — symmetric ±

    # ── arm actuator force / torque limits [N·m] ─────────────────────────────
    # Matches forcerange in robot.xml: 150/150/100/30/30/15 N·m.
    joint_force_max: np.ndarray   # shape (6,) — one-sided max

    # ── base (differential drive) ─────────────────────────────────────────────
    wheel_radius:       float   # m — must match robot.xml cylinder size
    wheelbase:          float   # m — centre-to-centre of drive wheels
    wheel_vel_max:      float   # rad/s — matches ctrlrange ±8 in robot.xml
    base_lin_vel_max:   float   # m/s  — derived: wheel_vel_max × wheel_radius
    base_ang_vel_max:   float   # rad/s — derived: 2 × base_lin_vel_max / wheelbase

    # ── wheel actuator force limit [N·m] ─────────────────────────────────────
    wheel_force_max: float   # matches forcerange ±40 in robot.xml

    # ── gripper ───────────────────────────────────────────────────────────────
    finger_pos_max:   float   # m — max open per finger (matches ctrlrange 0..0.04)
    finger_force_max: float   # N — matches forcerange ±50 in robot.xml

    # ── wrist force-torque safety thresholds ─────────────────────────────────
    # Exceeding these triggers the safety layer (Phase 6).
    # Set at 80 % of the actuator forcerange to allow a margin before hard stop.
    wrist_force_max:  float   # N   — 80 % of arm_j4..j6 force capability
    wrist_torque_max: float   # N·m

    # ── Cartesian workspace of the end-effector ───────────────────────────────
    # Sphere centred on shoulder_lift_link origin (world frame when standing).
    # reach_max ≈ upper_arm(0.30) + forearm(0.26) + wrist_links(0.13) + TCP(0.04)
    # reach_min: singularity avoidance — keep away from shoulder axis.
    reach_min: float   # m
    reach_max: float   # m

    # ── shoulder position in base frame ──────────────────────────────────────
    # Used to compute reach from known base pose.
    # = arm_mount offset (0,0,0.075) + shoulder pan height (0,0,0.040)
    shoulder_in_base: np.ndarray   # (3,) m

    # ── soft-limit fraction ───────────────────────────────────────────────────
    # Joints / forces above soft_limit × hard_limit trigger a warning
    # before the hard limit is reached.
    soft_limit_frac: float   # e.g. 0.90


def _make_default() -> WorkspaceLimits:
    """Build the canonical WorkspaceLimits instance (matches robot.xml)."""

    # Arm joint position limits — mirror joint range in robot.xml
    joint_pos_lo = np.array([-math.pi, -math.pi / 2, -math.pi,
                              -math.pi, -math.pi, -math.pi], dtype=np.float64)
    joint_pos_hi = np.array([ math.pi,  math.pi,      math.pi,
                               math.pi,  math.pi,  math.pi], dtype=np.float64)

    # Arm joint velocity limits [rad/s]
    # Proximal joints (1-3): π rad/s = 180°/s
    # Distal joints  (4-6): 2π rad/s = 360°/s
    joint_vel_max = np.array([math.pi, math.pi, math.pi,
                               2 * math.pi, 2 * math.pi, 2 * math.pi],
                              dtype=np.float64)

    # Arm actuator force limits [N·m] — matches forcerange in robot.xml
    joint_force_max = np.array([150.0, 150.0, 100.0, 30.0, 30.0, 15.0],
                                dtype=np.float64)

    wheel_radius = 0.10    # m — matches <geom type="cylinder" size="0.100 ..."/>
    wheelbase    = 0.450   # m — 2 × wheel_y_offset (0.225)
    wheel_vel_max     = 8.0                              # rad/s — matches ctrlrange ±8
    base_lin_vel_max  = wheel_vel_max * wheel_radius     # 0.80 m/s
    base_ang_vel_max  = 2.0 * base_lin_vel_max / wheelbase  # ≈ 3.56 rad/s

    # Upper arm + forearm + wrist links + TCP offset
    reach_max = 0.30 + 0.26 + 0.045 + 0.040 + 0.025 + 0.042   # ≈ 0.762 m

    shoulder_in_base = np.array([0.0, 0.0, 0.115], dtype=np.float64)
    # 0.075 (arm_mount above chassis) + 0.040 (shoulder_pan height)

    return WorkspaceLimits(
        joint_pos_lo      = joint_pos_lo,
        joint_pos_hi      = joint_pos_hi,
        joint_vel_max     = joint_vel_max,
        joint_force_max   = joint_force_max,
        wheel_radius      = wheel_radius,
        wheelbase         = wheelbase,
        wheel_vel_max     = wheel_vel_max,
        base_lin_vel_max  = base_lin_vel_max,
        base_ang_vel_max  = base_ang_vel_max,
        wheel_force_max   = 40.0,
        finger_pos_max    = 0.040,   # m — matches ctrlrange 0..0.040
        finger_force_max  = 50.0,    # N — matches forcerange ±50
        wrist_force_max   = 50.0,    # N
        wrist_torque_max  = 10.0,    # N·m
        reach_min         = 0.15,    # m — singularity avoidance
        reach_max         = reach_max,
        shoulder_in_base  = shoulder_in_base,
        soft_limit_frac   = 0.90,
    )


#: Canonical limits instance — import this everywhere.
DEFAULT_LIMITS: WorkspaceLimits = _make_default()


# ── result types ──────────────────────────────────────────────────────────────

class LimitCheck(NamedTuple):
    """Result of any limit-checking function."""
    ok:      bool
    reason:  str   # empty string when ok=True


# ── arm utilities ─────────────────────────────────────────────────────────────

def check_joint_positions(
    q: np.ndarray,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> LimitCheck:
    """
    Check whether all 6 arm joint positions are within their hard limits.

    Parameters
    ----------
    q : (6,) array of joint angles in radians

    Returns
    -------
    LimitCheck(ok=False, reason=...) if any joint is out of range.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (6,):
        return LimitCheck(False, f"q must be shape (6,), got {q.shape}")

    for i, (val, lo, hi) in enumerate(
        zip(q, limits.joint_pos_lo, limits.joint_pos_hi)
    ):
        if val < lo or val > hi:
            return LimitCheck(
                False,
                f"joint{i + 1} = {math.degrees(val):.1f}° "
                f"outside [{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°]",
            )
    return LimitCheck(True, "")


def check_joint_velocities(
    dq: np.ndarray,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> LimitCheck:
    """Check that all 6 joint speeds are within their velocity limits."""
    dq = np.asarray(dq, dtype=np.float64)
    if dq.shape != (6,):
        return LimitCheck(False, f"dq must be shape (6,), got {dq.shape}")

    violations = np.where(np.abs(dq) > limits.joint_vel_max)[0]
    if violations.size:
        i = violations[0]
        return LimitCheck(
            False,
            f"joint{i + 1} speed |{dq[i]:.3f}| rad/s "
            f"> limit {limits.joint_vel_max[i]:.3f} rad/s",
        )
    return LimitCheck(True, "")


def soft_joint_warnings(
    q: np.ndarray,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> list[str]:
    """
    Return a list of warning strings for joints approaching their hard limit.
    Empty list means all joints are comfortably within range.
    """
    q = np.asarray(q, dtype=np.float64)
    warnings: list[str] = []
    for i, (val, lo, hi) in enumerate(
        zip(q, limits.joint_pos_lo, limits.joint_pos_hi)
    ):
        span = hi - lo
        soft_lo = lo + span * (1 - limits.soft_limit_frac)
        soft_hi = hi - span * (1 - limits.soft_limit_frac)
        if val < soft_lo or val > soft_hi:
            warnings.append(
                f"joint{i + 1} = {math.degrees(val):.1f}° approaching limit "
                f"[{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°]"
            )
    return warnings


# ── base (differential drive) utilities ──────────────────────────────────────

def cmd_vel_to_wheels(
    v_lin: float,
    v_ang: float,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> tuple[float, float]:
    """
    Convert (linear, angular) body-frame velocity to (left, right) wheel
    angular velocities [rad/s].

    Returns unclamped wheel speeds — call clamp_wheel_commands() afterwards
    if you need to enforce hardware limits.

    v_lin : m/s   — positive = forward
    v_ang : rad/s — positive = turn left (CCW when viewed from above)
    """
    r = limits.wheel_radius
    d = limits.wheelbase
    v_left  = (v_lin - v_ang * d / 2.0) / r
    v_right = (v_lin + v_ang * d / 2.0) / r
    return v_left, v_right


def wheels_to_cmd_vel(
    v_left: float,
    v_right: float,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> tuple[float, float]:
    """
    Convert (left, right) wheel angular velocities [rad/s] to
    (linear [m/s], angular [rad/s]) body-frame velocity.
    """
    r = limits.wheel_radius
    d = limits.wheelbase
    v_lin = r * (v_left + v_right) / 2.0
    v_ang = r * (v_right - v_left) / d
    return v_lin, v_ang


def clamp_wheel_commands(
    v_left: float,
    v_right: float,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> tuple[float, float]:
    """
    Clamp wheel commands to ±wheel_vel_max [rad/s].

    If clamping is required both wheels are scaled by the same factor to
    preserve the intended turning radius (rather than clamping independently).
    """
    max_abs = max(abs(v_left), abs(v_right))
    if max_abs > limits.wheel_vel_max:
        scale = limits.wheel_vel_max / max_abs
        v_left  *= scale
        v_right *= scale
    return v_left, v_right


def clamp_cmd_vel(
    v_lin: float,
    v_ang: float,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> tuple[float, float]:
    """
    Clamp a (linear, angular) velocity command to the base limits.

    Converts to wheel space, clamps proportionally, converts back.
    """
    v_l, v_r = cmd_vel_to_wheels(v_lin, v_ang, limits)
    v_l, v_r = clamp_wheel_commands(v_l, v_r, limits)
    return wheels_to_cmd_vel(v_l, v_r, limits)


# ── gripper utilities ─────────────────────────────────────────────────────────

def clamp_gripper_pos(
    pos: float,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> float:
    """Clamp finger position to [0, finger_pos_max] metres."""
    return float(np.clip(pos, 0.0, limits.finger_pos_max))


def clamp_gripper_force(
    force: float,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> float:
    """Clamp gripper force command to [0, finger_force_max] N."""
    return float(np.clip(force, 0.0, limits.finger_force_max))


# ── wrist / safety utilities ──────────────────────────────────────────────────

def check_wrist_safety(
    force: np.ndarray,
    torque: np.ndarray,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> LimitCheck:
    """
    Check wrist force-torque sensor reading against safety thresholds.

    Parameters
    ----------
    force  : (3,) N   — Fx Fy Fz
    torque : (3,) N·m — Tx Ty Tz

    Returns LimitCheck(ok=False, ...) if either magnitude exceeds threshold.
    """
    f_mag = float(np.linalg.norm(force))
    t_mag = float(np.linalg.norm(torque))

    if f_mag > limits.wrist_force_max:
        return LimitCheck(
            False,
            f"wrist force {f_mag:.1f} N > limit {limits.wrist_force_max:.1f} N",
        )
    if t_mag > limits.wrist_torque_max:
        return LimitCheck(
            False,
            f"wrist torque {t_mag:.2f} N·m > limit {limits.wrist_torque_max:.2f} N·m",
        )
    return LimitCheck(True, "")


# ── Cartesian reachability ────────────────────────────────────────────────────

def is_ee_reachable(
    ee_pos_world: np.ndarray,
    base_pos_world: np.ndarray,
    base_quat_world: np.ndarray,
    limits: WorkspaceLimits = DEFAULT_LIMITS,
) -> bool:
    """
    Return True if the target EE position is inside the arm's reachable sphere.

    Uses a conservative sphere test — necessary but not sufficient for full
    inverse kinematics. The full IK check is done in Phase 5 (manipulation layer).

    Parameters
    ----------
    ee_pos_world    : (3,) target end-effector position in world frame
    base_pos_world  : (3,) current robot base position (qpos[0:3])
    base_quat_world : (4,) base orientation quaternion wxyz
    """
    ee_pos_world   = np.asarray(ee_pos_world,   dtype=np.float64)
    base_pos_world = np.asarray(base_pos_world, dtype=np.float64)

    # Rotate shoulder offset into world frame
    shoulder_world = base_pos_world + _rotate_vec(limits.shoulder_in_base, base_quat_world)

    dist = float(np.linalg.norm(ee_pos_world - shoulder_world))
    return limits.reach_min <= dist <= limits.reach_max


def _rotate_vec(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q (wxyz convention).
    Pure Python implementation — no external dependency.
    """
    w, x, y, z = q / np.linalg.norm(q)
    # Rodrigues' rotation via quaternion sandwich product
    t = 2.0 * np.cross([x, y, z], v)
    return v + w * t + np.cross([x, y, z], t)
