"""
robot/trajectory.py — Joint-space trajectory generation (Block 4.5).

Why joint-space interpolation
------------------------------
Cartesian-space interpolation (moving the EE in a straight line) requires
running IK at every waypoint, is sensitive to singularities mid-path, and
can produce unexpected joint flips.  Joint-space interpolation is simpler,
always produces smooth joint motion, and is standard for pick-and-place
where the exact Cartesian path between waypoints doesn't matter — only
the start and end EE poses do.

Cosine easing (smoothstep)
---------------------------
Linear interpolation produces abrupt velocity changes at start/stop that
stress the actuators.  Cosine easing s(t) = 0.5·(1 − cos(π·t)) gives
zero velocity at t=0 and t=1 (smooth ramp-up and ramp-down) while
remaining a monotone map [0,1]→[0,1] — so joint limits are respected as
long as the endpoints are within limits.

Trajectory structure
--------------------
A full grasp trajectory has two segments:

    Segment 0: q_current → q_pre_grasp   (N waypoints, cosine eased)
    Segment 1: q_pre_grasp → q_grasp     (N waypoints, cosine eased)

The pre-grasp hover ensures the arm is correctly aligned before it
descends/advances to the final grasp pose.

Collision checking
------------------
Each waypoint is checked by running mj_fwdPosition() and querying
data.ncon.  Any contact at a waypoint (arm self-collision or floor
contact) marks the waypoint as in collision.  The scene XML has no
external obstacles, so ncon > 0 ⟺ self-collision or floor contact.

Usage
-----
    from robot.kinematics import TidyBotKinematics
    from robot.trajectory import TrajectoryPlanner, Waypoint

    kin     = TidyBotKinematics()
    planner = TrajectoryPlanner(kin)
    traj    = planner.plan(q_current, pre_grasp_xyz, grasp_xyz)
    print(traj)                   # summary
    traj.is_collision_free()      # True/False
    traj.waypoints                # list[Waypoint]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from robot.kinematics import TidyBotKinematics, IKResult, ARM_QPOS_SLICE


# ── defaults ──────────────────────────────────────────────────────────────────

N_WAYPOINTS: int = 30     # waypoints per segment


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    """
    A single point in a joint-space trajectory.

    Attributes
    ----------
    t          : normalised time ∈ [0, 1] across the full trajectory
    q_arm      : (7,) joint angles for joint_1 … joint_7 (radians)
    ee_xyz     : (3,) end-effector position at this waypoint (world frame)
    in_collision : True if MuJoCo detected any contact at this configuration
    segment    : 0 = approach (current→pre-grasp), 1 = descend (pre-grasp→grasp)
    """
    t:            float
    q_arm:        np.ndarray   # (7,)
    ee_xyz:       np.ndarray   # (3,)
    in_collision: bool
    segment:      int


@dataclass
class JointTrajectory:
    """
    Full joint-space trajectory for a grasp motion.

    Attributes
    ----------
    waypoints      : ordered list of Waypoint (segment 0 then segment 1)
    q_start        : (7,) initial arm configuration
    q_pre_grasp    : (7,) IK solution for pre-grasp hover
    q_grasp        : (7,) IK solution for grasp contact point
    ik_pre_grasp   : IKResult for pre-grasp
    ik_grasp       : IKResult for grasp
    n_per_segment  : waypoints per segment
    """
    waypoints:      List[Waypoint]
    q_start:        np.ndarray
    q_pre_grasp:    np.ndarray
    q_grasp:        np.ndarray
    ik_pre_grasp:   IKResult
    ik_grasp:       IKResult
    n_per_segment:  int

    def is_collision_free(self) -> bool:
        """Return True if no waypoint is in collision."""
        return not any(w.in_collision for w in self.waypoints)

    def collision_count(self) -> int:
        """Number of waypoints with collisions."""
        return sum(w.in_collision for w in self.waypoints)

    def ee_path(self) -> np.ndarray:
        """(total_waypoints, 3) array of EE positions along the trajectory."""
        return np.array([w.ee_xyz for w in self.waypoints])

    def q_path(self) -> np.ndarray:
        """(total_waypoints, 7) array of joint angles along the trajectory."""
        return np.array([w.q_arm for w in self.waypoints])

    def __str__(self) -> str:
        n_total = len(self.waypoints)
        n_coll  = self.collision_count()
        status  = "COLLISION-FREE" if n_coll == 0 else f"{n_coll} COLLISIONS"
        pg = self.ik_pre_grasp
        g  = self.ik_grasp
        return (
            f"JointTrajectory [{status}]\n"
            f"  segments    : 2 × {self.n_per_segment} = {n_total} waypoints\n"
            f"  pre-grasp IK: converged={pg.converged}  "
            f"err={pg.final_error*1000:.2f} mm\n"
            f"  grasp IK    : converged={g.converged}  "
            f"err={g.final_error*1000:.2f} mm\n"
            f"  joint range : "
            f"Δq_seg0={float(np.max(np.abs(self.q_pre_grasp - self.q_start))):.3f} rad  "
            f"Δq_seg1={float(np.max(np.abs(self.q_grasp - self.q_pre_grasp))):.3f} rad"
        )


# ── easing ────────────────────────────────────────────────────────────────────

def _cosine_ease(n: int) -> np.ndarray:
    """
    Return n values in [0, 1] with cosine easing applied.

    s(t) = 0.5 · (1 − cos(π·t)) where t is uniformly spaced in [0, 1].
    s(0)=0, s(1)=1 with zero first derivative at both endpoints.
    """
    t = np.linspace(0.0, 1.0, n)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def _interpolate_segment(
    q_start: np.ndarray,
    q_end:   np.ndarray,
    n:       int,
) -> np.ndarray:
    """
    Interpolate n joint configs from q_start to q_end with cosine easing.

    Returns
    -------
    (n, 7) array of joint angles
    """
    s    = _cosine_ease(n)              # (n,) eased parameter
    diff = q_end - q_start             # (7,)
    return q_start[None, :] + s[:, None] * diff[None, :]   # (n, 7)


# ── planner ───────────────────────────────────────────────────────────────────

class TrajectoryPlanner:
    """
    Generates smooth joint-space trajectories for grasp motions.

    Parameters
    ----------
    kin           : TidyBotKinematics instance (model + data already loaded)
    n_per_segment : waypoints per segment  (default 30)
    """

    def __init__(
        self,
        kin:           TidyBotKinematics,
        n_per_segment: int = N_WAYPOINTS,
    ) -> None:
        self.kin           = kin
        self.n_per_segment = n_per_segment

    def plan(
        self,
        q_current:      np.ndarray,
        pre_grasp_xyz:  Tuple[float, float, float] | np.ndarray,
        grasp_xyz:      Tuple[float, float, float] | np.ndarray,
    ) -> JointTrajectory:
        """
        Plan a 2-segment trajectory: current → pre-grasp → grasp.

        Parameters
        ----------
        q_current     : (7,) current arm joint angles (joint_1 … joint_7)
        pre_grasp_xyz : (3,) pre-grasp hover position in world frame
        grasp_xyz     : (3,) grasp contact position in world frame

        Returns
        -------
        JointTrajectory with collision-checked waypoints
        """
        q0 = np.asarray(q_current, dtype=np.float64)

        # IK for both target poses, seeded from current config
        ik_pre = self.kin.ik(pre_grasp_xyz, q_init=q0)
        ik_grs = self.kin.ik(grasp_xyz,     q_init=ik_pre.q_arm)

        # Interpolate each segment
        seg0_q = _interpolate_segment(q0,           ik_pre.q_arm, self.n_per_segment)
        seg1_q = _interpolate_segment(ik_pre.q_arm, ik_grs.q_arm, self.n_per_segment)

        n_total = 2 * self.n_per_segment
        waypoints: List[Waypoint] = []

        for seg_idx, seg_q in enumerate((seg0_q, seg1_q)):
            for step_idx in range(self.n_per_segment):
                q_arm = seg_q[step_idx]

                # Global normalised time ∈ [0, 1]
                global_step = seg_idx * self.n_per_segment + step_idx
                t_global    = global_step / max(n_total - 1, 1)

                # FK + collision check
                ee, collides = self._fk_and_collision(q_arm)

                waypoints.append(Waypoint(
                    t            = t_global,
                    q_arm        = q_arm.copy(),
                    ee_xyz       = ee,
                    in_collision = collides,
                    segment      = seg_idx,
                ))

        return JointTrajectory(
            waypoints     = waypoints,
            q_start       = q0.copy(),
            q_pre_grasp   = ik_pre.q_arm.copy(),
            q_grasp       = ik_grs.q_arm.copy(),
            ik_pre_grasp  = ik_pre,
            ik_grasp      = ik_grs,
            n_per_segment = self.n_per_segment,
        )

    def _fk_and_collision(
        self, q_arm: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Run FK for the given arm configuration and check for contacts.

        Sets base joints to zero (robot stays put during arm motion).

        Returns
        -------
        (ee_xyz, in_collision)
        """
        import mujoco
        qpos = np.zeros(self.kin.model.nq)
        qpos[ARM_QPOS_SLICE] = q_arm

        self.kin.data.qpos[:] = qpos
        mujoco.mj_fwdPosition(self.kin.model, self.kin.data)

        # Collision detection — run broadphase+narrowphase
        mujoco.mj_collision(self.kin.model, self.kin.data)

        ee       = self.kin.data.site_xpos[self.kin._ee_id].copy()
        collides = self.kin.data.ncon > 0

        return ee, collides
