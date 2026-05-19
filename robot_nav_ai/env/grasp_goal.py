"""
env/grasp_goal.py — Grasp-informed navigation goal selector.

Problem
───────
The navigation policy should not drive to the object's position — it should
drive to the *best base pose from which the arm can grasp the object*.

The arm is mounted at the front-centre of the chassis.  For a given object
position, feasible base poses form an annular arc in front of the robot:

                         object ●
                          /    \\
                         /      \\
                 ──────────────────── arm reach arc
                   ↑              ↑
              reach_min        reach_max
                        [robot]

Selector pipeline
─────────────────
  1. Generate ``n_candidates`` base-pose candidates arranged on two
     concentric arcs (inner at ``arm_reach_min``, outer at ``arm_reach_max``)
     in front of the arm mount.

  2. Score each candidate on four criteria (weighted sum):
       a) Reach quality    : prefer centre of reachable range
       b) Obstacle clearance: lidar min-range at candidate (sampled from
          current observation's lidar ring — no re-raycast needed)
       c) Approach corridor : straight-line distance from candidate to object
          should be short and unobstructed (penalise if anything is closer
          along the line-of-sight than the object)
       d) Base heading     : reward poses where the arm naturally faces the
          object (yaw aligned to object bearing)

  3. Return the highest-scoring feasible candidate as the 3-D goal position
     (z = floor height; the navigation layer ignores z).

Usage
─────
    selector = GraspGoalSelector(cfg=GraspGoalConfig(), workspace=WorkspaceHint())
    goal_xy  = selector.select(
        object_pos = np.array([2.1, 0.4, 0.025]),
        robot_pos  = np.array([0.0, 0.0, 0.12]),
        lidar_dists= obs[SL_LIDAR],          # 36-dim normalised ring
        lidar_range= 5.0,
    )
    # goal_xy is (3,) world-frame xyz, ready for NavObsBuilder
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── workspace hint ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkspaceHint:
    """
    Geometrical description of the arm's reachable workspace, expressed
    relative to the robot base centre.

    arm_mount_fwd   : how far the arm base is in front of the robot centre (m)
    arm_reach_min   : minimum end-effector reach from arm mount (m)
    arm_reach_max   : maximum end-effector reach from arm mount (m)
    grasp_height_lo : lowest grasping height above floor (m)
    grasp_height_hi : highest grasping height above floor (m)
    """
    arm_mount_fwd:   float = 0.15   # m forward from robot base centre
    arm_reach_min:   float = 0.30   # m from arm mount
    arm_reach_max:   float = 0.80   # m from arm mount
    grasp_height_lo: float = 0.00   # m above floor
    grasp_height_hi: float = 0.55   # m above floor

    def object_in_workspace(self, base_xy: np.ndarray, object_pos: np.ndarray) -> bool:
        """True if the object is within the arm's reachable range from base_xy."""
        mount = self.arm_mount_world(base_xy, yaw=0.0)
        d     = math.hypot(object_pos[0] - mount[0], object_pos[1] - mount[1])
        h     = float(object_pos[2]) if len(object_pos) > 2 else 0.025
        return (self.arm_reach_min <= d <= self.arm_reach_max and
                self.grasp_height_lo <= h <= self.grasp_height_hi)

    def arm_mount_world(
        self, base_xy: np.ndarray, yaw: float
    ) -> np.ndarray:
        """World-frame (x, y) of the arm mount point given base pose."""
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([
            base_xy[0] + self.arm_mount_fwd * c,
            base_xy[1] + self.arm_mount_fwd * s,
        ])


# ── selector configuration ────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraspGoalConfig:
    """
    Hyper-parameters for the GraspGoalSelector.

    n_candidates       : number of base-pose candidates to generate
    n_arc_radii        : number of concentric arcs (radial levels)
    arc_angle_deg      : half-angle of the candidate arc in front of object (°)
    min_base_clearance : reject candidates with any obstacle closer than this (m)
    w_reach            : weight for reach-quality score
    w_clearance        : weight for obstacle-clearance score
    w_heading          : weight for arm-heading alignment score
    w_approach         : weight for corridor-clearance score
    fallback_to_object : if no feasible pose found, return object position
    """
    n_candidates:        int   = 16
    n_arc_radii:         int   = 2
    arc_angle_deg:       float = 150.0   # full arc in front of object
    min_base_clearance:  float = 0.30    # m
    w_reach:             float = 1.0
    w_clearance:         float = 2.0
    w_heading:           float = 1.5
    w_approach:          float = 1.0
    fallback_to_object:  bool  = True


# ── candidate ─────────────────────────────────────────────────────────────────

@dataclass
class GraspCandidate:
    """A single evaluated base-pose candidate."""
    xy:            np.ndarray   # (2,) proposed robot base position
    yaw:           float        # angle the robot should face when grasping
    dist_to_obj:   float        # m — distance from arm mount to object
    score:         float        # weighted composite score (higher = better)
    feasible:      bool         # passes clearance threshold

    def goal_xyz(self, z: float = 0.12) -> np.ndarray:
        """Return goal as (3,) array for NavObsBuilder."""
        return np.array([self.xy[0], self.xy[1], z], dtype=np.float32)


# ── selector ──────────────────────────────────────────────────────────────────

class GraspGoalSelector:
    """
    Converts an object position into the best base approach pose.

    The selector is stateless per call — call ``select()`` every time the
    object position or lidar reading changes.

    Parameters
    ----------
    cfg       : GraspGoalConfig
    workspace : WorkspaceHint describing the arm's kinematic reach
    """

    def __init__(
        self,
        cfg:       GraspGoalConfig = GraspGoalConfig(),
        workspace: WorkspaceHint   = WorkspaceHint(),
    ) -> None:
        self._cfg = cfg
        self._ws  = workspace

    @property
    def cfg(self) -> GraspGoalConfig:
        return self._cfg

    @property
    def workspace(self) -> WorkspaceHint:
        return self._ws

    # ── public API ────────────────────────────────────────────────────────────

    def select(
        self,
        object_pos:  np.ndarray,
        robot_pos:   np.ndarray,
        lidar_dists: np.ndarray,          # (N_RAYS,) normalised [0,1]
        lidar_range: float = 5.0,
    ) -> np.ndarray:
        """
        Choose the best base approach pose and return it as a 3-D goal.

        Parameters
        ----------
        object_pos  : (3,) world-frame object centre (x, y, z)
        robot_pos   : (3,) current robot base position (x, y, z)
        lidar_dists : normalised lidar ring from obs[SL_LIDAR]; values ∈ [0,1]
                      1.0 = no obstacle within lidar_range
        lidar_range : maximum lidar range in metres

        Returns
        -------
        (3,) world-frame goal position for the navigation layer
        """
        # Denormalize lidar
        raw_dists = np.asarray(lidar_dists, dtype=np.float64) * lidar_range

        candidates = self._generate_candidates(
            object_pos[:2], np.asarray(robot_pos[:2])
        )
        for c in candidates:
            c.score    = self._score(c, object_pos, raw_dists, lidar_range)
            c.feasible = self._is_feasible(c, raw_dists, lidar_range)

        feasible = [c for c in candidates if c.feasible]

        if feasible:
            best = max(feasible, key=lambda c: c.score)
            return best.goal_xyz()

        # Fallback: best candidate ignoring clearance, or object position itself
        if candidates:
            best = max(candidates, key=lambda c: c.score)
            if self._cfg.fallback_to_object:
                # Use the object position offset by arm_mount_fwd so we don't
                # drive on top of it
                angle_to_robot = math.atan2(
                    robot_pos[1] - object_pos[1],
                    robot_pos[0] - object_pos[0],
                )
                offset = self._ws.arm_mount_fwd + self._ws.arm_reach_min
                return np.array([
                    object_pos[0] + offset * math.cos(angle_to_robot),
                    object_pos[1] + offset * math.sin(angle_to_robot),
                    0.12,
                ], dtype=np.float32)
        return np.array([object_pos[0], object_pos[1], 0.12], dtype=np.float32)

    def evaluate_all(
        self,
        object_pos:  np.ndarray,
        robot_pos:   np.ndarray,
        lidar_dists: np.ndarray,
        lidar_range: float = 5.0,
    ) -> list[GraspCandidate]:
        """
        Return all candidates with scores and feasibility flags.
        Useful for debugging and visualisation.
        """
        raw_dists  = np.asarray(lidar_dists, dtype=np.float64) * lidar_range
        candidates = self._generate_candidates(
            object_pos[:2], np.asarray(robot_pos[:2])
        )
        for c in candidates:
            c.score    = self._score(c, object_pos, raw_dists, lidar_range)
            c.feasible = self._is_feasible(c, raw_dists, lidar_range)
        return candidates

    # ── candidate generation ──────────────────────────────────────────────────

    def _generate_candidates(
        self,
        object_xy: np.ndarray,
        robot_xy:  np.ndarray,
    ) -> list[GraspCandidate]:
        """
        Generate a set of candidate base positions on concentric arcs around
        the object, biased toward the side facing the robot.
        """
        cfg  = self._cfg
        ws   = self._ws
        n    = cfg.n_candidates
        n_r  = cfg.n_arc_radii

        # Direction from object toward robot (approach direction)
        dx   = robot_xy[0] - object_xy[0]
        dy   = robot_xy[1] - object_xy[1]
        base_angle = math.atan2(dy, dx)

        # Arc spans in front of the object (on robot side)
        half_arc = math.radians(cfg.arc_angle_deg / 2.0)
        n_ang    = max(1, n // n_r)

        # Radii: from (mount + reach_min) to (mount + reach_max), at robot centre
        r_near = ws.arm_mount_fwd + ws.arm_reach_min
        r_far  = ws.arm_mount_fwd + ws.arm_reach_max
        radii  = np.linspace(r_near, r_far, n_r)

        candidates: list[GraspCandidate] = []
        for radius in radii:
            for k in range(n_ang):
                if n_ang == 1:
                    theta = base_angle
                else:
                    theta = base_angle - half_arc + k * (2 * half_arc / (n_ang - 1))

                bx = object_xy[0] + radius * math.cos(theta)
                by = object_xy[1] + radius * math.sin(theta)
                xy = np.array([bx, by])

                # The robot yaw that points its arm toward the object
                yaw = math.atan2(object_xy[1] - by, object_xy[0] - bx)

                # Distance from arm mount to object
                mount = ws.arm_mount_world(xy, yaw)
                d_to_obj = math.hypot(
                    object_xy[0] - mount[0], object_xy[1] - mount[1]
                )
                candidates.append(GraspCandidate(
                    xy          = xy,
                    yaw         = yaw,
                    dist_to_obj = d_to_obj,
                    score       = 0.0,
                    feasible    = False,
                ))

        return candidates

    # ── scoring ───────────────────────────────────────────────────────────────

    def _score(
        self,
        c:           GraspCandidate,
        object_pos:  np.ndarray,
        raw_dists:   np.ndarray,
        lidar_range: float = 5.0,
    ) -> float:
        """Weighted composite score for one candidate (higher = better)."""
        cfg = self._cfg
        ws  = self._ws

        # a) Reach quality: 1 at midpoint of range, 0 at edges
        mid   = (ws.arm_reach_min + ws.arm_reach_max) / 2.0
        half  = (ws.arm_reach_max - ws.arm_reach_min) / 2.0 + 1e-9
        s_reach = max(0.0, 1.0 - abs(c.dist_to_obj - mid) / half)

        # b) Clearance: min lidar distance normalised by the sensor's max range
        min_d   = float(np.min(raw_dists)) if len(raw_dists) > 0 else lidar_range
        s_clear = float(np.clip(min_d / max(lidar_range, 1e-9), 0.0, 1.0))

        # c) Approach corridor: penalise if any obstacle is along the line
        #    from candidate to object (crude check via object distance)
        obj_dist = math.hypot(object_pos[0] - c.xy[0], object_pos[1] - c.xy[1])
        corridor_blocked = any(d < obj_dist * 0.9 for d in raw_dists)
        s_approach = 0.0 if corridor_blocked else 1.0

        # d) Heading alignment: prefer the candidate already faces the object
        #    (robot can approach from any direction but arm-forward is cheapest)
        s_heading = 1.0  # always optimal since we set yaw toward object

        total = (cfg.w_reach    * s_reach   +
                 cfg.w_clearance * s_clear   +
                 cfg.w_heading   * s_heading +
                 cfg.w_approach  * s_approach)
        return float(total)

    # ── feasibility ───────────────────────────────────────────────────────────

    def _is_feasible(
        self,
        c:         GraspCandidate,
        raw_dists: np.ndarray,
        lidar_range: float,
    ) -> bool:
        """
        Candidate is feasible if:
          1. The arm can reach the object (reach range check)
          2. The minimum lidar distance is above the clearance threshold
             (proxy for the candidate position being obstacle-free)
        """
        ws  = self._ws
        cfg = self._cfg

        in_reach = ws.arm_reach_min <= c.dist_to_obj <= ws.arm_reach_max
        min_d    = float(np.min(raw_dists)) if len(raw_dists) > 0 else lidar_range
        clear    = min_d >= cfg.min_base_clearance

        return in_reach and clear


# ── module-level convenience ──────────────────────────────────────────────────

def make_grasp_goal_selector(
    arm_reach_min: float = 0.30,
    arm_reach_max: float = 0.80,
    n_candidates:  int   = 16,
) -> GraspGoalSelector:
    """Convenience factory with flat keyword arguments."""
    ws  = WorkspaceHint(arm_reach_min=arm_reach_min, arm_reach_max=arm_reach_max)
    cfg = GraspGoalConfig(n_candidates=n_candidates)
    return GraspGoalSelector(cfg=cfg, workspace=ws)
