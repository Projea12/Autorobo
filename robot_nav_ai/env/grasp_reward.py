"""
env/grasp_reward.py — Reward function for the manipulation (grasp) layer.

Six independent reward components are computed and summed each step.
Every component is logged separately so W&B dashboards reveal which
signals are driving policy updates.

Component overview
══════════════════

1. APPROACH  (dense, signed)
   ─────────────────────────
   Reward proportional to EE distance closed toward the object centroid.

       r_approach = w_approach × (d_prev − d_curr)

   Positive when approaching, negative when retreating.

2. CONTACT  (dense, gated)
   ─────────────────────────
   Reward when ≥ 1 fingertip is in contact with the object.
   Scaled by the number of fingertips in contact (0, 1, or 2).

       r_contact = w_contact × (n_touching / 2)

3. LIFT  (dense, gated on contact)
   ─────────────────────────────────
   Zero until contact is established.  Once the object is being lifted,
   reward is proportional to height above the table surface.

       r_lift = w_lift × max(0, obj_z − table_z − lift_thresh)

   This ensures the policy learns: approach → contact → lift.

4. STABILITY  (dense, gated on lift)
   ────────────────────────────────────
   Rewards maintaining both fingertips in contact while the object is
   above the lift threshold.  Punishes drops (object height drops while
   gripper was closed).

       r_stability = w_stability × (n_touching / 2)   if obj_z > lift_thresh
       r_stability = −w_drop_penalty                   if drop detected

5. SYMMETRY  (dense, approach phase only)
   ─────────────────────────────────────────
   Rewards approaching along the object's grasp axis (e.g. from above for
   cylindrical objects), measured as the alignment of the EE approach
   vector with the preferred grasp direction.

       r_symmetry = w_symmetry × max(0, cos θ)

   where θ is the angle between (obj_pos − ee_pos) and grasp_axis.
   Positive only (clipped at 0) so misalignment is not actively penalised
   during approach — it simply earns no bonus.

6. TIME PENALTY  (dense, per step)
   ────────────────────────────────
   Constant per-step penalty to encourage efficient task completion.

       r_time = −w_time_step   every step

Terminal rewards
─────────────────
   SUCCESS  (+w_success) : object lifted ≥ success_height above table
   COLLISION (−w_collision): wrist F/T exceeds safety threshold mid-episode

Total per-step reward
──────────────────────
   r_total = r_approach + r_contact + r_lift + r_stability + r_symmetry
           + r_time
           + r_success      (on success step only)
           + r_collision    (on safety-stop step only)

Usage
─────
    fn = GraspRewardFunction(cfg=GraspRewardConfig())
    fn.reset(obj_pos=initial_object_position)

    info = fn.step(
        ee_pos        = current_ee_position,       # (3,) world frame
        obj_pos       = current_object_position,   # (3,) world frame
        n_touching    = 2,                         # 0, 1, or 2 fingertips
        wrist_safe    = True,                      # from arm controller F/T check
        grasp_axis    = np.array([0, 0, -1]),      # preferred approach direction
    )
    reward    = info.total
    success   = info.success
    terminated = info.terminated
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraspRewardConfig:
    """
    Weights and thresholds for all grasp reward components.

    approach      : weight for EE approach shaping (per metre)
    contact       : weight when fingertip(s) touch object
    lift          : weight for dense lift shaping (per metre above table)
    stability     : weight for maintaining contact during lift
    symmetry      : weight for approach-axis alignment bonus
    time_step     : per-step time penalty
    drop_penalty  : one-time penalty if object drops after being lifted
    success       : terminal bonus for successful lift
    collision     : terminal penalty for wrist F/T safety stop

    table_z       : table surface z-coordinate (metres, world frame)
    lift_thresh   : object must exceed table_z + lift_thresh to earn lift reward
    success_height: object must exceed table_z + success_height → episode success
    """
    # component weights
    approach:     float = 3.0
    contact:      float = 0.5
    lift:         float = 5.0
    stability:    float = 1.0
    symmetry:     float = 0.2
    time_step:    float = 0.01
    drop_penalty: float = 1.0
    success:      float = 10.0
    collision:    float = 5.0

    # thresholds
    table_z:        float = 0.0    # m — world-frame table surface height
    lift_thresh:    float = 0.02   # m — min lift above table to earn lift reward
    success_height: float = 0.20   # m — lift above table → episode success


# ── reward info (per step) ────────────────────────────────────────────────────

@dataclass
class GraspRewardInfo:
    """
    Per-step breakdown of all reward components.

    Fields
    ------
    approach   : dense EE-approach component
    contact    : fingertip contact component
    lift       : lift-height component
    stability  : grasp-stability component
    symmetry   : approach-axis alignment component
    time       : constant time penalty
    terminal   : success bonus or collision penalty (0 most steps)
    total      : sum of all components
    success    : True on the step the object is successfully lifted
    terminated : True if episode ends this step (success or collision)
    """
    approach:   float = 0.0
    contact:    float = 0.0
    lift:       float = 0.0
    stability:  float = 0.0
    symmetry:   float = 0.0
    time:       float = 0.0
    terminal:   float = 0.0
    total:      float = 0.0
    success:    bool  = False
    terminated: bool  = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self) -> str:
        return (
            f"GraspRewardInfo(total={self.total:.4f}, "
            f"approach={self.approach:.4f}, contact={self.contact:.4f}, "
            f"lift={self.lift:.4f}, stability={self.stability:.4f}, "
            f"sym={self.symmetry:.4f}, time={self.time:.4f}, "
            f"terminal={self.terminal:.4f}, "
            f"{'SUCCESS' if self.success else 'terminated' if self.terminated else 'running'})"
        )


# ── reward function ───────────────────────────────────────────────────────────

class GraspRewardFunction:
    """
    Computes per-step reward for the grasp policy.

    Parameters
    ----------
    cfg : GraspRewardConfig
    """

    def __init__(self, cfg: GraspRewardConfig = GraspRewardConfig()) -> None:
        self.cfg = cfg
        self._prev_ee_dist:  float = 0.0
        self._prev_obj_z:    float = 0.0
        self._contact_made:  bool  = False
        self._lifted:        bool  = False

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self, obj_pos: np.ndarray, ee_pos: Optional[np.ndarray] = None) -> None:
        """
        Call once per episode before the first step.

        Parameters
        ----------
        obj_pos : (3,) initial object position in world frame
        ee_pos  : (3,) initial EE position in world frame (optional)
        """
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
        if ee_pos is not None:
            self._prev_ee_dist = float(np.linalg.norm(obj_pos - ee_pos))
        else:
            self._prev_ee_dist = 1.0
        self._prev_obj_z   = float(obj_pos[2])
        self._contact_made = False
        self._lifted       = False

    def step(
        self,
        ee_pos:     np.ndarray,
        obj_pos:    np.ndarray,
        n_touching: int,
        wrist_safe: bool = True,
        grasp_axis: Optional[np.ndarray] = None,
    ) -> GraspRewardInfo:
        """
        Compute reward for one environment step.

        Parameters
        ----------
        ee_pos     : (3,) current end-effector position, world frame
        obj_pos    : (3,) current object centroid position, world frame
        n_touching : number of fingertips in contact with object (0, 1, 2)
        wrist_safe : False if wrist F/T exceeded safety threshold this step
        grasp_axis : (3,) preferred grasp approach direction (unit vector).
                     None → symmetry component is 0.

        Returns
        -------
        GraspRewardInfo with per-component breakdown and termination flags.
        """
        ee_pos  = np.asarray(ee_pos,  dtype=np.float64)
        obj_pos = np.asarray(obj_pos, dtype=np.float64)

        info = GraspRewardInfo()

        # ── 1. approach ───────────────────────────────────────────────────────
        curr_dist = float(np.linalg.norm(obj_pos - ee_pos))
        info.approach = self.cfg.approach * (self._prev_ee_dist - curr_dist)
        self._prev_ee_dist = curr_dist

        # ── 2. contact ────────────────────────────────────────────────────────
        contact_frac = float(np.clip(n_touching, 0, 2)) / 2.0
        info.contact = self.cfg.contact * contact_frac
        if n_touching > 0:
            self._contact_made = True

        # ── 3. lift (gated on contact) ────────────────────────────────────────
        obj_z    = float(obj_pos[2])
        lift_h   = obj_z - self.cfg.table_z - self.cfg.lift_thresh
        if self._contact_made and lift_h > 0.0:
            info.lift = self.cfg.lift * lift_h
            self._lifted = True

        # ── 4. stability (gated on lift) ──────────────────────────────────────
        if self._lifted:
            if obj_z > self.cfg.table_z + self.cfg.lift_thresh:
                info.stability = self.cfg.stability * contact_frac
            else:
                # Object dropped after being lifted
                info.stability = -self.cfg.drop_penalty
                self._lifted = False

        # ── 5. symmetry ───────────────────────────────────────────────────────
        if grasp_axis is not None and not self._contact_made:
            approach_vec = obj_pos - ee_pos
            dist = float(np.linalg.norm(approach_vec))
            if dist > 1e-6:
                axis = np.asarray(grasp_axis, dtype=np.float64)
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm > 1e-6:
                    cos_theta = float(
                        np.dot(approach_vec / dist, axis / axis_norm)
                    )
                    info.symmetry = self.cfg.symmetry * max(0.0, cos_theta)

        # ── 6. time penalty ───────────────────────────────────────────────────
        info.time = -self.cfg.time_step

        # ── terminal conditions ────────────────────────────────────────────────
        if not wrist_safe:
            info.terminal   = -self.cfg.collision
            info.terminated = True
        elif obj_z >= self.cfg.table_z + self.cfg.success_height and self._contact_made:
            info.terminal   = self.cfg.success
            info.success    = True
            info.terminated = True

        info.total = (
            info.approach + info.contact + info.lift
            + info.stability + info.symmetry + info.time + info.terminal
        )
        self._prev_obj_z = obj_z
        return info

    @property
    def contact_made(self) -> bool:
        """True if fingertip contact has been achieved at any point this episode."""
        return self._contact_made

    @property
    def lifted(self) -> bool:
        """True if object has been lifted above the lift threshold this episode."""
        return self._lifted

    def __repr__(self) -> str:
        return (
            f"GraspRewardFunction("
            f"w_approach={self.cfg.approach}, "
            f"w_lift={self.cfg.lift}, "
            f"success_height={self.cfg.success_height}m)"
        )
