"""
env/grasp_outcome.py — Grasp outcome detector.

Classifies each grasp episode as SUCCESS or one of four named failure modes
by tracking sensor signals across the episode.

Failure modes
─────────────
  MISS       : Gripper never made contact with the object.
               Detected when episode ends (timeout / truncation) and
               touch sensors never fired above contact_thresh.

  SLIP       : Contact was made and object was lifted above lift_thresh,
               but contact was subsequently lost while the object was
               still in the air.  The object fell from the gripper.

  COLLISION  : Wrist force-torque magnitude exceeded the safety threshold
               at any point during the episode.

  DROP       : Object was partially lifted (between lift_thresh and
               success_height) but then descended back below lift_thresh
               before the episode ended.  Distinct from SLIP — the fingers
               may still be touching (object slid down rather than fell).

  SUCCESS    : Object was held above success_height with contact maintained.

Detection priority (highest to lowest when multiple signals fire)
─────────────────────────────────────────────────────────────────
  COLLISION > SLIP > DROP > MISS > SUCCESS

Step-by-step usage
──────────────────
    detector = GraspOutcomeDetector()
    detector.reset()

    for obs, reward, terminated, truncated, info in episode:
        detector.update(obs, info)
        if terminated or truncated:
            outcome = detector.classify()
            print(outcome.result, outcome.reason)
            break

Batch usage (offline from recorded episodes)
────────────────────────────────────────────
    outcome = GraspOutcomeDetector.from_episode(obs_list, info_list)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ── failure / success labels ──────────────────────────────────────────────────

class GraspResult(Enum):
    SUCCESS   = "success"
    MISS      = "miss"
    SLIP      = "slip"
    COLLISION = "collision"
    DROP      = "drop"
    UNKNOWN   = "unknown"   # episode too short / no data


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OutcomeConfig:
    """
    Thresholds for grasp outcome classification.

    contact_thresh   : touch sensor value (normalised) above which
                       a fingertip is considered in contact
    lift_thresh_z    : object z (world frame) above which it is considered lifted
    success_height_z : object z above which episode is a success
    wrist_force_max  : normalised wrist force above which → COLLISION
                       (mirrors WorkspaceLimits.wrist_force_max / scale)

    Observation slice indices (match ManipulationEnv layout exactly):
      touch      : obs[24:26]  — normalised fingertip forces
      ee_pos     : obs[26:29]  — EE world xyz
      wrist_force: obs[33:36]  — normalised wrist force
      target_pos : obs[39:42]  — object centroid world xyz
    """
    contact_thresh:   float = 0.05    # touch > 0.05 → contact
    lift_thresh_z:    float = 0.045   # object z > floor(0.025) + 0.020 lift
    success_height_z: float = 0.225   # floor(0.025) + success(0.200)
    wrist_force_max:  float = 0.90    # fraction of normalised limit → collision

    # obs slice indices
    touch_slice:   tuple = (24, 26)
    wrist_slice:   tuple = (33, 36)
    target_slice:  tuple = (39, 42)


# ── per-episode outcome ───────────────────────────────────────────────────────

@dataclass
class GraspOutcome:
    """
    Classification result for one grasp episode.

    Fields
    ------
    result          : GraspResult enum value
    reason          : human-readable explanation
    contact_made    : True if any fingertip contact was detected
    max_lift_z      : highest object z reached during the episode
    contact_lost    : True if contact was lost after being made
    collision_step  : step index when collision was detected, or -1
    n_steps         : total episode length
    """
    result:         GraspResult
    reason:         str
    contact_made:   bool  = False
    max_lift_z:     float = 0.0
    contact_lost:   bool  = False
    collision_step: int   = -1
    n_steps:        int   = 0

    @property
    def success(self) -> bool:
        return self.result == GraspResult.SUCCESS

    @property
    def failure_mode(self) -> Optional[str]:
        if self.success:
            return None
        return self.result.value

    def to_dict(self) -> dict:
        return {
            "result":         self.result.value,
            "reason":         self.reason,
            "contact_made":   self.contact_made,
            "max_lift_z":     round(self.max_lift_z, 4),
            "contact_lost":   self.contact_lost,
            "collision_step": self.collision_step,
            "n_steps":        self.n_steps,
            "success":        self.success,
        }

    def __repr__(self) -> str:
        return (f"GraspOutcome({self.result.value.upper()}, "
                f"contact={self.contact_made}, "
                f"max_z={self.max_lift_z:.3f}m, "
                f"steps={self.n_steps})")


# ── detector ──────────────────────────────────────────────────────────────────

class GraspOutcomeDetector:
    """
    Tracks sensor signals across a grasp episode and classifies the outcome.

    Parameters
    ----------
    cfg : OutcomeConfig
    """

    def __init__(self, cfg: OutcomeConfig = OutcomeConfig()) -> None:
        self.cfg = cfg
        self.reset()

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Call once at the start of each episode."""
        self._step:           int   = 0
        self._contact_made:   bool  = False
        self._contact_active: bool  = False
        self._contact_lost:   bool  = False
        self._collision_step: int   = -1
        self._max_lift_z:     float = 0.0
        self._was_lifted:     bool  = False
        self._drop_detected:  bool  = False
        self._success:        bool  = False

    def update(self, obs: np.ndarray, info: dict) -> None:
        """
        Feed one step's observation and info dict to the detector.

        Parameters
        ----------
        obs  : (45,) float32 observation vector from ManipulationEnv
        info : step info dict (may contain "success" key)
        """
        obs = np.asarray(obs, dtype=np.float64).flatten()
        self._step += 1

        ts, te = self.cfg.touch_slice
        ws, we = self.cfg.wrist_slice
        ps, pe = self.cfg.target_slice

        touch       = obs[ts:te]
        wrist_force = obs[ws:we]
        obj_z       = float(obs[ps + 2]) if (pe - ps) >= 3 else 0.0

        # ── contact ───────────────────────────────────────────────────────────
        in_contact = bool(np.any(touch > self.cfg.contact_thresh))
        if in_contact:
            self._contact_made   = True
            self._contact_active = True
        elif self._contact_active:
            self._contact_lost   = True
            self._contact_active = False

        # ── lift tracking ─────────────────────────────────────────────────────
        self._max_lift_z = max(self._max_lift_z, obj_z)
        if obj_z > self.cfg.lift_thresh_z and self._contact_made:
            self._was_lifted = True

        # ── drop detection ────────────────────────────────────────────────────
        if (self._was_lifted
                and obj_z < self.cfg.lift_thresh_z
                and not self._success):
            self._drop_detected = True

        # ── collision ─────────────────────────────────────────────────────────
        wrist_mag = float(np.linalg.norm(wrist_force))
        if wrist_mag > self.cfg.wrist_force_max and self._collision_step < 0:
            self._collision_step = self._step

        # ── success ───────────────────────────────────────────────────────────
        if info.get("success", False):
            self._success = True
        if obj_z >= self.cfg.success_height_z and self._contact_made:
            self._success = True

    def classify(self) -> GraspOutcome:
        """
        Classify the episode outcome after it ends.

        Returns GraspOutcome with result, reason, and tracking fields.
        """
        if self._step == 0:
            return GraspOutcome(
                result  = GraspResult.UNKNOWN,
                reason  = "no steps recorded",
                n_steps = 0,
            )

        # Priority: COLLISION > SLIP > DROP > MISS > SUCCESS
        if self._collision_step >= 0:
            result = GraspResult.COLLISION
            reason = (f"wrist force exceeded limit at step {self._collision_step} "
                      f"(max_z={self._max_lift_z:.3f}m)")

        elif self._contact_lost and self._was_lifted:
            result = GraspResult.SLIP
            reason = (f"contact lost after lift "
                      f"(max_z={self._max_lift_z:.3f}m, "
                      f"step={self._step})")

        elif self._drop_detected:
            result = GraspResult.DROP
            reason = (f"object descended below lift threshold after being raised "
                      f"(max_z={self._max_lift_z:.3f}m)")

        elif not self._contact_made:
            result = GraspResult.MISS
            reason = "gripper never contacted the object"

        elif self._success:
            result = GraspResult.SUCCESS
            reason = f"object lifted to {self._max_lift_z:.3f}m with contact maintained"

        else:
            result = GraspResult.MISS
            reason = ("contact made but object not lifted to success height "
                      f"(max_z={self._max_lift_z:.3f}m)")

        return GraspOutcome(
            result         = result,
            reason         = reason,
            contact_made   = self._contact_made,
            max_lift_z     = self._max_lift_z,
            contact_lost   = self._contact_lost,
            collision_step = self._collision_step,
            n_steps        = self._step,
        )

    # ── class-level batch helper ──────────────────────────────────────────────

    @classmethod
    def from_episode(
        cls,
        obs_list:  list[np.ndarray],
        info_list: list[dict],
        cfg: OutcomeConfig = OutcomeConfig(),
    ) -> GraspOutcome:
        """
        Classify an episode from a pre-recorded list of observations and infos.

        Parameters
        ----------
        obs_list  : list of (45,) obs arrays, one per step
        info_list : list of step info dicts, one per step

        Returns
        -------
        GraspOutcome
        """
        detector = cls(cfg=cfg)
        for obs, info in zip(obs_list, info_list):
            detector.update(obs, info)
        return detector.classify()

    def __repr__(self) -> str:
        return (f"GraspOutcomeDetector("
                f"step={self._step}, "
                f"contact={self._contact_made}, "
                f"max_z={self._max_lift_z:.3f}m)")
