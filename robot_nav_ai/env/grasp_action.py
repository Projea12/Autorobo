"""
env/grasp_action.py — Action space design for the manipulation (grasp) layer.

Action layout  (7 floats, all in [−1, 1])
══════════════════════════════════════════
  [0]  Δx        End-effector X delta   (−1 = −MAX_POS_DELTA,  +1 = +MAX_POS_DELTA)
  [1]  Δy        End-effector Y delta
  [2]  Δz        End-effector Z delta
  [3]  Δroll     Roll delta             (−1 = −MAX_ROT_DELTA,  +1 = +MAX_ROT_DELTA)
  [4]  Δpitch    Pitch delta
  [5]  Δyaw      Yaw delta
  [6]  gripper   Gripper command        (< GRIPPER_THRESHOLD → open, ≥ → close)

Physical scaling
────────────────
  Δpos_m   = action[0:3] × MAX_POS_DELTA          (metres)
  Δrot_rad = action[3:6] × MAX_ROT_DELTA           (radians)
  gripper  = OPEN if action[6] < GRIPPER_THRESHOLD else CLOSE

Design rationale (see ADR-005)
───────────────────────────────
  Delta control is chosen over absolute pose control because:
  - Smaller, more robust actions are easier for SAC to learn
  - Errors in 3D position estimation from depth projection are tolerated
    by iterative corrections rather than a single open-loop move
  - Consistent with the Phase 8 recovery hierarchy (Level 1 = micro-adjust)

Action smoothing  (optional)
─────────────────────────────
  Exponential moving average applied to position deltas only.
  Rotation and gripper are passed through raw to avoid lag in orientation
  corrections and gripper timing.

  smoothed_t = (1 − α) × smoothed_{t−1} + α × raw_t
  α = 1.0 → no smoothing

Gripper state machine
─────────────────────
  The gripper is binary (open / close) in our parallel-jaw model.
  action[6] is a continuous signal, thresholded at GRIPPER_THRESHOLD (0.0).
  A hysteresis band of ±GRIPPER_HYSTERESIS prevents rapid open/close chatter.

Usage
─────
    proc = GraspActionProcessor(cfg=GraspActionConfig(), dt_env=0.010)
    proc.reset()
    result = proc.process(raw_action)      # raw_action: (7,) float32 in [-1,1]
    # result.delta_pos   : (3,) metres
    # result.delta_euler : (3,) radians
    # result.gripper_cmd : GripperCmd.OPEN or GripperCmd.CLOSE
    # result.gripper_changed : True if state flipped this step
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False


# ── physical constants ────────────────────────────────────────────────────────

MAX_POS_DELTA:   float = 0.05      # metres  — ±5 cm per step
MAX_ROT_DELTA:   float = 0.2618    # radians — ±15° per step
GRIPPER_THRESHOLD: float = 0.0     # action[6] threshold: <0 → open, ≥0 → close
GRIPPER_HYSTERESIS: float = 0.05   # dead-band to prevent chatter

ACTION_DIM: int = 7


# ── gripper state ─────────────────────────────────────────────────────────────

class GripperCmd(Enum):
    OPEN  = "open"
    CLOSE = "close"


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraspActionConfig:
    """
    Configuration for GraspActionProcessor.

    max_pos_delta   : maximum end-effector position change per step (metres)
    max_rot_delta   : maximum end-effector rotation change per step (radians)
    smooth_alpha    : EMA coefficient for position deltas [0,1]; 1.0 = no smoothing
    gripper_thresh  : threshold on action[6] for open/close decision
    gripper_hyst    : hysteresis band around threshold to prevent chatter
    """
    max_pos_delta:  float = MAX_POS_DELTA
    max_rot_delta:  float = MAX_ROT_DELTA
    smooth_alpha:   float = 0.6
    gripper_thresh: float = GRIPPER_THRESHOLD
    gripper_hyst:   float = GRIPPER_HYSTERESIS


# ── processed action ──────────────────────────────────────────────────────────

@dataclass
class GraspPhysicalAction:
    """
    Physical action after scaling and processing.

    Fields
    ------
    delta_pos       : (3,) float32 — X/Y/Z position delta in metres
    delta_euler     : (3,) float32 — roll/pitch/yaw delta in radians
    gripper_cmd     : GripperCmd.OPEN or GripperCmd.CLOSE
    gripper_changed : True if gripper state flipped this step
    raw             : (7,) float32 — original normalised action
    """
    delta_pos:       np.ndarray   # (3,) float32, metres
    delta_euler:     np.ndarray   # (3,) float32, radians
    gripper_cmd:     GripperCmd
    gripper_changed: bool
    raw:             np.ndarray   # (7,) float32

    def __repr__(self) -> str:
        dp = self.delta_pos.tolist()
        dr = [round(v, 4) for v in self.delta_euler.tolist()]
        return (f"GraspPhysicalAction("
                f"Δpos={[round(v*100,1) for v in dp]}cm, "
                f"Δeuler={dr}rad, "
                f"gripper={self.gripper_cmd.value}"
                f"{' [changed]' if self.gripper_changed else ''})")


# ── processor ─────────────────────────────────────────────────────────────────

class GraspActionProcessor:
    """
    Converts a normalised 7-dim SAC action into physical end-effector commands.

    Parameters
    ----------
    cfg    : GraspActionConfig
    dt_env : environment timestep in seconds (informational; not used for scaling)
    """

    def __init__(
        self,
        cfg:    GraspActionConfig = GraspActionConfig(),
        dt_env: float             = 0.010,
    ) -> None:
        self.cfg    = cfg
        self.dt_env = dt_env

        self._smooth_pos: np.ndarray = np.zeros(3, dtype=np.float64)
        self._gripper_state: GripperCmd = GripperCmd.OPEN

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Call once per episode before the first step."""
        self._smooth_pos    = np.zeros(3, dtype=np.float64)
        self._gripper_state = GripperCmd.OPEN

    def process(self, action: np.ndarray) -> GraspPhysicalAction:
        """
        Convert a normalised action vector to physical commands.

        Parameters
        ----------
        action : (7,) float32 in [-1, 1]

        Returns
        -------
        GraspPhysicalAction
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Expected action shape ({ACTION_DIM},), got {action.shape}")

        action = np.clip(action, -1.0, 1.0)

        # Position delta with EMA smoothing
        raw_pos = action[0:3] * self.cfg.max_pos_delta
        self._smooth_pos = (
            (1.0 - self.cfg.smooth_alpha) * self._smooth_pos
            + self.cfg.smooth_alpha * raw_pos
        )
        delta_pos = self._smooth_pos.astype(np.float32)

        # Rotation delta (no smoothing — precision needed for orientation)
        delta_euler = (action[3:6] * self.cfg.max_rot_delta).astype(np.float32)

        # Gripper with hysteresis
        gripper_signal = float(action[6])
        prev_state     = self._gripper_state
        if self._gripper_state == GripperCmd.OPEN:
            if gripper_signal >= self.cfg.gripper_thresh + self.cfg.gripper_hyst:
                self._gripper_state = GripperCmd.CLOSE
        else:
            if gripper_signal < self.cfg.gripper_thresh - self.cfg.gripper_hyst:
                self._gripper_state = GripperCmd.OPEN

        return GraspPhysicalAction(
            delta_pos       = delta_pos,
            delta_euler     = delta_euler,
            gripper_cmd     = self._gripper_state,
            gripper_changed = (self._gripper_state != prev_state),
            raw             = action.astype(np.float32),
        )

    @property
    def gripper_state(self) -> GripperCmd:
        """Current gripper state (may be queried between steps)."""
        return self._gripper_state

    def __repr__(self) -> str:
        return (f"GraspActionProcessor("
                f"max_pos={self.cfg.max_pos_delta*100:.0f}cm, "
                f"max_rot={math.degrees(self.cfg.max_rot_delta):.0f}deg, "
                f"alpha={self.cfg.smooth_alpha})")


# ── Gymnasium action space ────────────────────────────────────────────────────

def make_grasp_action_space() -> "spaces.Box":
    """
    Return the Gymnasium Box action space for the grasp policy.

    Shape : (7,) — all dims in [-1.0, 1.0]
    Dims  : [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
    """
    if not _GYM_AVAILABLE:
        raise ImportError("gymnasium is required. Install with: pip install gymnasium")
    return spaces.Box(
        low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
    )


# ── action space documentation ────────────────────────────────────────────────

ACTION_SPACE_SPEC = {
    "dims": ACTION_DIM,
    "dtype": "float32",
    "range": [-1.0, 1.0],
    "components": [
        {"index": 0, "name": "delta_x",     "unit": "normalised",
         "physical": f"±{MAX_POS_DELTA*100:.0f} cm",    "notes": "EMA smoothed"},
        {"index": 1, "name": "delta_y",     "unit": "normalised",
         "physical": f"±{MAX_POS_DELTA*100:.0f} cm",    "notes": "EMA smoothed"},
        {"index": 2, "name": "delta_z",     "unit": "normalised",
         "physical": f"±{MAX_POS_DELTA*100:.0f} cm",    "notes": "EMA smoothed"},
        {"index": 3, "name": "delta_roll",  "unit": "normalised",
         "physical": f"±{math.degrees(MAX_ROT_DELTA):.0f} deg", "notes": "raw"},
        {"index": 4, "name": "delta_pitch", "unit": "normalised",
         "physical": f"±{math.degrees(MAX_ROT_DELTA):.0f} deg", "notes": "raw"},
        {"index": 5, "name": "delta_yaw",   "unit": "normalised",
         "physical": f"±{math.degrees(MAX_ROT_DELTA):.0f} deg", "notes": "raw"},
        {"index": 6, "name": "gripper",     "unit": "normalised",
         "physical": "open / close",
         "notes": f"threshold={GRIPPER_THRESHOLD}, hysteresis=±{GRIPPER_HYSTERESIS}"},
    ],
}
