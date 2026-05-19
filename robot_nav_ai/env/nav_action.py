"""
env/nav_action.py — Action space design for the navigation layer.

Action layout  (2 floats, all in [−1, 1])
══════════════════════════════════════════
  [0]  v_linear   Normalised forward velocity   (−1 = full reverse, +1 = full forward)
  [1]  v_angular  Normalised yaw rate           (−1 = full right turn, +1 = full left turn)

Physical mapping
────────────────
  v_lin  = action[0] × LIN_VEL_MAX          (m/s)
  v_ang  = action[1] × ANG_VEL_MAX          (rad/s)

Differential-drive wheel velocities
────────────────────────────────────
  v_left  = v_lin − (WHEELBASE / 2) × v_ang     (m/s)
  v_right = v_lin + (WHEELBASE / 2) × v_ang     (m/s)
  ctrl_left  = v_left  / WHEEL_RADIUS            (rad/s → actuator input)
  ctrl_right = v_right / WHEEL_RADIUS

Action smoothing  (optional, configurable)
──────────────────────────────────────────
  Exponential moving average on the raw normalised action before physical
  scaling prevents high-frequency jitter from destabilising the chassis.

  smoothed_t = (1 − α) × smoothed_{t−1}  +  α × raw_t

  α = 1.0  → no smoothing (raw action used directly)
  α = 0.3  → moderate low-pass filter

Rate limiting  (optional, configurable)
────────────────────────────────────────
  After smoothing, the change in physical velocity is clipped per step:

    Δv_lin  ≤ lin_acc_max  × dt_env
    Δv_ang  ≤ ang_acc_max  × dt_env

  This models real actuator slew rates and discourages bang-bang policies.

Usage
─────
    proc = ActionProcessor(cfg=ActionConfig(), dt_env=0.010)
    proc.reset()                                # call once per episode
    phys = proc.process(raw_action)             # call each step
    env.ctrl[0] = phys.ctrl_left
    env.ctrl[1] = phys.ctrl_right
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False


# ── physical robot constants (shared with NavigationEnv) ──────────────────────

LIN_VEL_MAX:  float = 1.5    # m/s   — max forward/reverse speed
ANG_VEL_MAX:  float = 2.0    # rad/s — max yaw rate
WHEEL_RADIUS: float = 0.08   # m
WHEELBASE:    float = 0.30   # m     — centre-to-centre wheel separation

# derived max wheel velocity (rad/s) — used as actuator ctrlrange
WHEEL_VEL_MAX: float = (LIN_VEL_MAX + (WHEELBASE / 2) * ANG_VEL_MAX) / WHEEL_RADIUS


# ── action configuration ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionConfig:
    """
    Hyper-parameters governing action processing.

    lin_vel_max   : physical linear velocity limit (m/s)
    ang_vel_max   : physical angular velocity limit (rad/s)
    wheel_radius  : drive wheel radius (m)
    wheelbase     : lateral wheel separation (m)
    smoothing_alpha : EMA coefficient α ∈ (0, 1];  1.0 = no smoothing
    lin_acc_max   : max linear acceleration (m/s²); None = unlimited
    ang_acc_max   : max angular acceleration (rad/s²); None = unlimited
    """
    lin_vel_max:     float          = LIN_VEL_MAX
    ang_vel_max:     float          = ANG_VEL_MAX
    wheel_radius:    float          = WHEEL_RADIUS
    wheelbase:       float          = WHEELBASE
    smoothing_alpha: float          = 1.0    # default: no smoothing
    lin_acc_max:     Optional[float] = None  # default: unlimited
    ang_acc_max:     Optional[float] = None  # default: unlimited

    def __post_init__(self) -> None:
        if not (0.0 < self.smoothing_alpha <= 1.0):
            raise ValueError(
                f"smoothing_alpha must be in (0, 1]; got {self.smoothing_alpha}"
            )
        if self.lin_vel_max <= 0:
            raise ValueError(f"lin_vel_max must be > 0; got {self.lin_vel_max}")
        if self.ang_vel_max <= 0:
            raise ValueError(f"ang_vel_max must be > 0; got {self.ang_vel_max}")
        if self.wheel_radius <= 0:
            raise ValueError(f"wheel_radius must be > 0; got {self.wheel_radius}")
        if self.wheelbase <= 0:
            raise ValueError(f"wheelbase must be > 0; got {self.wheelbase}")

    @property
    def wheel_vel_max(self) -> float:
        """Maximum wheel angular speed (rad/s)."""
        return (self.lin_vel_max + (self.wheelbase / 2) * self.ang_vel_max) / self.wheel_radius


# ── physical action bundle ────────────────────────────────────────────────────

@dataclass
class PhysicalAction:
    """
    Processed action in physical units, ready for direct actuator application.

    v_linear    : forward velocity  (m/s)   — positive = forward
    v_angular   : yaw rate          (rad/s) — positive = CCW / left turn
    ctrl_left   : left wheel angular velocity target  (rad/s)
    ctrl_right  : right wheel angular velocity target (rad/s)
    raw         : original normalised action [v_lin_norm, v_ang_norm]
    smoothed    : EMA-smoothed normalised action
    """
    v_linear:   float
    v_angular:  float
    ctrl_left:  float
    ctrl_right: float
    raw:        np.ndarray   # (2,) float32
    smoothed:   np.ndarray   # (2,) float32

    def __repr__(self) -> str:
        return (
            f"PhysicalAction(v_lin={self.v_linear:+.3f} m/s, "
            f"v_ang={self.v_angular:+.3f} rad/s, "
            f"ctrl=[{self.ctrl_left:+.2f}, {self.ctrl_right:+.2f}] rad/s)"
        )


# ── action processor ──────────────────────────────────────────────────────────

class ActionProcessor:
    """
    Converts raw normalised actions ∈ [−1, 1]² into physical wheel commands,
    applying optional smoothing and rate limiting.

    Parameters
    ----------
    cfg    : ActionConfig — scaling, smoothing, and rate-limit parameters
    dt_env : environment step duration in seconds (used for rate limiting)
    """

    def __init__(
        self,
        cfg:    ActionConfig = ActionConfig(),
        dt_env: float        = 0.010,
    ) -> None:
        if dt_env <= 0:
            raise ValueError(f"dt_env must be > 0; got {dt_env}")
        self._cfg    = cfg
        self._dt_env = dt_env
        self._smoothed  = np.zeros(2, dtype=np.float32)
        self._prev_phys = np.zeros(2, dtype=np.float32)  # [v_lin, v_ang]

    @property
    def cfg(self) -> ActionConfig:
        return self._cfg

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Call at the start of every episode.

        Resets the EMA state and velocity history so the previous episode's
        momentum does not bleed into the new one.
        """
        self._smoothed[:]  = 0.0
        self._prev_phys[:] = 0.0

    def process(self, raw_action: np.ndarray) -> PhysicalAction:
        """
        Convert a raw normalised action to a PhysicalAction.

        Parameters
        ----------
        raw_action : array-like of shape (2,) with values in [−1, 1]

        Returns
        -------
        PhysicalAction with wheel ctrl values ready for env.data.ctrl
        """
        raw = np.asarray(raw_action, dtype=np.float32).flatten()[:2]
        raw = np.clip(raw, -1.0, 1.0)

        # 1. EMA smoothing in normalised space
        α = self._cfg.smoothing_alpha
        self._smoothed = (1.0 - α) * self._smoothed + α * raw

        # 2. Scale to physical units
        v_lin = float(self._smoothed[0]) * self._cfg.lin_vel_max
        v_ang = float(self._smoothed[1]) * self._cfg.ang_vel_max

        # 3. Rate limiting in physical space
        v_lin, v_ang = self._apply_rate_limit(v_lin, v_ang)
        self._prev_phys[0] = v_lin
        self._prev_phys[1] = v_ang

        # 4. Differential-drive kinematics → wheel velocities
        r     = self._cfg.wheel_radius
        hb    = self._cfg.wheelbase / 2.0
        ctrl_l = (v_lin - hb * v_ang) / r
        ctrl_r = (v_lin + hb * v_ang) / r

        # 5. Clip to actuator limits
        wmax   = self._cfg.wheel_vel_max
        ctrl_l = float(np.clip(ctrl_l, -wmax, wmax))
        ctrl_r = float(np.clip(ctrl_r, -wmax, wmax))

        return PhysicalAction(
            v_linear   = v_lin,
            v_angular  = v_ang,
            ctrl_left  = ctrl_l,
            ctrl_right = ctrl_r,
            raw        = raw.copy(),
            smoothed   = self._smoothed.copy(),
        )

    def action_space(self) -> "gym.spaces.Box":  # type: ignore[name-defined]
        """Return the Gymnasium Box action space for this processor."""
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium is required for action_space()")
        import gymnasium as gym
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _apply_rate_limit(self, v_lin: float, v_ang: float) -> tuple[float, float]:
        """Clip velocity change to stay within acceleration bounds."""
        cfg   = self._cfg
        dt    = self._dt_env
        prev_lin, prev_ang = float(self._prev_phys[0]), float(self._prev_phys[1])

        if cfg.lin_acc_max is not None:
            max_dv = cfg.lin_acc_max * dt
            v_lin  = float(np.clip(v_lin, prev_lin - max_dv, prev_lin + max_dv))

        if cfg.ang_acc_max is not None:
            max_dw = cfg.ang_acc_max * dt
            v_ang  = float(np.clip(v_ang, prev_ang - max_dw, prev_ang + max_dw))

        return v_lin, v_ang

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def smoothed_action(self) -> np.ndarray:
        """Current EMA-smoothed normalised action (2,)."""
        return self._smoothed.copy()

    @property
    def prev_physical(self) -> np.ndarray:
        """Previous step's physical velocities [v_lin, v_ang]."""
        return self._prev_phys.copy()


# ── module-level helpers ──────────────────────────────────────────────────────

def make_action_space() -> "gym.spaces.Box":  # type: ignore[name-defined]
    """Return the default Gymnasium action space for the navigation layer."""
    if not _GYM_AVAILABLE:
        raise ImportError("gymnasium is required")
    import gymnasium as gym
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def differential_drive(
    v_lin:    float,
    v_ang:    float,
    radius:   float = WHEEL_RADIUS,
    wheelbase: float = WHEELBASE,
) -> tuple[float, float]:
    """
    Convert (v_linear, v_angular) to (ctrl_left, ctrl_right) in rad/s.

    Parameters
    ----------
    v_lin     : linear velocity (m/s)
    v_ang     : angular velocity (rad/s), positive = CCW
    radius    : wheel radius (m)
    wheelbase : wheel separation (m)

    Returns
    -------
    (ctrl_left, ctrl_right) in rad/s
    """
    hb = wheelbase / 2.0
    return (v_lin - hb * v_ang) / radius, (v_lin + hb * v_ang) / radius


def inverse_differential_drive(
    ctrl_left:  float,
    ctrl_right: float,
    radius:     float = WHEEL_RADIUS,
    wheelbase:  float = WHEELBASE,
) -> tuple[float, float]:
    """
    Convert (ctrl_left, ctrl_right) in rad/s to (v_linear, v_angular).

    Parameters
    ----------
    ctrl_left  : left wheel angular velocity (rad/s)
    ctrl_right : right wheel angular velocity (rad/s)

    Returns
    -------
    (v_linear m/s, v_angular rad/s)
    """
    vl  = ctrl_left  * radius
    vr  = ctrl_right * radius
    hb  = wheelbase  / 2.0
    return (vl + vr) / 2.0, (vr - vl) / (2.0 * hb)
