"""
planning/conservative_policy.py — Conservative action policy for uncertain states.

Translates a DecisionGate result into a concrete robot behaviour.  The policy
enforces safe degradation: the robot always does *something* safe rather than
proceeding blindly or freezing without explanation.

Decision → behaviour mapping
─────────────────────────────
  ACT    → EXECUTE       : full speed, run planned action
  GATHER → layer-specific:
      perception bottleneck → RESCAN   : trigger extra camera frame / sweep
      navigation bottleneck → SLOW_DOWN: reduce velocity (default 30% of max)
      grasp bottleneck      → REPOSITION: move arm for better point-cloud view
  SAFER  → HALT          : zero velocity, execute safer_action from gate config

Velocity scaling
─────────────────
  EXECUTE    : scale = 1.0  (no reduction)
  SLOW_DOWN  : scale = cfg.slow_velocity_scale   (default 0.30)
  REPOSITION : scale = cfg.reposition_velocity_scale (default 0.15, cautious)
  RESCAN     : scale = 0.0  (robot stays still while scanning)
  HALT       : scale = 0.0  (full stop)

Usage
─────
    gate   = DecisionGate()
    policy = ConservativePolicy()

    result  = gate.evaluate(layered_confidence)
    c_action = policy.apply(result)

    if c_action.action_type == ConservativeActionType.EXECUTE:
        arm_controller.execute(plan)
    elif c_action.action_type == ConservativeActionType.SLOW_DOWN:
        robot.set_velocity(robot.max_velocity * c_action.velocity_scale)
    elif c_action.action_type == ConservativeActionType.RESCAN:
        perception.trigger_extra_frame()
    elif c_action.action_type == ConservativeActionType.REPOSITION:
        arm_controller.move_to_observe()
    else:  # HALT
        robot.stop()
        safety.execute(c_action.instruction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from planning.decision_gate import Decision, DecisionResult

log = logging.getLogger(__name__)


# ── action types ──────────────────────────────────────────────────────────────

class ConservativeActionType(Enum):
    EXECUTE    = "execute"     # ACT: run the planned action at full speed
    SLOW_DOWN  = "slow_down"   # GATHER/nav: reduce velocity, continue cautiously
    RESCAN     = "rescan"      # GATHER/perception: trigger extra sensor sweep
    REPOSITION = "reposition"  # GATHER/grasp: move arm for better sensor coverage
    HALT       = "halt"        # SAFER: full stop, execute safer fallback


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConservativePolicyConfig:
    """
    Velocity scales and behaviour overrides for ConservativePolicy.

    slow_velocity_scale       : fraction of max velocity when slowing for nav uncertainty.
    reposition_velocity_scale : fraction of max velocity while repositioning arm.
    rescan_extra_frames       : suggested number of extra frames for perception rescan.
    halt_requires_human       : if True, HALT always sets requires_human=True.
    """
    slow_velocity_scale:       float = 0.30
    reposition_velocity_scale: float = 0.15
    rescan_extra_frames:       int   = 3
    halt_requires_human:       bool  = True


# ── output ────────────────────────────────────────────────────────────────────

@dataclass
class ConservativeAction:
    """
    Output of ConservativePolicy.apply().

    Fields
    ──────
    action_type      : what the robot should do (ConservativeActionType)
    velocity_scale   : multiply max velocity by this factor before executing
                       (1.0 = full speed, 0.0 = stop)
    instruction      : human-readable description of the action to take
    requires_human   : True when a human operator must be involved
    bottleneck_layer : which layer triggered this conservative action
    bottleneck_score : raw score of the bottleneck layer
    decision_score   : propagated confidence score from the uncertainty pipeline
    extra_frames     : suggested number of extra perception frames (RESCAN only)
    """
    action_type:      ConservativeActionType
    velocity_scale:   float
    instruction:      str
    requires_human:   bool
    bottleneck_layer: str
    bottleneck_score: float
    decision_score:   float
    extra_frames:     int = 0

    def __repr__(self) -> str:
        return (f"ConservativeAction({self.action_type.value}, "
                f"scale={self.velocity_scale:.2f}, "
                f"bottleneck={self.bottleneck_layer}@{self.bottleneck_score:.2f})")


# ── policy ────────────────────────────────────────────────────────────────────

class ConservativePolicy:
    """
    Maps DecisionGate output to a concrete robot behaviour.

    The policy never raises exceptions — it always produces a safe action.
    Unknown bottleneck layer names fall back to HALT.

    Parameters
    ----------
    cfg : ConservativePolicyConfig
    """

    _GATHER_DISPATCH: dict[str, ConservativeActionType] = {
        "perception": ConservativeActionType.RESCAN,
        "navigation": ConservativeActionType.SLOW_DOWN,
        "grasp":      ConservativeActionType.REPOSITION,
    }

    def __init__(self, cfg: ConservativePolicyConfig = ConservativePolicyConfig()) -> None:
        self.cfg = cfg

    # ── public API ────────────────────────────────────────────────────────────

    def apply(self, decision_result: DecisionResult) -> ConservativeAction:
        """
        Convert a DecisionResult into a ConservativeAction.

        Parameters
        ----------
        decision_result : output of DecisionGate.evaluate()

        Returns
        -------
        ConservativeAction describing what the robot should do next.
        """
        d = decision_result.decision

        if d == Decision.ACT:
            return self._execute(decision_result)
        if d == Decision.GATHER:
            return self._gather(decision_result)
        return self._halt(decision_result)

    def velocity_for(self, decision_result: DecisionResult, max_velocity: float) -> float:
        """
        Return the scaled velocity given a decision and the robot's max velocity.

        Parameters
        ----------
        decision_result : output of DecisionGate.evaluate()
        max_velocity    : the robot's maximum velocity in m/s (or rad/s)

        Returns
        -------
        Scaled velocity in the same units as max_velocity.
        """
        action = self.apply(decision_result)
        return max_velocity * action.velocity_scale

    # ── private builders ──────────────────────────────────────────────────────

    def _execute(self, dr: DecisionResult) -> ConservativeAction:
        return ConservativeAction(
            action_type      = ConservativeActionType.EXECUTE,
            velocity_scale   = 1.0,
            instruction      = "confidence sufficient — execute planned action",
            requires_human   = False,
            bottleneck_layer = dr.bottleneck_layer,
            bottleneck_score = dr.bottleneck_score,
            decision_score   = dr.score,
        )

    def _gather(self, dr: DecisionResult) -> ConservativeAction:
        layer = dr.bottleneck_layer
        atype = self._GATHER_DISPATCH.get(layer, ConservativeActionType.HALT)

        if atype == ConservativeActionType.RESCAN:
            return ConservativeAction(
                action_type      = ConservativeActionType.RESCAN,
                velocity_scale   = 0.0,
                instruction      = (f"perception uncertain ({dr.bottleneck_score:.2f}) — "
                                    f"hold position and trigger {self.cfg.rescan_extra_frames} "
                                    f"extra sensor frames"),
                requires_human   = False,
                bottleneck_layer = layer,
                bottleneck_score = dr.bottleneck_score,
                decision_score   = dr.score,
                extra_frames     = self.cfg.rescan_extra_frames,
            )

        if atype == ConservativeActionType.SLOW_DOWN:
            scale = self.cfg.slow_velocity_scale
            return ConservativeAction(
                action_type      = ConservativeActionType.SLOW_DOWN,
                velocity_scale   = scale,
                instruction      = (f"navigation uncertain ({dr.bottleneck_score:.2f}) — "
                                    f"reduce velocity to {scale*100:.0f}% and gather "
                                    f"more localisation data"),
                requires_human   = False,
                bottleneck_layer = layer,
                bottleneck_score = dr.bottleneck_score,
                decision_score   = dr.score,
            )

        if atype == ConservativeActionType.REPOSITION:
            scale = self.cfg.reposition_velocity_scale
            return ConservativeAction(
                action_type      = ConservativeActionType.REPOSITION,
                velocity_scale   = scale,
                instruction      = (f"grasp uncertain ({dr.bottleneck_score:.2f}) — "
                                    f"reposition arm at {scale*100:.0f}% speed for "
                                    f"better point-cloud coverage"),
                requires_human   = False,
                bottleneck_layer = layer,
                bottleneck_score = dr.bottleneck_score,
                decision_score   = dr.score,
            )

        # Unknown layer — fall through to halt
        return self._halt(dr)

    def _halt(self, dr: DecisionResult) -> ConservativeAction:
        return ConservativeAction(
            action_type      = ConservativeActionType.HALT,
            velocity_scale   = 0.0,
            instruction      = dr.safer_action,
            requires_human   = self.cfg.halt_requires_human,
            bottleneck_layer = dr.bottleneck_layer,
            bottleneck_score = dr.bottleneck_score,
            decision_score   = dr.score,
        )

    def __repr__(self) -> str:
        return (f"ConservativePolicy("
                f"slow={self.cfg.slow_velocity_scale}, "
                f"reposition={self.cfg.reposition_velocity_scale})")
