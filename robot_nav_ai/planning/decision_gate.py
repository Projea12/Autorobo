"""
planning/decision_gate.py — Configurable threshold decision gate.

Converts a propagated confidence score into one of three actionable decisions:

    ACT    (≥ 0.90) : confidence sufficient — execute the planned action
    GATHER (0.60–0.89): confidence marginal — gather more sensor data first
    SAFER  (< 0.60) : confidence too low   — fall back to a defined safe action

This gate operates on the output of UncertaintyPipeline.propagate() so it
reflects uncertainty across all three layers (perception, navigation, grasp).

Why 0.90 / 0.60?
─────────────────
  0.90 — manipulation tasks have low tolerance for errors (collision, drop).
          Only act when the system is highly confident.
  0.60 — below this, sensor disagreement is severe enough that any action
          risks harm; a pre-defined safer fallback is always better.

SAFER actions (configurable)
─────────────────────────────
  DEFAULT: "halt and request human review"
  Can be overridden per deployment:
    • "retreat to last known safe position"
    • "open gripper and lower arm"
    • "announce uncertainty via TTS and wait"

Per-layer diagnostics
──────────────────────
When the gate returns GATHER or SAFER, it also reports which layer is the
bottleneck so the operator / recovery system knows where to focus:
    reason = "nav bottleneck (0.45) — gather more localisation data"

Usage
─────
    gate = DecisionGate()
    result = gate.evaluate(layered_confidence)
    if result.decision == Decision.ACT:
        arm_controller.execute(plan)
    elif result.decision == Decision.GATHER:
        perception.trigger_extra_frame()
    else:
        safety.execute_safer_action(result.safer_action)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from planning.uncertainty_pipeline import LayeredConfidence

log = logging.getLogger(__name__)


# ── decision ──────────────────────────────────────────────────────────────────

class Decision(Enum):
    ACT    = "act"     # ≥ act_threshold  — execute plan
    GATHER = "gather"  # ≥ gather_threshold — collect more data
    SAFER  = "safer"   # < gather_threshold — fall back to safe action


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DecisionGateConfig:
    """
    Threshold and fallback configuration for DecisionGate.

    act_threshold    : propagated score ≥ this → ACT.    Default 0.90.
    gather_threshold : propagated score ≥ this → GATHER.  Default 0.60.
                       Below gather_threshold → SAFER.
    safer_action     : human-readable description of the fallback safe action.
    annotate_bottleneck: if True, include bottleneck layer info in reason string.
    """
    act_threshold:       float = 0.90
    gather_threshold:    float = 0.60
    safer_action:        str   = "halt and request human review"
    annotate_bottleneck: bool  = True


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class DecisionResult:
    """
    Output of DecisionGate.evaluate().

    Fields
    ──────
    decision         : ACT, GATHER, or SAFER
    score            : propagated confidence score used for the decision
    reason           : human-readable explanation including bottleneck info
    safer_action     : the configured safe fallback (only meaningful on SAFER)
    bottleneck_layer : name of the weakest confidence layer
    bottleneck_score : score of the weakest layer
    """
    decision:         Decision
    score:            float
    reason:           str
    safer_action:     str
    bottleneck_layer: str
    bottleneck_score: float

    def __repr__(self) -> str:
        return (f"DecisionResult({self.decision.value}, "
                f"score={self.score:.3f}, "
                f"bottleneck={self.bottleneck_layer}@{self.bottleneck_score:.2f})")


# ── gate ──────────────────────────────────────────────────────────────────────

class DecisionGate:
    """
    Converts LayeredConfidence into an actionable Decision.

    Decision zones (configurable; defaults shown):
      ≥ 0.90  → ACT    — execute the planned action
      ≥ 0.60  → GATHER — collect more sensor data
      < 0.60  → SAFER  — execute the configured safe fallback

    Parameters
    ----------
    cfg : DecisionGateConfig
    """

    def __init__(self, cfg: DecisionGateConfig = DecisionGateConfig()) -> None:
        self.cfg = cfg

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self, layered: LayeredConfidence) -> DecisionResult:
        """
        Evaluate layered confidence and return an actionable decision.

        Parameters
        ----------
        layered : LayeredConfidence from UncertaintyPipeline.propagate()

        Returns
        -------
        DecisionResult with decision, score, reason, and bottleneck info.
        """
        score    = layered.propagated
        decision = self._classify(score)
        reason   = self._reason(decision, score, layered)

        if decision != Decision.ACT:
            log.info("DecisionGate: %s (score=%.3f, %s)",
                     decision.value, score, reason)

        return DecisionResult(
            decision         = decision,
            score            = score,
            reason           = reason,
            safer_action     = self.cfg.safer_action,
            bottleneck_layer = layered.bottleneck_layer,
            bottleneck_score = layered.bottleneck_score,
        )

    def evaluate_score(self, score: float) -> Decision:
        """
        Classify a raw scalar score without a LayeredConfidence object.

        Useful when only the overall propagated score is available
        (e.g. in unit tests or when layers are not tracked separately).

        Parameters
        ----------
        score : propagated confidence [0, 1]

        Returns
        -------
        Decision
        """
        return self._classify(float(score))

    def threshold_summary(self) -> str:
        """Return a one-line description of the configured zones."""
        return (f"ACT ≥ {self.cfg.act_threshold:.2f}  |  "
                f"GATHER [{self.cfg.gather_threshold:.2f}, {self.cfg.act_threshold:.2f})  |  "
                f"SAFER < {self.cfg.gather_threshold:.2f}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _classify(self, score: float) -> Decision:
        if score >= self.cfg.act_threshold:
            return Decision.ACT
        if score >= self.cfg.gather_threshold:
            return Decision.GATHER
        return Decision.SAFER

    def _reason(
        self,
        decision: Decision,
        score:    float,
        layered:  LayeredConfidence,
    ) -> str:
        base = f"propagated={score:.3f}"
        if decision == Decision.ACT:
            return f"{base} ≥ {self.cfg.act_threshold} → act"
        if not self.cfg.annotate_bottleneck:
            return base

        bn = layered.bottleneck_layer
        bs = layered.bottleneck_score
        if decision == Decision.GATHER:
            return (f"{base} ∈ [{self.cfg.gather_threshold}, {self.cfg.act_threshold}) "
                    f"— {bn} bottleneck ({bs:.2f}): gather more {bn} data")
        return (f"{base} < {self.cfg.gather_threshold} "
                f"— {bn} bottleneck ({bs:.2f}): {self.cfg.safer_action}")

    def __repr__(self) -> str:
        return (f"DecisionGate(act≥{self.cfg.act_threshold}, "
                f"gather≥{self.cfg.gather_threshold})")
