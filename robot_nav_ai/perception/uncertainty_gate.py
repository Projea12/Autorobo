"""
perception/uncertainty_gate.py — Uncertainty gate for perception-conditioned action.

Converts a SceneConfidence into one of four actionable decisions:

    ACT    — confidence sufficient; the agent should proceed
    GATHER — confidence marginal; collect more sensor data before acting
    FLAG   — confidence too low; request human review or abort
    SKIP   — no objects above minimum count; nothing to act on

Decision logic
──────────────
1. If n_objects < min_objects → SKIP (early exit)
2. Compute per-object decisions via evaluate_object():
      a. Mandatory checks (require_depth / require_seg) → FLAG on failure
      b. Otherwise threshold on ObjectConfidence.combined:
            combined ≥ act_threshold    → ACT
            combined ≥ gather_threshold → GATHER
            otherwise                   → FLAG
3. Scene decision = threshold on SceneConfidence.global_score
4. Scene decision is upgraded to the worst per-object decision
   (FLAG > GATHER > ACT), so a single flagged object prevents a premature ACT.

Usage
─────
    from perception.uncertainty_gate import UncertaintyGate, GateConfig, GateDecision
    from perception.confidence import SceneAggregator

    agg   = SceneAggregator()
    gate  = UncertaintyGate(GateConfig(act_threshold=0.75))

    scene = agg.aggregate(detections, projections)
    result = gate.evaluate(scene)
    if result.decision == GateDecision.ACT:
        robot.grasp(scene.objects[0].detection.position_3d)
    elif result.decision == GateDecision.GATHER:
        robot.look_closer()
    else:
        log.warning("low confidence: %s", result.reason)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from perception.confidence import ObjectConfidence, SceneConfidence
from perception.detector import Detection


# ── decision enum ─────────────────────────────────────────────────────────────

class GateDecision(Enum):
    ACT    = "act"     # confidence sufficient — execute planned action
    GATHER = "gather"  # confidence marginal  — collect more sensor data
    FLAG   = "flag"    # confidence too low   — request human review
    SKIP   = "skip"    # no objects detected  — nothing to act on


# Priority used to find the "worst" decision; higher = worse
_PRIORITY: dict[GateDecision, int] = {
    GateDecision.ACT:    0,
    GateDecision.GATHER: 1,
    GateDecision.FLAG:   2,
}


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GateConfig:
    """
    Threshold and mandatory-check configuration for UncertaintyGate.

    act_threshold    : combined score ≥ this → ACT
    gather_threshold : combined score ≥ this → GATHER; below → FLAG
    require_depth    : if True, any object with depth_score == 0.0 → FLAG
                       regardless of combined score
    require_seg      : if True, any object with seg_score == 0.0 → FLAG
    min_objects      : fewer detected objects than this → SKIP
    """
    act_threshold:    float = 0.70
    gather_threshold: float = 0.40
    require_depth:    bool  = True
    require_seg:      bool  = False
    min_objects:      int   = 1


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """
    Output of UncertaintyGate.evaluate().

    Fields
    ------
    decision         : scene-level GateDecision
    reason           : human-readable explanation (for logging / debugging)
    score            : global_score used for the scene-level decision
    object_decisions : per-object (Detection, GateDecision) pairs
    """
    decision:         GateDecision
    reason:           str
    score:            float
    object_decisions: list[tuple[Detection, GateDecision]]

    def __repr__(self) -> str:
        return (f"GateResult({self.decision.value}, score={self.score:.3f}, "
                f"reason={self.reason!r})")


# ── gate ──────────────────────────────────────────────────────────────────────

class UncertaintyGate:
    """
    Converts SceneConfidence into an actionable GateDecision.

    The scene decision is the worst of:
      • the score-based scene decision (threshold on global_score)
      • the worst per-object decision

    This prevents a premature ACT when even one object fails a mandatory check
    or falls below the act threshold.

    Parameters
    ----------
    cfg : GateConfig
    """

    def __init__(self, cfg: GateConfig = GateConfig()) -> None:
        self.cfg = cfg

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self, scene: SceneConfidence) -> GateResult:
        """
        Evaluate a SceneConfidence and return an actionable GateResult.

        Parameters
        ----------
        scene : SceneConfidence from SceneAggregator.aggregate()

        Returns
        -------
        GateResult with scene-level and per-object decisions.
        """
        if scene.n_objects < self.cfg.min_objects:
            return GateResult(
                decision         = GateDecision.SKIP,
                reason           = (f"n_objects={scene.n_objects} < "
                                    f"min_objects={self.cfg.min_objects}"),
                score            = scene.global_score,
                object_decisions = [],
            )

        obj_decisions = [
            (obj.detection, self.evaluate_object(obj)) for obj in scene.objects
        ]

        scene_dec = self._score_to_decision(scene.global_score)
        reason    = f"global_score={scene.global_score:.3f}"

        worst = max((d for _, d in obj_decisions), key=lambda d: _PRIORITY[d])
        if _PRIORITY[worst] > _PRIORITY[scene_dec]:
            scene_dec = worst
            reason   += f"; worst_object={worst.value}"

        return GateResult(
            decision         = scene_dec,
            reason           = reason,
            score            = scene.global_score,
            object_decisions = obj_decisions,
        )

    def evaluate_object(self, obj: ObjectConfidence) -> GateDecision:
        """
        Per-object decision considering mandatory checks and combined score.

        Mandatory check failures (require_depth, require_seg) return FLAG
        immediately, regardless of the combined score.
        """
        if self.cfg.require_depth and obj.depth_score == 0.0:
            return GateDecision.FLAG
        if self.cfg.require_seg and obj.seg_score == 0.0:
            return GateDecision.FLAG
        return self._score_to_decision(obj.combined)

    # ── internals ─────────────────────────────────────────────────────────────

    def _score_to_decision(self, score: float) -> GateDecision:
        if score >= self.cfg.act_threshold:
            return GateDecision.ACT
        if score >= self.cfg.gather_threshold:
            return GateDecision.GATHER
        return GateDecision.FLAG

    def __repr__(self) -> str:
        return (f"UncertaintyGate(act={self.cfg.act_threshold}, "
                f"gather={self.cfg.gather_threshold})")
