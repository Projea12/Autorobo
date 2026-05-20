"""
planning/uncertainty_pipeline.py — Cross-layer uncertainty propagation.

Chains perception → navigation → grasp confidence scores and propagates
uncertainty so that low confidence in any upstream layer degrades the
effective confidence of all downstream layers.

Propagation model
──────────────────
Each layer produces its own score s_i ∈ [0, 1].  The propagated score
is the weighted geometric mean of all layer scores:

    propagated = (s_perception^α  ×  s_nav^β  ×  s_grasp^γ)^(1/(α+β+γ))

Geometric mean is chosen over arithmetic mean because:
  • A single near-zero layer (e.g. perception score 0.1) drives the
    overall confidence down strongly, even if the other layers are good.
  • It is symmetric in the sense that equal weights give equal influence.
  • It degrades gracefully: doubling one bad score has diminishing returns.

Weight defaults (α, β, γ) = (0.40, 0.30, 0.30):
  Perception is weighted highest because all downstream reasoning depends
  on perceiving the scene correctly.  Nav and grasp share the remainder.

Effective (propagated) layer scores
─────────────────────────────────────
As well as the overall propagated score, the pipeline also exports the
effective score for each layer — the layer's own score multiplied by the
product of all upstream propagation factors.  This lets the decision gate
pinpoint which layer is the bottleneck:

    eff_perception = s_perception
    eff_nav        = s_nav  × f(s_perception)
    eff_grasp      = s_grasp × f(s_perception) × f(s_nav)

where f(s) = s^(weight / total_weight).

Usage
─────
    pipeline = UncertaintyPipeline()
    result = pipeline.propagate(
        perception_score = scene.global_score,      # from SceneAggregator
        nav_score        = nav_conf.combined,        # from NavigationConfidenceScorer
        grasp_score      = grasp_conf.combined,      # from GraspConfidenceScorer
    )
    print(result.propagated)          # overall confidence
    print(result.bottleneck_layer)    # which layer is pulling confidence down
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PropagationConfig:
    """
    Weights for the cross-layer geometric-mean propagation.

    perception_weight : exponent α for perception layer (highest: all downstream
                        reasoning depends on seeing the scene correctly)
    nav_weight        : exponent β for navigation layer
    grasp_weight      : exponent γ for grasp layer
    floor             : minimum propagated score — prevents division-by-zero
                        edge cases and keeps downstream decisions non-trivial
    """
    perception_weight: float = 0.40
    nav_weight:        float = 0.30
    grasp_weight:      float = 0.30
    floor:             float = 0.0


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class LayeredConfidence:
    """
    Full cross-layer confidence state after propagation.

    Fields
    ──────
    perception_score     : raw perception layer score [0,1]
    nav_score            : raw navigation layer score [0,1]
    grasp_score          : raw grasp layer score [0,1]
    propagated           : weighted geometric mean across all layers [0,1]
    eff_perception       : effective perception (= perception_score)
    eff_nav              : effective nav = nav_score × upstream perception factor
    eff_grasp            : effective grasp = grasp_score × upstream factors
    bottleneck_layer     : name of the layer with the lowest raw score
    bottleneck_score     : score of that layer
    """
    perception_score: float
    nav_score:        float
    grasp_score:      float
    propagated:       float
    eff_perception:   float
    eff_nav:          float
    eff_grasp:        float
    bottleneck_layer: str
    bottleneck_score: float

    def __repr__(self) -> str:
        return (f"LayeredConfidence("
                f"propagated={self.propagated:.3f}, "
                f"perception={self.perception_score:.2f}, "
                f"nav={self.nav_score:.2f}, "
                f"grasp={self.grasp_score:.2f}, "
                f"bottleneck={self.bottleneck_layer}@{self.bottleneck_score:.2f})")


# ── pipeline ──────────────────────────────────────────────────────────────────

class UncertaintyPipeline:
    """
    Propagates confidence across the perception → navigation → grasp chain.

    Parameters
    ----------
    cfg : PropagationConfig
    """

    def __init__(self, cfg: PropagationConfig = PropagationConfig()) -> None:
        self.cfg = cfg

    def propagate(
        self,
        perception_score: float,
        nav_score:        float,
        grasp_score:      float,
    ) -> LayeredConfidence:
        """
        Combine three layer scores via weighted geometric mean.

        Parameters
        ----------
        perception_score : [0,1] from SceneAggregator.aggregate().global_score
        nav_score        : [0,1] from NavigationConfidenceScorer.score().combined
        grasp_score      : [0,1] from GraspConfidenceScorer.score().combined

        Returns
        -------
        LayeredConfidence with propagated score and per-layer effective scores.
        """
        p = float(np.clip(perception_score, 0.0, 1.0))
        n = float(np.clip(nav_score,        0.0, 1.0))
        g = float(np.clip(grasp_score,      0.0, 1.0))

        cfg     = self.cfg
        alpha   = cfg.perception_weight
        beta    = cfg.nav_weight
        gamma   = cfg.grasp_weight
        total_w = alpha + beta + gamma

        if total_w <= 0.0:
            propagated = 0.0
        elif p == 0.0 or n == 0.0 or g == 0.0:
            propagated = max(cfg.floor, 0.0)
        else:
            log_prop   = (alpha * np.log(p) + beta * np.log(n) + gamma * np.log(g)) / total_w
            propagated = float(np.clip(np.exp(log_prop), cfg.floor, 1.0))

        # Effective per-layer scores: each degraded by upstream factors
        perc_factor = p ** (alpha / total_w) if p > 0 else 0.0
        nav_factor  = n ** (beta  / total_w) if n > 0 else 0.0

        eff_p = p
        eff_n = float(np.clip(n * perc_factor, 0.0, 1.0))
        eff_g = float(np.clip(g * perc_factor * nav_factor, 0.0, 1.0))

        # Bottleneck: whichever raw layer score is lowest
        scores = {"perception": p, "navigation": n, "grasp": g}
        bottleneck = min(scores, key=scores.__getitem__)

        return LayeredConfidence(
            perception_score = p,
            nav_score        = n,
            grasp_score      = g,
            propagated       = propagated,
            eff_perception   = eff_p,
            eff_nav          = eff_n,
            eff_grasp        = eff_g,
            bottleneck_layer = bottleneck,
            bottleneck_score = scores[bottleneck],
        )

    def propagate_partial(
        self,
        scores:  dict[str, float],
    ) -> float:
        """
        Propagate a subset of layers by name.

        Only the layers present in `scores` contribute to the geometric mean.
        Missing layers are ignored (their weight is not included in total_w).

        Parameters
        ----------
        scores : dict with any subset of keys
                 {"perception", "navigation", "grasp"} → score [0,1]

        Returns
        -------
        Propagated score in [0,1].  Returns 0.0 if scores is empty.
        """
        weight_map = {
            "perception": self.cfg.perception_weight,
            "navigation": self.cfg.nav_weight,
            "grasp":      self.cfg.grasp_weight,
        }
        total_w    = sum(weight_map[k] for k in scores if k in weight_map)
        if total_w <= 0.0 or not scores:
            return 0.0

        log_sum = 0.0
        for name, s in scores.items():
            if name not in weight_map:
                continue
            s = float(np.clip(s, 0.0, 1.0))
            if s == 0.0:
                return max(self.cfg.floor, 0.0)
            log_sum += weight_map[name] * np.log(s)

        return float(np.clip(np.exp(log_sum / total_w), self.cfg.floor, 1.0))

    def __repr__(self) -> str:
        cfg = self.cfg
        return (f"UncertaintyPipeline("
                f"α={cfg.perception_weight}, "
                f"β={cfg.nav_weight}, "
                f"γ={cfg.grasp_weight})")
