"""
planning/grasp_confidence.py — Grasp layer confidence scorer.

Converts observable grasp-planning signals into a normalised confidence score
[0, 1] that feeds the uncertainty propagation pipeline.

Component signals
──────────────────
  candidate_quality : score of the best grasp candidate from GraspPlanner
                      (the planner's own ranking, already [0, 1]).
  n_candidates      : number of viable grasp candidates generated.
                      More alternatives → more robust → higher confidence.
  depth_uncertainty : standard deviation of object depth estimate [m].
                      High depth noise makes grasp pose unreliable.
  reachability      : how well the target falls inside the arm's workspace [0,1].
                      1.0 = comfortably reachable, 0.0 = outside reach.
  point_cloud_pts   : number of points describing the object surface.
                      Sparse clouds produce poor PCA / axis estimates.

Combined score
──────────────
  grasp_score = w_candidate  × candidate_score
              + w_candidates × n_candidates_score
              + w_depth      × depth_score
              + w_reach      × reachability_score
              + w_cloud      × cloud_score
              (normalised by total weight, clipped to [0, 1])

Usage
─────
    scorer = GraspConfidenceScorer()
    signals = GraspSignals(
        best_candidate_score = 0.85,
        n_candidates         = 4,
        depth_std_m          = 0.02,
        reachability         = 0.90,
        n_cloud_points       = 300,
    )
    result = scorer.score(signals)
    print(result.combined)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# ── input signals ─────────────────────────────────────────────────────────────

@dataclass
class GraspSignals:
    """
    Observable grasp-planning signals for one planning step.

    Fields
    ──────
    best_candidate_score : planner's own quality score for the top candidate [0,1].
                           0.0 = no viable candidates found.
    n_candidates         : number of grasp candidates generated (0 = plan failed).
    depth_std_m          : 1-σ depth uncertainty of object centroid estimate [m].
    reachability         : workspace reachability of the target pose [0,1].
    n_cloud_points       : number of 3-D points on the object surface.
    """
    best_candidate_score: float = 0.0
    n_candidates:         int   = 0
    depth_std_m:          float = 0.05
    reachability:         float = 1.0
    n_cloud_points:       int   = 0


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraspConfidenceConfig:
    """
    Scaling parameters and component weights for GraspConfidenceScorer.

    n_candidates_sat  : n_candidates ≥ this → n_candidates_score = 1.0
    depth_std_max_m   : std ≥ this → depth_score = 0.0 (too uncertain)
    cloud_pts_min     : n_cloud_points below this → cloud_score = 0.0
    cloud_pts_sat     : n_cloud_points ≥ this → cloud_score = 1.0
    w_candidate       : weight for best-candidate-quality component
    w_candidates      : weight for number-of-candidates component
    w_depth           : weight for depth uncertainty component
    w_reachability    : weight for workspace reachability component
    w_cloud           : weight for point-cloud density component
    """
    n_candidates_sat:  int   = 5
    depth_std_max_m:   float = 0.10
    cloud_pts_min:     int   = 10
    cloud_pts_sat:     int   = 500
    w_candidate:       float = 0.35
    w_candidates:      float = 0.15
    w_depth:           float = 0.20
    w_reachability:    float = 0.20
    w_cloud:           float = 0.10


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class GraspConfidence:
    """
    Grasp confidence breakdown for one planning step.

    Fields
    ──────
    candidate_score    : [0,1] quality of best grasp candidate
    n_candidates_score : [0,1] diversity of viable candidates
    depth_score        : [0,1] inverse of depth uncertainty
    reachability_score : [0,1] workspace reachability
    cloud_score        : [0,1] point cloud density
    combined           : weighted aggregate [0,1]
    signals            : the raw GraspSignals used to compute this result
    """
    candidate_score:    float
    n_candidates_score: float
    depth_score:        float
    reachability_score: float
    cloud_score:        float
    combined:           float
    signals:            GraspSignals

    def __repr__(self) -> str:
        return (f"GraspConfidence(combined={self.combined:.3f}, "
                f"candidate={self.candidate_score:.2f}, "
                f"depth={self.depth_score:.2f}, "
                f"reach={self.reachability_score:.2f})")


# ── scorer ────────────────────────────────────────────────────────────────────

class GraspConfidenceScorer:
    """
    Converts GraspSignals into a GraspConfidence with per-component breakdown.

    Parameters
    ----------
    cfg : GraspConfidenceConfig
    """

    def __init__(self, cfg: GraspConfidenceConfig = GraspConfidenceConfig()) -> None:
        self.cfg = cfg

    def score(self, signals: GraspSignals) -> GraspConfidence:
        """
        Compute grasp confidence from observable signals.

        Parameters
        ----------
        signals : GraspSignals with current grasp-planning measurements

        Returns
        -------
        GraspConfidence — combined score in [0, 1] plus per-component breakdown.
        """
        candidate  = float(np.clip(signals.best_candidate_score, 0.0, 1.0))
        n_cands    = self._n_candidates_score(signals.n_candidates)
        depth      = self._depth_score(signals.depth_std_m)
        reach      = float(np.clip(signals.reachability, 0.0, 1.0))
        cloud      = self._cloud_score(signals.n_cloud_points)

        cfg = self.cfg
        total_w = (cfg.w_candidate + cfg.w_candidates + cfg.w_depth
                   + cfg.w_reachability + cfg.w_cloud)
        combined = float(np.clip(
            (cfg.w_candidate    * candidate
             + cfg.w_candidates * n_cands
             + cfg.w_depth      * depth
             + cfg.w_reachability * reach
             + cfg.w_cloud      * cloud) / total_w,
            0.0, 1.0,
        ))

        return GraspConfidence(
            candidate_score    = candidate,
            n_candidates_score = n_cands,
            depth_score        = depth,
            reachability_score = reach,
            cloud_score        = cloud,
            combined           = combined,
            signals            = signals,
        )

    # ── sub-scores ────────────────────────────────────────────────────────────

    def _n_candidates_score(self, n: int) -> float:
        if n <= 0:
            return 0.0
        return float(min(n / self.cfg.n_candidates_sat, 1.0))

    def _depth_score(self, std_m: float) -> float:
        return float(np.clip(1.0 - std_m / self.cfg.depth_std_max_m, 0.0, 1.0))

    def _cloud_score(self, n_pts: int) -> float:
        lo, hi = self.cfg.cloud_pts_min, self.cfg.cloud_pts_sat
        if n_pts <= lo:
            return 0.0
        if n_pts >= hi:
            return 1.0
        return float((n_pts - lo) / (hi - lo))

    def __repr__(self) -> str:
        return (f"GraspConfidenceScorer("
                f"w_cand={self.cfg.w_candidate}, "
                f"w_depth={self.cfg.w_depth}, "
                f"w_reach={self.cfg.w_reachability})")
