"""
planning/nav_confidence.py — Navigation layer confidence scorer.

Converts observable navigation signals into a normalised confidence score
[0, 1] that feeds the uncertainty propagation pipeline.

Component signals
──────────────────
  clearance    : minimum obstacle clearance along the planned path.
                 Low clearance → high collision risk → lower confidence.
  path_quality : derived from planned path length and waypoint count.
                 Very long paths or no valid waypoints → lower confidence.
  localisation : inverse of position uncertainty (localisation std).
                 High positional std → robot doesn't know where it is → lower.
  goal_dist    : distance to goal, penalised only when very far.
                 Nearby goals are always reachable; extreme distances add doubt.

Combined score
──────────────
  nav_score = w_clearance × clearance_score
            + w_path      × path_score
            + w_localisation × localisation_score
            + w_goal_dist × goal_dist_score
            (normalised by total weight, clipped to [0, 1])

Usage
─────
    scorer = NavigationConfidenceScorer()
    signals = NavSignals(
        min_clearance_m    = 0.8,
        path_length_m      = 3.0,
        n_waypoints        = 8,
        localisation_std_m = 0.05,
        goal_distance_m    = 2.5,
    )
    result = scorer.score(signals)
    print(result.combined, result.clearance_score, ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ── input signals ─────────────────────────────────────────────────────────────

@dataclass
class NavSignals:
    """
    Observable navigation signals for one planning step.

    Fields
    ──────
    min_clearance_m    : minimum clearance to any obstacle along path [m].
                         0.0 = blocked, large values = wide-open space.
    path_length_m      : total length of the planned path [m].
    n_waypoints        : number of valid waypoints in the plan (0 = no plan).
    localisation_std_m : 1-σ position uncertainty from localisation [m].
    goal_distance_m    : Euclidean distance to goal [m].
    """
    min_clearance_m:    float = 0.5
    path_length_m:      float = 2.0
    n_waypoints:        int   = 5
    localisation_std_m: float = 0.05
    goal_distance_m:    float = 1.0


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NavConfidenceConfig:
    """
    Scaling parameters and component weights.

    clearance_min_m      : clearance below this → clearance_score = 0
    clearance_sat_m      : clearance at or above this → clearance_score = 1
    path_length_max_m    : path at this length → path_score saturates low end
    n_waypoints_min      : fewer waypoints than this → path_score = 0
    localisation_std_max : std at or above this → localisation_score = 0
    goal_dist_near_m     : distance below this → goal_dist_score = 1 (always reachable)
    goal_dist_far_m      : distance above this → goal_dist_score = 0
    w_clearance          : weight for clearance component
    w_path               : weight for path quality component
    w_localisation       : weight for localisation component
    w_goal_dist          : weight for goal distance component
    """
    clearance_min_m:      float = 0.10
    clearance_sat_m:      float = 1.00
    path_length_max_m:    float = 15.0
    n_waypoints_min:      int   = 1
    localisation_std_max: float = 0.50
    goal_dist_near_m:     float = 3.0
    goal_dist_far_m:      float = 20.0
    w_clearance:          float = 0.35
    w_path:               float = 0.25
    w_localisation:       float = 0.25
    w_goal_dist:          float = 0.15


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class NavConfidence:
    """
    Navigation confidence breakdown for one planning step.

    Fields
    ──────
    clearance_score    : [0,1] based on obstacle clearance
    path_score         : [0,1] based on path length and waypoint count
    localisation_score : [0,1] based on position uncertainty
    goal_dist_score    : [0,1] based on distance to goal
    combined           : weighted aggregate [0,1]
    signals            : the raw NavSignals used to compute this result
    """
    clearance_score:    float
    path_score:         float
    localisation_score: float
    goal_dist_score:    float
    combined:           float
    signals:            NavSignals

    def __repr__(self) -> str:
        return (f"NavConfidence(combined={self.combined:.3f}, "
                f"clearance={self.clearance_score:.2f}, "
                f"path={self.path_score:.2f}, "
                f"loc={self.localisation_score:.2f}, "
                f"goal={self.goal_dist_score:.2f})")


# ── scorer ────────────────────────────────────────────────────────────────────

class NavigationConfidenceScorer:
    """
    Converts NavSignals into a NavConfidence with per-component breakdown.

    Parameters
    ----------
    cfg : NavConfidenceConfig
    """

    def __init__(self, cfg: NavConfidenceConfig = NavConfidenceConfig()) -> None:
        self.cfg = cfg

    def score(self, signals: NavSignals) -> NavConfidence:
        """
        Compute navigation confidence from observable signals.

        Parameters
        ----------
        signals : NavSignals with current navigation measurements

        Returns
        -------
        NavConfidence — combined score in [0, 1] plus per-component breakdown.
        """
        clearance = self._clearance_score(signals.min_clearance_m)
        path      = self._path_score(signals.path_length_m, signals.n_waypoints)
        loc       = self._localisation_score(signals.localisation_std_m)
        goal      = self._goal_dist_score(signals.goal_distance_m)

        cfg = self.cfg
        total_w = cfg.w_clearance + cfg.w_path + cfg.w_localisation + cfg.w_goal_dist
        combined = float(np.clip(
            (cfg.w_clearance * clearance
             + cfg.w_path * path
             + cfg.w_localisation * loc
             + cfg.w_goal_dist * goal) / total_w,
            0.0, 1.0,
        ))

        return NavConfidence(
            clearance_score    = clearance,
            path_score         = path,
            localisation_score = loc,
            goal_dist_score    = goal,
            combined           = combined,
            signals            = signals,
        )

    # ── sub-scores ────────────────────────────────────────────────────────────

    def _clearance_score(self, clearance_m: float) -> float:
        lo, hi = self.cfg.clearance_min_m, self.cfg.clearance_sat_m
        if clearance_m <= lo:
            return 0.0
        if clearance_m >= hi:
            return 1.0
        return float((clearance_m - lo) / (hi - lo))

    def _path_score(self, length_m: float, n_waypoints: int) -> float:
        if n_waypoints < self.cfg.n_waypoints_min:
            return 0.0
        length_score = float(np.clip(
            1.0 - length_m / self.cfg.path_length_max_m, 0.0, 1.0
        ))
        return length_score

    def _localisation_score(self, std_m: float) -> float:
        return float(np.clip(
            1.0 - std_m / self.cfg.localisation_std_max, 0.0, 1.0
        ))

    def _goal_dist_score(self, dist_m: float) -> float:
        near, far = self.cfg.goal_dist_near_m, self.cfg.goal_dist_far_m
        if dist_m <= near:
            return 1.0
        if dist_m >= far:
            return 0.0
        return float(1.0 - (dist_m - near) / (far - near))

    def __repr__(self) -> str:
        return (f"NavigationConfidenceScorer("
                f"w_clear={self.cfg.w_clearance}, "
                f"w_path={self.cfg.w_path}, "
                f"w_loc={self.cfg.w_localisation})")
