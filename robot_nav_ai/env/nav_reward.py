"""
env/nav_reward.py — Reward function for the navigation layer.

Six independent reward components are computed and summed each step.
Every component is separately logged in RewardInfo so training dashboards
can show which signals are driving policy updates.

Component overview
══════════════════

1. APPROACH  (dense, signed)
   ─────────────────────────
   Reward proportional to distance closed toward the goal this step.

       r_approach = w_approach × (d_prev − d_curr)

   Positive when approaching, negative when retreating.  The weight is
   tuned so a direct 1 m/s walk to a goal 2 m away earns roughly +0.2/s.

2. GOAL_REACHED  (sparse, terminal)
   ─────────────────────────────────
   One-time bonus when the robot enters the goal radius.

       r_goal = w_goal    if d_curr < goal_radius

   Terminates the episode.

3. COLLISION  (sparse, terminal)
   ─────────────────────────────
   One-time penalty when the lidar detects an obstacle below collision_r.
   Applied on the same step as termination.

       r_coll = −w_collision    if d_lidar_min < collision_r

4. OBSTACLE_PROXIMITY  (dense)
   ──────────────────────────────
   Soft penalty that grows as the robot approaches any obstacle, so the
   policy learns to keep a comfortable safety margin even when the hard
   collision threshold is not yet breached.

       proximity = clip(1 − d_lidar_min / danger_r, 0, 1)
       r_obst    = −w_obstacle × proximity    if d_lidar_min < danger_r

5. EXPLORATION  (dense, episode-scoped)
   ──────────────────────────────────────
   Bonus for visiting a map cell not yet seen in the current episode.
   Cells are defined by a discrete grid of size ``explore_cell_m`` metres.

       cell = (floor(x / cell_m), floor(y / cell_m))
       r_explore = w_explore    if cell ∉ visited_cells_this_episode

   Encourages the robot to cover new ground rather than looping near the
   spawn.  The exploration bonus decays per step as unvisited cells become
   scarce (episode-level novelty only; no cross-episode persistence).

6. UNCERTAINTY  (dense)
   ──────────────────────
   Penalises operating under perceptual uncertainty to push the policy
   toward areas and orientations where it can see well.  Two sub-signals:

     a) Perception confidence: if the YOLO detector is active but the
        target-confidence score is below a threshold, apply a penalty
        proportional to the shortfall.

           r_perc = −w_uncertainty × max(0, conf_thresh − confidence)
                    if perception is not None

     b) Occupancy unknowns: the fraction of occupancy-grid cells still
        marked "unknown" (value = 0.5) penalises operating in fog-of-war.

           unknown_frac = count(occ == 0.5) / occ.size
           r_occ  = −w_uncertainty × unknown_frac × occ_uncertainty_scale

   Both sub-signals use the same weight so the total uncertainty penalty
   sums over perceptual and spatial unknowns.

Total per-step reward
─────────────────────
   r_total = r_approach + r_obst + r_explore + r_uncertainty + r_time
           + r_goal          (on success step only)
           + r_collision      (on collision step only)

Time penalty
─────────────
   r_time = −w_time_step    every step (encourages efficiency)

Tuning guide
────────────
  The default weights are calibrated for:
    • Map size ≈ 5 m radius
    • Goal distances 1–4 m
    • Episode length ≤ 500 steps at 10 ms/step (5 s)
    • Speed ≈ 1 m/s

  A good starting ratio is approach:goal:collision ≈ 2:10:5 so the sparse
  goal signal dominates over accumulated dense rewards.

Usage
─────
    fn = NavRewardFunction(cfg=RewardConfig())
    fn.reset(robot_xy=np.array([0.0, 0.0]))       # call once per episode

    info = fn.step(
        robot_xy   = current_pos,
        d_prev     = prev_goal_dist,
        d_curr     = curr_goal_dist,
        d_lidar_min= min_lidar_range,
        occ_grid   = obs[SL_OCC],                 # 64-element slice
        perception = PerceptionInput(conf, brg, dist),   # or None
    )
    reward    = info.total
    terminated = info.terminated
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── reward configuration ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RewardConfig:
    """
    Weights and thresholds for all six reward components.

    approach          : weight for dense goal-approach shaping (per metre)
    goal              : sparse bonus on success
    collision         : sparse penalty on crash
    obstacle          : weight for dense obstacle-proximity penalty
    explore           : bonus per new map cell visited
    uncertainty       : weight for perception + occupancy uncertainty penalty
    time_step         : per-step time penalty
    goal_radius       : success threshold (m)
    collision_r       : collision threshold — lidar min range (m)
    danger_r          : obstacle proximity soft threshold (m)
    explore_cell_m    : discretisation cell size for exploration counting (m)
    conf_thresh       : YOLO confidence below which uncertainty penalty fires
    occ_unknown_scale : scale of occupancy-unknown sub-penalty relative to w
    """
    # component weights
    approach:          float = 2.0
    goal:              float = 10.0
    collision:         float = 5.0
    obstacle:          float = 0.5
    explore:           float = 0.10
    uncertainty:       float = 0.05
    time_step:         float = 0.01

    # thresholds
    goal_radius:       float = 0.25    # m
    collision_r:       float = 0.12    # m  — hard crash
    danger_r:          float = 0.25    # m  — soft proximity zone

    # exploration
    explore_cell_m:    float = 0.5     # m  — grid cell side length

    # uncertainty
    conf_thresh:       float = 0.30    # confidence ∈ [0,1]; below = penalise
    occ_unknown_scale: float = 0.20    # fraction of w_uncertainty for occ term


# ── per-step output bundle ────────────────────────────────────────────────────

@dataclass
class RewardInfo:
    """
    Per-step breakdown of the reward signal.

    total       : sum of all components (use this as the RL reward)
    approach    : goal-approach dense component (can be negative)
    goal        : sparse goal bonus (0 unless success this step)
    collision   : sparse collision penalty (0 unless crash this step)
    obstacle    : obstacle-proximity dense penalty (≤ 0)
    explore     : exploration bonus (≥ 0)
    uncertainty : perception + occupancy uncertainty penalty (≤ 0)
    time        : per-step time penalty (< 0)
    terminated  : True if the episode ends this step (success or collision)
    success     : True if the goal was reached
    collision_flag : True if a collision was detected
    new_cell    : True if the robot entered a new grid cell this step
    n_visited   : total distinct cells visited so far this episode
    """
    total:          float
    approach:       float
    goal:           float
    collision:      float
    obstacle:       float
    explore:        float
    uncertainty:    float
    time:           float
    terminated:     bool
    success:        bool
    collision_flag: bool
    new_cell:       bool
    n_visited:      int

    def __str__(self) -> str:
        tag = "SUCCESS" if self.success else ("CRASH" if self.collision_flag else "")
        parts = [
            f"r={self.total:+.4f}",
            f"app={self.approach:+.4f}",
            f"obst={self.obstacle:+.4f}",
            f"exp={self.explore:+.4f}",
            f"unc={self.uncertainty:+.4f}",
            f"t={self.time:+.4f}",
        ]
        if self.goal != 0.0:
            parts.append(f"goal={self.goal:+.4f}")
        if self.collision != 0.0:
            parts.append(f"coll={self.collision:+.4f}")
        if tag:
            parts.append(f"[{tag}]")
        return " | ".join(parts)


# ── reward function ───────────────────────────────────────────────────────────

class NavRewardFunction:
    """
    Stateful reward function for point-navigation.

    Maintains per-episode state:
      • set of visited exploration cells (reset each episode)
      • previous step's velocity (for smoothness penalty, future extension)

    Parameters
    ----------
    cfg : RewardConfig
    """

    def __init__(self, cfg: RewardConfig = RewardConfig()) -> None:
        self._cfg          = cfg
        self._visited:     set[tuple[int, int]] = set()
        self._n_visited:   int = 0

    @property
    def cfg(self) -> RewardConfig:
        return self._cfg

    @property
    def n_visited_cells(self) -> int:
        """Total distinct exploration cells visited so far this episode."""
        return self._n_visited

    # ── episode lifecycle ──────────────────────────────────────────────────────

    def reset(self, robot_xy: np.ndarray) -> None:
        """
        Clear per-episode state.  Call once at the start of each episode,
        after the robot has been placed at its spawn position.

        Parameters
        ----------
        robot_xy : (2,) world-frame (x, y) of the robot base
        """
        self._visited.clear()
        self._n_visited = 0
        self._mark_cell(float(robot_xy[0]), float(robot_xy[1]))

    # ── per-step computation ───────────────────────────────────────────────────

    def step(
        self,
        robot_xy:    np.ndarray,
        d_prev:      float,
        d_curr:      float,
        d_lidar_min: float,
        occ_grid:    np.ndarray,
        perception:  Optional["PerceptionInput"] = None,  # noqa: F821
    ) -> RewardInfo:
        """
        Compute the full reward for one environment step.

        Parameters
        ----------
        robot_xy    : (2,) current robot world-frame position (x, y)
        d_prev      : distance to goal at the PREVIOUS step (m)
        d_curr      : distance to goal at the CURRENT step (m)
        d_lidar_min : minimum lidar range in the forward arc (m)
        occ_grid    : flat occupancy-grid observation, shape (grid_n²,)
                      values: 0.0=free, 1.0=occupied, 0.5=unknown
        perception  : PerceptionInput bundle or None if detector inactive

        Returns
        -------
        RewardInfo with total reward and all component breakdowns
        """
        cfg = self._cfg

        # ── 1. approach ───────────────────────────────────────────────────────
        r_approach = cfg.approach * (d_prev - d_curr)

        # ── 2. goal reached ───────────────────────────────────────────────────
        success   = d_curr < cfg.goal_radius
        r_goal    = cfg.goal if success else 0.0

        # ── 3. collision ──────────────────────────────────────────────────────
        crash     = d_lidar_min < cfg.collision_r
        r_coll    = -cfg.collision if crash else 0.0

        # ── 4. obstacle proximity ─────────────────────────────────────────────
        r_obst = 0.0
        if d_lidar_min < cfg.danger_r:
            proximity = 1.0 - d_lidar_min / cfg.danger_r
            r_obst    = -cfg.obstacle * float(np.clip(proximity, 0.0, 1.0))

        # ── 5. exploration ────────────────────────────────────────────────────
        new_cell  = self._mark_cell(float(robot_xy[0]), float(robot_xy[1]))
        r_explore = cfg.explore if new_cell else 0.0

        # ── 6. uncertainty ────────────────────────────────────────────────────
        r_unc = self._uncertainty_penalty(occ_grid, perception)

        # ── 7. time ───────────────────────────────────────────────────────────
        r_time = -cfg.time_step

        total = r_approach + r_goal + r_coll + r_obst + r_explore + r_unc + r_time

        return RewardInfo(
            total          = float(total),
            approach       = float(r_approach),
            goal           = float(r_goal),
            collision      = float(r_coll),
            obstacle       = float(r_obst),
            explore        = float(r_explore),
            uncertainty    = float(r_unc),
            time           = float(r_time),
            terminated     = success or crash,
            success        = success,
            collision_flag = crash,
            new_cell       = new_cell,
            n_visited      = self._n_visited,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _mark_cell(self, x: float, y: float) -> bool:
        """
        Discretise (x, y) into an exploration cell and record the visit.

        Returns True if this is the first visit to the cell this episode.
        """
        cs   = self._cfg.explore_cell_m
        cell = (int(math.floor(x / cs)), int(math.floor(y / cs)))
        if cell not in self._visited:
            self._visited.add(cell)
            self._n_visited += 1
            return True
        return False

    def _uncertainty_penalty(
        self,
        occ_grid:   np.ndarray,
        perception: Optional[object],
    ) -> float:
        """
        Compute the combined uncertainty penalty for one step.

        Sub-signals (both scaled by cfg.uncertainty):

        a) Perception confidence shortfall:
             −w × max(0, conf_thresh − confidence)   if perception is active

        b) Occupancy unknown fraction:
             −w × unknown_frac × occ_unknown_scale
        """
        cfg  = self._cfg
        w    = cfg.uncertainty
        r    = 0.0

        # a) Perceptual uncertainty
        if perception is not None:
            conf = float(getattr(perception, "confidence", 0.0))
            shortfall = max(0.0, cfg.conf_thresh - conf)
            r -= w * shortfall

        # b) Occupancy-map unknown fraction
        if occ_grid is not None and occ_grid.size > 0:
            unknown_frac = float(np.mean(occ_grid == 0.5))
            r -= w * unknown_frac * cfg.occ_unknown_scale

        return r


# ── convenience factory ───────────────────────────────────────────────────────

def make_reward_function(
    approach:       float = 2.0,
    goal:           float = 10.0,
    collision:      float = 5.0,
    obstacle:       float = 0.5,
    explore:        float = 0.10,
    uncertainty:    float = 0.05,
    time_step:      float = 0.01,
    goal_radius:    float = 0.25,
    collision_r:    float = 0.12,
    danger_r:       float = 0.25,
) -> NavRewardFunction:
    """Convenience constructor accepting flat keyword arguments."""
    cfg = RewardConfig(
        approach    = approach,
        goal        = goal,
        collision   = collision,
        obstacle    = obstacle,
        explore     = explore,
        uncertainty = uncertainty,
        time_step   = time_step,
        goal_radius = goal_radius,
        collision_r = collision_r,
        danger_r    = danger_r,
    )
    return NavRewardFunction(cfg)
