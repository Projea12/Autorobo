"""
env/grasp_planner.py — Grasp pose planner for tabletop objects.

Computes ranked 6-DOF grasp candidates from an object point cloud
and/or segmentation mask, filtered by arm workspace reachability.

Pipeline
────────
  1. Point cloud analysis
     ─────────────────────
     Fit PCA to the object point cloud → three principal axes.
     The longest axis is the object's main body axis (e.g. bottle standing
     up → vertical; lying down → horizontal).

  2. Candidate generation
     ─────────────────────
     Three candidate families are always generated:
       a) TOP-DOWN  — approach from directly above the object centroid.
          Most reliable for cylindrical objects standing upright.
       b) SIDE-AXIS — approach along the object's horizontal principal axis.
          Best for elongated objects (bottles lying flat, tools).
       c) DIAGONAL  — 45° approach combining top-down and side-axis.
          Fallback for irregular shapes.

  3. Scoring
     ────────
     Each candidate is scored on four criteria:
       a) Reachability  : EE position inside arm's reachable sphere
       b) Approach angle: prefer approaches from above (avoid joint limits)
       c) Depth alignment: approach vector aligns with dominant point-cloud
          normal (reward approaching perpendicularly to object surface)
       d) Clearance     : prefer candidates where EE does not start inside
          the object bounding sphere

  4. Filtering
     ──────────
     Candidates outside the workspace sphere are discarded.
     Remaining candidates are ranked by score and returned top-k.

Usage
─────
    from env.grasp_planner import GraspPlanner, PlannerConfig
    from perception.depth_projector import ProjectionResult
    from perception.detector import Detection

    planner = GraspPlanner()
    candidates = planner.plan(
        obj_pos       = result.xyz,          # (3,) world frame
        point_cloud   = frame_cloud,         # (N, 3) or None
        mask          = detection.mask,      # (H, W) bool or None
        robot_pos     = robot_base_pos,      # (3,) world frame
        robot_quat    = robot_base_quat,     # (4,) wxyz
    )
    best = candidates[0]
    print(best.ee_pos, best.approach_vec, best.score)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from robot.workspace import DEFAULT_LIMITS, WorkspaceLimits, is_ee_reachable


# ── grasp candidate ───────────────────────────────────────────────────────────

@dataclass
class GraspCandidate:
    """
    A single 6-DOF grasp candidate.

    Fields
    ------
    ee_pos       : (3,) float32 — target end-effector position (world frame)
    approach_vec : (3,) float32 — unit vector pointing from EE toward object
                   (the direction the arm approaches from)
    score        : float — higher is better; used for ranking
    method       : "top_down", "side_axis", or "diagonal"
    reachable    : True if EE position is within arm workspace sphere
    """
    ee_pos:       np.ndarray   # (3,) float32
    approach_vec: np.ndarray   # (3,) float32
    score:        float
    method:       str
    reachable:    bool = True

    def __repr__(self) -> str:
        p = self.ee_pos.tolist()
        return (f"GraspCandidate(method={self.method!r}, score={self.score:.3f}, "
                f"pos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}], "
                f"{'reachable' if self.reachable else 'UNREACHABLE'})")


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlannerConfig:
    """
    Configuration for GraspPlanner.

    approach_dist   : standoff distance — how far above/beside the object
                      the EE is placed before closing the gripper (metres)
    top_k           : number of candidates to return (sorted best-first)
    min_points      : minimum point cloud points required for PCA axis fitting
    w_reachability  : weight for reachability score component
    w_approach_angle: weight for approach-angle score component
    w_depth_align   : weight for depth-alignment score component
    w_clearance     : weight for clearance score component
    diagonal_angles : list of elevation angles [rad] for diagonal candidates
    n_side_rotations: number of rotations around vertical axis for side candidates
    """
    approach_dist:    float = 0.12     # 12 cm standoff
    top_k:            int   = 5
    min_points:       int   = 10
    w_reachability:   float = 0.5
    w_approach_angle: float = 0.3
    w_depth_align:    float = 0.1
    w_clearance:      float = 0.1
    diagonal_angles:  tuple = (math.pi / 4,)     # 45°
    n_side_rotations: int   = 4                   # 0°, 90°, 180°, 270°


# ── planner ───────────────────────────────────────────────────────────────────

class GraspPlanner:
    """
    Generates and ranks 6-DOF grasp candidates for tabletop pick tasks.

    Parameters
    ----------
    cfg     : PlannerConfig
    limits  : WorkspaceLimits — arm reach bounds for reachability filtering
    """

    def __init__(
        self,
        cfg:    PlannerConfig    = PlannerConfig(),
        limits: WorkspaceLimits  = DEFAULT_LIMITS,
    ) -> None:
        self.cfg    = cfg
        self.limits = limits

    # ── public API ────────────────────────────────────────────────────────────

    def plan(
        self,
        obj_pos:     np.ndarray,
        robot_pos:   np.ndarray,
        robot_quat:  np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
        mask:        Optional[np.ndarray] = None,
    ) -> list[GraspCandidate]:
        """
        Compute and rank grasp candidates for one detected object.

        Parameters
        ----------
        obj_pos     : (3,) object centroid in world frame
        robot_pos   : (3,) robot base position in world frame
        robot_quat  : (4,) robot base orientation quaternion (wxyz)
        point_cloud : (N, 3) object point cloud in world frame, or None
        mask        : (H, W) bool segmentation mask, or None (unused here,
                      reserved for future surface-normal estimation)

        Returns
        -------
        List of GraspCandidate sorted by score descending, length ≤ top_k.
        Unreachable candidates are excluded.
        """
        obj_pos    = np.asarray(obj_pos,   dtype=np.float64)
        robot_pos  = np.asarray(robot_pos, dtype=np.float64)
        robot_quat = np.asarray(robot_quat, dtype=np.float64)

        # Principal axes from point cloud (falls back to world axes if too few pts)
        axes = self._principal_axes(point_cloud, obj_pos)

        candidates: list[GraspCandidate] = []

        # a) top-down family
        candidates.extend(self._top_down_candidates(obj_pos))

        # b) side-axis family (along horizontal principal axis)
        candidates.extend(self._side_axis_candidates(obj_pos, axes))

        # c) diagonal family
        candidates.extend(self._diagonal_candidates(obj_pos, axes))

        # Score and filter
        scored: list[GraspCandidate] = []
        for c in candidates:
            reachable = is_ee_reachable(c.ee_pos, robot_pos, robot_quat, self.limits)
            if not reachable:
                continue
            score = self._score(c, obj_pos, point_cloud)
            scored.append(GraspCandidate(
                ee_pos       = c.ee_pos.astype(np.float32),
                approach_vec = c.approach_vec.astype(np.float32),
                score        = score,
                method       = c.method,
                reachable    = True,
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: self.cfg.top_k]

    def plan_from_detection(
        self,
        obj_pos:    np.ndarray,
        robot_pos:  np.ndarray,
        robot_quat: np.ndarray,
        projection=None,
        detection=None,
    ) -> list[GraspCandidate]:
        """
        Convenience wrapper accepting ProjectionResult and Detection objects.

        Uses detection.mask and builds a partial point cloud from the
        projection's xyz + std if no full point cloud is available.
        """
        mask        = getattr(detection,  "mask", None)
        point_cloud = None

        if projection is not None and hasattr(projection, "xyz"):
            # Approximate 1-point cloud from projection — used for axis fallback
            point_cloud = projection.xyz.reshape(1, 3).astype(np.float64)

        return self.plan(
            obj_pos     = obj_pos,
            robot_pos   = robot_pos,
            robot_quat  = robot_quat,
            point_cloud = point_cloud,
            mask        = mask,
        )

    def best(
        self,
        obj_pos:     np.ndarray,
        robot_pos:   np.ndarray,
        robot_quat:  np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
        mask:        Optional[np.ndarray] = None,
    ) -> Optional[GraspCandidate]:
        """Return the single highest-scoring reachable candidate, or None."""
        results = self.plan(obj_pos, robot_pos, robot_quat, point_cloud, mask)
        return results[0] if results else None

    # ── candidate generators ──────────────────────────────────────────────────

    def _top_down_candidates(
        self, obj_pos: np.ndarray
    ) -> list[GraspCandidate]:
        """EE directly above the object, approaching downward."""
        approach = np.array([0.0, 0.0, -1.0])   # pointing down toward object
        ee_pos   = obj_pos + np.array([0.0, 0.0, self.cfg.approach_dist])
        return [GraspCandidate(
            ee_pos       = ee_pos,
            approach_vec = approach,
            score        = 0.0,
            method       = "top_down",
        )]

    def _side_axis_candidates(
        self, obj_pos: np.ndarray, axes: np.ndarray
    ) -> list[GraspCandidate]:
        """Candidates approaching along the object's horizontal principal axis."""
        candidates = []
        horiz_axis = axes[0].copy()   # longest principal axis
        horiz_axis[2] = 0.0           # project to horizontal plane
        norm = np.linalg.norm(horiz_axis)
        if norm < 1e-6:
            horiz_axis = np.array([1.0, 0.0, 0.0])
        else:
            horiz_axis /= norm

        n = self.cfg.n_side_rotations
        for i in range(n):
            angle = 2.0 * math.pi * i / n
            c, s  = math.cos(angle), math.sin(angle)
            # Rotate horiz_axis around Z by angle
            dir_vec = np.array([
                c * horiz_axis[0] - s * horiz_axis[1],
                s * horiz_axis[0] + c * horiz_axis[1],
                0.0,
            ])
            approach = -dir_vec                          # pointing toward object
            ee_pos   = obj_pos + dir_vec * self.cfg.approach_dist
            candidates.append(GraspCandidate(
                ee_pos       = ee_pos,
                approach_vec = approach,
                score        = 0.0,
                method       = "side_axis",
            ))
        return candidates

    def _diagonal_candidates(
        self, obj_pos: np.ndarray, axes: np.ndarray
    ) -> list[GraspCandidate]:
        """45° diagonal approach candidates (top-down + side combined)."""
        candidates = []
        horiz_axis = axes[0].copy()
        horiz_axis[2] = 0.0
        norm = np.linalg.norm(horiz_axis)
        if norm < 1e-6:
            horiz_axis = np.array([1.0, 0.0, 0.0])
        else:
            horiz_axis /= norm

        for elev in self.cfg.diagonal_angles:
            for sign in (+1.0, -1.0):
                dir_h = sign * horiz_axis
                dir_v = np.array([0.0, 0.0, 1.0])
                # Blend horizontal and vertical
                approach_raw = -(math.cos(elev) * dir_v + math.sin(elev) * dir_h)
                norm2 = np.linalg.norm(approach_raw)
                approach = approach_raw / norm2 if norm2 > 1e-6 else approach_raw
                ee_pos   = obj_pos - approach * self.cfg.approach_dist
                candidates.append(GraspCandidate(
                    ee_pos       = ee_pos,
                    approach_vec = approach,
                    score        = 0.0,
                    method       = "diagonal",
                ))
        return candidates

    # ── scoring ───────────────────────────────────────────────────────────────

    def _score(
        self,
        candidate:   GraspCandidate,
        obj_pos:     np.ndarray,
        point_cloud: Optional[np.ndarray],
    ) -> float:
        """Weighted sum of four scoring criteria, in [0, 1]."""
        cfg = self.cfg

        # a) Reachability: 1.0 (already filtered to reachable set)
        s_reach = 1.0

        # b) Approach angle: prefer top-down (approach_vec[2] close to -1)
        down  = np.array([0.0, 0.0, -1.0])
        app   = np.asarray(candidate.approach_vec, dtype=np.float64)
        s_ang = float(max(0.0, np.dot(app / (np.linalg.norm(app) + 1e-8), down)))

        # c) Depth alignment: align approach with dominant point-cloud normal
        s_depth = 0.5   # neutral if no point cloud
        if point_cloud is not None and len(point_cloud) >= cfg.min_points:
            cloud_normal = self._dominant_normal(point_cloud)
            if cloud_normal is not None:
                s_depth = float(abs(np.dot(
                    app / (np.linalg.norm(app) + 1e-8),
                    cloud_normal,
                )))

        # d) Clearance: EE should not start inside the object bounding sphere
        obj_radius   = self._object_radius(point_cloud, obj_pos)
        ee_obj_dist  = float(np.linalg.norm(candidate.ee_pos - obj_pos))
        s_clear = float(np.clip(ee_obj_dist / (obj_radius + 1e-3), 0.0, 1.0))

        score = (
            cfg.w_reachability   * s_reach  +
            cfg.w_approach_angle * s_ang    +
            cfg.w_depth_align    * s_depth  +
            cfg.w_clearance      * s_clear
        )
        return float(score)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _principal_axes(
        self,
        point_cloud: Optional[np.ndarray],
        obj_pos:     np.ndarray,
    ) -> np.ndarray:
        """
        Return (3, 3) matrix whose rows are the principal axes of the point
        cloud, sorted by explained variance descending.

        Falls back to world axes [X, Y, Z] if cloud is None or too small.
        """
        if point_cloud is None or len(point_cloud) < self.cfg.min_points:
            return np.eye(3, dtype=np.float64)

        cloud = np.asarray(point_cloud, dtype=np.float64)
        centred = cloud - cloud.mean(axis=0)
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        return Vt   # rows are principal axes, sorted by variance

    def _dominant_normal(
        self, point_cloud: np.ndarray
    ) -> Optional[np.ndarray]:
        """Estimate dominant surface normal via PCA on point cloud."""
        if len(point_cloud) < self.cfg.min_points:
            return None
        cloud = np.asarray(point_cloud, dtype=np.float64)
        centred = cloud - cloud.mean(axis=0)
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        # Smallest singular value → normal of the plane of best fit
        normal = Vt[-1]
        norm   = np.linalg.norm(normal)
        return normal / norm if norm > 1e-8 else None

    def _object_radius(
        self,
        point_cloud: Optional[np.ndarray],
        obj_pos:     np.ndarray,
    ) -> float:
        """Estimate object bounding sphere radius from point cloud."""
        if point_cloud is None or len(point_cloud) < 2:
            return 0.05   # default 5 cm for unknown object
        cloud = np.asarray(point_cloud, dtype=np.float64)
        dists = np.linalg.norm(cloud - obj_pos, axis=1)
        return float(np.percentile(dists, 90))   # robust to outliers

    def __repr__(self) -> str:
        return (f"GraspPlanner(approach_dist={self.cfg.approach_dist}m, "
                f"top_k={self.cfg.top_k})")
