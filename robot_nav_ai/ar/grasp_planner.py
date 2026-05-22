"""
ar/grasp_planner.py — Grasp pose generation (Block 3.3).

Given an object's 3D position and its approach vector, generates:

    Pre-grasp pose — gripper hovers 15 cm along the approach path
                     before reaching the object.  The arm moves here
                     first so it is aligned before descending/advancing.

    Grasp pose     — gripper is 2 cm from the object surface.
                     Fingertips are at the object; closing the gripper
                     from here captures the object.

Why offset backwards along the approach vector
----------------------------------------------
The approach vector points in the direction the gripper TRAVELS to reach
the object (e.g. downward [0,1,0] for a table-top grasp).  "Hover above"
means the gripper starts 15 cm in the *opposite* direction:

    pre_grasp = object_xyz - 0.15 * approach_vec

    Top-down  [0,1,0]:  pre_grasp is 15 cm above  (−Y in camera frame)
    Horizontal [0,0,1]:  pre_grasp is 15 cm in front (−Z in camera frame)

Gripper orientation
-------------------
The gripper must arrive aligned with the surface.  We build a right-handed
rotation matrix R whose third column (the gripper's local Z-axis) points
along the approach vector.  The other two axes are chosen by Gram-Schmidt
against a reference "up" hint that is not parallel to the approach:

    z_grip = approach_vec           (grasp axis)
    x_grip = normalise(up × z_grip) (sweep axis)
    y_grip = z_grip × x_grip        (side axis)

    R = [x_grip | y_grip | z_grip]  (column-wise, 3×3 SO(3) matrix)

Usage
-----
    from ar.grasp_planner import GraspPlanner
    from ar.grasp_pose    import estimate_approach

    approach = estimate_approach(depth_map, u, v, intrinsics)
    planner  = GraspPlanner()
    pose     = planner.plan(object_xyz_base, approach)

    # pose.pre_grasp_xyz  — move arm here first
    # pose.grasp_xyz      — then here to close gripper
    # pose.gripper_R      — orientation (3×3 SO(3))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ar.grasp_pose import ApproachType, GraspApproach


# ── defaults ──────────────────────────────────────────────────────────────────

PRE_GRASP_OFFSET_M: float = 0.15   # 15 cm hover before object
GRASP_OFFSET_M:     float = 0.02   # 2 cm — fingertips at object surface


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class GraspPose:
    """
    Complete grasp description in the robot's coordinate frame.

    Attributes
    ----------
    pre_grasp_xyz   : (3,) position the arm moves to first (hover point)
    grasp_xyz       : (3,) position where gripper closes on the object
    gripper_R       : (3,3) SO(3) rotation matrix; column 2 (Z-axis) is
                      the approach direction
    approach_type   : ApproachType — TOP_DOWN, HORIZONTAL, or LATERAL
    object_xyz      : (3,) object centre position (input, preserved)
    """
    pre_grasp_xyz:  np.ndarray     # (3,)
    grasp_xyz:      np.ndarray     # (3,)
    gripper_R:      np.ndarray     # (3,3)
    approach_type:  ApproachType
    object_xyz:     np.ndarray     # (3,)

    def __str__(self) -> str:
        pg = self.pre_grasp_xyz
        g  = self.grasp_xyz
        return (
            f"GraspPose [{self.approach_type.value}]\n"
            f"  pre_grasp : ({pg[0]:+.3f}, {pg[1]:+.3f}, {pg[2]:+.3f}) m\n"
            f"  grasp     : ({g[0]:+.3f},  {g[1]:+.3f},  {g[2]:+.3f}) m\n"
            f"  gripper_R :\n{np.round(self.gripper_R, 3)}"
        )


# ── orientation builder ───────────────────────────────────────────────────────

# Reference "up" hints tried in order until one is not parallel to approach.
# Camera frame: X=right, Y=down, Z=forward.
_UP_HINTS = [
    np.array([0.0, -1.0,  0.0]),   # world up  (−Y camera)
    np.array([0.0,  0.0,  1.0]),   # forward   (+Z camera)
    np.array([1.0,  0.0,  0.0]),   # right     (+X camera)
]


def _orientation_from_approach(approach_vec: np.ndarray) -> np.ndarray:
    """
    Build a 3×3 SO(3) rotation matrix whose Z-axis aligns with approach_vec.

    Uses Gram-Schmidt with a fallback up-hint sequence to guarantee the
    result is non-degenerate for any approach direction.

    Returns R such that R[:,2] ≈ approach_vec (normalised).
    """
    z = np.asarray(approach_vec, dtype=np.float64)
    z = z / np.linalg.norm(z)

    x = np.zeros(3)
    for hint in _UP_HINTS:
        x = np.cross(hint, z)
        if np.linalg.norm(x) > 0.1:   # not parallel
            break
    x = x / np.linalg.norm(x)

    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    return np.column_stack([x, y, z])   # columns: [x_grip, y_grip, z_grip]


# ── planner ───────────────────────────────────────────────────────────────────

class GraspPlanner:
    """
    Generates pre-grasp and grasp poses from an object position and approach.

    Parameters
    ----------
    pre_grasp_offset : metres the arm hovers before the object (default 0.15)
    grasp_offset     : metres from object surface where gripper closes (default 0.02)
    """

    def __init__(
        self,
        pre_grasp_offset: float = PRE_GRASP_OFFSET_M,
        grasp_offset:     float = GRASP_OFFSET_M,
    ) -> None:
        self.pre_grasp_offset = pre_grasp_offset
        self.grasp_offset     = grasp_offset

    def plan(
        self,
        object_xyz: Tuple[float, float, float] | np.ndarray,
        approach:   GraspApproach,
    ) -> GraspPose:
        """
        Generate a GraspPose for the given object and approach.

        Parameters
        ----------
        object_xyz : (3,) object centre in the robot base frame (metres)
        approach   : GraspApproach from estimate_approach()

        Returns
        -------
        GraspPose with pre_grasp_xyz, grasp_xyz, gripper_R
        """
        obj = np.asarray(object_xyz, dtype=np.float64)
        a   = np.asarray(approach.approach_vec, dtype=np.float64)
        a   = a / np.linalg.norm(a)   # ensure unit length

        # Gripper starts behind the object along the approach path, then
        # moves forward (in +approach direction) to complete the grasp.
        pre_grasp = obj - self.pre_grasp_offset * a
        grasp     = obj - self.grasp_offset     * a

        R = _orientation_from_approach(a)

        return GraspPose(
            pre_grasp_xyz = pre_grasp,
            grasp_xyz     = grasp,
            gripper_R     = R,
            approach_type = approach.approach_type,
            object_xyz    = obj.copy(),
        )
