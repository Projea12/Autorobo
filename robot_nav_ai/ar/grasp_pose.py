"""
ar/grasp_pose.py — Grasp pose estimation (Block 3).

Block 3.2 — Approach vector
----------------------------
The approach vector is the direction the gripper travels to reach the object.
It must arrive perpendicular to the surface so the fingers land flat on it:

    approach = -n_hat      (opposite of the outward surface normal)

Camera frame: X=right, Y=down, Z=forward.

Surface → normal → approach
    Horizontal table  : n=[0,-1,0]  → approach=[0,+1,0]  (downward)
    Vertical shelf    : n=[0, 0,-1] → approach=[0, 0,+1]  (forward)
    Tilted surface    : intermediate vector

Approach classification
-----------------------
Determined by which component of the approach vector dominates:

    |approach_y| largest → TOP_DOWN   (arm comes from above)
    |approach_z| largest → HORIZONTAL (arm comes from the front)
    |approach_x| largest → LATERAL    (arm comes from the side)

A confidence score (0–1) measures how unambiguous the classification is:
    confidence = max_component / sum_of_all_components
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from ar.surface_normal import estimate_normal


# ── approach type ─────────────────────────────────────────────────────────────

class ApproachType(Enum):
    TOP_DOWN   = "top_down"    # arm descends from above  (table, floor)
    HORIZONTAL = "horizontal"  # arm moves forward        (shelf, wall)
    LATERAL    = "lateral"     # arm moves sideways       (side of object)


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GraspApproach:
    """
    Full approach description for one detected object.

    Attributes
    ----------
    n_hat        : surface normal at the object centroid (unit vector, camera frame)
    approach_vec : -n_hat (direction the gripper travels, camera frame)
    approach_type: classified approach direction
    confidence   : 0–1, how dominant the primary approach axis is
    """
    n_hat:        np.ndarray    # (3,) unit vector
    approach_vec: np.ndarray    # (3,) unit vector = -n_hat
    approach_type: ApproachType
    confidence:   float

    def __str__(self) -> str:
        a = self.approach_vec
        return (f"{self.approach_type.value}  "
                f"approach=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f})  "
                f"conf={self.confidence:.2f}")


# ── core functions ────────────────────────────────────────────────────────────

def approach_vector(n_hat: np.ndarray) -> np.ndarray:
    """
    Compute the gripper approach vector from the surface normal.

    The robot approaches perpendicular to the surface, travelling in the
    direction opposite to the outward normal:

        approach = -n_hat

    Parameters
    ----------
    n_hat : (3,) unit surface normal in camera frame

    Returns
    -------
    (3,) unit approach vector in camera frame
    """
    return -np.asarray(n_hat, dtype=np.float64)


def classify_approach(approach_vec: np.ndarray) -> Tuple[ApproachType, float]:
    """
    Classify the approach direction and return a confidence score.

    Classification is based on the dominant component of the approach vector:
        |approach_y| largest → TOP_DOWN
        |approach_z| largest → HORIZONTAL
        |approach_x| largest → LATERAL

    Confidence = max_component_magnitude / sum_of_all_magnitudes.
    A perfectly top-down approach ([0,1,0]) scores 1.0.
    A 45° approach scores ~0.5.

    Parameters
    ----------
    approach_vec : (3,) approach vector (does not need to be unit length)

    Returns
    -------
    (ApproachType, confidence)
    """
    a = np.abs(np.asarray(approach_vec, dtype=np.float64))
    total = a.sum()
    if total < 1e-9:
        return ApproachType.TOP_DOWN, 0.0

    dominant = int(np.argmax(a))
    confidence = float(a[dominant] / total)

    approach_type = [ApproachType.LATERAL,
                     ApproachType.TOP_DOWN,
                     ApproachType.HORIZONTAL][dominant]
    return approach_type, confidence


def estimate_approach(
    depth_map: np.ndarray,
    u: int,
    v: int,
    intrinsics,
) -> GraspApproach:
    """
    Full pipeline: depth map → surface normal → approach vector → classification.

    Parameters
    ----------
    depth_map  : H×W float32 metric depth map (metres)
    u, v       : object centroid pixel coordinates
    intrinsics : CameraIntrinsics (.fx .fy .cx .cy)

    Returns
    -------
    GraspApproach with n_hat, approach_vec, approach_type, confidence
    """
    n_hat       = estimate_normal(depth_map, u, v, intrinsics)
    app_vec     = approach_vector(n_hat)
    app_type, conf = classify_approach(app_vec)
    return GraspApproach(
        n_hat        = n_hat,
        approach_vec = app_vec,
        approach_type= app_type,
        confidence   = conf,
    )
