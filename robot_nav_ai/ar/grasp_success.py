"""
ar/grasp_success.py — Object re-detection and grasp success check (Block 6.2).

Two complementary success signals
-----------------------------------
1. 3D position check  (primary, when depth/localiser available)
   Compare xyz_before and xyz_after from the Localiser.
   Success if the object moved upward (delta_z > MIN_LIFT_Z = 0.05 m).

2. Pixel centroid check  (fallback, from YOLO detections only)
   In image coordinates, Y increases downward (v=0 at top).
   A lifted object moves toward the top of the frame → smaller v.
   Success if centroid_v_after < centroid_v_before - MIN_PIXEL_RISE.

Why two checks
--------------
The 3D check is definitive but requires the depth estimator to be running.
The pixel check works on any frame with YOLO running — useful as a
lightweight sanity check or when depth is unavailable.

Distinguishing success vs failure
-----------------------------------
Success: object grasped and lifted — its 3D position (or pixel centroid)
         moved upward by at least 5 cm (or MIN_PIXEL_RISE pixels).

Failure: arm lifted but object stayed — the object is still detected at
         the same position it was before the grasp.  This happens when
         the gripper missed, slipped, or the object was not reachable.

Usage
-----
    from ar.grasp_success import GraspSuccessChecker, GraspSuccessResult

    checker = GraspSuccessChecker()

    # 3D check (from Localiser)
    result = checker.check_3d(xyz_before=[0,0.5,0.5], xyz_after=[0,0.5,0.6])
    # → success=True  delta_z=0.100

    # Pixel check (from YOLO detections)
    result = checker.check_detections(
        dets_before=[det_mug_before],
        dets_after=[det_mug_after],
        label="mug",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────

MIN_LIFT_Z:       float = 0.05    # minimum Δz to call grasp successful (5 cm)
MIN_PIXEL_RISE:   int   = 10      # minimum centroid rise in pixels (upward)


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class GraspSuccessResult:
    """
    Unified grasp success verdict.

    Attributes
    ----------
    success        : True if the object was detected at a higher position
    method         : "3d" | "pixel" | "none" — which check was used
    delta_z        : vertical displacement in metres (3D check) or None
    pixel_rise     : upward pixel displacement (pixel check) or None
    xyz_before     : object position before lift (3D check) or None
    xyz_after      : object position after lift  (3D check) or None
    label_matched  : True if target label was found in both detection sets
    reason         : human-readable explanation
    """
    success:      bool
    method:       str
    delta_z:      Optional[float]
    pixel_rise:   Optional[int]
    xyz_before:   Optional[np.ndarray]
    xyz_after:    Optional[np.ndarray]
    label_matched: bool
    reason:       str

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILURE"
        parts  = [f"GraspSuccessResult [{status}]  method={self.method}"]
        if self.delta_z is not None:
            parts.append(f"  Δz     = {self.delta_z*100:.1f} cm  (need >{MIN_LIFT_Z*100:.0f} cm)")
        if self.pixel_rise is not None:
            parts.append(f"  Δpx    = {self.pixel_rise} px  (need >{MIN_PIXEL_RISE} px)")
        parts.append(f"  reason = {self.reason}")
        return "\n".join(parts)


# ── checker ───────────────────────────────────────────────────────────────────

class GraspSuccessChecker:
    """
    Checks whether the object was successfully lifted after a grasp.

    Parameters
    ----------
    min_lift_z     : minimum vertical rise (metres) for 3D success
    min_pixel_rise : minimum centroid rise (pixels, upward in image) for 2D
    """

    def __init__(
        self,
        min_lift_z:     float = MIN_LIFT_Z,
        min_pixel_rise: int   = MIN_PIXEL_RISE,
    ) -> None:
        self.min_lift_z     = min_lift_z
        self.min_pixel_rise = min_pixel_rise

    # ── 3D check (primary) ────────────────────────────────────────────────────

    def check_3d(
        self,
        xyz_before: Tuple[float, float, float] | np.ndarray,
        xyz_after:  Tuple[float, float, float] | np.ndarray,
    ) -> GraspSuccessResult:
        """
        Compare 3D object positions before and after lift.

        Parameters
        ----------
        xyz_before : (3,) object position before grasp, in base/world frame
        xyz_after  : (3,) object position after lift

        Returns
        -------
        GraspSuccessResult with method="3d"
        """
        b = np.asarray(xyz_before, dtype=float)
        a = np.asarray(xyz_after,  dtype=float)
        delta_z = float(a[2] - b[2])
        success = delta_z > self.min_lift_z

        return GraspSuccessResult(
            success      = success,
            method       = "3d",
            delta_z      = delta_z,
            pixel_rise   = None,
            xyz_before   = b,
            xyz_after    = a,
            label_matched = True,
            reason       = (
                f"Δz = {delta_z*100:.1f} cm  "
                f"({'≥' if success else '<'} {self.min_lift_z*100:.0f} cm threshold)"
            ),
        )

    # ── pixel check (fallback) ────────────────────────────────────────────────

    def check_detections(
        self,
        dets_before: list,
        dets_after:  list,
        label:       str,
    ) -> GraspSuccessResult:
        """
        Compare YOLO centroid positions before and after lift.

        In image space, a lifted object has a SMALLER v (higher on screen).
        pixel_rise = v_before − v_after  (positive = moved up = success).

        Parameters
        ----------
        dets_before : list[Detection] from before the grasp
        dets_after  : list[Detection] from after the lift
        label       : target object label to match

        Returns
        -------
        GraspSuccessResult with method="pixel"
        """
        det_pre  = self._find_label(dets_before, label)
        det_post = self._find_label(dets_after,  label)

        if det_pre is None:
            return GraspSuccessResult(
                success=False, method="pixel", delta_z=None, pixel_rise=None,
                xyz_before=None, xyz_after=None, label_matched=False,
                reason=f"'{label}' not found in pre-grasp detections",
            )

        if det_post is None:
            # Object not detected after lift — could mean it was hidden in
            # the gripper (success) or the detection failed.  Treat as failure
            # to be conservative (require positive re-detection).
            return GraspSuccessResult(
                success=False, method="pixel", delta_z=None, pixel_rise=None,
                xyz_before=None, xyz_after=None, label_matched=False,
                reason=f"'{label}' not re-detected after lift — inconclusive",
            )

        v_before = det_pre.centroid_uv[1]
        v_after  = det_post.centroid_uv[1]
        rise     = v_before - v_after   # positive → moved UP in image
        success  = rise > self.min_pixel_rise

        return GraspSuccessResult(
            success       = success,
            method        = "pixel",
            delta_z       = None,
            pixel_rise    = rise,
            xyz_before    = None,
            xyz_after     = None,
            label_matched = True,
            reason        = (
                f"centroid v: {v_before} → {v_after}  "
                f"rise={rise} px  "
                f"({'≥' if success else '<'} {self.min_pixel_rise} px threshold)"
            ),
        )

    # ── combined check ────────────────────────────────────────────────────────

    def check(
        self,
        xyz_before:  Optional[Tuple] = None,
        xyz_after:   Optional[Tuple] = None,
        dets_before: Optional[list]  = None,
        dets_after:  Optional[list]  = None,
        label:       str             = "",
    ) -> GraspSuccessResult:
        """
        Run the best available check.

        Prefers 3D if xyz_before and xyz_after are provided.
        Falls back to pixel check if detections are provided.
        Returns a no-data failure if neither is available.
        """
        if xyz_before is not None and xyz_after is not None:
            return self.check_3d(xyz_before, xyz_after)
        if dets_before is not None and dets_after is not None and label:
            return self.check_detections(dets_before, dets_after, label)
        return GraspSuccessResult(
            success=False, method="none", delta_z=None, pixel_rise=None,
            xyz_before=None, xyz_after=None, label_matched=False,
            reason="No position data or detections provided",
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_label(detections: list, label: str):
        """Return the highest-confidence detection matching label, or None."""
        label = label.lower()
        best  = None
        best_conf = -1.0
        for d in detections:
            if label in d.label.lower() or d.label.lower() in label:
                if d.confidence > best_conf:
                    best      = d
                    best_conf = d.confidence
        return best
