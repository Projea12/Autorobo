"""
ar/grasp_reporter.py — Grasp success/failure reporter (Blocks 6.3 & 6.4).

Block 6.3 — Success/failure reporter
--------------------------------------
GraspReporter.report() classifies the outcome of a GraspExecutor run,
prints a console line, and optionally draws a coloured overlay on the
current camera frame.

Failure reason classification
------------------------------
    IK_FAILED      — IK did not converge for pre-grasp or grasp target
    OUT_OF_REACH   — target outside the robot's workspace
    NO_CONTACT     — gripper closed but driver joints did not indicate contact
    OBJECT_MOVED   — contact detected but object did not rise after lift
    (None for success)

Block 6.4 — Result handoff to Phase 8
---------------------------------------
GraspResult is the dataclass that the Phase 8 recovery layer consumes.
It captures everything Phase 8 needs to decide on a retry or escalation.

Usage
-----
    from ar.grasp_reporter import GraspReporter, GraspResult, FailureReason

    reporter = GraspReporter()
    result   = reporter.report(exec_result, label="mug", frame=frame)
    # → prints "[grasp] SUCCESS — mug secured"  (or FAILED variant)
    # → result.success / result.failure_reason / result.attempt_duration_ms
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from ar.grasp_executor import ExecutionResult, GraspState


# ── failure reasons ────────────────────────────────────────────────────────────

class FailureReason(Enum):
    NO_CONTACT   = auto()   # gripper closed but no contact indicated
    OBJECT_MOVED = auto()   # contact detected but object did not rise
    IK_FAILED    = auto()   # IK did not converge
    OUT_OF_REACH = auto()   # target outside workspace


# ── Phase-8 handoff dataclass (Block 6.4) ─────────────────────────────────────

@dataclass
class GraspResult:
    """
    Unified grasp outcome for Phase 8 consumption.

    Attributes
    ----------
    success              : True if the object was grasped and lifted
    object_label         : YOLO label of the target object
    failure_reason       : FailureReason enum member, or None on success
    attempt_duration_ms  : wall-clock duration of the execution attempt
    """
    success:             bool
    object_label:        str
    failure_reason:      Optional[FailureReason]
    attempt_duration_ms: float

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED ({self.failure_reason.name if self.failure_reason else '?'})"
        return (
            f"GraspResult [{status}]  label={self.object_label!r}  "
            f"duration={self.attempt_duration_ms:.0f} ms"
        )


# ── overlay helpers ────────────────────────────────────────────────────────────

_GREEN = (0, 200, 0)
_RED   = (0, 0, 200)
_WHITE = (255, 255, 255)

_OVERLAY_HEIGHT = 48    # px — banner height at bottom of frame
_ALPHA          = 0.55  # overlay opacity


def _draw_banner(frame: np.ndarray, text: str, colour: tuple) -> None:
    """
    Draw a semi-transparent coloured banner at the bottom of frame in-place.

    Parameters
    ----------
    frame  : BGR ndarray (H, W, 3) — modified in-place
    text   : message to display
    colour : BGR colour for the banner background
    """
    try:
        import cv2
    except ImportError:
        return

    h, w = frame.shape[:2]
    y0   = max(0, h - _OVERLAY_HEIGHT)
    roi  = frame[y0:h, 0:w]

    overlay = roi.copy()
    overlay[:] = colour
    cv2.addWeighted(overlay, _ALPHA, roi, 1 - _ALPHA, 0, roi)
    frame[y0:h, 0:w] = roi

    font_scale = 0.65
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = (w - tw) // 2
    ty = y0 + (_OVERLAY_HEIGHT + th) // 2
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, _WHITE, thickness,
                cv2.LINE_AA)


# ── reporter ──────────────────────────────────────────────────────────────────

class GraspReporter:
    """
    Classifies the outcome of a GraspExecutor run, reports it to the console,
    and optionally annotates the live camera frame.

    Usage
    -----
        reporter = GraspReporter()
        result   = reporter.report(exec_result, label="mug", frame=frame)
    """

    def report(
        self,
        exec_result: ExecutionResult,
        label:       str,
        frame:       Optional[np.ndarray] = None,
    ) -> GraspResult:
        """
        Classify the result, print a status line, draw overlay, and return
        a GraspResult for Phase 8.

        Parameters
        ----------
        exec_result : ExecutionResult from GraspExecutor.execute()
        label       : target object label (e.g. "mug")
        frame       : optional BGR camera frame to annotate in-place

        Returns
        -------
        GraspResult
        """
        duration_ms = exec_result.elapsed_s * 1000.0

        if exec_result.success:
            grasp_result = GraspResult(
                success             = True,
                object_label        = label,
                failure_reason      = None,
                attempt_duration_ms = duration_ms,
            )
            msg = f"[grasp] SUCCESS — {label} secured"
            print(msg)
            if frame is not None:
                _draw_banner(frame, msg, _GREEN)

        else:
            reason = self._classify(exec_result)
            grasp_result = GraspResult(
                success             = False,
                object_label        = label,
                failure_reason      = reason,
                attempt_duration_ms = duration_ms,
            )
            msg = f"[grasp] FAILED — reason: {reason.name.lower().replace('_', ' ')}"
            print(msg)
            if frame is not None:
                _draw_banner(frame, msg, _RED)

        print(grasp_result)
        return grasp_result

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _classify(exec_result: ExecutionResult) -> FailureReason:
        """
        Derive the most specific FailureReason from an ExecutionResult.

        Priority order (most specific first):
          1. OUT_OF_REACH — IK fail due to workspace limit
          2. IK_FAILED    — generic IK convergence failure
          3. NO_CONTACT   — gripper closed but object not detected
          4. OBJECT_MOVED — contact but no rise after lift
        """
        reason_text = exec_result.fail_reason.lower()

        if "out of reach" in reason_text or "unreachable" in reason_text or "reachability" in reason_text:
            return FailureReason.OUT_OF_REACH

        if "ik failed" in reason_text or "ik did not" in reason_text:
            return FailureReason.IK_FAILED

        # Executor reached DONE — examine gripper and lift results
        gr = exec_result.gripper_result
        lr = exec_result.lift_result

        if gr is not None and not gr.object_detected:
            return FailureReason.NO_CONTACT

        if lr is not None and not lr.success:
            return FailureReason.OBJECT_MOVED

        # Generic IK / timeout failures that don't match above
        if "timed out" in reason_text or "timeout" in reason_text:
            return FailureReason.IK_FAILED

        # Catch-all
        return FailureReason.IK_FAILED
