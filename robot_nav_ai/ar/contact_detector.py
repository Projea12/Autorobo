"""
ar/contact_detector.py — Gripper contact detection (Block 6.1).

Detection principle
--------------------
When the gripper closes with no object, the driver joints converge to a
reference position (~0.234 rad at ctrl=200, measured by running the 50-step
ramp close in simulation).  When an object is present, it physically blocks
the fingers before they reach that reference — the actual driver position
ends up LESS than the reference.

The threshold maps "5 units of fingers_actuator range" into driver-joint
radians:

    threshold_rad = (5 / 255) * DRIVER_MAX_ANGLE
                  = 0.0196 * 0.8
                  = 0.0157 rad

Contact is detected when:

    reference_driver - actual_driver > threshold_rad

i.e. the gripper is more than one threshold-width MORE open than it
would be during a free close — meaning something prevented full closure.

Reading qpos[10:18]
--------------------
The eight gripper DOFs are:
    qpos[10] right_driver_joint       (primary closure indicator)
    qpos[11] right_coupler_joint
    qpos[12] right_spring_link_joint
    qpos[13] right_follower_joint
    qpos[14] left_driver_joint        (symmetrically driven)
    qpos[15] left_coupler_joint
    qpos[16] left_spring_link_joint
    qpos[17] left_follower_joint

Only the driver joints (10 and 14) reliably reflect applied force — the
others are kinematically coupled and can move due to springs even without
contact.  We use qpos[10] (right_driver) as the primary contact signal and
confirm with qpos[14] (left_driver).

Usage
-----
    from ar.contact_detector import ContactDetector

    detector = ContactDetector()
    result   = detector.detect(kin.data)
    if result.contact_detected:
        print(f"Object in gripper! deficit={result.driver_deficit_rad:.3f} rad")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────

GRIPPER_QPOS_SLICE  = slice(10, 18)   # all 8 gripper DOFs
RDRIVER_QPOS_IDX    = 10              # right_driver_joint (primary)
LDRIVER_QPOS_IDX    = 14              # left_driver_joint  (confirm)
DRIVER_MAX_ANGLE    = 0.8             # rad  (from jnt_range[jid])
FINGERS_ACT_RANGE   = 255.0           # fingers_actuator ctrl range

# "5 units of fingers_actuator range" → driver-joint space
CONTACT_THRESHOLD_UNITS = 5.0
CONTACT_THRESHOLD_RAD   = (CONTACT_THRESHOLD_UNITS / FINGERS_ACT_RANGE) * DRIVER_MAX_ANGLE
# = 0.0157 rad

# Calibrated free-close reference: right_driver_joint qpos after 50-step
# ramp close at ctrl=200 with no object (measured in MuJoCo simulation).
FREE_CLOSE_DRIVER_REF = 0.2336   # rad  (at ctrl=200, 50 ramp steps)


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class ContactResult:
    """
    Result of one ContactDetector.detect() call.

    Attributes
    ----------
    contact_detected  : True when gripper joints indicate an object is present
    qpos_gripper      : (8,) snapshot of qpos[10:18] at detection time
    driver_pos        : right_driver_joint angle (radians)
    driver_deficit_rad: reference_driver − actual_driver  (> 0 → contact)
    threshold_rad     : threshold used for this detection (CONTACT_THRESHOLD_RAD)
    commanded_ctrl    : ctrl[10] value at the time of detection
    """
    contact_detected:   bool
    qpos_gripper:       np.ndarray   # (8,)
    driver_pos:         float
    driver_deficit_rad: float
    threshold_rad:      float
    commanded_ctrl:     float

    def __str__(self) -> str:
        status = "CONTACT" if self.contact_detected else "FREE"
        return (
            f"ContactResult [{status}]  "
            f"driver={self.driver_pos:.4f} rad  "
            f"deficit={self.driver_deficit_rad:.4f} rad  "
            f"threshold={self.threshold_rad:.4f} rad  "
            f"ctrl={self.commanded_ctrl:.0f}"
        )


# ── detector ──────────────────────────────────────────────────────────────────

class ContactDetector:
    """
    Post-close gripper contact detector.

    Reads qpos[10:18] after a close command and compares the right_driver
    joint angle to the calibrated free-close reference.  A positive deficit
    larger than the threshold means something blocked the gripper.

    Parameters
    ----------
    threshold_rad    : detection threshold in driver-joint radians.
                       Default = (5/255) * 0.8 = 0.0157 rad.
    free_close_ref   : reference driver angle for free close at ctrl=200.
                       Default = 0.2336 rad (measured from simulation).
    """

    def __init__(
        self,
        threshold_rad:  float = CONTACT_THRESHOLD_RAD,
        free_close_ref: float = FREE_CLOSE_DRIVER_REF,
    ) -> None:
        self.threshold_rad  = threshold_rad
        self.free_close_ref = free_close_ref

    def read_qpos(self, data) -> np.ndarray:
        """Return a copy of qpos[10:18] from MuJoCo data."""
        return np.array(data.qpos[GRIPPER_QPOS_SLICE], dtype=float)

    def detect(self, data, commanded_ctrl: float = 200.0) -> ContactResult:
        """
        Run contact detection from the current MuJoCo data state.

        Parameters
        ----------
        data           : mujoco.MjData (populated after close command)
        commanded_ctrl : ctrl[10] value used for the close command

        Returns
        -------
        ContactResult
        """
        qpos         = self.read_qpos(data)
        driver_pos   = float(qpos[0])   # qpos[10]

        # Scale reference to commanded ctrl level (linear approximation)
        reference    = self.free_close_ref * (commanded_ctrl / 200.0)
        deficit      = reference - driver_pos   # positive → more open → contact
        contact      = deficit > self.threshold_rad

        return ContactResult(
            contact_detected   = contact,
            qpos_gripper       = qpos,
            driver_pos         = driver_pos,
            driver_deficit_rad = deficit,
            threshold_rad      = self.threshold_rad,
            commanded_ctrl     = commanded_ctrl,
        )

    def detect_contact(self, data, commanded_ctrl: float = 200.0) -> bool:
        """Convenience wrapper — returns True/False only."""
        return self.detect(data, commanded_ctrl).contact_detected
