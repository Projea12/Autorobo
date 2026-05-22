"""
robot/kinematics.py — Forward kinematics via MuJoCo (Block 4.1 / 4.2).

Why MuJoCo instead of pinocchio/pink
--------------------------------------
TidyBot is already modelled as a MuJoCo XML.  MuJoCo 3.x ships with a
full kinematic tree solver that computes body/site positions in one call
(mj_kinematics / mj_fwdPosition) — no URDF conversion or secondary
library is needed.  This keeps the dependency surface minimal and
guarantees the FK matches the physics simulation exactly.

Coordinate frames
-----------------
MuJoCo world frame = robot base frame:
    X = right, Y = forward, Z = up

End-effector site
-----------------
The ``pinch_site`` site is attached to ``bracelet_link`` with a fixed
offset of (0, 0, -0.181525) m and a 180° rotation about X.  MuJoCo
resolves this automatically; we just query ``data.site_xpos["pinch_site"]``.

Usage
-----
    from robot.kinematics import TidyBotKinematics

    kin = TidyBotKinematics()
    xyz = kin.fk_home()        # EE position at home keyframe
    xyz = kin.fk(qpos_array)   # EE position for arbitrary joint angles
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# MuJoCo is a required dependency for this module
try:
    import mujoco
    _MUJOCO_OK = True
except ImportError:
    _MUJOCO_OK = False

# ── paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_SCENE_XML  = _REPO_ROOT / "robot" / "tidybot" / "scene.xml"

# Home keyframe joint positions (from tidybot.xml <keyframe>)
# Order: joint_x, joint_y, joint_th, joint_1…7, gripper joints (×8)
HOME_QPOS = np.array([
    0.0, 0.0, 0.0,                          # base x, y, heading
    0.0,                                    # joint_1
    0.26179939,                             # joint_2  (~15°)
    3.14159265,                             # joint_3  (π)
   -2.26892803,                             # joint_4  (~-130°)
    0.0,                                    # joint_5
    0.95993109,                             # joint_6  (~55°)
    1.57079633,                             # joint_7  (π/2)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # gripper (7 joints)
], dtype=np.float64)

# End-effector site name
EE_SITE = "pinch_site"


# ── kinematics class ──────────────────────────────────────────────────────────

class TidyBotKinematics:
    """
    Thin MuJoCo wrapper for TidyBot forward kinematics.

    Parameters
    ----------
    scene_xml : path to the MuJoCo scene XML (default: robot/tidybot/scene.xml)
    """

    def __init__(self, scene_xml: Optional[Path] = None) -> None:
        if not _MUJOCO_OK:
            raise ImportError(
                "mujoco is required for TidyBotKinematics. "
                "Install with: pip install mujoco"
            )
        xml_path = Path(scene_xml) if scene_xml else _SCENE_XML
        if not xml_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data  = mujoco.MjData(self.model)

        # Cache site id once
        self._ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE
        )
        if self._ee_id < 0:
            raise RuntimeError(f"Site '{EE_SITE}' not found in model")

    # ── public API ────────────────────────────────────────────────────────────

    def fk(self, qpos: np.ndarray) -> np.ndarray:
        """
        Compute end-effector position for given joint configuration.

        Parameters
        ----------
        qpos : (nq,) joint position array  (must match model.nq)

        Returns
        -------
        xyz : (3,) end-effector position in world/base frame (metres)
        """
        if qpos.shape[0] != self.model.nq:
            raise ValueError(
                f"qpos has {qpos.shape[0]} elements but model has {self.model.nq}"
            )
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self._ee_id].copy()

    def fk_home(self) -> np.ndarray:
        """
        Compute end-effector position at the TidyBot home keyframe.

        Returns
        -------
        xyz : (3,) position in world/base frame  [expected ≈ (0.0, 0.4, 1.1) m]
        """
        # Pad HOME_QPOS to model.nq if the model has extra DOFs
        qpos = np.zeros(self.model.nq)
        n    = min(len(HOME_QPOS), self.model.nq)
        qpos[:n] = HOME_QPOS[:n]
        return self.fk(qpos)

    def fk_keyframe(self, name: str) -> np.ndarray:
        """
        Compute FK for a named keyframe from the XML (e.g. 'home', 'retract').

        Parameters
        ----------
        name : keyframe name as defined in <keyframe> block of the XML

        Returns
        -------
        xyz : (3,) end-effector position
        """
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            raise ValueError(f"Keyframe '{name}' not found in model")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self._ee_id].copy()

    def __repr__(self) -> str:
        return (
            f"TidyBotKinematics(nq={self.model.nq}, "
            f"nsites={self.model.nsite}, ee='{EE_SITE}')"
        )
