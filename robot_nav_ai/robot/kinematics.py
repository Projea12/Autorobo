"""
robot/kinematics.py — FK, IK, and reachability for TidyBot (Blocks 4.1–4.4).

Why MuJoCo for all kinematics
-------------------------------
TidyBot is defined as a MuJoCo XML.  MuJoCo 3.x provides:
  • mj_kinematics()  — full forward kinematics in one call
  • mj_jacSite()     — analytical Jacobian (3×nv positional rows)
Using MuJoCo directly avoids URDF conversion and guarantees the FK/IK
results match the physics simulation exactly.

IK method — damped least-squares (DLS)
---------------------------------------
Standard pseudoinverse becomes numerically unstable near singularities
(J^T(JJ^T)^{-1} blows up when JJ^T is near-singular).  DLS adds a
damping term λ² to the diagonal:

    Δq = J^T (J J^T + λ²I)^{-1} Δx

This trades off accuracy for stability; the joint update is always finite
and the arm slows down gracefully as it approaches singular configurations.
λ=0.01 is small enough that it has negligible effect away from singularities.

Joint indexing
--------------
qpos layout (nq=18):
  [0]   joint_x    (base slide)
  [1]   joint_y    (base slide)
  [2]   joint_th   (base rotate)
  [3]   joint_1    (Kinova shoulder rotation)
  [4]   joint_2    (Kinova shoulder tilt,   limits ±2.2497 rad)
  [5]   joint_3    (Kinova elbow rotation)
  [6]   joint_4    (Kinova elbow tilt,      limits ±2.5796 rad)
  [7]   joint_5    (Kinova wrist rotation)
  [8]   joint_6    (Kinova wrist tilt,      limits ±2.0996 rad)
  [9]   joint_7    (Kinova wrist twist)
  [10–17] gripper joints

ARM_QPOS_SLICE = slice(3, 10)  — the 7 Kinova DOFs we solve for.

Reachability
------------
The Kinova Gen3 has a maximum reach of ~0.9 m from its base.
gen3/base_link is mounted at z=0.335 m above the robot base_link.
A target is unreachable if its Euclidean distance from the arm base
exceeds MAX_REACH, or if it is below the floor (z < 0).

Coordinate frames
-----------------
MuJoCo world frame = robot base frame:
    X = right, Y = forward, Z = up

Usage
-----
    from robot.kinematics import TidyBotKinematics

    kin = TidyBotKinematics()
    xyz  = kin.fk_home()                       # FK at home keyframe
    q, ok, iters = kin.ik([0.0, 0.5, 0.8])    # IK for a target position
    kin.check_reachable([3.0, 0.0, 0.5])       # raises ReachabilityError
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import mujoco
    _MUJOCO_OK = True
except ImportError:
    _MUJOCO_OK = False

# ── paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCENE_XML = _REPO_ROOT / "robot" / "tidybot" / "scene.xml"

# End-effector site
EE_SITE = "pinch_site"

# Kinova arm DOF slice in qpos (joint_1 … joint_7 → indices 3–9 inclusive)
ARM_QPOS_SLICE = slice(3, 10)

# Home keyframe (joint_x/y/th=0, joint_1–7 from XML, gripper open)
HOME_QPOS = np.array([
    0.0, 0.0, 0.0,
    0.0, 0.26179939, 3.14159265, -2.26892803,
    0.0, 0.95993109, 1.57079633,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float64)

# ── IK hyper-parameters ───────────────────────────────────────────────────────

IK_LAMBDA      = 0.01    # DLS damping factor
IK_MAX_ITER    = 150     # maximum DLS iterations
IK_POS_TOL     = 1e-4   # convergence threshold (metres)
IK_STEP_SCALE  = 1.0    # joint-update scale (1.0 = full Newton step per iter)

# ── reachability ──────────────────────────────────────────────────────────────

# gen3/base_link is at [0, 0, 0.335] in MuJoCo world frame.
ARM_BASE_XYZ = np.array([0.0, 0.0, 0.335], dtype=np.float64)
# Kinova Gen3 rated reach = 0.9024 m; we use a small safety margin.
MAX_REACH    = 0.89   # metres


# ── exceptions ────────────────────────────────────────────────────────────────

class ReachabilityError(Exception):
    """Raised when a target is outside the Kinova Gen3 reachable workspace."""


# ── IK result ─────────────────────────────────────────────────────────────────

@dataclass
class IKResult:
    """
    Result of a single IK solve.

    Attributes
    ----------
    q_arm       : (7,) solved joint angles for joint_1 … joint_7 (radians)
    converged   : True if position error < IK_POS_TOL
    iterations  : number of DLS iterations taken
    final_error : residual position error (metres)
    ee_xyz      : achieved end-effector position (metres, world frame)
    """
    q_arm:       np.ndarray   # (7,)
    converged:   bool
    iterations:  int
    final_error: float
    ee_xyz:      np.ndarray   # (3,)

    def __str__(self) -> str:
        status = "CONVERGED" if self.converged else "FAILED"
        return (
            f"IKResult [{status}]  iters={self.iterations}  "
            f"err={self.final_error*1000:.2f} mm\n"
            f"  EE  = ({self.ee_xyz[0]:+.4f}, {self.ee_xyz[1]:+.4f}, "
            f"{self.ee_xyz[2]:+.4f}) m\n"
            f"  q   = {np.round(self.q_arm, 4)}"
        )


# ── main class ────────────────────────────────────────────────────────────────

class TidyBotKinematics:
    """
    MuJoCo-based FK, IK, and reachability for TidyBot.

    Parameters
    ----------
    scene_xml : path to MuJoCo scene XML (default: robot/tidybot/scene.xml)
    """

    def __init__(self, scene_xml: Optional[Path] = None) -> None:
        if not _MUJOCO_OK:
            raise ImportError("pip install mujoco")
        xml_path = Path(scene_xml) if scene_xml else _SCENE_XML
        if not xml_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data  = mujoco.MjData(self.model)

        self._ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE
        )
        if self._ee_id < 0:
            raise RuntimeError(f"Site '{EE_SITE}' not found in model")

        # Pre-compute arm joint limit arrays (shape (7,2))
        self._arm_limits = self._build_arm_limits()

    # ── FK ────────────────────────────────────────────────────────────────────

    def fk(self, qpos: np.ndarray) -> np.ndarray:
        """
        Forward kinematics for a full qpos vector.

        Parameters
        ----------
        qpos : (nq,) joint positions

        Returns
        -------
        (3,) EE position in world frame
        """
        if qpos.shape[0] != self.model.nq:
            raise ValueError(f"qpos length {qpos.shape[0]} != model.nq {self.model.nq}")
        self.data.qpos[:] = qpos
        mujoco.mj_fwdPosition(self.model, self.data)
        return self.data.site_xpos[self._ee_id].copy()

    def fk_home(self) -> np.ndarray:
        """EE position at the home keyframe."""
        qpos = np.zeros(self.model.nq)
        n    = min(len(HOME_QPOS), self.model.nq)
        qpos[:n] = HOME_QPOS[:n]
        return self.fk(qpos)

    def fk_keyframe(self, name: str) -> np.ndarray:
        """EE position for a named keyframe (e.g. 'home', 'retract')."""
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            raise ValueError(f"Keyframe '{name}' not found")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_fwdPosition(self.model, self.data)
        return self.data.site_xpos[self._ee_id].copy()

    # ── IK ────────────────────────────────────────────────────────────────────

    def ik(
        self,
        target_xyz:  Tuple[float, float, float] | np.ndarray,
        q_init:      Optional[np.ndarray] = None,
        max_iter:    int   = IK_MAX_ITER,
        pos_tol:     float = IK_POS_TOL,
        lam:         float = IK_LAMBDA,
    ) -> IKResult:
        """
        Solve IK for a Cartesian target using damped least-squares.

        Keeps the robot base fixed (joint_x/y/th = 0) and solves only for
        the 7 Kinova arm joints (joint_1 … joint_7).

        Parameters
        ----------
        target_xyz : (3,) desired EE position in world/base frame (metres)
        q_init     : (7,) initial arm joint angles (default: home pose)
        max_iter   : maximum DLS iterations
        pos_tol    : convergence threshold in metres
        lam        : DLS damping factor (λ)

        Returns
        -------
        IKResult
        """
        target = np.asarray(target_xyz, dtype=np.float64)

        # Initialise full qpos from home, then overwrite arm joints
        qpos = np.zeros(self.model.nq)
        qpos[ARM_QPOS_SLICE] = (
            q_init if q_init is not None
            else HOME_QPOS[ARM_QPOS_SLICE]
        )

        nv   = self.model.nv
        # Jacobian buffers: jacp (3×nv) positional rows
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))   # rotational (unused for pos-only IK)

        for i in range(max_iter):
            # FK — mj_fwdPosition fills xmat/site_xpos AND the data
            # structures that mj_jacSite needs (unlike mj_kinematics alone).
            self.data.qpos[:] = qpos
            mujoco.mj_fwdPosition(self.model, self.data)

            # Position error
            ee  = self.data.site_xpos[self._ee_id]
            err = target - ee
            pos_error = float(np.linalg.norm(err))
            if pos_error < pos_tol:
                return IKResult(
                    q_arm       = qpos[ARM_QPOS_SLICE].copy(),
                    converged   = True,
                    iterations  = i,
                    final_error = pos_error,
                    ee_xyz      = ee.copy(),
                )

            # Jacobian (3×nv) at pinch_site
            mujoco.mj_jacSite(self.model, self.data,
                               jacp, jacr, self._ee_id)

            # Extract columns for arm joints only (nv == nq for hinge joints)
            # ARM dof indices in velocity space = same as qpos for hinge joints
            J_arm = jacp[:, ARM_QPOS_SLICE]   # (3, 7)

            # Damped least-squares: Δq = J^T (J J^T + λ²I)^{-1} Δx
            JJT  = J_arm @ J_arm.T                    # (3,3)
            damp = (lam ** 2) * np.eye(3)
            dq   = J_arm.T @ np.linalg.solve(JJT + damp, err)  # (7,)

            # Update and clamp to joint limits
            qpos[ARM_QPOS_SLICE] = np.clip(
                qpos[ARM_QPOS_SLICE] + IK_STEP_SCALE * dq,
                self._arm_limits[:, 0],
                self._arm_limits[:, 1],
            )

        # Did not converge — return best result so far
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        ee  = self.data.site_xpos[self._ee_id]
        err = float(np.linalg.norm(target - ee))
        return IKResult(
            q_arm       = qpos[ARM_QPOS_SLICE].copy(),
            converged   = False,
            iterations  = max_iter,
            final_error = err,
            ee_xyz      = ee.copy(),
        )

    # ── reachability ──────────────────────────────────────────────────────────

    def check_reachable(
        self,
        target_xyz: Tuple[float, float, float] | np.ndarray,
        max_reach:  float = MAX_REACH,
    ) -> None:
        """
        Check whether a Cartesian target is within the arm's reach envelope.

        Raises
        ------
        ReachabilityError
            If the target is too far from the arm base, or below the floor.
        """
        t   = np.asarray(target_xyz, dtype=np.float64)
        dist = float(np.linalg.norm(t - ARM_BASE_XYZ))

        if t[2] < 0.0:
            raise ReachabilityError(
                f"Target z={t[2]:.3f} m is below the floor (z < 0)"
            )
        if dist > max_reach:
            raise ReachabilityError(
                f"Target at {np.round(t,3)} m is {dist:.3f} m from arm base "
                f"(max reach = {max_reach:.2f} m)"
            )

    def is_reachable(
        self,
        target_xyz: Tuple[float, float, float] | np.ndarray,
        max_reach:  float = MAX_REACH,
    ) -> bool:
        """Return True if target is within reach, False otherwise."""
        try:
            self.check_reachable(target_xyz, max_reach)
            return True
        except ReachabilityError:
            return False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_arm_limits(self) -> np.ndarray:
        """
        Build (7, 2) array of [lo, hi] joint limits for joint_1 … joint_7.

        Joints without explicit limits get [-2π, 2π] (full rotation).
        """
        limits = np.full((7, 2), [-2 * np.pi, 2 * np.pi])
        for arm_idx, qpos_idx in enumerate(range(3, 10)):
            jnt_limited = self.model.jnt_limited[qpos_idx]
            if jnt_limited:
                limits[arm_idx] = self.model.jnt_range[qpos_idx]
        return limits

    def __repr__(self) -> str:
        return (
            f"TidyBotKinematics(nq={self.model.nq}, "
            f"nsites={self.model.nsite}, ee='{EE_SITE}')"
        )
