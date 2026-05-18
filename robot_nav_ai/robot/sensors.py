"""
robot/sensors.py — Full sensor suite for AutoRobo v1.

Wraps every sensor on the robot into a single read() call that returns a
typed SensorReading dataclass.  Rendering and raycasting are kept lazy so
headless training (no RGB, no LiDAR) has zero overhead.

Sensor inventory
────────────────
  Proprioceptive  (from sensordata, always available)
    IMU           : gyro (3), accel (3), orientation quat (4)
    Base velocity : linear (3), angular (3)
    Arm encoders  : joint position (6), joint velocity (6)
    Wrist F/T     : force (3), torque (3)
    Gripper       : finger positions (2), fingertip touch (2)
    End-effector  : position (3), orientation quat (4), velocity (3)

  Exteroceptive  (rendered / raycast on demand)
    RGB-D camera  : colour (H×W×3 uint8) + depth (H×W float32 m)
    2-D LiDAR     : range array (N_RAYS float32 m), horizontal plane scan

sensordata slice layout (48 floats total — matches robot.xml sensor order):
    [0 :3 ]  imu_gyro
    [3 :6 ]  imu_accel
    [6 :10]  imu_quat          (w x y z)
    [10:13]  base_linvel
    [13:16]  base_angvel
    [16:22]  joint positions   (q1 … q6)
    [22:28]  joint velocities  (dq1 … dq6)
    [28:31]  wrist_force
    [31:34]  wrist_torque
    [34]     finger_left pos
    [35]     finger_right pos
    [36]     touch_left
    [37]     touch_right
    [38:41]  ee_pos
    [41:45]  ee_quat           (w x y z)
    [45:48]  ee_linvel
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np


# ── sensordata slice constants ────────────────────────────────────────────────

class _S:
    """Sensordata index ranges — single source of truth."""
    GYRO         = slice(0,  3)
    ACCEL        = slice(3,  6)
    QUAT         = slice(6,  10)
    BASE_LINVEL  = slice(10, 13)
    BASE_ANGVEL  = slice(13, 16)
    JOINT_POS    = slice(16, 22)
    JOINT_VEL    = slice(22, 28)
    WRIST_FORCE  = slice(28, 31)
    WRIST_TORQUE = slice(31, 34)
    FINGER_POS   = slice(34, 36)
    TOUCH        = slice(36, 38)
    EE_POS       = slice(38, 41)
    EE_QUAT      = slice(41, 45)
    EE_LINVEL    = slice(45, 48)
    TOTAL        = 48


# ── data containers ───────────────────────────────────────────────────────────

@dataclass
class IMUReading:
    gyro:        np.ndarray   # (3,) rad/s
    accel:       np.ndarray   # (3,) m/s²
    orientation: np.ndarray   # (4,) quaternion wxyz


@dataclass
class ArmReading:
    joint_pos: np.ndarray   # (6,) rad  — joint1 … joint6
    joint_vel: np.ndarray   # (6,) rad/s


@dataclass
class WristReading:
    force:  np.ndarray   # (3,) N   — Fx Fy Fz in sensor frame
    torque: np.ndarray   # (3,) N·m — Tx Ty Tz in sensor frame


@dataclass
class GripperReading:
    finger_pos: np.ndarray   # (2,) m   — [left, right] opening
    touch:      np.ndarray   # (2,) N   — [left, right] contact force


@dataclass
class EndEffectorReading:
    pos:    np.ndarray   # (3,) m
    quat:   np.ndarray   # (4,) quaternion wxyz in world frame
    linvel: np.ndarray   # (3,) m/s in world frame


@dataclass
class BaseReading:
    linvel: np.ndarray   # (3,) m/s  in world frame
    angvel: np.ndarray   # (3,) rad/s in world frame


@dataclass
class SensorReading:
    """Complete snapshot of all robot sensors at one timestep."""
    timestamp:    float
    imu:          IMUReading
    base:         BaseReading
    arm:          ArmReading
    wrist:        WristReading
    gripper:      GripperReading
    ee:           EndEffectorReading
    # Exteroceptive — None when the respective sensor is disabled
    rgb:          Optional[np.ndarray] = None   # (H, W, 3) uint8
    depth:        Optional[np.ndarray] = None   # (H, W) float32 m
    lidar:        Optional[np.ndarray] = None   # (N_RAYS,) float32 m


# ── RGB-D camera ──────────────────────────────────────────────────────────────

class RGBDCamera:
    """
    Renders colour and depth images from a named MuJoCo camera.

    Usage
    -----
    cam = RGBDCamera(model, camera_name="rgbd_cam", height=480, width=640)
    rgb, depth = cam.read(data)

    Depth is returned in metres (converted from MuJoCo's normalised buffer).
    Near/far clip planes match the camera's znear/zfar in the model.
    """

    # RealSense D435 analogue defaults
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH  = 640
    DEFAULT_NEAR   = 0.10   # m
    DEFAULT_FAR    = 6.00   # m

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str = "rgbd_cam",
        height: int = DEFAULT_HEIGHT,
        width:  int = DEFAULT_WIDTH,
        near:   float = DEFAULT_NEAR,
        far:    float = DEFAULT_FAR,
    ) -> None:
        self._model       = model
        self._camera_name = camera_name
        self._near        = near
        self._far         = far

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")
        self._cam_id = cam_id

        self._renderer = mujoco.Renderer(model, height=height, width=width)

    # ── public API ────────────────────────────────────────────────────────────

    def read(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Return (rgb, depth) where depth is in metres."""
        rgb   = self._render_rgb(data)
        depth = self._render_depth(data)
        return rgb, depth

    def read_rgb(self, data: mujoco.MjData) -> np.ndarray:
        """Return colour image (H, W, 3) uint8."""
        return self._render_rgb(data)

    def read_depth(self, data: mujoco.MjData) -> np.ndarray:
        """Return depth map (H, W) float32 in metres."""
        return self._render_depth(data)

    def close(self) -> None:
        self._renderer.close()

    # ── internals ─────────────────────────────────────────────────────────────

    def _render_rgb(self, data: mujoco.MjData) -> np.ndarray:
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data, camera=self._camera_name)
        return self._renderer.render().copy()

    def _render_depth(self, data: mujoco.MjData) -> np.ndarray:
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=self._camera_name)
        raw = self._renderer.render()   # normalised [0, 1] or metres depending on MuJoCo version
        # MuJoCo 3: depth buffer is already in metres when depth rendering is enabled
        depth = np.asarray(raw, dtype=np.float32)
        # Clamp to valid range
        depth = np.clip(depth, self._near, self._far)
        return depth


# ── 2-D LiDAR ─────────────────────────────────────────────────────────────────

class LiDAR2D:
    """
    Simulates a 2-D horizontal LiDAR via MuJoCo raycasting.

    Rays are cast in the horizontal plane of the lidar_site frame.
    The robot's own body is excluded so the chassis never self-reports.

    Usage
    -----
    lidar = LiDAR2D(model, n_rays=360, max_dist=10.0)
    ranges = lidar.read(data)   # (360,) float32 metres
    """

    DEFAULT_N_RAYS   = 360     # 1° resolution
    DEFAULT_MAX_DIST = 10.0    # metres
    DEFAULT_SITE     = "lidar_site"
    DEFAULT_BODY     = "base"  # excluded from raycasting

    def __init__(
        self,
        model:       mujoco.MjModel,
        n_rays:      int   = DEFAULT_N_RAYS,
        max_dist:    float = DEFAULT_MAX_DIST,
        site_name:   str   = DEFAULT_SITE,
        exclude_body: str  = DEFAULT_BODY,
    ) -> None:
        self._n_rays   = n_rays
        self._max_dist = max_dist

        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in model")

        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, exclude_body)

        # Pre-compute ray directions in local frame (horizontal plane, CCW)
        angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
        # Local XY directions (Z=0 → horizontal scan)
        self._local_dirs = np.stack(
            [np.cos(angles), np.sin(angles), np.zeros(n_rays)], axis=1
        )  # (N, 3) float64
        self._geomid = np.array([-1], dtype=np.int32)

    # ── public API ────────────────────────────────────────────────────────────

    def read(self, data: mujoco.MjData) -> np.ndarray:
        """
        Cast all rays and return range array (N_RAYS,) float32 metres.
        Rays that miss everything are set to max_dist.
        """
        origin = data.site_xpos[self._site_id].copy()
        # 3×3 rotation matrix of site in world frame (row-major from MuJoCo)
        rot = data.site_xmat[self._site_id].reshape(3, 3)
        # Rotate all local directions to world frame at once
        world_dirs = self._local_dirs @ rot.T   # (N, 3)

        readings = np.full(self._n_rays, self._max_dist, dtype=np.float32)

        for i in range(self._n_rays):
            wdir = world_dirs[i]
            dist = mujoco.mj_ray(
                data._model,          # MjModel
                data,                 # MjData
                origin,               # ray origin
                wdir,                 # ray direction (need not be unit; mj_ray normalises)
                None,                 # geomgroup mask (None = all)
                1,                    # flg_static: include static geoms
                self._body_id,        # bodyexclude: skip robot base
                self._geomid,         # output geom id (mutated in-place)
            )
            if 0.0 <= dist < self._max_dist:
                readings[i] = float(dist)

        return readings

    @property
    def n_rays(self) -> int:
        return self._n_rays

    @property
    def angles(self) -> np.ndarray:
        """Ray angles in radians, CCW from +X axis. Shape (N_RAYS,)."""
        return np.linspace(0.0, 2.0 * np.pi, self._n_rays, endpoint=False)


# ── sensor suite ──────────────────────────────────────────────────────────────

class SensorSuite:
    """
    Aggregates all robot sensors into a single read() call.

    Proprioceptive sensors (IMU, encoders, F/T, touch) are always read from
    data.sensordata — zero extra computation.

    Exteroceptive sensors (RGB-D, LiDAR) are opt-in via enable_camera /
    enable_lidar flags to avoid rendering overhead during training phases
    that do not yet use them.

    Usage
    -----
    suite = SensorSuite(
        model, data,
        enable_camera=True,  camera_name="rgbd_cam",
        enable_lidar=True,   n_lidar_rays=360,
    )
    reading = suite.read()
    print(reading.lidar.shape)    # (360,)
    print(reading.rgb.shape)      # (480, 640, 3)
    print(reading.arm.joint_pos)  # (6,)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data:  mujoco.MjData,
        *,
        enable_camera:  bool  = False,
        camera_name:    str   = "rgbd_cam",
        camera_height:  int   = RGBDCamera.DEFAULT_HEIGHT,
        camera_width:   int   = RGBDCamera.DEFAULT_WIDTH,
        enable_lidar:   bool  = False,
        n_lidar_rays:   int   = LiDAR2D.DEFAULT_N_RAYS,
        lidar_max_dist: float = LiDAR2D.DEFAULT_MAX_DIST,
    ) -> None:
        self._model = model
        self._data  = data

        self._camera: Optional[RGBDCamera] = None
        if enable_camera:
            self._camera = RGBDCamera(
                model, camera_name=camera_name,
                height=camera_height, width=camera_width,
            )

        self._lidar: Optional[LiDAR2D] = None
        if enable_lidar:
            self._lidar = LiDAR2D(model, n_rays=n_lidar_rays, max_dist=lidar_max_dist)

    # ── public API ────────────────────────────────────────────────────────────

    def read(self) -> SensorReading:
        """Read all enabled sensors and return a SensorReading snapshot."""
        s = self._data.sensordata

        rgb: Optional[np.ndarray]   = None
        depth: Optional[np.ndarray] = None
        lidar: Optional[np.ndarray] = None

        if self._camera is not None:
            rgb, depth = self._camera.read(self._data)

        if self._lidar is not None:
            lidar = self._lidar.read(self._data)

        return SensorReading(
            timestamp = float(self._data.time),
            imu       = IMUReading(
                gyro        = s[_S.GYRO].copy(),
                accel       = s[_S.ACCEL].copy(),
                orientation = s[_S.QUAT].copy(),
            ),
            base      = BaseReading(
                linvel = s[_S.BASE_LINVEL].copy(),
                angvel = s[_S.BASE_ANGVEL].copy(),
            ),
            arm       = ArmReading(
                joint_pos = s[_S.JOINT_POS].copy(),
                joint_vel = s[_S.JOINT_VEL].copy(),
            ),
            wrist     = WristReading(
                force  = s[_S.WRIST_FORCE].copy(),
                torque = s[_S.WRIST_TORQUE].copy(),
            ),
            gripper   = GripperReading(
                finger_pos = s[_S.FINGER_POS].copy(),
                touch      = s[_S.TOUCH].copy(),
            ),
            ee        = EndEffectorReading(
                pos    = s[_S.EE_POS].copy(),
                quat   = s[_S.EE_QUAT].copy(),
                linvel = s[_S.EE_LINVEL].copy(),
            ),
            rgb   = rgb,
            depth = depth,
            lidar = lidar,
        )

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "SensorSuite":
        return self

    def __exit__(self, *_) -> None:
        self.close()
