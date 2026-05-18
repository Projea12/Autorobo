"""
Single source of truth for all robot model identifiers.
Import these instead of hardcoding strings anywhere else in the codebase.
"""

from __future__ import annotations

import os

# Absolute path to the MJCF model file
ROBOT_XML_PATH: str = os.path.join(os.path.dirname(__file__), "robot.xml")

# ── DOF counts ────────────────────────────────────────────────────────────────
NQ: int = 17   # generalised position dimension (qpos length)
NV: int = 16   # generalised velocity dimension (qvel length)
NU: int = 9    # number of actuators (ctrl length)

# ── Joint names (order matches qpos after freejoint) ─────────────────────────
BASE_FREE_JOINT: str = "root"

WHEEL_JOINTS: tuple[str, str] = ("wheel_left_joint", "wheel_right_joint")

ARM_JOINTS: tuple[str, ...] = (
    "joint1",   # shoulder pan   — z-axis ±π
    "joint2",   # shoulder lift  — y-axis [−π/2, π]
    "joint3",   # elbow          — y-axis ±π
    "joint4",   # wrist 1        — y-axis ±π
    "joint5",   # wrist 2        — z-axis ±π
    "joint6",   # wrist 3 / roll — x-axis ±π
)

GRIPPER_JOINTS: tuple[str, str] = ("finger_left_joint", "finger_right_joint")

ALL_JOINTS: tuple[str, ...] = WHEEL_JOINTS + ARM_JOINTS + GRIPPER_JOINTS

# ── Actuator names (order matches ctrl vector) ────────────────────────────────
WHEEL_ACTUATORS: tuple[str, str] = ("drive_left", "drive_right")

ARM_ACTUATORS: tuple[str, ...] = (
    "arm_j1", "arm_j2", "arm_j3",
    "arm_j4", "arm_j5", "arm_j6",
)

GRIPPER_ACTUATOR: str = "gripper"

ALL_ACTUATORS: tuple[str, ...] = WHEEL_ACTUATORS + ARM_ACTUATORS + (GRIPPER_ACTUATOR,)

# ── ctrl vector slice indices ─────────────────────────────────────────────────
CTRL_WHEEL_L: int = 0
CTRL_WHEEL_R: int = 1
CTRL_ARM     = slice(2, 8)   # arm_j1 … arm_j6
CTRL_GRIPPER: int = 8

# ── Sensor names ──────────────────────────────────────────────────────────────
SENSOR_IMU_GYRO:   str = "imu_gyro"
SENSOR_IMU_ACCEL:  str = "imu_accel"
SENSOR_IMU_QUAT:   str = "imu_quat"
SENSOR_BASE_LINVEL: str = "base_linvel"
SENSOR_BASE_ANGVEL: str = "base_angvel"

ARM_POSITION_SENSORS: tuple[str, ...] = ("q1", "q2", "q3", "q4", "q5", "q6")
ARM_VELOCITY_SENSORS: tuple[str, ...] = ("dq1", "dq2", "dq3", "dq4", "dq5", "dq6")

SENSOR_WRIST_FORCE:  str = "wrist_force"
SENSOR_WRIST_TORQUE: str = "wrist_torque"
SENSOR_FINGER_L:     str = "finger_l"
SENSOR_FINGER_R:     str = "finger_r"
SENSOR_TOUCH_L:      str = "touch_left"
SENSOR_TOUCH_R:      str = "touch_right"
SENSOR_EE_POS:       str = "ee_pos"
SENSOR_EE_QUAT:      str = "ee_quat"
SENSOR_EE_LINVEL:    str = "ee_linvel"

ALL_SENSORS: tuple[str, ...] = (
    SENSOR_IMU_GYRO, SENSOR_IMU_ACCEL, SENSOR_IMU_QUAT,
    SENSOR_BASE_LINVEL, SENSOR_BASE_ANGVEL,
    *ARM_POSITION_SENSORS, *ARM_VELOCITY_SENSORS,
    SENSOR_WRIST_FORCE, SENSOR_WRIST_TORQUE,
    SENSOR_FINGER_L, SENSOR_FINGER_R,
    SENSOR_TOUCH_L, SENSOR_TOUCH_R,
    SENSOR_EE_POS, SENSOR_EE_QUAT, SENSOR_EE_LINVEL,
)

# ── Site names ────────────────────────────────────────────────────────────────
SITE_IMU:        str = "imu_site"
SITE_FT:         str = "ft_site"
SITE_EE:         str = "ee_site"
SITE_TOUCH_L:    str = "touch_left_site"
SITE_TOUCH_R:    str = "touch_right_site"

# ── Physical limits ───────────────────────────────────────────────────────────
MAX_WHEEL_VEL:     float = 8.0    # rad/s (≈ 0.8 m/s at rim)
MAX_GRIPPER_OPEN:  float = 0.040  # metres per finger (118 mm total gap when open)
MAX_WRIST_FORCE:   float = 50.0   # Newton — safety threshold
MAX_WRIST_TORQUE:  float = 10.0   # Newton·metre

# ── Keyframe names ────────────────────────────────────────────────────────────
KF_HOME:       str = "home"
KF_READY:      str = "ready"
KF_GRASP_OPEN: str = "grasp_open"
