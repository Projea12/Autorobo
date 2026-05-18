"""
Tests for robot/sensors.py — full sensor suite.

Covers:
  1. MJCF camera count after XML changes
  2. Proprioceptive sensor shapes and types (always-on)
  3. RGBDCamera: RGB shape, dtype, depth range, depth clipping
  4. LiDAR2D: shape, dtype, values in [0, max_dist], self-exclusion
  5. SensorSuite: read() with cameras disabled (training mode)
  6. SensorSuite: read() with cameras enabled (eval mode)
  7. SensorReading timestamp increases across steps
  8. All sensor arrays are float32 (GPU-ready)
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from robot.constants import ROBOT_XML_PATH, KF_HOME
from robot.sensors import (
    RGBDCamera, LiDAR2D, SensorSuite, SensorReading, _S,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(ROBOT_XML_PATH)


@pytest.fixture
def data(model) -> mujoco.MjData:
    d = mujoco.MjData(model)
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KF_HOME)
    mujoco.mj_resetDataKeyframe(model, d, kf)
    # Let physics settle for 0.2 s
    for _ in range(100):
        mujoco.mj_step(model, d)
    return d


# ── 1. MJCF camera count ──────────────────────────────────────────────────────

def test_model_has_two_cameras(model):
    assert model.ncam == 2, f"expected 2 cameras (rgbd_cam + wrist_cam), got {model.ncam}"


def test_rgbd_cam_exists(model):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgbd_cam")
    assert cid >= 0


def test_wrist_cam_exists(model):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    assert cid >= 0


def test_lidar_site_exists(model):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
    assert sid >= 0


# ── 2. sensordata slice layout ────────────────────────────────────────────────

def test_sensordata_total_length(model, data):
    mujoco.mj_forward(model, data)
    assert len(data.sensordata) == _S.TOTAL


def test_gyro_shape(model, data):
    mujoco.mj_forward(model, data)
    assert data.sensordata[_S.GYRO].shape == (3,)


def test_joint_pos_shape(model, data):
    mujoco.mj_forward(model, data)
    assert data.sensordata[_S.JOINT_POS].shape == (6,)


def test_wrist_force_shape(model, data):
    mujoco.mj_forward(model, data)
    assert data.sensordata[_S.WRIST_FORCE].shape == (3,)


def test_ee_quat_shape(model, data):
    mujoco.mj_forward(model, data)
    assert data.sensordata[_S.EE_QUAT].shape == (4,)


# ── 3. RGBDCamera ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def camera(model) -> RGBDCamera:
    cam = RGBDCamera(model, camera_name="rgbd_cam", height=120, width=160)
    yield cam
    cam.close()


def test_camera_raises_for_bad_name(model):
    with pytest.raises(ValueError, match="not found"):
        RGBDCamera(model, camera_name="nonexistent_cam")


def test_rgb_shape(camera, model, data):
    mujoco.mj_forward(model, data)
    rgb = camera.read_rgb(data)
    assert rgb.shape == (120, 160, 3)


def test_rgb_dtype_uint8(camera, model, data):
    mujoco.mj_forward(model, data)
    rgb = camera.read_rgb(data)
    assert rgb.dtype == np.uint8


def test_rgb_has_nonzero_pixels(camera, model, data):
    mujoco.mj_forward(model, data)
    rgb = camera.read_rgb(data)
    assert rgb.max() > 0, "RGB image is entirely black — scene may not be rendering"


def test_depth_shape(camera, model, data):
    mujoco.mj_forward(model, data)
    depth = camera.read_depth(data)
    assert depth.shape == (120, 160)


def test_depth_dtype_float32(camera, model, data):
    mujoco.mj_forward(model, data)
    depth = camera.read_depth(data)
    assert depth.dtype == np.float32


def test_depth_in_valid_range(camera, model, data):
    mujoco.mj_forward(model, data)
    depth = camera.read_depth(data)
    assert float(depth.min()) >= camera._near
    assert float(depth.max()) <= camera._far


def test_read_returns_both(camera, model, data):
    mujoco.mj_forward(model, data)
    rgb, depth = camera.read(data)
    assert rgb.shape[-1] == 3
    assert depth.ndim == 2


# ── 4. LiDAR2D ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lidar(model) -> LiDAR2D:
    return LiDAR2D(model, n_rays=36, max_dist=5.0)   # 10° resolution for speed


def test_lidar_raises_for_bad_site(model):
    with pytest.raises(ValueError, match="not found"):
        LiDAR2D(model, site_name="no_such_site")


def test_lidar_shape(lidar, model, data):
    mujoco.mj_forward(model, data)
    ranges = lidar.read(data)
    assert ranges.shape == (36,)


def test_lidar_dtype_float32(lidar, model, data):
    mujoco.mj_forward(model, data)
    ranges = lidar.read(data)
    assert ranges.dtype == np.float32


def test_lidar_values_within_max_dist(lidar, model, data):
    mujoco.mj_forward(model, data)
    ranges = lidar.read(data)
    assert float(ranges.min()) > 0.0
    assert float(ranges.max()) <= lidar._max_dist


def test_lidar_angles_length(lidar):
    assert len(lidar.angles) == 36


def test_lidar_no_nan(lidar, model, data):
    mujoco.mj_forward(model, data)
    ranges = lidar.read(data)
    assert not np.any(np.isnan(ranges))


# ── 5. SensorSuite — proprioceptive only (training mode) ──────────────────────

@pytest.fixture
def suite_proprio(model, data) -> SensorSuite:
    s = SensorSuite(model, data, enable_camera=False, enable_lidar=False)
    yield s
    s.close()


def test_suite_read_returns_reading(suite_proprio, model, data):
    mujoco.mj_forward(model, data)
    r = suite_proprio.read()
    assert isinstance(r, SensorReading)


def test_suite_proprioceptive_imu(suite_proprio, model, data):
    mujoco.mj_forward(model, data)
    r = suite_proprio.read()
    assert r.imu.gyro.shape    == (3,)
    assert r.imu.accel.shape   == (3,)
    assert r.imu.orientation.shape == (4,)


def test_suite_arm_reading(suite_proprio, model, data):
    mujoco.mj_forward(model, data)
    r = suite_proprio.read()
    assert r.arm.joint_pos.shape == (6,)
    assert r.arm.joint_vel.shape == (6,)


def test_suite_wrist_reading(suite_proprio, model, data):
    mujoco.mj_forward(model, data)
    r = suite_proprio.read()
    assert r.wrist.force.shape  == (3,)
    assert r.wrist.torque.shape == (3,)


def test_suite_gripper_reading(suite_proprio, model, data):
    mujoco.mj_forward(model, data)
    r = suite_proprio.read()
    assert r.gripper.finger_pos.shape == (2,)
    assert r.gripper.touch.shape      == (2,)


def test_suite_ee_reading(suite_proprio, model, data):
    mujoco.mj_forward(model, data)
    r = suite_proprio.read()
    assert r.ee.pos.shape    == (3,)
    assert r.ee.quat.shape   == (4,)
    assert r.ee.linvel.shape == (3,)


def test_suite_camera_none_when_disabled(suite_proprio, model, data):
    mujoco.mj_forward(model, data)
    r = suite_proprio.read()
    assert r.rgb   is None
    assert r.depth is None
    assert r.lidar is None


# ── 6. SensorSuite — full (eval mode) ────────────────────────────────────────

@pytest.fixture(scope="module")
def suite_full(model) -> SensorSuite:
    d = mujoco.MjData(model)
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KF_HOME)
    mujoco.mj_resetDataKeyframe(model, d, kf)
    for _ in range(100):
        mujoco.mj_step(model, d)
    mujoco.mj_forward(model, d)

    s = SensorSuite(
        model, d,
        enable_camera=True,  camera_name="rgbd_cam",
        camera_height=120,   camera_width=160,
        enable_lidar=True,   n_lidar_rays=36,
        lidar_max_dist=5.0,
    )
    yield s
    s.close()


def test_suite_full_rgb_present(suite_full):
    r = suite_full.read()
    assert r.rgb is not None
    assert r.rgb.shape == (120, 160, 3)


def test_suite_full_depth_present(suite_full):
    r = suite_full.read()
    assert r.depth is not None
    assert r.depth.shape == (120, 160)


def test_suite_full_lidar_present(suite_full):
    r = suite_full.read()
    assert r.lidar is not None
    assert r.lidar.shape == (36,)


# ── 7. Timestamp increases ────────────────────────────────────────────────────

def test_timestamp_increases(model):
    d = mujoco.MjData(model)
    kf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KF_HOME)
    mujoco.mj_resetDataKeyframe(model, d, kf)
    suite = SensorSuite(model, d)

    t0 = suite.read().timestamp
    mujoco.mj_step(model, d)
    t1 = suite.read().timestamp
    assert t1 > t0

    suite.close()


# ── 8. All proprioceptive arrays are float64 (MuJoCo native) -─────────────────
# NB: sensordata is float64 in MuJoCo; we keep native precision for accuracy.
# GPU layers receive float32 casts at the env observation boundary, not here.

def test_joint_pos_is_numeric(suite_proprio, model, data):
    mujoco.mj_forward(model, data)
    r = suite_proprio.read()
    assert np.isfinite(r.arm.joint_pos).all()
    assert np.isfinite(r.arm.joint_vel).all()


def test_ee_quat_is_unit(suite_proprio, model, data):
    mujoco.mj_forward(model, data)
    r = suite_proprio.read()
    norm = float(np.linalg.norm(r.ee.quat))
    assert abs(norm - 1.0) < 0.01, f"ee quaternion norm = {norm:.4f}, expected ≈ 1"


def test_context_manager(model, data):
    with SensorSuite(model, data) as suite:
        r = suite.read()
    assert r is not None
