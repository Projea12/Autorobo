"""
Structural and physics validation for robot/robot.xml.

Tests verify:
  1. MJCF loads without error
  2. DOF counts match the header contract (nq=17, nv=16, nu=9)
  3. Every joint, actuator, sensor and site declared in constants.py exists
  4. Joint limits are set and in expected range
  5. Physics steps from the home keyframe without producing NaN / Inf
  6. Robot settles near the correct ground height (balance check)
  7. Gripper equality constraint is registered
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from robot.constants import (
    ROBOT_XML_PATH,
    NQ, NV, NU,
    ALL_JOINTS, ARM_JOINTS, GRIPPER_JOINTS, WHEEL_JOINTS,
    ALL_ACTUATORS, ARM_ACTUATORS, WHEEL_ACTUATORS, GRIPPER_ACTUATOR,
    ALL_SENSORS, SITE_EE, SITE_IMU, SITE_FT,
    KF_HOME, KF_READY, KF_GRASP_OPEN,
    MAX_GRIPPER_OPEN,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(ROBOT_XML_PATH)


@pytest.fixture(scope="module")
def data(model) -> mujoco.MjData:
    return mujoco.MjData(model)


# ── 1. MJCF loads ─────────────────────────────────────────────────────────────

def test_xml_loads(model):
    assert model is not None


# ── 2. DOF counts ─────────────────────────────────────────────────────────────

def test_nq(model):
    assert model.nq == NQ, f"expected nq={NQ}, got {model.nq}"


def test_nv(model):
    assert model.nv == NV, f"expected nv={NV}, got {model.nv}"


def test_nu(model):
    assert model.nu == NU, f"expected nu={NU}, got {model.nu}"


# ── 3. Joints exist ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_JOINTS)
def test_joint_exists(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"joint '{name}' not found in model"


def test_freejoint_exists(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    assert jid >= 0


# ── 4. Actuators exist ────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_ACTUATORS)
def test_actuator_exists(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    assert aid >= 0, f"actuator '{name}' not found in model"


# ── 5. Sensors exist ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ALL_SENSORS)
def test_sensor_exists(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    assert sid >= 0, f"sensor '{name}' not found in model"


# ── 6. Sites exist ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("site_name", [SITE_EE, SITE_IMU, SITE_FT])
def test_site_exists(model, site_name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    assert sid >= 0, f"site '{site_name}' not found in model"


# ── 7. Joint limits are set ───────────────────────────────────────────────────

def test_arm_joints_have_limits(model):
    for name in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        limited = model.jnt_limited[jid]
        assert limited, f"arm joint '{name}' has no limits set"


def test_gripper_joints_in_range(model):
    for name in GRIPPER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        assert lo == pytest.approx(0.0, abs=1e-4)
        assert hi == pytest.approx(MAX_GRIPPER_OPEN, abs=1e-4)


# ── 8. Equality constraint (gripper coupling) exists ─────────────────────────

def test_gripper_equality_constraint(model):
    assert model.neq >= 1, "expected at least one equality constraint (gripper coupling)"
    found = False
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
            j1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.eq_obj1id[i])
            j2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.eq_obj2id[i])
            if set([j1, j2]) == {"finger_left_joint", "finger_right_joint"}:
                found = True
                break
    assert found, "gripper_coupling equality constraint not found"


# ── 9. Keyframes exist and have correct size ──────────────────────────────────

@pytest.mark.parametrize("kf_name", [KF_HOME, KF_READY, KF_GRASP_OPEN])
def test_keyframe_exists(model, kf_name):
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, kf_name)
    assert kf_id >= 0, f"keyframe '{kf_name}' not found"


def test_home_keyframe_qpos_length(model):
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KF_HOME)
    # Each keyframe stores a full qpos slice
    start = kf_id * model.nq
    end   = start + model.nq
    assert end <= len(model.key_qpos)


# ── 10. Physics: no NaN after stepping from home pose ─────────────────────────

def test_physics_no_nan_from_home(model):
    d = mujoco.MjData(model)
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KF_HOME)
    mujoco.mj_resetDataKeyframe(model, d, kf_id)
    for _ in range(200):
        mujoco.mj_step(model, d)
    assert not np.any(np.isnan(d.qpos)), "NaN in qpos after 200 steps"
    assert not np.any(np.isnan(d.qvel)), "NaN in qvel after 200 steps"
    assert not np.any(np.isnan(d.sensordata)), "NaN in sensordata after 200 steps"


# ── 11. Physics: base settles near ground level ───────────────────────────────

def test_base_settles_at_correct_height(model):
    """
    After dropping from the home position, the base body should settle
    between 0.13 m and 0.17 m (nominal z = 0.15 m, wheels touching floor).
    """
    d = mujoco.MjData(model)
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KF_HOME)
    mujoco.mj_resetDataKeyframe(model, d, kf_id)

    for _ in range(500):   # 1 second of simulation at dt=0.002
        mujoco.mj_step(model, d)

    base_z = d.qpos[2]   # z coordinate from freejoint (index 2 of 7)
    assert 0.13 <= base_z <= 0.17, (
        f"base z = {base_z:.4f} after settling — expected 0.13–0.17 m"
    )


# ── 12. Sensor data has expected dimensionality ───────────────────────────────

def test_sensordata_length(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    mujoco.mj_forward(model, d)
    # 22 sensors: 3+3+4 + 3+3 + 6+6 + 3+3 + 1+1 + 1+1 + 3+4+3 = 48
    assert len(d.sensordata) == 48, (
        f"expected 48 sensor floats, got {len(d.sensordata)}"
    )
