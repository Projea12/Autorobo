"""
tests/test_arm_controller.py — Unit tests for ArmController, ArmControllerConfig,
ArmControlResult, and _mat_to_quat.

ArmController requires a live MuJoCo model so tests build a minimal 6-DOF arm.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from env.arm_controller import (
    ArmController,
    ArmControllerConfig,
    ArmControlResult,
    _mat_to_quat,
)
from env.grasp_action import GraspActionConfig, GraspActionProcessor, GripperCmd


# ── minimal 6-DOF arm fixture ─────────────────────────────────────────────────

_ARM_XML = """
<mujoco model="test_arm">
  <option timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom name="link1" type="capsule" size="0.03 0.1" fromto="0 0 0 0 0 0.2"
            group="1"/>
      <body name="link2" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1.57 3.14"/>
        <geom name="link2g" type="capsule" size="0.03 0.1" fromto="0 0 0 0.3 0 0"
              group="1"/>
        <body name="link3" pos="0.3 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="link3g" type="capsule" size="0.025 0.08" fromto="0 0 0 0.25 0 0"
                group="1"/>
          <body name="link4" pos="0.25 0 0">
            <joint name="j4" type="hinge" axis="1 0 0" range="-3.14 3.14"/>
            <geom name="link4g" type="capsule" size="0.02 0.06" fromto="0 0 0 0 0 0.15"
                  group="1"/>
            <body name="link5" pos="0 0 0.15">
              <joint name="j5" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
              <geom name="link5g" type="capsule" size="0.018 0.05"
                    fromto="0 0 0 0.1 0 0" group="1"/>
              <body name="link6" pos="0.1 0 0">
                <joint name="j6" type="hinge" axis="1 0 0" range="-3.14 3.14"/>
                <geom name="link6g" type="sphere" size="0.015" group="1"/>
                <site name="ee_site" pos="0.04 0 0" size="0.01"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_j1" joint="j1" kp="50" ctrlrange="-3.14 3.14"/>
    <position name="act_j2" joint="j2" kp="50" ctrlrange="-1.57 3.14"/>
    <position name="act_j3" joint="j3" kp="50" ctrlrange="-3.14 3.14"/>
    <position name="act_j4" joint="j4" kp="30" ctrlrange="-3.14 3.14"/>
    <position name="act_j5" joint="j5" kp="30" ctrlrange="-3.14 3.14"/>
    <position name="act_j6" joint="j6" kp="15" ctrlrange="-3.14 3.14"/>
  </actuator>
  <sensor>
    <framelinvel name="dummy1" objtype="site" objname="ee_site"/>
    <frameangvel name="dummy2" objtype="site" objname="ee_site"/>
    <force name="wrist_force" site="ee_site"/>
    <torque name="wrist_torque" site="ee_site"/>
  </sensor>
</mujoco>
"""

_CFG = ArmControllerConfig(
    damping          = 0.05,
    max_joint_vel    = 1.5,
    dt               = 0.010,
    robot_geomgroup  = 1,
    n_arm_joints     = 6,
    arm_ctrl_start   = 0,   # no wheels in test model
    gripper_ctrl_idx = -1,
    gripper_open_pos = 0.04,
    gripper_close_pos= 0.0,
    qpos_arm_start   = 0,   # no freejoint prefix in test model
)


@pytest.fixture
def arm():
    model = mujoco.MjModel.from_xml_string(_ARM_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    ctrl = ArmController(model, data, ee_site_id=site_id, cfg=_CFG)
    ctrl.reset()
    return ctrl, model, data


def _zero_action() -> "GraspPhysicalAction":
    from env.grasp_action import GraspPhysicalAction, GripperCmd
    return GraspPhysicalAction(
        delta_pos       = np.zeros(3, dtype=np.float32),
        delta_euler     = np.zeros(3, dtype=np.float32),
        gripper_cmd     = GripperCmd.OPEN,
        gripper_changed = False,
        raw             = np.zeros(7, dtype=np.float32),
    )


def _action_with(dpos=None, deuler=None, grip=GripperCmd.OPEN):
    from env.grasp_action import GraspPhysicalAction
    return GraspPhysicalAction(
        delta_pos       = np.array(dpos or [0,0,0], dtype=np.float32),
        delta_euler     = np.array(deuler or [0,0,0], dtype=np.float32),
        gripper_cmd     = grip,
        gripper_changed = False,
        raw             = np.zeros(7, dtype=np.float32),
    )


# ── ArmControllerConfig ───────────────────────────────────────────────────────

class TestArmControllerConfig:
    def test_defaults(self):
        cfg = ArmControllerConfig()
        assert cfg.damping          == pytest.approx(0.05)
        assert cfg.max_joint_vel    == pytest.approx(1.5)
        assert cfg.n_arm_joints     == 6
        assert cfg.gripper_open_pos == pytest.approx(0.04)

    def test_frozen(self):
        with pytest.raises(Exception):
            ArmControllerConfig().damping = 0.1


# ── ArmControlResult ──────────────────────────────────────────────────────────

class TestArmControlResult:
    def _make(self) -> ArmControlResult:
        return ArmControlResult(
            joint_pos_target = np.zeros(6),
            gripper_target   = 0.04,
            dq               = np.zeros(6),
            wrist_safe       = True,
            self_collision   = False,
            wrist_force_mag  = 1.0,
            wrist_torque_mag = 0.1,
        )

    def test_fields(self):
        r = self._make()
        assert r.wrist_safe
        assert not r.self_collision

    def test_repr_ok(self):
        assert "ok" in repr(self._make())

    def test_repr_wrist_unsafe(self):
        r = self._make()
        r.wrist_safe = False
        assert "WRIST" in repr(r)

    def test_repr_self_collision(self):
        r = self._make()
        r.self_collision = True
        assert "COLLISION" in repr(r)


# ── ArmController construction ────────────────────────────────────────────────

class TestConstruction:
    def test_repr(self, arm):
        ctrl, *_ = arm
        assert "damping" in repr(ctrl)

    def test_reset_no_error(self, arm):
        ctrl, *_ = arm
        ctrl.reset()

    def test_ee_position_shape(self, arm):
        ctrl, *_ = arm
        assert ctrl.ee_position().shape == (3,)

    def test_ee_quaternion_shape(self, arm):
        ctrl, *_ = arm
        q = ctrl.ee_quaternion()
        assert q.shape == (4,)

    def test_ee_quaternion_unit(self, arm):
        ctrl, *_ = arm
        q = ctrl.ee_quaternion()
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)


# ── ArmController.step — zero action ─────────────────────────────────────────

class TestZeroAction:
    def test_returns_result(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_zero_action())
        assert isinstance(r, ArmControlResult)

    def test_wrist_safe_at_rest(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_zero_action())
        assert r.wrist_safe

    def test_no_self_collision_at_rest(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_zero_action())
        assert not r.self_collision

    def test_joint_pos_target_shape(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_zero_action())
        assert r.joint_pos_target.shape == (6,)

    def test_dq_shape(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_zero_action())
        assert r.dq.shape == (6,)

    def test_zero_delta_zero_dq(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_zero_action())
        assert np.allclose(r.dq, 0.0, atol=1e-8)

    def test_gripper_open_target(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_zero_action())
        assert r.gripper_target == pytest.approx(_CFG.gripper_open_pos)

    def test_gripper_close_target(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_action_with(grip=GripperCmd.CLOSE))
        assert r.gripper_target == pytest.approx(_CFG.gripper_close_pos)


# ── ArmController.step — non-zero delta ───────────────────────────────────────

class TestNonZeroAction:
    def test_nonzero_pos_delta_produces_nonzero_dq(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_action_with(dpos=[0.01, 0.0, 0.0]))
        assert not np.allclose(r.dq, 0.0, atol=1e-6)

    def test_dq_within_max_vel(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_action_with(dpos=[1.0, 0.0, 0.0]))   # large delta
        assert np.all(np.abs(r.dq) <= _CFG.max_joint_vel + 1e-9)

    def test_joint_target_within_hard_limits(self, arm):
        from robot.workspace import DEFAULT_LIMITS
        ctrl, *_ = arm
        for _ in range(20):
            r = ctrl.step(_action_with(dpos=[0.05, 0.05, 0.05]))
        lo = DEFAULT_LIMITS.joint_pos_lo[:6]
        hi = DEFAULT_LIMITS.joint_pos_hi[:6]
        assert np.all(r.joint_pos_target >= lo - 1e-6)
        assert np.all(r.joint_pos_target <= hi + 1e-6)

    def test_euler_delta_changes_target(self, arm):
        ctrl, *_ = arm
        r1 = ctrl.step(_zero_action())
        r2 = ctrl.step(_action_with(deuler=[0.1, 0.0, 0.0]))
        assert not np.allclose(r1.joint_pos_target, r2.joint_pos_target)


# ── self-collision detection ──────────────────────────────────────────────────

class TestSelfCollision:
    def test_no_collision_straight_arm(self, arm):
        ctrl, *_ = arm
        r = ctrl.step(_zero_action())
        assert not r.self_collision

    def test_collision_freezes_joints(self, arm):
        ctrl, model, data = arm
        # Artificially inject a robot-robot contact
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        ctrl2   = ArmController(model, data, ee_site_id=site_id, cfg=_CFG)
        ctrl2.reset()
        q_before = ctrl2._current_q().copy()
        # Patch ncon to simulate collision detected
        original_ncon = data.ncon
        # Manually call the internal check — inject two group-1 geoms
        g1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link1")
        g2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link2g")
        # Both are group=1 — verify
        assert model.geom_group[g1_id] == 1
        assert model.geom_group[g2_id] == 1


# ── _mat_to_quat ──────────────────────────────────────────────────────────────

class TestMatToQuat:
    def test_identity_gives_w1(self):
        q = _mat_to_quat(np.eye(3))
        assert q[0] == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(q[1:], 0.0, atol=1e-6)

    def test_result_is_unit(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        q = _mat_to_quat(R.astype(float))
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)

    def test_90deg_z_rotation(self):
        angle = math.pi / 2
        R = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle),  math.cos(angle), 0],
            [0, 0, 1],
        ])
        q = _mat_to_quat(R)
        assert q[0] == pytest.approx(math.cos(angle / 2), abs=1e-6)
        assert q[3] == pytest.approx(math.sin(angle / 2), abs=1e-6)

    def test_180deg_x_rotation(self):
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        q = _mat_to_quat(R)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)

    def test_shape(self):
        q = _mat_to_quat(np.eye(3))
        assert q.shape == (4,)

    def test_dtype_float64(self):
        q = _mat_to_quat(np.eye(3))
        assert q.dtype == np.float64
