"""
tests/test_nav_obs.py — Navigation observation space tests.

Strategy
────────
Most groups are tested with a synthetic MuJoCo scene (floor + a few boxes)
so the lidar rays have real geometry to intersect.  Pure-math groups
(goal, perception, occupancy from mock lidar) use numpy only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import mujoco
from env.nav_obs import (
    NavObsConfig, NavObsBuilder, PerceptionInput,
    NAV_OBS_DIM, N_RAYS, N_NEAR, GRID_N,
    SL_ROBOT, SL_GOAL, SL_LIDAR, SL_NEAR, SL_PERCEPT, SL_OCC,
    IDX_X, IDX_Y, IDX_COS_YAW, IDX_SIN_YAW, IDX_VX, IDX_OMEGA,
    IDX_PROGRESS, IDX_GOAL_DIST, IDX_GOAL_COS_BRG, IDX_GOAL_SIN_BRG,
    IDX_GOAL_REACHED,
    make_nav_obs_space, _quat_to_yaw, _wrap_angle,
    LIDAR_MAX_RANGE, GOAL_DIST_MAX, MAP_HALF,
)


# ── MuJoCo scene fixture ──────────────────────────────────────────────────────

def _build_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Minimal MjSpec: floor + 4 box obstacles + freejoint 'robot'."""
    spec = mujoco.MjSpec()
    spec.option.gravity = np.array([0.0, 0.0, -9.81])

    floor = spec.worldbody.add_geom()
    floor.name  = "floor"
    floor.type  = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size  = np.array([10.0, 10.0, 0.1])

    # four walls / obstacles to give lidar hits
    for i, (px, py) in enumerate([(2, 0), (-2, 0), (0, 2), (0, -2)]):
        box = spec.worldbody.add_geom()
        box.name = f"obstacle_{i}"
        box.type = mujoco.mjtGeom.mjGEOM_BOX
        box.pos  = np.array([px, py, 0.5])
        box.size = np.array([0.2, 0.2, 0.5])

    # robot body (base_link) with freejoint
    robot = spec.worldbody.add_body()
    robot.name = "base_link"
    robot.pos  = np.array([0.0, 0.0, 0.15])
    fj = robot.add_freejoint()
    fj.name = "root"
    # tiny sphere so it doesn't block its own rays too much
    geom = robot.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = np.array([0.05, 0, 0])
    geom.name = "robot_col"

    model = spec.compile()
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture(scope="module")
def scene():
    return _build_scene()


@pytest.fixture(scope="module")
def builder(scene):
    model, _ = scene
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    qa = int(model.jnt_qposadr[jnt_id])
    qv = int(model.jnt_dofadr[jnt_id])
    cfg = NavObsConfig(robot_body_name="base_link")
    b = NavObsBuilder(cfg=cfg, model=model, robot_qpos_adr=qa, robot_qvel_adr=qv)
    b.reset(initial_goal_dist=3.0)
    return b


def _obs(builder, scene, goal=None, perception=None) -> np.ndarray:
    _, data = scene
    mujoco.mj_forward(builder.model, data)
    g = np.array(goal or [3.0, 0.0, 0.0])
    return builder.build(data, g, perception)


# ══════════════════════════════════════════════════════════════════════════════
# Constants and config
# ══════════════════════════════════════════════════════════════════════════════

def test_obs_dim_is_128():
    assert NAV_OBS_DIM == 128


def test_n_rays_is_36():
    assert N_RAYS == 36


def test_slice_lidar_length():
    assert SL_LIDAR.stop - SL_LIDAR.start == N_RAYS


def test_slice_near_length():
    assert SL_NEAR.stop - SL_NEAR.start == N_NEAR * 3


def test_slice_occ_length():
    assert SL_OCC.stop - SL_OCC.start == GRID_N * GRID_N


def test_slices_contiguous():
    slices = [SL_ROBOT, SL_GOAL, SL_LIDAR, SL_NEAR, SL_PERCEPT, SL_OCC]
    for a, b in zip(slices, slices[1:]):
        assert a.stop == b.start


def test_slices_cover_full_obs():
    assert SL_ROBOT.start == 0
    assert SL_OCC.stop    == NAV_OBS_DIM


def test_config_obs_dim():
    cfg = NavObsConfig()
    assert cfg.obs_dim == NAV_OBS_DIM


def test_config_ray_angles_length():
    cfg = NavObsConfig()
    assert len(cfg.ray_angles_deg) == N_RAYS


def test_config_ray_angles_start_at_zero():
    cfg = NavObsConfig()
    assert cfg.ray_angles_deg[0] == pytest.approx(0.0)


def test_config_ray_angles_cover_360():
    cfg = NavObsConfig()
    assert cfg.ray_angles_deg[-1] == pytest.approx(360.0 - 360.0 / N_RAYS)


def test_config_cell_size():
    cfg = NavObsConfig(grid_n=8, grid_size_m=4.0)
    assert cfg.cell_size == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# obs shape, dtype, finiteness
# ══════════════════════════════════════════════════════════════════════════════

def test_obs_shape(builder, scene):
    obs = _obs(builder, scene)
    assert obs.shape == (NAV_OBS_DIM,)


def test_obs_dtype(builder, scene):
    obs = _obs(builder, scene)
    assert obs.dtype == np.float32


def test_obs_finite(builder, scene):
    obs = _obs(builder, scene)
    assert np.isfinite(obs).all(), "obs contains NaN or Inf"


# ══════════════════════════════════════════════════════════════════════════════
# Group A — robot state
# ══════════════════════════════════════════════════════════════════════════════

def test_robot_xy_normalised(builder, scene):
    obs = _obs(builder, scene)
    assert abs(obs[IDX_X]) <= 1.0
    assert abs(obs[IDX_Y]) <= 1.0


def test_robot_cos_sin_yaw_unit(builder, scene):
    obs = _obs(builder, scene)
    mag = math.hypot(obs[IDX_COS_YAW], obs[IDX_SIN_YAW])
    assert mag == pytest.approx(1.0, abs=1e-5)


def test_robot_vel_normalised(builder, scene):
    obs = _obs(builder, scene)
    assert abs(obs[IDX_VX])    <= 1.0
    assert abs(obs[IDX_OMEGA]) <= 1.0


def test_progress_in_unit(builder, scene):
    obs = _obs(builder, scene)
    assert 0.0 <= obs[IDX_PROGRESS] <= 1.0


def test_robot_at_origin_xy_zero(builder, scene):
    _, data = scene
    qa = builder.robot_qpos_adr
    data.qpos[qa]     = 0.0
    data.qpos[qa + 1] = 0.0
    mujoco.mj_forward(builder.model, data)
    obs = _obs(builder, scene)
    assert obs[IDX_X] == pytest.approx(0.0, abs=1e-5)
    assert obs[IDX_Y] == pytest.approx(0.0, abs=1e-5)


def test_yaw_zero_heading(builder, scene):
    _, data = scene
    qa = builder.robot_qpos_adr
    # identity quaternion → yaw = 0
    data.qpos[qa + 3] = 1.0
    data.qpos[qa + 4:qa + 7] = 0.0
    mujoco.mj_forward(builder.model, data)
    obs = _obs(builder, scene)
    assert obs[IDX_COS_YAW] == pytest.approx(1.0, abs=1e-5)
    assert obs[IDX_SIN_YAW] == pytest.approx(0.0, abs=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# Group B — goal
# ══════════════════════════════════════════════════════════════════════════════

def test_goal_dist_normalised(builder, scene):
    obs = _obs(builder, scene, goal=[3.0, 0.0, 0.0])
    assert 0.0 <= obs[IDX_GOAL_DIST] <= 1.0


def test_goal_cos_sin_bearing_unit(builder, scene):
    obs = _obs(builder, scene)
    mag = math.hypot(obs[IDX_GOAL_COS_BRG], obs[IDX_GOAL_SIN_BRG])
    assert mag == pytest.approx(1.0, abs=1e-5)


def test_goal_directly_ahead_bearing_zero(builder, scene):
    _, data = scene
    qa = builder.robot_qpos_adr
    data.qpos[qa]     = 0.0
    data.qpos[qa + 1] = 0.0
    data.qpos[qa + 3] = 1.0      # identity quat → yaw=0
    data.qpos[qa + 4:qa + 7] = 0.0
    mujoco.mj_forward(builder.model, data)
    obs = _obs(builder, scene, goal=[2.0, 0.0, 0.0])
    assert obs[IDX_GOAL_COS_BRG] == pytest.approx(1.0, abs=1e-4)
    assert obs[IDX_GOAL_SIN_BRG] == pytest.approx(0.0, abs=1e-4)


def test_goal_reached_flag_near(builder, scene):
    _, data = scene
    qa = builder.robot_qpos_adr
    data.qpos[qa]     = 0.0
    data.qpos[qa + 1] = 0.0
    mujoco.mj_forward(builder.model, data)
    # goal at origin → distance ≈ 0 → reached
    obs = _obs(builder, scene, goal=[0.05, 0.0, 0.0])
    assert obs[IDX_GOAL_REACHED] == pytest.approx(1.0)


def test_goal_reached_flag_far(builder, scene):
    obs = _obs(builder, scene, goal=[10.0, 0.0, 0.0])
    assert obs[IDX_GOAL_REACHED] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Group C — lidar ring
# ══════════════════════════════════════════════════════════════════════════════

def test_lidar_shape(builder, scene):
    obs = _obs(builder, scene)
    assert obs[SL_LIDAR].shape == (N_RAYS,)


def test_lidar_values_in_unit(builder, scene):
    obs = _obs(builder, scene)
    assert np.all(obs[SL_LIDAR] >= 0.0)
    assert np.all(obs[SL_LIDAR] <= 1.0)


def test_lidar_hits_obstacles(builder, scene):
    """Obstacles at 2m in ±x, ±y should produce readings < max range."""
    obs  = _obs(builder, scene)
    lidar = obs[SL_LIDAR]
    # at least one ray should be significantly below 1.0 (hits an obstacle)
    assert np.any(lidar < 0.9), "No lidar hits detected — expected obstacles at 2m"


def test_lidar_forward_ray_hits_closest(builder, scene):
    """Ray 0 (forward) should hit obstacle at ~2m out of 5m max → ~0.4."""
    obs   = _obs(builder, scene)
    fwd   = float(obs[SL_LIDAR.start])
    assert fwd == pytest.approx(2.0 / LIDAR_MAX_RANGE, abs=0.05)


def test_lidar_deterministic(builder, scene):
    obs1 = _obs(builder, scene)
    obs2 = _obs(builder, scene)
    assert np.allclose(obs1[SL_LIDAR], obs2[SL_LIDAR])


# ══════════════════════════════════════════════════════════════════════════════
# Group D — nearest obstacles
# ══════════════════════════════════════════════════════════════════════════════

def test_near_shape(builder, scene):
    obs = _obs(builder, scene)
    assert obs[SL_NEAR].shape == (N_NEAR * 3,)


def test_near_in_unit(builder, scene):
    obs  = _obs(builder, scene)
    near = obs[SL_NEAR]
    assert np.all(near >= -1.0)
    assert np.all(near <= 1.0)


def test_near_first_dist_smallest(builder, scene):
    """First nearest-obstacle should have smallest distance."""
    obs   = _obs(builder, scene)
    near  = obs[SL_NEAR]
    d0    = near[0]
    d1    = near[3]
    assert d0 <= d1 + 1e-6


def test_near_cos_sin_unit_norm(builder, scene):
    """Each (cos, sin) pair should have unit magnitude."""
    obs  = _obs(builder, scene)
    near = obs[SL_NEAR]
    for k in range(N_NEAR):
        c  = near[k * 3 + 1]
        s  = near[k * 3 + 2]
        assert math.hypot(c, s) == pytest.approx(1.0, abs=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# Group E — perception
# ══════════════════════════════════════════════════════════════════════════════

def test_perception_no_detection(builder, scene):
    obs = _obs(builder, scene, perception=None)
    assert np.allclose(obs[SL_PERCEPT], 0.0)


def test_perception_confidence_clipped(builder, scene):
    p   = PerceptionInput(confidence=2.0, bearing_rad=0.0, dist_est_m=1.0)
    obs = _obs(builder, scene, perception=p)
    assert obs[SL_PERCEPT.start] <= 1.0


def test_perception_zero_confidence_zeroes_out(builder, scene):
    p   = PerceptionInput(confidence=0.0, bearing_rad=1.0, dist_est_m=2.0)
    obs = _obs(builder, scene, perception=p)
    assert np.allclose(obs[SL_PERCEPT], 0.0)


def test_perception_cos_sin_unit(builder, scene):
    p   = PerceptionInput(confidence=0.9, bearing_rad=math.pi / 4, dist_est_m=1.5)
    obs = _obs(builder, scene, perception=p)
    c   = obs[SL_PERCEPT.start + 1]
    s   = obs[SL_PERCEPT.start + 2]
    assert math.hypot(c, s) == pytest.approx(1.0, abs=1e-5)


def test_perception_bearing_forward(builder, scene):
    p   = PerceptionInput(confidence=0.8, bearing_rad=0.0, dist_est_m=2.0)
    obs = _obs(builder, scene, perception=p)
    assert obs[SL_PERCEPT.start + 1] == pytest.approx(1.0, abs=1e-5)
    assert obs[SL_PERCEPT.start + 2] == pytest.approx(0.0, abs=1e-5)


def test_perception_dist_normalised(builder, scene):
    p   = PerceptionInput(confidence=0.7, bearing_rad=0.0, dist_est_m=GOAL_DIST_MAX)
    obs = _obs(builder, scene, perception=p)
    assert obs[SL_PERCEPT.start + 3] == pytest.approx(1.0, abs=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# Group F — occupancy grid
# ══════════════════════════════════════════════════════════════════════════════

def test_occ_shape(builder, scene):
    obs = _obs(builder, scene)
    assert obs[SL_OCC].shape == (GRID_N * GRID_N,)


def test_occ_values_in_set(builder, scene):
    obs  = _obs(builder, scene)
    occ  = obs[SL_OCC]
    # valid values: 0.0 (free), 0.5 (unknown), 1.0 (occupied)
    valid = np.isin(occ, [0.0, 0.5, 1.0])
    assert valid.all(), f"Unexpected occupancy values: {np.unique(occ)}"


def test_occ_has_occupied_cells(builder, scene):
    """Obstacles at 2m should mark some cells as occupied."""
    obs  = _obs(builder, scene)
    occ  = obs[SL_OCC]
    assert np.any(occ == 1.0), "No occupied cells — expected obstacles"


def test_occ_has_free_cells(builder, scene):
    obs = _obs(builder, scene)
    assert np.any(obs[SL_OCC] == 0.0)


def test_occ_deterministic(builder, scene):
    obs1 = _obs(builder, scene)
    obs2 = _obs(builder, scene)
    assert np.allclose(obs1[SL_OCC], obs2[SL_OCC])


# ══════════════════════════════════════════════════════════════════════════════
# NavObsBuilder.reset (progress tracking)
# ══════════════════════════════════════════════════════════════════════════════

def test_reset_sets_initial_dist(builder, scene):
    builder.reset(initial_goal_dist=5.0)
    assert builder._initial_goal_dist == pytest.approx(5.0)


def test_progress_zero_at_start(builder, scene):
    _, data = scene
    qa = builder.robot_qpos_adr
    data.qpos[qa]     = 0.0
    data.qpos[qa + 1] = 0.0
    mujoco.mj_forward(builder.model, data)
    builder.reset(initial_goal_dist=3.0)
    obs = _obs(builder, scene, goal=[3.0, 0.0, 0.0])
    assert obs[IDX_PROGRESS] == pytest.approx(0.0, abs=0.05)


def test_progress_one_at_goal(builder, scene):
    _, data = scene
    qa = builder.robot_qpos_adr
    data.qpos[qa]     = 0.0
    data.qpos[qa + 1] = 0.0
    mujoco.mj_forward(builder.model, data)
    builder.reset(initial_goal_dist=3.0)
    obs = _obs(builder, scene, goal=[0.01, 0.0, 0.0])
    assert obs[IDX_PROGRESS] == pytest.approx(1.0, abs=0.05)


# ══════════════════════════════════════════════════════════════════════════════
# make_nav_obs_space
# ══════════════════════════════════════════════════════════════════════════════

def test_obs_space_shape():
    space = make_nav_obs_space()
    assert space.shape == (NAV_OBS_DIM,)


def test_obs_space_dtype():
    space = make_nav_obs_space()
    assert space.dtype == np.float32


def test_obs_space_bounds():
    space = make_nav_obs_space()
    assert np.all(space.low  == -1.0)
    assert np.all(space.high ==  1.0)


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def test_quat_to_yaw_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert _quat_to_yaw(q) == pytest.approx(0.0, abs=1e-9)


def test_quat_to_yaw_90():
    # 90° rotation around z: q = [cos45, 0, 0, sin45]
    q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    assert _quat_to_yaw(q) == pytest.approx(math.pi / 2, abs=1e-6)


def test_wrap_angle_near_pi():
    assert _wrap_angle(math.pi + 0.01) == pytest.approx(-math.pi + 0.01, abs=1e-9)


def test_wrap_angle_negative():
    assert _wrap_angle(-math.pi - 0.01) == pytest.approx(math.pi - 0.01, abs=1e-9)


def test_wrap_angle_zero():
    assert _wrap_angle(0.0) == pytest.approx(0.0, abs=1e-9)
