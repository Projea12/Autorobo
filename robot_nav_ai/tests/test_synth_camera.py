"""tests/test_synth_camera.py — Pure-numpy camera math unit tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.synth.camera import (
    CameraConfig, camera_pose_from_spherical, project_points,
    bbox_2d_from_3d, yolo_xywh,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return CameraConfig(image_w=640, image_h=480, fovy=45.0)


@pytest.fixture
def cam_overhead():
    """Camera directly above origin looking straight down."""
    lookat = np.zeros(3)
    pos, R = camera_pose_from_spherical(lookat, 2.0, 0.0, -90.0)
    return pos, R


# ── CameraConfig properties ───────────────────────────────────────────────────

def test_focal_length_positive(cfg):
    assert cfg.fy > 0 and cfg.fx > 0


def test_principal_point(cfg):
    assert cfg.cx == pytest.approx(320.0)
    assert cfg.cy == pytest.approx(240.0)


def test_fx_equals_fy(cfg):
    assert cfg.fx == pytest.approx(cfg.fy)


def test_focal_length_formula(cfg):
    # fy = (H/2) / tan(fovy/2)
    expected = 240.0 / math.tan(math.radians(22.5))
    assert cfg.fy == pytest.approx(expected, rel=1e-6)


def test_K_matrix_shape(cfg):
    assert cfg.K.shape == (3, 3)


def test_K_matrix_values(cfg):
    K = cfg.K
    assert K[0, 0] == pytest.approx(cfg.fx)
    assert K[1, 1] == pytest.approx(cfg.fy)
    assert K[0, 2] == pytest.approx(cfg.cx)
    assert K[1, 2] == pytest.approx(cfg.cy)
    assert K[2, 2] == pytest.approx(1.0)


def test_sample_pose_returns_keys(cfg):
    rng  = np.random.default_rng(0)
    pose = cfg.sample_pose(rng)
    assert {"lookat", "distance", "azimuth", "elevation"} <= pose.keys()


def test_sample_pose_within_ranges(cfg):
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = cfg.sample_pose(rng)
        assert cfg.distance_range[0] <= p["distance"] <= cfg.distance_range[1]
        assert cfg.elevation_range[0] <= p["elevation"] <= cfg.elevation_range[1]
        assert 0 <= p["azimuth"] <= 360


# ── camera_pose_from_spherical ────────────────────────────────────────────────

def test_pose_camera_above_origin():
    """elevation=-90 → camera directly above lookat."""
    pos, R = camera_pose_from_spherical(np.zeros(3), 2.0, 0.0, -90.0)
    assert pos[2] == pytest.approx(2.0, abs=1e-4)
    assert abs(pos[0]) < 1e-4
    assert abs(pos[1]) < 1e-4


def test_pose_position_at_correct_distance():
    lookat = np.array([0., 0., 0.])
    pos, _ = camera_pose_from_spherical(lookat, 1.5, 30.0, -30.0)
    dist = np.linalg.norm(pos - lookat)
    assert dist == pytest.approx(1.5, rel=1e-4)


def test_pose_rotation_matrix_orthogonal():
    _, R = camera_pose_from_spherical(np.zeros(3), 1.5, 45.0, -30.0)
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_pose_rotation_matrix_det_plus_one():
    _, R = camera_pose_from_spherical(np.zeros(3), 1.5, 45.0, -30.0)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


def test_pose_different_azimuths_different_positions():
    pos0, _ = camera_pose_from_spherical(np.zeros(3), 1.5, 0.0, -30.0)
    pos90, _ = camera_pose_from_spherical(np.zeros(3), 1.5, 90.0, -30.0)
    assert not np.allclose(pos0, pos90)


def test_pose_nonzero_lookat():
    lookat = np.array([1.0, 2.0, 0.5])
    pos, _ = camera_pose_from_spherical(lookat, 1.0, 0.0, -45.0)
    dist = np.linalg.norm(pos - lookat)
    assert dist == pytest.approx(1.0, rel=1e-4)


# ── project_points ────────────────────────────────────────────────────────────

def test_origin_projects_to_image_centre(cfg, cam_overhead):
    pos, R = cam_overhead
    uv, depth = project_points(np.zeros((1, 3)), pos, R, cfg)
    assert uv[0, 0] == pytest.approx(cfg.cx, abs=1.0)
    assert uv[0, 1] == pytest.approx(cfg.cy, abs=1.0)


def test_depth_positive_for_object_in_front(cfg):
    pos, R = camera_pose_from_spherical(np.zeros(3), 1.5, 0.0, -30.0)
    _, depth = project_points(np.zeros((1, 3)), pos, R, cfg)
    assert depth[0] > 0


def test_depth_behind_camera_is_nan(cfg):
    """Point behind camera should have depth ≤ 0 → uv becomes nan."""
    pos, R = camera_pose_from_spherical(np.zeros(3), 1.5, 0.0, -30.0)
    # Place point on the opposite side of camera from lookat
    behind = pos + (pos - np.zeros(3)) * 0.5
    uv, depth = project_points(behind[np.newaxis], pos, R, cfg)
    assert depth[0] <= 0 or not np.isfinite(uv[0, 0])


def test_project_multiple_points(cfg):
    pos, R = camera_pose_from_spherical(np.zeros(3), 1.5, 45.0, -30.0)
    pts = np.random.default_rng(0).uniform(-0.5, 0.5, size=(10, 3))
    pts[:, 2] = abs(pts[:, 2])   # keep above floor
    uv, depth = project_points(pts, pos, R, cfg)
    assert uv.shape == (10, 2)
    assert depth.shape == (10,)


def test_project_points_finite_for_visible(cfg):
    pos, R = camera_pose_from_spherical(np.zeros(3), 1.5, 0.0, -45.0)
    origin = np.zeros((1, 3))
    uv, depth = project_points(origin, pos, R, cfg)
    assert np.isfinite(uv).all()
    assert np.isfinite(depth).all()


def test_project_symmetry(cfg):
    """Point at (+0.3, 0, 0) and (-0.3, 0, 0) should project symmetrically."""
    pos, R = camera_pose_from_spherical(np.zeros(3), 2.0, 0.0, -90.0)
    pts = np.array([[0.3, 0, 0], [-0.3, 0, 0]])
    uv, _ = project_points(pts, pos, R, cfg)
    assert uv[0, 0] + uv[1, 0] == pytest.approx(2 * cfg.cx, abs=1.0)


# ── bbox_2d_from_3d ───────────────────────────────────────────────────────────

def test_bbox_visible_object(cfg):
    pos, R = camera_pose_from_spherical(np.zeros(3), 1.5, 0.0, -45.0)
    centre = np.zeros(3)
    half   = np.array([0.05, 0.04, 0.07])
    bbox, depth = bbox_2d_from_3d(centre, half, pos, R, cfg)
    assert bbox is not None
    assert depth > 0


def test_bbox_xyxy_ordering(cfg):
    pos, R = camera_pose_from_spherical(np.zeros(3), 1.5, 0.0, -45.0)
    bbox, _ = bbox_2d_from_3d(np.zeros(3), np.array([0.05, 0.04, 0.07]),
                               pos, R, cfg)
    assert bbox is not None
    u0, v0, u1, v1 = bbox
    assert u1 > u0
    assert v1 > v0


def test_bbox_within_image_bounds(cfg):
    pos, R = camera_pose_from_spherical(np.zeros(3), 1.5, 0.0, -45.0)
    bbox, _ = bbox_2d_from_3d(np.zeros(3), np.array([0.05, 0.04, 0.07]),
                               pos, R, cfg)
    assert bbox is not None
    assert 0 <= bbox[0] <= cfg.image_w
    assert 0 <= bbox[2] <= cfg.image_w
    assert 0 <= bbox[1] <= cfg.image_h
    assert 0 <= bbox[3] <= cfg.image_h


def test_bbox_returns_none_for_behind_camera(cfg):
    pos, R = camera_pose_from_spherical(np.zeros(3), 1.5, 0.0, -30.0)
    # Place object behind camera
    behind = pos + (pos - np.zeros(3)) * 2.0
    bbox, _ = bbox_2d_from_3d(behind, np.array([0.05, 0.05, 0.05]),
                               pos, R, cfg)
    assert bbox is None


def test_bbox_larger_object_bigger_projection(cfg):
    pos, R = camera_pose_from_spherical(np.zeros(3), 2.0, 0.0, -90.0)
    bbox_small, _ = bbox_2d_from_3d(np.zeros(3), np.array([0.02]*3), pos, R, cfg)
    bbox_large, _ = bbox_2d_from_3d(np.zeros(3), np.array([0.20]*3), pos, R, cfg)
    assert bbox_small is not None and bbox_large is not None
    w_small = bbox_small[2] - bbox_small[0]
    w_large = bbox_large[2] - bbox_large[0]
    assert w_large > w_small


def test_bbox_closer_object_bigger_projection(cfg):
    centre = np.array([0., 0., 0.1])
    half   = np.array([0.05]*3)
    pos_near, R_near = camera_pose_from_spherical(np.zeros(3), 0.9, 0.0, -60.0)
    pos_far,  R_far  = camera_pose_from_spherical(np.zeros(3), 2.2, 0.0, -60.0)
    b_near, _ = bbox_2d_from_3d(centre, half, pos_near, R_near, cfg)
    b_far,  _ = bbox_2d_from_3d(centre, half, pos_far,  R_far,  cfg)
    assert b_near is not None and b_far is not None
    area_near = (b_near[2]-b_near[0]) * (b_near[3]-b_near[1])
    area_far  = (b_far[2]-b_far[0])   * (b_far[3]-b_far[1])
    assert area_near > area_far


# ── yolo_xywh ─────────────────────────────────────────────────────────────────

def test_yolo_xywh_centred():
    # bbox spanning full image → cx=0.5, cy=0.5, w=1.0, h=1.0
    bbox = np.array([0.0, 0.0, 640.0, 480.0])
    xywh = yolo_xywh(bbox, 640, 480)
    assert xywh == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_yolo_xywh_quarter():
    # bbox covering top-left quarter
    bbox = np.array([0.0, 0.0, 320.0, 240.0])
    xywh = yolo_xywh(bbox, 640, 480)
    assert xywh == pytest.approx([0.25, 0.25, 0.5, 0.5])


def test_yolo_xywh_values_in_unit_interval():
    rng  = np.random.default_rng(7)
    for _ in range(20):
        u0, v0 = rng.uniform(0, 300, size=2)
        u1, v1 = u0 + rng.uniform(10, 200), v0 + rng.uniform(10, 150)
        xywh   = yolo_xywh(np.array([u0, v0, u1, v1]), 640, 480)
        assert np.all((xywh >= 0) & (xywh <= 1.0))
