"""
tests/test_rgbd_camera.py — Unit + integration tests for RGBDCamera.

Integration tests use the real NavigationEnv (which embeds a 'nav_cam')
so they exercise the full MuJoCo rendering path.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from perception.rgbd_camera import RGBDCamera, RGBDConfig, RGBDFrame, _build_K


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    from env.navigation_env import NavigationEnv
    e = NavigationEnv(max_steps=10, n_substeps=1, seed=0)
    e.reset(seed=0)
    yield e
    e.close()


@pytest.fixture(scope="module")
def camera(env):
    from env.navigation_env import NavigationEnv
    cfg = RGBDConfig(camera_name="nav_cam", width=320, height=240)
    return RGBDCamera(cfg, env._model)


@pytest.fixture(scope="module")
def frame(env, camera):
    import mujoco
    mujoco.mj_forward(env._model, env._data)
    return camera.capture(env._data, step=1)


# ── _build_K — unit tests ─────────────────────────────────────────────────────

class TestBuildK:
    def test_shape(self):
        K = _build_K(60.0, 640, 480)
        assert K.shape == (3, 3)

    def test_dtype(self):
        K = _build_K(60.0, 640, 480)
        assert K.dtype == np.float64

    def test_bottom_row(self):
        K = _build_K(60.0, 640, 480)
        np.testing.assert_array_equal(K[2], [0.0, 0.0, 1.0])

    def test_principal_point_at_image_centre(self):
        K = _build_K(60.0, 640, 480)
        assert K[0, 2] == pytest.approx(320.0)
        assert K[1, 2] == pytest.approx(240.0)

    def test_square_pixels_fx_eq_fy(self):
        K = _build_K(60.0, 640, 480)
        assert K[0, 0] == pytest.approx(K[1, 1])

    def test_fy_from_fovy(self):
        fovy = 60.0
        H = 480
        expected_fy = (H / 2.0) / math.tan(math.radians(fovy / 2.0))
        K = _build_K(fovy, 640, H)
        assert K[1, 1] == pytest.approx(expected_fy, rel=1e-9)

    def test_wider_fov_smaller_focal_length(self):
        K_narrow = _build_K(30.0, 640, 480)
        K_wide   = _build_K(90.0, 640, 480)
        assert K_wide[0, 0] < K_narrow[0, 0]

    def test_no_skew(self):
        K = _build_K(60.0, 640, 480)
        assert K[0, 1] == pytest.approx(0.0)

    def test_rectangular_image(self):
        K = _build_K(45.0, 1280, 720)
        assert K[0, 2] == pytest.approx(640.0)
        assert K[1, 2] == pytest.approx(360.0)


# ── RGBDConfig ────────────────────────────────────────────────────────────────

class TestRGBDConfig:
    def test_defaults(self):
        cfg = RGBDConfig()
        assert cfg.camera_name == "nav_cam"
        assert cfg.width  == 640
        assert cfg.height == 480
        assert cfg.min_depth == pytest.approx(0.05)
        assert cfg.max_depth == pytest.approx(8.0)
        assert cfg.depth_noise_sigma == pytest.approx(0.0)

    def test_frozen(self):
        cfg = RGBDConfig()
        with pytest.raises(Exception):
            cfg.width = 1280

    def test_custom(self):
        cfg = RGBDConfig(width=320, height=240, depth_noise_sigma=0.01)
        assert cfg.width == 320
        assert cfg.depth_noise_sigma == pytest.approx(0.01)


# ── RGBDCamera construction ───────────────────────────────────────────────────

class TestCameraConstruction:
    def test_invalid_camera_name_raises(self, env):
        with pytest.raises(ValueError, match="not found"):
            RGBDCamera(RGBDConfig(camera_name="nonexistent"), env._model)

    def test_K_shape(self, camera):
        assert camera.K.shape == (3, 3)

    def test_K_bottom_row(self, camera):
        np.testing.assert_array_equal(camera.K[2], [0.0, 0.0, 1.0])

    def test_width_height(self, camera):
        assert camera.width  == 320
        assert camera.height == 240

    def test_fovy_positive(self, camera):
        assert camera.fovy > 0.0

    def test_repr(self, camera):
        assert "nav_cam" in repr(camera)

    def test_cam_id_nonnegative(self, camera):
        assert camera.cam_id >= 0


# ── RGBDCamera intrinsics ─────────────────────────────────────────────────────

class TestCameraIntrinsics:
    def test_fx_positive(self, camera):
        assert camera.K[0, 0] > 0.0

    def test_fy_positive(self, camera):
        assert camera.K[1, 1] > 0.0

    def test_square_pixels(self, camera):
        assert camera.K[0, 0] == pytest.approx(camera.K[1, 1], rel=1e-6)

    def test_cx_near_image_centre(self, camera):
        assert abs(camera.K[0, 2] - camera.width / 2) < 1.0

    def test_cy_near_image_centre(self, camera):
        assert abs(camera.K[1, 2] - camera.height / 2) < 1.0


# ── RGBDFrame ─────────────────────────────────────────────────────────────────

class TestRGBDFrame:
    def test_rgb_shape(self, frame):
        assert frame.rgb.shape == (240, 320, 3)

    def test_rgb_dtype(self, frame):
        assert frame.rgb.dtype == np.uint8

    def test_depth_shape(self, frame):
        assert frame.depth.shape == (240, 320)

    def test_depth_dtype(self, frame):
        assert frame.depth.dtype == np.float32

    def test_K_shape(self, frame):
        assert frame.K.shape == (3, 3)

    def test_step_stored(self, frame):
        assert frame.step == 1

    def test_rgb_float_shape(self, frame):
        assert frame.rgb_float().shape == (240, 320, 3)

    def test_rgb_float_dtype(self, frame):
        assert frame.rgb_float().dtype == np.float32

    def test_rgb_float_range(self, frame):
        f = frame.rgb_float()
        assert f.min() >= 0.0
        assert f.max() <= 1.0

    def test_valid_mask_shape(self, frame):
        assert frame.valid_mask().shape == (240, 320)

    def test_valid_mask_dtype(self, frame):
        assert frame.valid_mask().dtype == bool

    def test_depth_nonnegative(self, frame):
        assert float(frame.depth.min()) >= 0.0

    def test_depth_within_range(self, frame):
        valid = frame.depth[frame.depth > 0]
        if len(valid) > 0:
            assert valid.max() <= RGBDConfig().max_depth + 1e-3

    def test_point_cloud_shape(self, frame):
        pts = frame.point_cloud()
        assert pts.ndim == 2
        assert pts.shape[1] == 3

    def test_point_cloud_dtype(self, frame):
        assert frame.point_cloud().dtype == np.float32

    def test_point_cloud_z_positive(self, frame):
        pts = frame.point_cloud()
        if len(pts) > 0:
            assert pts[:, 2].min() >= 0.0

    def test_point_cloud_count_le_pixels(self, frame):
        H, W = frame.shape
        assert len(frame.point_cloud()) <= H * W

    def test_shape_property(self, frame):
        assert frame.shape == (240, 320)

    def test_repr_contains_step(self, frame):
        assert "step=1" in repr(frame)


# ── NavigationEnv.capture_rgbd integration ───────────────────────────────────

class TestNavEnvCapture:
    def test_returns_rgbd_frame(self, env):
        from perception.rgbd_camera import RGBDFrame
        frame = env.capture_rgbd()
        assert isinstance(frame, RGBDFrame)

    def test_rgb_shape_default(self, env):
        frame = env.capture_rgbd()
        assert frame.rgb.shape == (480, 640, 3)

    def test_depth_shape_default(self, env):
        frame = env.capture_rgbd()
        assert frame.depth.shape == (480, 640)

    def test_rgb_not_all_zeros(self, env):
        frame = env.capture_rgbd()
        assert frame.rgb.sum() > 0

    def test_idempotent_capture(self, env):
        f1 = env.capture_rgbd()
        f2 = env.capture_rgbd()
        # Same state → same image
        np.testing.assert_array_equal(f1.rgb, f2.rgb)

    def test_custom_config(self, env):
        cfg   = RGBDConfig(camera_name="nav_cam", width=160, height=120)
        frame = env.capture_rgbd(cfg=cfg)
        assert frame.rgb.shape == (120, 160, 3)

    def test_step_count_stored(self, env):
        env.reset(seed=5)
        env.step(np.zeros(2))
        frame = env.capture_rgbd()
        assert frame.step == 1


# ── depth noise ───────────────────────────────────────────────────────────────

class TestDepthNoise:
    def test_noise_changes_depth(self, env):
        cfg_clean = RGBDConfig(camera_name="nav_cam", width=64, height=48,
                               depth_noise_sigma=0.0)
        cfg_noisy = RGBDConfig(camera_name="nav_cam", width=64, height=48,
                               depth_noise_sigma=0.05,
                               min_depth=0.01, max_depth=20.0)
        import mujoco
        mujoco.mj_forward(env._model, env._data)
        cam_clean = RGBDCamera(cfg_clean, env._model,
                               rng=np.random.default_rng(42))
        cam_noisy = RGBDCamera(cfg_noisy, env._model,
                               rng=np.random.default_rng(42))
        f_clean = cam_clean.capture(env._data)
        f_noisy = cam_noisy.capture(env._data)
        valid = f_clean.valid_mask() & f_noisy.valid_mask()
        if valid.sum() > 0:
            assert not np.allclose(f_clean.depth[valid], f_noisy.depth[valid])
        cam_clean.close()
        cam_noisy.close()
