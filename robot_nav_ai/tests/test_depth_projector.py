"""
tests/test_depth_projector.py — Unit tests for DepthProjector, ProjectorConfig,
and ProjectionResult.

Uses synthetic RGBDFrame objects — no MuJoCo / hardware required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from perception.rgbd_camera import RGBDFrame, _build_K
from perception.detector import Detection
from perception.depth_projector import (
    DepthProjector, ProjectorConfig, ProjectionResult,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_frame(H: int = 60, W: int = 80, z_val: float = 2.0,
                fovy: float = 60.0) -> RGBDFrame:
    """Synthetic RGBDFrame with uniform depth z_val."""
    rgb   = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), z_val, dtype=np.float32)
    K     = _build_K(fovy, W, H)
    return RGBDFrame(rgb=rgb, depth=depth, K=K, step=0)


def _make_det(x1: float = 10, y1: float = 10,
              x2: float = 70, y2: float = 50) -> Detection:
    return Detection(
        class_id   = 0,
        class_name = "mug",
        confidence = 0.9,
        bbox_xyxy  = np.array([x1, y1, x2, y2], dtype=np.float32),
        bbox_xywh  = np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1],
                               dtype=np.float32),
    )


# ── ProjectorConfig ───────────────────────────────────────────────────────────

class TestProjectorConfig:
    def test_defaults(self):
        cfg = ProjectorConfig()
        assert cfg.min_valid_pixels == 5
        assert cfg.z_min            == pytest.approx(0.05)
        assert cfg.z_max            == pytest.approx(10.0)
        assert cfg.bbox_shrink      == pytest.approx(0.15)

    def test_frozen(self):
        cfg = ProjectorConfig()
        with pytest.raises(Exception):
            cfg.min_valid_pixels = 99

    def test_custom(self):
        cfg = ProjectorConfig(min_valid_pixels=3, bbox_shrink=0.0)
        assert cfg.min_valid_pixels == 3
        assert cfg.bbox_shrink      == pytest.approx(0.0)


# ── ProjectionResult ──────────────────────────────────────────────────────────

class TestProjectionResult:
    def _make(self, method: str = "bbox_median") -> ProjectionResult:
        return ProjectionResult(
            xyz      = np.array([0.1, 0.2, 2.0], dtype=np.float32),
            std      = np.array([0.01, 0.01, 0.05], dtype=np.float32),
            n_points = 100,
            method   = method,
        )

    def test_repr_contains_z(self):
        r = self._make()
        assert "2.000" in repr(r)

    def test_repr_contains_method(self):
        r = self._make("mask_median")
        assert "mask_median" in repr(r)

    def test_n_points_stored(self):
        r = self._make()
        assert r.n_points == 100

    def test_method_stored(self):
        r = self._make("center")
        assert r.method == "center"

    def test_xyz_shape(self):
        assert self._make().xyz.shape == (3,)

    def test_std_shape(self):
        assert self._make().std.shape == (3,)


# ── _backproject (static) ─────────────────────────────────────────────────────

class TestBackproject:
    def test_center_pixel_x_zero(self):
        K = _build_K(60.0, 80, 60)
        xyz, _ = DepthProjector._backproject(K[0, 2], K[1, 2], 3.0, 0.0, K)
        assert xyz[0] == pytest.approx(0.0, abs=1e-6)

    def test_center_pixel_y_zero(self):
        K = _build_K(60.0, 80, 60)
        xyz, _ = DepthProjector._backproject(K[0, 2], K[1, 2], 3.0, 0.0, K)
        assert xyz[1] == pytest.approx(0.0, abs=1e-6)

    def test_z_propagated(self):
        K = _build_K(60.0, 80, 60)
        xyz, _ = DepthProjector._backproject(K[0, 2], K[1, 2], 3.0, 0.0, K)
        assert xyz[2] == pytest.approx(3.0)

    def test_right_of_center_positive_x(self):
        K = _build_K(60.0, 80, 60)
        xyz, _ = DepthProjector._backproject(K[0, 2] + 10.0, K[1, 2], 2.0, 0.0, K)
        assert xyz[0] > 0.0

    def test_below_center_positive_y(self):
        K = _build_K(60.0, 80, 60)
        xyz, _ = DepthProjector._backproject(K[0, 2], K[1, 2] + 10.0, 2.0, 0.0, K)
        assert xyz[1] > 0.0

    def test_zero_std_gives_zero_uncertainty(self):
        K = _build_K(60.0, 80, 60)
        _, std = DepthProjector._backproject(K[0, 2], K[1, 2], 2.0, 0.0, K)
        np.testing.assert_array_almost_equal(std, [0.0, 0.0, 0.0])

    def test_std_z_equals_z_std(self):
        K = _build_K(60.0, 80, 60)
        _, std = DepthProjector._backproject(K[0, 2], K[1, 2], 2.0, 0.5, K)
        assert std[2] == pytest.approx(0.5)

    def test_output_dtype_float32(self):
        K = _build_K(60.0, 80, 60)
        xyz, std = DepthProjector._backproject(K[0, 2], K[1, 2], 2.0, 0.0, K)
        assert xyz.dtype == np.float32
        assert std.dtype == np.float32

    def test_x_proportional_to_offset(self):
        K  = _build_K(60.0, 80, 60)
        du = 5.0;  z = 3.0
        xyz, _ = DepthProjector._backproject(K[0, 2] + du, K[1, 2], z, 0.0, K)
        assert xyz[0] == pytest.approx(du * z / float(K[0, 0]), rel=1e-5)


# ── bbox_median mode ──────────────────────────────────────────────────────────

class TestBboxMedian:
    def test_returns_projection_result(self):
        r = DepthProjector().project(_make_det(), _make_frame())
        assert isinstance(r, ProjectionResult)

    def test_method_is_bbox_median(self):
        r = DepthProjector().project(_make_det(), _make_frame(z_val=2.0))
        assert r.method == "bbox_median"

    def test_z_matches_uniform_depth(self):
        z_val = 3.5
        r = DepthProjector().project(_make_det(), _make_frame(z_val=z_val))
        assert r.xyz[2] == pytest.approx(z_val, rel=1e-4)

    def test_xyz_shape(self):
        r = DepthProjector().project(_make_det(), _make_frame())
        assert r.xyz.shape == (3,)

    def test_std_shape(self):
        r = DepthProjector().project(_make_det(), _make_frame())
        assert r.std.shape == (3,)

    def test_xyz_dtype_float32(self):
        r = DepthProjector().project(_make_det(), _make_frame())
        assert r.xyz.dtype == np.float32

    def test_std_dtype_float32(self):
        r = DepthProjector().project(_make_det(), _make_frame())
        assert r.std.dtype == np.float32

    def test_n_points_positive(self):
        r = DepthProjector().project(_make_det(), _make_frame(z_val=2.0))
        assert r.n_points > 0

    def test_uniform_depth_std_z_zero(self):
        r = DepthProjector().project(_make_det(), _make_frame(z_val=2.0))
        assert r.std[2] == pytest.approx(0.0, abs=1e-6)

    def test_bbox_centred_on_principal_point_xy_zero(self):
        H, W = 60, 80
        frame = _make_frame(H=H, W=W, z_val=2.0)
        # bbox centred exactly on principal point cx=40, cy=30
        det = _make_det(x1=30, y1=20, x2=50, y2=40)
        r   = DepthProjector(ProjectorConfig(bbox_shrink=0.0)).project(det, frame)
        assert r.xyz[0] == pytest.approx(0.0, abs=1e-3)
        assert r.xyz[1] == pytest.approx(0.0, abs=1e-3)

    def test_invalid_depth_excluded(self):
        H, W  = 60, 80
        frame = _make_frame(H=H, W=W, z_val=2.0)
        frame.depth[:, :] = 0.0        # all invalid
        frame.depth[20:40, 20:60] = 2.0  # only inner strip valid
        det = _make_det(x1=10, y1=10, x2=70, y2=50)
        r   = DepthProjector(ProjectorConfig(bbox_shrink=0.0)).project(det, frame)
        assert r.n_points == 20 * 40


# ── centre-pixel fallback ─────────────────────────────────────────────────────

class TestCenterFallback:
    def test_method_center_all_invalid(self):
        frame = _make_frame(z_val=0.0)   # all zeros → invalid
        r     = DepthProjector().project(_make_det(), frame)
        assert r.method == "center"

    def test_n_points_zero_when_center_depth_zero(self):
        frame = _make_frame(z_val=0.0)
        r     = DepthProjector().project(_make_det(), frame)
        assert r.n_points == 0

    def test_method_center_when_few_valid_pixels(self):
        frame = _make_frame(z_val=0.0)
        # Only one pixel valid — below min_valid_pixels=5
        frame.depth[30, 40] = 2.0
        r = DepthProjector(ProjectorConfig(min_valid_pixels=5)).project(
            _make_det(), frame)
        assert r.method == "center"

    def test_center_z_from_single_pixel(self):
        frame = _make_frame(z_val=0.0)
        frame.depth[30, 40] = 1.5   # center of 60×80 image
        r = DepthProjector(ProjectorConfig(min_valid_pixels=5)).project(
            _make_det(x1=20, y1=20, x2=60, y2=40), frame)
        # Fallback reads center pixel at (40, 30)
        assert r.xyz[2] == pytest.approx(1.5, abs=0.5)


# ── mask_median mode ──────────────────────────────────────────────────────────

class TestMaskMedian:
    def _mask(self, H=60, W=80):
        m = np.zeros((H, W), dtype=bool)
        m[10:50, 10:70] = True
        return m

    def test_method_mask_median(self):
        frame = _make_frame(z_val=2.0)
        r     = DepthProjector().project(_make_det(), frame, mask=self._mask())
        assert r.method == "mask_median"

    def test_z_matches_uniform_depth(self):
        z_val = 1.8
        frame = _make_frame(z_val=z_val)
        r     = DepthProjector().project(_make_det(), frame, mask=self._mask())
        assert r.xyz[2] == pytest.approx(z_val, rel=1e-4)

    def test_n_points_equals_valid_mask_pixels(self):
        frame = _make_frame(z_val=2.0)
        mask  = np.zeros((60, 80), dtype=bool)
        mask[10:30, 10:50] = True   # 20 × 40 = 800
        r = DepthProjector().project(_make_det(), frame, mask=mask)
        assert r.n_points == 800

    def test_empty_mask_falls_back(self):
        frame = _make_frame(z_val=2.0)
        mask  = np.zeros((60, 80), dtype=bool)   # all False
        r     = DepthProjector().project(_make_det(), frame, mask=mask)
        assert r.method in ("bbox_median", "center")

    def test_mask_centroid_used_for_xy(self):
        H, W  = 60, 80
        frame = _make_frame(H=H, W=W, z_val=2.0)
        # Mask centred exactly on principal point cx=40, cy=30
        # [30:51] → columns 30..50, mean=40.0; [20:41] → rows 20..40, mean=30.0
        mask  = np.zeros((H, W), dtype=bool)
        mask[20:41, 30:51] = True
        r = DepthProjector(ProjectorConfig(min_valid_pixels=1)).project(
            _make_det(), frame, mask=mask)
        assert r.xyz[0] == pytest.approx(0.0, abs=1e-3)
        assert r.xyz[1] == pytest.approx(0.0, abs=1e-3)


# ── project_batch ─────────────────────────────────────────────────────────────

class TestProjectBatch:
    def test_returns_list(self):
        frame = _make_frame(z_val=2.0)
        dets  = [_make_det(), _make_det(x1=5, y1=5, x2=40, y2=35)]
        r     = DepthProjector().project_batch(dets, frame)
        assert isinstance(r, list)

    def test_length_matches_input(self):
        frame = _make_frame(z_val=2.0)
        dets  = [_make_det() for _ in range(5)]
        assert len(DepthProjector().project_batch(dets, frame)) == 5

    def test_empty_input_returns_empty(self):
        assert DepthProjector().project_batch([], _make_frame()) == []

    def test_elements_are_projection_results(self):
        frame = _make_frame(z_val=2.0)
        r     = DepthProjector().project_batch([_make_det()], frame)
        assert isinstance(r[0], ProjectionResult)

    def test_uses_detection_mask(self):
        frame = _make_frame(z_val=2.0)
        det   = _make_det()
        mask  = np.zeros((60, 80), dtype=bool)
        mask[10:50, 10:70] = True
        det.mask = mask
        r = DepthProjector().project_batch([det], frame)
        assert r[0].method == "mask_median"


# ── annotate_detections ───────────────────────────────────────────────────────

class TestAnnotateDetections:
    def test_sets_position_3d_on_valid_detection(self):
        frame = _make_frame(z_val=2.0)
        det   = _make_det()
        assert det.position_3d is None
        DepthProjector().annotate_detections([det], frame)
        assert det.position_3d is not None

    def test_position_3d_shape(self):
        frame = _make_frame(z_val=2.0)
        det   = _make_det()
        DepthProjector().annotate_detections([det], frame)
        assert det.position_3d.shape == (3,)

    def test_position_3d_dtype_float32(self):
        frame = _make_frame(z_val=2.0)
        det   = _make_det()
        DepthProjector().annotate_detections([det], frame)
        assert det.position_3d.dtype == np.float32

    def test_no_position_3d_when_zero_depth(self):
        frame = _make_frame(z_val=0.0)
        det   = _make_det()
        DepthProjector().annotate_detections([det], frame)
        assert det.position_3d is None

    def test_returns_projection_results(self):
        frame   = _make_frame(z_val=2.0)
        det     = _make_det()
        results = DepthProjector().annotate_detections([det], frame)
        assert isinstance(results[0], ProjectionResult)

    def test_annotates_multiple_detections(self):
        frame = _make_frame(z_val=2.0)
        dets  = [_make_det(), _make_det(x1=5, y1=5, x2=40, y2=35)]
        DepthProjector().annotate_detections(dets, frame)
        for d in dets:
            assert d.position_3d is not None

    def test_empty_list_returns_empty(self):
        frame   = _make_frame()
        results = DepthProjector().annotate_detections([], frame)
        assert results == []
