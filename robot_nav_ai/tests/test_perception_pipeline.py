"""
tests/test_perception_pipeline.py — Integration tests for the full perception
pipeline under realistic and adversarial conditions.

Scenarios covered
─────────────────
  Known object, normal lighting       → expect ACT
  Partial occlusion (small mask)      → seg_score reduced
  Novel / unknown object              → low det_score → GATHER or FLAG
  Low light (dark RGB frame)          → pipeline handles gracefully
  Near distance (z ≈ 0.3 m)          → valid depth, high coverage
  Far distance  (z ≈ 5.0 m)          → valid depth, higher std possible
  Zero depth (no LiDAR return)        → require_depth=True → FLAG
  No SAM mask (novel scene)           → seg_score=0; require_seg=False → ACT
  Multiple objects (mixed quality)    → worst object determines scene decision
  Full pipeline integration           → DepthProjector → SceneAggregator → Gate

All tests use synthetic RGBDFrame objects so no MuJoCo / GPU is required.
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
from perception.depth_projector import DepthProjector, ProjectorConfig, ProjectionResult
from perception.confidence import AggregatorConfig, ObjectConfidence, SceneAggregator, SceneConfidence
from perception.uncertainty_gate import GateConfig, GateDecision, GateResult, UncertaintyGate


# ── shared helpers ────────────────────────────────────────────────────────────

_H, _W = 480, 640
_K     = _build_K(60.0, _W, _H)


def _frame(z: float = 2.0, brightness: int = 128) -> RGBDFrame:
    """Synthetic RGBDFrame with uniform depth z and constant RGB brightness."""
    rgb   = np.full((_H, _W, 3), brightness, dtype=np.uint8)
    depth = np.full((_H, _W), z, dtype=np.float32)
    return RGBDFrame(rgb=rgb, depth=depth, K=_K, step=0)


def _det(conf: float = 0.9, x1=100, y1=100, x2=540, y2=380,
         mask: np.ndarray = None) -> Detection:
    """Synthetic Detection with given confidence and bbox."""
    d = Detection(
        class_id   = 0,
        class_name = "025_mug",
        confidence = conf,
        bbox_xyxy  = np.array([x1, y1, x2, y2], dtype=np.float32),
        bbox_xywh  = np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1],
                               dtype=np.float32),
    )
    if mask is not None:
        d.mask = mask
    return d


def _full_mask(frac: float = 1.0) -> np.ndarray:
    """Boolean mask covering `frac` of the detection area [100:380, 100:540]."""
    mask = np.zeros((_H, _W), dtype=bool)
    row_h = int(280 * frac)
    mask[100:100 + row_h, 100:540] = True
    return mask


def _run(det: Detection, frame: RGBDFrame, *,
         agg_cfg: AggregatorConfig = None,
         gate_cfg: GateConfig = None) -> GateResult:
    """Project, aggregate, and gate-evaluate a single detection."""
    agg  = SceneAggregator(agg_cfg or AggregatorConfig())
    gate = UncertaintyGate(gate_cfg or GateConfig())
    proj = DepthProjector().project(det, frame, mask=det.mask)
    scene = agg.aggregate([det], [proj])
    return gate.evaluate(scene)


# ── known object, normal conditions ──────────────────────────────────────────

class TestKnownObjectNormal:
    """High confidence, good depth, full SAM mask → expect ACT."""

    def _setup(self):
        mask  = _full_mask(1.0)
        det   = _det(conf=0.92, mask=mask)
        frame = _frame(z=2.0, brightness=128)
        gate_cfg = GateConfig(act_threshold=0.65, require_depth=True,
                              require_seg=False)
        return _run(det, frame, gate_cfg=gate_cfg)

    def test_decision_is_act(self):
        assert self._setup().decision == GateDecision.ACT

    def test_score_above_act_threshold(self):
        r = self._setup()
        assert r.score >= 0.65

    def test_one_object_decision(self):
        assert len(self._setup().object_decisions) == 1

    def test_object_decision_is_act(self):
        _, obj_dec = self._setup().object_decisions[0]
        assert obj_dec == GateDecision.ACT

    def test_depth_score_nonzero(self):
        mask  = _full_mask(1.0)
        det   = _det(conf=0.92, mask=mask)
        frame = _frame(z=2.0)
        proj  = DepthProjector().project(det, frame)
        scene = SceneAggregator().aggregate([det], [proj])
        assert scene.objects[0].depth_score > 0.0

    def test_seg_score_near_one(self):
        mask  = _full_mask(1.0)
        det   = _det(conf=0.92, mask=mask)
        frame = _frame(z=2.0)
        proj  = DepthProjector().project(det, frame)
        scene = SceneAggregator().aggregate([det], [proj])
        assert scene.objects[0].seg_score >= 0.9


# ── partial occlusion ─────────────────────────────────────────────────────────

class TestPartialOcclusion:
    """SAM mask covers only 30% of bbox area (occluded object)."""

    def _scene(self, frac=0.30):
        mask  = _full_mask(frac)
        det   = _det(conf=0.85, mask=mask)
        frame = _frame(z=1.5)
        proj  = DepthProjector().project(det, frame, mask=mask)
        return SceneAggregator().aggregate([det], [proj])

    def test_seg_score_proportional_to_mask(self):
        scene = self._scene(frac=0.30)
        assert scene.objects[0].seg_score < 0.35

    def test_seg_score_lower_than_full_mask(self):
        full_scene = self._scene(frac=1.0)
        part_scene = self._scene(frac=0.30)
        assert part_scene.objects[0].seg_score < full_scene.objects[0].seg_score

    def test_combined_lower_than_full_mask(self):
        full_scene = self._scene(frac=1.0)
        part_scene = self._scene(frac=0.30)
        assert part_scene.objects[0].combined < full_scene.objects[0].combined

    def test_depth_still_valid(self):
        scene = self._scene(frac=0.30)
        assert scene.objects[0].depth_score > 0.0

    def test_projection_method_mask_median(self):
        mask  = _full_mask(0.30)
        det   = _det(conf=0.85, mask=mask)
        frame = _frame(z=1.5)
        proj  = DepthProjector().project(det, frame, mask=mask)
        assert proj.method == "mask_median"

    def test_half_occlusion_seg_around_half(self):
        scene = self._scene(frac=0.50)
        s     = scene.objects[0].seg_score
        assert 0.45 <= s <= 0.55


# ── novel / unknown object ────────────────────────────────────────────────────

class TestNovelObject:
    """Low YOLO confidence simulates a novel or out-of-distribution object."""

    def _scene(self, conf):
        det   = _det(conf=conf)
        frame = _frame(z=1.5)
        proj  = DepthProjector().project(det, frame)
        return SceneAggregator(
            AggregatorConfig(w_detection=0.5, w_depth=0.3, w_seg=0.2)
        ).aggregate([det], [proj])

    def test_very_low_conf_combined_below_mid(self):
        scene = self._scene(conf=0.15)
        assert scene.objects[0].combined < 0.5

    def test_low_conf_gates_to_gather_or_flag(self):
        det   = _det(conf=0.20)
        frame = _frame(z=1.5)
        r     = _run(det, frame,
                     gate_cfg=GateConfig(act_threshold=0.70, gather_threshold=0.40,
                                         require_depth=True))
        assert r.decision in (GateDecision.GATHER, GateDecision.FLAG)

    def test_moderate_conf_not_act(self):
        scene = self._scene(conf=0.35)
        assert scene.global_score < 0.70

    def test_det_score_matches_confidence(self):
        scene = self._scene(conf=0.22)
        assert scene.objects[0].detection_score == pytest.approx(0.22)

    def test_high_conf_known_vs_low_conf_novel(self):
        known = self._scene(conf=0.90)
        novel = self._scene(conf=0.20)
        assert known.global_score > novel.global_score


# ── low light ─────────────────────────────────────────────────────────────────

class TestLowLight:
    """Dark image (brightness ≈ 5/255). Pipeline should handle gracefully."""

    def test_pipeline_handles_dark_frame(self):
        det   = _det(conf=0.80)
        frame = _frame(z=2.0, brightness=5)   # nearly black
        result = _run(det, frame,
                      gate_cfg=GateConfig(require_depth=True))
        assert isinstance(result, GateResult)

    def test_depth_unaffected_by_brightness(self):
        det        = _det(conf=0.80)
        frame_dark = _frame(z=2.0, brightness=5)
        frame_norm = _frame(z=2.0, brightness=128)
        proj_dark  = DepthProjector().project(det, frame_dark)
        proj_norm  = DepthProjector().project(det, frame_norm)
        assert proj_dark.xyz[2] == pytest.approx(proj_norm.xyz[2], rel=1e-3)

    def test_low_light_simulated_via_conf_drop(self):
        # In real deployment, low light → lower YOLO conf; simulate here
        det   = _det(conf=0.35)   # low conf due to poor lighting
        frame = _frame(z=2.0, brightness=5)
        r     = _run(det, frame,
                     gate_cfg=GateConfig(act_threshold=0.70, require_depth=True))
        assert r.decision in (GateDecision.GATHER, GateDecision.FLAG)

    def test_rgb_all_dark(self):
        frame = _frame(brightness=5)
        assert frame.rgb.max() < 10

    def test_no_exception_on_all_zeros_rgb(self):
        det   = _det(conf=0.90)
        frame = _frame(z=2.0, brightness=0)
        _run(det, frame, gate_cfg=GateConfig(require_depth=True))


# ── near distance ─────────────────────────────────────────────────────────────

class TestNearDistance:
    """Object very close to camera (z ≈ 0.3 m)."""

    def test_near_depth_projected_correctly(self):
        det   = _det(conf=0.88)
        frame = _frame(z=0.30)
        proj  = DepthProjector().project(det, frame)
        assert proj.xyz[2] == pytest.approx(0.30, rel=1e-3)

    def test_near_depth_score_nonzero(self):
        det   = _det(conf=0.88)
        frame = _frame(z=0.30)
        proj  = DepthProjector().project(det, frame)
        scene = SceneAggregator().aggregate([det], [proj])
        assert scene.objects[0].depth_score > 0.0

    def test_near_uniform_depth_std_zero(self):
        det   = _det(conf=0.88)
        frame = _frame(z=0.30)
        proj  = DepthProjector().project(det, frame)
        assert proj.std[2] == pytest.approx(0.0, abs=1e-5)

    def test_near_n_points_large(self):
        det   = _det(conf=0.88)
        frame = _frame(z=0.30)
        proj  = DepthProjector().project(det, frame)
        assert proj.n_points > 50

    def test_near_can_act(self):
        det   = _det(conf=0.90)
        frame = _frame(z=0.30)
        r     = _run(det, frame,
                     gate_cfg=GateConfig(act_threshold=0.60, require_depth=True))
        assert r.decision == GateDecision.ACT


# ── far distance ──────────────────────────────────────────────────────────────

class TestFarDistance:
    """Object far from camera (z = 5.0 m)."""

    def test_far_depth_projected_correctly(self):
        det   = _det(conf=0.88)
        frame = _frame(z=5.0)
        proj  = DepthProjector().project(det, frame)
        assert proj.xyz[2] == pytest.approx(5.0, rel=1e-3)

    def test_far_depth_score_nonzero(self):
        det   = _det(conf=0.88)
        frame = _frame(z=5.0)
        proj  = DepthProjector().project(det, frame)
        scene = SceneAggregator().aggregate([det], [proj])
        assert scene.objects[0].depth_score > 0.0

    def test_xyz_z_increases_with_distance(self):
        det    = _det(conf=0.88)
        proj_n = DepthProjector().project(det, _frame(z=0.5))
        proj_f = DepthProjector().project(det, _frame(z=5.0))
        assert proj_f.xyz[2] > proj_n.xyz[2]

    def test_far_within_projector_z_max(self):
        det   = _det(conf=0.88)
        frame = _frame(z=5.0)
        proj  = DepthProjector(ProjectorConfig(z_max=10.0)).project(det, frame)
        assert proj.n_points > 0

    def test_beyond_z_max_gives_no_points(self):
        det   = _det(conf=0.88)
        frame = _frame(z=15.0)
        proj  = DepthProjector(ProjectorConfig(z_max=10.0)).project(det, frame)
        assert proj.n_points == 0


# ── zero depth (no valid LiDAR / sensor return) ───────────────────────────────

class TestZeroDepth:
    """All depth pixels are zero — sensor returned nothing valid."""

    def test_projection_n_points_zero(self):
        det   = _det(conf=0.90)
        frame = _frame(z=0.0)
        proj  = DepthProjector().project(det, frame)
        assert proj.n_points == 0

    def test_depth_score_is_zero(self):
        det   = _det(conf=0.90)
        frame = _frame(z=0.0)
        proj  = DepthProjector().project(det, frame)
        scene = SceneAggregator().aggregate([det], [proj])
        assert scene.objects[0].depth_score == pytest.approx(0.0)

    def test_require_depth_flags(self):
        det   = _det(conf=0.90)
        frame = _frame(z=0.0)
        r     = _run(det, frame,
                     gate_cfg=GateConfig(require_depth=True, act_threshold=0.65))
        assert r.decision == GateDecision.FLAG

    def test_no_require_depth_can_act(self):
        # High-confidence detection without depth can still ACT if threshold met
        det   = _det(conf=0.90)
        frame = _frame(z=0.0)
        agg_cfg  = AggregatorConfig(w_detection=1.0, w_depth=0.0, w_seg=0.0)
        gate_cfg = GateConfig(act_threshold=0.80, require_depth=False)
        r     = _run(det, frame, agg_cfg=agg_cfg, gate_cfg=gate_cfg)
        assert r.decision == GateDecision.ACT

    def test_projection_method_center(self):
        det   = _det(conf=0.90)
        frame = _frame(z=0.0)
        proj  = DepthProjector().project(det, frame)
        assert proj.method == "center"


# ── no SAM mask (novel / first-seen scene) ────────────────────────────────────

class TestNoSAMMask:
    """Detection has no SAM mask — seg_score=0."""

    def test_seg_score_is_zero(self):
        det   = _det(conf=0.85, mask=None)
        frame = _frame(z=2.0)
        proj  = DepthProjector().project(det, frame)
        scene = SceneAggregator().aggregate([det], [proj])
        assert scene.objects[0].seg_score == pytest.approx(0.0)

    def test_no_require_seg_can_still_act(self):
        det   = _det(conf=0.90, mask=None)
        frame = _frame(z=2.0)
        r     = _run(det, frame,
                     gate_cfg=GateConfig(act_threshold=0.60, require_seg=False,
                                         require_depth=True))
        assert r.decision == GateDecision.ACT

    def test_require_seg_flags(self):
        det   = _det(conf=0.90, mask=None)
        frame = _frame(z=2.0)
        r     = _run(det, frame,
                     gate_cfg=GateConfig(require_seg=True, require_depth=False))
        assert r.decision == GateDecision.FLAG

    def test_combined_lower_than_with_mask(self):
        frame    = _frame(z=2.0)
        no_mask  = _det(conf=0.90, mask=None)
        has_mask = _det(conf=0.90, mask=_full_mask(1.0))
        proj_n   = DepthProjector().project(no_mask, frame)
        proj_m   = DepthProjector().project(has_mask, frame, mask=has_mask.mask)
        scene_n  = SceneAggregator().aggregate([no_mask],  [proj_n])
        scene_m  = SceneAggregator().aggregate([has_mask], [proj_m])
        assert scene_n.global_score < scene_m.global_score


# ── multiple objects (mixed quality) ─────────────────────────────────────────

class TestMultipleObjects:
    """Scene with one high-quality and one low-quality detection."""

    def _mixed_scene(self):
        frame    = _frame(z=2.0)
        det_good = _det(conf=0.92, mask=_full_mask(1.0))
        det_bad  = _det(conf=0.20, x1=10, y1=10, x2=50, y2=40)   # small bbox, low conf
        proj_g   = DepthProjector().project(det_good, frame, mask=det_good.mask)
        proj_b   = DepthProjector().project(det_bad,  frame)
        return SceneAggregator().aggregate([det_good, det_bad], [proj_g, proj_b])

    def test_two_objects_detected(self):
        assert self._mixed_scene().n_objects == 2

    def test_global_score_between_best_and_worst(self):
        scene = self._mixed_scene()
        best  = max(o.combined for o in scene.objects)
        worst = min(o.combined for o in scene.objects)
        assert worst <= scene.global_score <= best

    def test_worst_object_lowers_scene_decision(self):
        # One object has zero depth → FLAG with require_depth=True
        frame    = _frame(z=2.0)
        det_good = _det(conf=0.92)
        det_bad  = _det(conf=0.20, x1=10, y1=10, x2=50, y2=40)
        proj_g   = DepthProjector().project(det_good, frame)
        proj_b   = DepthProjector().project(det_bad,  _frame(z=0.0))

        agg  = SceneAggregator()
        gate = UncertaintyGate(GateConfig(act_threshold=0.60, require_depth=True))
        scene = agg.aggregate([det_good, det_bad], [proj_g, proj_b])
        r     = gate.evaluate(scene)
        assert r.decision == GateDecision.FLAG

    def test_object_decisions_count(self):
        frame = _frame(z=2.0)
        dets  = [_det(conf=0.9), _det(conf=0.7, x1=50, y1=50, x2=200, y2=200)]
        projs = [DepthProjector().project(d, frame) for d in dets]
        scene = SceneAggregator().aggregate(dets, projs)
        gate  = UncertaintyGate(GateConfig(require_depth=False))
        r     = gate.evaluate(scene)
        assert len(r.object_decisions) == 2


# ── full pipeline integration ─────────────────────────────────────────────────

class TestFullPipelineIntegration:
    """DepthProjector → SceneAggregator → UncertaintyGate end-to-end."""

    def _pipeline(self, z, conf, mask_frac=1.0,
                  agg_cfg=None, gate_cfg=None):
        frame = _frame(z=z)
        mask  = _full_mask(mask_frac) if mask_frac > 0 else None
        det   = _det(conf=conf, mask=mask)
        proj  = DepthProjector().project(det, frame,
                                         mask=mask if mask is not None else None)
        agg   = SceneAggregator(agg_cfg or AggregatorConfig())
        gate  = UncertaintyGate(gate_cfg or GateConfig())
        scene = agg.aggregate([det], [proj])
        return gate.evaluate(scene), scene

    def test_gate_result_is_gate_result(self):
        r, _ = self._pipeline(z=2.0, conf=0.9)
        assert isinstance(r, GateResult)

    def test_scene_confidence_is_scene_confidence(self):
        _, s = self._pipeline(z=2.0, conf=0.9)
        assert isinstance(s, SceneConfidence)

    def test_position_3d_set_on_detection(self):
        frame = _frame(z=2.0)
        det   = _det(conf=0.88)
        proj  = DepthProjector().project(det, frame)
        DepthProjector().annotate_detections([det], frame)
        assert det.position_3d is not None

    def test_position_3d_z_matches_frame_depth(self):
        frame = _frame(z=3.0)
        det   = _det(conf=0.88)
        DepthProjector().annotate_detections([det], frame)
        assert det.position_3d[2] == pytest.approx(3.0, rel=1e-3)

    def test_high_quality_scene_acts(self):
        r, _ = self._pipeline(z=1.5, conf=0.92, mask_frac=1.0,
                               gate_cfg=GateConfig(act_threshold=0.60,
                                                   require_depth=True))
        assert r.decision == GateDecision.ACT

    def test_low_quality_scene_gathers_or_flags(self):
        r, _ = self._pipeline(z=2.0, conf=0.25, mask_frac=0.0,
                               gate_cfg=GateConfig(act_threshold=0.70,
                                                   gather_threshold=0.40,
                                                   require_depth=False))
        assert r.decision in (GateDecision.GATHER, GateDecision.FLAG)

    def test_depth_score_increases_with_more_valid_pixels(self):
        # Larger bbox → more valid depth pixels → higher depth coverage
        frame   = _frame(z=2.0)
        det_big = _det(conf=0.9, x1=10, y1=10, x2=630, y2=470)
        det_sml = _det(conf=0.9, x1=300, y1=200, x2=340, y2=230)
        proj_big = DepthProjector().project(det_big, frame)
        proj_sml = DepthProjector().project(det_sml, frame)
        agg   = SceneAggregator()
        s_big = agg.aggregate([det_big], [proj_big])
        s_sml = agg.aggregate([det_sml], [proj_sml])
        assert s_big.objects[0].depth_score >= s_sml.objects[0].depth_score

    def test_global_score_in_unit_range(self):
        _, s = self._pipeline(z=2.0, conf=0.85)
        assert 0.0 <= s.global_score <= 1.0

    def test_skip_with_zero_detections(self):
        gate  = UncertaintyGate()
        scene = SceneConfidence(objects=[], global_score=0.0, n_objects=0)
        r     = gate.evaluate(scene)
        assert r.decision == GateDecision.SKIP

    def test_consistent_decisions_same_inputs(self):
        r1, _ = self._pipeline(z=2.0, conf=0.90, mask_frac=1.0)
        r2, _ = self._pipeline(z=2.0, conf=0.90, mask_frac=1.0)
        assert r1.decision == r2.decision
        assert r1.score    == pytest.approx(r2.score)
