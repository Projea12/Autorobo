"""
benchmarks/grasp_latency.py — Grasp inference latency profiler.

Measures wall-clock time for each component of the grasp pipeline and
compares against a 100 ms total budget (Phase 5 requirement).

Pipeline components profiled
─────────────────────────────
  perception   : YOLO detect + depth project (reuses PerceptionProfiler)
  planner      : GraspPlanner.plan() — candidate generation + ranking
  policy       : SAC policy.predict() — neural network forward pass
  controller   : ArmController.step() — DLS IK solve + safety checks

Budget breakdown (target)
──────────────────────────
  perception  ≤ 50 ms   (Phase 4 requirement, already verified)
  planner     ≤ 10 ms
  policy      ≤  5 ms   (MLP forward pass, CPU)
  controller  ≤  5 ms
  ──────────────────────
  total       ≤ 70 ms   (30 ms headroom below 100 ms budget)

Usage
─────
    from benchmarks.grasp_latency import GraspLatencyProfiler, GraspLatencyConfig

    prof   = GraspLatencyProfiler()
    report = prof.profile_all(planner, policy_fn, controller, action)
    print(report)
    for hint in prof.hints(report):
        print(" !", hint)

    # Profile individual components:
    r = prof.profile_planner(planner, obj_pos, robot_pos, robot_quat)
    r = prof.profile_policy(predict_fn, obs)
    r = prof.profile_controller(controller, action)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from perception.latency_profiler import (
    LatencyReport,
    ProfilerConfig,
    StageResult,
    PerceptionProfiler,
)


# ── budget constants ──────────────────────────────────────────────────────────

BUDGET_TOTAL_MS:      float = 100.0
BUDGET_PERCEPTION_MS: float = 50.0
BUDGET_PLANNER_MS:    float = 10.0
BUDGET_POLICY_MS:     float = 5.0
BUDGET_CONTROLLER_MS: float = 5.0


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraspLatencyConfig:
    """
    Profiling configuration for the grasp pipeline.

    budget_ms    : total pipeline budget (milliseconds)
    n_warmup     : warm-up calls discarded before timing
    n_runs       : timed calls; mean/min/max computed over these
    """
    budget_ms: float = BUDGET_TOTAL_MS
    n_warmup:  int   = 3
    n_runs:    int   = 20


# ── profiler ──────────────────────────────────────────────────────────────────

class GraspLatencyProfiler:
    """
    Wall-clock latency profiler for all grasp pipeline components.

    Parameters
    ----------
    cfg : GraspLatencyConfig
    """

    def __init__(self, cfg: GraspLatencyConfig = GraspLatencyConfig()) -> None:
        self.cfg = cfg
        self._perc_profiler = PerceptionProfiler(
            ProfilerConfig(
                budget_ms = BUDGET_PERCEPTION_MS,
                n_warmup  = cfg.n_warmup,
                n_runs    = cfg.n_runs,
            )
        )

    # ── public API ────────────────────────────────────────────────────────────

    def profile_planner(
        self,
        planner,
        obj_pos:    np.ndarray,
        robot_pos:  np.ndarray,
        robot_quat: np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
    ) -> StageResult:
        """Profile GraspPlanner.plan() latency."""
        return self._time(
            name      = "grasp_planner",
            fn        = planner.plan,
            args      = (obj_pos, robot_pos, robot_quat),
            kwargs    = {"point_cloud": point_cloud},
            budget_ms = BUDGET_PLANNER_MS,
        )

    def profile_policy(
        self,
        predict_fn: Callable,
        obs:        np.ndarray,
    ) -> StageResult:
        """
        Profile SAC policy inference latency.

        predict_fn : callable(obs) → action  (e.g. model.predict wrapper)
        obs        : (obs_dim,) observation array
        """
        return self._time(
            name      = "sac_policy",
            fn        = predict_fn,
            args      = (obs,),
            budget_ms = BUDGET_POLICY_MS,
        )

    def profile_controller(
        self,
        controller,
        action,
    ) -> StageResult:
        """Profile ArmController.step() latency."""
        return self._time(
            name      = "arm_controller",
            fn        = controller.step,
            args      = (action,),
            budget_ms = BUDGET_CONTROLLER_MS,
        )

    def profile_all(
        self,
        planner,
        predict_fn:  Callable,
        controller,
        action,
        obj_pos:     np.ndarray,
        robot_pos:   np.ndarray,
        robot_quat:  np.ndarray,
        obs:         np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
    ) -> LatencyReport:
        """
        Profile all grasp pipeline components and return a combined report.

        Returns
        -------
        LatencyReport with one StageResult per component + total.
        """
        results = [
            self.profile_planner(planner, obj_pos, robot_pos, robot_quat,
                                 point_cloud=point_cloud),
            self.profile_policy(predict_fn, obs),
            self.profile_controller(controller, action),
        ]
        total = sum(s.mean_ms for s in results)
        return LatencyReport(
            stages    = results,
            total_ms  = total,
            budget_ms = self.cfg.budget_ms,
        )

    def hints(self, report: LatencyReport) -> list[str]:
        """Return optimisation hints for stages that exceed their budget."""
        hints: list[str] = []
        for s in report.stages:
            if not s.passed:
                hints.extend(self._hints_for(s.name, s.mean_ms))
        if not report.passed and not hints:
            hints.append(
                f"Total {report.total_ms:.1f}ms > {self.cfg.budget_ms:.0f}ms — "
                "run on GPU or reduce n_arm_joints / point cloud density"
            )
        return hints

    def run_standalone(
        self,
        planner,
        obj_pos:    np.ndarray,
        robot_pos:  np.ndarray,
        robot_quat: np.ndarray,
        point_cloud: Optional[np.ndarray] = None,
    ) -> LatencyReport:
        """Profile only the planner (no policy/controller needed)."""
        stage = self.profile_planner(
            planner, obj_pos, robot_pos, robot_quat, point_cloud=point_cloud
        )
        return LatencyReport(
            stages    = [stage],
            total_ms  = stage.mean_ms,
            budget_ms = BUDGET_PLANNER_MS,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _time(
        self,
        name:      str,
        fn:        Callable,
        args:      tuple     = (),
        kwargs:    dict      = {},
        budget_ms: float     = BUDGET_TOTAL_MS,
    ) -> StageResult:
        for _ in range(self.cfg.n_warmup):
            fn(*args, **kwargs)

        timings: list[float] = []
        for _ in range(self.cfg.n_runs):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            timings.append((time.perf_counter() - t0) * 1_000.0)

        arr = np.array(timings)
        return StageResult(
            name      = name,
            mean_ms   = float(arr.mean()),
            min_ms    = float(arr.min()),
            max_ms    = float(arr.max()),
            n_runs    = self.cfg.n_runs,
            budget_ms = budget_ms,
        )

    @staticmethod
    def _hints_for(name: str, ms: float) -> list[str]:
        nl = name.lower()
        if "planner" in nl:
            return [
                f"{name} ({ms:.1f}ms): reduce n_side_rotations or top_k; "
                "skip PCA for point clouds < 20 points"
            ]
        if "policy" in nl or "sac" in nl:
            return [
                f"{name} ({ms:.1f}ms): export policy to ONNX or TorchScript; "
                "reduce net_arch from [256,256] to [128,128]"
            ]
        if "controller" in nl or "arm" in nl:
            return [
                f"{name} ({ms:.1f}ms): DLS solve is O(n³) — "
                "consider caching Jacobian or reducing n_substeps"
            ]
        return [f"{name} ({ms:.1f}ms): profile sub-operations to find bottleneck"]

    def __repr__(self) -> str:
        return (f"GraspLatencyProfiler("
                f"budget={self.cfg.budget_ms}ms, "
                f"n_runs={self.cfg.n_runs})")
