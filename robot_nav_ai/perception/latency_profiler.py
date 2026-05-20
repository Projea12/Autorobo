"""
perception/latency_profiler.py — Inference latency profiler for the perception pipeline.

Measures wall-clock time for each pipeline stage, compares against a total
budget (default 50 ms), and emits optimisation hints for stages that are slow.

Typical deployment timings (guideline):
    YOLO detect (yolov8n, GPU)  :  8–15 ms
    SAM segment (vit_b, GPU)    : 15–30 ms
    DepthProjector              :  < 1 ms
    SceneAggregator             :  < 1 ms
    UncertaintyGate             :  < 1 ms
    ─────────────────────────────────────
    Target total                : < 50 ms

Usage
─────
    from perception.latency_profiler import PerceptionProfiler, ProfilerConfig

    prof = PerceptionProfiler(ProfilerConfig(budget_ms=50.0, n_warmup=3, n_runs=20))

    r = prof.profile_stages([
        ("yolo_detect",  detector.detect,          (frame.rgb,), {}),
        ("depth_proj",   projector.project_batch,  (dets, frame), {}),
        ("aggregator",   aggregator.aggregate,      (dets, projs), {}),
        ("gate",         gate.evaluate,             (scene,), {}),
    ])
    print(r)
    for hint in prof.optimization_hints(r):
        print(" !", hint)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ── per-stage result ──────────────────────────────────────────────────────────

@dataclass
class StageResult:
    """
    Timing statistics for a single pipeline stage.

    Fields
    ------
    name      : stage identifier (e.g. "yolo_detect")
    mean_ms   : mean wall-clock time across timed runs (milliseconds)
    min_ms    : fastest timed run
    max_ms    : slowest timed run
    n_runs    : number of timed runs (warmup excluded)
    budget_ms : per-stage budget used for the passed property
    """
    name:      str
    mean_ms:   float
    min_ms:    float
    max_ms:    float
    n_runs:    int
    budget_ms: float

    @property
    def passed(self) -> bool:
        """True if mean_ms is strictly below budget_ms."""
        return self.mean_ms < self.budget_ms

    def __repr__(self) -> str:
        ok = "PASS" if self.passed else "FAIL"
        return (f"StageResult({self.name!r}, mean={self.mean_ms:.2f}ms, "
                f"min={self.min_ms:.2f}ms, max={self.max_ms:.2f}ms, "
                f"budget={self.budget_ms:.0f}ms, {ok})")


# ── report ────────────────────────────────────────────────────────────────────

@dataclass
class LatencyReport:
    """
    Aggregated timing report for all profiled pipeline stages.

    Fields
    ------
    stages    : ordered list of StageResult, one per profiled stage
    total_ms  : sum of mean_ms across all stages
    budget_ms : total pipeline budget (from ProfilerConfig)
    """
    stages:    list[StageResult]
    total_ms:  float
    budget_ms: float

    @property
    def passed(self) -> bool:
        """True if total_ms < budget_ms."""
        return self.total_ms < self.budget_ms

    @property
    def slowest(self) -> Optional[str]:
        """Name of the stage with the highest mean_ms, or None if no stages."""
        if not self.stages:
            return None
        return max(self.stages, key=lambda s: s.mean_ms).name

    def to_dict(self) -> dict:
        return {
            "stages": [
                {
                    "name":      s.name,
                    "mean_ms":   round(s.mean_ms, 3),
                    "min_ms":    round(s.min_ms, 3),
                    "max_ms":    round(s.max_ms, 3),
                    "n_runs":    s.n_runs,
                    "budget_ms": s.budget_ms,
                    "passed":    s.passed,
                }
                for s in self.stages
            ],
            "total_ms":  round(self.total_ms, 3),
            "budget_ms": self.budget_ms,
            "passed":    self.passed,
            "slowest":   self.slowest,
        }

    def __str__(self) -> str:
        w = max((len(s.name) for s in self.stages), default=8)
        lines = ["LatencyReport"]
        for s in self.stages:
            ok = "✓" if s.passed else "✗"
            lines.append(
                f"  {ok} {s.name:<{w}}  {s.mean_ms:6.2f}ms"
                f"  [min {s.min_ms:.2f} / max {s.max_ms:.2f}]"
                f"  budget {s.budget_ms:.0f}ms"
            )
        sep = "  " + "─" * (w + 40)
        lines.append(sep)
        status = "PASS" if self.passed else "FAIL"
        lines.append(f"  Total: {self.total_ms:.2f}ms / {self.budget_ms:.0f}ms  [{status}]")
        return "\n".join(lines)


# ── config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProfilerConfig:
    """
    Configuration for PerceptionProfiler.

    budget_ms : total pipeline latency budget (milliseconds); default 50 ms
    n_warmup  : number of warm-up calls before timing (JIT, cache fill)
    n_runs    : number of timed calls; mean/min/max computed over these
    """
    budget_ms: float = 50.0
    n_warmup:  int   = 2
    n_runs:    int   = 10


# ── profiler ──────────────────────────────────────────────────────────────────

class PerceptionProfiler:
    """
    Wall-clock profiler for perception pipeline stages.

    Parameters
    ----------
    cfg : ProfilerConfig
    """

    def __init__(self, cfg: ProfilerConfig = ProfilerConfig()) -> None:
        self.cfg = cfg

    # ── public API ────────────────────────────────────────────────────────────

    def time_stage(
        self,
        name:      str,
        fn:        Callable,
        *args,
        budget_ms: Optional[float] = None,
        n_warmup:  Optional[int]   = None,
        n_runs:    Optional[int]   = None,
        **kwargs,
    ) -> StageResult:
        """
        Time a single callable and return a StageResult.

        Parameters
        ----------
        name      : human-readable stage label
        fn        : callable to time
        *args     : positional arguments forwarded to fn
        budget_ms : per-stage budget; defaults to cfg.budget_ms
        n_warmup  : warmup calls; defaults to cfg.n_warmup
        n_runs    : timed calls; defaults to cfg.n_runs
        **kwargs  : keyword arguments forwarded to fn

        Returns
        -------
        StageResult with mean/min/max timings.
        """
        budget = budget_ms if budget_ms is not None else self.cfg.budget_ms
        warmup = n_warmup  if n_warmup  is not None else self.cfg.n_warmup
        runs   = n_runs    if n_runs    is not None else self.cfg.n_runs

        for _ in range(warmup):
            fn(*args, **kwargs)

        timings: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            timings.append((time.perf_counter() - t0) * 1_000.0)

        return StageResult(
            name      = name,
            mean_ms   = float(np.mean(timings)),
            min_ms    = float(np.min(timings)),
            max_ms    = float(np.max(timings)),
            n_runs    = runs,
            budget_ms = budget,
        )

    def profile_stages(
        self,
        stages: list[tuple],
    ) -> LatencyReport:
        """
        Profile a list of stages and return a LatencyReport.

        Each element of ``stages`` is a tuple:
            (name, fn)                     — no arguments
            (name, fn, args)               — positional args tuple
            (name, fn, args, kwargs)       — positional + keyword dicts

        Parameters
        ----------
        stages : ordered list of (name, callable[, args[, kwargs]]) tuples

        Returns
        -------
        LatencyReport with one StageResult per stage.
        """
        results: list[StageResult] = []
        for entry in stages:
            if len(entry) == 2:
                name, fn = entry; args = (); kw = {}
            elif len(entry) == 3:
                name, fn, args = entry; kw = {}
            else:
                name, fn, args, kw = entry
            results.append(self.time_stage(name, fn, *args, **kw))

        total = sum(s.mean_ms for s in results)
        return LatencyReport(stages=results, total_ms=total,
                             budget_ms=self.cfg.budget_ms)

    def profile_projector(self, projector, detections, frame) -> LatencyReport:
        """Profile DepthProjector.project_batch on the provided detections."""
        stage = self.time_stage("depth_projector",
                                projector.project_batch, detections, frame)
        return LatencyReport(stages=[stage], total_ms=stage.mean_ms,
                             budget_ms=self.cfg.budget_ms)

    def profile_aggregator(self, aggregator, detections,
                            projections=None) -> LatencyReport:
        """Profile SceneAggregator.aggregate."""
        stage = self.time_stage("aggregator",
                                aggregator.aggregate, detections, projections)
        return LatencyReport(stages=[stage], total_ms=stage.mean_ms,
                             budget_ms=self.cfg.budget_ms)

    def profile_gate(self, gate, scene) -> LatencyReport:
        """Profile UncertaintyGate.evaluate."""
        stage = self.time_stage("gate", gate.evaluate, scene)
        return LatencyReport(stages=[stage], total_ms=stage.mean_ms,
                             budget_ms=self.cfg.budget_ms)

    def optimization_hints(self, report: LatencyReport) -> list[str]:
        """
        Return a list of optimisation suggestions for stages that are slow.

        Hints are keyed on stage name substrings:
          "detect" / "yolo"  → model size, TensorRT, imgsz
          "sam" / "seg"      → vit_b, gating SAM to high-conf detections
          "projector"        → already fast; redirect attention elsewhere
        A final hint is appended when the total budget is exceeded.
        """
        hints: list[str] = []
        for s in report.stages:
            if not s.passed:
                hints.extend(self._hints_for(s.name, s.mean_ms))
        if not report.passed and not hints:
            hints.append(
                f"Total {report.total_ms:.1f}ms exceeds budget — "
                "run inference on GPU (device='cuda' or 'mps')"
            )
        return hints

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _hints_for(name: str, ms: float) -> list[str]:
        nl = name.lower()
        if any(k in nl for k in ("detect", "yolo")):
            return [
                f"{name} ({ms:.1f}ms): use yolov8n.pt; "
                "export to TensorRT/ONNX; reduce imgsz (e.g. 320)"
            ]
        if any(k in nl for k in ("sam", "seg")):
            return [
                f"{name} ({ms:.1f}ms): use vit_b model; "
                "gate SAM to detections with confidence > 0.5 only"
            ]
        if any(k in nl for k in ("projector", "proj", "depth")):
            return [
                f"{name} ({ms:.1f}ms): projector is already fast — "
                "check detector/SAM first"
            ]
        if any(k in nl for k in ("aggregator", "aggr")):
            return [
                f"{name} ({ms:.1f}ms): aggregator overhead — "
                "reduce number of detections passed in"
            ]
        if "gate" in nl:
            return [
                f"{name} ({ms:.1f}ms): gate is trivially fast — "
                "profile other stages"
            ]
        return [
            f"{name} ({ms:.1f}ms): profile sub-operations to find bottleneck"
        ]

    def __repr__(self) -> str:
        return (f"PerceptionProfiler(budget={self.cfg.budget_ms}ms, "
                f"n_warmup={self.cfg.n_warmup}, n_runs={self.cfg.n_runs})")
