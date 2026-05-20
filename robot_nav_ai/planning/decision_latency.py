"""
planning/decision_latency.py — End-to-end uncertainty gate latency profiler.

Measures the wall-clock time for the complete uncertainty decision pipeline:

    NavSignals  → NavigationConfidenceScorer.score()    ┐
    GraspSignals → GraspConfidenceScorer.score()        ├─ scored_layers
    perception_score (scalar)                           ┘
        ↓
    UncertaintyPipeline.propagate()     → LayeredConfidence
        ↓
    DecisionGate.evaluate()             → DecisionResult
        ↓
    ConservativePolicy.apply()          → ConservativeAction
                                              total ≤ 20 ms

Budget breakdown (default)
───────────────────────────
  nav_score_budget_ms     :  3 ms  — linear arithmetic, always fast
  grasp_score_budget_ms   :  3 ms  — linear arithmetic, always fast
  propagate_budget_ms     :  5 ms  — one exp / log per layer
  gate_budget_ms          :  3 ms  — two comparisons
  policy_budget_ms        :  3 ms  — dict lookup + dataclass construction
  total_budget_ms         : 20 ms  — hard deadline for the full gate

Usage
─────
    profiler = DecisionLatencyProfiler()
    report   = profiler.profile(
        nav_signals    = NavSignals(...),
        grasp_signals  = GraspSignals(...),
        perception_score = 0.85,
    )
    assert report.passed, f"Gate too slow: {report.total_ms:.2f} ms"
    print(report.summary())
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from planning.nav_confidence      import NavSignals, NavigationConfidenceScorer
from planning.grasp_confidence    import GraspSignals, GraspConfidenceScorer
from planning.uncertainty_pipeline import UncertaintyPipeline, PropagationConfig
from planning.decision_gate       import DecisionGate, DecisionGateConfig, DecisionResult
from planning.conservative_policy import ConservativePolicy, ConservativePolicyConfig, ConservativeAction

log = logging.getLogger(__name__)


# ── budget config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LatencyBudgetConfig:
    """
    Per-phase and overall latency budgets in milliseconds.

    All values are wall-clock upper bounds.  Violation is reported in the
    LatencyReport but does NOT raise — the caller decides how to respond.
    """
    nav_score_budget_ms:   float = 3.0
    grasp_score_budget_ms: float = 3.0
    propagate_budget_ms:   float = 5.0
    gate_budget_ms:        float = 3.0
    policy_budget_ms:      float = 3.0
    total_budget_ms:       float = 20.0


# ── per-phase timing ──────────────────────────────────────────────────────────

@dataclass
class PhaseLatency:
    """Wall-clock time for one pipeline phase."""
    name:     str
    elapsed_ms: float
    budget_ms:  float

    @property
    def passed(self) -> bool:
        return self.elapsed_ms <= self.budget_ms

    def __repr__(self) -> str:
        status = "OK" if self.passed else "SLOW"
        return f"PhaseLatency({self.name}: {self.elapsed_ms:.3f}ms / {self.budget_ms:.1f}ms [{status}])"


# ── full report ───────────────────────────────────────────────────────────────

@dataclass
class LatencyReport:
    """
    Full profiling result for one end-to-end gate invocation.

    Fields
    ──────
    phases          : per-phase PhaseLatency objects in pipeline order
    total_ms        : total wall-clock time for the complete gate
    budget_ms       : configured total budget
    decision_result : the DecisionResult produced by the gate
    conservative_action : the ConservativeAction produced by the policy
    n_warmup_runs   : number of warm-up iterations that preceded this measurement
    """
    phases:              list[PhaseLatency]
    total_ms:            float
    budget_ms:           float
    decision_result:     DecisionResult
    conservative_action: ConservativeAction
    n_warmup_runs:       int = 0

    @property
    def passed(self) -> bool:
        """True if total_ms ≤ budget_ms and every phase is within its budget."""
        return self.total_ms <= self.budget_ms and all(p.passed for p in self.phases)

    @property
    def slowest_phase(self) -> PhaseLatency:
        return max(self.phases, key=lambda p: p.elapsed_ms)

    def summary(self) -> str:
        """Return a multi-line human-readable summary."""
        lines = [
            f"DecisionLatency: {'PASS' if self.passed else 'FAIL'} "
            f"({self.total_ms:.3f} ms / {self.budget_ms:.1f} ms budget)",
        ]
        for p in self.phases:
            flag = "  OK  " if p.passed else " SLOW "
            lines.append(f"  [{flag}] {p.name:<20} {p.elapsed_ms:6.3f} ms "
                         f"(budget {p.budget_ms:.1f} ms)")
        lines.append(f"  Decision : {self.decision_result.decision.value} "
                     f"(score={self.decision_result.score:.3f})")
        lines.append(f"  Action   : {self.conservative_action.action_type.value} "
                     f"(scale={self.conservative_action.velocity_scale:.2f})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"LatencyReport({'PASS' if self.passed else 'FAIL'}, "
                f"total={self.total_ms:.3f}ms, "
                f"decision={self.decision_result.decision.value})")


# ── profiler ──────────────────────────────────────────────────────────────────

class DecisionLatencyProfiler:
    """
    Profiles the full uncertainty → decision → policy pipeline.

    Internally constructs the full pipeline from configurable components.
    Supports warm-up runs to stabilise JIT / import caches before measurement.

    Parameters
    ----------
    budget          : LatencyBudgetConfig
    pipeline_cfg    : PropagationConfig for UncertaintyPipeline
    gate_cfg        : DecisionGateConfig for DecisionGate
    policy_cfg      : ConservativePolicyConfig for ConservativePolicy
    """

    def __init__(
        self,
        budget:       LatencyBudgetConfig       = LatencyBudgetConfig(),
        pipeline_cfg: PropagationConfig          = PropagationConfig(),
        gate_cfg:     DecisionGateConfig         = DecisionGateConfig(),
        policy_cfg:   ConservativePolicyConfig   = ConservativePolicyConfig(),
    ) -> None:
        self.budget   = budget
        self._nav_scorer   = NavigationConfidenceScorer()
        self._grasp_scorer = GraspConfidenceScorer()
        self._pipeline     = UncertaintyPipeline(pipeline_cfg)
        self._gate         = DecisionGate(gate_cfg)
        self._policy       = ConservativePolicy(policy_cfg)

    # ── public API ────────────────────────────────────────────────────────────

    def profile(
        self,
        nav_signals:       NavSignals,
        grasp_signals:     GraspSignals,
        perception_score:  float,
        n_warmup_runs:     int = 3,
    ) -> LatencyReport:
        """
        Run the full pipeline once (with optional warm-up) and return a report.

        Parameters
        ----------
        nav_signals      : NavSignals for this planning step
        grasp_signals    : GraspSignals for this planning step
        perception_score : [0,1] from SceneAggregator / perception layer
        n_warmup_runs    : number of unmetered warm-up passes (default 3)
                           eliminates first-call import/JIT overhead

        Returns
        -------
        LatencyReport — total_ms, per-phase PhaseLatency, decision, action.
        """
        for _ in range(n_warmup_runs):
            self._run_pipeline(nav_signals, grasp_signals, perception_score)

        return self._run_pipeline(
            nav_signals, grasp_signals, perception_score,
            n_warmup_runs=n_warmup_runs,
        )

    def profile_repeated(
        self,
        nav_signals:      NavSignals,
        grasp_signals:    GraspSignals,
        perception_score: float,
        n_runs:           int = 100,
        n_warmup_runs:    int = 3,
    ) -> dict[str, float]:
        """
        Run the pipeline n_runs times and return aggregate timing statistics.

        Parameters
        ----------
        n_runs        : number of measured iterations
        n_warmup_runs : warm-up iterations before measurement begins

        Returns
        -------
        dict with keys: min_ms, max_ms, mean_ms, p95_ms, p99_ms, budget_ms,
                        pass_rate (fraction of runs under budget).
        """
        for _ in range(n_warmup_runs):
            self._run_pipeline(nav_signals, grasp_signals, perception_score)

        totals: list[float] = []
        for _ in range(n_runs):
            report = self._run_pipeline(nav_signals, grasp_signals, perception_score)
            totals.append(report.total_ms)

        totals_sorted = sorted(totals)
        n = len(totals_sorted)
        return {
            "min_ms":    totals_sorted[0],
            "max_ms":    totals_sorted[-1],
            "mean_ms":   sum(totals) / n,
            "p95_ms":    totals_sorted[int(0.95 * n)],
            "p99_ms":    totals_sorted[int(0.99 * n)],
            "budget_ms": self.budget.total_budget_ms,
            "pass_rate": sum(1 for t in totals if t <= self.budget.total_budget_ms) / n,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        nav_signals:      NavSignals,
        grasp_signals:    GraspSignals,
        perception_score: float,
        n_warmup_runs:    int = 0,
    ) -> LatencyReport:
        phases: list[PhaseLatency] = []
        b = self.budget

        # Phase 1 — navigation confidence score
        t0 = time.perf_counter()
        nav_conf = self._nav_scorer.score(nav_signals)
        phases.append(PhaseLatency("nav_score",
                                   (time.perf_counter() - t0) * 1000,
                                   b.nav_score_budget_ms))

        # Phase 2 — grasp confidence score
        t0 = time.perf_counter()
        grasp_conf = self._grasp_scorer.score(grasp_signals)
        phases.append(PhaseLatency("grasp_score",
                                   (time.perf_counter() - t0) * 1000,
                                   b.grasp_score_budget_ms))

        # Phase 3 — uncertainty propagation
        t0 = time.perf_counter()
        layered = self._pipeline.propagate(
            perception_score = perception_score,
            nav_score        = nav_conf.combined,
            grasp_score      = grasp_conf.combined,
        )
        phases.append(PhaseLatency("propagate",
                                   (time.perf_counter() - t0) * 1000,
                                   b.propagate_budget_ms))

        # Phase 4 — decision gate
        t0 = time.perf_counter()
        decision_result = self._gate.evaluate(layered)
        phases.append(PhaseLatency("gate",
                                   (time.perf_counter() - t0) * 1000,
                                   b.gate_budget_ms))

        # Phase 5 — conservative policy
        t0 = time.perf_counter()
        conservative_action = self._policy.apply(decision_result)
        phases.append(PhaseLatency("policy",
                                   (time.perf_counter() - t0) * 1000,
                                   b.policy_budget_ms))

        total_ms = sum(p.elapsed_ms for p in phases)

        if total_ms > b.total_budget_ms:
            log.warning(
                "DecisionLatencyProfiler: gate exceeded budget "
                "(%.3f ms > %.1f ms budget, slowest: %s)",
                total_ms, b.total_budget_ms,
                max(phases, key=lambda p: p.elapsed_ms).name,
            )

        return LatencyReport(
            phases              = phases,
            total_ms            = total_ms,
            budget_ms           = b.total_budget_ms,
            decision_result     = decision_result,
            conservative_action = conservative_action,
            n_warmup_runs       = n_warmup_runs,
        )

    def __repr__(self) -> str:
        return (f"DecisionLatencyProfiler("
                f"budget={self.budget.total_budget_ms}ms, "
                f"phases={len(self.budget.__dataclass_fields__)-1})")
