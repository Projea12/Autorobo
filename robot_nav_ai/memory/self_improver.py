"""
self_improver.py — Self-Improvement Loop (Phase 11)

Analyses logged episodes to identify failure patterns and triggers targeted
retraining on the failure cases. Implements a data flywheel:

1. Episodes are logged by EpisodeLogger
2. SelfImprover analyses failure episodes to find common failure modes
3. Failed episodes are sampled into a targeted retraining dataset
4. Training is triggered with higher weight on failure cases

This implements the "self-improvement" component of Physical AI:
the robot gets better by learning from its own mistakes.

Usage:
    from memory.self_improver import SelfImprover

    improver = SelfImprover(episode_log_dir="logs/episodes")
    report = improver.analyse_failures()
    print(report.most_common_failure_mode)
    if improver.should_retrain(report):
        improver.trigger_retraining(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class FailureAnalysisReport:
    """
    Summary of failure analysis across a set of episodes.

    Attributes:
        total_episodes: Number of episodes analysed.
        failure_count: Number of failed episodes.
        success_rate: Fraction of successful episodes.
        failure_modes: Dict mapping failure mode → count.
        most_common_failure_mode: The most frequent failure mode.
        failure_episodes: List of episode IDs that failed.
        recommended_action: High-level recommendation (e.g., "retrain_grasp").
    """
    total_episodes: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    failure_modes: dict[str, int] = field(default_factory=dict)
    most_common_failure_mode: str | None = None
    failure_episodes: list[int] = field(default_factory=list)
    recommended_action: str = "none"


class SelfImprover:
    """
    Analyses episode logs and triggers targeted retraining on failure cases.

    Self-improvement strategy:
    1. After every N episodes, scan log dir for failure episodes
    2. Classify failures by mode (nav_failure, grasp_failure, place_failure, timeout)
    3. If grasp_failure > threshold → trigger SAC fine-tuning on failure episodes
    4. If nav_failure > threshold → trigger PPO fine-tuning on failure cases
    5. Log the triggered retraining to W&B for tracking
    """

    # Success rate thresholds below which retraining is triggered
    RETRAIN_THRESHOLD_GRASP = 0.60      # retrain grasp if < 60% success
    RETRAIN_THRESHOLD_NAV = 0.70        # retrain nav if < 70% success
    MIN_EPISODES_FOR_ANALYSIS = 20      # need at least 20 episodes to analyse

    def __init__(
        self,
        episode_log_dir: str | Path = "logs/episodes",
        model_dir: str | Path = "models/",
    ) -> None:
        """
        Initialise the self-improver.

        Args:
            episode_log_dir: Directory containing episode logs from EpisodeLogger.
            model_dir: Directory containing trained model checkpoints.
        """
        self.episode_log_dir = Path(episode_log_dir)
        self.model_dir = Path(model_dir)
        self._analysis_history: list[FailureAnalysisReport] = []
        log.info(f"SelfImprover initialised. Log dir: {self.episode_log_dir}")

    def analyse_failures(
        self,
        last_n_episodes: int | None = None,
    ) -> FailureAnalysisReport:
        """
        Analyse recent episode logs to identify failure patterns.

        Args:
            last_n_episodes: Analyse only the last N episodes. None = all.

        Returns:
            FailureAnalysisReport with failure statistics and recommendations.

        TODO: Phase 11 — implement:
            metadata_files = sorted(self.episode_log_dir.glob("*_meta.json"))
            if last_n_episodes:
                metadata_files = metadata_files[-last_n_episodes:]

            failure_modes = {}
            failures = []
            for meta_file in metadata_files:
                meta = json.load(open(meta_file))
                if not meta["success"]:
                    mode = meta.get("failure_mode", "unknown")
                    failure_modes[mode] = failure_modes.get(mode, 0) + 1
                    failures.append(meta["episode_id"])

            report = FailureAnalysisReport(
                total_episodes=len(metadata_files),
                failure_count=len(failures),
                success_rate=1 - len(failures)/len(metadata_files),
                failure_modes=failure_modes,
                most_common_failure_mode=max(failure_modes, key=failure_modes.get),
                failure_episodes=failures,
                recommended_action=self._recommend_action(failure_modes),
            )
        """
        raise NotImplementedError(
            "TODO: Phase 11 — implement analyse_failures() by scanning "
            "episode_log_dir for *_meta.json files and aggregating failure modes."
        )

    def should_retrain(self, report: FailureAnalysisReport) -> bool:
        """
        Determine if retraining should be triggered based on analysis report.

        Args:
            report: FailureAnalysisReport from analyse_failures().

        Returns:
            True if retraining is recommended.
        """
        if report.total_episodes < self.MIN_EPISODES_FOR_ANALYSIS:
            log.info(
                f"Only {report.total_episodes} episodes — need at least "
                f"{self.MIN_EPISODES_FOR_ANALYSIS} before triggering retraining."
            )
            return False

        if report.success_rate < self.RETRAIN_THRESHOLD_GRASP:
            log.warning(
                f"Success rate {report.success_rate:.1%} < threshold "
                f"{self.RETRAIN_THRESHOLD_GRASP:.1%} — retraining recommended."
            )
            return True

        return False

    def trigger_retraining(self, report: FailureAnalysisReport) -> None:
        """
        Trigger targeted retraining based on the failure analysis.

        Actions taken:
        1. Create a targeted dataset of failure episodes
        2. Adjust training config (higher weight on failure cases)
        3. Launch fine-tuning script as subprocess or queue job

        Args:
            report: FailureAnalysisReport from analyse_failures().

        TODO: Phase 11 — implement:
            failure_demo_path = self._create_failure_dataset(report.failure_episodes)
            if report.most_common_failure_mode in ["grasp_failure", "slip"]:
                subprocess.run([
                    "python", "scripts/train_grasp.py",
                    f"data.failure_demos={failure_demo_path}",
                    "training.total_timesteps=200000",
                    "training.sac.learning_rate=1e-5",
                ])
        """
        log.warning(
            f"Triggering retraining for failure mode: "
            f"{report.most_common_failure_mode} "
            f"({report.failure_count}/{report.total_episodes} episodes failed)"
        )
        raise NotImplementedError(
            "TODO: Phase 11 — implement trigger_retraining(): "
            "build failure dataset, adjust training config, launch fine-tuning."
        )

    def _recommend_action(self, failure_modes: dict[str, int]) -> str:
        """
        Determine the recommended retraining action based on failure modes.

        Args:
            failure_modes: Dict of failure mode → count.

        Returns:
            Recommendation string: "retrain_grasp", "retrain_nav",
            "retrain_perception", "collect_more_demos", or "none".
        """
        if not failure_modes:
            return "none"

        most_common = max(failure_modes, key=lambda k: failure_modes[k])
        mode_to_action = {
            "grasp_failure": "retrain_grasp",
            "slip": "retrain_grasp",
            "miss": "retrain_grasp",
            "nav_failure": "retrain_nav",
            "stuck": "retrain_nav",
            "detection_failure": "retrain_perception",
            "timeout": "collect_more_demos",
        }
        return mode_to_action.get(most_common, "none")
