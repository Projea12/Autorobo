"""
task_executor.py — Task Graph Executor (Phase 10)

Iterates through a TaskGraph and dispatches each node to the appropriate
sub-system: navigation policy, perception stack, manipulation system.

The executor handles:
  - Sequential node execution following graph edges
  - Parallel node execution when multiple nodes are ready simultaneously
  - Recovery system integration when a node fails
  - Episode logging of all execution events
  - Timeout handling per node

Usage:
    from planning.task_executor import TaskExecutor
    from planning.task_graph import TaskGraph

    executor = TaskExecutor(nav_model, grasp_model, interface, perception, cfg)
    result = executor.execute(task_graph)
    print(result["success"], result["failure_phase"])
"""

from __future__ import annotations

import logging
import time
from typing import Any

from planning.task_graph import TaskGraph, TaskNode, TaskStatus

log = logging.getLogger(__name__)

# Maximum time per action (seconds) — prevents hanging
ACTION_TIMEOUTS = {
    "find_object": 15.0,
    "navigate_to": 60.0,
    "grasp": 30.0,
    "place": 20.0,
    "open_gripper": 5.0,
    "close_gripper": 5.0,
    "wait": None,  # uses the wait duration itself
    "ask_human": 120.0,
}


class TaskExecutor:
    """
    Executes a TaskGraph by dispatching nodes to sub-systems.

    Sub-systems dispatched to:
      - navigate_to: navigation PPO policy via BaseRobotInterface
      - find_object: ObjectDetector + DepthEstimator
      - grasp: GraspPlanner + ArmController + GripperController
      - place: ArmController + GripperController
      - open/close_gripper: GripperController
      - ask_human: human interaction handler

    Integrates with RecoveryManager on node failure.
    """

    def __init__(
        self,
        nav_model: Any,
        grasp_model: Any,
        interface: Any,
        perception: Any,
        recovery_manager: Any = None,
        episode_logger: Any = None,
        cfg: Any = None,
    ) -> None:
        """
        Initialise the task executor with all sub-system references.

        Args:
            nav_model: Loaded PPO navigation model.
            grasp_model: Loaded SAC grasp model.
            interface: BaseRobotInterface (MuJoCo or ROS2).
            perception: GraspEstimator (full perception pipeline).
            recovery_manager: RecoveryManager for failure handling. Optional.
            episode_logger: EpisodeLogger for logging. Optional.
            cfg: Hydra config.

        TODO: Phase 10 — wire up ArmController, GripperController
        from interface + cfg, attach all sub-systems.
        """
        self.nav_model = nav_model
        self.grasp_model = grasp_model
        self.interface = interface
        self.perception = perception
        self.recovery_manager = recovery_manager
        self.episode_logger = episode_logger
        self.cfg = cfg
        log.info("TaskExecutor initialised")

    def execute(self, task_graph: TaskGraph) -> dict[str, Any]:
        """
        Execute a complete TaskGraph from start to finish.

        Args:
            task_graph: The TaskGraph to execute (from TaskPlanner).

        Returns:
            Execution result dict:
            {
                "success": bool,
                "failure_phase": str | None,  # which node type failed
                "failure_reason": str | None,
                "n_nodes_completed": int,
                "total_time": float,          # seconds
                "task_id": str,
            }

        TODO: Phase 10 — implement execution loop:
            while not task_graph.is_complete() and not task_graph.has_failed():
                ready_nodes = task_graph.get_ready_nodes()
                for node in ready_nodes:
                    node.status = TaskStatus.RUNNING
                    try:
                        result = self._execute_node(node)
                        node.mark_succeeded(result)
                    except Exception as e:
                        node.mark_failed(str(e))
                        if self.recovery_manager:
                            self.recovery_manager.handle_failure(node, e)
        """
        log.info(
            f"Executing TaskGraph '{task_graph.task_id}': "
            f"{task_graph.description}"
        )
        start_time = time.time()

        raise NotImplementedError(
            "TODO: Phase 10 — implement execute() loop with ready-node dispatch, "
            "recovery manager integration, and episode logging."
        )

    def _execute_node(self, node: TaskNode) -> dict[str, Any]:
        """
        Execute a single TaskNode by dispatching to the appropriate sub-system.

        Args:
            node: The TaskNode to execute (status must be RUNNING).

        Returns:
            Result dict (stored in node.result on success).

        Raises:
            NotImplementedError: If the action type is unknown.
            TimeoutError: If the action exceeds its timeout.
            RuntimeError: If the action fails.

        TODO: Phase 10 — implement dispatch table:
            dispatchers = {
                "navigate_to": self._execute_navigate_to,
                "find_object": self._execute_find_object,
                "grasp": self._execute_grasp,
                "place": self._execute_place,
                "open_gripper": self._execute_open_gripper,
                "close_gripper": self._execute_close_gripper,
                "wait": self._execute_wait,
                "ask_human": self._execute_ask_human,
            }
            dispatcher = dispatchers.get(node.action)
            if dispatcher is None:
                raise NotImplementedError(f"Unknown action: {node.action}")
            return dispatcher(node)
        """
        raise NotImplementedError(
            f"TODO: Phase 10 — implement dispatch for action '{node.action}'."
        )

    def _execute_navigate_to(self, node: TaskNode) -> dict[str, Any]:
        """
        Execute a navigate_to action using the PPO navigation policy.

        Args:
            node: Node with args["target"]: str | [x, y, z].

        Returns:
            {"reached": bool, "final_position": [x, y], "steps": int}

        TODO: Phase 10 — implement navigation rollout:
            target = node.args["target"]
            # Set goal in env, run PPO policy until done or timeout
        """
        raise NotImplementedError(
            "TODO: Phase 10 — implement navigate_to dispatcher using PPO policy."
        )

    def _execute_find_object(self, node: TaskNode) -> dict[str, Any]:
        """
        Execute a find_object action using the perception stack.

        Args:
            node: Node with args["class_name"] and optional args["description"].

        Returns:
            {"object_id": str, "position_3d": [x, y, z], "detection": Detection}

        TODO: Phase 10 — get obs from interface, run detector, match class_name.
        """
        raise NotImplementedError(
            "TODO: Phase 10 — implement find_object dispatcher using ObjectDetector."
        )

    def _execute_grasp(self, node: TaskNode) -> dict[str, Any]:
        """
        Execute a grasp action using the SAC policy + manipulation stack.

        Args:
            node: Node with args["object_id"]: str.

        Returns:
            {"grasped": bool, "grasp_quality": float}

        TODO: Phase 10 — get grasp candidate from perception, plan with GraspPlanner,
        execute with ArmController, verify with GripperController.close_and_verify().
        """
        raise NotImplementedError(
            "TODO: Phase 10 — implement grasp dispatcher using SAC policy "
            "or GraspPlanner + ArmController + GripperController."
        )

    def _execute_place(self, node: TaskNode) -> dict[str, Any]:
        """
        Execute a place action.

        Args:
            node: Node with args["object_id"] and args["target"].

        Returns:
            {"placed": bool, "final_position": [x, y, z]}

        TODO: Phase 10 — navigate near target, open gripper at target position.
        """
        raise NotImplementedError(
            "TODO: Phase 10 — implement place dispatcher."
        )

    def _execute_wait(self, node: TaskNode) -> dict[str, Any]:
        """
        Execute a wait action — pause for a specified duration.

        Args:
            node: Node with args["seconds"]: float.

        Returns:
            {"waited_seconds": float}
        """
        seconds = float(node.args.get("seconds", 1.0))
        log.info(f"Waiting {seconds}s...")
        time.sleep(seconds)
        return {"waited_seconds": seconds}

    def _execute_ask_human(self, node: TaskNode) -> dict[str, Any]:
        """
        Execute an ask_human action — prompt for human input.

        Args:
            node: Node with args["question"]: str.

        Returns:
            {"human_response": str}

        TODO: Phase 14 — implement via speech synthesis (TTS) + speech recognition (STT),
        or text I/O for development.
        """
        question = node.args.get("question", "How can I help you?")
        log.info(f"[HUMAN INPUT REQUIRED] {question}")
        # Development fallback: command-line input
        response = input(f"Robot asks: {question}\nYour response: ")
        return {"human_response": response}
