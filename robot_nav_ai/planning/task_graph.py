"""
task_graph.py — TaskGraph and TaskNode Dataclasses (Phase 10)

Defines the data structures for representing structured task plans
produced by the TaskPlanner and consumed by the TaskExecutor.

A TaskGraph is a directed acyclic graph (DAG) where:
  - Nodes are individual action steps (find_object, navigate_to, grasp, etc.)
  - Edges represent execution order (node A must complete before node B)

For linear tasks: a simple chain of edges [0→1, 1→2, 2→3]
For parallel tasks: branching edges (e.g., find two objects simultaneously)

Usage:
    from planning.task_graph import TaskGraph, TaskNode, TaskStatus

    graph = TaskGraph(task_id="task_001", description="Pick up mug")
    node0 = TaskNode(id=0, action="find_object", args={"class_name": "mug"})
    node1 = TaskNode(id=1, action="grasp", args={"object_id": "obj_0"})
    graph.add_node(node0)
    graph.add_node(node1)
    graph.add_edge(0, 1)
    print(graph.get_ready_nodes())  # [node0]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TaskStatus(Enum):
    """Execution status of a TaskNode."""
    PENDING = auto()     # not yet started
    READY = auto()       # all dependencies completed, ready to execute
    RUNNING = auto()     # currently being executed
    SUCCEEDED = auto()   # completed successfully
    FAILED = auto()      # failed — may trigger recovery
    SKIPPED = auto()     # skipped (e.g., dependency failed)


@dataclass
class TaskNode:
    """
    Represents a single step in a task plan.

    Attributes:
        id: Unique integer identifier within the task graph.
        action: Name of the action primitive to execute.
            One of: navigate_to, find_object, grasp, place,
                    open_gripper, close_gripper, wait, ask_human
        args: Dict of arguments for the action primitive.
            E.g., {"class_name": "mug"} for find_object.
        description: Human-readable description of this step.
        status: Current execution status.
        result: Output of the action (set when status = SUCCEEDED).
            E.g., {"object_id": "obj_0", "position": [x, y, z]} for find_object.
        error: Error message if status = FAILED.
        retry_count: Number of times this node has been retried.
        max_retries: Maximum allowed retries before marking as FAILED.
    """
    id: int
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 2

    def __repr__(self) -> str:
        return (
            f"TaskNode(id={self.id}, action='{self.action}', "
            f"status={self.status.name}, args={self.args})"
        )

    def can_retry(self) -> bool:
        """Return True if this node has retry attempts remaining."""
        return self.retry_count < self.max_retries

    def mark_succeeded(self, result: dict[str, Any] | None = None) -> None:
        """Mark node as successfully completed with optional result."""
        self.status = TaskStatus.SUCCEEDED
        self.result = result or {}

    def mark_failed(self, error: str) -> None:
        """Mark node as failed with error message."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.retry_count += 1

    def reset(self) -> None:
        """Reset node to PENDING for retry."""
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None


class TaskGraph:
    """
    Directed acyclic graph representing a structured task plan.

    Nodes are TaskNode objects. Edges define execution dependencies:
    an edge (A → B) means node B cannot start until node A succeeds.

    Provides methods for the TaskExecutor to query execution state:
      - get_ready_nodes(): nodes whose dependencies are all SUCCEEDED
      - is_complete(): True if all nodes SUCCEEDED
      - has_failed(): True if any node FAILED with no retries remaining
    """

    def __init__(
        self,
        task_id: str,
        description: str = "",
    ) -> None:
        """
        Initialise an empty task graph.

        Args:
            task_id: Unique identifier for this task instance.
            description: Human-readable task description.
        """
        self.task_id = task_id
        self.description = description
        self._nodes: dict[int, TaskNode] = {}
        self._edges: list[tuple[int, int]] = []  # (from_id, to_id)

    def add_node(self, node: TaskNode) -> None:
        """
        Add a TaskNode to the graph.

        Args:
            node: TaskNode to add.

        Raises:
            ValueError: If a node with this ID already exists.
        """
        if node.id in self._nodes:
            raise ValueError(
                f"Node with id={node.id} already exists in task graph '{self.task_id}'."
            )
        self._nodes[node.id] = node

    def add_edge(self, from_id: int, to_id: int) -> None:
        """
        Add a directed edge (dependency) from one node to another.

        Args:
            from_id: ID of the prerequisite node.
            to_id: ID of the dependent node.

        Raises:
            KeyError: If either node ID does not exist.
        """
        if from_id not in self._nodes:
            raise KeyError(f"Node {from_id} not found in graph.")
        if to_id not in self._nodes:
            raise KeyError(f"Node {to_id} not found in graph.")
        self._edges.append((from_id, to_id))

    def get_node(self, node_id: int) -> TaskNode:
        """Get a node by ID."""
        return self._nodes[node_id]

    def get_ready_nodes(self) -> list[TaskNode]:
        """
        Return nodes that are ready to execute.

        A node is ready if:
        1. Its status is PENDING
        2. All nodes that have edges pointing to it are SUCCEEDED

        Returns:
            List of TaskNode objects ready for execution.
        """
        # Build set of all prerequisite (from_id) → {to_ids}
        dependencies: dict[int, set[int]] = {nid: set() for nid in self._nodes}
        for from_id, to_id in self._edges:
            dependencies[to_id].add(from_id)

        ready = []
        for node_id, node in self._nodes.items():
            if node.status != TaskStatus.PENDING:
                continue
            # Check all prerequisites are SUCCEEDED
            prereqs = dependencies[node_id]
            if all(
                self._nodes[p].status == TaskStatus.SUCCEEDED
                for p in prereqs
            ):
                node.status = TaskStatus.READY
                ready.append(node)

        return ready

    def is_complete(self) -> bool:
        """Return True if all nodes have SUCCEEDED."""
        return all(
            node.status == TaskStatus.SUCCEEDED
            for node in self._nodes.values()
        )

    def has_failed(self) -> bool:
        """Return True if any node has FAILED with no retries remaining."""
        return any(
            node.status == TaskStatus.FAILED and not node.can_retry()
            for node in self._nodes.values()
        )

    @property
    def nodes(self) -> list[TaskNode]:
        """Return all nodes as a list, sorted by ID."""
        return [self._nodes[k] for k in sorted(self._nodes.keys())]

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Return all edges as (from_id, to_id) tuples."""
        return list(self._edges)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskGraph":
        """
        Construct a TaskGraph from the dict returned by Claude API.

        Args:
            data: Dict with "task_id", "description", "nodes", "edges".

        Returns:
            Parsed TaskGraph.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If node IDs referenced in edges don't exist.
        """
        graph = cls(
            task_id=data["task_id"],
            description=data.get("description", ""),
        )
        for node_data in data["nodes"]:
            node = TaskNode(
                id=node_data["id"],
                action=node_data["action"],
                args=node_data.get("args", {}),
                description=node_data.get("description", ""),
            )
            graph.add_node(node)
        for from_id, to_id in data.get("edges", []):
            graph.add_edge(from_id, to_id)
        return graph

    def to_dict(self) -> dict[str, Any]:
        """Serialise the TaskGraph to a JSON-compatible dict."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "nodes": [
                {
                    "id": node.id,
                    "action": node.action,
                    "args": node.args,
                    "description": node.description,
                    "status": node.status.name,
                }
                for node in self.nodes
            ],
            "edges": [[from_id, to_id] for from_id, to_id in self._edges],
        }

    def __repr__(self) -> str:
        return (
            f"TaskGraph(id='{self.task_id}', "
            f"nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, "
            f"complete={self.is_complete()})"
        )
