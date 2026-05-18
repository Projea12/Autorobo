"""
task_planner.py — Language-to-Task-Graph Planner using Claude API (Phase 14)

Takes a natural language instruction from the user and converts it into a
structured TaskGraph that the TaskExecutor can execute.

Uses the Anthropic Claude API with structured output to parse instructions.
Claude is given:
  - The instruction text
  - The current world state (objects detected, their positions, robot state)
  - Available action primitives
And returns a JSON task graph.

See ADR-004 for the rationale behind using Claude API for task planning.

Usage:
    from planning.task_planner import TaskPlanner

    planner = TaskPlanner(anthropic_api_key=os.environ["ANTHROPIC_API_KEY"])
    task_graph = planner.plan("Pick up the red mug and put it on the shelf", world_state)
    print(task_graph.nodes)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from planning.task_graph import TaskGraph, TaskNode

log = logging.getLogger(__name__)

# ── System prompt for Claude task planner ─────────────────────────────────────
TASK_PLANNER_SYSTEM_PROMPT = """
You are a task planner for a mobile manipulation robot. Your job is to convert
natural language instructions into a structured JSON task graph that the robot
can execute.

## Available Action Primitives

- navigate_to(target: str | [x, y, z])  — drive base to object or position
- find_object(class_name: str, description: str = "")  — detect and locate an object
- grasp(object_id: str)  — pick up an object with the gripper
- place(object_id: str, target: str, relation: str = "on")  — place object at target
- open_gripper()  — open the gripper
- close_gripper()  — close the gripper
- wait(seconds: float)  — pause execution
- ask_human(question: str)  — ask the human for clarification

## Output Format

Always respond with a valid JSON object (no markdown, no extra text):
{
  "task_id": "<unique task identifier>",
  "description": "<human-readable task description>",
  "nodes": [
    {"id": 0, "action": "<primitive_name>", "args": {...}, "description": "..."},
    ...
  ],
  "edges": [[0, 1], [1, 2], ...]
}

## Rules

1. Keep task graphs minimal — use only the primitives needed
2. Always find_object before navigate_to or grasp
3. Always navigate_to before grasp (unless robot is already close)
4. If the instruction is ambiguous, use ask_human as the first node
5. If the instruction is unsafe or impossible, return a single ask_human node
"""


class TaskPlanner:
    """
    Language-conditioned task planner using the Claude API.

    Converts natural language instructions into executable TaskGraph objects
    that can be dispatched by TaskExecutor.

    The planner maintains a conversation context for multi-turn interactions
    (e.g., follow-up clarifications) and supports prompt caching for the
    system prompt to reduce latency and cost on repeated calls.
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
    ) -> None:
        """
        Initialise the task planner with Anthropic client.

        Args:
            anthropic_api_key: API key. Falls back to ANTHROPIC_API_KEY env var.
            model: Claude model to use (default: claude-sonnet-4-6).
            max_tokens: Maximum tokens for task graph output.

        Raises:
            ValueError: If no API key is provided or found in environment.

        TODO: Phase 14 — the client is initialised here. Add prompt caching
        by setting cache_control on the system prompt (reduces cost by ~90%
        on repeated calls with the same system prompt).
        """
        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass anthropic_api_key argument."
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._call_count = 0
        log.info(f"TaskPlanner initialised with model: {model}")

    def plan(
        self,
        instruction: str,
        world_state: dict[str, Any],
    ) -> TaskGraph:
        """
        Convert a natural language instruction to a TaskGraph.

        Args:
            instruction: Natural language instruction from the user.
                E.g., "Pick up the banana and put it in the bowl."
            world_state: Current state of the world:
                {
                    "objects": [{"id": "obj_0", "class": "mug", "position": [x, y, z]}, ...],
                    "robot_position": [x, y, theta],
                    "gripper_open": bool,
                    "holding": str | None  # object_id if holding something
                }

        Returns:
            TaskGraph with nodes and edges representing the execution plan.

        Raises:
            ValueError: If Claude returns unparseable JSON.
            anthropic.APIError: If the API call fails.

        TODO: Phase 14 — implement with prompt caching:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": TASK_PLANNER_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},  # prompt caching
                    }
                ],
                messages=[{
                    "role": "user",
                    "content": self._build_user_message(instruction, world_state),
                }],
            )
            return self._parse_response(response.content[0].text)
        """
        log.info(f"Planning task: '{instruction}'")
        log.debug(f"World state: {world_state}")

        user_message = self._build_user_message(instruction, world_state)

        raise NotImplementedError(
            "TODO: Phase 14 — implement plan() using anthropic client.messages.create(). "
            "Enable prompt caching on system prompt for cost reduction. "
            "See ADR-004 for design rationale."
        )

    def plan_with_fallback(
        self,
        instruction: str,
        world_state: dict[str, Any],
    ) -> TaskGraph:
        """
        Plan with automatic fallback to rule-based planner on API failure.

        Args:
            instruction: Natural language instruction.
            world_state: Current world state.

        Returns:
            TaskGraph from Claude API, or fallback rule-based plan.

        TODO: Phase 14 — try plan(), catch APIError, fall back to
        _rule_based_fallback(instruction, world_state).
        """
        try:
            return self.plan(instruction, world_state)
        except (anthropic.APIError, ValueError) as e:
            log.warning(
                f"Claude API planning failed: {e}. "
                "Falling back to rule-based planner."
            )
            return self._rule_based_fallback(instruction, world_state)

    def _build_user_message(
        self,
        instruction: str,
        world_state: dict[str, Any],
    ) -> str:
        """
        Format the user message for Claude with instruction and world state.

        Args:
            instruction: The user's natural language instruction.
            world_state: Current world state dict.

        Returns:
            Formatted prompt string.
        """
        world_state_str = json.dumps(world_state, indent=2)
        return (
            f"Instruction: {instruction}\n\n"
            f"Current world state:\n{world_state_str}"
        )

    def _parse_response(self, response_text: str) -> TaskGraph:
        """
        Parse Claude's JSON response into a TaskGraph.

        Args:
            response_text: Raw text response from Claude.

        Returns:
            Parsed TaskGraph.

        Raises:
            ValueError: If response is not valid JSON or missing required fields.

        TODO: Phase 14 — implement with JSON schema validation:
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Claude returned invalid JSON: {e}")
            return TaskGraph.from_dict(data)
        """
        raise NotImplementedError(
            "TODO: Phase 14 — implement _parse_response() with JSON validation "
            "against the TaskGraph schema."
        )

    def _rule_based_fallback(
        self,
        instruction: str,
        world_state: dict[str, Any],
    ) -> TaskGraph:
        """
        Simple rule-based task planner as fallback when Claude API is unavailable.

        Handles the 5 most common task templates via keyword matching:
          - "pick up X" → find_object + navigate_to + grasp
          - "put X on Y" → find_object(X) + navigate_to(X) + grasp(X) + find_object(Y) + place
          - "bring me X" → find_object + navigate_to + grasp + navigate_to(human)
          - "go to X" → navigate_to(X)
          - anything else → ask_human

        TODO: Phase 14 — implement keyword extraction and template matching.
        """
        log.warning(
            "Using rule-based fallback planner (Claude API not available). "
            "This handles only simple, template-matching instructions."
        )
        raise NotImplementedError(
            "TODO: Phase 14 — implement rule-based fallback for top-5 task templates."
        )

    @property
    def call_count(self) -> int:
        """Return the total number of Claude API calls made."""
        return self._call_count
