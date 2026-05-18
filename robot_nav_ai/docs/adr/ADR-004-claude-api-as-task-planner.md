# ADR-004: Claude API as Language-to-Task-Graph Planner

**Status:** Accepted  
**Date:** 2026-05-18  
**Author:** AutoRobo Team  
**Deciders:** John Olugbemi

---

## Context

The system's goal is to accept natural language instructions from a user and execute
them autonomously. This requires a **language-to-task-graph translator** that can:

1. Parse arbitrarily complex natural language instructions
2. Ground language to physical objects ("the red mug" → specific detected object)
3. Decompose instructions into a sequence of executable sub-tasks
4. Handle ambiguity by asking clarifying questions
5. Adapt to failure by replanning at the task level

Example:
```
Input:  "Can you clear the table? Put all the dishes in the sink."
Output: TaskGraph([
    find_objects(class="dish"),
    for each dish:
        navigate_to(dish),
        grasp(dish),
        navigate_to("sink"),
        place(dish, in="sink")
])
```

### Approaches Considered

1. **Rule-based parser:** Hard-coded grammar rules map specific phrases to task templates
2. **Fine-tuned small LM:** Fine-tune a small language model (e.g., T5, Phi-3) on task-plan pairs
3. **GPT-4/Claude API:** Use a frontier LLM via API for zero-shot task planning
4. **LMPC (Language Model Predictive Control):** LLM generates cost functions for MPC
5. **SayCan:** Grounded affordance-weighted LLM planning

### Evaluation Criteria

| Criteria | Rule-based | Fine-tuned LM | Claude API | LMPC |
|----------|-----------|--------------|------------|------|
| Handles novel instructions | No | Partial | Yes | Yes |
| Handles ambiguity | No | No | Yes | Partial |
| Zero-shot generalisation | No | No | Yes | Partial |
| Development speed | Fast | Slow | Fast | Medium |
| Inference cost | Free | Low | API cost | Medium |
| Works offline | Yes | Yes | No | No |
| Reliability | High | Medium | High | Medium |
| Context window for scene | No | Limited | 200K tokens | Limited |

---

## Decision

**Use the Anthropic Claude API (claude-sonnet-4-6) for task planning.**

Claude is invoked as a structured output generator. The system prompt encodes:
- Available action primitives (`navigate_to`, `grasp`, `place`, `find_object`, etc.)
- Current world state (detected objects, their positions, room map)
- Robot capabilities and workspace limits

Claude returns a structured JSON task graph that `task_executor.py` interprets.

```python
# planning/task_planner.py
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """
You are a task planner for a mobile manipulation robot.
Given a natural language instruction and the current world state,
output a JSON task graph with nodes and edges.

Available actions: navigate_to, find_object, grasp, place, open_gripper, close_gripper

Output format:
{
  "task_id": "...",
  "nodes": [{"id": 0, "action": "...", "args": {...}}, ...],
  "edges": [[0, 1], [1, 2], ...]
}
"""

def plan(instruction: str, world_state: dict) -> TaskGraph:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Instruction: {instruction}\nWorld state: {world_state}"
        }]
    )
    return TaskGraph.from_json(response.content[0].text)
```

### Why Claude Specifically

1. **200K token context window:** Can include the full scene description, object inventory,
   and task history in a single prompt. Critical for complex multi-step tasks.

2. **Instruction following reliability:** Claude consistently follows structured output
   format requirements (JSON schema). This is essential for reliable parsing.

3. **Safety alignment:** Claude refuses unsafe requests (e.g., "throw the object at the person").
   This adds a layer of safety filtering above the robot's own safety systems.

4. **Tool use support:** Claude's tool use API will enable Phase 14+ where Claude can
   call planning primitives directly rather than generating JSON strings.

5. **Prompt caching:** Anthropic's prompt caching reduces cost and latency when the
   system prompt (which includes action primitives) is repeated across many calls.

### Cost and Latency Estimates

- Claude Sonnet: ~$3/M input tokens, ~$15/M output tokens (as of 2026-05)
- Typical task planning call: ~2K tokens in, ~500 tokens out
- Cost per task plan: ~$0.014 (1.4 cents)
- Latency: ~500ms–1.5s (acceptable for task-level planning, not reactive control)

Task planning runs once at the start of each task, not in the control loop.
Reactive control (50 Hz) uses the RL policies — Claude is not on the hot path.

---

## Alternatives Considered

### Rule-Based Parser — Rejected

Too brittle. "Put the mug next to (not in) the bowl" would require explicitly coding
the spatial relationship parser. Impossible to handle the full range of user language.

### Fine-Tuned Small LM — Deferred

A fine-tuned Phi-3 or Qwen2.5 model could work offline and at lower cost.
Deferred to Phase 17+ — requires collecting a task-plan dataset first. Claude API
enables rapid prototyping without upfront dataset collection.

### SayCan Affordance Model — Rejected

SayCan requires training a value function over (instruction, action, state) that
scores affordance-weighted actions. This is significantly more engineering effort.
The zero-shot Claude approach achieves comparable generalisation without a training dataset.

---

## Consequences

### Positive
- Zero-shot generalisation to novel instructions without any training data
- Handles ambiguity, clarification requests, and multi-step reasoning
- Development speed: task planner is functional from day 1 of Phase 14
- Safety filtering via Claude's alignment

### Negative
- Requires internet connection and `ANTHROPIC_API_KEY` for task planning
- API cost at scale (~$0.014/task) is negligible for development but non-trivial in production
- Latency (~1s) means task planning cannot be real-time reactive — acceptable for coarse planning
- Output JSON must be validated and sanitised before execution (Claude can hallucinate action names)

### Mitigations
- Implement a local fallback rule-based planner for the 10 most common task templates
- Cache Claude responses for identical (instruction, world_state) pairs
- Validate all Claude output against the action primitive schema before execution
- Log all Claude calls for debugging and future fine-tuning dataset creation

### Future Work
- Phase 17+: Fine-tune a small local model on logged Claude task plans for offline deployment
- Investigate Claude's tool use API for direct action dispatching (remove JSON parsing layer)
