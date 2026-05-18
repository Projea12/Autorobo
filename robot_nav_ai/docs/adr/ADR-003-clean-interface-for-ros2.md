# ADR-003: Abstract Interface Layer for ROS2 Swappability

**Status:** Accepted  
**Date:** 2026-05-18  
**Author:** AutoRobo Team  
**Deciders:** John Olugbemi

---

## Context

The project has two distinct deployment targets:

1. **MuJoCo simulation** (Phases 1–15): all training and evaluation happens in simulation
2. **Physical robot via ROS2** (Phase 17): policies are deployed to a real robot

The challenge is to write all policy code, perception code, and task execution code
**once** and have it run on both the simulator and the real robot without modification.

### The Coupling Problem

A naive implementation would directly call MuJoCo APIs from training code:

```python
# BAD: tightly coupled to MuJoCo
import mujoco
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
obs = data.qpos  # directly access MuJoCo state
```

This cannot be swapped for ROS2 without rewriting every caller.

A naive ROS2 implementation calls ROS2 APIs directly:

```python
# BAD: tightly coupled to ROS2
import rclpy
from sensor_msgs.msg import JointState
# requires ROS2 to be installed even in sim
```

This prevents running on macOS (where ROS2 Jazzy is not installed) during development.

### Requirements

1. All policy/planning/perception code should call a **single, stable interface**
2. Swapping from simulation to real robot should require **one config change**,
   not code changes
3. The interface should be testable in isolation (without MuJoCo or ROS2 installed)
4. ROS2 dependencies should be **optional** — the project must install cleanly on macOS
   without ROS2

---

## Decision

**Implement an abstract base class `BaseRobotInterface` in `interfaces/base_interface.py`
with concrete implementations for MuJoCo and ROS2.**

```
interfaces/
├── __init__.py              # exports BaseRobotInterface
├── base_interface.py        # abstract base class
├── mujoco_interface.py      # Phase 1-15: simulation
└── ros2_interface.py        # Phase 17: real robot (conditional import)
```

The abstract interface defines exactly five methods:
- `reset() → dict` — reset environment, return initial observation
- `step(action) → tuple` — apply action, return (obs, reward, done, info)
- `get_observation() → dict` — return current sensor readings
- `apply_action(action)` — send commands to actuators
- `close()` — clean up resources

All policy, planning, and perception code takes a `BaseRobotInterface` argument.
The concrete implementation is selected at runtime via the Hydra config:

```yaml
# configs/env/mujoco.yaml
interface: mujoco

# To switch to real robot (Phase 17):
interface: ros2
```

The `ros2_interface.py` file uses **conditional imports**:

```python
try:
    import rclpy
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
```

This means `ros2_interface.py` can exist in the codebase without ROS2 installed —
it simply raises a helpful error if instantiated without ROS2.

---

## Alternatives Considered

### Alternative 1: Gymnasium Env Wrapper

Wrap everything in a `gymnasium.Env`. Both MuJoCo and ROS2 would be Gym envs.

**Rejected because:**
- Gymnasium's `step()` → `(obs, reward, done, truncated, info)` is training-centric
- Real robot deployment does not naturally map to episode-based Gym interface
- Gymnasium adds a dependency to the deployment runtime unnecessarily

### Alternative 2: ROS2 Everywhere (from Day 1)

Write all code as ROS2 nodes from the start, using `rosbag` for sim-to-real.

**Rejected because:**
- ROS2 Jazzy does not support macOS 26 cleanly (still in beta as of 2026-05-18)
- ROS2 adds massive development overhead for sim-only phases (1–15)
- Debugging is significantly harder with ROS2 node architecture vs plain Python

### Alternative 3: Separate Codebases

Maintain one codebase for sim, one for real robot, sync manually.

**Rejected because:**
- Any bug fix or feature must be applied twice
- Codebases diverge over time, defeating the purpose of sim-to-real

---

## Consequences

### Positive
- All policy/planning code is written once and works in both sim and real
- The interface can be mocked for unit testing without MuJoCo or ROS2
- ROS2 integration in Phase 17 requires only implementing `ros2_interface.py`
  — no changes to training or planning code
- Clean separation of concerns: the interface is the only layer that knows about
  the underlying hardware or simulator

### Negative
- Requires discipline to never call MuJoCo or ROS2 APIs directly from policy code
- The abstract interface must be carefully designed upfront — adding methods later
  requires updating all implementations
- Some MuJoCo-specific features (ground-truth state, reward from simulator) cannot
  be exposed through the interface without polluting the abstraction

### Interface Stability Commitment
The five abstract methods (`reset`, `step`, `get_observation`, `apply_action`, `close`)
are frozen from Phase 1. Any extension requires a new ADR.
