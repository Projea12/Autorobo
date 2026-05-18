# AutoRobo Physical AI — System Architecture

**Project:** Language-Conditioned Mobile Manipulation System  
**Version:** 0.1.0

---

## Overview

AutoRobo is a 17-phase development plan to build a Physical AI system capable of
understanding natural language instructions and executing them through coordinated
mobile navigation and robotic manipulation in the real world.

The system takes a sentence like *"Pick up the red mug and put it on the shelf"*
and autonomously navigates, detects the object, grasps it, and places it correctly.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LANGUAGE INTERFACE                           │
│  User instruction → Claude API (task_planner.py)               │
│                   → TaskGraph (task_graph.py)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ structured task graph
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TASK EXECUTOR                                │
│  task_executor.py — dispatches to sub-systems                  │
│  ├── NavigationPolicy (PPO)                                     │
│  ├── GraspPolicy (SAC)                                          │
│  └── PlacePolicy (SAC / scripted)                               │
└────────┬────────────────────┬────────────────────────┬──────────┘
         │                    │                        │
         ▼                    ▼                        ▼
┌────────────────┐  ┌─────────────────────┐  ┌────────────────────┐
│  NAVIGATION    │  │    MANIPULATION     │  │    PERCEPTION      │
│                │  │                     │  │                    │
│ PPO policy     │  │ grasp_planner.py    │  │ ObjectDetector     │
│ LiDAR + odom  │  │ arm_controller.py   │  │ (YOLOv8)           │
│ → vel cmds    │  │ gripper_controller  │  │ ObjectSegmenter    │
│                │  │ workspace_limits    │  │ (SAM2)             │
└───────┬────────┘  └──────────┬──────────┘  │ DepthEstimator    │
        │                      │             │ (DepthAnything v2) │
        └──────────────────────┴─────────────│ GraspEstimator    │
                                             └─────────┬──────────┘
                                                       │
                           ┌───────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ROBOT INTERFACE LAYER                        │
│  interfaces/base_interface.py (abstract)                        │
│  ├── MuJoCoInterface   — simulation (Phases 1–15)               │
│  └── ROS2Interface     — real robot (Phase 17)                  │
└─────────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┴──────────────────┐
          ▼                                   ▼
┌──────────────────┐               ┌──────────────────────────┐
│  MEMORY          │               │  SAFETY & RECOVERY       │
│                  │               │                          │
│ world_memory.py  │               │ recovery_manager.py      │
│ (ChromaDB)       │               │ fault_detector.py        │
│ episode_logger   │               │ emergency_stop.py        │
│ self_improver    │               │ force_limits.py          │
└──────────────────┘               │ proximity_monitor.py     │
                                   └──────────────────────────┘
```

---

## 17-Phase Development Plan

### Phase 1 — Foundation & Environment Setup
**Goal:** Working MuJoCo environment and project skeleton.  
**Deliverables:** `interfaces/mujoco_interface.py`, scene XML, sanity check notebook.

### Phase 2 — Observation & Action Space Design
**Goal:** Define all obs/action spaces for navigation and manipulation.  
**Deliverables:** Gymnasium-compliant env wrappers, obs normalisation.

### Phase 3 — Navigation Policy (PPO)
**Goal:** Train mobile base to navigate to goal positions.  
**Deliverables:** `scripts/train_nav.py`, trained PPO model, SPL > 0.6.

### Phase 4 — Navigation Evaluation & Benchmarking
**Goal:** Measure SPL, success rate, collision rate systematically.  
**Deliverables:** `benchmarks/eval_navigation.py`, navigation benchmark report.

### Phase 5 — Scene Understanding
**Goal:** Build obstacle map, room segmentation, semantic labels.  
**Deliverables:** Semantic map data structure, obstacle detection integration.

### Phase 6 — Perception Stack Integration
**Goal:** Integrate YOLOv8 + SAM2 + DepthAnything v2.  
**Deliverables:** `perception/detector.py`, `segmenter.py`, `depth.py`, debug notebook.

### Phase 7 — Demonstration Data Collection
**Goal:** Collect oracle + keyboard demonstrations for grasp learning.  
**Deliverables:** `scripts/collect_data.py`, 1000+ episodes, HDF5 dataset.

### Phase 8 — Grasp Policy (SAC + HER)
**Goal:** Train arm to grasp YCB objects from tabletop.  
**Deliverables:** `scripts/train_grasp.py`, SAC model, >70% grasp success.

### Phase 9 — Grasp Evaluation & Failure Analysis
**Goal:** Per-object success rates, failure mode classification.  
**Deliverables:** `benchmarks/eval_grasping.py`, per-object breakdown.

### Phase 10 — Full Pipeline Integration (Nav → Grasp → Place)
**Goal:** Chain navigation, perception, grasping, and placing into one pipeline.  
**Deliverables:** `planning/task_executor.py`, end-to-end pick-and-place demo.

### Phase 11 — Memory & Episode Logging
**Goal:** Build world memory and self-improvement loop.  
**Deliverables:** `memory/world_memory.py` (ChromaDB), `episode_logger.py`.

### Phase 12 — Recovery System
**Goal:** Robust 5-level recovery hierarchy for failure handling.  
**Deliverables:** `recovery/recovery_manager.py`, `fault_detector.py`.

### Phase 13 — Safety Systems
**Goal:** Emergency stop, force limits, human proximity monitoring.  
**Deliverables:** `safety/` module, validated with deliberate fault injection.

### Phase 14 — Language-Conditioned Planning (Claude API)
**Goal:** Natural language instruction → structured task graph.  
**Deliverables:** `planning/task_planner.py` with Claude API integration.

### Phase 15 — Model Export & Optimisation
**Goal:** Export trained models to ONNX/TorchScript for deployment.  
**Deliverables:** `scripts/export_model.py`, deployment bundles.

### Phase 16 — Sim-to-Real Transfer
**Goal:** Transfer trained policies to physical robot.  
**Deliverables:** Domain randomisation, camera calibration, real-robot evaluation.

### Phase 17 — ROS2 Integration & Production Deployment
**Goal:** Full ROS2 Jazzy integration, production-grade deployment.  
**Deliverables:** `interfaces/ros2_interface.py`, ROS2 nodes, deployment runbook.

---

## Key Design Decisions

For detailed rationale on each design decision, see the ADR documents in `docs/adr/`:

| ADR | Decision |
|-----|----------|
| ADR-001 | MuJoCo over PyBullet |
| ADR-002 | PPO for navigation, SAC for grasping |
| ADR-003 | Abstract interface layer for ROS2 swap |
| ADR-004 | Claude API as language-to-task-graph planner |

---

## Data Flow: Full Pick-and-Place Task

```
User: "Pick up the banana and put it in the bowl"
            │
            ▼
    task_planner.py (Claude API)
            │
            ▼ TaskGraph:
    [find_object("banana")] → [navigate_to(banana_location)]
    → [grasp("banana")] → [find_object("bowl")]
    → [navigate_to(bowl_location)] → [place("banana", in="bowl")]
            │
            ▼
    task_executor.py iterates graph nodes:
    
    Node 1: find_object("banana")
      → perception/detector.py → bounding box
      → perception/depth.py → 3D position
      → world_memory.py → cache position
    
    Node 2: navigate_to(banana_location)
      → PPO policy → velocity commands → MuJoCoInterface.step()
      → recovery_manager if stuck
    
    Node 3: grasp("banana")
      → perception/grasp_estimator.py → grasp pose
      → manipulation/grasp_planner.py → approach trajectory
      → manipulation/arm_controller.py → joint commands
      → manipulation/gripper_controller.py → close gripper
      → fault_detector.py → verify grasp success
    
    ... (continue for place)
```

---

## Directory Structure

```
robot_nav_ai/
├── configs/          # Hydra YAML configs
├── scripts/          # Training and utility scripts
├── notebooks/        # Jupyter analysis notebooks
├── docs/             # Architecture docs, ADRs, runbook
├── benchmarks/       # Evaluation scripts
├── docker/           # Containerisation
├── interfaces/       # Swappable sim/real interfaces
├── perception/       # Computer vision stack
├── manipulation/     # Arm control and grasp planning
├── planning/         # Task planning (language → actions)
├── memory/           # World memory and episode logging
├── recovery/         # Fault tolerance and recovery
└── safety/           # Safety monitoring systems
```
