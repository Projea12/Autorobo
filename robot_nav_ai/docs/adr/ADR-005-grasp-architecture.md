# ADR-005: Grasp Architecture — Learned from Scratch with SAC

**Status:** Accepted  
**Date:** 2026-05-20  
**Author:** AutoRobo Team  
**Deciders:** John Olugbemi

---

## Context

Phase 5 requires a grasp policy that takes object perception output (3D position,
segmentation mask, point cloud) and produces arm commands that successfully pick up
objects of varying shapes, sizes, and poses.

Three candidate architectures were evaluated:

### Candidate 1: GraspNet (Fang et al., 2020)
A deep network that predicts dense 6DOF grasp poses directly from a point cloud.
Trained on the GraspNet-1Billion dataset (1B annotated real-world grasps).

### Candidate 2: Contact-GraspNet (Sundermeyer et al., 2021)
Predicts grasp poses from partial point clouds by reasoning about contact geometry.
Trained on simulated grasps + real-world fine-tuning. State-of-the-art on YCB-Video.

### Candidate 3: Learned from Scratch with SAC
Train a SAC policy end-to-end in MuJoCo. The policy takes perception features as
observations and outputs delta end-effector poses. No pre-trained grasp model required.

---

## Decision

**Learned from scratch with SAC** (Candidate 3).

---

## Justification

### Why not GraspNet

| Issue | Detail |
|---|---|
| Dataset dependency | Requires GraspNet-1Billion (real-world scans, ~30GB). Not reproducible in pure sim. |
| Sim mismatch | Trained on real depth cameras. MuJoCo synthetic depth is structurally different. |
| No RL loop | GraspNet outputs pose candidates — execution still requires a separate controller. |
| Integration cost | Would require a full point cloud processing pipeline (Open3D, normals, voxelisation) before any grasp candidate is produced. |

### Why not Contact-GraspNet

| Issue | Detail |
|---|---|
| Pre-training required | Depends on ShapeNet + real fine-tuning data. Cannot train from sim alone. |
| Partial point cloud assumptions | Designed for single-view real depth cameras, not MuJoCo rendered depth. |
| Inference latency | Contact-GraspNet adds 80–150ms inference — already at our 100ms budget before execution. |
| Complexity | Two-stage pipeline (grasp prediction → IK → execution) adds failure modes at each stage. |

### Why SAC from scratch

1. **Full sim compatibility.** SAC trains entirely in MuJoCo — no external dataset required.
   The policy learns what works in our specific simulator with our specific robot.

2. **Already decided (ADR-002).** The grasping algorithm is SAC + HER. Adding a separate
   grasp network on top would create two competing systems.

3. **Natural perception integration.** Phase 4 outputs (3D position, depth score, seg mask)
   feed directly as SAC observation features. No additional point cloud processing.

4. **Delta control.** SAC learns small incremental end-effector corrections, which is
   more robust than predicting absolute grasp poses (less sensitive to pose estimation error).

5. **Failure recovery compatible.** SAC's stochastic policy naturally produces varied
   retry attempts. GraspNet would produce the same (failed) grasp pose on retry.

6. **Sim-to-real path.** Domain randomisation (friction, object mass, visual textures) +
   delta control gives the best-known sim-to-real transfer for manipulation.
   Contact-GraspNet's sim-to-real gap has not been characterised for MuJoCo synthetic data.

---

## Architecture

```
Observation (per step)
──────────────────────
  object_xyz        (3,)   — 3D position from depth projector
  object_bbox_norm  (4,)   — normalised bbox [x1,y1,x2,y2]
  seg_score         (1,)   — mask quality from confidence aggregator
  ee_pos            (3,)   — current end-effector position (world frame)
  ee_quat           (4,)   — current end-effector orientation (quaternion)
  gripper_state     (1,)   — current gripper opening [0=closed, 1=open]
  joint_pos         (6,)   — arm joint angles
  joint_vel         (6,)   — arm joint velocities
  contact_force     (3,)   — wrist force-torque sensor reading
  ─────────────────────────
  Total             (31,)

Action (delta, per step)
────────────────────────
  Δx, Δy, Δz       (3,)   — end-effector position delta, scaled ±5 cm
  Δroll, Δpitch, Δyaw (3,) — orientation delta, scaled ±15°
  gripper           (1,)   — [-1=open, +1=close], threshold at 0.0
  ─────────────────────────
  Total             (7,)   — all in [-1, 1]

Policy
──────
  SAC with HER (Hindsight Experience Replay)
  MLP: [256, 256] hidden, ReLU, LayerNorm
  Replay buffer: 500k transitions
  Batch size: 256
```

---

## Alternatives Considered

| Architecture | Dataset needed | Sim-to-real | Integration | Latency | Decision |
|---|---|---|---|---|---|
| GraspNet | GraspNet-1B (real) | Poor (sim mismatch) | Complex | ~50ms | Rejected |
| Contact-GraspNet | ShapeNet + real | Uncharacterised | Complex | 80–150ms | Rejected |
| SAC from scratch | None (sim only) | Good (domain rand) | Native | <5ms | **Accepted** |

---

## Consequences

### Positive
- No external dataset dependency — fully reproducible from simulation
- Perception → action pipeline is end-to-end learnable
- Natural failure recovery through stochastic policy
- Compatible with Phase 8 recovery hierarchy (Level 1: SAC retry, Level 2: SAC replan)
- Inference latency < 5ms (MLP policy), well within 100ms budget

### Negative
- Cold start: SAC needs ~500k steps before grasps succeed (mitigated by HER + demo seeding)
- Cannot leverage pre-trained grasp knowledge from millions of real grasps
- Policy is specific to our robot's kinematics — must retrain for different hardware

### Future Consideration
- Phase 13+: If success rate plateaus below 80%, supplement SAC with a Contact-GraspNet
  initialisation on simulated grasps to warm-start the policy
- Phase 16+: Fine-tune on real robot data using the sim policy as starting point
