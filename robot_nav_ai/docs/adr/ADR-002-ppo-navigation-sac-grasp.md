# ADR-002: PPO for Navigation, SAC for Grasping

**Status:** Accepted  
**Date:** 2026-05-18  
**Author:** AutoRobo Team  
**Deciders:** John Olugbemi

---

## Context

The system requires two distinct RL policies:

1. **Navigation policy** — mobile base drives to goal positions
2. **Grasp policy** — arm picks up objects from a tabletop

These are fundamentally different problems with different reward structures, action spaces,
and sample efficiency requirements. We need to select appropriate RL algorithms for each.

### Navigation Policy Characteristics

- **Reward structure:** Dense (continuous distance-to-goal reward at every step)
- **Action space:** Continuous but low-dimensional (2D: linear + angular velocity)
- **Episode length:** Medium (100–500 steps)
- **Sample efficiency:** Less critical — dense reward enables fast learning
- **Multiple parallel envs:** Strongly beneficial (more diverse experience)
- **On-policy acceptable:** Yes — navigation generalisation improves with on-policy data

### Grasp Policy Characteristics

- **Reward structure:** Sparse (reward only on successful lift — binary)
- **Action space:** Continuous and high-dimensional (6D end-effector delta + 1D gripper)
- **Episode length:** Short-to-medium (50–200 steps)
- **Sample efficiency:** Critical — sparse rewards are expensive to learn
- **Multiple parallel envs:** Less critical — replay buffer amortises sample cost
- **Off-policy required:** Yes — HER (Hindsight Experience Replay) requires replay buffer

---

## Decision

### Navigation: Proximal Policy Optimisation (PPO)

PPO is selected for navigation because:

1. **On-policy stability:** Dense reward navigation is well-solved by on-policy methods.
   PPO's clipped objective prevents destructive policy updates.

2. **Parallelisation:** PPO naturally benefits from many parallel environments (n_envs=8+).
   With 8 envs, PPO collects diverse navigation experiences across the map simultaneously.

3. **Simplicity:** PPO has few sensitive hyperparameters compared to SAC. This reduces
   tuning time during Phase 3 when we want to establish a working baseline quickly.

4. **Literature support:** PPO is the standard choice for navigation RL
   (e.g., DD-PPO in Habitat, NavigationNet). Our results are directly comparable.

5. **Memory efficiency:** On-policy methods do not require a large replay buffer,
   making PPO more practical when RAM is limited.

**Configuration:** `configs/training/ppo.yaml`  
**Implementation:** `scripts/train_nav.py`

### Grasping: Soft Actor-Critic (SAC)

SAC is selected for grasping because:

1. **Sample efficiency:** SAC is an off-policy algorithm that reuses past experience
   via a replay buffer. This is essential for sparse-reward grasping where successful
   grasps are rare events.

2. **Entropy maximisation:** SAC's maximum entropy objective encourages diverse exploration,
   which helps discover successful grasp strategies rather than collapsing to a single
   (possibly suboptimal) grasp approach.

3. **HER compatibility:** Hindsight Experience Replay (HER) requires an off-policy
   algorithm with a replay buffer. SAC + HER is the state-of-the-art combination for
   goal-conditioned manipulation (e.g., OpenAI Fetch, Franka kitchen tasks).

4. **Continuous action quality:** SAC learns stochastic policies that handle the
   precision required for finger-object contact better than deterministic off-policy
   methods (TD3) in practice.

5. **Automatic entropy tuning:** SAC automatically adjusts the temperature parameter,
   removing a critical hyperparameter that is hard to tune for sparse reward settings.

**Configuration:** `configs/training/sac.yaml`  
**Implementation:** `scripts/train_grasp.py`

---

## Alternatives Considered

| Algorithm | Navigation | Grasping | Reason Not Chosen |
|-----------|-----------|---------|-------------------|
| TD3 | Possible | Possible | Less stable than SAC; no entropy bonus |
| DDPG | Possible | Possible | Sample-inefficient; sensitive to hyperparams |
| A3C/IMPALA | Good | Poor | Poor replay buffer support, no HER |
| DreamerV3 | Good | Good | High implementation complexity; slower iteration |
| TD-MPC2 | Possible | Good | Requires world model; overkill for Phase 3 |

---

## Consequences

### Positive
- PPO + SAC is the most battle-tested combination for mobile manipulation in the literature
- Both algorithms have excellent Stable-Baselines3 implementations — no custom RL code needed
- SAC + HER directly addresses the sparse reward challenge in grasping
- Separate algorithms allow independent tuning and evaluation of nav and grasp

### Negative
- Two separate training pipelines require more engineering overhead
- SAC replay buffer requires ~8 GB RAM for 1M transitions with image observations
- Combining nav and grasp into a single end-to-end policy (future work) requires rethinking

### Future Consideration
- Phase 10+: Investigate using a single TD-MPC2 or DreamerV3 world model for both sub-tasks
- Phase 14+: Hierarchical RL with Claude API providing high-level goal selection
