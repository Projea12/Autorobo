# ADR-001: MuJoCo Over PyBullet

**Status:** Accepted  
**Date:** 2026-05-18  
**Author:** AutoRobo Team  
**Deciders:** John Olugbemi

---

## Context

This project requires a physics simulator to train reinforcement learning agents for
mobile manipulation tasks. The primary candidates evaluated were:

- **PyBullet** (pybullet 3.x) — open-source, historically popular in robotics RL
- **MuJoCo** (mujoco 3.x) — developed by DeepMind, industry standard for dexterous manipulation
- **Isaac Sim** (NVIDIA) — GPU-native, requires NVIDIA hardware

### PyBullet Disqualifier: macOS 26 SDK Incompatibility

PyBullet 3.x has a critical incompatibility with the macOS 26 SDK (Xcode 26 / Apple Clang 18).
The build fails during `pip install pybullet` with linker errors related to deprecated
C runtime symbols removed in macOS 26:

```
ld: warning: ignoring file libBulletDynamics.a: file was built for archive which is not the architecture being linked (x86_64)
clang: error: linker command failed with exit code 1
```

This is a known upstream issue with no ETA for a fix. PyBullet's last release was in 2022
and the project has low maintenance activity. Workarounds (Rosetta 2, conda-forge builds)
add unacceptable complexity to the development environment.

### MuJoCo Advantages

1. **macOS compatibility:** MuJoCo 3.x ships as a pre-built wheel for macOS (arm64 and x86_64).
   Installs cleanly on macOS 26 via `pip install mujoco`. Zero system dependencies.

2. **Contact physics quality:** MuJoCo uses a convex optimisation-based contact solver
   (via the Duality Gap contact model) that produces significantly more realistic and
   stable contact forces than PyBullet's iterative LCP solver. This is critical for
   precise grasp simulation — finger-object contacts must be physically accurate for
   sim-to-real transfer.

3. **MJX GPU acceleration:** MuJoCo 3.x ships with MJX (MuJoCo XLA), a JAX-based
   reimplementation of the MuJoCo physics engine that runs on GPU/TPU. This enables
   thousands of parallel simulation environments for large-scale RL training (Phase 10+).

4. **DeepMind/Google support:** MuJoCo is actively maintained with monthly releases.
   The Python bindings are first-class (not an afterthought).

5. **Robotics ecosystem:** dm_control, Gymnasium's MuJoCo envs, and MuJoCo Menagerie
   (pre-built robot models) are all built on MuJoCo. Extensive community support.

6. **Benchmark standard:** Almost all state-of-the-art manipulation RL papers
   (IQL, TD-MPC2, DreamerV3) use MuJoCo environments. Our benchmarks are directly
   comparable to the literature.

### Isaac Sim Rejection

Isaac Sim was rejected because:
- Requires NVIDIA GPU (not available on the primary development machine, Apple Silicon)
- Licence complexity and large installation footprint (>50 GB)
- Python API is less mature for custom environment development

---

## Decision

**Use MuJoCo 3.x as the sole physics simulator for all training and evaluation.**

- Primary interface: `interfaces/mujoco_interface.py`
- Scene description: MJCF XML files in `assets/scenes/`
- Robot models: MJCF from MuJoCo Menagerie (and custom MJCF for the mobile base)
- GPU training (Phase 10+): MJX backend via `env.mujoco.use_gpu=true`

---

## Consequences

### Positive
- Clean cross-platform development (macOS + Linux) with a single `pip install mujoco`
- Superior contact physics improves grasp simulation fidelity and sim-to-real transfer
- MJX enables scaling to 1000+ parallel envs on a single GPU in later phases
- Our results are comparable to published RL manipulation benchmarks

### Negative
- MJCF XML learning curve for custom scene/robot authoring (vs URDF which PyBullet uses)
- Some existing robot models are in URDF format — conversion tool (`mujoco.mjcf.from_urdf`) required
- MJX (GPU mode) requires JAX installation in addition to PyTorch (for RL training)

### Neutral
- ROS2 (Phase 17) uses URDF natively; we maintain separate URDF for ROS2 while using MJCF in sim
