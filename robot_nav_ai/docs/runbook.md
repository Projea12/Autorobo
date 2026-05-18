# AutoRobo Physical AI — Operations Runbook

**Project:** Language-Conditioned Mobile Manipulation System  
**Version:** 0.1.0  
**Last updated:** 2026-05-18

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [First-Time Environment Setup](#2-first-time-environment-setup)
3. [Training Navigation Policy from Scratch (Phase 3)](#3-training-navigation-policy-from-scratch-phase-3)
4. [Training Grasp Policy from Scratch (Phase 8)](#4-training-grasp-policy-from-scratch-phase-8)
5. [Resuming Training from Checkpoint](#5-resuming-training-from-checkpoint)
6. [Running Evaluation Benchmarks](#6-running-evaluation-benchmarks)
7. [Visualising Results (W&B, TensorBoard)](#7-visualising-results-wb-tensorboard)
8. [Sim-to-Real Transfer (Phase 16)](#8-sim-to-real-transfer-phase-16)
9. [Troubleshooting Common Issues](#9-troubleshooting-common-issues)

---

## 1. Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8-core (e.g. M2 Pro) | 16-core (e.g. M3 Max / Ryzen 9) |
| RAM | 16 GB | 32 GB |
| GPU | None (CPU training) | NVIDIA RTX 3090 / A100 |
| Disk | 50 GB free | 200 GB SSD |
| Robot (real) | — | Mobile arm with RGB-D camera + LiDAR |

### Software Requirements

- **macOS 13+ (Sonoma/Sequoia)** or **Ubuntu 22.04 LTS**
- **Python 3.11** (exact version — 3.12 has breaking changes with some RL libs)
- **Git** 2.40+
- **Conda** (optional but recommended for environment isolation)

### macOS-Specific Notes

> **Critical:** PyBullet is incompatible with macOS 26 SDK (see ADR-001). This project uses MuJoCo exclusively.

MuJoCo 3.x installs cleanly on macOS via pip — no system library dependencies.  
Apple MPS (Metal Performance Shaders) provides ~3–5× speedup over CPU for neural network inference. Set `device=mps` in configs.

### Required API Keys / Tokens

| Service | Environment Variable | Required For |
|---------|---------------------|--------------|
| Anthropic | `ANTHROPIC_API_KEY` | Phase 14: Claude task planner |
| Weights & Biases | `WANDB_API_KEY` | Experiment tracking (optional) |

Set them in your shell profile:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export WANDB_API_KEY="..."
```

---

## 2. First-Time Environment Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/your-org/autorobo.git
cd autorobo/robot_nav_ai
```

### Step 2: Run the automated setup script

```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

This script:
- Checks your Python 3.11 installation
- Creates a virtual environment at `.venv/`
- Installs all dependencies (MuJoCo, SB3, Hydra, YOLO, Anthropic SDK, etc.)
- Installs the project in editable mode
- Verifies MuJoCo is working

### Step 3: Activate the environment

```bash
source .venv/bin/activate
```

Add this to your `.zshrc`/`.bashrc` to activate automatically:

```bash
# In ~/.zshrc
alias autorobo="source /path/to/autorobo/robot_nav_ai/.venv/bin/activate"
```

### Step 4: Verify the installation

```bash
# Check MuJoCo
python -c "import mujoco; print(mujoco.__version__)"

# Check SB3
python -c "import stable_baselines3; print(stable_baselines3.__version__)"

# Check Hydra
python -c "import hydra; print(hydra.__version__)"

# Run the env sanity check notebook
jupyter lab notebooks/01_env_sanity_check.ipynb
```

### Step 5: Download model checkpoints (perception)

```bash
mkdir -p models/perception

# YOLOv8 nano (auto-downloads on first inference via ultralytics)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# DepthAnything v2 small
# Download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Small
# Place at: models/perception/depth_anything_v2_vits.pth
```

### Step 6: Create directory structure for outputs

```bash
mkdir -p logs/{navigation,grasping,eval} models/{navigation,grasping,perception,exported} data/{demonstrations,eval}
```

---

## 3. Training Navigation Policy from Scratch (Phase 3)

### Overview

The navigation policy trains a PPO agent to drive the mobile base to goal positions
while avoiding obstacles. Input: LiDAR scan + goal vector + base odometry.
Output: linear velocity + angular velocity commands.

**Expected training time:** ~4 hours on 8-core CPU / ~45 minutes with GPU

### Quick Start

```bash
python scripts/train_nav.py
```

### Custom Configuration

Override any config value from the command line (Hydra syntax):

```bash
# Longer training
python scripts/train_nav.py training.total_timesteps=10000000

# More parallel environments (requires more RAM)
python scripts/train_nav.py training.n_envs=16

# Different learning rate
python scripts/train_nav.py training.ppo.learning_rate=1e-4

# Log to W&B with custom run name
python scripts/train_nav.py training.logging.wandb_run_name=nav_v2_longer
```

### Hyperparameter Sweep (Hydra multirun)

```bash
python scripts/train_nav.py --multirun \
  training.ppo.learning_rate=1e-4,3e-4,1e-3 \
  training.ppo.n_steps=1024,2048,4096
```

Results go to `multirun/YYYY-MM-DD/HH-MM-SS/`.

### Training Phases

| Phase | Steps | Expected Reward | Notes |
|-------|-------|----------------|-------|
| Exploration | 0–100K | Random (~0) | Agent learns basic movement |
| Early learning | 100K–500K | 1–5 | Starts reaching some goals |
| Convergence | 500K–2M | 8–12 | Consistent navigation |
| Fine-tuning | 2M–5M | 12–15 | Optimising path efficiency |

### Checkpoints

Checkpoints are saved every 50,000 steps to `models/navigation/ppo/`.  
The best model (by eval reward) is saved as `best_model.zip`.

### Success Criteria for Phase 3

- Navigation success rate > 80% (within 0.3m of goal)
- SPL (Success weighted by Path Length) > 0.6
- Collision rate < 5%

---

## 4. Training Grasp Policy from Scratch (Phase 8)

### Overview

The grasp policy trains a SAC agent to pick YCB objects from a tabletop.
Input: RGB-D image (84×84) + proprioception (joint positions + gripper state).
Output: end-effector delta pose + gripper open/close.

**Expected training time:** ~6 hours on GPU / ~24 hours on CPU

### Prerequisites

Collect demonstration data first (strongly recommended):

```bash
# Collect 1000 oracle demonstrations (~30 minutes)
python scripts/collect_data.py mode=oracle n_demos=1000

# Verify demos
python scripts/collect_data.py mode=replay
```

### Quick Start

```bash
python scripts/train_grasp.py training=sac
```

### Important: Sparse Reward Strategy

Grasping uses sparse rewards (reward only on success). This is hard to learn from scratch.  
Three techniques are applied:

1. **Demonstration seeding:** Pre-fill replay buffer with oracle demos
2. **Hindsight Experience Replay (HER):** Relabel failed episodes as successes toward achieved goals
3. **Reward shaping:** Dense distance reward + sparse success bonus (configurable)

To enable HER:

```bash
python scripts/train_grasp.py training.sac.use_her=true
```

### Grasp Training Phases

| Phase | Steps | Grasp Success | Notes |
|-------|-------|--------------|-------|
| Demo-seeded | 0–10K | ~30% | From demonstrations only |
| RL exploration | 10K–100K | 20–40% | May temporarily drop |
| RL learning | 100K–500K | 40–65% | Rapid improvement |
| Convergence | 500K–2M | 65–80% | Fine-tuning approach |

### Success Criteria for Phase 8

- Grasp success rate > 70% on YCB objects (in simulation)
- No single object class below 50% success rate
- Average grasp time < 10 seconds per attempt

---

## 5. Resuming Training from Checkpoint

### Automatic Resume (recommended)

The training scripts automatically detect and resume from the latest checkpoint
if the checkpoint directory exists:

```bash
# Just re-run the training script — it will find the latest checkpoint
python scripts/train_nav.py
python scripts/train_grasp.py training=sac
```

### Manual Checkpoint Specification

```bash
# Resume navigation from specific checkpoint
python scripts/train_nav.py \
  training.checkpoint.save_path=models/navigation/ppo/ \
  training.resume_from=models/navigation/ppo/ppo_nav_2500000_steps.zip

# Resume grasp from specific checkpoint
python scripts/train_grasp.py training=sac \
  training.resume_from=models/grasping/sac/sac_grasp_500000_steps.zip
```

### After Crash Recovery

If training crashes, identify the last valid checkpoint:

```bash
ls -lt models/navigation/ppo/*.zip | head -5
```

Then resume from the second-to-last checkpoint (the latest may be corrupt):

```bash
python scripts/train_nav.py \
  training.resume_from=models/navigation/ppo/ppo_nav_2450000_steps.zip
```

### Changing Hyperparameters on Resume

You can change hyperparameters when resuming (useful for fine-tuning):

```bash
# Resume with reduced learning rate (fine-tuning)
python scripts/train_nav.py \
  training.resume_from=models/navigation/ppo/best_model.zip \
  training.ppo.learning_rate=1e-5 \
  training.total_timesteps=1000000
```

---

## 6. Running Evaluation Benchmarks

### Full Evaluation Suite

```bash
python scripts/evaluate.py
```

Runs all three benchmarks and saves results to `logs/eval/eval_results.json`.

### Individual Benchmarks

```bash
# Navigation only
python benchmarks/eval_navigation.py

# Grasping only (YCB objects)
python benchmarks/eval_grasping.py

# Full pipeline (pick-and-place end-to-end)
python benchmarks/eval_full_pipeline.py
```

### Evaluation Configuration

```bash
# More evaluation episodes for statistical significance
python scripts/evaluate.py eval.n_episodes=100

# Specify model paths explicitly
python scripts/evaluate.py \
  eval.nav_model=models/navigation/ppo/best_model.zip \
  eval.grasp_model=models/grasping/sac/sac_grasp_final.zip

# Deterministic evaluation (no exploration noise)
python scripts/evaluate.py eval.deterministic=true
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| SPL | Success weighted by Path Length | > 0.6 |
| Navigation success rate | % episodes reaching goal | > 80% |
| Collision rate | % episodes with collision | < 5% |
| Grasp success rate | % grasps that lift object | > 70% |
| Task completion rate | % full pick-and-place tasks | > 60% |
| Average task time | Seconds per task | < 45s |

---

## 7. Visualising Results (W&B, TensorBoard)

### TensorBoard

```bash
# Navigation training logs
tensorboard --logdir logs/navigation/ppo/ --port 6006

# Grasping training logs
tensorboard --logdir logs/grasping/sac/ --port 6007

# All logs together
tensorboard --logdir logs/ --port 6006
```

Open `http://localhost:6006` in your browser.

**Key metrics to monitor:**
- `rollout/ep_rew_mean` — episode reward (should trend upward)
- `rollout/ep_len_mean` — episode length (should trend downward for nav)
- `train/value_loss` — value function loss (should decrease)
- `train/policy_gradient_loss` — should stay bounded
- `eval/mean_reward` — held-out evaluation reward

### Weights & Biases

Ensure `WANDB_API_KEY` is set, then training automatically logs to W&B.

```bash
# View runs in browser
wandb login
# Navigate to: https://wandb.ai/your-org/autorobo-navigation
```

**Useful W&B features for this project:**
- **Parallel coordinates plot:** compare hyperparameter sweeps
- **Run comparison:** overlay reward curves from multiple runs
- **Artifact versioning:** link model checkpoints to training runs
- **Alert on plateau:** set W&B alert if reward doesn't improve in 500K steps

### Jupyter Analysis

```bash
jupyter lab
```

Then open the analysis notebooks:
- `notebooks/02_reward_analysis.ipynb` — training curves
- `notebooks/04_grasp_success_rate.ipynb` — per-object success breakdown

---

## 8. Sim-to-Real Transfer (Phase 16)

### Overview

Sim-to-real transfer bridges the gap between the MuJoCo simulation and the physical robot.
The main challenges are:
- **Perception gap:** Real camera images differ from rendered images
- **Dynamics gap:** Real joint friction, mass properties differ from simulation
- **Latency:** Real robot has ~50ms communication latency vs near-zero in sim

### Step 1: Domain Randomisation (enable before Phase 16)

Enable domain randomisation in simulation training:

```bash
python scripts/train_grasp.py \
  env.randomise_lighting=true \
  env.mujoco.use_domain_rand=true \
  env.domain_rand.mass_range=[0.8, 1.2] \
  env.domain_rand.friction_range=[0.5, 2.0] \
  env.domain_rand.camera_noise=0.02
```

### Step 2: Real-World Camera Calibration

```bash
# Run camera calibration with checkerboard (ROS2 required)
ros2 run camera_calibration cameracalibrator \
  --size 8x6 --square 0.025 \
  --ros-args -r image:=/camera/rgb/image_raw
```

Update calibration in `configs/perception/yolo.yaml`:

```yaml
camera:
  fx: <measured>
  fy: <measured>
  cx: <measured>
  cy: <measured>
```

### Step 3: Switch to ROS2 Interface

Replace the MuJoCo interface with the ROS2 interface:

```bash
python scripts/evaluate.py interface=ros2
```

This uses `interfaces/ros2_interface.py` which publishes/subscribes to real robot topics.  
See ADR-003 for the interface design rationale.

### Step 4: Fine-Tune on Real Data

After initial real-world trials, collect failure episodes and fine-tune:

```bash
# Collect real-robot demonstrations
python scripts/collect_data.py mode=keyboard interface=ros2 n_demos=200

# Fine-tune grasp policy on real data
python scripts/train_grasp.py \
  training.resume_from=models/grasping/sac/best_sim_model.zip \
  training.total_timesteps=200000 \
  training.sac.learning_rate=1e-5 \
  data.use_real_demos=true
```

### Step 5: Safety Validation

Before autonomous operation on the real robot:

1. Verify all safety systems: `python -c "from safety.emergency_stop import EmergencyStop; EmergencyStop().test()"`
2. Confirm workspace limits: `cat configs/robot/base.yaml` — check x/y/z bounds
3. Test emergency stop physically — press E-stop, verify arm halts within 100ms
4. Run in shadow mode first: arm moves but gripper doesn't close
5. Gradually increase autonomy: manual → semi-auto → full auto

### Sim-to-Real Gap Mitigation Checklist

- [ ] Domain randomisation enabled during final sim training
- [ ] Camera calibrated with physical robot
- [ ] Robot URDF matches physical robot joint limits
- [ ] Contact physics validated on 3+ real object types
- [ ] Recovery system tested with deliberate failures
- [ ] Human proximity monitor tested (1.0m / 0.5m / 0.2m zones)
- [ ] Emergency stop tested and <100ms response confirmed
- [ ] All force limits validated against robot hardware specs

---

## 9. Troubleshooting Common Issues

### MuJoCo Issues

**Problem:** `mujoco.FatalError: gladLoadGL error`  
**Cause:** No display available (headless server)  
**Fix:**
```bash
export MUJOCO_GL=egl     # for servers with EGL
export MUJOCO_GL=osmesa  # for pure software rendering
# Or use offscreen rendering in config:
# env.render.mode=rgb_array
```

**Problem:** `mujoco.MujocoException: XML Error: ...`  
**Cause:** Scene XML file not found or malformed  
**Fix:** Check `assets/scenes/tabletop_scene.xml` path in `configs/env/mujoco.yaml`

---

### Training Issues

**Problem:** Reward stays flat at 0 for >500K steps  
**Cause:** Likely reward function not triggered, or env reset bug  
**Fix:**
1. Run `notebooks/01_env_sanity_check.ipynb` to verify env
2. Check `env.reward.navigation_success` is non-zero in config
3. Reduce `env.episode.max_steps` to force more resets
4. Enable dense reward shaping: `env.reward.distance_shaping=true`

**Problem:** Training crashes with `CUDA out of memory`  
**Cause:** Batch size too large or too many parallel envs  
**Fix:**
```bash
python scripts/train_nav.py training.ppo.batch_size=32 training.n_envs=4
```

**Problem:** PPO `value_loss` explodes (NaN)  
**Cause:** Learning rate too high or bad reward scale  
**Fix:**
```bash
python scripts/train_nav.py \
  training.ppo.learning_rate=1e-4 \
  training.ppo.max_grad_norm=0.3 \
  training.ppo.clip_range_vf=0.1
```

---

### Perception Issues

**Problem:** YOLOv8 detects 0 objects on tabletop  
**Cause:** Confidence threshold too high, or wrong object classes  
**Fix:** Lower threshold: `configs/perception/yolo.yaml` → `confidence_threshold: 0.3`

**Problem:** Depth estimation wildly inaccurate  
**Cause:** Model expecting metric depth but running in relative mode  
**Fix:** Ensure `metric_depth: true` in `configs/perception/yolo.yaml`

---

### Grasp Policy Issues

**Problem:** Grasp success rate stuck below 20% after 500K steps  
**Cause:** Sparse reward too hard without demonstrations  
**Fix:**
1. Collect oracle demonstrations: `python scripts/collect_data.py mode=oracle n_demos=2000`
2. Enable HER: `training.sac.use_her=true`
3. Enable reward shaping: `env.reward.distance_shaping=true`

**Problem:** Gripper closes but object falls immediately  
**Cause:** Grasp pose offset — gripper closing around object edge, not centre  
**Fix:** Adjust `perception.grasp_estimator.grasp_width_range` to match object size

---

### Real Robot Issues

**Problem:** ROS2 topics not publishing  
**Fix:**
```bash
ros2 topic list  # check topics exist
ros2 topic hz /camera/rgb/image_raw  # check camera is publishing
ros2 topic hz /joint_states  # check robot state is publishing
```

**Problem:** Emergency stop triggered spuriously  
**Cause:** Force limit too conservative  
**Fix:** Increase `safety.force_limits.max_joint_torque` in `configs/robot/base.yaml` (with caution)

**Problem:** Navigation works in sim but fails on real robot  
**Cause:** Sim-to-real gap in LiDAR scan format  
**Fix:** Check LiDAR frame_id in config matches ROS2 sensor frame. Verify angle_min/max match physical scanner.

---

*For issues not covered here, open a GitHub issue with:*
- *Full error traceback*
- *Hydra config (paste output of `python train_nav.py --cfg job`)*
- *Platform info (`uname -a`, `python --version`)*
- *Training step at which issue occurred*
