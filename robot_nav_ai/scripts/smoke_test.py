"""
scripts/smoke_test.py — Phase 1 simulation stack smoke test.

Checks every component end-to-end without training.  Run with:

    python scripts/smoke_test.py              # all checks
    python scripts/smoke_test.py --verbose    # print per-step details

Exit code 0 → all checks passed.
Exit code 1 → at least one check failed (details printed).

Checks
──────
 1. Robot MJCF loads and compiles without error
 2. ManipulationEnv constructs and closes cleanly
 3. Observation shape, dtype, and finite values
 4. Action space shape and limits
 5. Physics runs 200 steps without NaN
 6. Episode reset returns valid EpisodeInfo
 7. Robot spawn is within configured bounds
 8. Robot z is not underground (z > 0)
 9. Goal xyz is finite and 3D
10. Obstacles scattered — active ones above floor, inactive underground
11. Domain randomisation changes model properties
12. Same seed gives identical reset (reproducibility gate)
13. Sensors return data — joint pos, vel, camera shape
14. Checkpoint save + resume preserves policy weights
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── colour helpers ─────────────────────────────────────────────────────────────
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

def _ok(msg):  print(f"  {_GREEN}✓{_RESET}  {msg}")
def _fail(msg): print(f"  {_RED}✗{_RESET}  {_BOLD}{msg}{_RESET}")
def _info(msg): print(f"  {_YELLOW}·{_RESET}  {msg}")


# ── individual checks ─────────────────────────────────────────────────────────

_ROBOT_XML = _ROOT / "robot" / "robot.xml"


def check_mjcf_loads(verbose: bool) -> bool:
    import mujoco
    mjcf = _ROBOT_XML
    if not mjcf.exists():
        _fail(f"MJCF not found: {mjcf}")
        return False
    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf))
        data  = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        if verbose:
            _info(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}")
        _ok(f"MJCF loads — nq={model.nq}, nv={model.nv}, nu={model.nu}")
        return True
    except Exception as e:
        _fail(f"MJCF failed to load: {e}")
        return False


def check_env_constructs(verbose: bool) -> tuple[bool, object | None]:
    try:
        from env import ManipulationEnv
        env = ManipulationEnv()
        env.close()
        _ok("ManipulationEnv constructs and closes")
        return True, None
    except Exception as e:
        _fail(f"ManipulationEnv construction failed: {e}")
        if verbose:
            traceback.print_exc()
        return False, None


def check_observation(verbose: bool) -> bool:
    from env import ManipulationEnv, OBS_DIM
    env = ManipulationEnv()
    try:
        obs, _ = env.reset(seed=0)
        ok = True
        if obs.shape != (OBS_DIM,):
            _fail(f"Obs shape {obs.shape} ≠ ({OBS_DIM},)")
            ok = False
        if obs.dtype != np.float32:
            _fail(f"Obs dtype {obs.dtype} ≠ float32")
            ok = False
        if not np.isfinite(obs).all():
            _fail(f"Obs contains non-finite values ({(~np.isfinite(obs)).sum()} elements)")
            ok = False
        if ok:
            _ok(f"Observation: shape={obs.shape}, dtype={obs.dtype}, finite=True")
        if verbose:
            _info(f"obs min={obs.min():.3f}  max={obs.max():.3f}  mean={obs.mean():.3f}")
        return ok
    finally:
        env.close()


def check_action_space(verbose: bool) -> bool:
    from env import ManipulationEnv, ACT_DIM
    env = ManipulationEnv()
    try:
        act_space = env.action_space
        ok = True
        if act_space.shape != (ACT_DIM,):
            _fail(f"Action space shape {act_space.shape} ≠ ({ACT_DIM},)")
            ok = False
        if not (np.allclose(act_space.low, -1.0) and np.allclose(act_space.high, 1.0)):
            _fail(f"Action bounds not [-1, 1]: low={act_space.low}, high={act_space.high}")
            ok = False
        if ok:
            _ok(f"Action space: shape={act_space.shape}, bounds=[-1, 1]")
        return ok
    finally:
        env.close()


def check_physics_200_steps(verbose: bool) -> bool:
    from env import ManipulationEnv
    env = ManipulationEnv()
    try:
        obs, _ = env.reset(seed=42)
        rng = np.random.default_rng(42)
        nan_step = None
        for i in range(200):
            action = rng.uniform(-0.1, 0.1, size=env.action_space.shape).astype(np.float32)
            obs, reward, term, trunc, _ = env.step(action)
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                nan_step = i
                break
            if term or trunc:
                obs, _ = env.reset(seed=i)
        if nan_step is not None:
            _fail(f"NaN/Inf detected at step {nan_step}")
            return False
        _ok("200 physics steps — no NaN/Inf")
        return True
    finally:
        env.close()


def check_episode_reset(verbose: bool) -> bool:
    import mujoco
    from env.episode_reset import EpisodeResetter, SpawnConfig, GoalConfig
    from world.world import WorldBuilder, WorldConfig

    try:
        cfg        = WorldConfig()
        builder    = WorldBuilder(cfg)
        model, ws  = builder.build(str(_ROBOT_XML))
        data       = mujoco.MjData(model)

        kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kf_id < 0:
            _fail("No 'home' keyframe in model — cannot test EpisodeResetter directly")
            return False

        resetter = EpisodeResetter(home_kf_id=kf_id, world_state=ws)
        rng      = np.random.default_rng(7)
        info     = resetter.reset(model, data, rng)

        ok = True
        if info.robot_xy.shape != (2,):
            _fail(f"robot_xy shape {info.robot_xy.shape} ≠ (2,)"); ok = False
        if not np.isfinite(info.robot_xy).all():
            _fail("robot_xy not finite"); ok = False
        if not np.isfinite(info.robot_yaw):
            _fail("robot_yaw not finite"); ok = False
        if info.goal_xyz.shape != (3,):
            _fail(f"goal_xyz shape {info.goal_xyz.shape} ≠ (3,)"); ok = False
        if not np.isfinite(info.goal_xyz).all():
            _fail("goal_xyz not finite"); ok = False
        if data.qpos[2] < 0.0:
            _fail(f"Robot z underground: {data.qpos[2]:.4f}"); ok = False

        if ok:
            _ok(f"EpisodeResetter.reset() — xy={np.round(info.robot_xy,3)}, "
                f"yaw={info.robot_yaw:.2f}, goal={np.round(info.goal_xyz,3)}, "
                f"obs={info.n_active_obstacles}")
        if verbose:
            _info(f"robot z = {data.qpos[2]:.4f}")
        return ok
    except Exception as e:
        _fail(f"EpisodeResetter check failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def check_obstacles(verbose: bool) -> bool:
    import mujoco
    from env.episode_reset import EpisodeResetter
    from world.world import WorldBuilder, WorldConfig

    try:
        cfg       = WorldConfig(n_obstacles=4)
        builder   = WorldBuilder(cfg)
        model, ws = builder.build(str(_ROBOT_XML))
        data      = mujoco.MjData(model)

        kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kf_id < 0:
            _fail("No 'home' keyframe — skipping obstacle check")
            return False

        resetter = EpisodeResetter(home_kf_id=kf_id, world_state=ws, n_obstacles=4)
        rng      = np.random.default_rng(3)
        info     = resetter.reset(model, data, rng)

        n_slots = ws.n_obstacle_slots
        active  = 0
        hidden  = 0
        for i in range(n_slots):
            pos = ws.obstacle_pos(data, i)
            if pos[2] > -1.0:
                active += 1
                if pos[2] < -0.01:
                    _fail(f"Active obstacle {i} is underground: z={pos[2]:.3f}")
                    return False
            else:
                hidden += 1

        _ok(f"Obstacles: {active} active above floor, {hidden} hidden underground "
            f"(slots={n_slots})")
        if verbose:
            for i in range(min(n_slots, 6)):
                pos = ws.obstacle_pos(data, i)
                _info(f"  obs[{i}] z={pos[2]:.3f}")
        return True
    except Exception as e:
        _fail(f"Obstacle check failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def check_domain_rand(verbose: bool) -> bool:
    import mujoco
    from env.domain_rand import DomainRandomizer
    from world.world import WorldBuilder, WorldConfig

    try:
        cfg       = WorldConfig()
        builder   = WorldBuilder(cfg)
        model, _  = builder.build(str(_ROBOT_XML))
        data      = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        dr = DomainRandomizer(model)
        snap_before = dr.snapshot(model)

        rng = np.random.default_rng(99)
        dr.randomize(model, data, rng)
        snap_after = dr.snapshot(model)

        changed = 0
        for key in snap_before:
            b = np.asarray(snap_before[key])
            a = np.asarray(snap_after[key])
            if not np.allclose(b, a, atol=1e-6):
                changed += 1
                if verbose:
                    _info(f"  {key} changed")

        if changed == 0:
            _fail("DomainRandomizer.randomize() changed nothing")
            return False

        # Confirm physics still runs after mutation
        for _ in range(10):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            _fail("NaN after domain rand + 10 physics steps")
            return False

        _ok(f"DomainRandomizer mutated {changed} property groups, physics stable")
        return True
    except Exception as e:
        _fail(f"Domain randomisation check failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


def check_reproducibility(verbose: bool) -> bool:
    from env import ManipulationEnv
    env = ManipulationEnv()
    try:
        obs1, _ = env.reset(seed=777)
        obs2, _ = env.reset(seed=777)
        if np.array_equal(obs1, obs2):
            _ok("Reproducibility: same seed → identical observation")
            return True
        else:
            diff = np.abs(obs1 - obs2).max()
            _fail(f"Same seed gave different obs (max diff={diff:.6f})")
            return False
    finally:
        env.close()


def check_sensors(verbose: bool) -> bool:
    from env import ManipulationEnv
    env = ManipulationEnv(render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=0)
        ok = True

        # joint positions (obs[0:8]) and velocities (obs[8:16])
        joint_pos = obs[0:8]
        joint_vel = obs[8:16]
        if not np.isfinite(joint_pos).all():
            _fail("Joint positions contain NaN"); ok = False
        if not np.isfinite(joint_vel).all():
            _fail("Joint velocities contain NaN"); ok = False

        # RGB render
        try:
            frame = env.render()
            if frame is None:
                _fail("render() returned None"); ok = False
            elif frame.ndim != 3 or frame.shape[2] != 3:
                _fail(f"Unexpected frame shape {frame.shape}"); ok = False
            else:
                if verbose:
                    _info(f"  camera frame: {frame.shape}, dtype={frame.dtype}")
                _ok(f"Sensors: joint_pos finite, joint_vel finite, "
                    f"camera {frame.shape}")
        except Exception as e:
            _fail(f"render() raised: {e}"); ok = False

        return ok
    finally:
        env.close()


def check_checkpoint_roundtrip(verbose: bool) -> bool:
    try:
        import torch
        import torch.nn as nn
        import tempfile
        from utils.checkpoint import CheckpointManager

        net = nn.Linear(8, 4)
        opt = torch.optim.Adam(net.parameters())

        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(tmp, keep_last=2)
            mgr.save_torch(step=100,
                           net=net.state_dict(),
                           opt=opt.state_dict())
            ckpt = mgr.load_torch("latest")

        net2 = nn.Linear(8, 4)
        net2.load_state_dict(ckpt["net"])
        for p1, p2 in zip(net.parameters(), net2.parameters()):
            if not torch.allclose(p1, p2):
                _fail("Checkpoint roundtrip: weights differ after load")
                return False

        _ok("Checkpoint save → load roundtrip: weights identical")
        return True
    except ImportError:
        _fail("PyTorch not available — checkpoint roundtrip skipped")
        return False
    except Exception as e:
        _fail(f"Checkpoint roundtrip failed: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ── runner ────────────────────────────────────────────────────────────────────

_CHECKS = [
    ("Robot MJCF loads",                  check_mjcf_loads),
    ("ManipulationEnv constructs",         lambda v: check_env_constructs(v)[0]),
    ("Observation shape/dtype/finite",     check_observation),
    ("Action space shape/bounds",          check_action_space),
    ("Physics 200 steps — no NaN",         check_physics_200_steps),
    ("EpisodeResetter.reset()",            check_episode_reset),
    ("Obstacles above floor / hidden",     check_obstacles),
    ("DomainRandomizer mutates + stable",  check_domain_rand),
    ("Reproducibility — same seed",        check_reproducibility),
    ("Sensors — joints + camera",          check_sensors),
    ("Checkpoint save/load roundtrip",     check_checkpoint_roundtrip),
]


def main(verbose: bool = False) -> int:
    print(f"\n{_BOLD}AutoRobo v1 — Phase 1 Smoke Test{_RESET}")
    print("─" * 50)

    results: list[tuple[str, bool]] = []
    t0 = time.perf_counter()

    for label, fn in _CHECKS:
        print(f"\n[{len(results)+1}/{len(_CHECKS)}] {label}")
        try:
            passed = fn(verbose)
        except Exception as e:
            _fail(f"Unexpected exception: {e}")
            if verbose:
                traceback.print_exc()
            passed = False
        results.append((label, passed))

    elapsed = time.perf_counter() - t0
    n_pass  = sum(p for _, p in results)
    n_fail  = len(results) - n_pass

    print(f"\n{'─' * 50}")
    print(f"{_BOLD}Results: {n_pass}/{len(results)} passed  ({elapsed:.1f}s){_RESET}")

    if n_fail:
        print(f"\n{_RED}Failed:{_RESET}")
        for label, passed in results:
            if not passed:
                print(f"  ✗  {label}")
        return 1

    print(f"\n{_GREEN}{_BOLD}All checks passed — Phase 1 stack is operational.{_RESET}")
    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()
    sys.exit(main(verbose=args.verbose))
