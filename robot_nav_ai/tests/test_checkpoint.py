"""
tests/test_checkpoint.py — Unit tests for utils/checkpoint.py.

All tests use a tmp_path fixture so no disk state leaks between runs.
PyTorch is used for save_torch/load_torch tests; SB3 tests are skipped if
either SB3 or MuJoCo is unavailable in the test environment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

# ── import under test ─────────────────────────────────────────────────────────
import sys
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils.checkpoint import CheckpointManager, make_run_dir

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    from env import ManipulationEnv
    HAS_MUJOCO = True
except Exception:
    HAS_MUJOCO = False


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mgr(tmp_path):
    return CheckpointManager(tmp_path / "run", keep_last=3)


@pytest.fixture
def torch_state():
    """Minimal PyTorch state dicts for save_torch tests."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    import torch.nn as nn
    net = nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    return dict(
        net    = net.state_dict(),
        opt    = opt.state_dict(),
        net_obj = net,
        opt_obj = opt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

def test_manager_creates_directory(tmp_path):
    run_dir = tmp_path / "myrun"
    mgr = CheckpointManager(run_dir)
    assert mgr.ckpt_root.exists()


def test_manager_stores_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    mgr = CheckpointManager(run_dir, keep_last=7)
    assert mgr.run_dir == run_dir
    assert mgr.keep_last == 7


# ─────────────────────────────────────────────────────────────────────────────
# make_run_dir
# ─────────────────────────────────────────────────────────────────────────────

def test_make_run_dir_creates_directory(tmp_path):
    d = make_run_dir(tmp_path, "exp")
    assert d.exists() and d.is_dir()


def test_make_run_dir_unique_on_repeated_calls(tmp_path):
    d1 = make_run_dir(tmp_path, "exp")
    time.sleep(1.1)   # ensure timestamp differs
    d2 = make_run_dir(tmp_path, "exp")
    assert d1 != d2


def test_make_run_dir_starts_with_run_name(tmp_path):
    d = make_run_dir(tmp_path, "mymodel")
    assert d.name.startswith("mymodel")


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch save / load
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_save_torch_creates_directory(mgr, torch_state):
    path = mgr.save_torch(step=1000, net=torch_state["net"], opt=torch_state["opt"])
    assert path.exists()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_save_torch_creates_pt_file(mgr, torch_state):
    path = mgr.save_torch(step=1000, net=torch_state["net"])
    assert (path / "torch.pt").exists()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_save_torch_creates_meta(mgr, torch_state):
    path = mgr.save_torch(step=2000, episode=5, net=torch_state["net"])
    meta = json.loads((path / "meta.json").read_text())
    assert meta["step"] == 2000
    assert meta["episode"] == 5


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_save_torch_extra_fields_in_meta(mgr, torch_state):
    mgr.save_torch(step=3000, extra={"best_return": 12.5}, net=torch_state["net"])
    ckpt = mgr.load_torch("latest")
    assert ckpt["meta"]["best_return"] == pytest.approx(12.5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_load_torch_latest(mgr, torch_state):
    mgr.save_torch(step=1000, net=torch_state["net"])
    mgr.save_torch(step=2000, net=torch_state["net"])
    ckpt = mgr.load_torch("latest")
    assert ckpt["meta"]["step"] == 2000


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_load_torch_named_step(mgr, torch_state):
    mgr.save_torch(step=5000, net=torch_state["net"])
    ckpt = mgr.load_torch("step_0000005000")
    assert ckpt["meta"]["step"] == 5000


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_load_torch_state_dict_roundtrip(mgr, torch_state):
    import torch
    import torch.nn as nn
    net = torch_state["net_obj"]
    mgr.save_torch(step=100, net=net.state_dict())
    ckpt = mgr.load_torch("latest")
    net2 = nn.Linear(4, 2)
    net2.load_state_dict(ckpt["net"])
    # weights must match
    for p1, p2 in zip(net.parameters(), net2.parameters()):
        assert torch.allclose(p1, p2)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_load_torch_optimizer_roundtrip(mgr, torch_state):
    import torch
    import torch.nn as nn
    net = torch_state["net_obj"]
    opt = torch_state["opt_obj"]
    mgr.save_torch(step=200, net=net.state_dict(), opt=opt.state_dict())
    ckpt = mgr.load_torch("latest")
    opt2 = torch.optim.Adam(nn.Linear(4, 2).parameters())
    opt2.load_state_dict(ckpt["opt"])
    assert "state" in opt2.state_dict()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_load_torch_nonexistent_raises(mgr):
    with pytest.raises(FileNotFoundError):
        mgr.load_torch("step_9999999999")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_load_torch_no_checkpoints_raises(mgr):
    with pytest.raises(FileNotFoundError):
        mgr.load_torch("latest")


# ─────────────────────────────────────────────────────────────────────────────
# best slot
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_is_best_creates_best_dir(mgr, torch_state):
    mgr.save_torch(step=1000, is_best=True, net=torch_state["net"])
    assert (mgr.ckpt_root / "best").exists()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_is_best_overwrites_previous_best(mgr, torch_state):
    mgr.save_torch(step=1000, is_best=True, net=torch_state["net"])
    mgr.save_torch(step=2000, is_best=True, net=torch_state["net"])
    ckpt = mgr.load_torch("best")
    assert ckpt["meta"]["step"] == 2000


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_load_best_no_best_raises(mgr):
    with pytest.raises(FileNotFoundError):
        mgr.load_torch("best")


# ─────────────────────────────────────────────────────────────────────────────
# Rotation (keep_last)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_rotation_keeps_at_most_keep_last(tmp_path, torch_state):
    mgr = CheckpointManager(tmp_path / "run", keep_last=2)
    for step in [1000, 2000, 3000, 4000]:
        mgr.save_torch(step=step, net=torch_state["net"])
    step_dirs = [d for d in mgr.ckpt_root.iterdir()
                 if d.is_dir() and d.name.startswith("step_")]
    assert len(step_dirs) == 2


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_rotation_keeps_latest_checkpoints(tmp_path, torch_state):
    mgr = CheckpointManager(tmp_path / "run", keep_last=2)
    for step in [1000, 2000, 3000]:
        mgr.save_torch(step=step, net=torch_state["net"])
    steps = sorted(
        int(d.name.split("_")[1])
        for d in mgr.ckpt_root.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    )
    assert steps == [2000, 3000]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_rotation_does_not_delete_best(tmp_path, torch_state):
    mgr = CheckpointManager(tmp_path / "run", keep_last=1)
    mgr.save_torch(step=1000, is_best=True, net=torch_state["net"])
    for step in [2000, 3000]:
        mgr.save_torch(step=step, net=torch_state["net"])
    assert (mgr.ckpt_root / "best").exists()


# ─────────────────────────────────────────────────────────────────────────────
# latest_step()
# ─────────────────────────────────────────────────────────────────────────────

def test_latest_step_none_when_empty(mgr):
    assert mgr.latest_step() is None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_latest_step_returns_last_saved(mgr, torch_state):
    mgr.save_torch(step=1000, net=torch_state["net"])
    mgr.save_torch(step=3000, net=torch_state["net"])
    assert mgr.latest_step() == 3000


# ─────────────────────────────────────────────────────────────────────────────
# list_checkpoints()
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_list_checkpoints_sorted(mgr, torch_state):
    for step in [3000, 1000, 2000]:
        mgr.save_torch(step=step, net=torch_state["net"])
    listing = mgr.list_checkpoints()
    steps = [e["step"] for e in listing]
    assert steps == sorted(steps)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_list_checkpoints_excludes_best(mgr, torch_state):
    mgr.save_torch(step=1000, is_best=True, net=torch_state["net"])
    listing = mgr.list_checkpoints()
    names = [e.get("path", "") for e in listing]
    assert not any("best" in n for n in names)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_list_checkpoints_has_path_field(mgr, torch_state):
    mgr.save_torch(step=500, net=torch_state["net"])
    listing = mgr.list_checkpoints()
    assert "path" in listing[0]


# ─────────────────────────────────────────────────────────────────────────────
# SB3 save / resume
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (HAS_SB3 and HAS_MUJOCO),
                    reason="SB3 or MuJoCo not available")
def test_save_sb3_creates_zip(tmp_path):
    mgr = CheckpointManager(tmp_path / "run", keep_last=3)
    env = make_vec_env(ManipulationEnv, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=0)
    path = mgr.save_sb3(model, step=1000)
    assert (path / "model.zip").exists()
    env.close()


@pytest.mark.skipif(not (HAS_SB3 and HAS_MUJOCO),
                    reason="SB3 or MuJoCo not available")
def test_save_sb3_creates_meta(tmp_path):
    mgr = CheckpointManager(tmp_path / "run", keep_last=3)
    env = make_vec_env(ManipulationEnv, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=0)
    path = mgr.save_sb3(model, step=5000, episode=10)
    meta = json.loads((path / "meta.json").read_text())
    assert meta["step"] == 5000
    assert meta["episode"] == 10
    env.close()


@pytest.mark.skipif(not (HAS_SB3 and HAS_MUJOCO),
                    reason="SB3 or MuJoCo not available")
def test_resume_sb3_latest(tmp_path):
    mgr = CheckpointManager(tmp_path / "run", keep_last=3)
    env = make_vec_env(ManipulationEnv, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=0)
    mgr.save_sb3(model, step=1000)
    mgr.save_sb3(model, step=2000)
    model2, meta = mgr.resume_sb3(PPO, env, ckpt_name="latest")
    assert meta["step"] == 2000
    assert model2 is not None
    env.close()


@pytest.mark.skipif(not (HAS_SB3 and HAS_MUJOCO),
                    reason="SB3 or MuJoCo not available")
def test_resume_sb3_preserves_policy_weights(tmp_path):
    """Weights loaded from checkpoint must match the saved model."""
    import torch
    mgr = CheckpointManager(tmp_path / "run", keep_last=3)
    env = make_vec_env(ManipulationEnv, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=0)
    mgr.save_sb3(model, step=1000)
    model2, _ = mgr.resume_sb3(PPO, env, ckpt_name="latest")
    p1 = list(model.policy.parameters())[0].detach()
    p2 = list(model2.policy.parameters())[0].detach()
    assert torch.allclose(p1, p2)
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# SB3CheckpointCallback
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not (HAS_SB3 and HAS_MUJOCO),
                    reason="SB3 or MuJoCo not available")
def test_sb3_callback_saves_during_learn(tmp_path):
    from utils.checkpoint import SB3CheckpointCallback
    mgr = CheckpointManager(tmp_path / "run", keep_last=5)
    env = make_vec_env(ManipulationEnv, n_envs=1)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
    cb = SB3CheckpointCallback(mgr, save_freq=128, verbose=0)
    model.learn(total_timesteps=256, callback=cb._cb)
    assert mgr.latest_step() is not None
    env.close()
