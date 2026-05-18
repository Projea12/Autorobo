"""
Centralised seeding — call set_global_seed(seed) once at the start of every
training run or evaluation session. Covers every RNG that can break
reproducibility across NumPy, PyTorch, Python stdlib, and MuJoCo.

MuJoCo's own randomness comes entirely from NumPy (mjData is deterministic
given fixed model + ctrl). The Gymnasium env receives the seed through
reset(seed=...) which sets env.np_random internally via super().reset(seed=seed).
Pass the same SeedConfig.env_seed to env.reset() to close the loop.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeedConfig:
    """All seeds derived from a single root integer."""

    root: int
    numpy_seed: int = field(init=False)
    torch_seed: int = field(init=False)
    env_seed: int = field(init=False)
    python_seed: int = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.root)
        seeds = rng.integers(0, 2**31, size=4)
        self.numpy_seed = int(seeds[0])
        self.torch_seed = int(seeds[1])
        self.env_seed = int(seeds[2])
        self.python_seed = int(seeds[3])


def set_global_seed(seed: int) -> SeedConfig:
    """
    Seed every RNG in the process from a single integer.

    Call once before constructing any env, model, or data loader.
    Returns the SeedConfig so callers can pass env_seed to env.reset().

    Example
    -------
    >>> cfg = set_global_seed(42)
    >>> obs, _ = env.reset(seed=cfg.env_seed)
    >>> env.action_space.seed(cfg.env_seed)   # action_space has its own RNG
    """
    cfg = SeedConfig(root=seed)

    # Python stdlib
    random.seed(cfg.python_seed)

    # Hash randomisation — must be set before process forks
    os.environ["PYTHONHASHSEED"] = str(cfg.python_seed)

    # NumPy legacy API (used by SB3 internally)
    np.random.seed(cfg.numpy_seed)

    # PyTorch — import is deferred so the module works without torch installed
    try:
        import torch

        torch.manual_seed(cfg.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.torch_seed)
        # Makes CUDA ops deterministic at a small speed cost
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return cfg
