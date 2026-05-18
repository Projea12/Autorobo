"""
base_interface.py — Abstract Robot Interface (ADR-003)

Defines the contract that all robot interface implementations must satisfy.
All policy, planning, and perception code interacts only with this interface —
never with MuJoCo or ROS2 APIs directly.

This enables swapping the underlying implementation (simulation → real robot)
with a single config change. See ADR-003 for rationale.

Implementations:
    MuJoCoInterface  — mujoco_interface.py  (Phases 1–15)
    ROS2Interface    — ros2_interface.py    (Phase 17)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRobotInterface(ABC):
    """
    Abstract base class for all robot interfaces.

    Provides a uniform API over simulation (MuJoCo) and real robot (ROS2).
    All five abstract methods must be implemented by subclasses.

    Example usage:
        interface: BaseRobotInterface = MuJoCoInterface(cfg)
        obs = interface.reset()
        for _ in range(1000):
            action = policy.predict(obs)
            obs, reward, done, info = interface.step(action)
            if done:
                break
        interface.close()
    """

    def __init__(self, cfg: Any) -> None:
        """
        Initialise the interface with a Hydra config.

        Args:
            cfg: Hydra DictConfig (or any config object) with robot/env settings.
        """
        self.cfg = cfg
        self._is_closed = False

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """
        Reset the environment to an initial state.

        This is called at the start of each episode. The implementation
        should randomise object positions, robot pose, and any other
        stochastic elements according to the config.

        Returns:
            Initial observation dict. Keys match the observation space
            defined in the config:
            {
                "rgb": np.ndarray of shape (H, W, 3), dtype=uint8,
                "depth": np.ndarray of shape (H, W), dtype=float32,
                "lidar": np.ndarray of shape (N,), dtype=float32,
                "proprioception": np.ndarray of shape (D,), dtype=float32,
            }
        """
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Apply an action and advance the environment by one control step.

        Args:
            action: Action to apply. Format depends on the active policy:
                - Navigation: np.ndarray [linear_vel, angular_vel]
                - Grasping: np.ndarray [dx, dy, dz, droll, dpitch, dyaw, gripper]

        Returns:
            Tuple of (observation, reward, done, info):
                observation: dict matching reset() return format
                reward: float reward for this step
                done: True if episode has ended (success or failure)
                info: dict with extra info:
                    {
                        "success": bool,
                        "collision": bool,
                        "timeout": bool,
                        "distance_to_goal": float,
                    }
        """
        ...

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """
        Return the current observation without stepping the simulation.

        Useful for initialising perception at the start of an episode
        before the first action is taken.

        Returns:
            Current observation dict (same format as reset() return value).
        """
        ...

    @abstractmethod
    def apply_action(self, action: Any) -> None:
        """
        Send a command to the robot actuators without computing reward or done.

        Used in real-time control loops where reward/done computation is
        handled externally (e.g., by the safety monitor or task executor).

        Args:
            action: Action in the same format as step().
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Release all resources held by this interface.

        Should be called when training or evaluation is complete.
        For MuJoCo: frees the model/data memory.
        For ROS2: shuts down the ROS2 node cleanly.

        After calling close(), the interface must not be used.
        """
        ...

    # ── Concrete utility methods ──────────────────────────────────────────────

    def __enter__(self) -> "BaseRobotInterface":
        """Support use as a context manager: `with MuJoCoInterface(cfg) as iface:`"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Automatically close on context manager exit."""
        if not self._is_closed:
            self.close()
            self._is_closed = True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(closed={self._is_closed})"
