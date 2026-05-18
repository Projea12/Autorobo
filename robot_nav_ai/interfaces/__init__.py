"""
interfaces/ — Swappable Robot Interface Layer

Exports the abstract base class so all callers import from here:

    from interfaces import BaseRobotInterface

Concrete implementations:
    from interfaces.mujoco_interface import MuJoCoInterface   # simulation
    from interfaces.ros2_interface import ROS2Interface        # real robot (Phase 17)
"""

from interfaces.base_interface import BaseRobotInterface

__all__ = ["BaseRobotInterface"]
