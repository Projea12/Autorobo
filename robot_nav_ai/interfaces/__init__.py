"""
interfaces/ — Swappable Robot Interface Layer

Exports the abstract base class so all callers import from here:

    from interfaces import BaseRobotInterface

Concrete implementations — swap with a one-line change:
    from interfaces.video_interface  import VideoInterface    # video / webcam (demo/AR)
    from interfaces.mujoco_interface import MuJoCoInterface   # physics simulation
    from interfaces.ros2_interface   import ROS2Interface     # real robot hardware
"""

from interfaces.base_interface import BaseRobotInterface

__all__ = ["BaseRobotInterface"]
