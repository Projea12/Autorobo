"""
ros2_interface.py — ROS2 Real Robot Interface (Phase 17)

Implements BaseRobotInterface using ROS2 Jazzy for real robot deployment.
Activate in Phase 17 — requires ROS2 Jazzy install on target machine.

This file can exist in the codebase without ROS2 installed. It will raise
an informative ImportError if instantiated on a system without ROS2.

See ADR-003 for the interface design rationale.

Usage (Phase 17 only, on ROS2-enabled machine):
    # Ensure ROS2 is sourced: source /opt/ros/jazzy/setup.bash
    from interfaces.ros2_interface import ROS2Interface
    interface = ROS2Interface(cfg)
    obs = interface.reset()
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from interfaces.base_interface import BaseRobotInterface

log = logging.getLogger(__name__)

# ── Conditional ROS2 import ───────────────────────────────────────────────────
# This allows the file to be imported without ROS2 installed.
# ROS2Interface will raise ImportError only when instantiated.
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState, LaserScan, Image
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float64MultiArray, Bool
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    log.debug(
        "ROS2 not available (rclpy not found). "
        "ROS2Interface cannot be instantiated. "
        "Activate in Phase 17 — requires ROS2 Jazzy: "
        "source /opt/ros/jazzy/setup.bash"
    )


class ROS2Interface(BaseRobotInterface):
    """
    ROS2 Jazzy implementation of BaseRobotInterface.

    Bridges the abstract robot interface to ROS2 topics/services:

    Subscribes to:
        /camera/rgb/image_raw       (sensor_msgs/Image) → RGB obs
        /camera/depth/image_raw     (sensor_msgs/Image) → depth obs
        /scan                       (sensor_msgs/LaserScan) → LiDAR obs
        /joint_states               (sensor_msgs/JointState) → proprioception

    Publishes to:
        /cmd_vel                    (geometry_msgs/Twist) → base velocity
        /arm_controller/commands    (std_msgs/Float64MultiArray) → arm joints
        /gripper/command            (std_msgs/Bool) → gripper open/close

    Activate in Phase 17 — requires ROS2 Jazzy install.
    On non-ROS2 systems, instantiating this class raises ImportError.
    """

    def __init__(self, cfg: Any) -> None:
        """
        Initialise the ROS2 interface.

        Args:
            cfg: Hydra DictConfig with robot/env settings.

        Raises:
            ImportError: If ROS2 (rclpy) is not installed.

        TODO: Phase 17 — implement:
            rclpy.init()
            self._node = rclpy.create_node("autorobo_interface")
            # Set up subscribers and publishers
            # Wait for all topics to be available
        """
        if not ROS2_AVAILABLE:
            raise ImportError(
                "ROS2 (rclpy) is not installed. "
                "ROS2Interface requires ROS2 Jazzy. "
                "Install via: source /opt/ros/jazzy/setup.bash\n"
                "See docs/adr/ADR-003-clean-interface-for-ros2.md for details."
            )

        super().__init__(cfg)

        # These will be set in Phase 17 implementation
        self._node = None           # rclpy.Node
        self._latest_rgb = None     # np.ndarray
        self._latest_depth = None   # np.ndarray
        self._latest_lidar = None   # np.ndarray
        self._latest_joints = None  # np.ndarray

        # Publishers (set in Phase 17)
        self._cmd_vel_pub = None
        self._arm_cmd_pub = None
        self._gripper_pub = None

        log.info(
            "ROS2Interface created (not yet initialised — TODO: Phase 17). "
            "Requires ROS2 Jazzy: source /opt/ros/jazzy/setup.bash"
        )

    def reset(self) -> dict[str, Any]:
        """
        Reset the real robot to a safe home position.

        For the real robot, "reset" means:
        1. Send arm to home configuration (all joints at zero)
        2. Open gripper
        3. Wait for movement to complete
        4. Return current sensor observation

        Note: Unlike simulation, we cannot randomise the environment.
        Object placement must be done manually for each episode.

        TODO: Phase 17 — implement:
            self._send_arm_to_home()
            self._open_gripper()
            self._wait_for_motion_complete(timeout=10.0)
            return self.get_observation()
        """
        raise NotImplementedError(
            "TODO: Phase 17 — implement ROS2 reset: "
            "home arm, open gripper, wait for completion, return obs."
        )

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Apply action to the real robot and return next observation.

        Note: On the real robot, reward is computed externally (e.g., by checking
        gripper force sensors or verifying object position via perception).
        There is no ground-truth reward signal.

        TODO: Phase 17 — implement:
            self.apply_action(action)
            time.sleep(self.cfg.env.mujoco.control_timestep)  # 20ms
            obs = self.get_observation()
            reward = self._estimate_reward(obs)  # perception-based
            done = self._check_done_real(obs)
            info = {"success": done and reward > 0, "collision": False}
            return obs, reward, done, info
        """
        raise NotImplementedError(
            "TODO: Phase 17 — implement ROS2 step: "
            "apply_action(), sleep(control_timestep), get_observation(), "
            "estimate reward from perception."
        )

    def get_observation(self) -> dict[str, Any]:
        """
        Return the latest sensor readings from ROS2 topics.

        Spins the ROS2 node briefly to consume any pending callbacks,
        then returns the most recent sensor data.

        TODO: Phase 17 — implement:
            rclpy.spin_once(self._node, timeout_sec=0.05)
            return {
                "rgb": self._latest_rgb,
                "depth": self._latest_depth,
                "lidar": self._latest_lidar,
                "proprioception": self._latest_joints,
            }
        """
        raise NotImplementedError(
            "TODO: Phase 17 — implement get_observation: "
            "spin node, return latest sensor readings."
        )

    def apply_action(self, action: Any) -> None:
        """
        Publish action commands to ROS2 topics.

        Args:
            action: np.ndarray — format depends on active sub-system:
                Navigation: [linear_vel, angular_vel]
                Arm control: [j1, j2, j3, j4, j5, j6, gripper]

        TODO: Phase 17 — implement:
            if action.shape == (2,):
                # Navigation action → cmd_vel
                twist = Twist()
                twist.linear.x = float(action[0])
                twist.angular.z = float(action[1])
                self._cmd_vel_pub.publish(twist)
            else:
                # Arm action → joint commands
                cmd = Float64MultiArray(data=action[:6].tolist())
                self._arm_cmd_pub.publish(cmd)
                # Gripper
                gripper_open = bool(action[6] > 0.5)
                self._gripper_pub.publish(Bool(data=gripper_open))
        """
        raise NotImplementedError(
            "TODO: Phase 17 — implement apply_action: "
            "publish Twist for base, Float64MultiArray for arm, Bool for gripper."
        )

    def close(self) -> None:
        """
        Shutdown the ROS2 node and release all resources.

        Sends the arm to the home position before shutting down
        to leave the robot in a safe state.

        TODO: Phase 17 — implement:
            self._send_arm_to_home()
            self._open_gripper()
            self._node.destroy_node()
            rclpy.shutdown()
            self._is_closed = True
        """
        log.info("ROS2Interface.close() called — homing arm before shutdown.")
        self._is_closed = True
        raise NotImplementedError(
            "TODO: Phase 17 — implement close(): "
            "home arm, open gripper, destroy node, rclpy.shutdown()."
        )

    # ── ROS2 callback helpers ─────────────────────────────────────────────────

    def _rgb_callback(self, msg: Any) -> None:
        """
        ROS2 subscriber callback for RGB camera images.

        TODO: Phase 17 — implement:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            self._latest_rgb = bridge.imgmsg_to_cv2(msg, "rgb8")
        """
        raise NotImplementedError(
            "TODO: Phase 17 — implement _rgb_callback using cv_bridge."
        )

    def _depth_callback(self, msg: Any) -> None:
        """
        ROS2 subscriber callback for depth images.

        TODO: Phase 17 — implement:
            self._latest_depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width
            )
        """
        raise NotImplementedError(
            "TODO: Phase 17 — implement _depth_callback."
        )

    def _lidar_callback(self, msg: Any) -> None:
        """
        ROS2 subscriber callback for LiDAR scan.

        TODO: Phase 17 — implement:
            self._latest_lidar = np.array(msg.ranges, dtype=np.float32)
            # Clip infinite readings to max_range
            self._latest_lidar = np.clip(
                self._latest_lidar, 0, self.cfg.robot.sensors.lidar.max_range
            )
        """
        raise NotImplementedError(
            "TODO: Phase 17 — implement _lidar_callback."
        )

    def _joint_state_callback(self, msg: Any) -> None:
        """
        ROS2 subscriber callback for joint states.

        TODO: Phase 17 — implement:
            positions = np.array(msg.position, dtype=np.float32)
            velocities = np.array(msg.velocity, dtype=np.float32)
            self._latest_joints = np.concatenate([positions, velocities])
        """
        raise NotImplementedError(
            "TODO: Phase 17 — implement _joint_state_callback."
        )
