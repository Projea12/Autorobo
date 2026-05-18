"""
Launch PyBullet GUI, load robot.urdf, and let you inspect the model.

Usage:
    python visualise_robot.py            # interactive — press Ctrl+C to quit
    python visualise_robot.py --headless # off-screen, runs 240 steps then exits
"""

import argparse
import os
import time

import pybullet as p
import pybullet_data

URDF_PATH = os.path.join(os.path.dirname(__file__), "robot", "robot.urdf")

# ── joint-type labels for the info printout ──────────────────────────────────
JOINT_TYPE = {
    p.JOINT_REVOLUTE:   "revolute",
    p.JOINT_PRISMATIC:  "prismatic",
    p.JOINT_SPHERICAL:  "spherical",
    p.JOINT_PLANAR:     "planar",
    p.JOINT_FIXED:      "fixed",
}


def load_robot(client: int) -> int:
    """Load ground plane + robot URDF, return robot body id."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=client)

    # flat ground plane
    p.loadURDF("plane.urdf", physicsClientId=client)

    # spawn robot 0.05 m above ground so wheels rest on the plane
    robot_id = p.loadURDF(
        URDF_PATH,
        basePosition=[0, 0, 0.05],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client,
    )
    return robot_id


def print_robot_info(robot_id: int, client: int) -> None:
    """Print every link and joint loaded from the URDF."""
    n_joints = p.getNumJoints(robot_id, physicsClientId=client)
    base_info = p.getBodyInfo(robot_id, physicsClientId=client)

    print("\n" + "═" * 60)
    print(f"  Robot loaded  —  body id : {robot_id}")
    print(f"  Base link     : {base_info[0].decode()}")
    print(f"  Total joints  : {n_joints}")
    print("═" * 60)
    print(f"  {'#':<4} {'Joint name':<26} {'Type':<12} {'Child link'}")
    print("  " + "─" * 56)

    for i in range(n_joints):
        info = p.getJointInfo(robot_id, i, physicsClientId=client)
        j_idx  = info[0]
        j_name = info[1].decode()
        j_type = JOINT_TYPE.get(info[2], "unknown")
        l_name = info[12].decode()
        print(f"  {j_idx:<4} {j_name:<26} {j_type:<12} {l_name}")

    print("═" * 60 + "\n")


def configure_camera(client: int) -> None:
    """Position the debug camera for a good initial view."""
    p.resetDebugVisualizerCamera(
        cameraDistance=0.9,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.1],
        physicsClientId=client,
    )


def draw_sensor_frame(robot_id: int, client: int) -> None:
    """Draw XYZ axes on the lidar_link frame so it's easy to spot."""
    n_joints = p.getNumJoints(robot_id, physicsClientId=client)
    lidar_joint_idx = None
    for i in range(n_joints):
        name = p.getJointInfo(robot_id, i, physicsClientId=client)[1].decode()
        if name == "lidar_joint":
            lidar_joint_idx = i
            break

    if lidar_joint_idx is None:
        return

    state = p.getLinkState(robot_id, lidar_joint_idx, physicsClientId=client)
    origin = state[4]   # world position of link frame
    length = 0.08

    # X — red, Y — green, Z — blue
    for axis, colour in zip([[length,0,0],[0,length,0],[0,0,length]],
                             [[1,0,0],[0,1,0],[0,0,1]]):
        end = [origin[i] + axis[i] for i in range(3)]
        p.addUserDebugLine(origin, end, colour, lineWidth=2,
                           physicsClientId=client)

    p.addUserDebugText("LiDAR", [origin[0], origin[1], origin[2]+0.06],
                       textColorRGB=[0, 0.6, 1], textSize=1.2,
                       physicsClientId=client)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="Run DIRECT (no GUI), 240 steps then exit")
    args = parser.parse_args()

    mode   = p.DIRECT if args.headless else p.GUI
    client = p.connect(mode)

    try:
        robot_id = load_robot(client)
        print_robot_info(robot_id, client)

        if not args.headless:
            configure_camera(client)
            draw_sensor_frame(robot_id, client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1,
                                       physicsClientId=client)
            print("PyBullet GUI open — rotate with left-drag, zoom with scroll.")
            print("Press Ctrl+C in terminal to exit.\n")

            while True:
                p.stepSimulation(physicsClientId=client)
                time.sleep(1 / 240)
        else:
            for _ in range(240):
                p.stepSimulation(physicsClientId=client)
            print("Headless run complete — robot loaded successfully.")

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()
