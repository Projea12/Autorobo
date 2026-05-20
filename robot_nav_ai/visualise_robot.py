"""
Autorobo — robot viewer using MuJoCo's built-in 3D renderer.

Loads robot/robot.xml, sets the home keyframe so the robot stays
in place, then opens the interactive MuJoCo viewer window.

Controls inside the viewer:
  Left-drag       rotate view
  Right-drag      pan
  Scroll          zoom
  Ctrl+A          show / hide axes
  Ctrl+G          show / hide geom groups
  Space           pause / resume physics
  Backspace       reset to initial pose

Usage:
    python visualise_robot.py            # home pose (arm up)
    python visualise_robot.py --pose ready    # arm reaching forward
    python visualise_robot.py --pose grasp_open
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _check_mujoco():
    try:
        import mujoco
        import mujoco.viewer
        return mujoco
    except ModuleNotFoundError:
        print("MuJoCo not installed.  Run:  pip install mujoco")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", default="home",
                        choices=["home", "ready", "grasp_open"],
                        help="Starting keyframe pose")
    args = parser.parse_args()

    mujoco = _check_mujoco()

    xml_path = ROOT / "robot" / "robot.xml"
    if not xml_path.exists():
        print(f"robot.xml not found at {xml_path}")
        sys.exit(1)

    print(f"\nLoading robot from:  {xml_path}")
    print(f"Starting pose     :  {args.pose}")
    print("\nViewer controls:")
    print("  Left-drag  = rotate   |  Scroll = zoom")
    print("  Space      = pause    |  Backspace = reset")
    print("  Ctrl+A     = axes     |  Esc = exit\n")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)

    # Apply the keyframe so the robot starts in a stable, visible pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.pose)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        print(f"Warning: keyframe '{args.pose}' not found — using default.")

    # Forward kinematics so all body positions are correct before viewer opens
    mujoco.mj_forward(model, data)

    # Launch the interactive viewer (blocks until window is closed)
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Point camera at the robot from a good angle
        viewer.cam.lookat[:] = [0.1, 0.0, 0.35]
        viewer.cam.distance  = 2.5
        viewer.cam.azimuth   = -60
        viewer.cam.elevation = -18

        print("MuJoCo viewer open.  Close the window or press Ctrl+C to exit.")

        while viewer.is_running():
            # Do NOT step physics — robot stays locked in keyframe pose.
            # Drag to rotate, scroll to zoom, Backspace to reset.
            viewer.sync()
            time.sleep(0.016)   # ~60 fps refresh


if __name__ == "__main__":
    main()
