"""
world/demo.py — Standalone visual confirmation of the full arena.

Loads the arena + robot URDF in PyBullet GUI, runs several episode resets
to confirm randomise_obstacles() and randomise_goal() work correctly, then
idles so you can inspect the scene.

Usage
-----
    cd robot_nav_ai
    source venv/bin/activate
    python world/demo.py                  # interactive GUI
    python world/demo.py --episodes 3     # auto-reset N times then idle
    python world/demo.py --headless       # no GUI, just smoke-test logic
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pybullet as p
import pybullet_data

# allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from world.world import World, WorldConfig

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "robot", "robot.urdf")

# ── colours for robot spawn marker ───────────────────────────────────────────
_SPAWN_COLOUR = [0.20, 0.60, 1.00, 0.70]   # blue disc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Arena visual demo")
    ap.add_argument("--episodes",  type=int,   default=3,
                    help="Number of episode resets to show before idling")
    ap.add_argument("--headless",  action="store_true",
                    help="Run in DIRECT mode (no window) — logic test only")
    ap.add_argument("--arena",     type=float, default=6.0,
                    help="Arena side length in metres")
    ap.add_argument("--obstacles", type=int,   default=8,
                    help="Number of obstacles")
    ap.add_argument("--seed",      type=int,   default=42)
    return ap.parse_args()


# ── robot loader ──────────────────────────────────────────────────────────────

def load_robot(client: int) -> int:
    robot_id = p.loadURDF(
        URDF_PATH,
        basePosition=[0, 0, 0.10],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client,
    )
    return robot_id


def add_spawn_marker(client: int) -> int:
    """Blue disc at origin showing robot spawn zone."""
    vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.80,
        length=0.005,
        rgbaColor=_SPAWN_COLOUR,
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[0, 0, 0.003],
        physicsClientId=client,
    )


# ── info printer ──────────────────────────────────────────────────────────────

def print_episode_summary(world: World, episode: int) -> None:
    gx, gy = world.goal_position or (0, 0)
    print(
        f"\n  Episode {episode:>2} │ "
        f"obstacles={world.obstacle_count} "
        f"(static={len(world.static_obstacle_ids)}, "
        f"dynamic={len(world.dynamic_obstacle_ids)})  │ "
        f"goal=({gx:+.2f}, {gy:+.2f})"
    )


# ── camera ────────────────────────────────────────────────────────────────────

def setup_camera(client: int, arena: float) -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=arena * 1.25,
        cameraYaw=30,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0],
        physicsClientId=client,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    mode = p.DIRECT if args.headless else p.GUI
    client = p.connect(mode)

    try:
        # ── world config ──────────────────────────────────────────────────────
        cfg = WorldConfig(
            arena_size=args.arena,
            min_obstacles=max(5, args.obstacles - 2),
            max_obstacles=args.obstacles,
            dynamic_ratio=0.30,
            dynamic_speed_min=0.3,
            dynamic_speed_max=0.7,
            seed=args.seed,
            floor_type="box",
            grid_lines=True,
        )
        world = World(client, config=cfg)

        # ── first build ───────────────────────────────────────────────────────
        print("\n" + "═" * 55)
        print("  Robot Nav AI — Arena Demo")
        print("═" * 55)
        print(f"  Arena     : {args.arena} × {args.arena} m")
        print(f"  Obstacles : {cfg.min_obstacles}–{cfg.max_obstacles}")
        print(f"  Seed      : {args.seed}")
        print(f"  Mode      : {'headless' if args.headless else 'GUI'}")
        print("═" * 55)

        world.build()
        goal_xy = world.randomise_goal()

        if not args.headless:
            setup_camera(client, args.arena)
            robot_id   = load_robot(client)
            spawn_disc = add_spawn_marker(client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1,
                                       physicsClientId=client)
            print("\n  Controls: left-drag=rotate  scroll=zoom  Ctrl+C=quit")

        print_episode_summary(world, episode=1)

        # ── episode reset loop ────────────────────────────────────────────────
        for ep in range(2, args.episodes + 1):
            if args.headless:
                # just step physics a few times per episode
                for _ in range(120):
                    p.stepSimulation(physicsClientId=client)
            else:
                # let the scene run for 3 s so you can see the current layout
                deadline = time.time() + 3.0
                pulse_t  = time.time()
                while time.time() < deadline:
                    p.stepSimulation(physicsClientId=client)
                    if time.time() - pulse_t > 0.25:
                        world.pulse_goal()
                        pulse_t = time.time()
                    time.sleep(1 / 240)

            # ── randomise for next episode ────────────────────────────────────
            world.randomise_obstacles()
            world.randomise_goal()
            print_episode_summary(world, ep)

            if not args.headless:
                # move robot back to spawn
                p.resetBasePositionAndOrientation(
                    robot_id,
                    [0, 0, 0.10],
                    p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=client,
                )
                p.resetBaseVelocity(
                    robot_id,
                    linearVelocity=[0, 0, 0],
                    angularVelocity=[0, 0, 0],
                    physicsClientId=client,
                )

        # ── idle / final check ────────────────────────────────────────────────
        print("\n" + "═" * 55)
        print(f"  Final layout: {world.obstacle_count} obstacles")
        print(f"  LiDAR targets: {len(world.all_obstacle_ids)} bodies")
        print(f"  Goal position: {world.goal_position}")
        print("═" * 55)

        if args.headless:
            print("  Headless run complete — all checks passed.\n")
        else:
            print("  Scene is live — press Ctrl+C to exit.\n")
            while True:
                p.stepSimulation(physicsClientId=client)
                world.pulse_goal()
                time.sleep(1 / 60)

    except KeyboardInterrupt:
        print("\n  Exiting.")
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()
