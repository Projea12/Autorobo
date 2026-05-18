"""
world/world.py — Arena builder for the robot navigation simulation.

Constructs:
  • Textured ground plane
  • Four boundary walls enclosing the arena
  • Random or fixed box/cylinder obstacles
  • Goal marker (visual only, no collision)

Usage:
    import pybullet as p
    from world import World

    client = p.connect(p.GUI)
    world  = World(client, arena_size=6.0, num_obstacles=10)
    world.build()
    goal   = world.sample_goal()
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pybullet as p
import pybullet_data


# ── Colour helpers ────────────────────────────────────────────────────────────
_FLOOR_COLOUR    = [0.55, 0.55, 0.55, 1.0]
_WALL_COLOUR     = [0.30, 0.30, 0.35, 1.0]
_BOX_COLOUR      = [0.70, 0.40, 0.10, 1.0]
_CYLINDER_COLOUR = [0.20, 0.55, 0.20, 1.0]
_GOAL_COLOUR     = [0.00, 0.85, 0.20, 0.6]


@dataclass
class ObstacleConfig:
    """Specification for a single obstacle."""
    shape:    str            # "box" | "cylinder"
    position: Tuple[float, float, float]
    size:     Tuple          # box: (lx, ly, lz)  cylinder: (radius, height)
    yaw:      float = 0.0   # rotation about Z in radians


@dataclass
class WorldConfig:
    """Top-level arena configuration."""
    arena_size:    float = 6.0    # half-extent → arena spans ±arena_size/2
    wall_height:   float = 0.5
    wall_thickness: float = 0.05
    num_obstacles:  int  = 10
    seed:           Optional[int] = None
    # fixed obstacles — if empty, obstacles are randomly generated
    fixed_obstacles: List[ObstacleConfig] = field(default_factory=list)


class World:
    """
    Builds and manages the PyBullet simulation arena.

    Parameters
    ----------
    client      : PyBullet physics client id (from p.connect)
    arena_size  : full side length of the square arena in metres
    num_obstacles: number of random obstacles (ignored when fixed_obstacles given)
    seed        : RNG seed for reproducible layouts
    config      : full WorldConfig override (overrides individual kwargs)
    """

    def __init__(
        self,
        client:        int,
        arena_size:    float = 6.0,
        num_obstacles: int   = 10,
        seed:          Optional[int] = None,
        config:        Optional[WorldConfig] = None,
    ) -> None:
        self._client = client
        self._cfg    = config or WorldConfig(
            arena_size=arena_size,
            num_obstacles=num_obstacles,
            seed=seed,
        )
        self._rng = random.Random(self._cfg.seed)

        # body-id registries
        self._floor_id:     Optional[int] = None
        self._wall_ids:     List[int]     = []
        self._obstacle_ids: List[int]     = []
        self._goal_id:      Optional[int] = None

    # ── public API ────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Construct the full arena: floor + walls + obstacles."""
        self._setup_physics()
        self._build_floor()
        self._build_walls()
        self._build_obstacles()

    def rebuild(self) -> None:
        """Remove all arena bodies and rebuild (useful between episodes)."""
        self.reset()
        self.build()

    def reset(self) -> None:
        """Remove every arena body from the simulation."""
        for body_id in (
            [self._floor_id] + self._wall_ids + self._obstacle_ids + [self._goal_id]
        ):
            if body_id is not None:
                try:
                    p.removeBody(body_id, physicsClientId=self._client)
                except Exception:
                    pass
        self._floor_id     = None
        self._wall_ids     = []
        self._obstacle_ids = []
        self._goal_id      = None

    def sample_goal(
        self,
        min_dist_from_centre: float = 0.5,
        margin: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Return a random (x, y) goal position inside the arena.

        Parameters
        ----------
        min_dist_from_centre : avoid spawning right at origin
        margin               : keep this far from each wall
        """
        half = self._cfg.arena_size / 2.0 - margin
        for _ in range(200):
            x = self._rng.uniform(-half, half)
            y = self._rng.uniform(-half, half)
            if math.hypot(x, y) >= min_dist_from_centre:
                self._place_goal_marker(x, y)
                return (x, y)
        return (half * 0.5, half * 0.5)   # deterministic fallback

    def place_goal_marker(self, x: float, y: float, z: float = 0.01) -> None:
        """Draw a flat green disc at (x, y) to show the goal visually."""
        self._place_goal_marker(x, y, z)

    # ── body-id accessors ─────────────────────────────────────────────────────

    @property
    def obstacle_ids(self) -> List[int]:
        return list(self._obstacle_ids)

    @property
    def wall_ids(self) -> List[int]:
        return list(self._wall_ids)

    @property
    def all_obstacle_ids(self) -> List[int]:
        """Walls + obstacles — everything a LiDAR ray can hit."""
        return self._wall_ids + self._obstacle_ids

    # ── internal builders ─────────────────────────────────────────────────────

    def _setup_physics(self) -> None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)

    def _build_floor(self) -> None:
        """Load PyBullet's built-in plane and tint it."""
        self._floor_id = p.loadURDF(
            "plane.urdf",
            basePosition=[0, 0, 0],
            physicsClientId=self._client,
        )
        p.changeVisualShape(
            self._floor_id, -1,
            rgbaColor=_FLOOR_COLOUR,
            physicsClientId=self._client,
        )

    def _build_walls(self) -> None:
        """
        Four axis-aligned walls that form a closed square boundary.

        Wall layout (top-down view, arena_size = S):
                        North  (y = +S/2)
              ┌──────────────────────────┐
         West │                          │ East
         (-S/2│                          │+S/2)
              └──────────────────────────┘
                        South  (y = -S/2)
        """
        S  = self._cfg.arena_size
        H  = self._cfg.wall_height
        T  = self._cfg.wall_thickness

        # (cx, cy, cz,  half_lx, half_ly, half_lz)
        wall_specs = [
            # North
            ( 0,  S / 2, H / 2,   S / 2 + T,  T / 2,  H / 2),
            # South
            ( 0, -S / 2, H / 2,   S / 2 + T,  T / 2,  H / 2),
            # East
            ( S / 2, 0,  H / 2,   T / 2,  S / 2,  H / 2),
            # West
            (-S / 2, 0,  H / 2,   T / 2,  S / 2,  H / 2),
        ]

        for cx, cy, cz, hx, hy, hz in wall_specs:
            body_id = self._make_box(
                half_extents=(hx, hy, hz),
                position=(cx, cy, cz),
                colour=_WALL_COLOUR,
                mass=0,          # static
            )
            self._wall_ids.append(body_id)

    def _build_obstacles(self) -> None:
        if self._cfg.fixed_obstacles:
            for obs in self._cfg.fixed_obstacles:
                self._place_obstacle(obs)
        else:
            self._build_random_obstacles()

    def _build_random_obstacles(self) -> None:
        """
        Scatter a mix of boxes and cylinders inside the arena.
        Keeps a clear zone of radius 0.8 m around the origin so the
        robot always has room to spawn.
        """
        half    = self._cfg.arena_size / 2.0
        margin  = 0.4          # stay this far from walls
        clear_r = 0.8          # clear zone around origin

        shapes = ["box", "cylinder"]
        placed = 0
        attempts = 0

        while placed < self._cfg.num_obstacles and attempts < 500:
            attempts += 1
            x = self._rng.uniform(-half + margin, half - margin)
            y = self._rng.uniform(-half + margin, half - margin)

            if math.hypot(x, y) < clear_r:
                continue

            shape = self._rng.choice(shapes)

            if shape == "box":
                lx = self._rng.uniform(0.15, 0.50)
                ly = self._rng.uniform(0.15, 0.50)
                lz = self._rng.uniform(0.20, 0.60)
                yaw = self._rng.uniform(0, math.pi)
                obs = ObstacleConfig("box", (x, y, lz / 2), (lx, ly, lz), yaw)
            else:
                r  = self._rng.uniform(0.08, 0.20)
                lz = self._rng.uniform(0.20, 0.60)
                obs = ObstacleConfig("cylinder", (x, y, lz / 2), (r, lz), 0.0)

            self._place_obstacle(obs)
            placed += 1

    def _place_obstacle(self, obs: ObstacleConfig) -> None:
        orn = p.getQuaternionFromEuler([0, 0, obs.yaw])

        if obs.shape == "box":
            lx, ly, lz = obs.size
            body_id = self._make_box(
                half_extents=(lx / 2, ly / 2, lz / 2),
                position=obs.position,
                orientation=orn,
                colour=_BOX_COLOUR,
                mass=0,
            )
        elif obs.shape == "cylinder":
            radius, height = obs.size
            body_id = self._make_cylinder(
                radius=radius,
                height=height,
                position=obs.position,
                colour=_CYLINDER_COLOUR,
                mass=0,
            )
        else:
            raise ValueError(f"Unknown obstacle shape: {obs.shape!r}")

        self._obstacle_ids.append(body_id)

    def _place_goal_marker(self, x: float, y: float, z: float = 0.01) -> None:
        if self._goal_id is not None:
            try:
                p.removeBody(self._goal_id, physicsClientId=self._client)
            except Exception:
                pass

        # flat disc — visual only (mass=0, no collision shape needed)
        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.15,
            length=0.02,
            rgbaColor=_GOAL_COLOUR,
            physicsClientId=self._client,
        )
        self._goal_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=[x, y, z],
            physicsClientId=self._client,
        )

    # ── primitive factories ───────────────────────────────────────────────────

    def _make_box(
        self,
        half_extents: Tuple[float, float, float],
        position:     Tuple[float, float, float],
        colour:       List[float],
        mass:         float = 0,
        orientation:  Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> int:
        col_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=list(half_extents),
            physicsClientId=self._client,
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=list(half_extents),
            rgbaColor=colour,
            physicsClientId=self._client,
        )
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=list(position),
            baseOrientation=list(orientation),
            physicsClientId=self._client,
        )

    def _make_cylinder(
        self,
        radius:    float,
        height:    float,
        position:  Tuple[float, float, float],
        colour:    List[float],
        mass:      float = 0,
    ) -> int:
        col_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self._client,
        )
        vis_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=colour,
            physicsClientId=self._client,
        )
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=list(position),
            physicsClientId=self._client,
        )
