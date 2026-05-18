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
_FLOOR_ALT_COLOUR = [0.45, 0.45, 0.45, 1.0]   # checker dark tile
_GRID_COLOUR     = [0.25, 0.25, 0.25, 1.0]
_WALL_COLOUR     = [0.30, 0.30, 0.35, 1.0]
_WALL_STRIPE     = [0.80, 0.80, 0.10, 1.0]   # yellow warning stripe near top
_BOX_COLOUR      = [0.70, 0.40, 0.10, 1.0]
_CYLINDER_COLOUR = [0.20, 0.55, 0.20, 1.0]
_GOAL_DISC       = [0.00, 0.90, 0.25, 0.75]   # translucent green flat disc
_GOAL_SPHERE     = [0.00, 1.00, 0.30, 0.85]   # bright green floating sphere
_GOAL_BEACON     = [0.00, 1.00, 0.50, 0.30]   # tall translucent beacon pillar
_GOAL_LINE       = [0.00, 0.95, 0.30]          # RGB for debug lines (no alpha)
_GOAL_PULSE_A    = [0.00, 1.00, 0.30, 0.85]   # bright state for pulse
_GOAL_PULSE_B    = [0.00, 0.55, 0.15, 0.40]   # dim  state for pulse


@dataclass
class ObstacleConfig:
    """
    Specification for a single obstacle.

    shape    : "box" | "cylinder"
    position : (x, y, z) — z is the centre height, not the base
    size     : box → (lx, ly, lz) full lengths
               cylinder → (radius, height)
    yaw      : rotation about Z in radians (boxes only)
    mass     : kg — 0 = static/immovable,  >0 = dynamic (physics-driven)
    dynamic  : convenience flag; sets mass to 1.0 if True and mass==0
    colour   : override RGBA; None uses shape-default colour
    velocity : initial (vx, vy) for dynamic obstacles (m/s)
    """
    shape:    str
    position: Tuple[float, float, float]
    size:     Tuple
    yaw:      float                          = 0.0
    mass:     float                          = 0.0
    dynamic:  bool                           = False
    colour:   Optional[List[float]]          = None
    velocity: Tuple[float, float]            = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.dynamic and self.mass == 0.0:
            self.mass = 1.0


@dataclass
class WorldConfig:
    """Top-level arena configuration."""
    arena_size:      float = 6.0
    wall_height:     float = 0.5
    wall_thickness:  float = 0.05
    # obstacle count — random count chosen in [min_obstacles, max_obstacles]
    min_obstacles:   int   = 5
    max_obstacles:   int   = 10
    # fraction of random obstacles that are dynamic (0.0–1.0)
    dynamic_ratio:   float = 0.3
    # dynamic obstacle speed range (m/s)
    dynamic_speed_min: float = 0.3
    dynamic_speed_max: float = 0.8
    seed:            Optional[int] = None
    # "plane" → infinite plane.urdf  |  "box" → bounded explicit box
    floor_type:      str   = "plane"
    floor_thickness: float = 0.02
    grid_lines:      bool  = True
    # supply fixed_obstacles to skip random generation entirely
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
        num_obstacles: int   = 8,    # shortcut — sets min=max=num_obstacles
        seed:          Optional[int] = None,
        config:        Optional[WorldConfig] = None,
    ) -> None:
        self._client = client
        if config is not None:
            self._cfg = config
        else:
            self._cfg = WorldConfig(
                arena_size=arena_size,
                min_obstacles=num_obstacles,
                max_obstacles=num_obstacles,
                seed=seed,
            )
        self._rng = random.Random(self._cfg.seed)

        # body-id registries
        self._floor_id:       Optional[int]  = None
        self._wall_ids:       List[int]       = []
        self._named_wall_ids: dict            = {}
        self._static_ids:     List[int]       = []   # mass = 0
        self._dynamic_ids:    List[int]       = []   # mass > 0, physics-driven
        # goal marker — multiple body + debug-line ids
        self._goal_id:         Optional[int]   = None   # disc body
        self._goal_sphere_id:  Optional[int]   = None   # floating sphere
        self._goal_beacon_id:  Optional[int]   = None   # tall pillar
        self._goal_line_ids:   List[int]        = []     # debug lines
        self._goal_pos:        Optional[Tuple[float, float]] = None
        self._pulse_bright:    bool             = True   # toggle for pulse()

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
        all_ids = (
            [self._floor_id]
            + self._wall_ids
            + self._static_ids
            + self._dynamic_ids
            + [self._goal_id, self._goal_sphere_id, self._goal_beacon_id]
        )
        for body_id in all_ids:
            if body_id is not None:
                try:
                    p.removeBody(body_id, physicsClientId=self._client)
                except Exception:
                    pass
        for line_id in self._goal_line_ids:
            try:
                p.removeUserDebugItem(line_id, physicsClientId=self._client)
            except Exception:
                pass
        self._floor_id        = None
        self._wall_ids        = []
        self._named_wall_ids  = {}
        self._static_ids      = []
        self._dynamic_ids     = []
        self._goal_id         = None
        self._goal_sphere_id  = None
        self._goal_beacon_id  = None
        self._goal_line_ids   = []
        self._goal_pos        = None

    def sample_goal(
        self,
        min_dist_from_centre: float = 0.8,
        margin: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Pick a random (x, y) goal clear of walls and the robot spawn zone,
        draw the full goal marker, and return the position.
        """
        half = self._cfg.arena_size / 2.0 - margin
        for _ in range(300):
            x = self._rng.uniform(-half, half)
            y = self._rng.uniform(-half, half)
            if math.hypot(x, y) >= min_dist_from_centre:
                self._place_goal_marker(x, y)
                return (x, y)
        fallback = (half * 0.6, half * 0.6)
        self._place_goal_marker(*fallback)
        return fallback

    def place_goal_marker(self, x: float, y: float) -> None:
        """Explicitly place the goal marker at (x, y)."""
        self._place_goal_marker(x, y)

    @property
    def goal_position(self) -> Optional[Tuple[float, float]]:
        """Last placed goal (x, y), or None if not yet placed."""
        return self._goal_pos

    def pulse_goal(self) -> None:
        """
        Alternate the goal sphere between bright and dim colours.
        Call once per sim step (or every N steps) to create a pulsing effect.

        Example in training loop:
            if step % 10 == 0:
                world.pulse_goal()
        """
        if self._goal_sphere_id is None:
            return
        colour = _GOAL_PULSE_A if self._pulse_bright else _GOAL_PULSE_B
        p.changeVisualShape(
            self._goal_sphere_id, -1,
            rgbaColor=colour,
            physicsClientId=self._client,
        )
        self._pulse_bright = not self._pulse_bright

    # ── body-id accessors ─────────────────────────────────────────────────────

    @property
    def obstacle_ids(self) -> List[int]:
        """All obstacle body ids (static + dynamic)."""
        return self._static_ids + self._dynamic_ids

    @property
    def static_obstacle_ids(self) -> List[int]:
        return list(self._static_ids)

    @property
    def dynamic_obstacle_ids(self) -> List[int]:
        return list(self._dynamic_ids)

    @property
    def obstacle_count(self) -> int:
        return len(self._static_ids) + len(self._dynamic_ids)

    @property
    def wall_ids(self) -> List[int]:
        return list(self._wall_ids)

    def wall_id(self, face: str) -> int:
        """Return body id for a named wall: 'north','south','east','west'."""
        try:
            return self._named_wall_ids[face]
        except KeyError:
            raise KeyError(
                f"No wall named {face!r}. "
                f"Choose from {list(self._named_wall_ids)}"
            )

    @property
    def all_obstacle_ids(self) -> List[int]:
        """Walls + all obstacles — everything a LiDAR ray can hit."""
        return self._wall_ids + self._static_ids + self._dynamic_ids

    def step_dynamic_obstacles(self) -> None:
        """
        Called once per simulation step to keep dynamic obstacles moving.

        PyBullet's physics engine handles wall/obstacle collisions automatically
        because each dynamic body has mass > 0 and restitution = 0.8.
        This method does nothing extra — it exists as a hook so the env loop
        can call `world.step_dynamic_obstacles()` each step without needing
        to know whether dynamics are enabled.
        """
        pass   # physics engine drives movement; hook kept for custom overrides

    # ── internal builders ─────────────────────────────────────────────────────

    def _setup_physics(self) -> None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)

    def _build_floor(self) -> None:
        """
        Build the ground surface.

        floor_type = "plane"
            Loads PyBullet's built-in plane.urdf — infinite, zero-thickness,
            no visual edges.  Fastest option; good for open-world training.

        floor_type = "box"
            Creates an explicit flat box sized exactly to arena_size × arena_size.
            Shows a visible boundary edge and optional 1 m debug grid so the
            agent can perceive its position relative to the arena.
        """
        if self._cfg.floor_type == "plane":
            self._build_floor_plane()
        elif self._cfg.floor_type == "box":
            self._build_floor_box()
        else:
            raise ValueError(
                f"Unknown floor_type {self._cfg.floor_type!r}. "
                "Choose 'plane' or 'box'."
            )

    def _build_floor_plane(self) -> None:
        """Infinite ground plane from PyBullet's built-in plane.urdf."""
        self._floor_id = p.loadURDF(
            "plane.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client,
        )
        p.changeVisualShape(
            self._floor_id, -1,
            rgbaColor=_FLOOR_COLOUR,
            physicsClientId=self._client,
        )

    def _build_floor_box(self) -> None:
        """
        Explicit flat box floor bounded to the arena footprint.

        Geometry
        --------
        S  = arena_size          (e.g. 6.0 m)
        T  = floor_thickness     (e.g. 0.02 m)
        Top surface sits at z = 0 (world origin).
        Box centre is therefore at z = −T/2.

        The collision shape is a single box so PyBullet's broadphase
        handles it in O(1) regardless of arena size.
        """
        S = self._cfg.arena_size
        T = self._cfg.floor_thickness

        col_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[S / 2, S / 2, T / 2],
            physicsClientId=self._client,
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[S / 2, S / 2, T / 2],
            rgbaColor=_FLOOR_COLOUR,
            physicsClientId=self._client,
        )
        self._floor_id = p.createMultiBody(
            baseMass=0,                         # static — never moves
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[0, 0, -T / 2],        # top face flush with z = 0
            physicsClientId=self._client,
        )

        # ── border edge lines ────────────────────────────────────────────────
        # Draw the four top edges so the boundary is clearly visible in GUI.
        h = S / 2
        corners = [(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)]
        for i in range(4):
            p.addUserDebugLine(
                corners[i], corners[(i + 1) % 4],
                lineColorRGB=[0.1, 0.1, 0.1],
                lineWidth=2,
                physicsClientId=self._client,
            )

        # ── 1 m grid lines ───────────────────────────────────────────────────
        if self._cfg.grid_lines:
            self._draw_grid(S)

    def _draw_grid(self, arena_size: float, cell: float = 1.0) -> None:
        """
        Draw a debug grid of cell × cell metre squares on the floor.

        Lines run from -arena_size/2 to +arena_size/2 in both axes,
        spaced `cell` metres apart.  Skips the centre lines (already clear).
        """
        half  = arena_size / 2.0
        steps = int(arena_size / cell) + 1
        z     = 0.002   # just above floor surface to avoid z-fighting

        for i in range(steps):
            offset = -half + i * cell
            if abs(offset) > half:
                continue
            # parallel to X axis
            p.addUserDebugLine(
                [-half, offset, z], [half, offset, z],
                lineColorRGB=_GRID_COLOUR[:3],
                lineWidth=1,
                physicsClientId=self._client,
            )
            # parallel to Y axis
            p.addUserDebugLine(
                [offset, -half, z], [offset, half, z],
                lineColorRGB=_GRID_COLOUR[:3],
                lineWidth=1,
                physicsClientId=self._client,
            )

    def _build_walls(self) -> None:
        """
        Four solid boundary walls built with explicit PyBullet primitives.

        Top-down layout (arena_size = S, wall_thickness = T):
        ┌─────────────────────────────────────────┐  y = +S/2  NORTH
        │◄──── S + 2T ────────────────────────────►│
        │                                         │
        │                 arena                   │  x = ±S/2
        │                                         │
        └─────────────────────────────────────────┘  y = −S/2  SOUTH

        Corner strategy — N/S walls span the full width INCLUDING the corner
        blocks (half_x = S/2 + T) so E/W walls slot flush inside them:

            ┌──┬──────────────────────────┬──┐  ← North wall (full width)
            │  │                          │  │
            │W │         arena            │E │  ← East/West (inner length only)
            │  │                          │  │
            └──┴──────────────────────────┴──┘  ← South wall (full width)

        Each wall is created with:
          1. p.createCollisionShape  — solid box the physics engine resolves
          2. p.createVisualShape     — rendered geometry (same box, tinted)
          3. p.createMultiBody       — combines both into a zero-mass static body
          4. p.createCollisionShape  — thin stripe for the yellow warning band
          5. A debug text label      — face name visible in GUI
        """
        S = self._cfg.arena_size
        H = self._cfg.wall_height
        T = self._cfg.wall_thickness

        # ── wall geometry table ───────────────────────────────────────────────
        # name, centre (cx,cy,cz), half-extents (hx,hy,hz)
        #
        # N/S walls: hx = S/2+T  so they fill the corner blocks
        # E/W walls: hy = S/2    so they fit exactly between the corners
        wall_table = {
            "north": ((  0,       S/2, H/2), (S/2+T, T/2,  H/2)),
            "south": ((  0,      -S/2, H/2), (S/2+T, T/2,  H/2)),
            "east":  (( S/2,       0,  H/2), (T/2,   S/2,  H/2)),
            "west":  ((-S/2,       0,  H/2), (T/2,   S/2,  H/2)),
        }

        for face, (centre, half_ext) in wall_table.items():
            cx, cy, cz = centre
            hx, hy, hz = half_ext

            # ── 1. collision shape ────────────────────────────────────────────
            col_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy, hz],
                physicsClientId=self._client,
            )

            # ── 2. visual shape (main body) ───────────────────────────────────
            vis_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy, hz],
                rgbaColor=_WALL_COLOUR,
                physicsClientId=self._client,
            )

            # ── 3. static rigid body ──────────────────────────────────────────
            #   baseMass = 0  →  immovable; robot cannot push it
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[cx, cy, cz],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self._client,
            )

            # ── 4. warning stripe (visual-only, no collision) ─────────────────
            # A thin yellow band near the top of each wall — helps the agent
            # and the human viewer quickly spot the boundary.
            stripe_h  = min(0.06, hz * 0.25)           # 6 cm or 25 % of height
            stripe_z  = cz + hz - stripe_h             # flush with wall top
            stripe_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy + 0.001, stripe_h],  # 1 mm proud of wall
                rgbaColor=_WALL_STRIPE,
                physicsClientId=self._client,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,             # no collision — visual only
                baseVisualShapeIndex=stripe_vis,
                basePosition=[cx, cy, stripe_z],
                physicsClientId=self._client,
            )

            # ── 5. GUI label ──────────────────────────────────────────────────
            label_pos = [cx, cy, cz + hz + 0.08]
            p.addUserDebugText(
                face.upper(),
                label_pos,
                textColorRGB=[1, 1, 1],
                textSize=0.9,
                physicsClientId=self._client,
            )

            # register
            self._wall_ids.append(body_id)
            self._named_wall_ids[face] = body_id

    def _build_obstacles(self) -> None:
        if self._cfg.fixed_obstacles:
            for obs in self._cfg.fixed_obstacles:
                self._place_obstacle(obs)
        else:
            self._build_random_obstacles()

    def _build_random_obstacles(self) -> None:
        """
        Place 5–10 obstacles of mixed shape, size, and dynamics.

        Algorithm
        ---------
        1. Pick a random count in [min_obstacles, max_obstacles].
        2. For each obstacle, sample a candidate (x, y) and reject if:
             a. Inside the robot spawn clear-zone  (r < 0.8 m from origin)
             b. Closer than `sep` metres to any already-placed obstacle
             c. Within `margin` metres of a wall
           Up to 500 attempts; placed count may be less than target if the
           arena is too crowded.
        3. Randomly assign "box" or "cylinder".
        4. Randomly mark `dynamic_ratio` fraction of obstacles as dynamic;
           give them a random initial velocity so they move from the start.

        Size ranges
        -----------
        Box      : L  0.15–0.50 m,  W  0.15–0.50 m,  H  0.20–0.60 m
        Cylinder : r  0.08–0.20 m,  H  0.20–0.60 m

        Colours
        -------
        Static box      → orange-brown
        Static cylinder → green
        Dynamic box     → red-orange  (signals moving hazard)
        Dynamic cylinder→ magenta
        """
        target   = self._rng.randint(self._cfg.min_obstacles,
                                     self._cfg.max_obstacles)
        half     = self._cfg.arena_size / 2.0
        margin   = 0.45    # keep this far from each wall
        clear_r  = 0.80    # robot spawn safe zone (centred at origin)
        sep      = 0.35    # minimum distance between obstacle footprints

        # footprints of already-placed obstacles: list of (x, y, footprint_r)
        footprints: List[Tuple[float, float, float]] = []

        placed   = 0
        attempts = 0

        while placed < target and attempts < 600:
            attempts += 1

            x = self._rng.uniform(-half + margin, half - margin)
            y = self._rng.uniform(-half + margin, half - margin)

            # ── reject: inside robot spawn zone ──────────────────────────────
            if math.hypot(x, y) < clear_r:
                continue

            # ── reject: too close to an existing obstacle ─────────────────────
            shape = self._rng.choice(["box", "cylinder"])
            if shape == "box":
                lx  = self._rng.uniform(0.15, 0.50)
                ly  = self._rng.uniform(0.15, 0.50)
                lz  = self._rng.uniform(0.20, 0.60)
                yaw = self._rng.uniform(0, math.pi)
                footprint_r = math.hypot(lx, ly) / 2 + sep
            else:
                r   = self._rng.uniform(0.08, 0.20)
                lz  = self._rng.uniform(0.20, 0.60)
                yaw = 0.0
                footprint_r = r + sep

            too_close = any(
                math.hypot(x - fx, y - fy) < footprint_r + fr
                for fx, fy, fr in footprints
            )
            if too_close:
                continue

            # ── decide static vs dynamic ──────────────────────────────────────
            is_dynamic = self._rng.random() < self._cfg.dynamic_ratio
            mass       = 1.0 if is_dynamic else 0.0

            speed = self._rng.uniform(self._cfg.dynamic_speed_min,
                                      self._cfg.dynamic_speed_max)
            angle = self._rng.uniform(0, 2 * math.pi)
            vel   = (speed * math.cos(angle), speed * math.sin(angle)) if is_dynamic \
                    else (0.0, 0.0)

            # ── build ObstacleConfig ──────────────────────────────────────────
            if shape == "box":
                obs = ObstacleConfig(
                    shape="box",
                    position=(x, y, lz / 2),
                    size=(lx, ly, lz),
                    yaw=yaw,
                    mass=mass,
                    dynamic=is_dynamic,
                    velocity=vel,
                )
            else:
                obs = ObstacleConfig(
                    shape="cylinder",
                    position=(x, y, lz / 2),
                    size=(r, lz),
                    yaw=0.0,
                    mass=mass,
                    dynamic=is_dynamic,
                    velocity=vel,
                )

            self._place_obstacle(obs)
            footprints.append((x, y, footprint_r - sep))
            placed += 1

    # ── colours per obstacle category ────────────────────────────────────────
    _COLOUR_MAP = {
        ("box",      False): [0.70, 0.40, 0.10, 1.0],   # orange-brown static box
        ("cylinder", False): [0.20, 0.55, 0.20, 1.0],   # green static cylinder
        ("box",      True):  [0.85, 0.20, 0.10, 1.0],   # red dynamic box
        ("cylinder", True):  [0.75, 0.10, 0.75, 1.0],   # magenta dynamic cylinder
    }

    def _place_obstacle(self, obs: ObstacleConfig) -> None:
        """
        Spawn one obstacle with explicit p.createCollisionShape /
        p.createVisualShape / p.createMultiBody calls.

        Static  (mass=0) → immovable; only walls and the floor can stop the robot.
        Dynamic (mass>0) → full rigid-body physics; bounces off walls and other
                           obstacles; initial velocity set via resetBaseVelocity.
        """
        colour  = obs.colour or self._COLOUR_MAP.get(
            (obs.shape, obs.dynamic), _BOX_COLOUR
        )
        orn     = p.getQuaternionFromEuler([0, 0, obs.yaw])

        # ── 1. collision shape ────────────────────────────────────────────────
        if obs.shape == "box":
            lx, ly, lz = obs.size
            col_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[lx / 2, ly / 2, lz / 2],
                physicsClientId=self._client,
            )
            vis_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[lx / 2, ly / 2, lz / 2],
                rgbaColor=colour,
                physicsClientId=self._client,
            )
        elif obs.shape == "cylinder":
            radius, height = obs.size
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
        else:
            raise ValueError(f"Unknown obstacle shape: {obs.shape!r}")

        # ── 2. rigid body ─────────────────────────────────────────────────────
        body_id = p.createMultiBody(
            baseMass=obs.mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=list(obs.position),
            baseOrientation=list(orn),
            physicsClientId=self._client,
        )

        # ── 3. physics material properties ───────────────────────────────────
        p.changeDynamics(
            body_id, -1,
            lateralFriction=0.5,
            restitution=0.8,          # bouncy — dynamic obstacles bounce off walls
            linearDamping=0.05,       # light air drag to prevent infinite drift
            angularDamping=0.1,
            physicsClientId=self._client,
        )

        # ── 4. initial velocity (dynamic only) ────────────────────────────────
        if obs.dynamic and (obs.velocity[0] != 0 or obs.velocity[1] != 0):
            p.resetBaseVelocity(
                body_id,
                linearVelocity=[obs.velocity[0], obs.velocity[1], 0],
                angularVelocity=[0, 0, 0],
                physicsClientId=self._client,
            )

        # ── 5. register ───────────────────────────────────────────────────────
        if obs.dynamic:
            self._dynamic_ids.append(body_id)
        else:
            self._static_ids.append(body_id)

    def _clear_goal(self) -> None:
        """Remove all existing goal visuals before placing a new one."""
        for body_id in [self._goal_id, self._goal_sphere_id, self._goal_beacon_id]:
            if body_id is not None:
                try:
                    p.removeBody(body_id, physicsClientId=self._client)
                except Exception:
                    pass
        for line_id in self._goal_line_ids:
            try:
                p.removeUserDebugItem(line_id, physicsClientId=self._client)
            except Exception:
                pass
        self._goal_id        = None
        self._goal_sphere_id = None
        self._goal_beacon_id = None
        self._goal_line_ids  = []

    def _place_goal_marker(self, x: float, y: float) -> None:
        """
        Build the goal marker — 4 visual layers + debug annotations.

        Layer 1 — Flat disc (ground level)
        ────────────────────────────────────
        A wide flat cylinder (r=0.20 m, h=0.01 m) lying on the floor.
        Translucent green — easy to see without occluding the robot.
        No collision shape (collisionIndex=-1) so the robot drives over it.

        Layer 2 — Floating sphere
        ──────────────────────────
        A sphere (r=0.10 m) hovering 0.35 m above the disc centre.
        This is the primary visual target the agent should navigate toward.
        Supports pulse_goal() colour toggling between steps.

        Layer 3 — Beacon pillar
        ─────────────────────────
        A tall thin cylinder (r=0.03 m, h=1.20 m) rising from the centre.
        Visible from anywhere in the arena — acts like a lighthouse.

        Layer 4 — Debug crosshairs + vertical beacon line
        ──────────────────────────────────────────────────
        Four p.addUserDebugLine rays from the disc centre outward (±X, ±Y),
        plus a vertical line from floor to sphere height.
        Lines are cheap to draw and do not cost collision checks.

        Layer 5 — Text label
        ──────────────────────
        "GOAL" floating above the sphere.
        """
        self._clear_goal()
        self._goal_pos = (x, y)

        # ── Layer 1: ground disc ─────────────────────────────────────────────
        disc_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.20,
            length=0.01,
            rgbaColor=_GOAL_DISC,
            physicsClientId=self._client,
        )
        self._goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,   # no collision — robot rolls over it
            baseVisualShapeIndex=disc_vis,
            basePosition=[x, y, 0.005],
            physicsClientId=self._client,
        )

        # ── Layer 2: floating sphere ─────────────────────────────────────────
        sphere_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.10,
            rgbaColor=_GOAL_SPHERE,
            physicsClientId=self._client,
        )
        self._goal_sphere_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=sphere_vis,
            basePosition=[x, y, 0.35],
            physicsClientId=self._client,
        )

        # ── Layer 3: beacon pillar ───────────────────────────────────────────
        beacon_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.025,
            length=1.20,
            rgbaColor=_GOAL_BEACON,
            physicsClientId=self._client,
        )
        self._goal_beacon_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=beacon_vis,
            basePosition=[x, y, 0.60],
            physicsClientId=self._client,
        )

        # ── Layer 4: debug lines ─────────────────────────────────────────────
        arm   = 0.30   # crosshair arm length
        z_cr  = 0.02   # just above floor
        z_top = 0.40   # top of vertical beacon line

        # crosshair — 4 outward rays in ±X and ±Y
        crosshair_specs = [
            ([x,       y,       z_cr], [x + arm, y,       z_cr]),  # +X
            ([x,       y,       z_cr], [x - arm, y,       z_cr]),  # −X
            ([x,       y,       z_cr], [x,       y + arm, z_cr]),  # +Y
            ([x,       y,       z_cr], [x,       y - arm, z_cr]),  # −Y
        ]
        for start, end in crosshair_specs:
            lid = p.addUserDebugLine(
                start, end,
                lineColorRGB=_GOAL_LINE,
                lineWidth=2,
                physicsClientId=self._client,
            )
            self._goal_line_ids.append(lid)

        # vertical beacon line floor → sphere
        lid = p.addUserDebugLine(
            [x, y, 0.01], [x, y, z_top],
            lineColorRGB=_GOAL_LINE,
            lineWidth=1,
            physicsClientId=self._client,
        )
        self._goal_line_ids.append(lid)

        # ── Layer 5: text label ──────────────────────────────────────────────
        lid = p.addUserDebugText(
            "GOAL",
            [x, y, 0.55],
            textColorRGB=[0.0, 1.0, 0.4],
            textSize=1.4,
            physicsClientId=self._client,
        )
        self._goal_line_ids.append(lid)

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
