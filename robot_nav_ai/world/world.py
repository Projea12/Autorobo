"""
world/world.py — MuJoCo arena builder for AutoRobo v1.

Replaces the previous PyBullet World class entirely.

Architecture
────────────
The builder uses MuJoCo's MjSpec API to programmatically construct a
complete simulation scene from a robot MJCF file:

  worldbody
    floor         — infinite plane geom  (z = 0, static)
    walls × 4     — solid box geoms      (north/south/east/west, static)
    obstacle_i    — box/cylinder bodies  + freejoints  (pool of N)
    pickable_i    — graspable box bodies + freejoints  (pool of M)

Pool approach — avoids per-episode recompilation
─────────────────────────────────────────────────
All dynamic objects are baked into the compiled MjModel at build() time.
The pool size (max_obstacles, max_pickable) is fixed.  At each episode reset,
WorldState.randomize() teleports active objects to new positions via qpos and
hides inactive pool slots underground at z = −100 m.  This is O(n) in pool
size and does not touch the model.

Sizes (obstacle geometry) are randomised once at build() time.
Positions are randomised every episode reset via WorldState.randomize().

Usage
─────
    from world.world import WorldBuilder, WorldConfig

    cfg     = WorldConfig(arena_size=6.0, n_obstacles=6, n_pickable=1)
    builder = WorldBuilder(cfg)
    model, state = builder.build(ROBOT_XML_PATH)

    data = mujoco.MjData(model)
    rng  = np.random.default_rng(42)

    # At each episode reset:
    state.randomize(data, rng)
    mujoco.mj_forward(model, data)

    # Query object positions at runtime:
    pick_pos = state.pickable_pos(data, 0)   # (3,) world xyz
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import mujoco
import numpy as np


# ── hidden-away z for inactive pool slots ─────────────────────────────────────

_HIDDEN_Z: float = -100.0   # underground — never visible or reachable


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorldConfig:
    """
    All world-builder parameters in one immutable object.

    Attributes
    ----------
    arena_size      : side length of the square arena in metres
    wall_height     : metres above floor
    wall_thickness  : metres (collision width of each wall)
    max_obstacles   : pool size baked into the model (≥ n_obstacles)
    n_obstacles     : active obstacles per episode (randomised positions)
    max_pickable    : pool size baked into the model (≥ n_pickable)
    n_pickable      : active pickable objects per episode
    obstacle_clear_r: min radius around origin kept obstacle-free (robot spawn)
    wall_margin     : min distance between an obstacle and a wall
    sep             : min distance between any two obstacle centres
    build_seed      : RNG seed for geometry sizes (None = random each build)
    """
    arena_size:       float = 6.0
    wall_height:      float = 0.50
    wall_thickness:   float = 0.10
    max_obstacles:    int   = 12
    n_obstacles:      int   = 6
    max_pickable:     int   = 4
    n_pickable:       int   = 1
    obstacle_clear_r: float = 0.80
    wall_margin:      float = 0.40
    sep:              float = 0.35
    build_seed:       Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_obstacles > self.max_obstacles:
            raise ValueError("n_obstacles must be ≤ max_obstacles")
        if self.n_pickable > self.max_pickable:
            raise ValueError("n_pickable must be ≤ max_pickable")


# ── runtime object entry (one per pool slot) ──────────────────────────────────

class ObjectEntry(NamedTuple):
    """Index bookkeeping for one dynamic object in the MuJoCo model."""
    name:  str
    qadr:  int   # qpos start (freejoint = 7 floats: xyz + wxyz quat)
    vadr:  int   # qvel start (freejoint = 6 floats: v3 + ω3)
    half:  np.ndarray   # geom half-extents (3,) — for ground-plane z calculation


# ── world state ───────────────────────────────────────────────────────────────

class WorldState:
    """
    Runtime companion to a compiled WorldBuilder model.

    Holds the qpos/qvel addresses of every dynamic pool object and exposes
    randomize() so environments can scatter objects each episode without
    knowing any model internals.
    """

    def __init__(
        self,
        obstacle_entries: list[ObjectEntry],
        pickable_entries: list[ObjectEntry],
        cfg: WorldConfig,
    ) -> None:
        self._obs     = obstacle_entries
        self._pick    = pickable_entries
        self._cfg     = cfg

    # ── public API ────────────────────────────────────────────────────────────

    def randomize(
        self,
        data:         mujoco.MjData,
        rng:          np.random.Generator,
        n_obstacles:  Optional[int] = None,
        n_pickable:   Optional[int] = None,
    ) -> None:
        """
        Scatter active objects to random collision-free positions and hide
        inactive pool slots underground.

        Parameters
        ----------
        data        : live MjData (qpos/qvel are mutated in place)
        rng         : NumPy Generator (caller controls the seed)
        n_obstacles : override active count (default = WorldConfig.n_obstacles)
        n_pickable  : override active count (default = WorldConfig.n_pickable)
        """
        n_obs  = n_obstacles if n_obstacles is not None else self._cfg.n_obstacles
        n_pick = n_pickable  if n_pickable  is not None else self._cfg.n_pickable

        placed: list[tuple[float, float, float]] = []   # (x, y, r) of placed objects

        for i, entry in enumerate(self._obs):
            if i < n_obs:
                pos = self._sample_position(rng, placed, entry.half)
                if pos is not None:
                    self._place(data, entry, pos)
                    placed.append((*pos[:2], float(np.max(entry.half[:2]))))
                else:
                    self._hide(data, entry)
            else:
                self._hide(data, entry)

        for i, entry in enumerate(self._pick):
            if i < n_pick:
                pos = self._sample_position(rng, placed, entry.half,
                                            clear_r=0.0)   # pickable may be closer to robot
                if pos is not None:
                    self._place(data, entry, pos)
                    placed.append((*pos[:2], float(np.max(entry.half[:2]))))
                else:
                    self._hide(data, entry)
            else:
                self._hide(data, entry)

    def randomize_obstacles(
        self,
        data:        mujoco.MjData,
        rng:         np.random.Generator,
        n_obstacles: Optional[int] = None,
    ) -> None:
        """Randomise only the obstacle pool (leave pickable objects untouched)."""
        n_obs  = n_obstacles if n_obstacles is not None else self._cfg.n_obstacles
        placed: list[tuple[float, float, float]] = []
        for i, entry in enumerate(self._obs):
            if i < n_obs:
                pos = self._sample_position(rng, placed, entry.half)
                if pos is not None:
                    self._place(data, entry, pos)
                    placed.append((*pos[:2], float(np.max(entry.half[:2]))))
                else:
                    self._hide(data, entry)
            else:
                self._hide(data, entry)

    def randomize_pickable(
        self,
        data:       mujoco.MjData,
        rng:        np.random.Generator,
        n_pickable: Optional[int] = None,
        *,
        x_range: tuple[float, float] = (0.40, 0.85),
        y_range: tuple[float, float] = (-0.30, 0.30),
    ) -> None:
        """
        Randomise only the pickable object pool.

        x_range, y_range — placement region in world frame.  Defaults match
        ManipulationEnv._TARGET_{X,Y}_RANGE so the two can stay in sync.
        """
        n_pick = n_pickable if n_pickable is not None else self._cfg.n_pickable
        for i, entry in enumerate(self._pick):
            if i < n_pick:
                x = float(rng.uniform(*x_range))
                y = float(rng.uniform(*y_range))
                z = float(entry.half[2])          # sit exactly on the floor
                self._place(data, entry, (x, y, z))
            else:
                self._hide(data, entry)

    def pickable_pos(self, data: mujoco.MjData, idx: int) -> np.ndarray:
        """Return current world xyz of the idx-th pickable object (3,)."""
        entry = self._pick[idx]
        return data.qpos[entry.qadr : entry.qadr + 3].copy()

    def obstacle_pos(self, data: mujoco.MjData, idx: int) -> np.ndarray:
        """Return current world xyz of the idx-th obstacle (3,)."""
        entry = self._obs[idx]
        return data.qpos[entry.qadr : entry.qadr + 3].copy()

    @property
    def n_obstacle_slots(self) -> int:
        return len(self._obs)

    @property
    def n_pickable_slots(self) -> int:
        return len(self._pick)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _sample_position(
        self,
        rng:     np.random.Generator,
        placed:  list[tuple[float, float, float]],
        half:    np.ndarray,
        clear_r: Optional[float] = None,
        n_tries: int = 400,
    ) -> Optional[tuple[float, float, float]]:
        """
        Sample a collision-free (x, y, z) for an obstacle.

        Returns None if no valid position found within n_tries attempts.
        z is set so the object sits exactly on the floor.
        """
        cfg    = self._cfg
        half_a = cfg.arena_size / 2.0 - cfg.wall_margin
        obj_r  = float(np.max(half[:2]))
        cr     = clear_r if clear_r is not None else cfg.obstacle_clear_r

        for _ in range(n_tries):
            x = float(rng.uniform(-half_a + obj_r, half_a - obj_r))
            y = float(rng.uniform(-half_a + obj_r, half_a - obj_r))

            if cr > 0 and math.hypot(x, y) < cr:
                continue

            too_close = any(
                math.hypot(x - px, y - py) < obj_r + pr + cfg.sep
                for px, py, pr in placed
            )
            if too_close:
                continue

            z = float(half[2])   # centre height = half-height above floor
            return (x, y, z)

        return None

    @staticmethod
    def _place(
        data:  mujoco.MjData,
        entry: ObjectEntry,
        pos:   tuple[float, float, float],
    ) -> None:
        qa = entry.qadr
        va = entry.vadr
        data.qpos[qa]     = pos[0]
        data.qpos[qa + 1] = pos[1]
        data.qpos[qa + 2] = pos[2]
        data.qpos[qa + 3] = 1.0   # quaternion w  (identity orientation)
        data.qpos[qa + 4] = 0.0
        data.qpos[qa + 5] = 0.0
        data.qpos[qa + 6] = 0.0
        data.qvel[va : va + 6] = 0.0

    @staticmethod
    def _hide(data: mujoco.MjData, entry: ObjectEntry) -> None:
        qa = entry.qadr
        va = entry.vadr
        data.qpos[qa]     = 0.0
        data.qpos[qa + 1] = 0.0
        data.qpos[qa + 2] = _HIDDEN_Z
        data.qpos[qa + 3] = 1.0
        data.qpos[qa + 4] = 0.0
        data.qpos[qa + 5] = 0.0
        data.qpos[qa + 6] = 0.0
        data.qvel[va : va + 6] = 0.0


# ── builder ───────────────────────────────────────────────────────────────────

# RGBA colours for scene elements
_FLOOR_RGBA    = [0.55, 0.55, 0.55, 1.0]
_WALL_RGBA     = [0.30, 0.30, 0.35, 1.0]
_OBS_BOX_RGBA  = [0.70, 0.40, 0.10, 1.0]   # warm brown static box
_OBS_CYL_RGBA  = [0.20, 0.55, 0.20, 1.0]   # green static cylinder
_PICK_RGBA     = [1.00, 0.35, 0.05, 1.0]   # orange — matches ManipulationEnv target


class WorldBuilder:
    """
    Programmatically constructs a MuJoCo simulation arena.

    Loads the robot MJCF, appends the world (floor, walls, obstacle pool,
    pickable-object pool) via MjSpec, compiles once, and returns the
    immutable (MjModel, WorldState) pair.

    Parameters
    ----------
    config : WorldConfig controlling arena geometry and pool sizes
    """

    def __init__(self, config: WorldConfig = WorldConfig()) -> None:
        self._cfg = config

    # ── public API ────────────────────────────────────────────────────────────

    def build(
        self,
        robot_xml_path: str,
        *,
        build_rng: Optional[np.random.Generator] = None,
    ) -> tuple[mujoco.MjModel, WorldState]:
        """
        Load robot MJCF, add the world, compile, return (model, state).

        Parameters
        ----------
        robot_xml_path : absolute path to robot.xml
        build_rng      : RNG for obstacle geometry sizes.  If None a fresh
                         generator seeded by WorldConfig.build_seed is used.

        Returns
        -------
        model : compiled MjModel (nq / nv increased by pool objects)
        state : WorldState with all qpos/qvel addresses resolved
        """
        rng = build_rng or np.random.default_rng(self._cfg.build_seed)

        spec = mujoco.MjSpec.from_file(robot_xml_path)

        self._add_floor(spec)
        self._add_walls(spec)

        obs_specs  = self._add_obstacle_pool(spec, rng)
        pick_specs = self._add_pickable_pool(spec)

        model = spec.compile()

        obs_entries  = self._resolve_entries(model, obs_specs)
        pick_entries = self._resolve_entries(model, pick_specs)

        state = WorldState(obs_entries, pick_entries, self._cfg)
        return model, state

    # ── floor ─────────────────────────────────────────────────────────────────

    def _add_floor(self, spec: mujoco.MjSpec) -> None:
        """
        Infinite ground plane at z = 0.

        MuJoCo plane geom: size[2] = grid spacing for rendering (0.05 m).
        The plane extends infinitely — size[0] and size[1] are irrelevant for
        collision but set here to match arena size for the visual texture grid.
        """
        g = spec.worldbody.add_geom()
        g.name     = "world_floor"
        g.type     = mujoco.mjtGeom.mjGEOM_PLANE
        g.size     = [self._cfg.arena_size / 2, self._cfg.arena_size / 2, 0.05]
        g.pos      = [0.0, 0.0, 0.0]
        g.rgba     = _FLOOR_RGBA
        g.friction = [1.0, 0.005, 0.0001]
        g.contype  = 1
        g.conaffinity = 1

    # ── walls ─────────────────────────────────────────────────────────────────

    def _add_walls(self, spec: mujoco.MjSpec) -> None:
        """
        Four solid box geoms anchored to worldbody forming a closed arena.

        Corner strategy: north/south walls span the full width + 2×thickness
        so east/west walls slot inside them with no gap.

            ╔══════════════╗  ← north  (full width)
            ║              ║
            ║    arena     ║  ← east/west (inner span only)
            ║              ║
            ╚══════════════╝  ← south  (full width)
        """
        S = self._cfg.arena_size
        H = self._cfg.wall_height
        T = self._cfg.wall_thickness

        walls = {
            "wall_north": ([0.0,   S/2,   H/2], [S/2 + T, T/2, H/2]),
            "wall_south": ([0.0,  -S/2,   H/2], [S/2 + T, T/2, H/2]),
            "wall_east":  ([ S/2,  0.0,   H/2], [T/2,  S/2,    H/2]),
            "wall_west":  ([-S/2,  0.0,   H/2], [T/2,  S/2,    H/2]),
        }

        for name, (pos, half) in walls.items():
            g = spec.worldbody.add_geom()
            g.name        = name
            g.type        = mujoco.mjtGeom.mjGEOM_BOX
            g.size        = half
            g.pos         = pos
            g.rgba        = _WALL_RGBA
            g.contype     = 1
            g.conaffinity = 1

    # ── obstacle pool ─────────────────────────────────────────────────────────

    def _add_obstacle_pool(
        self,
        spec: mujoco.MjSpec,
        rng:  np.random.Generator,
    ) -> list[tuple[str, np.ndarray]]:
        """
        Add max_obstacles bodies with freejoints to the worldbody.

        Half the pool is boxes, half are cylinders.
        Sizes are randomised at build time and fixed for the model lifetime.
        All are initially hidden at z = −100 m; WorldState.randomize() places
        the active ones at episode reset.

        Returns list of (joint_name, half_extents) for address resolution.
        """
        n        = self._cfg.max_obstacles
        n_box    = n // 2
        n_cyl    = n - n_box
        entries: list[tuple[str, np.ndarray]] = []

        for i in range(n_box):
            lx   = float(rng.uniform(0.15, 0.45))
            ly   = float(rng.uniform(0.15, 0.45))
            lz   = float(rng.uniform(0.25, 0.55))
            half = np.array([lx / 2, ly / 2, lz / 2])
            name = f"obstacle_box_{i}"
            jname = f"obstacle_box_{i}_joint"

            body       = spec.worldbody.add_body()
            body.name  = name
            body.pos   = [0.0, 0.0, _HIDDEN_Z]

            fj      = body.add_freejoint()
            fj.name = jname

            g          = body.add_geom()
            g.name     = f"{name}_geom"
            g.type     = mujoco.mjtGeom.mjGEOM_BOX
            g.size     = half.tolist()
            g.mass     = float(lx * ly * lz * 500)   # 500 kg/m³ ≈ dense wood
            g.rgba     = _OBS_BOX_RGBA
            g.friction = [0.8, 0.005, 0.0001]

            entries.append((jname, half))

        for i in range(n_cyl):
            r    = float(rng.uniform(0.08, 0.18))
            lz   = float(rng.uniform(0.25, 0.55))
            half = np.array([r, r, lz / 2])
            name = f"obstacle_cyl_{i}"
            jname = f"obstacle_cyl_{i}_joint"

            body       = spec.worldbody.add_body()
            body.name  = name
            body.pos   = [0.0, 0.0, _HIDDEN_Z]

            fj      = body.add_freejoint()
            fj.name = jname

            g          = body.add_geom()
            g.name     = f"{name}_geom"
            g.type     = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size     = [r, lz / 2, 0.0]   # MuJoCo cylinder: size[0]=r, size[1]=half-h
            g.mass     = float(math.pi * r * r * lz * 500)
            g.rgba     = _OBS_CYL_RGBA
            g.friction = [0.8, 0.005, 0.0001]

            entries.append((jname, half))

        return entries

    # ── pickable pool ─────────────────────────────────────────────────────────

    def _add_pickable_pool(
        self,
        spec: mujoco.MjSpec,
    ) -> list[tuple[str, np.ndarray]]:
        """
        Add max_pickable graspable box bodies with freejoints.

        Pickable objects are 5×5×5 cm boxes — matches the ManipulationEnv
        target object so reward shaping stays consistent.

        Returns list of (joint_name, half_extents) for address resolution.
        """
        entries: list[tuple[str, np.ndarray]] = []
        side = 0.025   # 5 cm box: half = 0.025 m
        half = np.array([side, side, side])

        for i in range(self._cfg.max_pickable):
            name  = f"pickable_{i}"
            jname = f"pickable_{i}_joint"

            body       = spec.worldbody.add_body()
            body.name  = name
            body.pos   = [0.0, 0.0, _HIDDEN_Z]

            fj      = body.add_freejoint()
            fj.name = jname

            g          = body.add_geom()
            g.name     = f"{name}_geom"
            g.type     = mujoco.mjtGeom.mjGEOM_BOX
            g.size     = half.tolist()
            g.mass     = 0.20
            g.rgba     = _PICK_RGBA
            g.friction = [1.5, 0.5, 0.5]

            entries.append((jname, half))

        return entries

    # ── qpos address resolution ───────────────────────────────────────────────

    @staticmethod
    def _resolve_entries(
        model:   mujoco.MjModel,
        specs:   list[tuple[str, np.ndarray]],
    ) -> list[ObjectEntry]:
        """
        After compilation, resolve joint names → qpos/qvel addresses.
        """
        entries: list[ObjectEntry] = []
        for jname, half in specs:
            jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"Joint '{jname}' not found after compile")
            qadr = int(model.jnt_qposadr[jid])
            vadr = int(model.jnt_dofadr[jid])
            entries.append(ObjectEntry(name=jname, qadr=qadr, vadr=vadr,
                                       half=half.copy()))
        return entries


# ── convenience factory ───────────────────────────────────────────────────────

def build_world(
    robot_xml_path: str,
    *,
    arena_size:    float        = 6.0,
    n_obstacles:   int          = 6,
    n_pickable:    int          = 1,
    build_seed:    Optional[int] = None,
    **cfg_kwargs,
) -> tuple[mujoco.MjModel, WorldState]:
    """
    One-line world construction shortcut.

        model, state = build_world(ROBOT_XML_PATH, n_obstacles=6)

    Extra keyword arguments are forwarded to WorldConfig.
    """
    cfg     = WorldConfig(
        arena_size=arena_size,
        n_obstacles=n_obstacles,
        n_pickable=n_pickable,
        build_seed=build_seed,
        **cfg_kwargs,
    )
    builder = WorldBuilder(cfg)
    return builder.build(robot_xml_path)
