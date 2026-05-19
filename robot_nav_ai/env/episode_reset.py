"""
env/episode_reset.py — Episode reset logic for AutoRobo v1.

Provides three independent randomisers and an EpisodeResetter that composes
them into a single reset() call used by training environments.

Randomisers
───────────
  randomise_robot_spawn   — freejoint (x, y, yaw); z kept from keyframe
  randomise_obstacles     — delegates to WorldState.randomize_obstacles()
  randomise_goal          — pickable object OR nav-goal position

Ordering inside reset()
────────────────────────
  1. mj_resetDataKeyframe      → home arm pose, pool objects at z=0 (harmless)
  2. randomise_obstacles        → scatter obstacles; clears spawn zone
  3. randomise_robot_spawn      → robot placed clear of obstacles
  4. randomise_goal             → goal placed relative to robot (or world frame)
  5. domain_rand.randomize()   → visual + physical variation (optional)
  6. mj_forward                 → propagate all changes

Usage
─────
    from env.episode_reset import EpisodeResetter, SpawnConfig, GoalConfig

    resetter = EpisodeResetter(
        home_kf_id   = kf_id,
        world_state  = state,          # optional — WorldState from WorldBuilder
        domain_rand  = randomizer,     # optional — DomainRandomizer
    )

    episode_info = resetter.reset(model, data, rng)
    print(episode_info.robot_xy, episode_info.goal_xyz)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from world.world import WorldState
from env.domain_rand import DomainRandomizer


# ── configuration dataclasses ─────────────────────────────────────────────────

@dataclass(frozen=True)
class SpawnConfig:
    """
    Robot spawn constraints.

    The robot base freejoint is set to (x, y, z_fixed, yaw) where z_fixed
    comes from the home keyframe so the robot rests correctly on the floor.

    clear_r_robot : geoms in the robot footprint — stay at least this far from
                    any obstacle centre so the robot does not spawn overlapping
                    an obstacle.  Checked via rejection sampling.
    """
    x_range:       tuple[float, float] = (-1.50, 1.50)
    y_range:       tuple[float, float] = (-1.50, 1.50)
    yaw_range:     tuple[float, float] = (-math.pi, math.pi)
    clear_r_robot: float               = 0.40   # m — robot chassis radius + margin
    max_tries:     int                 = 300    # rejection-sample tries before fallback


@dataclass(frozen=True)
class GoalConfig:
    """
    Pickable-object / navigation-goal placement constraints.

    mode = "relative"
        Goal is placed at (fwd, lat) in the robot's own frame, then rotated
        to world frame by the robot's spawn yaw.  This ensures the goal is
        always in front of the robot regardless of spawn orientation.

    mode = "world"
        Goal is placed at an absolute (x, y) in the world frame, ignoring the
        robot's spawn orientation.  Useful when the robot always spawns at the
        origin facing +X (ManipulationEnv default).
    """
    mode:           str                = "relative"   # "relative" | "world"

    # relative-mode ranges (in robot frame)
    fwd_range:      tuple[float, float] = (0.40, 0.85)
    lat_range:      tuple[float, float] = (-0.30, 0.30)

    # world-mode ranges
    x_range:        tuple[float, float] = (0.40, 0.85)
    y_range:        tuple[float, float] = (-0.30, 0.30)

    # z is set so the object sits on the floor
    z:              float               = 0.025   # box half-height


# ── episode information ───────────────────────────────────────────────────────

@dataclass
class EpisodeInfo:
    """Return value of EpisodeResetter.reset() — actual sampled values."""
    robot_xy:          np.ndarray   # (2,) world-frame base position
    robot_yaw:         float        # radians, CCW from +X
    goal_xyz:          np.ndarray   # (3,) world-frame goal centre
    n_active_obstacles: int


# ── main class ────────────────────────────────────────────────────────────────

class EpisodeResetter:
    """
    Orchestrates a full episode reset for any MuJoCo environment built on top
    of the WorldBuilder + DomainRandomizer stack.

    Parameters
    ----------
    home_kf_id    : index of the "home" keyframe in MjModel.key_qpos
    world_state   : WorldState from WorldBuilder.build() — required for obstacle
                    and pickable randomisation; None → only robot spawn changes
    domain_rand   : DomainRandomizer — applied after layout randomisation;
                    None → no visual/physical variation
    spawn_cfg     : SpawnConfig controlling robot spawn distribution
    goal_cfg      : GoalConfig controlling goal placement
    n_obstacles   : override active obstacle count (None → WorldState default)
    n_pickable    : override active pickable count (None → WorldState default)
    """

    def __init__(
        self,
        home_kf_id:  int,
        world_state: Optional[WorldState]      = None,
        domain_rand: Optional[DomainRandomizer] = None,
        spawn_cfg:   SpawnConfig               = SpawnConfig(),
        goal_cfg:    GoalConfig                = GoalConfig(),
        n_obstacles: Optional[int]             = None,
        n_pickable:  Optional[int]             = None,
    ) -> None:
        self._kf_id       = home_kf_id
        self._world_state = world_state
        self._domain_rand = domain_rand
        self._spawn_cfg   = spawn_cfg
        self._goal_cfg    = goal_cfg
        self._n_obstacles = n_obstacles
        self._n_pickable  = n_pickable

    # ── public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        model: mujoco.MjModel,
        data:  mujoco.MjData,
        rng:   np.random.Generator,
    ) -> EpisodeInfo:
        """
        Full episode reset — call at the start of every episode.

        Execution order (see module docstring):
          1. Keyframe reset
          2. Obstacle randomisation
          3. Robot spawn
          4. Goal placement
          5. Domain randomisation (visual + physical)
          6. mj_forward

        Parameters
        ----------
        model : MjModel — shared; domain_rand may mutate it in-place
        data  : MjData — reset in place
        rng   : NumPy Generator seeded by caller for reproducibility

        Returns
        -------
        EpisodeInfo with the actual positions sampled this episode.
        """
        # 1. Arm and gripper back to home pose; pool objects placed at origin
        mujoco.mj_resetDataKeyframe(model, data, self._kf_id)

        # 2. Scatter obstacles (must come before spawn so clearance check works)
        n_obs = 0
        if self._world_state is not None:
            self.randomise_obstacles(data, rng)
            n_obs = (self._n_obstacles
                     if self._n_obstacles is not None
                     else self._world_state._cfg.n_obstacles)

        # 3. Place robot at a random collision-free spawn
        robot_xy, robot_yaw = self.randomise_robot_spawn(model, data, rng)

        # 4. Place goal (pickable object or nav-goal marker)
        goal_xyz = self.randomise_goal(data, rng,
                                       robot_xy=robot_xy,
                                       robot_yaw=robot_yaw)

        # 5. Visual + physical variation
        if self._domain_rand is not None:
            self._domain_rand.randomize(model, data, rng)

        # 6. Propagate all position / property changes
        mujoco.mj_forward(model, data)

        return EpisodeInfo(
            robot_xy           = robot_xy,
            robot_yaw          = robot_yaw,
            goal_xyz           = goal_xyz,
            n_active_obstacles = n_obs,
        )

    # ── individual randomisers ────────────────────────────────────────────────

    def randomise_robot_spawn(
        self,
        model: mujoco.MjModel,
        data:  mujoco.MjData,
        rng:   np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        """
        Place the robot at a random (x, y, yaw) within the configured ranges.

        z is taken from the home keyframe (already in qpos[2]) so the robot
        rests correctly on the floor without needing physics settle time.

        Obstacle clearance is enforced via rejection sampling against the
        current obstacle qpos positions (if WorldState is available).

        Returns
        -------
        (robot_xy, robot_yaw) — (2,) array and float
        """
        cfg   = self._spawn_cfg
        z_fix = float(data.qpos[2])   # keyframe z — keep unchanged

        # Build list of active obstacle (x, y) positions for clearance check
        obstacle_xy: list[tuple[float, float]] = []
        if self._world_state is not None:
            ws  = self._world_state
            n   = (self._n_obstacles if self._n_obstacles is not None
                   else ws._cfg.n_obstacles)
            for i in range(min(n, ws.n_obstacle_slots)):
                pos = ws.obstacle_pos(data, i)
                if pos[2] > -1.0:   # only active (non-hidden) obstacles
                    obstacle_xy.append((float(pos[0]), float(pos[1])))

        # Rejection-sample spawn position
        xy, yaw = self._sample_spawn(rng, cfg, obstacle_xy)

        # Write freejoint qpos: [x, y, z, w, qx, qy, qz]
        data.qpos[0] = xy[0]
        data.qpos[1] = xy[1]
        data.qpos[2] = z_fix
        data.qpos[3] = math.cos(yaw / 2.0)   # quaternion w
        data.qpos[4] = 0.0                     # quaternion x
        data.qpos[5] = 0.0                     # quaternion y
        data.qpos[6] = math.sin(yaw / 2.0)   # quaternion z  (Z-axis rotation)

        # Zero base velocities
        data.qvel[0:6] = 0.0

        return xy, yaw

    def randomise_obstacles(
        self,
        data: mujoco.MjData,
        rng:  np.random.Generator,
    ) -> None:
        """
        Scatter obstacle pool to new random collision-free positions.

        No-op if no WorldState was provided at construction.
        Delegates entirely to WorldState.randomize_obstacles() which
        enforces spawn-zone clearance and inter-obstacle separation.
        """
        if self._world_state is None:
            return
        self._world_state.randomize_obstacles(
            data, rng,
            n_obstacles=self._n_obstacles,
        )

    def randomise_goal(
        self,
        data:      mujoco.MjData,
        rng:       np.random.Generator,
        robot_xy:  Optional[np.ndarray] = None,
        robot_yaw: float                = 0.0,
    ) -> np.ndarray:
        """
        Place the goal (pickable object or navigation target) and return its
        world-frame xyz position.

        Parameters
        ----------
        data      : MjData — pickable object qpos is mutated in place
        rng       : NumPy Generator
        robot_xy  : (2,) world-frame robot base position (for relative mode)
        robot_yaw : robot yaw in radians (for relative mode)

        Returns
        -------
        (3,) goal world-frame xyz.
        """
        cfg = self._goal_cfg

        if cfg.mode == "relative":
            goal_xyz = self._goal_relative(rng, robot_xy, robot_yaw, cfg)
        elif cfg.mode == "world":
            goal_xyz = self._goal_world(rng, cfg)
        else:
            raise ValueError(f"Unknown goal mode {cfg.mode!r}. Use 'relative' or 'world'.")

        # Place first active pickable object at the goal position
        if self._world_state is not None and self._world_state.n_pickable_slots > 0:
            n_pick = (self._n_pickable if self._n_pickable is not None
                      else self._world_state._cfg.n_pickable)
            if n_pick > 0:
                entry = self._world_state._pick[0]
                WorldState._place(data, entry,
                                  (float(goal_xyz[0]),
                                   float(goal_xyz[1]),
                                   float(goal_xyz[2])))
                # Remaining pickable slots → hidden
                for i in range(1, self._world_state.n_pickable_slots):
                    WorldState._hide(data, self._world_state._pick[i])

        return goal_xyz

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sample_spawn(
        rng:         np.random.Generator,
        cfg:         SpawnConfig,
        obstacle_xy: list[tuple[float, float]],
    ) -> tuple[np.ndarray, float]:
        """Rejection-sample a collision-free spawn (x, y, yaw)."""
        for _ in range(cfg.max_tries):
            x   = float(rng.uniform(*cfg.x_range))
            y   = float(rng.uniform(*cfg.y_range))
            too_close = any(
                math.hypot(x - ox, y - oy) < cfg.clear_r_robot
                for ox, oy in obstacle_xy
            )
            if not too_close:
                yaw = float(rng.uniform(*cfg.yaw_range))
                return np.array([x, y]), yaw

        # Fallback: origin (always obstacle-free due to WorldConfig.obstacle_clear_r)
        yaw = float(rng.uniform(*cfg.yaw_range))
        return np.array([0.0, 0.0]), yaw

    @staticmethod
    def _goal_relative(
        rng:       np.random.Generator,
        robot_xy:  Optional[np.ndarray],
        robot_yaw: float,
        cfg:       GoalConfig,
    ) -> np.ndarray:
        """
        Sample goal in robot frame then rotate to world frame.

        Robot frame: +X forward, +Y left (CCW positive yaw convention).
        Goal at (fwd, lat) in robot frame maps to world frame as:

            gx = rx + fwd·cos(θ) − lat·sin(θ)
            gy = ry + fwd·sin(θ) + lat·cos(θ)
        """
        fwd = float(rng.uniform(*cfg.fwd_range))
        lat = float(rng.uniform(*cfg.lat_range))
        c, s = math.cos(robot_yaw), math.sin(robot_yaw)
        rx, ry = (float(robot_xy[0]), float(robot_xy[1])) if robot_xy is not None else (0.0, 0.0)
        gx = rx + fwd * c - lat * s
        gy = ry + fwd * s + lat * c
        return np.array([gx, gy, cfg.z])

    @staticmethod
    def _goal_world(
        rng: np.random.Generator,
        cfg: GoalConfig,
    ) -> np.ndarray:
        """Sample goal at an absolute world-frame (x, y, z)."""
        gx = float(rng.uniform(*cfg.x_range))
        gy = float(rng.uniform(*cfg.y_range))
        return np.array([gx, gy, cfg.z])


# ── module-level convenience ──────────────────────────────────────────────────

def make_resetter(
    model:       mujoco.MjModel,
    world_state: Optional[WorldState]       = None,
    domain_rand: Optional[DomainRandomizer]  = None,
    spawn_cfg:   SpawnConfig                = SpawnConfig(),
    goal_cfg:    GoalConfig                 = GoalConfig(),
    n_obstacles: Optional[int]              = None,
    n_pickable:  Optional[int]              = None,
) -> EpisodeResetter:
    """
    Build an EpisodeResetter from a compiled model, resolving home_kf_id.

    Raises RuntimeError if the model has no "home" keyframe.
    """
    kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kf_id < 0:
        raise RuntimeError("Model has no 'home' keyframe — add one to the robot MJCF.")
    return EpisodeResetter(
        home_kf_id  = kf_id,
        world_state = world_state,
        domain_rand = domain_rand,
        spawn_cfg   = spawn_cfg,
        goal_cfg    = goal_cfg,
        n_obstacles = n_obstacles,
        n_pickable  = n_pickable,
    )
