"""
env/nav_obs.py — Navigation observation space for mobile robot RL.

Observation layout  (NAV_OBS_DIM = 128 floats, all float32)
════════════════════════════════════════════════════════════

Group A — Robot state  [0:8]  8 dims
──────────────────────────────────────
  [0]    x  position  (m, normalised by MAP_HALF)
  [1]    y  position  (m, normalised by MAP_HALF)
  [2]    cos(yaw)     — avoids wrap-around discontinuity
  [3]    sin(yaw)
  [4]    v_x  linear velocity  (m/s, / VEL_MAX)
  [5]    v_y  lateral velocity (m/s, / VEL_MAX)   ← always ~0 for diff-drive
  [6]    ω    angular velocity (rad/s, / ANG_VEL_MAX)
  [7]    progress  = 1 − dist_to_goal/initial_dist   (0→1 as robot approaches)

Group B — Goal  [8:12]  4 dims
──────────────────────────────
  [8]    dist_to_goal  (m, / GOAL_DIST_MAX, clipped [0,1])
  [9]    cos(bearing)  — bearing = angle from robot heading to goal
  [10]   sin(bearing)
  [11]   goal_reached  flag  (0.0 / 1.0)

Group C — Lidar ring  [12:48]  36 dims  (N_RAYS = 36, 10° spacing)
────────────────────────────────────────────────────────────────────
  [12:48]  ray_i = dist_i / LIDAR_MAX_RANGE, clipped [0,1]
            ray_0 points along robot's forward axis (+x body frame)
            rays proceed counter-clockwise at 10° intervals
            1.0 = no obstacle within LIDAR_MAX_RANGE
            Body of robot excluded from ray intersections

Group D — Nearest obstacles  [48:60]  12 dims  (N_NEAR = 4 × 3)
──────────────────────────────────────────────────────────────────
  For each of the 4 closest lidar hits (sorted by distance):
    [3k+48]   dist   / LIDAR_MAX_RANGE, clipped [0,1]
    [3k+49]   cos(angle in robot frame)
    [3k+50]   sin(angle in robot frame)
  Padding (1.0, 0.0, 0.0) if fewer than 4 obstacles in range.

Group E — Perception  [60:64]  4 dims
──────────────────────────────────────
  [60]   target_confidence  (0=not detected, 1=certain)
  [61]   cos(target_bearing)  — bearing in robot frame; 0 if not detected
  [62]   sin(target_bearing)
  [63]   target_dist / GOAL_DIST_MAX  (estimated from bounding-box size; 0 if undetected)

Group F — Egocentric occupancy map  [64:128]  64 dims  (8×8 grid)
──────────────────────────────────────────────────────────────────
  Local 8×8 occupancy grid aligned to robot heading.
  Row-major, row 0 = furthest forward, row 7 = furthest backward.
  Cell (r, c):
    0.0  = free
    1.0  = occupied (obstacle within cell volume)
    0.5  = unknown  (not covered by any lidar ray)
  Grid spans GRID_SIZE_M × GRID_SIZE_M metres (default 4.0 m × 4.0 m).
  Cell resolution = GRID_SIZE_M / GRID_N = 0.5 m.

Total: 8 + 4 + 36 + 12 + 4 + 64 = 128 dims
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import mujoco
import numpy as np

# ── observation dimension ─────────────────────────────────────────────────────

N_RAYS:       int   = 36           # lidar rays (360° / 10°)
N_NEAR:       int   = 4            # nearest-obstacle slots
GRID_N:       int   = 8            # occupancy grid side length
NAV_OBS_DIM:  int   = 8 + 4 + N_RAYS + N_NEAR * 3 + 4 + GRID_N * GRID_N
assert NAV_OBS_DIM == 128

# ── slice constants (use these everywhere instead of magic numbers) ────────────

SL_ROBOT   = slice(0,  8)           # Group A
SL_GOAL    = slice(8,  12)          # Group B
SL_LIDAR   = slice(12, 12 + N_RAYS) # Group C  [12:48]
SL_NEAR    = slice(48, 60)          # Group D
SL_PERCEPT = slice(60, 64)          # Group E
SL_OCC     = slice(64, 128)         # Group F

# named sub-slices for Group A
IDX_X        = 0
IDX_Y        = 1
IDX_COS_YAW  = 2
IDX_SIN_YAW  = 3
IDX_VX       = 4
IDX_VY       = 5
IDX_OMEGA    = 6
IDX_PROGRESS = 7

# named sub-slices for Group B (relative to SL_GOAL.start)
IDX_GOAL_DIST    = 8
IDX_GOAL_COS_BRG = 9
IDX_GOAL_SIN_BRG = 10
IDX_GOAL_REACHED = 11


# ── normalization constants ───────────────────────────────────────────────────

MAP_HALF:        float = 10.0    # m  — arena half-width for position norm
VEL_MAX:         float = 2.0     # m/s
ANG_VEL_MAX:     float = 3.0     # rad/s
LIDAR_MAX_RANGE: float = 5.0     # m
GOAL_DIST_MAX:   float = 15.0    # m
GRID_SIZE_M:     float = 4.0     # m  — total footprint of local map


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NavObsConfig:
    """
    Hyper-parameters for the navigation observation builder.

    n_rays           : number of lidar rays (must divide 360 evenly)
    n_near           : number of nearest-obstacle slots (Group D)
    grid_n           : side length of egocentric occupancy grid (Group F)
    lidar_max_range  : maximum lidar range in metres
    lidar_height     : height above floor at which rays are cast
    goal_dist_max    : distance normalisation constant for goal
    map_half         : arena half-width used to normalise (x, y)
    vel_max          : linear velocity normalisation
    ang_vel_max      : angular velocity normalisation
    grid_size_m      : side length of local occupancy map in metres
    robot_body_name  : MuJoCo body name to exclude from raycasts
    """
    n_rays:          int   = N_RAYS
    n_near:          int   = N_NEAR
    grid_n:          int   = GRID_N
    lidar_max_range: float = LIDAR_MAX_RANGE
    lidar_height:    float = 0.15          # m above floor
    goal_dist_max:   float = GOAL_DIST_MAX
    map_half:        float = MAP_HALF
    vel_max:         float = VEL_MAX
    ang_vel_max:     float = ANG_VEL_MAX
    grid_size_m:     float = GRID_SIZE_M
    robot_body_name: str   = "base_link"
    geomgroup_mask:  Optional[tuple[int, ...]] = None   # None = all groups

    @property
    def obs_dim(self) -> int:
        return 8 + 4 + self.n_rays + self.n_near * 3 + 4 + self.grid_n ** 2

    @property
    def ray_angles_deg(self) -> np.ndarray:
        """CCW angles in degrees starting from robot forward axis."""
        step = 360.0 / self.n_rays
        return np.arange(self.n_rays) * step

    @property
    def cell_size(self) -> float:
        return self.grid_size_m / self.grid_n


# ── perception input bundle ───────────────────────────────────────────────────

@dataclass
class PerceptionInput:
    """
    YOLO detection result passed to the observation builder.
    Set confidence=0.0 to signal 'target not detected'.
    """
    confidence:      float = 0.0
    bearing_rad:     float = 0.0    # angle to target in robot frame
    dist_est_m:      float = 0.0    # distance estimate (e.g. from bbox size)


# ── observation builder ───────────────────────────────────────────────────────

class NavObsBuilder:
    """
    Builds the 128-dim navigation observation vector from MuJoCo state.

    Parameters
    ----------
    cfg         : NavObsConfig
    model       : compiled MjModel
    robot_qpos_adr  : first qpos index of the robot's freejoint (7 values)
    robot_qvel_adr  : first qvel index of the robot's freejoint (6 values)
    """

    def __init__(
        self,
        cfg:             NavObsConfig,
        model:           mujoco.MjModel,
        robot_qpos_adr:  int = 0,
        robot_qvel_adr:  int = 0,
    ) -> None:
        self.cfg            = cfg
        self.model          = model
        self.robot_qpos_adr = robot_qpos_adr
        self.robot_qvel_adr = robot_qvel_adr

        # resolve body id for ray exclusion
        self._robot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, cfg.robot_body_name
        )

        # pre-compute ray direction unit vectors in body frame (z=0 plane)
        angles = np.deg2rad(cfg.ray_angles_deg)
        self._ray_body = np.stack([
            np.cos(angles),   # x component
            np.sin(angles),   # y component
            np.zeros(cfg.n_rays),
        ], axis=1)             # (N_RAYS, 3)

        self._initial_goal_dist: float = 1.0   # set on each episode reset

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self, initial_goal_dist: float) -> None:
        """Call once per episode with the initial distance to goal."""
        self._initial_goal_dist = max(initial_goal_dist, 0.1)

    def build(
        self,
        data:       mujoco.MjData,
        goal_world: np.ndarray,              # (3,) world-frame goal position
        perception: Optional[PerceptionInput] = None,
    ) -> np.ndarray:
        """
        Compute the full 128-dim observation.

        Parameters
        ----------
        data        : current MjData (after mj_forward)
        goal_world  : goal position in world frame (x, y, z) — z ignored
        perception  : YOLO detection bundle (None → no target visible)

        Returns
        -------
        obs : float32 array of shape (obs_dim,)
        """
        obs = np.zeros(self.cfg.obs_dim, dtype=np.float32)

        robot_pos, yaw, vel_lin, vel_ang = self._robot_state(data)

        # ── Group A: robot state ──────────────────────────────────────────────
        obs[IDX_X]        = float(np.clip(robot_pos[0] / self.cfg.map_half, -1, 1))
        obs[IDX_Y]        = float(np.clip(robot_pos[1] / self.cfg.map_half, -1, 1))
        obs[IDX_COS_YAW]  = float(math.cos(yaw))
        obs[IDX_SIN_YAW]  = float(math.sin(yaw))
        obs[IDX_VX]       = float(np.clip(vel_lin[0] / self.cfg.vel_max,    -1, 1))
        obs[IDX_VY]       = float(np.clip(vel_lin[1] / self.cfg.vel_max,    -1, 1))
        obs[IDX_OMEGA]    = float(np.clip(vel_ang    / self.cfg.ang_vel_max, -1, 1))

        # ── Group B: goal ─────────────────────────────────────────────────────
        goal_dist, goal_bearing = self._goal_polar(robot_pos, yaw, goal_world)
        obs[IDX_PROGRESS]    = float(np.clip(
            1.0 - goal_dist / self._initial_goal_dist, 0.0, 1.0
        ))
        obs[IDX_GOAL_DIST]    = float(np.clip(goal_dist / self.cfg.goal_dist_max, 0, 1))
        obs[IDX_GOAL_COS_BRG] = float(math.cos(goal_bearing))
        obs[IDX_GOAL_SIN_BRG] = float(math.sin(goal_bearing))
        obs[IDX_GOAL_REACHED] = float(goal_dist < 0.25)

        # ── Group C: lidar ring ───────────────────────────────────────────────
        lidar_dists, lidar_angles = self._cast_lidar(data, robot_pos, yaw)
        obs[SL_LIDAR] = np.clip(
            lidar_dists / self.cfg.lidar_max_range, 0.0, 1.0
        ).astype(np.float32)

        # ── Group D: nearest obstacles ────────────────────────────────────────
        obs[SL_NEAR] = self._nearest_obstacles(lidar_dists, lidar_angles)

        # ── Group E: perception ───────────────────────────────────────────────
        obs[SL_PERCEPT] = self._perception_group(perception)

        # ── Group F: egocentric occupancy grid ────────────────────────────────
        obs[SL_OCC] = self._occupancy_grid(lidar_dists, lidar_angles)

        return obs

    # ── internal helpers ──────────────────────────────────────────────────────

    def _robot_state(
        self, data: mujoco.MjData
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Return (pos_xy, yaw, lin_vel_xy_body, ang_vel_z)."""
        qa = self.robot_qpos_adr
        qv = self.robot_qvel_adr
        pos = data.qpos[qa: qa + 3].copy()       # world xyz
        q   = data.qpos[qa + 3: qa + 7].copy()   # wxyz quaternion
        yaw = _quat_to_yaw(q)
        vel_world  = data.qvel[qv: qv + 3].copy()
        # rotate to body frame: vx_body = vx_world*cos(yaw) + vy_world*sin(yaw)
        cy, sy     = math.cos(yaw), math.sin(yaw)
        vel_body   = np.array([
            vel_world[0] * cy + vel_world[1] * sy,
            -vel_world[0] * sy + vel_world[1] * cy,
        ])
        ang_vel_z  = float(data.qvel[qv + 5])    # yaw rate
        return pos, yaw, vel_body, ang_vel_z

    def _goal_polar(
        self,
        robot_pos:  np.ndarray,
        yaw:        float,
        goal_world: np.ndarray,
    ) -> tuple[float, float]:
        """Return (distance_m, bearing_rad) — bearing in robot frame."""
        dx   = goal_world[0] - robot_pos[0]
        dy   = goal_world[1] - robot_pos[1]
        dist = math.hypot(dx, dy)
        world_angle   = math.atan2(dy, dx)
        bearing       = _wrap_angle(world_angle - yaw)
        return dist, bearing

    def _cast_lidar(
        self,
        data:      mujoco.MjData,
        robot_pos: np.ndarray,
        yaw:       float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cast N_RAYS from robot position at lidar_height.

        Returns
        -------
        dists  : (N_RAYS,) distances in metres; LIDAR_MAX_RANGE if no hit
        angles : (N_RAYS,) corresponding angles in robot frame (rad)
        """
        cfg    = self.cfg
        pnt    = np.array([robot_pos[0], robot_pos[1], cfg.lidar_height])
        cy, sy = math.cos(yaw), math.sin(yaw)
        R2d    = np.array([[cy, -sy], [sy, cy]])   # body→world rotation

        dists  = np.full(cfg.n_rays, cfg.lidar_max_range, dtype=np.float64)
        angles = np.deg2rad(cfg.ray_angles_deg)
        geomid = np.array([-1], dtype=np.int32)
        ggroup = (np.array(cfg.geomgroup_mask, dtype=np.uint8)
                  if cfg.geomgroup_mask is not None else None)

        for i, (bx, by, _) in enumerate(self._ray_body):
            # rotate body-frame ray direction to world frame
            wx, wy = R2d @ np.array([bx, by])
            vec    = np.array([wx, wy, 0.0])
            dist   = mujoco.mj_ray(
                self.model, data,
                pnt, vec,
                ggroup, 1,                       # geomgroup filter, include statics
                self._robot_body_id,             # additionally exclude robot body
                geomid,
            )
            if dist >= 0 and dist < cfg.lidar_max_range:
                dists[i] = dist

        return dists, angles

    def _nearest_obstacles(
        self,
        dists:  np.ndarray,
        angles: np.ndarray,
    ) -> np.ndarray:
        """
        Group D — 4 nearest lidar hits packed as (dist_norm, cos_a, sin_a).
        Returns float32 array of shape (N_NEAR * 3,).
        """
        cfg     = self.cfg
        out     = np.zeros(cfg.n_near * 3, dtype=np.float32)
        # default: no obstacle (dist=1.0, angle=0)
        for k in range(cfg.n_near):
            out[k * 3] = 1.0

        hit_mask = dists < cfg.lidar_max_range
        if not hit_mask.any():
            return out

        hit_dists  = dists[hit_mask]
        hit_angles = angles[hit_mask]
        order      = np.argsort(hit_dists)
        for k, idx in enumerate(order[:cfg.n_near]):
            d = float(hit_dists[idx]) / cfg.lidar_max_range
            a = float(hit_angles[idx])
            out[k * 3]     = float(np.clip(d, 0, 1))
            out[k * 3 + 1] = float(math.cos(a))
            out[k * 3 + 2] = float(math.sin(a))

        return out

    def _perception_group(
        self,
        perc: Optional[PerceptionInput],
    ) -> np.ndarray:
        """Group E — 4-dim perception vector."""
        out = np.zeros(4, dtype=np.float32)
        if perc is None or perc.confidence <= 0.0:
            return out
        out[0] = float(np.clip(perc.confidence, 0.0, 1.0))
        out[1] = float(math.cos(perc.bearing_rad))
        out[2] = float(math.sin(perc.bearing_rad))
        out[3] = float(np.clip(perc.dist_est_m / self.cfg.goal_dist_max, 0, 1))
        return out

    def _occupancy_grid(
        self,
        dists:  np.ndarray,
        angles: np.ndarray,
    ) -> np.ndarray:
        """
        Group F — 8×8 egocentric occupancy grid, flattened row-major.

        Row 0 = furthest forward (+x body), row 7 = furthest backward.
        Columns: col 0 = leftmost (−y body), col 7 = rightmost (+y body).
        Cell value: 0.0=free, 1.0=occupied, 0.5=unknown.
        """
        n   = self.cfg.grid_n
        cs  = self.cfg.cell_size      # metres per cell
        half = self.cfg.grid_size_m / 2.0

        # initialise as unknown
        grid = np.full((n, n), 0.5, dtype=np.float32)

        for dist, angle in zip(dists, angles):
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            hit   = dist < self.cfg.lidar_max_range

            # mark cells along the ray as free up to the hit (or max range)
            ray_len = dist if hit else self.cfg.lidar_max_range
            n_steps = max(int(ray_len / (cs * 0.5)), 1)
            for step in range(n_steps):
                t  = step * cs * 0.5
                rx = cos_a * t       # forward (+x body)
                ry = sin_a * t       # left (+y body)
                col = int((ry + half) / cs)
                row = int((half - rx) / cs)   # row 0 = furthest fwd
                if 0 <= row < n and 0 <= col < n:
                    if grid[row, col] != 1.0:   # don't overwrite occupied
                        grid[row, col] = 0.0

            # mark hit cell as occupied
            if hit:
                rx  = cos_a * dist
                ry  = sin_a * dist
                col = int((ry + half) / cs)
                row = int((half - rx) / cs)
                if 0 <= row < n and 0 <= col < n:
                    grid[row, col] = 1.0

        return grid.ravel()


# ── helpers ───────────────────────────────────────────────────────────────────

def _quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw from wxyz quaternion."""
    w, x, y, z = q
    return float(math.atan2(2.0 * (w * z + x * y),
                             1.0 - 2.0 * (y * y + z * z)))


def _wrap_angle(a: float) -> float:
    """Wrap angle to (−π, π]."""
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def make_nav_obs_space() -> "gymnasium.spaces.Box":
    """Return the Gymnasium observation space for the navigation layer."""
    import gymnasium as gym
    return gym.spaces.Box(
        low=-1.0, high=1.0,
        shape=(NAV_OBS_DIM,),
        dtype=np.float32,
    )
