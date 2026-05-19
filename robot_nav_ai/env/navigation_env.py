"""
env/navigation_env.py — Gymnasium navigation environment (MuJoCo).

Task
────
Drive from a random spawn to a goal position.  The episode ends when the robot
base reaches within ``goal_radius`` of the goal, gets too close to an obstacle
(collision), or the step budget runs out.

Architecture
────────────
  Observation  : 128-dim NavObsBuilder vector (see nav_obs.py)
  Action       : 2-dim float32 Box in [−1, 1]
                   [0] v_linear  → both wheels at ±NAV_WHEEL_VEL m/s
                   [1] v_angular → differential wheel offset
  Reward       : dense approach + obstacle-proximity penalty + sparse bonuses
  Termination  : success (dist < goal_radius) | collision (lidar < collision_r)
  Truncation   : step count >= max_steps

Physics
───────
  Self-contained MjSpec model — no robot.xml dependency.
  A box chassis (30×20×8 cm) with two cylindrical drive wheels and two front
  caster spheres.  Velocity actuators with kv=20 on wheel joints.
  dt_env = n_substeps × 5 ms  (default 2 × 5 ms = 10 ms per env step).
  Robot body geoms are moved to geom group 1 so lidar skips them.

Differential-drive action mapping
──────────────────────────────────
  v_l = (v_lin − v_ang) × NAV_WHEEL_VEL      [left wheel velocity, m/s]
  v_r = (v_lin + v_ang) × NAV_WHEEL_VEL      [right wheel velocity, m/s]
  ctrl[0] = v_l / WHEEL_RADIUS   (rad/s applied to wheel joint velocity)
  ctrl[1] = v_r / WHEEL_RADIUS

Reward shaping
──────────────
  R_APPROACH   : r_approach × (prev_dist − curr_dist)      [dense, ~2 per m]
  R_OBSTACLE   : −r_obstacle × (1 − d_min/danger_r)        [dense, proximity]
  R_TIME       : −r_time_step                               [dense, −0.01/step]
  R_SUCCESS    : +r_success                                 [sparse, +10]
  R_COLLISION  : −r_collision                               [sparse, −5]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.nav_obs import (
    NavObsBuilder, NavObsConfig, PerceptionInput,
    make_nav_obs_space,
)
from env.episode_reset import SpawnConfig, GoalConfig

# ── model constants ───────────────────────────────────────────────────────────

NAV_WHEEL_VEL:  float = 1.5   # m/s — max linear speed
WHEEL_RADIUS:   float = 0.08  # m
WHEELBASE:      float = 0.30  # m — distance between wheel centres

# ── reward configuration ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class NavRewardConfig:
    """Weights and thresholds for navigation reward shaping."""
    approach:    float = 2.0    # per metre of progress toward goal (dense)
    obstacle:    float = 0.5    # obstacle-proximity penalty multiplier (dense)
    time_step:   float = 0.01   # per-step time penalty (dense)
    success:     float = 10.0   # goal-reached bonus (sparse)
    collision:   float = 5.0    # collision penalty (sparse)
    danger_r:    float = 0.25   # m — lidar hits below this trigger penalty
    collision_r: float = 0.12   # m — lidar hits below this terminate episode
    goal_radius: float = 0.25   # m — success threshold


# ── action constants ──────────────────────────────────────────────────────────

_ACT_DIM: int = 2
_OBS_CFG_DEFAULT = NavObsConfig(
    robot_body_name="nav_base",
    lidar_height=0.15,
    geomgroup_mask=(1, 0, 0, 0, 0, 0),  # only hit environment geoms (group 0)
)


# ══════════════════════════════════════════════════════════════════════════════

class NavigationEnv(gym.Env):
    """
    MuJoCo-based point-navigation environment for mobile robot RL.

    Parameters
    ----------
    render_mode  : "rgb_array" or None
    max_steps    : hard episode timeout (in env steps)
    n_substeps   : physics steps per env step
    obs_cfg      : NavObsConfig — sensor geometry and normalization
    reward_cfg   : NavRewardConfig — reward shaping weights
    spawn_cfg    : SpawnConfig — robot spawn distribution
    goal_cfg     : GoalConfig — goal placement distribution
    seed         : initial RNG seed (overridable per-reset)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode: Optional[str]   = None,
        max_steps:   int             = 500,
        n_substeps:  int             = 2,
        obs_cfg:     NavObsConfig    = _OBS_CFG_DEFAULT,
        reward_cfg:  NavRewardConfig = NavRewardConfig(),
        spawn_cfg:   SpawnConfig     = SpawnConfig(),
        goal_cfg:    GoalConfig      = GoalConfig(),
        seed:        Optional[int]   = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._max_steps  = max_steps
        self._n_substeps = n_substeps
        self._obs_cfg    = obs_cfg
        self._rew_cfg    = reward_cfg
        self._spawn_cfg  = spawn_cfg
        self._goal_cfg   = goal_cfg

        self._model = self._build_model()
        self._data  = mujoco.MjData(self._model)
        self._renderer: Optional[mujoco.Renderer] = None

        self._cache_indices()

        self.observation_space = make_nav_obs_space()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_ACT_DIM,), dtype=np.float32
        )

        self._obs_builder = NavObsBuilder(
            cfg            = obs_cfg,
            model          = self._model,
            robot_qpos_adr = self._base_qadr,
            robot_qvel_adr = self._base_vadr,
        )

        # Per-episode state — filled in reset()
        self._step_count: int        = 0
        self._prev_dist:  float      = 1.0
        self._goal_world: np.ndarray = np.zeros(3, dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)

        robot_xy, robot_yaw = self._spawn_robot(self._rng)
        self._goal_world = self._place_goal(self._rng, robot_xy, robot_yaw)

        mujoco.mj_forward(self._model, self._data)

        self._step_count = 0
        self._prev_dist  = self._goal_dist()
        self._obs_builder.reset(initial_goal_dist=self._prev_dist)

        obs = self._get_obs()
        return obs, self._build_info(success=False, collision=False)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._apply_action(np.asarray(action, dtype=np.float64))

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        obs = self._get_obs()

        curr_dist = self._goal_dist()
        reward    = self._compute_reward(curr_dist)
        self._prev_dist = curr_dist

        success   = curr_dist < self._rew_cfg.goal_radius
        collision = self._is_collision()
        terminated = success or collision
        truncated  = self._step_count >= self._max_steps

        if success:
            reward += self._rew_cfg.success
        elif collision:
            reward -= self._rew_cfg.collision

        return obs, float(reward), terminated, truncated, self._build_info(success, collision)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── model construction ─────────────────────────────────────────────────────

    @staticmethod
    def _build_model() -> mujoco.MjModel:
        """
        Build a minimal differential-drive model from scratch (no robot.xml).

        Geom groups:
          group 0 — environment (floor, obstacles) — visible to lidar
          group 1 — robot chassis, wheels, casters — excluded from lidar
        """
        spec = mujoco.MjSpec()
        spec.option.timestep   = 0.005    # 5 ms — stable with kv=20
        spec.option.gravity    = [0, 0, -9.81]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

        # ── floor ──────────────────────────────────────────────────────────
        floor        = spec.worldbody.add_geom()
        floor.name   = "floor"
        floor.type   = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size   = [0.0, 0.0, 0.1]
        floor.rgba   = [0.8, 0.8, 0.8, 1.0]
        floor.group  = 0

        # ── robot base ─────────────────────────────────────────────────────
        base        = spec.worldbody.add_body()
        base.name   = "nav_base"
        base.pos    = [0.0, 0.0, 0.12]   # z = wheel_r + chassis_h/2

        fj          = base.add_freejoint()
        fj.name     = "nav_root"

        chassis      = base.add_geom()
        chassis.name = "nav_chassis"
        chassis.type = mujoco.mjtGeom.mjGEOM_BOX
        chassis.size = [0.15, 0.10, 0.04]
        chassis.mass = 6.0
        chassis.rgba = [0.2, 0.4, 0.8, 1.0]
        chassis.group = 1                 # robot — excluded from lidar

        # Front caster balls (fixed, low friction)
        for name, pos in [("caster_l", [0.13,  0.08, -0.08]),
                           ("caster_r", [0.13, -0.08, -0.08])]:
            c           = base.add_geom()
            c.name      = name
            c.type      = mujoco.mjtGeom.mjGEOM_SPHERE
            c.size      = [0.03, 0, 0]
            c.pos       = pos
            c.mass      = 0.05
            c.group     = 1
            c.friction  = [0.01, 0.005, 0.005]

        # ── drive wheels ───────────────────────────────────────────────────
        for side, y_sign in [("l", 1), ("r", -1)]:
            wb = base.add_body()
            wb.name  = f"wheel_{side}"
            wb.pos   = [0.0, y_sign * (WHEELBASE / 2), -0.04]

            wj       = wb.add_joint()
            wj.name  = f"wheel_{side}_joint"
            wj.type  = mujoco.mjtJoint.mjJNT_HINGE
            wj.axis  = [0, 1, 0]
            wj.damping = 0.1

            wg       = wb.add_geom()
            wg.name  = f"wheel_{side}_geom"
            wg.type  = mujoco.mjtGeom.mjGEOM_CYLINDER
            wg.size  = [WHEEL_RADIUS, 0.025, 0]
            wg.mass  = 0.4
            wg.rgba  = [0.1, 0.1, 0.1, 1.0]
            wg.group = 1                  # robot — excluded from lidar
            wg.friction = [1.5, 0.01, 0.001]

        # ── wheel velocity actuators ───────────────────────────────────────
        for side in ("l", "r"):
            act          = spec.add_actuator()
            act.name     = f"drive_{side}"
            act.trntype  = mujoco.mjtTrn.mjTRN_JOINT
            act.target   = f"wheel_{side}_joint"
            act.dyntype  = mujoco.mjtDyn.mjDYN_NONE
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm  = [20.0] + [0.0] * 9
            act.biasprm  = [0.0, 0.0, -20.0] + [0.0] * 7
            max_rad_s    = NAV_WHEEL_VEL / WHEEL_RADIUS
            act.ctrlrange = [-max_rad_s, max_rad_s]

        # ── default keyframe (robot at origin, upright) ────────────────────
        kf        = spec.add_key()
        kf.name   = "home"
        # qpos: base freejoint (7) + wheel_l (1) + wheel_r (1) = 9
        kf.qpos   = [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0,  # base
                     0.0,                                    # wheel_l
                     0.0]                                    # wheel_r

        return spec.compile()

    def _cache_indices(self) -> None:
        m = self._model

        root_jid        = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "nav_root")
        self._base_qadr = int(m.jnt_qposadr[root_jid])
        self._base_vadr = int(m.jnt_dofadr[root_jid])

        self._robot_body_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "nav_base"
        )

        wl_jid       = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wheel_l_joint")
        wr_jid       = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wheel_r_joint")
        self._wl_qadr = int(m.jnt_qposadr[wl_jid])
        self._wr_qadr = int(m.jnt_qposadr[wr_jid])

    # ── episode reset helpers ──────────────────────────────────────────────────

    def _spawn_robot(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        """Place robot at a random collision-free spawn within SpawnConfig ranges."""
        cfg = self._spawn_cfg
        qa  = self._base_qadr

        for _ in range(cfg.max_tries):
            x   = float(rng.uniform(*cfg.x_range))
            y   = float(rng.uniform(*cfg.y_range))
            yaw = float(rng.uniform(*cfg.yaw_range))
            # No obstacles in base model, so any spawn is fine
            self._data.qpos[qa]     = x
            self._data.qpos[qa + 1] = y
            self._data.qpos[qa + 2] = 0.12         # fixed floor height
            self._data.qpos[qa + 3] = math.cos(yaw / 2.0)  # qw
            self._data.qpos[qa + 4] = 0.0
            self._data.qpos[qa + 5] = 0.0
            self._data.qpos[qa + 6] = math.sin(yaw / 2.0)  # qz
            self._data.qvel[self._base_vadr : self._base_vadr + 6] = 0.0
            return np.array([x, y]), yaw

        return np.array([0.0, 0.0]), 0.0

    def _place_goal(
        self,
        rng:       np.random.Generator,
        robot_xy:  np.ndarray,
        robot_yaw: float,
    ) -> np.ndarray:
        """Sample goal position per GoalConfig and return world-frame xyz."""
        cfg = self._goal_cfg
        if cfg.mode == "relative":
            fwd = float(rng.uniform(*cfg.fwd_range))
            lat = float(rng.uniform(*cfg.lat_range))
            c, s = math.cos(robot_yaw), math.sin(robot_yaw)
            gx   = robot_xy[0] + fwd * c - lat * s
            gy   = robot_xy[1] + fwd * s + lat * c
        else:   # "world"
            gx = float(rng.uniform(*cfg.x_range))
            gy = float(rng.uniform(*cfg.y_range))
        return np.array([gx, gy, cfg.z], dtype=np.float32)

    # ── observation ────────────────────────────────────────────────────────────

    def _get_obs(
        self,
        perception: Optional[PerceptionInput] = None,
    ) -> np.ndarray:
        return self._obs_builder.build(
            self._data,
            self._goal_world,
            perception=perception,
        )

    # ── reward ─────────────────────────────────────────────────────────────────

    def _compute_reward(self, curr_dist: float) -> float:
        cfg    = self._rew_cfg
        reward = cfg.approach * (self._prev_dist - curr_dist)
        reward -= cfg.time_step

        d_min = self._forward_arc_min_range()
        if d_min < cfg.danger_r:
            reward -= cfg.obstacle * (1.0 - d_min / cfg.danger_r)

        return reward

    # ── action ─────────────────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> None:
        """
        Map [v_lin, v_ang] ∈ [−1,1]² → wheel joint velocity targets (rad/s).

          v_left_ms  = (v_lin − v_ang) × NAV_WHEEL_VEL
          v_right_ms = (v_lin + v_ang) × NAV_WHEEL_VEL
          ctrl[i]    = v_i / WHEEL_RADIUS          (rad/s)
        """
        v_lin = float(np.clip(action[0], -1.0, 1.0))
        v_ang = float(np.clip(action[1], -1.0, 1.0))
        vl    = (v_lin - v_ang) * NAV_WHEEL_VEL
        vr    = (v_lin + v_ang) * NAV_WHEEL_VEL
        self._data.ctrl[0] = vl / WHEEL_RADIUS
        self._data.ctrl[1] = vr / WHEEL_RADIUS

    # ── helpers ────────────────────────────────────────────────────────────────

    def _goal_dist(self) -> float:
        qa = self._base_qadr
        bx = float(self._data.qpos[qa])
        by = float(self._data.qpos[qa + 1])
        return math.hypot(self._goal_world[0] - bx, self._goal_world[1] - by)

    def _forward_arc_min_range(self) -> float:
        """Cast 7 rays in the forward ±90° arc; return minimum hit distance."""
        cfg    = self._obs_cfg
        qa     = self._base_qadr
        pos    = self._data.qpos[qa : qa + 3]
        q      = self._data.qpos[qa + 3 : qa + 7]
        yaw    = math.atan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
        )
        pnt    = np.array([pos[0], pos[1], cfg.lidar_height])
        geomid = np.array([-1], dtype=np.int32)
        min_d  = cfg.lidar_max_range
        ggroup = (np.array(cfg.geomgroup_mask, dtype=np.uint8)
                  if cfg.geomgroup_mask is not None else None)

        for deg in range(-90, 91, 30):
            a   = yaw + math.radians(deg)
            vec = np.array([math.cos(a), math.sin(a), 0.0])
            d   = mujoco.mj_ray(
                self._model, self._data,
                pnt, vec, ggroup, 1,
                self._robot_body_id, geomid,
            )
            if 0.0 <= d < min_d:
                min_d = d

        return min_d

    def _is_collision(self) -> bool:
        """Lidar-based collision: any forward-arc ray hit within collision_r."""
        return self._forward_arc_min_range() < self._rew_cfg.collision_r

    def _build_info(self, success: bool, collision: bool) -> dict[str, Any]:
        return {
            "success":      success,
            "collision":    collision,
            "step":         self._step_count,
            "dist_to_goal": self._prev_dist,
            "goal":         self._goal_world[:2].tolist(),
        }
