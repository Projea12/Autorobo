"""
env/navigation_env.py — Gymnasium navigation environment (MuJoCo).

Task
────
Drive from a random spawn to a goal position.  The episode ends when the
robot reaches within ``goal_radius`` of the goal (success), gets too close
to an obstacle (collision), or exhausts the step budget (truncation).

Architecture
────────────
  Observation  : 128-dim NavObsBuilder vector  (see nav_obs.py)
  Action       : 2-dim  ActionProcessor output (see nav_action.py)
                   [0] v_linear  ∈ [−1, 1]  — forward / reverse
                   [1] v_angular ∈ [−1, 1]  — left / right yaw rate
  Reward       : NavRewardFunction with 6 components (see nav_reward.py)
  Termination  : success | collision
  Truncation   : step count ≥ max_steps

Physics
───────
  Self-contained MjSpec model — no robot.xml dependency.
  Box chassis + two velocity-controlled drive wheels + two caster balls.
  Implicit integrator at 5 ms, kv=20.  dt_env = n_substeps × 5 ms.
  Robot geoms in group 1 (excluded from lidar); env geoms in group 0.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.nav_obs import (
    NavObsBuilder, NavObsConfig, PerceptionInput,
    make_nav_obs_space, SL_OCC,
)
from env.nav_action import (
    ActionConfig, ActionProcessor,
    LIN_VEL_MAX, ANG_VEL_MAX, WHEEL_RADIUS, WHEELBASE,
)
from env.nav_reward import RewardConfig, NavRewardFunction
from env.episode_reset import SpawnConfig, GoalConfig


# ── constants ─────────────────────────────────────────────────────────────────

_OBS_CFG_DEFAULT = NavObsConfig(
    robot_body_name="nav_base",
    lidar_height=0.15,
    geomgroup_mask=(1, 0, 0, 0, 0, 0),
)

# dt_env = n_substeps × model timestep
_MODEL_DT: float = 0.005    # 5 ms
_N_SUBSTEPS_DEFAULT: int = 2  # → 10 ms per env step


# ══════════════════════════════════════════════════════════════════════════════

class NavigationEnv(gym.Env):
    """
    MuJoCo point-navigation environment for mobile robot RL.

    Parameters
    ----------
    render_mode  : "rgb_array" or None
    max_steps    : hard episode timeout in env steps
    n_substeps   : MuJoCo steps per env step  (dt_env = n × 5 ms)
    obs_cfg      : NavObsConfig — sensor geometry and normalisation
    action_cfg   : ActionConfig — velocity limits, smoothing, rate limits
    reward_cfg   : RewardConfig — reward component weights and thresholds
    spawn_cfg    : SpawnConfig  — robot spawn distribution
    goal_cfg     : GoalConfig   — goal placement distribution
    seed         : initial RNG seed (overridable in reset())
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        render_mode: Optional[str]  = None,
        max_steps:   int            = 500,
        n_substeps:  int            = _N_SUBSTEPS_DEFAULT,
        obs_cfg:     NavObsConfig   = _OBS_CFG_DEFAULT,
        action_cfg:  ActionConfig   = ActionConfig(),
        reward_cfg:  RewardConfig   = RewardConfig(),
        spawn_cfg:   SpawnConfig    = SpawnConfig(),
        goal_cfg:    GoalConfig     = GoalConfig(),
        seed:        Optional[int]  = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._max_steps  = max_steps
        self._n_substeps = n_substeps
        self._obs_cfg    = obs_cfg
        self._spawn_cfg  = spawn_cfg
        self._goal_cfg   = goal_cfg

        dt_env = n_substeps * _MODEL_DT

        self._model = self._build_model()
        self._data  = mujoco.MjData(self._model)
        self._renderer:  Optional[mujoco.Renderer] = None
        self._rgbd_cam:  Optional[Any]             = None   # RGBDCamera, lazy

        self._cache_indices()

        self.observation_space = make_nav_obs_space()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._obs_builder   = NavObsBuilder(
            cfg            = obs_cfg,
            model          = self._model,
            robot_qpos_adr = self._base_qadr,
            robot_qvel_adr = self._base_vadr,
        )
        self._action_proc   = ActionProcessor(cfg=action_cfg, dt_env=dt_env)
        self._reward_fn     = NavRewardFunction(cfg=reward_cfg)

        # Per-episode state — filled in reset()
        self._step_count: int        = 0
        self._prev_dist:  float      = 1.0
        self._goal_world: np.ndarray = np.zeros(3, dtype=np.float32)
        self._last_reward_info: Optional[Any] = None
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
        self._goal_world     = self._place_goal(self._rng, robot_xy, robot_yaw)

        mujoco.mj_forward(self._model, self._data)

        self._step_count = 0
        self._prev_dist  = self._goal_dist()

        # Reset sub-systems
        self._obs_builder.reset(initial_goal_dist=self._prev_dist)
        self._action_proc.reset()
        self._reward_fn.reset(robot_xy=robot_xy)

        obs = self._get_obs()
        return obs, {"step": 0, "dist_to_goal": self._prev_dist,
                     "goal": self._goal_world[:2].tolist(),
                     "success": False, "collision": False}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Action processing → physical wheel commands
        phys = self._action_proc.process(np.asarray(action, dtype=np.float32))
        self._data.ctrl[0] = phys.ctrl_left
        self._data.ctrl[1] = phys.ctrl_right

        # 2. Physics simulation
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1

        # 3. Observation
        obs = self._get_obs()

        # 4. Reward computation
        curr_dist = self._goal_dist()
        d_lidar   = self._forward_arc_min_range()
        ri = self._reward_fn.step(
            robot_xy    = self._robot_xy(),
            d_prev      = self._prev_dist,
            d_curr      = curr_dist,
            d_lidar_min = d_lidar,
            occ_grid    = obs[SL_OCC],
            perception  = None,   # YOLO not wired in navigation-only mode
        )
        self._prev_dist       = curr_dist
        self._last_reward_info = ri

        truncated = self._step_count >= self._max_steps
        return obs, ri.total, ri.terminated, truncated, self._build_info(ri)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data)
        return self._renderer.render().copy()

    def capture_rgbd(self, cfg=None):
        """
        Render one RGB-D frame from the onboard 'nav_cam' camera.

        Parameters
        ----------
        cfg : RGBDConfig or None — uses default 640×480 60° fov if None

        Returns
        -------
        RGBDFrame with rgb (H,W,3 uint8) and depth (H,W float32 metres).
        """
        from perception.rgbd_camera import RGBDCamera, RGBDConfig
        if self._rgbd_cam is None:
            self._rgbd_cam = RGBDCamera(cfg or RGBDConfig(), self._model)
        mujoco.mj_forward(self._model, self._data)
        return self._rgbd_cam.capture(self._data, step=self._step_count)

    def close(self) -> None:
        if self._rgbd_cam is not None:
            self._rgbd_cam.close()
            self._rgbd_cam = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── public accessors ───────────────────────────────────────────────────────

    @property
    def reward_function(self) -> NavRewardFunction:
        """Access the underlying reward function (e.g. to read visited cells)."""
        return self._reward_fn

    @property
    def action_processor(self) -> ActionProcessor:
        """Access the underlying action processor."""
        return self._action_proc

    @property
    def goal_world(self) -> np.ndarray:
        """Current episode goal position in world frame (3,)."""
        return self._goal_world.copy()

    # ── model construction ─────────────────────────────────────────────────────

    @staticmethod
    def _build_model() -> mujoco.MjModel:
        """
        Minimal differential-drive model built with MjSpec (no robot.xml).

        Geom groups:
          0 — environment (floor, future walls/obstacles) — visible to lidar
          1 — robot chassis, wheels, casters              — skipped by lidar
        """
        spec = mujoco.MjSpec()
        spec.option.timestep   = _MODEL_DT
        spec.option.gravity    = [0, 0, -9.81]
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

        # Floor
        floor        = spec.worldbody.add_geom()
        floor.name   = "floor"
        floor.type   = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size   = [0.0, 0.0, 0.1]
        floor.rgba   = [0.75, 0.75, 0.75, 1.0]
        floor.group  = 0

        # Robot base body (freejoint)
        base       = spec.worldbody.add_body()
        base.name  = "nav_base"
        base.pos   = [0.0, 0.0, 0.12]   # z = wheel_r + chassis_h/2

        fj         = base.add_freejoint()
        fj.name    = "nav_root"

        chassis       = base.add_geom()
        chassis.name  = "nav_chassis"
        chassis.type  = mujoco.mjtGeom.mjGEOM_BOX
        chassis.size  = [0.15, 0.10, 0.04]
        chassis.mass  = 6.0
        chassis.rgba  = [0.2, 0.4, 0.8, 1.0]
        chassis.group = 1

        for name, pos in [("caster_l", [0.13,  0.08, -0.08]),
                           ("caster_r", [0.13, -0.08, -0.08])]:
            c          = base.add_geom()
            c.name     = name
            c.type     = mujoco.mjtGeom.mjGEOM_SPHERE
            c.size     = [0.03, 0, 0]
            c.pos      = pos
            c.mass     = 0.05
            c.group    = 1
            c.friction = [0.01, 0.005, 0.005]

        # Drive wheels
        for side, y_sign in [("l", 1), ("r", -1)]:
            wb       = base.add_body()
            wb.name  = f"wheel_{side}"
            wb.pos   = [0.0, y_sign * (WHEELBASE / 2), -0.04]

            wj         = wb.add_joint()
            wj.name    = f"wheel_{side}_joint"
            wj.type    = mujoco.mjtJoint.mjJNT_HINGE
            wj.axis    = [0, 1, 0]
            wj.damping = 0.1

            wg          = wb.add_geom()
            wg.name     = f"wheel_{side}_geom"
            wg.type     = mujoco.mjtGeom.mjGEOM_CYLINDER
            wg.size     = [WHEEL_RADIUS, 0.025, 0]
            wg.mass     = 0.4
            wg.rgba     = [0.1, 0.1, 0.1, 1.0]
            wg.group    = 1
            wg.friction = [1.5, 0.01, 0.001]

        # Velocity actuators (kv=20, no gear — ctrl in rad/s at wheel joint)
        max_rad_s = LIN_VEL_MAX / WHEEL_RADIUS + (WHEELBASE / 2) * ANG_VEL_MAX / WHEEL_RADIUS
        for side in ("l", "r"):
            act           = spec.add_actuator()
            act.name      = f"drive_{side}"
            act.trntype   = mujoco.mjtTrn.mjTRN_JOINT
            act.target    = f"wheel_{side}_joint"
            act.dyntype   = mujoco.mjtDyn.mjDYN_NONE
            act.gaintype  = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype  = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm   = [20.0] + [0.0] * 9
            act.biasprm   = [0.0, 0.0, -20.0] + [0.0] * 7
            act.ctrlrange = [-max_rad_s, max_rad_s]

        # RGBD camera — front-centre of chassis, 10° downward tilt
        # xyaxes = [cam_X_in_body, cam_Y_in_body]
        #   cam X (right in image)  = body -Y
        #   cam Y (up in image)     = body Z tilted 10° toward forward
        _tilt = math.radians(10.0)
        cam          = base.add_camera()
        cam.name     = "nav_cam"
        cam.pos      = [0.15, 0.0, 0.06]
        cam.xyaxes   = [0.0, -1.0, 0.0,
                        -math.sin(_tilt), 0.0, math.cos(_tilt)]
        cam.fovy     = 60.0

        # Home keyframe
        kf       = spec.add_key()
        kf.name  = "home"
        kf.qpos  = [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return spec.compile()

    def _cache_indices(self) -> None:
        m = self._model

        root_jid        = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "nav_root")
        self._base_qadr = int(m.jnt_qposadr[root_jid])
        self._base_vadr = int(m.jnt_dofadr[root_jid])

        self._robot_body_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "nav_base"
        )

    # ── episode reset helpers ──────────────────────────────────────────────────

    def _spawn_robot(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, float]:
        cfg = self._spawn_cfg
        qa  = self._base_qadr
        x   = float(rng.uniform(*cfg.x_range))
        y   = float(rng.uniform(*cfg.y_range))
        yaw = float(rng.uniform(*cfg.yaw_range))
        self._data.qpos[qa]     = x
        self._data.qpos[qa + 1] = y
        self._data.qpos[qa + 2] = 0.12
        self._data.qpos[qa + 3] = math.cos(yaw / 2.0)
        self._data.qpos[qa + 4] = 0.0
        self._data.qpos[qa + 5] = 0.0
        self._data.qpos[qa + 6] = math.sin(yaw / 2.0)
        self._data.qvel[self._base_vadr : self._base_vadr + 6] = 0.0
        return np.array([x, y]), yaw

    def _place_goal(
        self,
        rng: np.random.Generator,
        robot_xy: np.ndarray,
        robot_yaw: float,
    ) -> np.ndarray:
        cfg = self._goal_cfg
        if cfg.mode == "relative":
            fwd = float(rng.uniform(*cfg.fwd_range))
            lat = float(rng.uniform(*cfg.lat_range))
            c, s = math.cos(robot_yaw), math.sin(robot_yaw)
            gx   = robot_xy[0] + fwd * c - lat * s
            gy   = robot_xy[1] + fwd * s + lat * c
        else:
            gx = float(rng.uniform(*cfg.x_range))
            gy = float(rng.uniform(*cfg.y_range))
        return np.array([gx, gy, cfg.z], dtype=np.float32)

    # ── observation ────────────────────────────────────────────────────────────

    def _get_obs(
        self, perception: Optional[PerceptionInput] = None
    ) -> np.ndarray:
        return self._obs_builder.build(
            self._data, self._goal_world, perception=perception
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    def _goal_dist(self) -> float:
        qa = self._base_qadr
        return math.hypot(
            self._goal_world[0] - float(self._data.qpos[qa]),
            self._goal_world[1] - float(self._data.qpos[qa + 1]),
        )

    def _robot_xy(self) -> np.ndarray:
        qa = self._base_qadr
        return np.array([
            float(self._data.qpos[qa]),
            float(self._data.qpos[qa + 1]),
        ])

    def _forward_arc_min_range(self) -> float:
        """Cast 7 rays ±90° forward; return minimum hit distance (m)."""
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

    def _build_info(self, ri: Any) -> dict[str, Any]:
        return {
            "success":        ri.success,
            "collision":      ri.collision_flag,
            "step":           self._step_count,
            "dist_to_goal":   self._prev_dist,
            "goal":           self._goal_world[:2].tolist(),
            "n_visited_cells": ri.n_visited,
            "reward_approach": ri.approach,
            "reward_obstacle": ri.obstacle,
            "reward_explore":  ri.explore,
            "reward_uncertainty": ri.uncertainty,
            "reward_time":     ri.time,
        }
