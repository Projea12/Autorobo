"""
env/domain_rand.py — Per-episode domain randomisation for AutoRobo v1.

Randomises visual, physical, and dynamic properties of the MuJoCo model
in-place each episode.  All changes are applied directly to MjModel
numpy arrays — no recompilation required.

Randomised properties
──────────────────────
  Lighting      headlight ambient/diffuse, sun/fill direction + intensity,
                fill-light active probability
  Floor         surface colour (grey tone), sliding friction
  Walls         surface colour (grey range)
  Obstacles     RGBA hue shift, sliding friction per geom
  Pickable objs RGBA hue shift, sliding friction, mass, size (±frac)
  Wheels        sliding friction (traction variation)
  Arm joints    DOF damping ±frac of nominal value

Calling pattern
───────────────
    rand = DomainRandomizer(model)          # once — caches all IDs
    ...
    rand.randomize(model, data, rng)        # every episode reset

After randomize():
  • Call mujoco.mj_forward(model, data) to propagate size/inertia changes.
  • The model is shared — if you have multiple data objects, call randomize()
    before creating a new episode rather than mid-rollout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DomainRandConfig:
    """
    Controls which properties are randomised and their ranges.

    All ranges are [lo, hi] — uniform distribution unless stated otherwise.
    Set a flag to False to skip that category entirely.
    """

    # ── lighting ──────────────────────────────────────────────────────────────
    randomize_lighting:      bool  = True
    headlight_ambient_lo:    float = 0.05   # per-channel, same for R/G/B
    headlight_ambient_hi:    float = 0.45
    headlight_diffuse_lo:    float = 0.25
    headlight_diffuse_hi:    float = 0.90
    sun_diffuse_lo:          float = 0.40
    sun_diffuse_hi:          float = 1.00
    sun_dir_noise_std:       float = 0.20   # Gaussian noise on normalised sun dir
    fill_active_prob:        float = 0.70   # probability fill light is enabled
    fill_diffuse_lo:         float = 0.10
    fill_diffuse_hi:         float = 0.50

    # ── floor ─────────────────────────────────────────────────────────────────
    randomize_floor_color:    bool  = True
    floor_grey_lo:            float = 0.30
    floor_grey_hi:            float = 0.70
    randomize_floor_friction: bool  = True
    floor_friction_lo:        float = 0.50
    floor_friction_hi:        float = 1.50

    # ── walls ─────────────────────────────────────────────────────────────────
    randomize_wall_color:     bool  = True
    wall_grey_lo:             float = 0.20
    wall_grey_hi:             float = 0.55

    # ── obstacles ─────────────────────────────────────────────────────────────
    randomize_obstacle_color:    bool  = True
    obs_hue_shift:               float = 0.30   # max channel-wise shift ±
    randomize_obstacle_friction: bool  = True
    obs_friction_lo:             float = 0.30
    obs_friction_hi:             float = 1.80

    # ── pickable objects ──────────────────────────────────────────────────────
    randomize_pickable_color:    bool  = True
    pick_hue_shift:              float = 0.25
    randomize_pickable_friction: bool  = True
    pick_friction_lo:            float = 0.80
    pick_friction_hi:            float = 2.50
    randomize_pickable_mass:     bool  = True
    pick_mass_lo:                float = 0.08   # kg
    pick_mass_hi:                float = 0.50
    randomize_pickable_size:     bool  = True
    pick_size_frac:              float = 0.20   # ±20 % of nominal half-extents

    # ── wheels ────────────────────────────────────────────────────────────────
    randomize_wheel_friction:    bool  = True
    wheel_friction_lo:           float = 0.80
    wheel_friction_hi:           float = 2.20

    # ── arm joint damping ─────────────────────────────────────────────────────
    randomize_joint_damping:     bool  = True
    joint_damping_frac:          float = 0.20   # ±20 % of nominal damping


DEFAULT_CONFIG: DomainRandConfig = DomainRandConfig()


# ── main class ────────────────────────────────────────────────────────────────

class DomainRandomizer:
    """
    Applies per-episode domain randomisation to a compiled MuJoCo model.

    Initialise once with the compiled model; call randomize() each episode.

    Parameters
    ----------
    model  : compiled MjModel (from WorldBuilder.build or ManipulationEnv)
    config : DomainRandConfig controlling which properties are randomised
    """

    # Arm DOF indices in the (robot + world) compiled model.
    # Robot DOFs: 0-5 base freejoint, 6-7 wheels, 8-13 arm, 14-15 fingers.
    _ARM_DOF_SLICE = slice(8, 14)

    def __init__(
        self,
        model:  mujoco.MjModel,
        config: DomainRandConfig = DEFAULT_CONFIG,
    ) -> None:
        self._cfg = config

        # ── geom ID caches ────────────────────────────────────────────────────
        self._floor_gid:  int        = _find_geom(model, "world_floor")
        self._wall_gids:  list[int]  = _find_geoms_prefix(model, "wall_")
        self._obs_gids:   list[int]  = _find_geoms_prefix(model, "obstacle_")
        self._pick_gids:  list[int]  = _find_geoms_prefix(model, "pickable_")
        self._wheel_gids: list[int]  = [
            g for g in [
                _find_geom(model, "wheel_left_geom"),
                _find_geom(model, "wheel_right_geom"),
            ] if g >= 0
        ]

        # ── body ID caches (for mass / inertia) ───────────────────────────────
        self._pick_bids: list[int] = [
            int(model.geom_bodyid[g]) for g in self._pick_gids
        ]

        # ── light ID caches ───────────────────────────────────────────────────
        self._sun_lid:  int = _find_light(model, "sun")
        self._fill_lid: int = _find_light(model, "fill")

        # ── nominal values (for ±frac randomisation) ──────────────────────────
        self._nom_pick_size = (
            model.geom_size[self._pick_gids].copy()
            if self._pick_gids else np.zeros((0, 3))
        )
        self._nom_dof_damping = model.dof_damping[self._ARM_DOF_SLICE].copy()

        # ── nominal obstacle colours (base for hue shift) ─────────────────────
        self._nom_obs_rgba = (
            model.geom_rgba[self._obs_gids].copy()
            if self._obs_gids else np.zeros((0, 4))
        )
        self._nom_pick_rgba = (
            model.geom_rgba[self._pick_gids].copy()
            if self._pick_gids else np.zeros((0, 4))
        )

        # ── nominal sun direction ─────────────────────────────────────────────
        if self._sun_lid >= 0:
            raw = model.light_dir[self._sun_lid].copy()
            norm = np.linalg.norm(raw)
            self._nom_sun_dir = raw / norm if norm > 1e-8 else np.array([0., 0., -1.])
        else:
            self._nom_sun_dir = np.array([0., 0., -1.])

    # ── public API ────────────────────────────────────────────────────────────

    def randomize(
        self,
        model: mujoco.MjModel,
        data:  mujoco.MjData,
        rng:   np.random.Generator,
    ) -> None:
        """
        Randomise all enabled model properties in-place.

        Call before mj_forward() each episode.  Mutates model arrays directly;
        calls mj_setConst() if geom sizes are changed.

        Parameters
        ----------
        model : MjModel to mutate (same instance as passed to __init__)
        data  : MjData needed by mj_setConst()
        rng   : NumPy Generator (caller controls seed for reproducibility)
        """
        cfg = self._cfg
        need_set_const = False

        if cfg.randomize_lighting:
            self._rand_lighting(model, rng)

        if cfg.randomize_floor_color or cfg.randomize_floor_friction:
            self._rand_floor(model, rng)

        if cfg.randomize_wall_color:
            self._rand_walls(model, rng)

        if cfg.randomize_obstacle_color or cfg.randomize_obstacle_friction:
            self._rand_obstacles(model, rng)

        if (cfg.randomize_pickable_color or cfg.randomize_pickable_friction
                or cfg.randomize_pickable_mass):
            self._rand_pickable(model, rng)

        if cfg.randomize_pickable_size and self._pick_gids:
            self._rand_pickable_size(model, rng)
            need_set_const = True

        if cfg.randomize_wheel_friction and self._wheel_gids:
            self._rand_wheel_friction(model, rng)

        if cfg.randomize_joint_damping:
            self._rand_joint_damping(model, rng)

        if need_set_const:
            mujoco.mj_setConst(model, data)

    # ── snapshot / restore ────────────────────────────────────────────────────

    def snapshot(self, model: mujoco.MjModel) -> dict:
        """
        Capture current randomisable model state for later restore().

        Useful when you need to compare before/after randomisation in tests.
        """
        snap: dict = {}
        if self._floor_gid >= 0:
            snap["floor_rgba"]    = model.geom_rgba[self._floor_gid].copy()
            snap["floor_fric"]    = model.geom_friction[self._floor_gid].copy()
        if self._obs_gids:
            snap["obs_rgba"]      = model.geom_rgba[self._obs_gids].copy()
            snap["obs_fric"]      = model.geom_friction[self._obs_gids].copy()
        if self._pick_gids:
            snap["pick_rgba"]     = model.geom_rgba[self._pick_gids].copy()
            snap["pick_fric"]     = model.geom_friction[self._pick_gids].copy()
            snap["pick_size"]     = model.geom_size[self._pick_gids].copy()
            snap["pick_mass"]     = np.array([model.body_mass[b] for b in self._pick_bids])
        snap["dof_damping"]       = model.dof_damping[self._ARM_DOF_SLICE].copy()
        snap["hl_ambient"]        = np.array(model.vis.headlight.ambient[:])
        snap["hl_diffuse"]        = np.array(model.vis.headlight.diffuse[:])
        return snap

    # ── lighting ──────────────────────────────────────────────────────────────

    def _rand_lighting(
        self,
        model: mujoco.MjModel,
        rng:   np.random.Generator,
    ) -> None:
        cfg = self._cfg

        # Headlight — single value applied to all 3 channels
        ambient = float(rng.uniform(cfg.headlight_ambient_lo, cfg.headlight_ambient_hi))
        diffuse = float(rng.uniform(cfg.headlight_diffuse_lo, cfg.headlight_diffuse_hi))
        for i in range(3):
            model.vis.headlight.ambient[i] = ambient
            model.vis.headlight.diffuse[i] = diffuse

        # Sun — directional light with perturbed direction and randomised intensity
        if self._sun_lid >= 0:
            # Add Gaussian noise to the nominal direction, then renormalise
            noisy = self._nom_sun_dir + rng.normal(0.0, cfg.sun_dir_noise_std, 3)
            norm  = np.linalg.norm(noisy)
            model.light_dir[self._sun_lid] = noisy / max(norm, 1e-8)

            sun_d = float(rng.uniform(cfg.sun_diffuse_lo, cfg.sun_diffuse_hi))
            model.light_diffuse[self._sun_lid] = [sun_d, sun_d, sun_d]

        # Fill — toggle on/off + vary intensity
        if self._fill_lid >= 0:
            active = int(rng.random() < cfg.fill_active_prob)
            model.light_active[self._fill_lid] = active
            if active:
                fill_d = float(rng.uniform(cfg.fill_diffuse_lo, cfg.fill_diffuse_hi))
                model.light_diffuse[self._fill_lid] = [fill_d, fill_d, fill_d]

    # ── floor ─────────────────────────────────────────────────────────────────

    def _rand_floor(
        self,
        model: mujoco.MjModel,
        rng:   np.random.Generator,
    ) -> None:
        if self._floor_gid < 0:
            return
        cfg = self._cfg
        if cfg.randomize_floor_color:
            grey = float(rng.uniform(cfg.floor_grey_lo, cfg.floor_grey_hi))
            model.geom_rgba[self._floor_gid, :3] = grey
        if cfg.randomize_floor_friction:
            model.geom_friction[self._floor_gid, 0] = float(
                rng.uniform(cfg.floor_friction_lo, cfg.floor_friction_hi)
            )

    # ── walls ─────────────────────────────────────────────────────────────────

    def _rand_walls(
        self,
        model: mujoco.MjModel,
        rng:   np.random.Generator,
    ) -> None:
        cfg = self._cfg
        for gid in self._wall_gids:
            grey = float(rng.uniform(cfg.wall_grey_lo, cfg.wall_grey_hi))
            model.geom_rgba[gid, :3] = grey

    # ── obstacles ─────────────────────────────────────────────────────────────

    def _rand_obstacles(
        self,
        model: mujoco.MjModel,
        rng:   np.random.Generator,
    ) -> None:
        cfg = self._cfg
        for i, gid in enumerate(self._obs_gids):
            if cfg.randomize_obstacle_color:
                base = self._nom_obs_rgba[i, :3]
                shift = rng.uniform(-cfg.obs_hue_shift, cfg.obs_hue_shift, 3)
                model.geom_rgba[gid, :3] = np.clip(base + shift, 0.05, 1.0)
            if cfg.randomize_obstacle_friction:
                model.geom_friction[gid, 0] = float(
                    rng.uniform(cfg.obs_friction_lo, cfg.obs_friction_hi)
                )

    # ── pickable objects ──────────────────────────────────────────────────────

    def _rand_pickable(
        self,
        model: mujoco.MjModel,
        rng:   np.random.Generator,
    ) -> None:
        cfg = self._cfg
        for i, (gid, bid) in enumerate(zip(self._pick_gids, self._pick_bids)):
            if cfg.randomize_pickable_color:
                base  = self._nom_pick_rgba[i, :3]
                shift = rng.uniform(-cfg.pick_hue_shift, cfg.pick_hue_shift, 3)
                model.geom_rgba[gid, :3] = np.clip(base + shift, 0.05, 1.0)
            if cfg.randomize_pickable_friction:
                model.geom_friction[gid, 0] = float(
                    rng.uniform(cfg.pick_friction_lo, cfg.pick_friction_hi)
                )
            if cfg.randomize_pickable_mass:
                mass = float(rng.uniform(cfg.pick_mass_lo, cfg.pick_mass_hi))
                model.body_mass[bid] = mass
                # Recompute inertia for uniform box of current half-extents
                h = model.geom_size[gid]   # [hx, hy, hz]
                model.body_inertia[bid] = _box_inertia(mass, h)

    def _rand_pickable_size(
        self,
        model: mujoco.MjModel,
        rng:   np.random.Generator,
    ) -> None:
        """Scale pickable geom half-extents by a random factor ∈ [1−frac, 1+frac]."""
        frac = self._cfg.pick_size_frac
        for i, (gid, bid) in enumerate(zip(self._pick_gids, self._pick_bids)):
            scale = float(rng.uniform(1.0 - frac, 1.0 + frac))
            new_half = self._nom_pick_size[i] * scale
            model.geom_size[gid] = new_half
            # Update inertia to match new size + current (possibly just-set) mass
            mass = float(model.body_mass[bid])
            model.body_inertia[bid] = _box_inertia(mass, new_half)

    # ── wheels ────────────────────────────────────────────────────────────────

    def _rand_wheel_friction(
        self,
        model: mujoco.MjModel,
        rng:   np.random.Generator,
    ) -> None:
        cfg = self._cfg
        for gid in self._wheel_gids:
            model.geom_friction[gid, 0] = float(
                rng.uniform(cfg.wheel_friction_lo, cfg.wheel_friction_hi)
            )

    # ── joint damping ─────────────────────────────────────────────────────────

    def _rand_joint_damping(
        self,
        model: mujoco.MjModel,
        rng:   np.random.Generator,
    ) -> None:
        frac = self._cfg.joint_damping_frac
        for i, nom in enumerate(self._nom_dof_damping):
            if nom == 0.0:
                continue   # skip undamped DOFs
            lo = nom * (1.0 - frac)
            hi = nom * (1.0 + frac)
            model.dof_damping[self._ARM_DOF_SLICE.start + i] = float(
                rng.uniform(lo, hi)
            )


# ── helper utilities ──────────────────────────────────────────────────────────

def _find_geom(model: mujoco.MjModel, name: str) -> int:
    """Return geom ID for the given name, or −1 if not found."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _find_geoms_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    """Return IDs of all geoms whose name starts with prefix, in ID order."""
    found = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.startswith(prefix):
            found.append(gid)
    return found


def _find_light(model: mujoco.MjModel, name: str) -> int:
    """Return light ID for the given name, or −1 if not found."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name)


def _box_inertia(mass: float, half: np.ndarray) -> np.ndarray:
    """
    Diagonal inertia tensor of a solid uniform box (body frame).

    Parameters
    ----------
    mass : kg
    half : (3,) half-extents [hx, hy, hz] in metres

    Returns
    -------
    (3,) array  [Ixx, Iyy, Izz]
    """
    hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
    Ixx = mass * (hy**2 + hz**2) / 3.0
    Iyy = mass * (hx**2 + hz**2) / 3.0
    Izz = mass * (hx**2 + hy**2) / 3.0
    return np.array([Ixx, Iyy, Izz])
