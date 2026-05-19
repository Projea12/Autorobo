"""
data/synth/scene.py — MuJoCo scene builder for synthetic data generation.

Builds a single compiled model whose object pool is fixed at construction
time.  Each call to reset() mutates data.qpos in-place to scatter a fresh
random layout — no recompilation needed per image.

Object pool strategy
────────────────────
  • Every YCB object in the registry gets one slot in the pool.
  • Geometry: cans → cylinder, everything else → box.
  • If a processed collision mesh exists in processed_dir the pipeline can
    later swap the geom; for now primitives match the registry half_extents.
  • Inactive slots are hidden at z = -100 m (same pool trick as world.py).

Pool qpos layout (per object, freejoint)
────────────────────────────────────────
  [x, y, z,  qw, qx, qy, qz]   (7 floats)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import mujoco
import numpy as np

from data.ycb.registry import REGISTRY, YCBCategory, YCBObject


# ── slot descriptor ───────────────────────────────────────────────────────────

class ObjectSlot(NamedTuple):
    """Describes one object slot baked into the compiled model."""
    name:        str             # canonical YCB name
    qadr:        int             # first qpos index of this slot's freejoint
    vadr:        int             # first qvel index
    half_extents: np.ndarray    # (3,) metres — used for bbox projection
    category:    YCBCategory


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SceneConfig:
    """
    Parameters controlling the synthetic scene.

    image_w / image_h   : rendered image resolution
    fovy                : vertical field of view in degrees (camera intrinsic)
    floor_size          : half-width of the visible floor plane in metres
    object_spread       : objects placed uniformly within ±spread in X and Y
    object_names        : subset of REGISTRY names to use; None → all 21
    min_objects         : min active objects per image (inclusive)
    max_objects         : max active objects per image (inclusive)
    enable_headlight    : add ambient headlight to the model
    """
    image_w:       int              = 640
    image_h:       int              = 480
    fovy:          float            = 45.0
    floor_size:    float            = 1.5
    object_spread: float            = 0.70    # m
    object_names:  tuple | None     = None    # None → all registry objects
    min_objects:   int              = 2
    max_objects:   int              = 5
    enable_headlight: bool          = True


# ── scene class ───────────────────────────────────────────────────────────────

class SynthScene:
    """
    Compiled MuJoCo model with a randomisable YCB object pool.

    Parameters
    ----------
    cfg          : SceneConfig
    processed_dir: path to YCBPreprocessor output (optional — used later for
                   real mesh injection; currently unused in primitive mode)
    """

    _HIDE_Z = -100.0   # z for inactive slots

    def __init__(
        self,
        cfg:           SceneConfig       = SceneConfig(),
        processed_dir: Optional[str]     = None,
    ) -> None:
        self.cfg           = cfg
        self.processed_dir = processed_dir

        names = list(cfg.object_names) if cfg.object_names else REGISTRY.names()
        self._ycb_objects  = [REGISTRY[n] for n in names]

        self.model, self.slots = self._build()
        self._data_template = mujoco.MjData(self.model)

    # ── public API ────────────────────────────────────────────────────────────

    def make_data(self) -> mujoco.MjData:
        """Return a fresh MjData for this model (cheap copy of template)."""
        return mujoco.MjData(self.model)

    def reset(
        self,
        data: mujoco.MjData,
        rng:  np.random.Generator,
    ) -> list[ObjectSlot]:
        """
        Scatter a random subset of objects on the floor; hide the rest.

        Parameters
        ----------
        data : MjData to mutate in-place
        rng  : seeded NumPy Generator

        Returns
        -------
        List of active ObjectSlot instances (in order placed).
        """
        n_active = int(rng.integers(self.cfg.min_objects,
                                    self.cfg.max_objects + 1))
        n_active = min(n_active, len(self.slots))

        chosen_idx = rng.choice(len(self.slots), size=n_active, replace=False)
        chosen_idx.sort()

        active: list[ObjectSlot] = []
        placed_xy: list[np.ndarray] = []

        for slot_i in chosen_idx:
            slot = self.slots[slot_i]
            xy   = self._sample_position(rng, placed_xy,
                                         min_sep=max(slot.half_extents[:2]) * 2 + 0.05)
            placed_xy.append(xy)
            z    = slot.half_extents[2]   # resting height on floor
            yaw  = float(rng.uniform(-math.pi, math.pi))
            self._place(data, slot, xy[0], xy[1], z, yaw)
            active.append(slot)

        # hide all unused slots
        for i, slot in enumerate(self.slots):
            if i not in chosen_idx:
                self._hide(data, slot)

        mujoco.mj_forward(self.model, data)
        return active

    def object_pos(self, data: mujoco.MjData, slot: ObjectSlot) -> np.ndarray:
        """Return current world-frame position of a slot as (3,) array."""
        return data.qpos[slot.qadr: slot.qadr + 3].copy()

    def object_quat(self, data: mujoco.MjData, slot: ObjectSlot) -> np.ndarray:
        """Return current wxyz quaternion of a slot as (4,) array."""
        return data.qpos[slot.qadr + 3: slot.qadr + 7].copy()

    # ── model build ───────────────────────────────────────────────────────────

    def _build(self) -> tuple[mujoco.MjModel, list[ObjectSlot]]:
        spec = mujoco.MjSpec()
        spec.option.gravity = np.array([0.0, 0.0, -9.81])

        # headlight
        if self.cfg.enable_headlight:
            spec.visual.headlight.ambient  = np.array([0.4, 0.4, 0.4])
            spec.visual.headlight.diffuse  = np.array([0.8, 0.8, 0.8])
            spec.visual.headlight.specular = np.array([0.3, 0.3, 0.3])

        # directional sun
        sun = spec.worldbody.add_light()
        sun.name      = "sun"
        sun.pos       = np.array([0.0, 0.0, 3.0])
        sun.dir       = np.array([0.0, -0.3, -1.0])
        sun.type      = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        sun.diffuse   = np.array([0.7, 0.7, 0.7])
        sun.specular  = np.array([0.2, 0.2, 0.2])
        sun.castshadow = True

        # floor
        floor = spec.worldbody.add_geom()
        floor.name    = "synth_floor"
        floor.type    = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size    = np.array([self.cfg.floor_size,
                                   self.cfg.floor_size, 0.05])
        floor.rgba    = np.array([0.75, 0.75, 0.75, 1.0])
        floor.friction = np.array([1.0, 0.005, 0.0001])

        # object pool — primitive geoms with correct YCB dimensions
        slot_specs: list[tuple[str, str, np.ndarray, YCBCategory]] = []
        for obj in self._ycb_objects:
            body_name  = f"pool_{obj.name}"
            body       = spec.worldbody.add_body()
            body.name  = body_name
            body.pos   = np.array([0.0, 0.0, self._HIDE_Z])

            joint       = body.add_freejoint()
            joint.name  = f"{body_name}_jnt"

            geom        = body.add_geom()
            geom.name   = f"{body_name}_col"
            geom.rgba   = _object_rgba(obj)
            geom.friction = np.array([obj.friction, 0.005, 0.0001])
            geom.mass   = obj.mass_kg

            hx, hy, hz = obj.half_extents
            if obj.category == YCBCategory.CAN:
                r = (hx + hy) / 2.0        # average horizontal radius
                geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                geom.size = np.array([r, hz, 0.0])
            else:
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size = np.array([hx, hy, hz])

            slot_specs.append((body_name, joint.name,
                               np.array(obj.half_extents, dtype=np.float64),
                               obj.category))

        model = spec.compile()

        # resolve qpos / qvel addresses
        slots: list[ObjectSlot] = []
        for (body_name, jnt_name, half, cat), obj in zip(slot_specs, self._ycb_objects):
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            qadr   = int(model.jnt_qposadr[jnt_id])
            vadr   = int(model.jnt_dofadr[jnt_id])
            slots.append(ObjectSlot(
                name         = obj.name,
                qadr         = qadr,
                vadr         = vadr,
                half_extents = half,
                category     = cat,
            ))

        return model, slots

    # ── per-frame helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _place(
        data: mujoco.MjData,
        slot: ObjectSlot,
        x: float, y: float, z: float,
        yaw: float,
    ) -> None:
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        data.qpos[slot.qadr + 0] = x
        data.qpos[slot.qadr + 1] = y
        data.qpos[slot.qadr + 2] = z
        data.qpos[slot.qadr + 3] = qw
        data.qpos[slot.qadr + 4] = 0.0
        data.qpos[slot.qadr + 5] = 0.0
        data.qpos[slot.qadr + 6] = qz
        data.qvel[slot.vadr: slot.vadr + 6] = 0.0

    @staticmethod
    def _hide(data: mujoco.MjData, slot: ObjectSlot) -> None:
        data.qpos[slot.qadr + 2] = SynthScene._HIDE_Z
        data.qpos[slot.qadr + 3] = 1.0   # identity quaternion
        data.qpos[slot.qadr + 4] = 0.0
        data.qpos[slot.qadr + 5] = 0.0
        data.qpos[slot.qadr + 6] = 0.0
        data.qvel[slot.vadr: slot.vadr + 6] = 0.0

    def _sample_position(
        self,
        rng:      np.random.Generator,
        placed:   list[np.ndarray],
        min_sep:  float,
        max_tries: int = 50,
    ) -> np.ndarray:
        s = self.cfg.object_spread
        for _ in range(max_tries):
            xy = rng.uniform(-s, s, size=2)
            if all(np.linalg.norm(xy - p) >= min_sep for p in placed):
                return xy
        return rng.uniform(-s, s, size=2)   # fallback


# ── helpers ───────────────────────────────────────────────────────────────────

# Fixed per-category colours — consistent appearance per class, varied enough
# that a YOLO network can learn to distinguish them even without real textures.
_CATEGORY_RGBA: dict[YCBCategory, np.ndarray] = {
    YCBCategory.CAN:    np.array([0.60, 0.60, 0.62, 1.0]),  # metal silver
    YCBCategory.BOX:    np.array([0.85, 0.78, 0.55, 1.0]),  # cardboard tan
    YCBCategory.BOTTLE: np.array([0.90, 0.95, 0.80, 1.0]),  # plastic yellow-green
    YCBCategory.BOWL:   np.array([0.95, 0.95, 0.95, 1.0]),  # white ceramic
    YCBCategory.MUG:    np.array([0.70, 0.30, 0.20, 1.0]),  # terracotta
    YCBCategory.TOOL:   np.array([0.20, 0.20, 0.22, 1.0]),  # dark grey
    YCBCategory.FOOD:   np.array([0.95, 0.85, 0.30, 1.0]),  # banana yellow
    YCBCategory.CLAMP:  np.array([0.80, 0.20, 0.10, 1.0]),  # red clamp
    YCBCategory.MARKER: np.array([0.10, 0.10, 0.80, 1.0]),  # blue marker
    YCBCategory.FOAM:   np.array([0.10, 0.60, 0.90, 1.0]),  # blue foam
}


def _object_rgba(obj: YCBObject) -> np.ndarray:
    base = _CATEGORY_RGBA.get(obj.category, np.array([0.6, 0.6, 0.6, 1.0]))
    # Per-object slight hue perturbation using name hash (deterministic)
    h = abs(hash(obj.name)) % 1000 / 1000.0
    jitter = (h - 0.5) * 0.12
    return np.clip(base + np.array([jitter, -jitter * 0.5, jitter * 0.3, 0]), 0, 1)
