"""
data/ycb/registry.py — YCB object catalog for AutoRobo v1.

Contains the 21 YCB objects most commonly used in manipulation benchmarks
(ContactGraspNet, GraspNet-1B, DexGraspNet, RoboSuite evaluations).

Each entry records:
  - canonical YCB name (matches the S3 archive name)
  - short label for logging / display
  - physical category for scene generation logic
  - approximate mass from YCB official specs (kg)
  - axis-aligned bounding box in metres (x, y, z half-extents)
  - surface friction coefficient (used as MuJoCo geom friction[0])
  - whether the object is suited to single-arm top-down grasps

Sources
───────
  Masses / dims:  YCB Benchmarks official object data sheets
                  (https://www.ycbbenchmarks.org/object-models/)
  Friction:       tuned for MuJoCo 3 steel-on-rubber contact model

Usage
─────
    from data.ycb import REGISTRY

    can  = REGISTRY["002_master_chef_can"]
    cans = REGISTRY.by_category("can")
    easy = REGISTRY.graspable()           # all single-arm graspable objects
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class YCBCategory(str, Enum):
    CAN      = "can"
    BOX      = "box"
    BOTTLE   = "bottle"
    BOWL     = "bowl"
    MUG      = "mug"
    TOOL     = "tool"
    FOOD     = "food"
    CLAMP    = "clamp"
    MARKER   = "marker"
    FOAM     = "foam"


@dataclass(frozen=True)
class YCBObject:
    """
    Physical description of one YCB object.

    Attributes
    ----------
    name        : canonical YCB archive name, e.g. "002_master_chef_can"
    label       : short human-readable name, e.g. "master_chef_can"
    category    : YCBCategory enum value
    mass_kg     : approximate mass from official YCB data sheets
    half_extents: (x, y, z) bounding-box half-extents in metres.
                  The full bounding box is 2 × half_extents on each axis.
    friction    : MuJoCo geom friction[0] (sliding friction coefficient)
    graspable   : True → object suited to single-arm top-down parallel-jaw grasp
    """
    name:         str
    label:        str
    category:     YCBCategory
    mass_kg:      float
    half_extents: tuple[float, float, float]   # (x, y, z) metres
    friction:     float
    graspable:    bool = True

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def size(self) -> tuple[float, float, float]:
        """Full bounding-box dimensions (2 × half_extents)."""
        return (
            2.0 * self.half_extents[0],
            2.0 * self.half_extents[1],
            2.0 * self.half_extents[2],
        )

    @property
    def resting_height(self) -> float:
        """z offset so the object base sits exactly on z = 0."""
        return self.half_extents[2]

    @property
    def download_name(self) -> str:
        """Stem of the S3 archive, e.g. '002_master_chef_can_google_16k'."""
        return f"{self.name}_google_16k"

    def __str__(self) -> str:
        return (
            f"YCBObject({self.name!r}, mass={self.mass_kg:.3f} kg, "
            f"dims={tuple(round(v*100, 1) for v in self.size)} cm, "
            f"friction={self.friction})"
        )


# ── catalog ───────────────────────────────────────────────────────────────────

_CATALOG: list[YCBObject] = [
    # ── cans ─────────────────────────────────────────────────────────────────
    YCBObject("002_master_chef_can",   "master_chef_can",   YCBCategory.CAN,
              mass_kg=0.414, half_extents=(0.0424, 0.0424, 0.0714), friction=0.80),
    YCBObject("005_tomato_soup_can",   "tomato_soup_can",   YCBCategory.CAN,
              mass_kg=0.349, half_extents=(0.0330, 0.0330, 0.0510), friction=0.80),
    YCBObject("007_tuna_fish_can",     "tuna_fish_can",     YCBCategory.CAN,
              mass_kg=0.171, half_extents=(0.0425, 0.0425, 0.0170), friction=0.80),
    YCBObject("010_potted_meat_can",   "potted_meat_can",   YCBCategory.CAN,
              mass_kg=0.370, half_extents=(0.0498, 0.0352, 0.0330), friction=0.75),

    # ── boxes ────────────────────────────────────────────────────────────────
    YCBObject("003_cracker_box",       "cracker_box",       YCBCategory.BOX,
              mass_kg=0.411, half_extents=(0.1070, 0.0720, 0.0385), friction=0.60),
    YCBObject("004_sugar_box",         "sugar_box",         YCBCategory.BOX,
              mass_kg=0.514, half_extents=(0.0455, 0.0920, 0.0455), friction=0.60),
    YCBObject("008_pudding_box",       "pudding_box",       YCBCategory.BOX,
              mass_kg=0.187, half_extents=(0.0855, 0.0360, 0.0155), friction=0.55),
    YCBObject("009_gelatin_box",       "gelatin_box",       YCBCategory.BOX,
              mass_kg=0.097, half_extents=(0.0755, 0.0280, 0.0165), friction=0.55),
    YCBObject("036_wood_block",        "wood_block",        YCBCategory.BOX,
              mass_kg=0.729, half_extents=(0.0850, 0.0850, 0.0450), friction=0.65),
    YCBObject("061_foam_brick",        "foam_brick",        YCBCategory.FOAM,
              mass_kg=0.070, half_extents=(0.0525, 0.0375, 0.0265), friction=0.90),

    # ── bottles ──────────────────────────────────────────────────────────────
    YCBObject("006_mustard_bottle",    "mustard_bottle",    YCBCategory.BOTTLE,
              mass_kg=0.603, half_extents=(0.0450, 0.0385, 0.0975), friction=0.70),
    YCBObject("021_bleach_cleanser",   "bleach_cleanser",   YCBCategory.BOTTLE,
              mass_kg=1.131, half_extents=(0.0500, 0.0685, 0.1250), friction=0.65),
    YCBObject("019_pitcher_base",      "pitcher_base",      YCBCategory.BOTTLE,
              mass_kg=0.178, half_extents=(0.0685, 0.0685, 0.0975), friction=0.70,
              graspable=False),

    # ── bowls / mugs ─────────────────────────────────────────────────────────
    YCBObject("024_bowl",              "bowl",              YCBCategory.BOWL,
              mass_kg=0.147, half_extents=(0.0795, 0.0795, 0.0305), friction=0.65,
              graspable=False),
    YCBObject("025_mug",               "mug",               YCBCategory.MUG,
              mass_kg=0.118, half_extents=(0.0400, 0.0600, 0.0415), friction=0.70),

    # ── food ─────────────────────────────────────────────────────────────────
    YCBObject("011_banana",            "banana",            YCBCategory.FOOD,
              mass_kg=0.066, half_extents=(0.0885, 0.0330, 0.0215), friction=0.60),

    # ── tools ────────────────────────────────────────────────────────────────
    YCBObject("035_power_drill",       "power_drill",       YCBCategory.TOOL,
              mass_kg=0.895, half_extents=(0.0855, 0.0490, 0.1095), friction=0.75,
              graspable=False),
    YCBObject("037_scissors",          "scissors",          YCBCategory.TOOL,
              mass_kg=0.082, half_extents=(0.0415, 0.0200, 0.0790), friction=0.50,
              graspable=False),
    YCBObject("040_large_marker",      "large_marker",      YCBCategory.MARKER,
              mass_kg=0.016, half_extents=(0.0075, 0.0075, 0.0650), friction=0.55),

    # ── clamps ───────────────────────────────────────────────────────────────
    YCBObject("051_large_clamp",       "large_clamp",       YCBCategory.CLAMP,
              mass_kg=0.125, half_extents=(0.0435, 0.0230, 0.0740), friction=0.60,
              graspable=False),
    YCBObject("052_extra_large_clamp", "extra_large_clamp", YCBCategory.CLAMP,
              mass_kg=0.202, half_extents=(0.0610, 0.0230, 0.0935), friction=0.60,
              graspable=False),
]


# ── registry class ────────────────────────────────────────────────────────────

class YCBRegistry:
    """
    Queryable catalog of YCB objects.

    Indexing
    ────────
        reg = YCBRegistry()
        obj = reg["002_master_chef_can"]   # by canonical name
        obj = reg["master_chef_can"]       # by short label

    Iteration
    ─────────
        for obj in reg:
            print(obj.name, obj.mass_kg)

    Filtering
    ─────────
        reg.by_category("can")             # list[YCBObject]
        reg.graspable()                    # list[YCBObject] — top-down graspable
        reg.by_mass_range(0.0, 0.5)        # list[YCBObject] — light objects
        reg.names()                        # list[str] — all canonical names
    """

    def __init__(self, objects: list[YCBObject] | None = None) -> None:
        src = objects if objects is not None else _CATALOG
        self._by_name:  dict[str, YCBObject] = {o.name:  o for o in src}
        self._by_label: dict[str, YCBObject] = {o.label: o for o in src}

    # ── lookup ────────────────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> YCBObject:
        if key in self._by_name:
            return self._by_name[key]
        if key in self._by_label:
            return self._by_label[key]
        raise KeyError(
            f"YCB object {key!r} not found. "
            f"Use a canonical name (e.g. '002_master_chef_can') or "
            f"a short label (e.g. 'master_chef_can')."
        )

    def __contains__(self, key: str) -> bool:
        return key in self._by_name or key in self._by_label

    def __iter__(self) -> Iterator[YCBObject]:
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def get(self, key: str, default: YCBObject | None = None) -> YCBObject | None:
        try:
            return self[key]
        except KeyError:
            return default

    # ── filtering ─────────────────────────────────────────────────────────────

    def by_category(self, category: str | YCBCategory) -> list[YCBObject]:
        """Return all objects in a category (accepts string or YCBCategory)."""
        cat = YCBCategory(category) if isinstance(category, str) else category
        return [o for o in self if o.category == cat]

    def graspable(self) -> list[YCBObject]:
        """Return all objects suited to single-arm parallel-jaw grasps."""
        return [o for o in self if o.graspable]

    def by_mass_range(self, lo: float, hi: float) -> list[YCBObject]:
        """Return objects whose mass_kg is within [lo, hi]."""
        return [o for o in self if lo <= o.mass_kg <= hi]

    def names(self) -> list[str]:
        """Return all canonical YCB names, sorted."""
        return sorted(self._by_name)

    def labels(self) -> list[str]:
        """Return all short labels, sorted."""
        return sorted(self._by_label)

    def categories(self) -> list[YCBCategory]:
        """Return the set of categories present in this registry."""
        return sorted({o.category for o in self}, key=lambda c: c.value)

    # ── sampling ──────────────────────────────────────────────────────────────

    def sample(
        self,
        rng,
        n:          int  = 1,
        graspable:  bool = False,
        category:   str | YCBCategory | None = None,
    ) -> list[YCBObject]:
        """
        Sample n objects without replacement.

        Parameters
        ----------
        rng       : numpy.random.Generator
        n         : number of objects to draw
        graspable : if True restrict to graspable objects
        category  : if given restrict to that category
        """
        pool = list(self)
        if graspable:
            pool = [o for o in pool if o.graspable]
        if category is not None:
            cat  = YCBCategory(category) if isinstance(category, str) else category
            pool = [o for o in pool if o.category == cat]
        if n > len(pool):
            raise ValueError(
                f"Requested {n} objects but only {len(pool)} match the filters."
            )
        indices = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in indices]

    # ── display ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a multi-line summary table."""
        lines = [
            f"{'Name':<30} {'Label':<22} {'Cat':<8} {'Mass':>6} {'Dims (cm)':>22} {'Fric':>5} {'Grasp'}",
            "─" * 100,
        ]
        for o in sorted(self, key=lambda x: x.name):
            dims = tuple(round(v * 100, 1) for v in o.size)
            lines.append(
                f"{o.name:<30} {o.label:<22} {o.category.value:<8} "
                f"{o.mass_kg:>5.3f}  "
                f"{str(dims):>22}  "
                f"{o.friction:>5.2f}  {'✓' if o.graspable else '✗'}"
            )
        lines.append(f"\n{len(self)} objects total")
        return "\n".join(lines)


# ── module-level singleton ────────────────────────────────────────────────────

REGISTRY = YCBRegistry()
