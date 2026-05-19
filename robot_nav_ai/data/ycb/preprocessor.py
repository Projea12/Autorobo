"""
data/ycb/preprocessor.py — YCB mesh preprocessing for MuJoCo.

Converts raw YCB archives (downloaded by YCBDownloader) into assets the
world builder and ManipulationEnv can load directly:

  <out_dir>/<name>/
    collision.stl          ← convex hull of nontextured.stl (physics)
    visual.obj             ← copy of textured.obj (rendering)
    object.xml             ← standalone MJCF body snippet
    meta.json              ← dimensions, mass, friction, hull vertex count

Pipeline per object
───────────────────
  1. Parse vertices from nontextured.stl (binary or ASCII)
  2. Compute convex hull via scipy.spatial.ConvexHull
  3. Write hull as binary STL (collision.stl)
  4. Copy textured.obj + .mtl to visual.obj (rendering)
  5. Generate object.xml — a <body> with:
       - <freejoint/> (for pick-and-place episodes)
       - collision geom: type=mesh, mesh=collision.stl
       - visual geom:   type=mesh, mesh=visual.obj, contype=0
  6. Write meta.json with AABB half-extents, mass, friction

No external dependencies beyond scipy (already needed for physics).
If scipy is unavailable the hull step is skipped and a box proxy is used.

Usage
─────
    from data.ycb import YCBDownloader, YCBPreprocessor

    dl  = YCBDownloader(dest_dir="data/ycb/raw")
    pre = YCBPreprocessor(raw_dir="data/ycb/raw", out_dir="data/ycb/processed")

    result = pre.process("002_master_chef_can")
    print(result.mjcf_path, result.meta)

    pre.process_all()    # all downloaded objects
"""

from __future__ import annotations

import json
import shutil
import struct
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ProcessedObject:
    """Output of YCBPreprocessor.process()."""
    name:          str
    success:       bool
    out_dir:       Path | None  = None
    mjcf_path:     Path | None  = None
    collision_stl: Path | None  = None
    visual_obj:    Path | None  = None
    meta:          dict         = field(default_factory=dict)
    error:         str | None   = None
    elapsed_s:     float        = 0.0

    def __str__(self) -> str:
        if not self.success:
            return f"[fail] {self.name}: {self.error}"
        hv = self.meta.get("hull_vertices", "box")
        return (
            f"[ok]   {self.name}  hull_verts={hv}  "
            f"dims={self.meta.get('size_cm')}cm  "
            f"({self.elapsed_s:.2f}s)"
        )


# ── preprocessor ─────────────────────────────────────────────────────────────

class YCBPreprocessor:
    """
    Converts raw YCB mesh archives to MuJoCo-ready assets.

    Parameters
    ----------
    raw_dir : directory written by YCBDownloader (contains <name>/ sub-dirs)
    out_dir : destination for processed assets
    """

    def __init__(self, raw_dir: str | Path, out_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, name: str) -> ProcessedObject:
        """
        Process one YCB object.

        Parameters
        ----------
        name : canonical YCB name, e.g. "002_master_chef_can"

        Returns
        -------
        ProcessedObject with all output paths and metadata.
        """
        from .registry import REGISTRY

        t0      = time.perf_counter()
        raw_obj = self.raw_dir / name / "google_16k"
        out_obj = self.out_dir / name
        out_obj.mkdir(parents=True, exist_ok=True)

        stl_src = raw_obj / "nontextured.stl"
        obj_src = raw_obj / "textured.obj"
        mtl_src = raw_obj / "textured.mtl"

        # ── verify raw files ──────────────────────────────────────────────────
        if not stl_src.exists():
            return ProcessedObject(
                name=name, success=False,
                error=f"nontextured.stl not found: {stl_src}. "
                      "Run YCBDownloader.download(name) first.",
            )

        # ── step 1: parse vertices ────────────────────────────────────────────
        try:
            verts = self._read_stl_vertices(stl_src)
        except Exception as e:
            return ProcessedObject(name=name, success=False,
                                   error=f"STL parse error: {e}")

        # ── step 2: compute AABB ──────────────────────────────────────────────
        lo, hi        = verts.min(axis=0), verts.max(axis=0)
        centre        = (lo + hi) / 2.0
        half_extents  = (hi - lo) / 2.0

        # ── step 3: convex hull (scipy) or AABB fallback ──────────────────────
        hull_verts, hull_faces, n_hull_verts = self._convex_hull(verts)
        use_box = hull_verts is None

        # ── step 4: write collision.stl ───────────────────────────────────────
        coll_stl = out_obj / "collision.stl"
        if use_box:
            self._write_box_stl(coll_stl, centre, half_extents)
        else:
            self._write_stl(coll_stl, hull_verts, hull_faces)

        # ── step 5: copy visual mesh ──────────────────────────────────────────
        vis_obj = out_obj / "visual.obj"
        if obj_src.exists():
            shutil.copy2(obj_src, vis_obj)
            if mtl_src.exists():
                shutil.copy2(mtl_src, out_obj / "visual.mtl")
                self._fix_mtl_ref(vis_obj)
        else:
            # Fallback: write a minimal box OBJ for visual
            self._write_box_obj(vis_obj, centre, half_extents)

        # ── step 6: look up registry properties ───────────────────────────────
        reg_obj = REGISTRY.get(name)
        mass    = reg_obj.mass_kg if reg_obj else 0.300
        fric    = reg_obj.friction if reg_obj else 0.70

        # Use registry half_extents if available (more accurate than mesh AABB)
        if reg_obj is not None:
            half_extents = np.array(reg_obj.half_extents)

        # ── step 7: generate MJCF body snippet ───────────────────────────────
        mjcf_path = out_obj / "object.xml"
        self._write_mjcf(
            mjcf_path, name,
            coll_stl  = coll_stl,
            vis_obj   = vis_obj,
            mass      = mass,
            friction  = fric,
            half      = half_extents,
        )

        # ── step 8: write meta.json ───────────────────────────────────────────
        meta = {
            "name":         name,
            "mass_kg":      mass,
            "friction":     fric,
            "half_extents": [float(v) for v in half_extents],
            "size_cm":      [round(float(v) * 200, 1) for v in half_extents],
            "hull_vertices": n_hull_verts if not use_box else "box_proxy",
            "collision_stl": str(coll_stl),
            "visual_obj":    str(vis_obj),
            "mjcf":          str(mjcf_path),
        }
        (out_obj / "meta.json").write_text(json.dumps(meta, indent=2))

        return ProcessedObject(
            name          = name,
            success       = True,
            out_dir       = out_obj,
            mjcf_path     = mjcf_path,
            collision_stl = coll_stl,
            visual_obj    = vis_obj,
            meta          = meta,
            elapsed_s     = time.perf_counter() - t0,
        )

    def process_all(
        self,
        names: list[str] | None = None,
    ) -> list[ProcessedObject]:
        """
        Process all downloaded objects found in raw_dir.

        Parameters
        ----------
        names : explicit list; None → discover from raw_dir
        """
        if names is None:
            names = [
                d.name for d in sorted(self.raw_dir.iterdir())
                if d.is_dir() and (d / "google_16k" / "nontextured.stl").exists()
            ]

        results = []
        for name in names:
            result = self.process(name)
            print(result)
            results.append(result)
        return results

    def is_processed(self, name: str) -> bool:
        """Return True if out_dir/<name>/meta.json exists (already processed)."""
        return (self.out_dir / name / "meta.json").exists()

    def load_meta(self, name: str) -> dict:
        """Load and return the meta.json for a processed object."""
        p = self.out_dir / name / "meta.json"
        if not p.exists():
            raise FileNotFoundError(
                f"Object {name!r} not yet processed. "
                "Run YCBPreprocessor.process(name) first."
            )
        return json.loads(p.read_text())

    # ── STL parsing ───────────────────────────────────────────────────────────

    @staticmethod
    def _read_stl_vertices(path: Path) -> np.ndarray:
        """
        Parse an STL file (binary or ASCII) and return all vertices as (N, 3).

        Binary STL layout: 80-byte header + uint32 n_triangles +
          n_triangles × (12-byte normal + 3×12-byte vertex + 2-byte attr)
        """
        raw = path.read_bytes()

        # Detect binary vs ASCII
        is_ascii = raw[:5] == b"solid" and b"\n" in raw[:80]

        if is_ascii:
            return YCBPreprocessor._parse_ascii_stl(raw.decode("utf-8", errors="ignore"))
        else:
            return YCBPreprocessor._parse_binary_stl(raw)

    @staticmethod
    def _parse_binary_stl(raw: bytes) -> np.ndarray:
        if len(raw) < 84:
            raise ValueError("Binary STL too short")
        n_tri = struct.unpack_from("<I", raw, 80)[0]
        expected = 84 + n_tri * 50
        if len(raw) < expected:
            raise ValueError(
                f"Binary STL truncated: expected {expected} bytes, got {len(raw)}"
            )
        verts = np.frombuffer(
            raw, dtype=np.float32,
            count  = n_tri * 12,
            offset = 80 + 4,
        ).reshape(n_tri, 12)[:, 3:12].reshape(-1, 3).astype(np.float64)
        return verts

    @staticmethod
    def _parse_ascii_stl(text: str) -> np.ndarray:
        verts = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("vertex "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not verts:
            raise ValueError("No vertices found in ASCII STL")
        return np.array(verts, dtype=np.float64)

    # ── convex hull ───────────────────────────────────────────────────────────

    @staticmethod
    def _convex_hull(
        verts: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
        """
        Compute convex hull via scipy.  Returns (hull_verts, hull_faces, n_verts).
        Falls back to (None, None, None) if scipy is unavailable.
        """
        try:
            from scipy.spatial import ConvexHull
        except ImportError:
            return None, None, None

        if len(verts) < 4:
            return None, None, None

        try:
            hull  = ConvexHull(verts)
            hverts = verts[hull.vertices]
            # Re-index faces to use dense hull.vertices indexing
            old_to_new = {old: new for new, old in enumerate(hull.vertices)}
            hfaces = np.array(
                [[old_to_new[i] for i in tri] for tri in hull.simplices],
                dtype=np.int32,
            )
            return hverts, hfaces, len(hverts)
        except Exception:
            return None, None, None

    # ── STL writing ───────────────────────────────────────────────────────────

    @staticmethod
    def _write_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
        """Write a binary STL from dense vertex/face arrays."""
        n_tri  = len(faces)
        buf    = bytearray(80 + 4 + n_tri * 50)
        struct.pack_into("<I", buf, 80, n_tri)
        offset = 84
        for tri in faces:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            e1  = v1 - v0
            e2  = v2 - v0
            cross = np.cross(e1, e2)
            nlen  = np.linalg.norm(cross)
            n     = (cross / nlen) if nlen > 0 else np.array([0.0, 0.0, 1.0])
            struct.pack_into("<fff", buf, offset, *n.tolist());  offset += 12
            struct.pack_into("<fff", buf, offset, *v0.tolist()); offset += 12
            struct.pack_into("<fff", buf, offset, *v1.tolist()); offset += 12
            struct.pack_into("<fff", buf, offset, *v2.tolist()); offset += 12
            struct.pack_into("<H",   buf, offset, 0);             offset += 2
        path.write_bytes(bytes(buf))

    @staticmethod
    def _write_box_stl(
        path: Path,
        centre: np.ndarray,
        half:   np.ndarray,
    ) -> None:
        """Write a box (12 triangles) as binary STL — used when hull fails."""
        lo = centre - half
        hi = centre + half
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ], dtype=np.float64)
        faces = np.array([
            [0,1,2],[0,2,3],  # -Z
            [4,6,5],[4,7,6],  # +Z
            [0,4,5],[0,5,1],  # -Y
            [2,6,7],[2,7,3],  # +Y
            [0,3,7],[0,7,4],  # -X
            [1,5,6],[1,6,2],  # +X
        ], dtype=np.int32)
        YCBPreprocessor._write_stl(path, corners, faces)

    # ── OBJ fallback ──────────────────────────────────────────────────────────

    @staticmethod
    def _write_box_obj(
        path: Path,
        centre: np.ndarray,
        half:   np.ndarray,
    ) -> None:
        """Write a minimal box as Wavefront OBJ (visual fallback)."""
        lo = centre - half
        hi = centre + half
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        lines = [
            "# Box proxy — generated by YCBPreprocessor",
            f"v {x0} {y0} {z0}", f"v {x1} {y0} {z0}",
            f"v {x1} {y1} {z0}", f"v {x0} {y1} {z0}",
            f"v {x0} {y0} {z1}", f"v {x1} {y0} {z1}",
            f"v {x1} {y1} {z1}", f"v {x0} {y1} {z1}",
            "f 1 2 3 4", "f 8 7 6 5",
            "f 1 5 6 2", "f 3 7 8 4",
            "f 1 4 8 5", "f 2 6 7 3",
        ]
        path.write_text("\n".join(lines))

    # ── MTL path fix ──────────────────────────────────────────────────────────

    @staticmethod
    def _fix_mtl_ref(obj_path: Path) -> None:
        """Rewrite the mtllib line in .obj to point at visual.mtl."""
        text = obj_path.read_text(errors="ignore")
        lines = []
        for line in text.splitlines():
            if line.strip().lower().startswith("mtllib"):
                lines.append("mtllib visual.mtl")
            else:
                lines.append(line)
        obj_path.write_text("\n".join(lines))

    # ── MJCF generation ───────────────────────────────────────────────────────

    @staticmethod
    def _write_mjcf(
        path:      Path,
        name:      str,
        coll_stl:  Path,
        vis_obj:   Path,
        mass:      float,
        friction:  float,
        half:      np.ndarray,
    ) -> None:
        """
        Write a self-contained MJCF body snippet for one YCB object.

        The file is a complete <mujoco> document so it can be included via
        MjSpec.from_file() or merged with the main model.  The body has a
        freejoint so the object can be grasped and moved.

        Inertia is set analytically from the bounding-box half-extents to
        avoid MuJoCo needing to compute it from the mesh.
        """
        hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
        Ixx = mass * (hy**2 + hz**2) / 3.0
        Iyy = mass * (hx**2 + hz**2) / 3.0
        Izz = mass * (hx**2 + hy**2) / 3.0

        # Use relative paths so the XML is portable
        coll_rel = coll_stl.name
        vis_rel  = vis_obj.name
        label    = name.split("_", 1)[1] if "_" in name else name

        xml = f"""<mujoco model="{name}">
  <!--
    YCB object: {name}
    mass={mass:.4f} kg  friction={friction:.2f}
    half_extents=[{hx:.4f}, {hy:.4f}, {hz:.4f}] m
    Generated by YCBPreprocessor.
  -->
  <asset>
    <mesh name="{name}_col" file="{coll_rel}" scale="1 1 1"/>
    <mesh name="{name}_vis" file="{vis_rel}"  scale="1 1 1"/>
  </asset>

  <worldbody>
    <body name="{name}" pos="0 0 {hz:.4f}">
      <freejoint name="{name}_joint"/>
      <inertial mass="{mass:.6f}"
                 pos="0 0 0"
                 diaginertia="{Ixx:.8f} {Iyy:.8f} {Izz:.8f}"/>
      <!-- collision geom -->
      <geom name="{name}_col"
            type="mesh"
            mesh="{name}_col"
            friction="{friction:.2f} 0.005 0.0001"
            contype="1"
            conaffinity="1"
            group="0"
            rgba="0.8 0.8 0.8 1"/>
      <!-- visual geom (no collision) -->
      <geom name="{name}_vis"
            type="mesh"
            mesh="{name}_vis"
            contype="0"
            conaffinity="0"
            group="1"
            rgba="1 1 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""
        path.write_text(xml)
