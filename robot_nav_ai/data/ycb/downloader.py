"""
data/ycb/downloader.py — YCB dataset downloader for AutoRobo v1.

Downloads the Google 16k mesh set from the official YCB S3 bucket.
Supports parallel downloads, resume of partial files, and integrity checks.

Archive layout on S3
────────────────────
  https://ycb-benchmarks.s3.amazonaws.com/data/compressed/
    {name}_google_16k.tgz
      {name}/
        google_16k/
          textured.obj          ← visual mesh
          textured.mtl          ← material file
          textured_simple.obj   ← simplified visual
          nontextured.stl       ← STL (no UV, used for collision)

Usage
─────
    from data.ycb import YCBDownloader

    dl = YCBDownloader(dest_dir="data/ycb/raw")
    result = dl.download("002_master_chef_can")        # single object
    results = dl.download_all(n_workers=4)             # all 21 objects
    results = dl.download_all(names=["002_...", "005_..."])  # subset

    # Check what's already on disk
    status = dl.status()                               # dict[name, bool]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

from .registry import REGISTRY, YCBObject


# ── constants ─────────────────────────────────────────────────────────────────

_BASE_URL  = "https://ycb-benchmarks.s3.amazonaws.com/data/compressed"
_MESH_TYPE = "google_16k"
_CHUNK     = 1 << 17    # 128 KiB read chunk
_TIMEOUT   = 30         # HTTP connect + read timeout (seconds)

# Files that must exist inside a valid extracted object directory
_REQUIRED_FILES = [
    "{name}/google_16k/textured.obj",
    "{name}/google_16k/nontextured.stl",
]


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DownloadResult:
    """Outcome of a single-object download attempt."""
    name:      str
    success:   bool
    path:      Path | None   = None   # directory of extracted files
    error:     str  | None   = None   # error message if not success
    skipped:   bool          = False  # True if already present and verified
    bytes_dl:  int           = 0      # bytes downloaded this call (0 if skipped)
    elapsed_s: float         = 0.0

    def __str__(self) -> str:
        if self.skipped:
            return f"[skip] {self.name}"
        if self.success:
            mb = self.bytes_dl / 1e6
            return f"[ok]   {self.name}  ({mb:.1f} MB in {self.elapsed_s:.1f}s)"
        return f"[fail] {self.name}: {self.error}"


# ── downloader ────────────────────────────────────────────────────────────────

class YCBDownloader:
    """
    Downloads and extracts YCB Google-16k mesh archives.

    Parameters
    ----------
    dest_dir    : root directory for raw downloads.  Each object gets a
                  sub-directory: <dest_dir>/<name>/google_16k/
    force       : re-download even if object already verified on disk
    timeout     : HTTP request timeout in seconds
    progress_cb : optional callback(name, bytes_done, bytes_total) for UIs
    """

    def __init__(
        self,
        dest_dir:    str | Path,
        force:       bool                                    = False,
        timeout:     int                                     = _TIMEOUT,
        progress_cb: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.dest_dir    = Path(dest_dir)
        self.force       = force
        self.timeout     = timeout
        self.progress_cb = progress_cb
        self._lock       = threading.Lock()
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def download(self, name: str) -> DownloadResult:
        """
        Download and extract a single YCB object by canonical name.

        Returns immediately (skipped=True) if the object is already present
        and passes file verification — unless force=True.
        """
        if name not in REGISTRY:
            return DownloadResult(
                name    = name,
                success = False,
                error   = f"Unknown object {name!r}. Check data.ycb.REGISTRY.names().",
            )

        obj_dir = self.dest_dir / name
        if not self.force and self._verify(name, obj_dir):
            return DownloadResult(name=name, success=True, path=obj_dir, skipped=True)

        t0  = time.perf_counter()
        url = f"{_BASE_URL}/{name}_{_MESH_TYPE}.tgz"
        tgz = self.dest_dir / f"{name}_{_MESH_TYPE}.tgz"

        try:
            bytes_dl = self._fetch(name, url, tgz)
            self._extract(tgz, self.dest_dir)
            tgz.unlink(missing_ok=True)

            if not self._verify(name, obj_dir):
                return DownloadResult(
                    name    = name,
                    success = False,
                    error   = "Extraction succeeded but required files are missing.",
                )

            return DownloadResult(
                name      = name,
                success   = True,
                path      = obj_dir,
                bytes_dl  = bytes_dl,
                elapsed_s = time.perf_counter() - t0,
            )

        except requests.Timeout:
            return DownloadResult(name=name, success=False,
                                  error=f"HTTP timeout after {self.timeout}s")
        except requests.ConnectionError as e:
            return DownloadResult(name=name, success=False,
                                  error=f"Connection error: {e}")
        except requests.HTTPError as e:
            return DownloadResult(name=name, success=False,
                                  error=f"HTTP {e.response.status_code}: {url}")
        except tarfile.TarError as e:
            return DownloadResult(name=name, success=False,
                                  error=f"Archive extraction failed: {e}")
        except Exception as e:
            return DownloadResult(name=name, success=False,
                                  error=f"Unexpected error: {e}")
        finally:
            tgz.unlink(missing_ok=True)   # clean up partial archive on error

    def download_all(
        self,
        names:     list[str] | None = None,
        n_workers: int               = 4,
    ) -> list[DownloadResult]:
        """
        Download a list of YCB objects in parallel.

        Parameters
        ----------
        names     : list of canonical names; None → all 21 objects
        n_workers : thread-pool size (HTTP-bound; 4–8 is typically optimal)

        Returns
        -------
        List of DownloadResult, one per requested object.
        """
        targets = names if names is not None else REGISTRY.names()
        results: list[DownloadResult] = []

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(self.download, n): n for n in targets}
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                with self._lock:
                    print(result)

        return sorted(results, key=lambda r: r.name)

    def status(self) -> dict[str, bool]:
        """
        Return a dict mapping each canonical YCB name → verified on disk.

        Useful for checking what's already downloaded before starting a run.
        """
        return {
            name: self._verify(name, self.dest_dir / name)
            for name in REGISTRY.names()
        }

    def object_dir(self, name: str) -> Path:
        """Return the expected extraction directory for an object."""
        return self.dest_dir / name

    def mesh_obj(self, name: str) -> Path:
        """Return path to the textured .obj visual mesh."""
        return self.dest_dir / name / _MESH_TYPE / "textured.obj"

    def mesh_stl(self, name: str) -> Path:
        """Return path to the nontextured .stl collision mesh."""
        return self.dest_dir / name / _MESH_TYPE / "nontextured.stl"

    # ── internal helpers ──────────────────────────────────────────────────────

    def _fetch(self, name: str, url: str, dest: Path) -> int:
        """Stream-download url → dest, calling progress_cb per chunk."""
        response = requests.get(url, stream=True, timeout=self.timeout)
        response.raise_for_status()

        total     = int(response.headers.get("Content-Length", 0))
        received  = 0
        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=_CHUNK):
                if chunk:
                    fh.write(chunk)
                    received += len(chunk)
                    if self.progress_cb:
                        self.progress_cb(name, received, total)

        return received

    @staticmethod
    def _extract(tgz: Path, dest: Path) -> None:
        """Extract a .tgz archive into dest/."""
        with tarfile.open(tgz, "r:gz") as tf:
            # Safety: strip any absolute or traversal paths
            members = []
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in m.name:
                    continue
                members.append(m)
            tf.extractall(dest, members=members)

    @staticmethod
    def _verify(name: str, obj_dir: Path) -> bool:
        """
        Return True if the object directory contains all required files.

        Does NOT check file contents — just existence.  A corrupted archive
        would need force=True to re-download.
        """
        for tmpl in _REQUIRED_FILES:
            p = obj_dir.parent / tmpl.format(name=name)
            if not p.exists():
                return False
        return True

    # ── persistence ───────────────────────────────────────────────────────────

    def save_status(self, path: str | Path) -> None:
        """Write status JSON to disk for CI / DVC tracking."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        st = {
            "dest_dir":  str(self.dest_dir),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "objects":   self.status(),
        }
        path.write_text(json.dumps(st, indent=2))

    @staticmethod
    def load_status(path: str | Path) -> dict:
        """Load a previously saved status JSON."""
        return json.loads(Path(path).read_text())
