"""
data/dvc_utils.py — Dataset lineage utilities.

Every manifest.json written by the pipeline is stamped with:
  - the DVC run-cache ID (md5 of locked stage outputs)
  - the params.yaml values that produced it
  - the git commit that was current when generation ran

This means any model checkpoint can be traced back to the exact
dataset state that trained it by reading its manifest.json lineage block.

Usage (inside pipeline.py or any generation script)
────────────────────────────────────────────────────
    from data.dvc_utils import lineage_stamp
    manifest = {...}
    manifest["lineage"] = lineage_stamp(params_path="params.yaml")
    json.dumps(manifest)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional


def _run(cmd: list[str], cwd: Optional[Path] = None) -> str:
    """Run a subprocess, return stdout stripped; return "" on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(cwd) if cwd else None, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def git_commit(repo_root: Optional[Path] = None) -> str:
    """Current git HEAD commit hash (short). Empty string if not in a repo."""
    return _run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)


def git_dirty(repo_root: Optional[Path] = None) -> bool:
    """True if the working tree has uncommitted changes."""
    return bool(_run(["git", "status", "--porcelain"], cwd=repo_root))


def dvc_repro_id(repo_root: Optional[Path] = None) -> str:
    """
    Return the DVC run-id for the most recently reproduced stage.
    Falls back to "" when DVC is unavailable or no run has completed.
    """
    # DVC stores run IDs in .dvc/tmp/repro_lock — not a stable API,
    # so we use `dvc status --json` to detect whether outputs are cached.
    status = _run(["dvc", "status", "--json"], cwd=repo_root)
    if not status:
        return ""
    try:
        d = json.loads(status)
        # If status is empty dict → all stages committed → pipeline is clean
        return "clean" if not d else "modified"
    except json.JSONDecodeError:
        return ""


def load_params(params_path: str | Path) -> dict[str, Any]:
    """Load params.yaml into a dict.  Returns {} if file not found."""
    p = Path(params_path)
    if not p.exists():
        return {}
    try:
        import yaml  # optional dependency
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def load_dvc_lock(dvc_lock_path: str | Path) -> dict[str, Any]:
    """Parse dvc.lock to extract per-stage md5 hashes of outputs."""
    p = Path(dvc_lock_path)
    if not p.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def stage_output_hashes(
    stage: str,
    dvc_lock_path: str | Path = "dvc.lock",
) -> dict[str, str]:
    """
    Return {path: md5} for every output of the named stage from dvc.lock.
    Empty dict if dvc.lock doesn't exist or stage is not present.
    """
    lock = load_dvc_lock(dvc_lock_path)
    stages = lock.get("stages", {})
    if stage not in stages:
        return {}
    outs = stages[stage].get("outs", [])
    result: dict[str, str] = {}
    for entry in outs:
        if isinstance(entry, dict):
            for path, meta in entry.items():
                if isinstance(meta, dict):
                    h = meta.get("md5") or meta.get("sha256") or ""
                    result[path] = h
    return result


def lineage_stamp(
    params_path: str | Path = "params.yaml",
    dvc_lock_path: str | Path = "dvc.lock",
    repo_root: Optional[Path] = None,
    stage: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build a lineage block suitable for embedding in manifest.json.

    Parameters
    ----------
    params_path   : path to params.yaml (relative to cwd or absolute)
    dvc_lock_path : path to dvc.lock
    repo_root     : git/dvc repo root (defaults to cwd)
    stage         : if given, include output hashes for that stage

    Returns
    -------
    dict with keys:
      git_commit      — short commit hash or ""
      git_dirty       — bool, True if working tree has changes
      dvc_status      — "clean" | "modified" | ""
      params          — contents of params.yaml
      stage_hashes    — {path: md5} from dvc.lock for the named stage
      generated_at    — ISO-8601 UTC timestamp
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    params = load_params(root / params_path if not Path(params_path).is_absolute() else params_path)
    hashes = stage_output_hashes(stage, root / dvc_lock_path) if stage else {}

    return {
        "git_commit":   git_commit(root),
        "git_dirty":    git_dirty(root),
        "dvc_status":   dvc_repro_id(root),
        "params":       params,
        "stage_hashes": hashes,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ── manifest helpers ──────────────────────────────────────────────────────────

def read_lineage(manifest_path: str | Path) -> dict[str, Any]:
    """Read the lineage block from an existing manifest.json."""
    p = Path(manifest_path)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        return d.get("lineage", {})
    except Exception:
        return {}


def datasets_match(manifest_a: str | Path, manifest_b: str | Path) -> bool:
    """
    Return True if two manifests were produced by identical pipeline runs.
    Compares git commit + params only (ignores timestamp).
    """
    a = read_lineage(manifest_a)
    b = read_lineage(manifest_b)
    if not a or not b:
        return False
    return (
        a.get("git_commit") == b.get("git_commit")
        and a.get("params")  == b.get("params")
    )
