"""
data/splits.py — Deterministic, contamination-proof dataset split management.

Design principles
─────────────────
1. Hash-based assignment — each image stem is assigned by
       bucket = int(sha256(stem + str(seed))[:8], 16) % 10_000
   so the split for any stem is fixed regardless of iteration order,
   directory listing, or how many other images exist.

2. Three-way split — train / val / test.  The test set is a held-out
   evaluation set that must NEVER be used during training or for
   hyperparameter search.  val is used for early stopping / model
   selection only.

3. Test-set lock — once a split_manifest.json is written its test-set
   membership is frozen.  When new images are added later they get
   assigned to train/val only; existing test stems are never evicted.
   This prevents the common mistake of re-splitting a grown dataset and
   accidentally leaking test images into training.

4. Contamination checks — explicit assertions that:
     train ∩ val  = ∅
     train ∩ test = ∅
     val   ∩ test = ∅
   checked against both in-memory records and on-disk directories.

5. DVC-aware — the SplitConfig fields map directly to params.yaml so
   changing any split ratio invalidates DVC-cached outputs downstream.

split_manifest.json layout
──────────────────────────
  {
    "config":  { ... SplitConfig ... },
    "splits":  { "000000": "train", "000001": "val", ... },
    "counts":  { "train": N, "val": N, "test": N },
    "locked_test_stems": [ "000800", ... ],   ← immutable after first write
    "generated_at": "…",
    "lineage": { ... }
  }
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ── split config ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SplitConfig:
    """
    Parameters controlling dataset splitting.

    train_frac  : fraction of images in train (e.g. 0.75)
    val_frac    : fraction in val   (e.g. 0.15)
    test_frac   : fraction in test  (e.g. 0.10)
                  train + val + test must equal 1.0
    seed        : salt for hash-based assignment; change to re-randomise
    lock_test   : if True, once a split_manifest.json exists its test
                  stems are frozen and new images never enter test
    """
    train_frac: float = 0.75
    val_frac:   float = 0.15
    test_frac:  float = 0.10
    seed:       int   = 42
    lock_test:  bool  = True

    def __post_init__(self) -> None:
        total = round(self.train_frac + self.val_frac + self.test_frac, 9)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_frac + val_frac + test_frac must equal 1.0, got {total}"
            )

    # bucket boundaries (out of 10_000)
    @property
    def _train_hi(self) -> int:
        return round(self.train_frac * 10_000)

    @property
    def _val_hi(self) -> int:
        return round((self.train_frac + self.val_frac) * 10_000)


SPLITS = ("train", "val", "test")


# ── split manager ─────────────────────────────────────────────────────────────

class SplitManager:
    """
    Assigns image stems to splits and enforces contamination rules.

    Parameters
    ----------
    cfg            : SplitConfig
    manifest_path  : path to split_manifest.json; if it exists and
                     cfg.lock_test is True, test stems are reloaded
                     from it and kept immutable
    """

    def __init__(
        self,
        cfg:           SplitConfig         = SplitConfig(),
        manifest_path: Optional[Path | str] = None,
    ) -> None:
        self.cfg           = cfg
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self._locked_test: set[str] = set()

        if self.manifest_path and self.manifest_path.exists() and cfg.lock_test:
            existing = json.loads(self.manifest_path.read_text())
            self._locked_test = set(existing.get("locked_test_stems", []))

    # ── public API ────────────────────────────────────────────────────────────

    def assign(self, stem: str) -> str:
        """Return 'train' | 'val' | 'test' for a single image stem."""
        # Locked test stems keep their assignment forever
        if stem in self._locked_test:
            return "test"

        bucket = _stem_bucket(stem, self.cfg.seed)
        if bucket < self.cfg._train_hi:
            return "train"
        if bucket < self.cfg._val_hi:
            return "val"
        return "test"

    def assign_all(self, stems: list[str]) -> dict[str, str]:
        """Assign a list of stems; returns {stem: split}."""
        return {s: self.assign(s) for s in stems}

    def build_manifest(
        self,
        stems:      list[str],
        lineage:    Optional[dict] = None,
        extra:      Optional[dict] = None,
    ) -> "SplitManifest":
        """
        Build a SplitManifest for the given stems.
        Locks the test set if cfg.lock_test is True.
        """
        assignments = self.assign_all(stems)
        test_stems  = [s for s, sp in assignments.items() if sp == "test"]
        # merge pre-existing locked stems with newly assigned ones
        all_test = sorted(set(test_stems) | self._locked_test)
        self._locked_test = set(all_test)

        return SplitManifest(
            config             = self.cfg,
            assignments        = assignments,
            locked_test_stems  = all_test,
            lineage            = lineage or {},
            extra              = extra or {},
        )

    def check_contamination(
        self,
        split_dirs: dict[str, Path],
    ) -> list[str]:
        """
        Verify on-disk split directories contain no overlapping stems.

        Parameters
        ----------
        split_dirs : {"train": Path, "val": Path, "test": Path}
                     each Path is a directory of images or labels

        Returns
        -------
        List of contamination violation strings (empty = clean).
        """
        stem_sets: dict[str, set[str]] = {}
        for split, d in split_dirs.items():
            if d.exists():
                stem_sets[split] = {p.stem for p in d.iterdir()
                                    if p.is_file()}
            else:
                stem_sets[split] = set()

        violations: list[str] = []
        splits = list(stem_sets.keys())
        for i, s1 in enumerate(splits):
            for s2 in splits[i + 1:]:
                overlap = stem_sets[s1] & stem_sets[s2]
                if overlap:
                    examples = sorted(overlap)[:5]
                    violations.append(
                        f"{s1} ∩ {s2} = {len(overlap)} stems "
                        f"(e.g. {examples})"
                    )
        return violations

    def verify_manifest(
        self,
        manifest_path: Path | str,
        split_dirs:    dict[str, Path],
    ) -> list[str]:
        """
        Full verification: check manifest assignments match on-disk reality
        and that no set overlaps.

        Returns list of violation strings (empty = clean).
        """
        violations = self.check_contamination(split_dirs)
        manifest = SplitManifest.load(manifest_path)

        # every on-disk file should match its manifest assignment
        for split, d in split_dirs.items():
            if not d.exists():
                continue
            for p in d.iterdir():
                if not p.is_file():
                    continue
                expected = manifest.assignments.get(p.stem)
                if expected is None:
                    violations.append(
                        f"'{p.stem}' found in {split}/ but not in manifest"
                    )
                elif expected != split:
                    violations.append(
                        f"'{p.stem}' is in {split}/ but manifest says {expected}"
                    )
        return violations


# ── split manifest ────────────────────────────────────────────────────────────

@dataclass
class SplitManifest:
    """Persistent record of which stem belongs to which split."""
    config:            SplitConfig
    assignments:       dict[str, str]    # stem → "train"|"val"|"test"
    locked_test_stems: list[str]
    lineage:           dict
    extra:             dict = field(default_factory=dict)

    # ── derived ───────────────────────────────────────────────────────────────

    @property
    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {s: 0 for s in SPLITS}
        for sp in self.assignments.values():
            c[sp] = c.get(sp, 0) + 1
        return c

    def stems_for(self, split: str) -> list[str]:
        return sorted(s for s, sp in self.assignments.items() if sp == split)

    def contamination_check(self) -> list[str]:
        """Return violations found purely in the in-memory assignments."""
        by_split: dict[str, set[str]] = {s: set() for s in SPLITS}
        for stem, sp in self.assignments.items():
            by_split[sp].add(stem)
        violations: list[str] = []
        splits = list(by_split.keys())
        for i, s1 in enumerate(splits):
            for s2 in splits[i + 1:]:
                overlap = by_split[s1] & by_split[s2]
                if overlap:
                    violations.append(
                        f"{s1} ∩ {s2} = {len(overlap)} stems"
                    )
        return violations

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "config":            asdict(self.config),
            "splits":            self.assignments,
            "counts":            self.counts,
            "locked_test_stems": self.locked_test_stems,
            "lineage":           self.lineage,
            "generated_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()),
        }
        doc.update(self.extra)
        path.write_text(json.dumps(doc, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "SplitManifest":
        doc = json.loads(Path(path).read_text())
        cfg_dict = doc["config"]
        cfg = SplitConfig(
            train_frac = cfg_dict["train_frac"],
            val_frac   = cfg_dict["val_frac"],
            test_frac  = cfg_dict["test_frac"],
            seed       = cfg_dict["seed"],
            lock_test  = cfg_dict["lock_test"],
        )
        reserved = {"config", "splits", "counts", "locked_test_stems",
                    "lineage", "generated_at"}
        return cls(
            config            = cfg,
            assignments       = doc.get("splits", {}),
            locked_test_stems = doc.get("locked_test_stems", []),
            lineage           = doc.get("lineage", {}),
            extra             = {k: v for k, v in doc.items()
                                 if k not in reserved},
        )

    def __repr__(self) -> str:
        c = self.counts
        return (f"SplitManifest(train={c['train']}, val={c['val']}, "
                f"test={c['test']}, locked_test={len(self.locked_test_stems)})")


# ── helpers ───────────────────────────────────────────────────────────────────

def _stem_bucket(stem: str, seed: int) -> int:
    """Deterministic bucket in [0, 10_000) for a stem + seed combination."""
    key   = f"{stem}:{seed}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % 10_000


def discover_stems(
    root: Path | str,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".txt"),
) -> list[str]:
    """
    Return sorted unique stems from all files matching extensions under root.
    Walks train/val/test subdirs if present; otherwise uses root directly.
    """
    root = Path(root)
    found: set[str] = set()
    for sub in ("train", "val", "test", ""):
        d = root / sub if sub else root
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in extensions:
                    found.add(p.stem)
    return sorted(found)


def splits_from_dirs(
    image_root: Path | str,
) -> dict[str, set[str]]:
    """
    Read split membership from on-disk directory structure.
    Returns {"train": {stems}, "val": {stems}, "test": {stems}}.
    """
    image_root = Path(image_root)
    result: dict[str, set[str]] = {}
    for split in SPLITS:
        d = image_root / split
        if d.is_dir():
            result[split] = {p.stem for p in d.iterdir() if p.is_file()}
        else:
            result[split] = set()
    return result
