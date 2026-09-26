"""
perception/clip_identifier.py — CLIP-based object identifier.

Identifies unknown objects by matching their visual appearance to natural
language descriptions. Called only when YOLOv8 confidence is low and SAM 2
has produced a segmented mask of an unrecognised object.

Uses OpenAI CLIP via HuggingFace transformers.
Runs on MPS (Apple M1). Falls back to CPU.

Design:
  - Takes a cropped object image (from SAM 2 mask bounding box)
  - Returns a natural language description and confidence score
  - Description is passed to Claude API for spatial reasoning
  - Result cached per spatial region — never reprocesses a known object

Failure modes handled:
  - Model load failure   → RuntimeError with clear message
  - Empty crop           → returns ("unknown object", 0.0)
  - Inference failure    → returns ("unknown object", 0.0), logs warning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

log = logging.getLogger(__name__)

_MODEL_ID = "openai/clip-vit-base-patch32"

# Candidate descriptions CLIP scores against.
# Covers common objects in Nigerian home environments + universal objects.
# Descriptions are intentionally visual — what something looks like, not its name.
_CANDIDATE_LABELS = [
    # Universal indoor navigation obstacles
    "a chair", "a table", "a sofa", "a bed", "a door", "a wall",
    "a person", "a bag", "a box", "a bottle", "a cup", "a book",
    "a laptop", "a television", "a shelf", "a cabinet", "stairs",
    # Common Nigerian household objects
    "a plastic container", "a bucket", "a large pot", "a cooking stove",
    "a generator", "a rechargeable lamp", "a fan", "a plastic bag",
    "a mattress on the floor", "a wooden bench", "a plastic chair",
    "shoes on the floor", "clothing on the floor", "a rubber slipper",
    "a large bag of grain", "a water dispenser", "a gas cylinder",
    # Generic fallbacks
    "a small object on the floor", "a large object", "furniture",
    "an unknown object",
]


@dataclass(frozen=True)
class CLIPConfig:
    model_id:   str        = _MODEL_ID
    candidates: list[str]  = field(default_factory=lambda: _CANDIDATE_LABELS)
    device:     str        = ""
    top_k:      int        = 3


@dataclass
class IdentificationResult:
    label:       str    # best matching description
    confidence:  float  # CLIP similarity score [0, 1]
    top_k:       list[tuple[str, float]]  # top-k (label, score) pairs


class CLIPIdentifier:
    """
    CLIP-based identifier for objects SAM 2 segmented but YOLOv8 could not name.

    Matches cropped object image against candidate natural language descriptions.
    Returns the best matching description and confidence score.
    Result is passed to Claude API for spatial reasoning and map labelling.
    """

    def __init__(self, cfg: CLIPConfig = CLIPConfig()) -> None:
        self.cfg     = cfg
        self._model  = None
        self._proc   = None
        self._device = cfg.device or self._auto_device()

        try:
            self._proc  = CLIPProcessor.from_pretrained(cfg.model_id, local_files_only=True)
            self._model = CLIPModel.from_pretrained(cfg.model_id, local_files_only=True).to(self._device)
            self._model.eval()
            log.info("CLIPIdentifier loaded on %s (local cache)", self._device)
        except Exception:
            try:
                self._proc  = CLIPProcessor.from_pretrained(cfg.model_id)
                self._model = CLIPModel.from_pretrained(cfg.model_id).to(self._device)
                self._model.eval()
                log.info("CLIPIdentifier loaded on %s (downloaded)", self._device)
            except Exception as exc:
                raise RuntimeError(
                    f"CLIPIdentifier failed to load '{cfg.model_id}': {exc}"
                ) from exc

    # ── public API ─────────────────────────────────────────────────────────────

    def identify(
        self,
        image:         np.ndarray,
        mask:          Optional[np.ndarray] = None,
    ) -> IdentificationResult:
        """
        Identify an unknown object from its image or masked region.

        Parameters
        ----------
        image : (H, W, 3) uint8 RGB — full frame or cropped object region
        mask  : (H, W) bool — if provided, crops to mask bounding box first

        Returns
        -------
        IdentificationResult with label, confidence, and top_k alternatives.
        Returns ("unknown object", 0.0) on any failure.
        """
        if image.size == 0 or image.ndim != 3:
            return IdentificationResult("unknown object", 0.0, [])

        try:
            crop = self._crop_to_mask(image, mask) if mask is not None else image
            if crop.size == 0:
                return IdentificationResult("unknown object", 0.0, [])

            pil_img = Image.fromarray(crop)
            inputs  = self._proc(
                text   = self.cfg.candidates,
                images = pil_img,
                return_tensors = "pt",
                padding        = True,
            ).to(self._device)

            with torch.inference_mode():
                outputs = self._model(**inputs)
                logits  = outputs.logits_per_image[0]
                probs   = logits.softmax(dim=0).cpu().numpy()

            top_indices = np.argsort(probs)[::-1][:self.cfg.top_k]
            top_k       = [
                (self.cfg.candidates[i], float(probs[i]))
                for i in top_indices
            ]

            return IdentificationResult(
                label      = top_k[0][0],
                confidence = top_k[0][1],
                top_k      = top_k,
            )

        except Exception as exc:
            log.warning("CLIPIdentifier inference failed: %s", exc)
            return IdentificationResult("unknown object", 0.0, [])

    def identify_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[IdentificationResult]:
        """
        Identify multiple cropped object images in one forward pass.
        More efficient than calling identify() in a loop.

        Parameters
        ----------
        crops : list of (H, W, 3) uint8 RGB arrays

        Returns
        -------
        List of IdentificationResult, one per crop.
        """
        if not crops:
            return []

        try:
            pil_imgs = [Image.fromarray(c) for c in crops if c.size > 0]
            inputs   = self._proc(
                text   = self.cfg.candidates,
                images = pil_imgs,
                return_tensors = "pt",
                padding        = True,
            ).to(self._device)

            with torch.inference_mode():
                outputs    = self._model(**inputs)
                logits     = outputs.logits_per_image
                probs_all  = logits.softmax(dim=1).cpu().numpy()

            results = []
            for probs in probs_all:
                top_indices = np.argsort(probs)[::-1][:self.cfg.top_k]
                top_k       = [(self.cfg.candidates[i], float(probs[i])) for i in top_indices]
                results.append(IdentificationResult(
                    label      = top_k[0][0],
                    confidence = top_k[0][1],
                    top_k      = top_k,
                ))
            return results

        except Exception as exc:
            log.warning("CLIPIdentifier batch inference failed: %s", exc)
            return [IdentificationResult("unknown object", 0.0, []) for _ in crops]

    # ── internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_to_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop image to the bounding box of a boolean mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return np.zeros((1, 1, 3), dtype=np.uint8)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return image[rmin:rmax+1, cmin:cmax+1]

    @staticmethod
    def _auto_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def __repr__(self) -> str:
        return (f"CLIPIdentifier(model='{self.cfg.model_id}', "
                f"candidates={len(self.cfg.candidates)}, "
                f"device='{self._device}')")
