"""
scripts/test_perception.py — Live MacBook webcam perception test.

Runs the full Option B tiered pipeline on MacBook FaceTime camera.
Displays live annotated feed:
  - Green boxes   : high-confidence YOLOv8 detections
  - Orange boxes  : Tier 2 (SAM 2 + CLIP) identified unknowns
  - Depth overlay : top-left corner colourised depth map (small)
  - HUD           : FPS, detection counts, Tier 2 status, cache size

Press Q to quit.
Press S to save current frame to /tmp/perception_frame.png
"""

from __future__ import annotations

import sys
import os
import time
import logging

import cv2
import numpy as np

# Resolve repo root so imports work from any working directory
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "robot_nav_ai"))

from perception.pipeline import PerceptionPipeline, PipelineConfig, PerceptionFrame

logging.basicConfig(
    level  = logging.WARNING,
    format = "%(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("test_perception")


# ── colours ───────────────────────────────────────────────────────────────────
GREEN  = (0, 200, 0)
ORANGE = (0, 140, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GREY   = (120, 120, 120)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.45
FONT_MED   = 0.55


# ── drawing ───────────────────────────────────────────────────────────────────

def _draw_detections(frame: np.ndarray, result: PerceptionFrame) -> np.ndarray:
    vis = frame.copy()

    for det in result.detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), GREEN, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        _put_label(vis, label, x1, y1 - 6, GREEN)

    for unk in result.unknowns:
        x1, y1, x2, y2 = (int(v) for v in unk.bbox_xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), ORANGE, 2)
        label = f"{unk.label} {unk.confidence:.2f} ({unk.depth:.1f}m)"
        _put_label(vis, label, x1, y1 - 6, ORANGE)

        if unk.mask is not None:
            overlay = vis.copy()
            overlay[unk.mask] = (0, 80, 200)
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    return vis


def _put_label(img, text, x, y, colour):
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SMALL, 1)
    cv2.rectangle(img, (x, y - th - 2), (x + tw, y + 2), BLACK, -1)
    cv2.putText(img, text, (x, y), FONT, FONT_SMALL, colour, 1, cv2.LINE_AA)


def _draw_hud(
    frame:      np.ndarray,
    result:     PerceptionFrame,
    fps:        float,
    tier2:      bool,
    cache_size: int,
) -> np.ndarray:
    lines = [
        f"FPS: {fps:.1f}  |  frame {result.frame_id}",
        f"Detections: {len(result.detections)}  |  Unknowns: {len(result.unknowns)}",
        f"Pipeline: {result.elapsed_ms:.0f}ms",
        f"Tier 2 (SAM2+CLIP): {'ON' if tier2 else 'OFF'}  |  cache: {cache_size}",
    ]
    y = 20
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, FONT, FONT_MED, 1)
        cv2.rectangle(frame, (8, y - th - 3), (14 + tw, y + 3), BLACK, -1)
        cv2.putText(frame, line, (10, y), FONT, FONT_MED, WHITE, 1, cv2.LINE_AA)
        y += th + 8
    return frame


def _depth_thumbnail(depth_map: np.ndarray, size: tuple[int, int] = (160, 120)) -> np.ndarray:
    """Colourised depth map thumbnail (Inferno palette)."""
    vmin = float(np.percentile(depth_map, 2))
    vmax = float(np.percentile(depth_map, 98))
    norm = np.clip((depth_map - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    grey = (norm * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)
    return cv2.resize(coloured, size)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading PerceptionPipeline...")
    pipe = PerceptionPipeline(cfg=PipelineConfig(
        low_conf_trigger = 0.40,
        min_conf_discard = 0.15,
    ))
    print(f"Ready: {pipe}")
    print("Opening MacBook webcam (device 0)...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_window: list[float] = []
    print("Running. Press Q to quit, S to save frame.")

    while True:
        t_start = time.perf_counter()

        ok, bgr = cap.read()
        if not ok:
            log.warning("Frame capture failed — skipping")
            continue

        # Pipeline expects RGB
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = pipe.process(rgb)

        # Annotate on BGR for display
        vis = _draw_detections(bgr, result)
        vis = _draw_hud(vis, result, fps=_fps(fps_window, t_start),
                        tier2=pipe.tier2_active, cache_size=pipe.cache_size)

        # Depth thumbnail — bottom-right corner
        thumb = _depth_thumbnail(result.depth_map)
        H, W  = vis.shape[:2]
        th, tw = thumb.shape[:2]
        vis[H - th: H, W - tw: W] = thumb

        cv2.imshow("Autorobo — Perception Pipeline", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = "/tmp/perception_frame.png"
            cv2.imwrite(path, vis)
            print(f"Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


def _fps(window: list[float], t_start: float) -> float:
    window.append(time.perf_counter() - t_start)
    if len(window) > 30:
        window.pop(0)
    avg = sum(window) / len(window)
    return 1.0 / avg if avg > 0 else 0.0


if __name__ == "__main__":
    main()
