"""
ar/object_detector.py — YOLO-based object detection for Autorobo.

Runs inference in a background thread every N frames so the display loop
is never blocked.  The main loop reads the latest detections at any time.

Usage
-----
    detector = ObjectDetector()
    detector.start()

    # in display loop:
    frame_with_boxes = detector.draw(frame)
    detections       = detector.latest          # list[Detection]

    detector.stop()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ── Detection result ──────────────────────────────────────────────────────────

@dataclass
class Detection:
    label:      str
    confidence: float
    bbox_xyxy:  Tuple[int, int, int, int]   # x1, y1, x2, y2  (pixels)
    centroid_uv: Tuple[int, int]             # (u, v) centre pixel

    def __str__(self) -> str:
        u, v = self.centroid_uv
        return f"{self.label} {self.confidence:.0%} @ ({u},{v})"


# ── Detector ──────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Background-threaded YOLO detector.

    Parameters
    ----------
    weights    : path to .pt weights file
    every_n    : run inference every N frames fed via update()
    conf_thresh: minimum confidence to report a detection
    device     : 'mps' (M1 GPU), 'cpu', or 'cuda'
    """

    def __init__(
        self,
        weights:     str   = "yolov8n.pt",
        every_n:     int   = 3,
        conf_thresh: float = 0.30,
        device:      str   = "mps",
    ) -> None:
        from ultralytics import YOLO
        print(f"[detector] Loading {weights} on {device} ...")
        self._model      = YOLO(weights)
        self._device     = device
        self._conf       = conf_thresh
        self._every_n    = every_n

        self._lock        = threading.Lock()
        self._frame_in    = None          # latest frame from main thread
        self._detections: List[Detection] = []
        self._frame_count = 0
        self._fps         = 0.0
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._loop, daemon=True)

        # Colours per class (consistent across frames)
        self._colours: dict[str, Tuple[int,int,int]] = {}
        print("[detector] Ready.")

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def update(self, frame: np.ndarray) -> None:
        """Feed a new frame. Inference runs every N calls."""
        with self._lock:
            self._frame_in    = frame
            self._frame_count += 1

    @property
    def latest(self) -> List[Detection]:
        """Thread-safe snapshot of the most recent detections."""
        with self._lock:
            return list(self._detections)

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    def query(self, label: str) -> Optional[Detection]:
        """
        Return the highest-confidence detection matching label,
        or None if not currently visible.
        e.g. detector.query('bottle')
        """
        matches = [d for d in self.latest if d.label == label]
        if not matches:
            return None
        return max(matches, key=lambda d: d.confidence)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of frame."""
        out  = frame.copy()
        dets = self.latest
        for det in dets:
            x1, y1, x2, y2 = det.bbox_xyxy
            colour = self._class_colour(det.label)

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            # Label background
            label_txt = f"{det.label} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(
                label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(out,
                          (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          colour, -1)
            cv2.putText(out, label_txt,
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Centroid dot
            u, v = det.centroid_uv
            cv2.circle(out, (u, v), 4, colour, -1)

        # FPS counter
        cv2.putText(out, f"det {self._fps:.1f}fps",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return out

    # ── background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        skip = 0
        while not self._stop.is_set():
            with self._lock:
                frame = self._frame_in
                count = self._frame_count

            if frame is None:
                time.sleep(0.01)
                continue

            # Only run inference every N frames
            if count % self._every_n != 0:
                time.sleep(0.005)
                continue

            t0  = time.perf_counter()
            results = self._model(
                frame,
                conf    = self._conf,
                device  = self._device,
                verbose = False,
            )[0]
            elapsed = time.perf_counter() - t0

            dets = self._parse(results)

            with self._lock:
                self._detections = dets
                self._fps = 1.0 / elapsed if elapsed > 0 else 0.0

    def _parse(self, result) -> List[Detection]:
        dets = []
        boxes = result.boxes
        if boxes is None or len(boxes.cls) == 0:
            return dets

        xyxy  = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clses):
            label = self._model.names[cls]
            cx    = int((x1 + x2) / 2)
            cy    = int((y1 + y2) / 2)
            dets.append(Detection(
                label       = label,
                confidence  = float(conf),
                bbox_xyxy   = (x1, y1, x2, y2),
                centroid_uv = (cx, cy),
            ))
        return dets

    def _class_colour(self, label: str) -> Tuple[int, int, int]:
        if label not in self._colours:
            # Deterministic colour from label hash
            h = hash(label) % 256
            self._colours[label] = tuple(
                int(c) for c in cv2.cvtColor(
                    np.uint8([[[h, 200, 180]]]), cv2.COLOR_HSV2BGR
                )[0][0]
            )
        return self._colours[label]


# ── standalone test ───────────────────────────────────────────────────────────

def main() -> None:
    import sys, argparse
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="Object detector test")
    parser.add_argument("--video", required=True)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--every-n", type=int, default=3)
    args = parser.parse_args()

    video_path = ROOT / args.video if not Path(args.video).is_absolute() \
                 else Path(args.video)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open {video_path}"); return

    detector = ObjectDetector(conf_thresh=args.conf, every_n=args.every_n)
    detector.start()

    fps_times = []
    print("\n[test] Running — press Q to quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        t0 = time.perf_counter()

        detector.update(frame)
        out = detector.draw(frame)

        # Display FPS
        fps_times.append(time.perf_counter() - t0)
        if len(fps_times) > 30:
            fps_times.pop(0)
        display_fps = 1.0 / (sum(fps_times) / len(fps_times))

        cv2.putText(out, f"display {display_fps:.1f}fps",
                    (10, out.shape[0] - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # Print detections to terminal
        dets = detector.latest
        if dets:
            print(f"\r[det] {' | '.join(str(d) for d in dets)}   ", end="")

        dh = 720; dw = int(out.shape[1] * dh / out.shape[0])
        cv2.imshow("Autorobo — Object Detector (Q to quit)",
                   cv2.resize(out, (dw, dh)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[test] Done.")


if __name__ == "__main__":
    main()
