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

# ── Home robot vocabulary ─────────────────────────────────────────────────────
# Used when running YOLO-World (open vocabulary).
# Add anything you want the robot to be able to see and pick up.
HOME_CLASSES = [
    # Furniture
    "chair", "couch", "sofa", "bed", "table", "desk", "shelf", "wardrobe",
    "cabinet", "drawer", "door", "window", "mirror", "curtain", "pillow",
    # Containers / storage
    "box", "basket", "bag", "backpack", "suitcase", "bin", "bucket",
    # Kitchen
    "cup", "mug", "bottle", "bowl", "plate", "glass", "fork", "knife",
    "spoon", "kettle", "thermos", "can",
    # Electronics
    "phone", "laptop", "remote", "keyboard", "mouse", "charger", "cable",
    "tv", "speaker", "headphones", "tablet", "camera",
    # Everyday objects
    "book", "pen", "pencil", "scissors", "tape", "keys", "wallet",
    "glasses", "watch", "shoe", "clothing", "towel", "umbrella",
    # Food / plants
    "apple", "banana", "orange", "bottle", "potted plant",
    # Cleaning
    "broom", "mop", "dustpan", "spray bottle",
]


# ── Detection result ──────────────────────────────────────────────────────────

@dataclass
class Detection:
    label:       str
    confidence:  float
    bbox_xyxy:   Tuple[int, int, int, int]   # x1, y1, x2, y2  (pixels)
    centroid_uv: Tuple[int, int]             # (u, v) centre pixel
    track_id:    int = -1                    # assigned by IoUTracker (-1 = untracked)

    def __str__(self) -> str:
        u, v = self.centroid_uv
        tid  = f" #{self.track_id}" if self.track_id >= 0 else ""
        return f"{self.label}{tid} {self.confidence:.0%} @ ({u},{v})"


# ── IoU centroid tracker ───────────────────────────────────────────────────────

class IoUTracker:
    """
    Assigns stable integer track IDs to detections across frames.

    Algorithm (greedy IoU matching):
      1. Compute IoU between every new detection and every live track.
      2. Greedily match highest-IoU pairs (IoU > iou_thresh).
      3. Matched tracks keep their ID.
      4. Unmatched new detections get a fresh ID.
      5. Tracks unseen for max_age frames are removed.

    Same-label constraint: only match detections to tracks of the same class
    so a bottle never steals a cup's ID.
    """

    def __init__(self, iou_thresh: float = 0.30, max_age: int = 8) -> None:
        self._iou_thresh = iou_thresh
        self._max_age    = max_age
        self._next_id    = 0
        # track_id → {bbox, label, age (frames since last seen)}
        self._tracks: dict[int, dict] = {}

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Assign / update track IDs. Returns detections with track_id set."""
        if not detections:
            # Age out all tracks
            for tid in list(self._tracks):
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self._max_age:
                    del self._tracks[tid]
            return []

        live_ids  = list(self._tracks.keys())
        det_boxes = [d.bbox_xyxy for d in detections]
        trk_boxes = [self._tracks[t]["bbox"] for t in live_ids]

        # IoU matrix: rows=detections, cols=tracks
        iou_mat = np.zeros((len(detections), len(live_ids)), dtype=np.float32)
        for i, db in enumerate(det_boxes):
            for j, tb in enumerate(trk_boxes):
                if detections[i].label == self._tracks[live_ids[j]]["label"]:
                    iou_mat[i, j] = _iou(db, tb)

        matched_det  = set()
        matched_trk  = set()

        # Greedy: match highest IoU pairs first
        flat = np.argsort(-iou_mat, axis=None)
        for idx in flat:
            i, j = divmod(int(idx), len(live_ids))
            if iou_mat[i, j] < self._iou_thresh:
                break
            if i in matched_det or j in matched_trk:
                continue
            tid = live_ids[j]
            detections[i].track_id = tid
            self._tracks[tid]["bbox"] = det_boxes[i]
            self._tracks[tid]["age"]  = 0
            matched_det.add(i)
            matched_trk.add(j)

        # New tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det:
                det.track_id = self._next_id
                self._tracks[self._next_id] = {
                    "bbox":  det.bbox_xyxy,
                    "label": det.label,
                    "age":   0,
                }
                self._next_id += 1

        # Age out unmatched tracks
        for j, tid in enumerate(live_ids):
            if j not in matched_trk:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self._max_age:
                    del self._tracks[tid]

        return detections


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
        weights:     str        = "yolov8s-world.pt",
        every_n:     int        = 3,
        conf_thresh: float      = 0.20,
        device:      str        = "mps",
        classes:     list       = None,
    ) -> None:
        from ultralytics import YOLO

        self._is_world = "world" in weights.lower()
        if self._is_world and self._clip_ready():
            print(f"[detector] Loading {weights} (open-vocabulary) on {device} ...")
            self._model = YOLO(weights)
            vocab = classes if classes is not None else HOME_CLASSES
            self._model.set_classes(vocab)
            print(f"[detector] YOLO-World ready — {len(vocab)} home classes.")
        else:
            if self._is_world:
                print("[detector] YOLO-World CLIP encoder not ready.")
                print("[detector] Falling back to yolov8n.pt — run once to download CLIP (~338MB).")
            else:
                print(f"[detector] Loading {weights} on {device} ...")
            self._model    = YOLO("yolov8n.pt")
            self._is_world = False

        self._device  = device
        self._conf    = conf_thresh
        self._every_n = every_n

        self._lock        = threading.Lock()
        self._frame_in    = None          # latest frame from main thread
        self._detections: List[Detection] = []
        self._frame_count = 0
        self._fps         = 0.0
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._loop, daemon=True)

        # Colours per track ID (stable colour = stable ID)
        self._colours: dict[int, Tuple[int,int,int]] = {}
        self._target_label: Optional[str] = None
        self._tracker = IoUTracker(iou_thresh=0.30, max_age=8)
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
        Return the best detection matching a natural-language label.
        Tries exact match first, then partial/synonym match.
        e.g. query('bottle'), query('mug'), query('pick up the cup')
        """
        label = _extract_object_label(label)
        dets  = self.latest

        # 1 — exact class name match
        matches = [d for d in dets if d.label == label]

        # 2 — partial match (e.g. 'cell phone' contains 'phone')
        if not matches:
            matches = [d for d in dets if label in d.label or d.label in label]

        # 3 — synonym match
        if not matches:
            syns = _SYNONYMS.get(label, [])
            matches = [d for d in dets if d.label in syns]

        if not matches:
            return None

        # When multiple instances, pick the one closest to frame centre
        if len(matches) == 1:
            return matches[0]
        return min(matches, key=lambda d: _dist_to_centre(d, self._frame_in))

    def set_target(self, label: str) -> bool:
        """
        Lock onto an object by label. Returns True if found.
        The locked target is highlighted differently in draw().
        """
        with self._lock:
            self._target_label = _extract_object_label(label)
        found = self.query(label) is not None
        if found:
            print(f"[detector] Target locked: '{self._target_label}'")
        else:
            print(f"[detector] '{self._target_label}' not visible yet — will highlight when found")
        return found

    def clear_target(self) -> None:
        with self._lock:
            self._target_label = None

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and labels. Target object is highlighted green."""
        out    = frame.copy()
        dets   = self.latest
        target = self._target_label

        for det in dets:
            is_target = (target is not None and
                         (det.label == target or target in det.label
                          or det.label in _SYNONYMS.get(target, [])))

            colour    = (0, 255, 80) if is_target else self._track_colour(det.track_id)
            thickness = 3            if is_target else 2
            x1, y1, x2, y2 = det.bbox_xyxy

            if is_target:
                cv2.rectangle(out, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 80), 1)

            cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)

            tid_str   = f"#{det.track_id} " if det.track_id >= 0 else ""
            label_txt = f"{'► ' if is_target else ''}{tid_str}{det.label} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), colour, -1)
            cv2.putText(out, label_txt, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0) if is_target else (255, 255, 255),
                        1, cv2.LINE_AA)

            u, v = det.centroid_uv
            cv2.circle(out, (u, v), 5 if is_target else 3, colour, -1)

        # FPS + target status
        status = f"det {self._fps:.1f}fps"
        if target:
            found  = any(d.label == target or target in d.label for d in dets)
            status += f"  |  target: {target} {'✓' if found else '(searching...)'}"
        cv2.putText(out, status, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1, cv2.LINE_AA)
        return out

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clip_ready() -> bool:
        """Return True only if the CLIP ViT-B-32 encoder is fully downloaded."""
        import hashlib
        clip_path = Path.home() / ".cache" / "clip" / "ViT-B-32.pt"
        expected_size = 354_226_516   # 338 MB
        if not clip_path.exists():
            return False
        if clip_path.stat().st_size < expected_size * 0.99:
            return False
        return True

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
            dets = self._tracker.update(dets)   # assign stable track IDs

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

    def _track_colour(self, track_id: int) -> Tuple[int, int, int]:
        """Stable colour per track ID — same object always same colour."""
        if track_id not in self._colours:
            h = (track_id * 47) % 180   # spread hues evenly
            self._colours[track_id] = tuple(
                int(c) for c in cv2.cvtColor(
                    np.uint8([[[h, 220, 190]]]), cv2.COLOR_HSV2BGR
                )[0][0]
            )
        return self._colours[track_id]


# ── label helpers ────────────────────────────────────────────────────────────

# Common synonyms → COCO class names
_SYNONYMS: dict[str, list] = {
    "mug":        ["cup"],
    "glass":      ["wine glass", "cup"],
    "phone":      ["cell phone"],
    "mobile":     ["cell phone"],
    "couch":      ["couch", "sofa"],
    "sofa":       ["couch"],
    "fridge":     ["refrigerator"],
    "tv":         ["tv", "monitor"],
    "laptop":     ["laptop"],
    "bag":        ["handbag", "backpack", "suitcase"],
    "luggage":    ["suitcase"],
    "plant":      ["potted plant"],
    "remote":     ["remote"],
    "controller": ["remote"],
    "toothbrush": ["toothbrush"],
}

# Words to strip from natural language commands
_STRIP = {"pick", "up", "grab", "get", "fetch", "bring", "take",
          "the", "a", "an", "that", "this", "please", "me", "for", "i"}


def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Intersection-over-Union of two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _extract_object_label(text: str) -> str:
    """'pick up the bottle' → 'bottle'  (word-level strip, not substring)"""
    words = text.strip().lower().split()
    kept  = [w for w in words if w not in _STRIP]
    return " ".join(kept) if kept else text.strip().lower()


def _dist_to_centre(det: "Detection", frame) -> float:
    """Distance from detection centroid to frame centre."""
    if frame is None:
        return 0.0
    h, w = frame.shape[:2]
    u, v = det.centroid_uv
    return ((u - w/2)**2 + (v - h/2)**2) ** 0.5


# ── standalone test ───────────────────────────────────────────────────────────

def main() -> None:
    import sys, argparse
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="Object detector test")
    parser.add_argument("--video", required=True)
    parser.add_argument("--conf",    type=float, default=0.20)
    parser.add_argument("--every-n", type=int,   default=3)
    parser.add_argument("--weights",  type=str,  default="yolov8s-world.pt",
                        help="Model weights: yolov8n.pt (fast) or yolov8s-world.pt (anything)")
    args = parser.parse_args()

    video_path = ROOT / args.video if not Path(args.video).is_absolute() \
                 else Path(args.video)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open {video_path}"); return

    detector = ObjectDetector(
        weights     = args.weights,
        conf_thresh = args.conf,
        every_n     = args.every_n,
    )
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
