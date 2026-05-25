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
    # Graspable tabletop objects (primary targets)
    "cup", "mug", "bottle", "bowl", "can", "glass",
    "book", "remote", "phone", "keys",
    # Furniture / surfaces (navigation landmarks)
    "chair", "table", "couch", "shelf",
    # Plants / food
    "potted plant", "apple", "banana",
]


# ── Detection result ──────────────────────────────────────────────────────────

@dataclass
class Detection:
    label:       str
    confidence:  float
    bbox_xyxy:   Tuple[int, int, int, int]   # x1, y1, x2, y2  (pixels)
    centroid_uv: Tuple[int, int]             # (u, v) centre pixel
    track_id:    int = -1                    # assigned by IoUTracker (-1 = untracked)
    position_3d: Optional[Tuple[float, float, float]] = None  # (X,Y,Z) metres, camera frame

    def __str__(self) -> str:
        u, v = self.centroid_uv
        tid  = f" #{self.track_id}" if self.track_id >= 0 else ""
        xyz  = (f" [{self.position_3d[0]:+.1f},{self.position_3d[1]:+.1f},{self.position_3d[2]:.1f}m]"
                if self.position_3d is not None else "")
        return f"{self.label}{tid} {self.confidence:.0%} @ ({u},{v}){xyz}"


# ── per-track Kalman filter ───────────────────────────────────────────────────

class _KalmanBox:
    """
    Constant-velocity Kalman filter for one bounding box.

    State  x = [cx, cy, w, h, vx, vy, vw, vh]  (8-D)
    Obs    z = [cx, cy, w, h]                   (4-D)
    """

    def __init__(self, bbox: Tuple[int,int,int,int]) -> None:
        cx, cy, w, h = _bbox_to_cwh(bbox)

        # Transition matrix (constant velocity)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i+4] = 1.0

        # Observation matrix
        self.H = np.eye(4, 8, dtype=np.float32)

        # Noise
        self.Q = np.diag([1,1,1,1, 10,10,5,5]).astype(np.float32)   # process
        self.R = np.diag([4,4,16,16]).astype(np.float32)             # measurement

        self.P = np.diag([10,10,20,20, 100,100,50,50]).astype(np.float32)

        self.x = np.array([cx,cy,w,h, 0,0,0,0], dtype=np.float32)

    def predict(self) -> Tuple[float,float,float,float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        cx,cy,w,h = self.x[:4]
        return cx,cy,max(1,w),max(1,h)

    def update(self, bbox: Tuple[int,int,int,int]) -> None:
        z = np.array(_bbox_to_cwh(bbox), dtype=np.float32)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def predicted_bbox(self) -> Tuple[int,int,int,int]:
        cx,cy,w,h = self.x[:4]
        return _cwh_to_bbox(cx,cy,w,h)


# ── Kalman multi-object tracker ───────────────────────────────────────────────

class IoUTracker:
    """
    Kalman-filter-based multi-object tracker.

    Each track has its own Kalman filter that predicts where the object
    will appear next frame.  Detections are matched against PREDICTIONS
    (not last observed positions) — this survives fast camera motion and
    noisy YOLO bounding boxes.

    Same-label constraint: a bottle can never steal a cup's ID.
    Two-stage matching:
      Stage 1 — high-confidence IoU (≥ iou_hi) against predicted boxes
      Stage 2 — centroid distance fallback for remaining pairs
    """

    def __init__(
        self,
        iou_hi:      float = 0.20,   # strong IoU match threshold
        dist_thresh: float = 120.0,  # px fallback centroid distance
        max_age:     int   = 15,     # frames to keep an unmatched track
        min_hits:    int   = 2,      # detections needed before track is confirmed
    ) -> None:
        self._iou_hi      = iou_hi
        self._dist_thresh = dist_thresh
        self._max_age     = max_age
        self._min_hits    = min_hits
        self._next_id     = 0
        # tid → {kf, label, age, hits, confirmed}
        self._tracks: dict[int, dict] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Match detections to tracks, update Kalman filters, return detections
        with stable track_id.  Unconfirmed tracks (< min_hits) are hidden."""

        # Step 1 — predict all tracks one step forward
        predictions: dict[int, Tuple] = {}
        for tid, trk in self._tracks.items():
            predictions[tid] = trk["kf"].predict()

        live_ids = list(self._tracks.keys())
        matched_det: dict[int, int] = {}   # det_idx → tid
        matched_trk: set[int]       = set()

        if detections and live_ids:
            # Stage 1: high-confidence IoU against predicted bbox
            pred_boxes = {tid: trk["kf"].predicted_bbox
                          for tid, trk in self._tracks.items()}
            score = self._build_score(detections, live_ids, pred_boxes, iou_weight=2.0)
            matched_det, matched_trk = self._greedy_match(
                score, detections, live_ids, matched_det, matched_trk, thresh=self._iou_hi
            )

            # Stage 2: centroid distance for remaining pairs
            remaining_dets = [i for i in range(len(detections)) if i not in matched_det]
            remaining_trks = [j for j, tid in enumerate(live_ids) if j not in matched_trk]
            if remaining_dets and remaining_trks:
                score2 = self._build_dist_score(
                    detections, remaining_dets,
                    live_ids, remaining_trks, pred_boxes
                )
                matched_det, matched_trk = self._greedy_match(
                    score2, detections, live_ids, matched_det, matched_trk, thresh=0.01
                )

        # Step 2 — update matched tracks
        for det_i, tid in matched_det.items():
            self._tracks[tid]["kf"].update(detections[det_i].bbox_xyxy)
            self._tracks[tid]["age"]  = 0
            self._tracks[tid]["hits"] += 1
            if self._tracks[tid]["hits"] >= self._min_hits:
                self._tracks[tid]["confirmed"] = True
            detections[det_i].track_id = tid

        # Step 3 — new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det:
                tid = self._next_id
                self._tracks[tid] = {
                    "kf":        _KalmanBox(det.bbox_xyxy),
                    "label":     det.label,
                    "age":       0,
                    "hits":      1,
                    "confirmed": False,
                }
                det.track_id   = tid
                self._next_id += 1

        # Step 4 — age out dead tracks
        for tid in list(self._tracks):
            if tid not in matched_det.values():
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self._max_age:
                    del self._tracks[tid]

        # Only return confirmed tracks (suppress single-frame noise)
        return [d for d in detections
                if d.track_id in self._tracks
                and self._tracks[d.track_id]["confirmed"]]

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_score(self, dets, live_ids, pred_boxes, iou_weight=1.0):
        n, m = len(dets), len(live_ids)
        score = np.zeros((n, m), np.float32)
        for i, det in enumerate(dets):
            for j, tid in enumerate(live_ids):
                if det.label != self._tracks[tid]["label"]:
                    continue
                score[i, j] = _iou(det.bbox_xyxy, pred_boxes[tid]) * iou_weight
        return score

    def _build_dist_score(self, dets, det_idxs, live_ids, trk_idxs, pred_boxes):
        n, m = len(dets), len(live_ids)
        score = np.zeros((n, m), np.float32)
        for i in det_idxs:
            det = dets[i]
            dc  = det.centroid_uv
            for j in trk_idxs:
                tid = live_ids[j]
                if det.label != self._tracks[tid]["label"]:
                    continue
                pb  = pred_boxes[tid]
                tc  = ((pb[0]+pb[2])//2, (pb[1]+pb[3])//2)
                dist = ((dc[0]-tc[0])**2 + (dc[1]-tc[1])**2) ** 0.5
                if dist < self._dist_thresh:
                    score[i, j] = 1.0 - dist / self._dist_thresh
        return score

    @staticmethod
    def _greedy_match(score, dets, live_ids, matched_det, matched_trk, thresh):
        flat = np.argsort(-score, axis=None)
        for idx in flat:
            i, j = divmod(int(idx), len(live_ids))
            if score[i, j] < thresh:
                break
            if i in matched_det or j in matched_trk:
                continue
            matched_det[i] = live_ids[j]
            matched_trk.add(j)
        return matched_det, matched_trk


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
        weights:     str        = "yolov8l-worldv2.pt",
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
        self._tracker = IoUTracker(iou_hi=0.20, dist_thresh=120.0, max_age=20, min_hits=1)
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
        boxes = result.boxes
        if boxes is None or len(boxes.cls) == 0:
            return []

        xyxy  = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        raw = []
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clses):
            label = self._model.names[cls]
            cx    = int((x1 + x2) / 2)
            cy    = int((y1 + y2) / 2)
            raw.append(Detection(
                label       = label,
                confidence  = float(conf),
                bbox_xyxy   = (x1, y1, x2, y2),
                centroid_uv = (cx, cy),
            ))

        # Per-label NMS: suppress overlapping boxes of the same class.
        # Low threshold (0.25) catches YOLO-World over-splitting large objects.
        return _label_nms(raw, iou_thresh=0.25)

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


def _label_nms(dets: List[Detection], iou_thresh: float = 0.40) -> List[Detection]:
    """
    Per-label Non-Maximum Suppression.
    Within each label, suppress lower-confidence boxes that overlap
    a higher-confidence box by more than iou_thresh.
    This prevents YOLO-World from creating 3 'bed' tracks for one bed.
    """
    by_label: dict[str, List[Detection]] = {}
    for d in dets:
        by_label.setdefault(d.label, []).append(d)

    kept = []
    for label, group in by_label.items():
        group.sort(key=lambda d: -d.confidence)
        suppressed = set()
        for i, d in enumerate(group):
            if i in suppressed:
                continue
            kept.append(d)
            for j in range(i + 1, len(group)):
                if j not in suppressed and _iou(d.bbox_xyxy, group[j].bbox_xyxy) > iou_thresh:
                    suppressed.add(j)
    return kept


def _bbox_to_cwh(b):
    """xyxy → (cx, cy, w, h)"""
    return (b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]

def _cwh_to_bbox(cx, cy, w, h):
    return (int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2))

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
    parser.add_argument("--weights",  type=str,  default="yolov8l-worldv2.pt",
                        help="Model weights: yolov8l-worldv2.pt (default) or yolov8n.pt (fast)")
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
