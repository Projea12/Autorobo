"""
scripts/webcam_demo.py — Live webcam perception demo.

Opens your Mac webcam, runs YOLO object detection on every frame,
and draws bounding boxes + confidence scores in real time.

Uses yolov8n.pt (COCO classes) by default so it works immediately
without training. Switch to --weights path/to/ycb.pt once you have
a trained YCB model.

Usage
─────
    cd robot_nav_ai
    python scripts/webcam_demo.py                       # default webcam
    python scripts/webcam_demo.py --camera 1            # second camera
    python scripts/webcam_demo.py --weights yolov8n.pt  # explicit model
    python scripts/webcam_demo.py --conf 0.4            # confidence threshold
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from perception.detector import DetectorConfig, ObjectDetector


# ── colours per class id (cycles through palette) ────────────────────────────

_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
    (255, 152, 0),  (255, 87, 34),  (255, 61, 0),   (255, 45, 0),
    (255, 0, 0),    (196, 43, 28),  (139, 0, 0),    (0, 128, 0),
    (0, 255, 127),
]


def _colour(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def _draw(frame: np.ndarray, detections, fps: float) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
        col  = _colour(det.class_id)
        label = f"{det.class_name}  {det.confidence:.0%}"

        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # FPS + object count overlay
    info = f"FPS: {fps:.1f}   Objects: {len(detections)}"
    cv2.putText(out, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(out, "Press Q to quit", (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def run(camera: int, weights: str, conf: float, iou: float) -> None:
    print(f"Loading model: {weights}")
    cfg      = DetectorConfig(weights_path=weights, conf_thresh=conf, iou_thresh=iou)
    detector = ObjectDetector(cfg)

    print(f"Opening camera {camera} ...")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit(f"ERROR: Could not open camera {camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Running — press Q in the window to quit\n")
    t_prev = time.perf_counter()
    fps    = 0.0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Camera read failed — exiting")
            break

        # OpenCV gives BGR; detector accepts BGR (auto-converts internally)
        detections = detector.detect(frame_bgr)

        t_now = time.perf_counter()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        for det in detections:
            print(f"  {det.class_name:<20} conf={det.confidence:.2f}  "
                  f"bbox={det.bbox_xyxy.astype(int).tolist()}")

        out = _draw(frame_bgr, detections, fps)
        cv2.imshow("Autorobo — Live Perception", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(description="Autorobo live webcam demo")
    p.add_argument("--camera",  type=int,   default=0,           help="Camera index (default 0)")
    p.add_argument("--weights", type=str,   default="yolov8n.pt", help="YOLO weights (default yolov8n.pt)")
    p.add_argument("--conf",    type=float, default=0.35,         help="Confidence threshold (default 0.35)")
    p.add_argument("--iou",     type=float, default=0.45,         help="IoU threshold (default 0.45)")
    args = p.parse_args()
    run(args.camera, args.weights, args.conf, args.iou)


if __name__ == "__main__":
    main()
