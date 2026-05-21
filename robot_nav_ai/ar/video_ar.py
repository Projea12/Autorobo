"""
ar/video_ar.py — Robot navigates through a recorded room video.

The video IS the robot's camera feed — what the robot sees as it moves.
Commands control how the robot moves through the room:

    forward       → video plays forward  (robot advances through room)
    back          → video plays backward (robot reverses)
    stop          → video pauses         (robot stops)
    left          → slow crawl + turn animation (robot pivots left)
    right         → slow crawl + turn animation (robot pivots right)
    arm up        → arm raises, video pauses
    arm down      → arm lowers
    open / close  → gripper opens/closes
    wave          → robot waves, video pauses
    home          → reset arm + pause
    quit / q      → exit

Usage:
    python ar/video_ar.py --video video/room_video.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # robot_nav_ai/
sys.path.insert(0, str(ROOT))


# ── video playback speeds (frames to advance per display tick) ────────────────

PLAYBACK = {
    "FORWARD"      : 2.0,
    "BACKWARD"     : -2.0,
    "TURN_LEFT"    : 0.4,
    "TURN_RIGHT"   : 0.4,
    "STOP"         : 0.0,
    "ARM_UP"       : 0.0,
    "ARM_DOWN"     : 0.0,
    "GRIPPER_OPEN" : 0.0,
    "GRIPPER_CLOSE": 0.0,
    "WAVE"         : 0.0,
    "HOME"         : 0.0,
}


# ── video player ──────────────────────────────────────────────────────────────

class VideoPlayer:
    """
    Wraps a cv2.VideoCapture and advances frames based on the current command.
    Moving forward plays the video forward — that IS the robot moving through
    the room. Stopping pauses the video. Reversing plays it backward.
    """

    def __init__(self, path: str) -> None:
        self._cap   = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {path}")
        self._total  = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._pos    = 0.0          # fractional frame position
        self._speed  = 0.0          # frames to advance per tick (from command)
        self._lock   = threading.Lock()
        self.width   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height  = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps     = self._cap.get(cv2.CAP_PROP_FPS)

    def set_speed(self, frames_per_tick: float) -> None:
        with self._lock:
            self._speed = frames_per_tick

    def read(self) -> Optional[np.ndarray]:
        """Advance by current speed and return the frame."""
        with self._lock:
            self._pos = (self._pos + self._speed) % self._total
            pos = int(self._pos)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


# ── fixed robot placement ─────────────────────────────────────────────────────

def robot_screen_pos(w: int, h: int):
    """
    Robot is always at the bottom-centre of the frame — like a robot
    whose onboard camera always shows it in the same spot.
    Returns (u, v, sprite_height_px).
    """
    u  = w // 2
    v  = int(h * 0.88)       # 88% down = near floor
    sh = int(h * 0.38)       # robot takes up ~38% of frame height
    return u, v, sh


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autorobo — robot navigates through room video"
    )
    parser.add_argument("--video", required=True,
                        help="Path to room video (e.g. video/room_video.mp4)")
    args = parser.parse_args()

    video_path = Path(args.video) if Path(args.video).is_absolute() \
                 else ROOT / args.video

    from ar.ar_renderer       import ARConfig, ARRenderer, CameraIntrinsics
    from ar.command_interface import CommandInterface, RobotController, Cmd

    # ── open video ────────────────────────────────────────────────────────────
    player = VideoPlayer(str(video_path))
    print(f"[video_ar] {video_path.name}  "
          f"{player.width}×{player.height}  {player.fps:.1f}fps  "
          f"{player.width/player.fps:.0f}s")

    # ── build renderer ────────────────────────────────────────────────────────
    ar_cfg = ARConfig(
        render_width  = player.width,
        render_height = player.height,
        intrinsics    = CameraIntrinsics(
            fx = player.width  * 1.1,
            fy = player.width  * 1.1,
            cx = player.width  / 2.0,
            cy = player.height / 2.0,
        ),
    )
    renderer   = ARRenderer(ar_cfg)
    controller = RobotController()

    # ── command interface ─────────────────────────────────────────────────────
    quit_event = threading.Event()

    def on_command(cmd: Cmd) -> None:
        controller.set_command(cmd)
        speed = PLAYBACK.get(cmd.name, 0.0)
        player.set_speed(speed)
        print(f"[video_ar] {cmd.name}  →  "
              f"{'playing forward' if speed > 0 else 'playing backward' if speed < 0 else 'paused'}")

    cmd_iface = CommandInterface(on_command=on_command, on_quit=quit_event.set)
    renderer.start_physics(controller=controller)
    cmd_iface.start()

    # ── fixed robot position for this video ──────────────────────────────────
    u, v, sprite_h = robot_screen_pos(player.width, player.height)

    print("\n[video_ar] Ready.  Type a command to move the robot.\n")
    print("           forward / back / left / right / stop")
    print("           arm up / arm down / open / close / wave / home / quit\n")

    try:
        while not quit_event.is_set():
            frame = player.read()
            if frame is None:
                break

            # Render robot sprite at fixed screen position
            rgb, mask = renderer._get_sprite(sprite_h)
            out = frame.copy()
            renderer._draw_shadow(out, u, v, rgb.shape[1])
            renderer._blit(out, rgb, mask,
                           u - rgb.shape[1] // 2,
                           v - rgb.shape[0])

            # Status overlay
            cmd_name = controller.current().name
            label    = f"{cmd_name}  |  {video_path.name}  |  Q=quit"
            cv2.putText(out, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 1, cv2.LINE_AA)

            # Scale up for display
            dh      = 720
            dw      = int(out.shape[1] * dh / out.shape[0])
            display = cv2.resize(out, (dw, dh))

            cv2.imshow("Autorobo — Room Navigation (Q to quit)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        renderer.close()
        player.release()
        cv2.destroyAllWindows()
        print("[video_ar] Stopped.")


if __name__ == "__main__":
    main()
