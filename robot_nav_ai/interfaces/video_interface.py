"""
interfaces/video_interface.py — Video/Webcam Robot Interface

Presents a pre-recorded video or live webcam as a BaseRobotInterface so the
rest of the pipeline (detection, planning, grasp execution) is identical whether
the camera feed is synthetic, recorded, or real.

Swap table:
    VideoInterface(path=...)       — pre-recorded room walkthrough (demo/AR)
    VideoInterface(webcam_index=0) — live webcam (demo/AR)
    MuJoCoInterface()              — physics simulation (training / offline eval)
    ROS2Interface()                — real robot hardware (deployment)

Camera feed:
    get_observation()["rgb"]   — current video frame (BGR uint8, H×W×3)

Unavailable modalities (no depth sensor or proprioception in a plain video):
    get_observation()["depth"]          — zeros (H×W float32)
    get_observation()["lidar"]          — zeros (360,  float32)
    get_observation()["proprioception"] — zeros (45,   float32)

Action interpretation (step / apply_action):
    action[0] wheel_left  }  average → playback speed
    action[1] wheel_right }  (±1 normalised → ±MAX_FRAMES frames/tick)
    action[2:]  arm / gripper — ignored (video has no actuators)

Playback mapping that matches the original PLAYBACK dict in video_ar.py:
    FORWARD      [+1, +1, …]  avg = +1.0  → +2.0 frames/tick
    BACKWARD     [−1, −1, …]  avg = −1.0  → −2.0 frames/tick
    TURN_LEFT    [ 0, +½, …]  avg = +0.25 → +0.5 frames/tick
    TURN_RIGHT   [+½,  0, …]  avg = +0.25 → +0.5 frames/tick
    STOP / ARM   [ 0,  0, …]  avg =  0.0  →  0.0 (paused)
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

from interfaces.base_interface import BaseRobotInterface

# frames/tick at normalised wheel speed = 1.0
_MAX_FRAMES_PER_TICK: float = 2.0

# observation dimensions — kept in sync with MuJoCoInterface
_LIDAR_N:   int = 360
_PROPRIO_N: int = 45


@contextlib.contextmanager
def _silent_stderr():
    """
    Redirect C-library stderr (fd 2) to /dev/null for the duration of the block.

    ffmpeg writes H264 decode warnings directly to file-descriptor 2, bypassing
    Python's logging entirely.  This is the only way to silence them.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved   = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class VideoInterface(BaseRobotInterface):
    """
    Pre-recorded video or live webcam as a BaseRobotInterface.

    Parameters
    ----------
    path : str or Path, optional
        Path to a video file.  Mutually exclusive with webcam_index.
    webcam_index : int, optional
        OpenCV device index (e.g. 0).  Mutually exclusive with path.
    cfg : any
        Optional config object (passed to BaseRobotInterface, otherwise unused).

    Properties
    ----------
    width, height : int   — frame dimensions in pixels
    fps           : float — source frame rate (webcam defaults to 30)
    source_name   : str   — human-readable label for display overlays
    frame_index   : int   — current frame position (video only; 0 for webcam)
    is_webcam     : bool
    """

    def __init__(
        self,
        *,
        path: Optional[Union[str, Path]] = None,
        webcam_index: Optional[int] = None,
        cfg: Any = None,
    ) -> None:
        super().__init__(cfg)

        if (path is None) == (webcam_index is None):
            raise ValueError("provide exactly one of: path= or webcam_index=")

        self._lock  = threading.Lock()
        self._speed: float = 0.0
        self._pos:   float = 0.0
        self._last_frame: Optional[np.ndarray] = None

        if path is not None:
            with _silent_stderr():
                self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise FileNotFoundError(f"VideoInterface: cannot open '{path}'")
            reported = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._is_webcam  = False
            self.source_name = Path(path).name
            # The file may be truncated — probe for the actual last readable frame
            # so playback never seeks into the corrupt tail region.
            self._total = self._probe_safe_total(reported)
            print(f"VideoInterface: {self.source_name} — {self._total}/{reported} frames usable")
        else:
            with _silent_stderr():
                self._cap = cv2.VideoCapture(webcam_index)
            if not self._cap.isOpened():
                raise IOError(f"VideoInterface: cannot open webcam {webcam_index}")
            self._total      = 0
            self._is_webcam  = True
            self.source_name = f"webcam:{webcam_index}"

        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._zero_depth   = np.zeros((self.height, self.width), dtype=np.float32)
        self._zero_lidar   = np.zeros(_LIDAR_N,   dtype=np.float32)
        self._zero_proprio = np.zeros(_PROPRIO_N, dtype=np.float32)

    # ── BaseRobotInterface API ────────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        """Seek to frame 0 (no-op for webcam) and return the first observation."""
        with self._lock:
            self._pos   = 0.0
            self._speed = 0.0
        if not self._is_webcam:
            with _silent_stderr():
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self.get_observation()

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Apply action (sets playback speed) and return the next observation."""
        self.apply_action(action)
        obs  = self.get_observation()
        info = {
            "success":          False,
            "collision":        False,
            "timeout":          False,
            "distance_to_goal": 0.0,
        }
        return obs, 0.0, False, info

    def get_observation(self) -> dict[str, Any]:
        """Advance the video by the current playback speed and return the frame."""
        frame = self._read_frame()
        return {
            "rgb":            frame,
            "depth":          self._zero_depth.copy(),
            "lidar":          self._zero_lidar.copy(),
            "proprioception": self._zero_proprio.copy(),
        }

    def apply_action(self, action: Any) -> None:
        """
        Translate normalised wheel velocities into video playback speed.

        speed = average(wheel_left, wheel_right) × MAX_FRAMES_PER_TICK
        """
        action = np.asarray(action, dtype=np.float32)
        if len(action) >= 2:
            avg_wheel = float(action[0] + action[1]) / 2.0
        elif len(action) == 1:
            avg_wheel = float(action[0])
        else:
            avg_wheel = 0.0
        with self._lock:
            self._speed = avg_wheel * _MAX_FRAMES_PER_TICK

    def close(self) -> None:
        """Release the video capture device."""
        self._cap.release()
        self._is_closed = True

    # ── Extra helpers used by video_ar.py ─────────────────────────────────────

    @property
    def frame_index(self) -> int:
        """Current integer frame position (always 0 for webcam)."""
        with self._lock:
            return int(self._pos)

    @property
    def playback_speed(self) -> float:
        """Current playback speed in frames/tick (read-only)."""
        with self._lock:
            return self._speed

    # ── Private helpers ───────────────────────────────────────────────────────

    def _read_frame(self) -> np.ndarray:
        """Advance position by current speed and read the frame at new position."""
        if self._is_webcam:
            with _silent_stderr():
                ok, frame = self._cap.read()
            if ok:
                self._last_frame = frame
                return frame
            return (self._last_frame
                    if self._last_frame is not None
                    else np.zeros((self.height, self.width, 3), dtype=np.uint8))

        with self._lock:
            speed = self._speed
            if speed == 0.0 and self._last_frame is not None:
                return self._last_frame      # paused — hold current frame
            total = max(1, self._total)
            self._pos = (self._pos + speed) % total
            pos = int(self._pos)

        with _silent_stderr():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = self._cap.read()

        if ok:
            self._last_frame = frame
            return frame

        # Decode failed — shrink _total so we never seek this region again
        with self._lock:
            self._total = max(1, pos)
            self._pos   = 0.0
        return (self._last_frame
                if self._last_frame is not None
                else np.zeros((self.height, self.width, 3), dtype=np.uint8))

    def _probe_safe_total(self, reported: int) -> int:
        """
        Binary-search for the actual last readable frame in a potentially
        truncated video file.  Returns a safe upper bound for _total.

        Uses _silent_stderr so the probe itself produces no H264 noise.
        """
        if reported <= 0:
            return 0

        lo, hi = 0, reported - 1
        last_good = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            with _silent_stderr():
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ok, _ = self._cap.read()
            if ok:
                last_good = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # Reset to start after probing
        with _silent_stderr():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return max(1, last_good + 1)
