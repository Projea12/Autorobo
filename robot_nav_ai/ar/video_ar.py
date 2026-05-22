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

SLAM runs in a background thread, building a sparse 3D map from the video
and tracking where in the room the robot is. A minimap overlay shows the
robot's trajectory.

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
    "TURN_LEFT"    : 0.5,
    "TURN_RIGHT"   : 0.5,
    "STOP"         : 0.0,
    "ARM_UP"       : 0.0,
    "ARM_DOWN"     : 0.0,
    "GRIPPER_OPEN" : 0.0,
    "GRIPPER_CLOSE": 0.0,
    "WAVE"         : 0.0,
    "HOME"         : 0.0,
    "PICK"         : 0.0,   # video pauses during grasp execution
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
        self._speed  = 0.0          # frames to advance per tick
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

    @property
    def frame_index(self) -> int:
        with self._lock:
            return int(self._pos)

    def release(self) -> None:
        self._cap.release()


# ── fixed robot placement ─────────────────────────────────────────────────────

def robot_screen_pos(w: int, h: int):
    """
    Robot is always at the bottom-centre of the frame.
    Returns (u, v, sprite_height_px).
    """
    u  = w // 2
    v  = int(h * 0.88)       # 88% down = near floor
    sh = int(h * 0.42)       # robot takes up ~42% of frame height
    return u, v, sh


# ── SLAM worker ───────────────────────────────────────────────────────────────

class SLAMWorker:
    """
    Runs VisualSLAM in a background thread on every new frame.
    Only starts processing when the video is playing (speed != 0).
    """

    def __init__(self, slam) -> None:
        self._slam   = slam
        self._frame  = None
        self._speed  = 0.0
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, frame: np.ndarray, speed: float) -> None:
        with self._lock:
            self._frame = frame
            self._speed = speed

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame = self._frame
                speed = self._speed

            if frame is None or speed == 0.0:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._slam.process(gray)
            time.sleep(0.033)   # ~30 fps SLAM cap


# ── grasp session ─────────────────────────────────────────────────────────────

class GraspSession:
    """
    Ties together detection → localise → plan → execute for one pick command.

    Called from the main video loop when a PICK command arrives.  The video
    is already paused (speed=0) before this is invoked.  The session runs
    the full pipeline in a background thread so the display loop can keep
    drawing the status overlay.

    Usage
    -----
        session = GraspSession(detector, localiser, ar_cfg.intrinsics)
        session.start("mug")          # non-blocking
        while session.running:
            overlay_text = session.status
        result = session.result
    """

    # Human-readable labels for each GraspState
    _STATE_LABELS = {
        "IDLE"               : "READY",
        "MOVING_TO_PREGRASP" : "MOVING TO PRE-GRASP...",
        "MOVING_TO_GRASP"    : "MOVING TO GRASP...",
        "CLOSING"            : "CLOSING GRIPPER...",
        "LIFTING"            : "LIFTING...",
        "DONE"               : "GRASP COMPLETE",
        "FAILED"             : "GRASP FAILED",
    }

    def __init__(self, detector, localiser, intrinsics) -> None:
        self._detector   = detector
        self._localiser  = localiser
        self._intrinsics = intrinsics
        self.status      = "IDLE"
        self.running     = False
        self.result      = None
        self._thread     = None

    def start(self, target_label: str) -> None:
        """Launch the grasp pipeline in a daemon thread."""
        self.running = True
        self.status  = f"SEARCHING FOR '{target_label.upper()}'..."
        self._thread = threading.Thread(
            target=self._run, args=(target_label,), daemon=True
        )
        self._thread.start()

    def _run(self, target_label: str) -> None:
        try:
            self._execute(target_label)
        except Exception as e:
            self.status  = f"ERROR: {e}"
            self.result  = None
        finally:
            self.running = False

    def _execute(self, target_label: str) -> None:
        from ar.grasp_planner   import GraspPlanner
        from ar.grasp_pose      import ApproachType, GraspApproach
        from ar.grasp_executor  import GraspExecutor, GraspState
        from ar.localiser       import Localiser
        from robot.kinematics   import TidyBotKinematics
        from robot.robot_controller import RobotController

        # ── find object ──────────────────────────────────────────────────
        self.status = f"LOCATING '{target_label.upper()}'..."
        if self._detector is None or self._localiser is None:
            self.status = "NO DETECTOR/LOCALISER — cannot grasp"
            return

        dets      = self._detector.latest
        depth_map = self._localiser.latest_depth()
        if not dets or depth_map is None:
            self.status = "NO DETECTIONS — try again when object is visible"
            return

        xyz_list = self._localiser.localise(dets, depth_map, self._intrinsics)

        # Pick the best match to target_label (highest confidence)
        target_xyz = None
        best_conf  = -1.0
        for det, xyz in zip(dets, xyz_list):
            if xyz is None:
                continue
            label_match = (target_label in det.label.lower() or
                           det.label.lower() in target_label)
            if label_match and det.confidence > best_conf:
                target_xyz = xyz
                best_conf  = det.confidence
                break

        if target_xyz is None:
            # Fall back: use highest-confidence detection regardless of label
            for det, xyz in zip(dets, xyz_list):
                if xyz is not None and det.confidence > best_conf:
                    target_xyz = xyz
                    best_conf  = det.confidence

        if target_xyz is None:
            self.status = "OBJECT NOT LOCALISED"
            return

        obj_xyz = np.array(target_xyz, dtype=float)

        # ── reachability ─────────────────────────────────────────────────
        self.status = "CHECKING REACH..."
        try:
            kin = TidyBotKinematics()
            kin.check_reachable(obj_xyz)
        except Exception as e:
            self.status = f"OUT OF REACH: {e}"
            return

        # ── plan ─────────────────────────────────────────────────────────
        self.status = "PLANNING..."
        approach_vec = np.array([0.0, 0.0, -1.0])   # downward (TOP_DOWN)
        v            = approach_vec / np.linalg.norm(approach_vec)
        approach     = GraspApproach(
            n_hat=v, approach_vec=-v,
            approach_type=ApproachType.TOP_DOWN, confidence=1.0,
        )
        pose = GraspPlanner().plan(obj_xyz, approach)

        # ── execute ───────────────────────────────────────────────────────
        ctrl     = RobotController(kin)
        executor = GraspExecutor(ctrl, kin)

        # Intercept state transitions for live overlay
        original_transition = executor._transition
        state_labels = self._STATE_LABELS

        def _tracked_transition(new_state) -> None:
            original_transition(new_state)
            label = state_labels.get(new_state.name, new_state.name)
            self.status = label

        executor._transition = _tracked_transition

        self.result = executor.execute(pose)
        self.status = (
            "GRASP COMPLETE ✓" if self.result.success
            else f"FAILED: {self.result.fail_reason}"
        )


def _draw_grasp_status(frame: np.ndarray, status: str) -> None:
    """Draw grasp status banner at the top of the frame."""
    H, W = frame.shape[:2]
    # Dark banner
    cv2.rectangle(frame, (0, 0), (W, 48), (20, 20, 20), -1)
    cv2.putText(frame, status, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, status, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 1, cv2.LINE_AA)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autorobo — TidyBot navigates through room video with SLAM"
    )
    parser.add_argument("--video", required=True,
                        help="Path to room video (e.g. video/room_video.mp4)")
    parser.add_argument("--no-slam", action="store_true",
                        help="Disable visual SLAM (faster startup)")
    parser.add_argument("--no-detect", action="store_true",
                        help="Disable object detection")
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable 3D localisation (depth estimation)")
    args = parser.parse_args()

    video_path = Path(args.video) if Path(args.video).is_absolute() \
                 else ROOT / args.video

    from ar.ar_renderer       import ARConfig, ARRenderer, CameraIntrinsics
    from ar.command_interface import CommandInterface, RobotController, Cmd
    from ar.slam              import VisualSLAM, SLAMConfig
    from ar.object_detector   import ObjectDetector
    from ar.localiser         import Localiser

    # ── open video ────────────────────────────────────────────────────────────
    player = VideoPlayer(str(video_path))
    W, H   = player.width, player.height
    print(f"[video_ar] {video_path.name}  {W}×{H}  {player.fps:.1f}fps  "
          f"{player._total} frames")

    # ── build AR renderer (TidyBot) ───────────────────────────────────────────
    ar_cfg = ARConfig(
        render_width  = W,
        render_height = H,
        intrinsics    = CameraIntrinsics(
            fx = W * 1.1,
            fy = W * 1.1,
            cx = W / 2.0,
            cy = H / 2.0,
        ),
    )
    renderer   = ARRenderer(ar_cfg)
    controller = RobotController()

    # ── SLAM ──────────────────────────────────────────────────────────────────
    use_slam = not args.no_slam
    if use_slam:
        slam_cfg = SLAMConfig(
            fx = ar_cfg.intrinsics.fx,
            fy = ar_cfg.intrinsics.fy,
            cx = ar_cfg.intrinsics.cx,
            cy = ar_cfg.intrinsics.cy,
        )
        slam       = VisualSLAM(slam_cfg)
        slam_worker = SLAMWorker(slam)
        slam_worker.start()
        print("[video_ar] Visual SLAM enabled.")
    else:
        slam = None
        print("[video_ar] SLAM disabled.")

    # ── object detector ───────────────────────────────────────────────────────
    use_detect = not args.no_detect
    if use_detect:
        detector = ObjectDetector(conf_thresh=0.30, every_n=3)
        detector.start()
        print("[video_ar] Object detector enabled (YOLOv8n).")
    else:
        detector = None

    # ── 3-D localiser (DepthAnything V2) ─────────────────────────────────────
    use_depth = use_detect and not args.no_depth
    if use_depth:
        localiser = Localiser(every_n=5)
        localiser.start()
        print("[video_ar] 3D localiser enabled (DepthAnything V2).")
    else:
        localiser = None

    # ── command interface ─────────────────────────────────────────────────────
    quit_event    = threading.Event()
    _cur_speed    = [0.0]   # shared mutable for SLAM worker
    _grasp_session: list = [None]   # [GraspSession | None]

    def on_command(cmd: Cmd, raw_text: str = "") -> None:
        controller.set_command(cmd)
        speed = PLAYBACK.get(cmd.name, 0.0)
        player.set_speed(speed)
        _cur_speed[0] = speed
        direction = ("playing forward" if speed > 0
                     else "playing backward" if speed < 0 else "paused")
        print(f"[video_ar] {cmd.name}  →  {direction}")

        if cmd == Cmd.PICK:
            from ar.command_interface import parse_pick_target
            target = parse_pick_target(raw_text) if raw_text else "object"
            print(f"[video_ar] PICK command — target: '{target}'")
            session = GraspSession(detector if use_detect else None,
                                   localiser if use_depth else None,
                                   ar_cfg.intrinsics)
            _grasp_session[0] = session
            session.start(target)

    cmd_iface = CommandInterface(on_command=on_command, on_quit=quit_event.set)
    renderer.start_physics(controller=controller)
    cmd_iface.start()

    # ── fixed robot position ──────────────────────────────────────────────────
    u, v, sprite_h = robot_screen_pos(W, H)

    print("\n[video_ar] Ready.  TidyBot loaded.\n")
    print("           forward / back / left / right / stop")
    print("           arm up / arm down / open / close / wave / home / quit\n")

    try:
        while not quit_event.is_set():
            frame = player.read()
            if frame is None:
                break

            # Feed frame to SLAM, detector and localiser (non-blocking)
            if use_slam:
                slam_worker.update(frame, _cur_speed[0])
            if use_detect and detector is not None:
                detector.update(frame)
            if use_depth and localiser is not None:
                localiser.update(frame)

            # Render TidyBot sprite at fixed screen position (lock prevents
            # physics thread corrupting MjData mid-render)
            with renderer._physics_lock:
                rgb, mask = renderer._get_sprite(sprite_h)
            out = frame.copy()
            renderer._draw_shadow(out, u, v, rgb.shape[1])
            renderer._blit(out, rgb, mask,
                           u - rgb.shape[1] // 2,
                           v - rgb.shape[0])

            # Draw detection boxes + 3D positions (before minimap)
            if use_detect and detector is not None:
                out = detector.draw(out)
                if use_depth and localiser is not None:
                    depth_map = localiser.latest_depth()
                    if depth_map is not None:
                        dets = detector.latest
                        xyz_list = localiser.localise(dets, depth_map, ar_cfg.intrinsics)
                        Localiser.draw_3d(out, dets, xyz_list)

            # SLAM minimap overlay
            if use_slam and slam is not None:
                slam.draw_minimap(out)

            # Grasp status overlay (takes priority when session is active)
            session = _grasp_session[0]
            if session is not None and (session.running or session.status not in ("IDLE", "READY")):
                _draw_grasp_status(out, session.status)
                if not session.running:
                    _grasp_session[0] = None   # clear after done

            # Status overlay
            cmd_name = controller.current().name
            if use_slam and slam is not None:
                rx, rz = slam.position_xz
                status = f"{cmd_name}  |  pos ({rx:.1f},{rz:.1f})m  |  Q=quit"
            else:
                status = f"{cmd_name}  |  {video_path.name}  |  Q=quit"

            cv2.putText(out, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 1, cv2.LINE_AA)

            # Scale up for display
            dh      = 720
            dw      = int(out.shape[1] * dh / out.shape[0])
            display = cv2.resize(out, (dw, dh))

            cv2.imshow("Autorobo — TidyBot Navigation (Q to quit)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if use_slam:
            slam_worker.stop()
        if use_detect and detector is not None:
            detector.stop()
        if use_depth and localiser is not None:
            localiser.stop()
        renderer.close()
        player.release()
        cv2.destroyAllWindows()
        print("[video_ar] Stopped.")


if __name__ == "__main__":
    main()
