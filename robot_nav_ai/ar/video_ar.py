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
        self.status       = "IDLE"
        self.running      = False
        self.result       = None
        self.grasp_result = None   # GraspResult for Phase 8
        self._thread      = None

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

        exec_result = executor.execute(pose)
        self.result = exec_result

        # Block 6.3/6.4 — report outcome and produce Phase-8 GraspResult
        from ar.grasp_reporter import GraspReporter
        reporter      = GraspReporter()
        self.grasp_result = reporter.report(exec_result, label=target_label)

        self.status = (
            "GRASP COMPLETE ✓" if exec_result.success
            else f"FAILED: {exec_result.fail_reason}"
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


def _draw_trajectory_arc(
    frame: np.ndarray,
    robot_px: tuple,
    pre_px:   tuple,
    grs_px:   tuple,
    phase:    float,
) -> None:
    """
    Draw an animated trajectory arc on the frame in-place.

    Two segments:  robot_sprite → pre_grasp  (yellow dots)
                   pre_grasp   → grasp        (orange dots + arrow)

    phase ∈ [0, 1) drives the dot animation (use time.time() % 1.0).
    """
    H, W = frame.shape[:2]

    def _ok(p):
        return p is not None and 0 <= p[0] < W and 0 <= p[1] < H

    segments = []
    if _ok(robot_px) and _ok(pre_px):
        segments.append((robot_px, pre_px, (0, 220, 220)))   # cyan — approach
    if _ok(pre_px) and _ok(grs_px):
        segments.append((pre_px, grs_px, (0, 165, 255)))     # orange — descent

    N_DOTS = 18
    for p1, p2, col in segments:
        # Static dim guide line
        cv2.line(frame, p1, p2, (50, 50, 50), 2, cv2.LINE_AA)
        # Animated travelling dots
        for i in range(N_DOTS):
            t = (i / N_DOTS + phase) % 1.0
            px = int(p1[0] + t * (p2[0] - p1[0]))
            py = int(p1[1] + t * (p2[1] - p1[1]))
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(frame, (px, py), 3, col, -1, cv2.LINE_AA)

    # Key-point markers
    if _ok(pre_px):
        cv2.circle(frame, pre_px, 9,  (0, 220, 220), 2, cv2.LINE_AA)
        cv2.circle(frame, pre_px, 3,  (0, 220, 220), -1, cv2.LINE_AA)
        cv2.putText(frame, "PRE", (pre_px[0] + 11, pre_px[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 220), 1, cv2.LINE_AA)

    if _ok(grs_px):
        cv2.circle(frame, grs_px, 9,  (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, grs_px, 3,  (0, 165, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "GRASP", (grs_px[0] + 11, grs_px[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1, cv2.LINE_AA)
        # Approach arrow pre → grasp
        if _ok(pre_px):
            cv2.arrowedLine(frame, pre_px, grs_px,
                            (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.30)
            cv2.arrowedLine(frame, pre_px, grs_px,
                            (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.30)


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
    from robot.kinematics     import TidyBotKinematics
    from ar.transforms        import project_to_pixel

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

    # ── click-to-grasp state ──────────────────────────────────────────────────
    # _click_state[0]     = (u_frame, v_frame) in original frame coords, or None.
    # _click_xyz[0]       = (X, Y, Z) in robot base frame, or None.
    # _click_reachable[0] = True | False | None (None = no xyz yet)
    # _display_dims[0]    = (disp_w, disp_h) — updated each frame so the callback
    #                       can scale display-space clicks back to frame space.
    _click_state:     list = [None]   # [(u, v)] | [None]
    _click_xyz:       list = [None]   # [(X, Y, Z) base frame] | [None]
    _click_reachable: list = [None]   # [True | False | None]
    _display_dims:    list = [(W, H)]

    # Shared kinematics instance for the click pipeline (reachability + IK)
    print("[video_ar] Loading kinematics for click-to-grasp pipeline...")
    _kin = TidyBotKinematics()
    print("[video_ar] Kinematics ready.")

    # Plan state — populated by background thread after each new reachable click
    _click_plan:         list = [None]   # GraspPose | None
    _click_plan_status:  list = [""]     # "" | "PLANNING..." | "PLAN READY" | "PLAN FAILED: ..."
    _last_planned_click: list = [None]   # (u,v) of last triggered plan
    _plan_thread:        list = [None]   # threading.Thread | None

    # Execute state — populated by _run_grasp_click() background thread
    _exec_running:      list = [False]
    _exec_status:       list = [""]
    _exec_banner_until: list = [0.0]   # time.time() deadline for banner hold
    _exec_banner_col:   list = [(0, 0, 0)]
    _exec_trigger:      list = [False]

    def _run_grasp_click(pose) -> None:
        """Run full GraspExecutor pipeline in a daemon thread for a clicked point."""
        from robot.kinematics       import TidyBotKinematics
        from robot.robot_controller import RobotController
        from ar.grasp_executor      import GraspExecutor
        from ar.grasp_reporter      import GraspReporter

        _exec_running[0] = True
        _exec_status[0]  = "STARTING GRASP..."
        try:
            kin2     = TidyBotKinematics()
            ctrl2    = RobotController(kin2)
            executor = GraspExecutor(ctrl2, kin2)

            # Intercept state transitions for live overlay
            orig = executor._transition
            def _tracked(new_state):
                orig(new_state)
                _exec_status[0] = GraspSession._STATE_LABELS.get(
                    new_state.name, new_state.name)
            executor._transition = _tracked

            result       = executor.execute(pose)
            grasp_result = GraspReporter().report(result, label="object")

            if grasp_result.success:
                _exec_banner_col[0] = (0, 180, 0)
                _exec_status[0]     = "SUCCESS — object secured  ✓"
            else:
                reason = (grasp_result.failure_reason.name
                          .lower().replace("_", " ")
                          if grasp_result.failure_reason else "unknown")
                _exec_banner_col[0] = (0, 0, 180)
                _exec_status[0]     = f"FAILED — {reason}  ✗"

            _exec_banner_until[0] = time.time() + 3.0

        except Exception as exc:
            _exec_status[0]       = f"ERROR: {exc}"
            _exec_banner_col[0]   = (0, 0, 180)
            _exec_banner_until[0] = time.time() + 3.0
            print(f"[click] execute error: {exc}")
        finally:
            _exec_running[0] = False

    def _launch_plan(xyz_base: tuple) -> None:
        """Run IK + GraspPlanner in a daemon thread for the clicked point."""
        from ar.grasp_planner import GraspPlanner
        from ar.grasp_pose    import GraspApproach, ApproachType

        _click_plan_status[0] = "PLANNING..."
        _click_plan[0]        = None
        try:
            approach_vec = np.array([0.0, 0.0, -1.0])   # TOP_DOWN
            v            = approach_vec / np.linalg.norm(approach_vec)
            approach     = GraspApproach(
                n_hat=v, approach_vec=-v,
                approach_type=ApproachType.TOP_DOWN, confidence=1.0,
            )
            pose = GraspPlanner().plan(np.asarray(xyz_base, dtype=float), approach)
            # Verify IK is solvable for both waypoints
            ik_pre = _kin.ik(pose.pre_grasp_xyz)
            ik_grs = _kin.ik(pose.grasp_xyz)
            if not ik_pre.converged or not ik_grs.converged:
                _click_plan_status[0] = "PLAN FAILED: IK did not converge"
                return
            _click_plan[0]       = pose
            _click_plan_status[0] = "PLAN READY — click again to execute"
            print(f"[click] Plan ready  pre_grasp={np.round(pose.pre_grasp_xyz,3)}  "
                  f"grasp={np.round(pose.grasp_xyz,3)}")
        except Exception as exc:
            _click_plan_status[0] = f"PLAN FAILED: {exc}"
            print(f"[click] Plan failed: {exc}")

    WIN = "Autorobo — TidyBot Navigation (Q to quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def _on_mouse(event, cx, cy, flags, param) -> None:
        """Convert display-space click → frame-space pixel and store it."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Second click when plan is ready → execute
            if (_click_plan[0] is not None and
                    "READY" in _click_plan_status[0] and
                    not _exec_running[0]):
                _exec_trigger[0] = True
                print("[click] execute triggered")
                return
            # First click — set new target pixel
            dw, dh = _display_dims[0]
            fx = int(cx * W / dw)
            fy = int(cy * H / dh)
            fx = max(0, min(W - 1, fx))
            fy = max(0, min(H - 1, fy))
            _click_state[0] = (fx, fy)
            print(f"[click] pixel ({fx}, {fy})  →  queued for grasp")
        elif event == cv2.EVENT_RBUTTONDOWN:
            _click_state[0]       = None
            _click_xyz[0]         = None
            _click_reachable[0]   = None
            _click_plan[0]        = None
            _click_plan_status[0] = ""
            _last_planned_click[0]= None
            print("[click] cleared")

    cv2.setMouseCallback(WIN, _on_mouse)

    print("\n[video_ar] Ready.  TidyBot loaded.\n")
    print("           forward / back / left / right / stop")
    print("           arm up / arm down / open / close / wave / home / quit")
    print("           LEFT CLICK on any object to pick it up\n")

    _last_frame: list = [None]   # hold last valid frame so pause doesn't break loop

    try:
        while not quit_event.is_set():
            frame = player.read()
            if frame is None:
                # Paused or seek failed — reuse last frame so display keeps updating
                if _cur_speed[0] == 0.0 and _last_frame[0] is not None:
                    frame = _last_frame[0]
                else:
                    break
            else:
                _last_frame[0] = frame

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

            # Grasp status overlay — click-execute takes priority, then voice session
            now = time.time()
            if _exec_running[0]:
                _draw_grasp_status(out, _exec_status[0])
            elif now < _exec_banner_until[0]:
                # Timed coloured result banner (3 s)
                from ar.grasp_reporter import _draw_banner as _draw_col_banner
                _draw_col_banner(out, _exec_status[0], _exec_banner_col[0])
            elif _exec_banner_until[0] > 0 and now >= _exec_banner_until[0]:
                # Banner expired — reset click pipeline for next pick
                _exec_banner_until[0] = 0.0
                _exec_status[0]       = ""
                _click_state[0]       = None
                _click_xyz[0]         = None
                _click_reachable[0]   = None
                _click_plan[0]        = None
                _click_plan_status[0] = ""
                _last_planned_click[0]= None
            else:
                session = _grasp_session[0]
                if session is not None and (session.running or session.status not in ("IDLE", "READY")):
                    _draw_grasp_status(out, session.status)
                    if not session.running:
                        _grasp_session[0] = None

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

            # Pixel → 3D unproject for clicked point
            click = _click_state[0]
            if click is not None:
                cu, cv_ = click
                xyz_base = None

                # Primary: per-pixel depth from DepthAnything
                if use_depth and localiser is not None:
                    depth_map = localiser.latest_depth()
                    if depth_map is not None:
                        dh_dm, dw_dm = depth_map.shape[:2]
                        su = int(cu * dw_dm / W)
                        sv = int(cv_ * dh_dm / H)
                        su = max(0, min(dw_dm - 1, su))
                        sv = max(0, min(dh_dm - 1, sv))
                        d  = float(depth_map[sv, su])
                        if d > 0.05:
                            xyz_cam  = Localiser.back_project(cu, cv_, d, ar_cfg.intrinsics)
                            _candidate = Localiser.to_base_frame(xyz_cam)
                            if _kin.is_reachable(_candidate):
                                xyz_base = _candidate

                # Fallback: ray-march along click ray to find nearest reachable depth.
                # The camera is at 1.2m looking horizontally — objects the arm can
                # reach are below the camera horizon and only visible at close range.
                # Scanning d from 0.8m down to 0.15m finds the reachable depth (if any)
                # along this pixel's view ray without needing the depth estimator.
                if xyz_base is None:
                    for _d in np.linspace(0.8, 0.15, 14):
                        _xyz_cam  = Localiser.back_project(cu, cv_, _d, ar_cfg.intrinsics)
                        _candidate = Localiser.to_base_frame(_xyz_cam)
                        if _kin.is_reachable(_candidate):
                            xyz_base = _candidate
                            print(f"[click] ray-march found reachable depth {_d:.2f}m  "
                                  f"xyz={tuple(round(x,3) for x in xyz_base)}")
                            break

                if xyz_base is not None:
                    _click_xyz[0]       = xyz_base
                    _click_reachable[0] = True
                else:
                    # No reachable depth along this ray — upper frame / too far
                    _xyz_cam  = Localiser.back_project(cu, cv_, 0.8, ar_cfg.intrinsics)
                    _click_xyz[0]       = Localiser.to_base_frame(_xyz_cam)
                    _click_reachable[0] = False

            # Trigger IK + plan when click lands on a new reachable point
            _c  = _click_state[0]
            _xb = _click_xyz[0]
            if (_c is not None and _xb is not None and
                    _click_reachable[0] is True and
                    _c != _last_planned_click[0] and
                    (_plan_thread[0] is None or not _plan_thread[0].is_alive())):
                _last_planned_click[0] = _c
                _plan_thread[0] = threading.Thread(
                    target=_launch_plan, args=(_xb,), daemon=True
                )
                _plan_thread[0].start()

            # Launch execution when user clicks again on a ready plan
            if _exec_trigger[0] and not _exec_running[0] and _click_plan[0] is not None:
                _exec_trigger[0] = False
                threading.Thread(
                    target=_run_grasp_click, args=(_click_plan[0],), daemon=True
                ).start()

            # Trajectory arc overlay — shown whenever a plan is ready
            plan = _click_plan[0]
            if plan is not None:
                phase   = time.time() % 1.0
                pre_px  = project_to_pixel(plan.pre_grasp_xyz, ar_cfg.intrinsics)
                grs_px  = project_to_pixel(plan.grasp_xyz,     ar_cfg.intrinsics)
                robot_px = (u, v)   # sprite screen position
                _draw_trajectory_arc(out, robot_px, pre_px, grs_px, phase)

            # Draw click-to-grasp crosshair + 3D label + reachability colour
            click = _click_state[0]
            if click is not None:
                cu, cv_ = click
                r         = 18
                xyz       = _click_xyz[0]
                reachable = _click_reachable[0]

                # Colour: green = reachable, red = out of reach, cyan = no depth
                if reachable is True:
                    col      = (0, 220, 0)
                    reach_lbl = "REACHABLE"
                elif reachable is False:
                    col      = (0, 0, 220)
                    reach_lbl = "OUT OF REACH"
                else:
                    col      = (0, 255, 255)
                    reach_lbl = "NO DEPTH"

                cv2.circle(out, (cu, cv_), r, col, 2, cv2.LINE_AA)
                cv2.line(out, (cu - r, cv_), (cu + r, cv_), col, 1, cv2.LINE_AA)
                cv2.line(out, (cu, cv_ - r), (cu, cv_ + r), col, 1, cv2.LINE_AA)

                # XYZ label above crosshair
                if xyz is not None:
                    label3d = f"({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:.2f}) m"
                    cv2.putText(out, label3d, (cu + r + 4, cv_ - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(out, label3d, (cu + r + 4, cv_ - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)

                # Reachability label below XYZ
                cv2.putText(out, reach_lbl, (cu + r + 4, cv_ - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, reach_lbl, (cu + r + 4, cv_ - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

                # Plan status below reachability
                plan_lbl = _click_plan_status[0]
                if plan_lbl:
                    p_col = (0, 220, 0) if "READY" in plan_lbl else (0, 180, 255)
                    cv2.putText(out, plan_lbl, (cu + r + 4, cv_ + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(out, plan_lbl, (cu + r + 4, cv_ + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, p_col, 1, cv2.LINE_AA)

            # Scale up for display
            disp_h  = 720
            disp_w  = int(out.shape[1] * disp_h / out.shape[0])
            display = cv2.resize(out, (disp_w, disp_h))
            _display_dims[0] = (disp_w, disp_h)

            cv2.imshow(WIN, display)
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
