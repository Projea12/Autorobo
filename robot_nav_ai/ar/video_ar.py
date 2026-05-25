"""
ar/video_ar.py — Robot navigates through a room via a BaseRobotInterface.

Camera feed, playback, and physics all come from the active interface:

    VideoInterface(path=...)       — pre-recorded walkthrough (default demo)
    VideoInterface(webcam_index=0) — live webcam
    MuJoCoInterface()              — full physics simulation

Commands control how the robot moves:

    forward / back           → video plays forward / backward
    left  / right            → slow crawl + turn animation
    stop                     → video pauses
    arm up / arm down        → arm raises / lowers, video pauses
    open / close             → gripper opens / closes, video pauses
    wave                     → robot waves, video pauses
    home                     → reset arm, video pauses
    LEFT CLICK on any object → click-to-grasp (plan then execute on second click)
    RIGHT CLICK              → clear click target
    quit / Q                 → exit

SLAM runs in a background thread, building a sparse 3D map and tracking
where in the room the robot is. A minimap overlay shows the trajectory.

Usage:
    python ar/video_ar.py --video video/room_video.mp4
    python ar/video_ar.py --webcam 0
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


# ── command → normalised action mapping ───────────────────────────────────────
#
# Action format: [wheel_left, wheel_right, arm_j1..j6, gripper] ∈ [−1, 1]
#
# VideoInterface interprets wheel average as playback speed:
#   avg = +1.0 → +2.0 frames/tick  (FORWARD)
#   avg = +0.25 → +0.5 frames/tick (TURN)
#   avg =  0.0 →  0.0 frames/tick  (STOP / ARM / GRIPPER)
#
# MuJoCoInterface passes the vector straight to ManipulationEnv actuators.

_Z = np.zeros(9, dtype=np.float32)

def _cmd_action(wl: float, wr: float) -> np.ndarray:
    a = np.zeros(9, dtype=np.float32)
    a[0], a[1] = wl, wr
    return a


# ── SLAM worker ───────────────────────────────────────────────────────────────

class SLAMWorker:
    """Runs VisualSLAM in a background thread on every new frame."""

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
            time.sleep(0.033)


# ── fixed robot placement ─────────────────────────────────────────────────────

def robot_screen_pos(w: int, h: int):
    """Robot at bottom-centre of the frame. Returns (u, v, sprite_height_px)."""
    u  = w // 2
    v  = int(h * 0.88)
    sh = int(h * 0.42)
    return u, v, sh


# ── grasp session (voice / text command path) ─────────────────────────────────

class GraspSession:
    """
    Ties together detection → localise → plan → execute for one PICK command.
    Runs in a background thread so the display loop keeps drawing the overlay.
    """

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
        self.grasp_result = None
        self._thread      = None

    def start(self, target_label: str) -> None:
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
        from ar.grasp_executor  import GraspExecutor
        from ar.localiser       import Localiser
        from robot.kinematics   import TidyBotKinematics
        from robot.robot_controller import RobotController

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
            for det, xyz in zip(dets, xyz_list):
                if xyz is not None and det.confidence > best_conf:
                    target_xyz = xyz
                    best_conf  = det.confidence

        if target_xyz is None:
            self.status = "OBJECT NOT LOCALISED"
            return

        obj_xyz = np.array(target_xyz, dtype=float)

        self.status = "CHECKING REACH..."
        try:
            kin = TidyBotKinematics()
            kin.check_reachable(obj_xyz)
        except Exception as e:
            self.status = f"OUT OF REACH: {e}"
            return

        self.status = "PLANNING..."
        approach_vec = np.array([0.0, 0.0, -1.0])
        v            = approach_vec / np.linalg.norm(approach_vec)
        approach     = GraspApproach(
            n_hat=v, approach_vec=-v,
            approach_type=ApproachType.TOP_DOWN, confidence=1.0,
        )
        pose = GraspPlanner().plan(obj_xyz, approach)

        ctrl     = RobotController(kin)
        executor = GraspExecutor(ctrl, kin)

        original_transition = executor._transition
        state_labels = self._STATE_LABELS

        def _tracked_transition(new_state) -> None:
            original_transition(new_state)
            self.status = state_labels.get(new_state.name, new_state.name)

        executor._transition = _tracked_transition
        exec_result = executor.execute(pose)
        self.result = exec_result

        from ar.grasp_reporter import GraspReporter
        reporter      = GraspReporter()
        self.grasp_result = reporter.report(exec_result, label=target_label)

        self.status = (
            "GRASP COMPLETE [OK]" if exec_result.success
            else f"FAILED: {exec_result.fail_reason}"
        )


# ── overlay helpers ───────────────────────────────────────────────────────────

def _draw_grasp_status(frame: np.ndarray, status: str) -> None:
    H, W = frame.shape[:2]
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
    H, W = frame.shape[:2]

    def _ok(p):
        return p is not None and 0 <= p[0] < W and 0 <= p[1] < H

    segments = []
    if _ok(robot_px) and _ok(pre_px):
        segments.append((robot_px, pre_px, (0, 220, 220)))
    if _ok(pre_px) and _ok(grs_px):
        segments.append((pre_px, grs_px, (0, 165, 255)))

    N_DOTS = 18
    for p1, p2, col in segments:
        cv2.line(frame, p1, p2, (50, 50, 50), 2, cv2.LINE_AA)
        for i in range(N_DOTS):
            t  = (i / N_DOTS + phase) % 1.0
            px = int(p1[0] + t * (p2[0] - p1[0]))
            py = int(p1[1] + t * (p2[1] - p1[1]))
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(frame, (px, py), 3, col, -1, cv2.LINE_AA)

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
        if _ok(pre_px):
            cv2.arrowedLine(frame, pre_px, grs_px,
                            (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.30)
            cv2.arrowedLine(frame, pre_px, grs_px,
                            (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.30)


# ── main ──────────────────────────────────────────────────────────────────────

def main(interface=None) -> None:
    """
    Run the AR navigation loop.

    Parameters
    ----------
    interface : BaseRobotInterface, optional
        Any interface implementation.  If None, one is created from CLI args.
        Pass a MuJoCoInterface for physics-backed execution, or a VideoInterface
        for video/webcam mode.
    """
    parser = argparse.ArgumentParser(
        description="Autorobo — TidyBot navigates through a room"
    )
    parser.add_argument("--video", default=None,
                        help="Path to room video (e.g. video/room_video.mp4)")
    parser.add_argument("--webcam", type=int, default=None, metavar="INDEX",
                        help="Use webcam instead of video file (e.g. --webcam 0)")
    parser.add_argument("--no-slam",    action="store_true", help="Disable SLAM")
    parser.add_argument("--no-detect",  action="store_true", help="Disable YOLO detection")
    parser.add_argument("--no-depth",   action="store_true", help="Disable 3D localisation")
    parser.add_argument("--no-simview", action="store_true", help="Disable MuJoCo sim panel")
    args = parser.parse_args()

    from ar.ar_renderer       import ARConfig, ARRenderer, CameraIntrinsics
    from ar.command_interface import CommandInterface, RobotController as CmdController, Cmd
    from ar.slam              import VisualSLAM, SLAMConfig
    from ar.object_detector   import ObjectDetector
    from ar.localiser         import Localiser
    from robot.kinematics     import TidyBotKinematics
    from ar.transforms        import project_to_pixel
    from interfaces.video_interface import VideoInterface

    # ── build or accept interface ─────────────────────────────────────────────
    _owns_interface = interface is None
    if interface is None:
        if args.video is None and args.webcam is None:
            parser.error("provide --video <path> or --webcam <index>")

        if args.webcam is not None:
            interface = VideoInterface(webcam_index=args.webcam)
            print(f"[video_ar] Webcam {args.webcam}  "
                  f"{interface.width}×{interface.height}")
        else:
            video_path = (Path(args.video) if Path(args.video).is_absolute()
                          else ROOT / args.video)
            interface = VideoInterface(path=video_path)
            print(f"[video_ar] {interface.source_name}  "
                  f"{interface.width}×{interface.height}  {interface.fps:.1f} fps")

    W, H = interface.width, interface.height

    # ── AR renderer ───────────────────────────────────────────────────────────
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
    renderer      = ARRenderer(ar_cfg)
    cmd_controller = CmdController()

    # ── SLAM ──────────────────────────────────────────────────────────────────
    use_slam = not args.no_slam
    if use_slam:
        slam_cfg = SLAMConfig(
            fx=ar_cfg.intrinsics.fx, fy=ar_cfg.intrinsics.fy,
            cx=ar_cfg.intrinsics.cx, cy=ar_cfg.intrinsics.cy,
        )
        slam        = VisualSLAM(slam_cfg)
        slam_worker = SLAMWorker(slam)
        slam_worker.start()
        print("[video_ar] Visual SLAM enabled.")
    else:
        slam = None
        print("[video_ar] SLAM disabled.")

    # ── object detector ───────────────────────────────────────────────────────
    use_detect = not args.no_detect
    if use_detect:
        detector = ObjectDetector(weights="yolov8l-worldv2.pt",
                                  conf_thresh=0.25, every_n=3)
        detector.start()
        print("[video_ar] Object detector enabled (YOLO-World v2 large).")
    else:
        detector = None

    # ── 3-D localiser ─────────────────────────────────────────────────────────
    use_depth = use_detect and not args.no_depth
    if use_depth:
        localiser = Localiser(every_n=5)
        localiser.start()
        print("[video_ar] 3D localiser enabled (DepthAnything V2).")
    else:
        localiser = None

    # ── command interface ─────────────────────────────────────────────────────
    quit_event    = threading.Event()
    _cur_speed    = [0.0]   # scalar playback speed for SLAM worker
    _grasp_session: list = [None]

    # Maps each Cmd to a 9-dim normalised action vector.
    # Wheel avg → VideoInterface playback speed (matches original PLAYBACK dict).
    # Arm/gripper dims are used by MuJoCoInterface; VideoInterface ignores them.
    _CMD_ACTIONS = {
        Cmd.FORWARD      : _cmd_action( 1.0,  1.0),   # avg=+1.0 → +2.0 fps
        Cmd.BACKWARD     : _cmd_action(-1.0, -1.0),   # avg=−1.0 → −2.0 fps
        Cmd.TURN_LEFT    : _cmd_action( 0.0,  0.5),   # avg=+0.25 → +0.5 fps
        Cmd.TURN_RIGHT   : _cmd_action( 0.5,  0.0),   # avg=+0.25 → +0.5 fps
        # All following commands pause the video (zero wheels) while actuating
        Cmd.STOP         : _Z.copy(),
        Cmd.ARM_UP       : _Z.copy(),
        Cmd.ARM_DOWN     : _Z.copy(),
        Cmd.GRIPPER_OPEN : _Z.copy(),
        Cmd.GRIPPER_CLOSE: _Z.copy(),
        Cmd.WAVE         : _Z.copy(),
        Cmd.HOME         : _Z.copy(),
        Cmd.PICK         : _Z.copy(),
    }

    def on_command(cmd: Cmd, raw_text: str = "") -> None:
        cmd_controller.set_command(cmd)
        action = _CMD_ACTIONS.get(cmd, _Z)
        interface.apply_action(action)
        # Track playback speed for SLAM worker (zero = don't process SLAM)
        _cur_speed[0] = float((action[0] + action[1]) / 2.0 * 2.0)  # frames/tick

        if cmd == Cmd.PICK:
            from ar.command_interface import parse_pick_target
            target = parse_pick_target(raw_text) if raw_text else "object"
            print(f"[video_ar] PICK command — target: '{target}'")
            session = GraspSession(
                detector  if use_detect else None,
                localiser if use_depth  else None,
                ar_cfg.intrinsics,
            )
            _grasp_session[0] = session
            session.start(target)

    cmd_iface = CommandInterface(on_command=on_command, on_quit=quit_event.set)
    renderer.start_physics(controller=cmd_controller)
    cmd_iface.start()

    # ── fixed robot screen position ───────────────────────────────────────────
    u, v, sprite_h = robot_screen_pos(W, H)

    # ── click-to-grasp state ──────────────────────────────────────────────────
    _click_state:     list = [None]
    _click_xyz:       list = [None]
    _click_reachable: list = [None]
    _display_dims:    list = [(W, H)]

    print("[video_ar] Loading kinematics for click-to-grasp pipeline...")
    _kin = TidyBotKinematics()
    print("[video_ar] Kinematics ready.")

    _click_plan:         list = [None]
    _click_plan_status:  list = [""]
    _last_planned_click: list = [None]
    _plan_thread:        list = [None]

    _exec_running:      list = [False]
    _exec_status:       list = [""]
    _exec_banner_until: list = [0.0]
    _exec_banner_col:   list = [(0, 0, 0)]
    _exec_trigger:      list = [False]

    def _run_grasp_click(pose) -> None:
        """
        Run the full grasp pipeline in a background thread for a clicked point.

        At each state transition the active interface is stepped with a zero-
        action vector.  For VideoInterface this keeps the video paused.  For
        MuJoCoInterface it advances physics by one step so the simulation stays
        in sync with the executor's kinematic progress.
        """
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

            # Mirror arm state → AR renderer panel at 30 fps
            _mirror_stop = threading.Event()
            def _mirror_loop():
                while not _mirror_stop.is_set():
                    renderer.mirror_arm_state(ctrl2.get_joints(), ctrl2._gripper)
                    time.sleep(1 / 30)
            threading.Thread(target=_mirror_loop, daemon=True).start()

            # Intercept state transitions:
            #  1. update the status overlay
            #  2. step the interface so it stays in sync (pauses video / advances physics)
            _zero_action = np.zeros(9, dtype=np.float32)
            orig = executor._transition
            def _tracked(new_state):
                orig(new_state)
                _exec_status[0] = GraspSession._STATE_LABELS.get(
                    new_state.name, new_state.name)
                interface.step(_zero_action)   # ← interface.step() called here

            executor._transition = _tracked

            result       = executor.execute(pose)
            _mirror_stop.set()
            grasp_result = GraspReporter().report(result, label="object")

            if grasp_result.success:
                _exec_banner_col[0] = (0, 180, 0)
                _exec_status[0]     = "SUCCESS — object secured  [OK]"
            else:
                reason = (grasp_result.failure_reason.name
                          .lower().replace("_", " ")
                          if grasp_result.failure_reason else "unknown")
                _exec_banner_col[0] = (0, 0, 180)
                _exec_status[0]     = f"FAILED — {reason}  [X]"

            _exec_banner_until[0] = time.time() + 3.0

        except Exception as exc:
            _mirror_stop.set()
            _exec_status[0]       = f"ERROR: {exc}"
            _exec_banner_col[0]   = (0, 0, 180)
            _exec_banner_until[0] = time.time() + 3.0
            print(f"[click] execute error: {exc}")
        finally:
            _exec_running[0] = False

    def _launch_plan(xyz_base: tuple, click_uv: tuple,
                     frame_snap: np.ndarray) -> None:
        """Run IK + GraspPlanner in a background thread for the clicked point."""
        from ar.grasp_planner import GraspPlanner
        from ar.grasp_pose    import GraspApproach, ApproachType

        _click_plan_status[0] = "PLANNING..."
        _click_plan[0]        = None

        cu, cv_ = click_uv
        crop_bgr = None
        if frame_snap is not None:
            fh, fw = frame_snap.shape[:2]
            best_bbox = None
            if use_detect and detector is not None:
                for det in detector.latest:
                    x1, y1, x2, y2 = det.bbox_xyxy
                    if x1 <= cu <= x2 and y1 <= cv_ <= y2:
                        best_bbox = det.bbox_xyxy
                        break
            if best_bbox is not None:
                x1, y1, x2, y2 = best_bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x2), min(fh, y2)
                if x2 > x1 and y2 > y1:
                    crop_bgr = frame_snap[y1:y2, x1:x2].copy()
            else:
                r = 40
                x1, y1 = max(0, cu-r), max(0, cv_-r)
                x2, y2 = min(fw, cu+r), min(fh, cv_+r)
                if x2 > x1 and y2 > y1:
                    crop_bgr = frame_snap[y1:y2, x1:x2].copy()
        renderer.set_target_crop(crop_bgr)

        try:
            approach_vec = np.array([0.0, 0.0, -1.0])
            v            = approach_vec / np.linalg.norm(approach_vec)
            approach     = GraspApproach(
                n_hat=v, approach_vec=-v,
                approach_type=ApproachType.TOP_DOWN, confidence=1.0,
            )
            pose      = GraspPlanner().plan(np.asarray(xyz_base, dtype=float), approach)
            ik_pre    = _kin.ik(pose.pre_grasp_xyz)
            ik_grs    = _kin.ik(pose.grasp_xyz)
            if not ik_pre.converged or not ik_grs.converged:
                _click_plan_status[0] = "PLAN FAILED: IK did not converge"
                return
            _click_plan[0]        = pose
            _click_plan_status[0] = "PLAN READY — click again to execute"
            renderer.set_target_object_pos(xyz_base)
            print(f"[click] Plan ready  pre_grasp={np.round(pose.pre_grasp_xyz,3)}  "
                  f"grasp={np.round(pose.grasp_xyz,3)}")
        except Exception as exc:
            _click_plan_status[0] = f"PLAN FAILED: {exc}"
            print(f"[click] Plan failed: {exc}")

    WIN = "Autorobo | TidyBot Navigation (Q to quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def _on_mouse(event, cx, cy, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if (_click_plan[0] is not None and
                    "READY" in _click_plan_status[0] and
                    not _exec_running[0]):
                _exec_trigger[0] = True
                print("[click] execute triggered")
                return
            dw, dh = _display_dims[0]
            fx = max(0, min(W - 1, int(cx * W / dw)))
            fy = max(0, min(H - 1, int(cy * H / dh)))
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

    print(f"\n[video_ar] Ready.  Interface: {interface.__class__.__name__}\n")
    print("           forward / back / left / right / stop")
    print("           arm up / arm down / open / close / wave / home / quit")
    print("           LEFT CLICK on any object to pick it up\n")

    _last_frame: list = [None]

    try:
        while not quit_event.is_set():
            # ── get camera frame from interface ───────────────────────────────
            obs   = interface.get_observation()
            frame = obs["rgb"]

            if frame is None:
                break
            _last_frame[0] = frame

            # ── feed frame to background workers ─────────────────────────────
            if use_slam:
                slam_worker.update(frame, _cur_speed[0])
            if use_detect and detector is not None:
                detector.update(frame)
            if use_depth and localiser is not None:
                localiser.update(frame)

            # ── AR overlay ────────────────────────────────────────────────────
            out = frame.copy()
            is_video_interface = hasattr(interface, "is_webcam")
            if is_video_interface and not interface.is_webcam:
                with renderer._physics_lock:
                    rgb, mask = renderer._get_sprite(sprite_h)
                renderer._draw_shadow(out, u, v, rgb.shape[1])
                renderer._blit(out, rgb, mask,
                               u - rgb.shape[1] // 2,
                               v - rgb.shape[0])

            # ── detection + 3D positions ──────────────────────────────────────
            if use_detect and detector is not None:
                out = detector.draw(out)
                if use_depth and localiser is not None:
                    depth_map = localiser.latest_depth()
                    if depth_map is not None:
                        dets     = detector.latest
                        xyz_list = localiser.localise(dets, depth_map, ar_cfg.intrinsics)
                        Localiser.draw_3d(out, dets, xyz_list)

            # ── SLAM minimap ──────────────────────────────────────────────────
            if use_slam and slam is not None:
                slam.draw_minimap(out)

            # ── grasp status overlay ──────────────────────────────────────────
            now = time.time()
            if _exec_running[0]:
                _draw_grasp_status(out, _exec_status[0])
            elif now < _exec_banner_until[0]:
                from ar.grasp_reporter import _draw_banner as _draw_col_banner
                _draw_col_banner(out, _exec_status[0], _exec_banner_col[0])
            elif _exec_banner_until[0] > 0 and now >= _exec_banner_until[0]:
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
                if session is not None and (session.running or
                        session.status not in ("IDLE", "READY")):
                    _draw_grasp_status(out, session.status)
                    if not session.running:
                        _grasp_session[0] = None

            # ── status bar ────────────────────────────────────────────────────
            cmd_name = cmd_controller.current().name
            if use_slam and slam is not None:
                rx, rz = slam.position_xz
                status = f"{cmd_name}  |  pos ({rx:.1f},{rz:.1f})m  |  Q=quit"
            else:
                status = (f"{cmd_name}  |  {interface.source_name}  |  Q=quit")

            cv2.putText(out, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 1, cv2.LINE_AA)

            # ── click → 3D ───────────────────────────────────────────────────
            click = _click_state[0]
            if click is not None:
                cu, cv_ = click
                xyz_base = None

                if use_depth and localiser is not None:
                    depth_map = localiser.latest_depth()
                    if depth_map is not None:
                        dh_dm, dw_dm = depth_map.shape[:2]
                        su = max(0, min(dw_dm-1, int(cu * dw_dm / W)))
                        sv = max(0, min(dh_dm-1, int(cv_ * dh_dm / H)))
                        d  = float(depth_map[sv, su])
                        if d > 0.05:
                            xyz_cam    = Localiser.back_project(cu, cv_, d, ar_cfg.intrinsics)
                            _candidate = Localiser.to_base_frame(xyz_cam)
                            if _kin.is_reachable(_candidate):
                                xyz_base = _candidate

                if xyz_base is None:
                    X_base = ((cu - W / 2.0) / (W / 2.0)) * 0.35
                    Z_base = 0.65 - (cv_ / H) * 0.40
                    Y_base = 0.50
                    _cand  = (X_base, Y_base, Z_base)
                    if _kin.is_reachable(_cand):
                        xyz_base = _cand
                        print(f"[click] workspace map  px=({cu},{cv_})  "
                              f"xyz={tuple(round(x, 3) for x in xyz_base)}")

                if xyz_base is not None:
                    _click_xyz[0]       = xyz_base
                    _click_reachable[0] = True
                else:
                    _click_xyz[0]       = (0.0, 0.5, 0.5)
                    _click_reachable[0] = False

            # ── trigger plan when new reachable click arrives ─────────────────
            _c  = _click_state[0]
            _xb = _click_xyz[0]
            if (_c is not None and _xb is not None and
                    _click_reachable[0] is True and
                    _c != _last_planned_click[0] and
                    (_plan_thread[0] is None or not _plan_thread[0].is_alive())):
                _last_planned_click[0] = _c
                _plan_thread[0] = threading.Thread(
                    target=_launch_plan,
                    args=(_xb, _c, _last_frame[0]),
                    daemon=True
                )
                _plan_thread[0].start()

            # ── trigger execution on second click ─────────────────────────────
            if _exec_trigger[0] and not _exec_running[0] and _click_plan[0] is not None:
                _exec_trigger[0] = False
                threading.Thread(
                    target=_run_grasp_click,
                    args=(_click_plan[0],),
                    daemon=True,
                ).start()

            # ── trajectory arc ────────────────────────────────────────────────
            plan = _click_plan[0]
            if plan is not None:
                phase    = time.time() % 1.0
                pre_px   = project_to_pixel(plan.pre_grasp_xyz, ar_cfg.intrinsics)
                grs_px   = project_to_pixel(plan.grasp_xyz,     ar_cfg.intrinsics)
                robot_px = (u, v)
                _draw_trajectory_arc(out, robot_px, pre_px, grs_px, phase)

            # ── crosshair + reachability label ────────────────────────────────
            click = _click_state[0]
            if click is not None:
                cu, cv_ = click
                r         = 18
                xyz       = _click_xyz[0]
                reachable = _click_reachable[0]

                if reachable is True:
                    col, reach_lbl = (0, 220, 0),   "REACHABLE"
                elif reachable is False:
                    col, reach_lbl = (0, 0, 220),   "OUT OF REACH"
                else:
                    col, reach_lbl = (0, 255, 255), "NO DEPTH"

                cv2.circle(out, (cu, cv_), r, col, 2, cv2.LINE_AA)
                cv2.line(out, (cu-r, cv_), (cu+r, cv_), col, 1, cv2.LINE_AA)
                cv2.line(out, (cu, cv_-r), (cu, cv_+r), col, 1, cv2.LINE_AA)

                if xyz is not None:
                    label3d = f"({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:.2f}) m"
                    cv2.putText(out, label3d, (cu+r+4, cv_-18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(out, label3d, (cu+r+4, cv_-18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)

                cv2.putText(out, reach_lbl, (cu+r+4, cv_-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(out, reach_lbl, (cu+r+4, cv_-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

                plan_lbl = _click_plan_status[0]
                if plan_lbl:
                    p_col = (0, 220, 0) if "READY" in plan_lbl else (0, 180, 255)
                    cv2.putText(out, plan_lbl, (cu+r+4, cv_+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(out, plan_lbl, (cu+r+4, cv_+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, p_col, 1, cv2.LINE_AA)

            # ── scale + split-screen sim panel ────────────────────────────────
            disp_h = 720
            disp_w = int(out.shape[1] * disp_h / out.shape[0])
            left   = cv2.resize(out, (disp_w, disp_h))

            if not args.no_simview:
                sim_w   = int(disp_h * 4 / 3)
                sim_bgr = renderer.render_sim_panel(sim_w, disp_h)
                cv2.putText(sim_bgr, "MuJoCo Simulation",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(sim_bgr, "MuJoCo Simulation",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (200, 200, 200), 1, cv2.LINE_AA)
                div = np.full((disp_h, 3, 3), 60, dtype=np.uint8)
                display = np.concatenate([left, div, sim_bgr], axis=1)
            else:
                display = left

            _display_dims[0] = (display.shape[1], disp_h)

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
        if _owns_interface:
            interface.close()
        cv2.destroyAllWindows()
        print("[video_ar] Stopped.")


if __name__ == "__main__":
    main()
