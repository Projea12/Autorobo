"""
ar/ar_renderer.py — AR overlay: renders the MuJoCo robot into a live webcam frame.

Pipeline per frame:
  1. Render robot sprite from a fixed 3/4-angle virtual camera (MuJoCo offscreen)
  2. Use depth buffer to extract robot-only pixels (ground plane hidden)
  3. Project robot's 3D world position through real camera intrinsics → screen (u, v)
  4. Scale sprite by perspective depth
  5. Draw soft shadow ellipse at (u, v)
  6. Alpha-blend robot onto webcam frame

Usage:
    python ar/ar_renderer.py              # live webcam AR preview
    python ar/ar_renderer.py --no-preview # headless
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # robot_nav_ai/


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0


@dataclass
class ARConfig:
    xml_path:      Path             = ROOT / "robot" / "tidybot" / "ar_scene.xml"
    render_width:  int              = 640
    render_height: int              = 480
    intrinsics:    CameraIntrinsics = field(default_factory=CameraIntrinsics)

    # Where the robot stands in the AR world (OpenCV world frame, metres)
    # x=0 centre, y=1.6 below camera (floor level), z=2.5 in front
    robot_world_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.6, 1.8])
    )

    # Virtual camera for TidyBot sprite (Z-up; robot centre ≈ z=0.55)
    sprite_lookat:    np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.55])
    )
    sprite_distance:  float = 2.8
    sprite_azimuth:   float = -30.0
    sprite_elevation: float = -18.0

    # Rendering
    robot_height_m:   float = 1.3    # TidyBot with retracted arm
    shadow_opacity:   float = 0.50
    robot_opacity:    float = 0.92   # blend factor for robot pixels
    keyframe:         str   = "retract"
    depth_far_thresh: float = 60.0   # depth (MuJoCo units) above = background


# ── renderer ──────────────────────────────────────────────────────────────────

class ARRenderer:
    """
    Composites the MuJoCo robot model into a BGR webcam frame.

    Parameters
    ----------
    cfg : ARConfig
    """

    def __init__(self, cfg: ARConfig = ARConfig()) -> None:
        import mujoco
        self._mj = mujoco
        self.cfg = cfg

        if not cfg.xml_path.exists():
            raise FileNotFoundError(f"robot.xml not found: {cfg.xml_path}")

        print(f"[ARRenderer] Loading {cfg.xml_path.name} ...")
        self._model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        self._data  = mujoco.MjData(self._model)

        key_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_KEY, cfg.keyframe
        )
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
        mujoco.mj_forward(self._model, self._data)

        self._renderer = mujoco.Renderer(
            self._model, cfg.render_height, cfg.render_width
        )

        # Virtual camera for sprite rendering
        self._sprite_cam = mujoco.MjvCamera()
        self._sprite_cam.lookat[:]  = cfg.sprite_lookat
        self._sprite_cam.distance   = cfg.sprite_distance
        self._sprite_cam.azimuth    = cfg.sprite_azimuth
        self._sprite_cam.elevation  = cfg.sprite_elevation

        # Hide ground plane, floor geom, and static scene elements
        self._opt = mujoco.MjvOption()
        self._opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC]    = False
        self._opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = False

        # Cache the sprite (re-rendered on pose change)
        self._sprite_rgb:  Optional[np.ndarray] = None
        self._sprite_mask: Optional[np.ndarray] = None
        self._sprite_dirty = True

        # Track last virtual camera angles to avoid redundant re-renders
        self._last_az  = cfg.sprite_azimuth
        self._last_el  = cfg.sprite_elevation
        self._az_thresh = 3.0   # degrees change needed to re-render

        # Physics thread state
        self._physics_lock   = threading.Lock()
        self._physics_stop   = threading.Event()
        self._physics_thread: Optional[threading.Thread] = None

        print("[ARRenderer] Ready.")

    def start_physics(
        self,
        fps: float = 30.0,
        controller=None,
    ) -> None:
        """Start MuJoCo physics in a background thread."""
        if self._physics_thread and self._physics_thread.is_alive():
            return
        self._controller = controller
        self._physics_stop.clear()
        self._physics_thread = threading.Thread(
            target=self._physics_loop, args=(fps,), daemon=True
        )
        self._physics_thread.start()
        print("[ARRenderer] Physics thread started.")

    def stop_physics(self) -> None:
        self._physics_stop.set()

    def _physics_loop(self, fps: float) -> None:
        mj    = self._mj
        model = self._model
        dt    = 1.0 / fps

        while not self._physics_stop.is_set():
            t0 = time.perf_counter()

            with self._physics_lock:
                if self._controller is not None:
                    self._controller.apply(
                        self._data.ctrl, float(self._data.time)
                    )
                mj.mj_step(model, self._data)

            self._sprite_dirty = True

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    # ── public API ────────────────────────────────────────────────────────────

    def set_robot_qpos(self, qpos: np.ndarray) -> None:
        """Update robot joint positions (e.g. from MuJoCo physics)."""
        self._data.qpos[:len(qpos)] = qpos
        self._mj.mj_forward(self._model, self._data)
        self._sprite_dirty = True

    def composite(
        self,
        frame_bgr: np.ndarray,
        camera_pose: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render robot and blend into frame_bgr.

        Parameters
        ----------
        frame_bgr    : H×W×3 uint8 webcam frame
        camera_pose  : 4×4 world-to-camera matrix (from CameraTracker),
                       or None to use identity (robot centred in frame)
        """
        if camera_pose is None:
            camera_pose = np.eye(4)

        # Update virtual camera angle to match real camera rotation
        self._update_sprite_camera(camera_pose)

        # Project robot world position onto screen
        u, v, depth_z = self._project(camera_pose)

        if depth_z <= 0.05:
            return frame_bgr

        # Scale sprite so robot occupies correct pixel height
        cfg = self.cfg
        K   = cfg.intrinsics
        sprite_h_px = int(K.fy * cfg.robot_height_m / depth_z)
        sprite_h_px = max(40, min(sprite_h_px, frame_bgr.shape[0]))

        with self._physics_lock:
            rgb, mask = self._get_sprite(sprite_h_px)

        out = frame_bgr.copy()
        self._draw_shadow(out, u, v, rgb.shape[1])
        self._blit(out, rgb, mask, u - rgb.shape[1] // 2, v - rgb.shape[0])
        return out

    def close(self) -> None:
        self.stop_physics()
        self._renderer.close()

    # ── internals ─────────────────────────────────────────────────────────────

    def _set_frustum(self) -> None:
        """
        Override the MuJoCo scene camera frustum to match real camera intrinsics.
        Call this after every update_scene() and before render().
        This gives true foreshortening: steep angles squash the robot correctly.
        """
        K    = self.cfg.intrinsics
        h, w = self.cfg.render_height, self.cfg.render_width
        near, far = 0.05, 50.0
        cam  = self._renderer.scene.camera[0]
        cam.frustum_near   = near
        cam.frustum_far    = far
        # Map pixel principal point to frustum planes at near distance
        cam.frustum_bottom = -near * K.cy / K.fy
        cam.frustum_top    =  near * (h - K.cy) / K.fy
        cam.frustum_width  =  near * w / K.fx
        cam.frustum_center =  near * (w / 2 - K.cx) / K.fx   # ≈ 0 for centred lens

    def _update_sprite_camera(self, pose: np.ndarray) -> None:
        """
        Derive virtual camera azimuth/elevation AND distance from the real
        camera's pose so the robot sprite shows the correct face at the correct
        perspective scale.

        pose : world-to-camera 4×4 (OpenCV convention, Y-down, Z-forward)
        """
        R, t = pose[:3, :3], pose[:3, 3]

        # Actual distance from real camera to robot in world space
        p_robot_cam = R @ self.cfg.robot_world_pos + t
        dist = float(np.linalg.norm(p_robot_cam))
        dist = max(0.5, min(dist, 5.0))

        # Camera forward direction in world (OpenCV: +Z is forward)
        fwd = R.T @ np.array([0.0, 0.0, 1.0])

        # Yaw (horizontal rotation) and pitch (vertical tilt) from world forward
        yaw_rad   = np.arctan2(fwd[0], fwd[2])
        pitch_rad = np.arctan2(-fwd[1], np.sqrt(fwd[0] ** 2 + fwd[2] ** 2))

        # Virtual cam rotates opposite to real camera so the robot faces the viewer
        new_az   = self.cfg.sprite_azimuth - float(np.degrees(yaw_rad))
        new_el   = self.cfg.sprite_elevation + float(np.degrees(pitch_rad))
        new_dist = dist

        changed = (
            abs(new_az   - self._last_az)  > self._az_thresh or
            abs(new_el   - self._last_el)  > self._az_thresh or
            abs(new_dist - self._sprite_cam.distance) > 0.1
        )
        if changed:
            self._sprite_cam.azimuth   = new_az
            self._sprite_cam.elevation = new_el
            self._sprite_cam.distance  = new_dist
            self._last_az  = new_az
            self._last_el  = new_el
            self._sprite_dirty = True

    def _project(
        self, pose: np.ndarray
    ) -> Tuple[int, int, float]:
        """Project robot_world_pos through camera pose → (u, v, depth_z)."""
        R, t = pose[:3, :3], pose[:3, 3]
        p_cam = R @ self.cfg.robot_world_pos + t
        z = float(p_cam[2])
        if z <= 0.1:
            z = 0.1   # clamp behind-camera to near plane — keeps robot visible
        K  = self.cfg.intrinsics
        u  = int(K.fx * p_cam[0] / z + K.cx)
        v  = int(K.fy * p_cam[1] / z + K.cy)
        # Clamp to frame so robot never fully disappears
        h, w = self.cfg.render_height, self.cfg.render_width
        u = int(np.clip(u, w // 6, w * 5 // 6))
        v = int(np.clip(v, h // 3, h - 10))   # lower 2/3 = floor region
        return u, v, z

    def _get_sprite(
        self, target_h: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rgb_bgr, mask) sprite resized to target_h rows."""
        if self._sprite_dirty or self._sprite_rgb is None:
            self._render_sprite()

        src_h, src_w = self._sprite_rgb.shape[:2]
        scale   = target_h / src_h
        new_w   = max(1, int(src_w * scale))
        new_h   = target_h

        rgb  = cv2.resize(self._sprite_rgb,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(
            self._sprite_mask.astype(np.uint8), (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        return rgb, mask

    def _render_sprite(self) -> None:
        """Offscreen-render robot and extract robot-only pixels."""
        # Keep sprite camera locked onto the robot's current base position so
        # base motion (joint_x / joint_y) never walks the robot out of frame.
        bx = float(self._data.qpos[0])
        by = float(self._data.qpos[1])
        self._sprite_cam.lookat[0] = bx
        self._sprite_cam.lookat[1] = by

        self._renderer.update_scene(
            self._data, camera=self._sprite_cam, scene_option=self._opt
        )
        self._set_frustum()                    # match real camera intrinsics
        rgb = self._renderer.render().copy()   # H×W×3 uint8

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(
            self._data, camera=self._sprite_cam, scene_option=self._opt
        )
        self._set_frustum()   # reapply after second update_scene
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()

        mask = depth < self.cfg.depth_far_thresh
        if not mask.any():
            mask = depth < depth.max() * 0.99

        # Crop to bounding box of robot pixels to remove empty border
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) and len(cols):
            r0, r1 = rows[0], rows[-1] + 1
            c0, c1 = cols[0], cols[-1] + 1
            rgb  = rgb[r0:r1, c0:c1]
            mask = mask[r0:r1, c0:c1]

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Boost brightness to match indoor webcam exposure
        bgr = cv2.convertScaleAbs(bgr, alpha=1.6, beta=20)
        self._sprite_rgb  = bgr
        self._sprite_mask = mask
        self._sprite_dirty = False

    def _draw_shadow(
        self,
        canvas: np.ndarray,
        cx: int, cy: int,
        sprite_w: int,
    ) -> None:
        """Draw a soft elliptical shadow at the robot's feet."""
        ax = max(8, sprite_w // 2)
        ay = max(4, sprite_w // 8)
        overlay = canvas.copy()
        cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, (10, 10, 10), -1)
        cv2.addWeighted(overlay, self.cfg.shadow_opacity,
                        canvas, 1 - self.cfg.shadow_opacity, 0, canvas)

    @staticmethod
    def _blit(
        canvas: np.ndarray,
        sprite: np.ndarray,
        mask: np.ndarray,
        x: int, y: int,
    ) -> None:
        """Paste sprite onto canvas at top-left (x, y) using mask."""
        ch, cw = canvas.shape[:2]
        sh, sw = sprite.shape[:2]

        x0, y0 = max(0, x),   max(0, y)
        x1, y1 = min(cw, x+sw), min(ch, y+sh)
        sx0 = x0 - x
        sy0 = y0 - y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        if x1 <= x0 or y1 <= y0:
            return

        roi  = canvas[y0:y1, x0:x1]
        spr  = sprite[sy0:sy1, sx0:sx1]
        msk  = mask[sy0:sy1, sx0:sx1]
        roi[msk] = spr[msk]


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(
        description="Autorobo AR renderer — robot overlaid on video or webcam"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video",  type=str, default=None,
                        help="Path to video file (e.g. video/room_video.mp4)")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    from ar.depth_estimator    import DepthEstimator, DepthConfig
    from ar.camera_tracker     import CameraTracker
    from ar.command_interface  import CommandInterface, RobotController

    # ── open video or webcam ──────────────────────────────────────────────────
    use_video = args.video is not None
    if use_video:
        video_path = Path(args.video) if Path(args.video).is_absolute() \
                     else ROOT / args.video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return
        vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[AR] Video: {video_path.name}  {vid_w}×{vid_h}  {vid_fps:.1f}fps")

        # Phone camera intrinsics for this resolution
        ar_cfg = ARConfig(
            render_width  = vid_w,
            render_height = vid_h,
            intrinsics    = CameraIntrinsics(
                fx = vid_w * 1.1,   # phone camera ~70° horizontal FOV
                fy = vid_w * 1.1,
                cx = vid_w / 2.0,
                cy = vid_h / 2.0,
            ),
        )

        def read_frame():
            ok, frame = cap.read()
            if not ok:                   # video ended — loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            return frame if ok else None

        def release(): cap.release()

    else:
        depth_cfg = DepthConfig(camera_index=args.camera)
        _est = DepthEstimator(depth_cfg)
        _est.open_camera()
        ar_cfg = ARConfig()
        vid_fps = 30.0

        def read_frame(): return _est.read_frame()
        def release():    _est.close_camera()

    # ── build pipeline ────────────────────────────────────────────────────────
    estimator  = DepthEstimator(DepthConfig())   # for depth inference only
    renderer   = ARRenderer(ar_cfg)
    controller = RobotController()

    quit_event = threading.Event()
    cmd_iface  = CommandInterface(
        on_command = controller.set_command,
        on_quit    = quit_event.set,
    )
    renderer.start_physics(controller=controller)
    cmd_iface.start()

    if args.no_preview:
        print("ARRenderer ready.")
        renderer.close()
        release()
        return

    lock   = threading.Lock()
    shared = {"frame": None, "pose": np.eye(4), "fps": 0.0}

    def inference_loop() -> None:
        while not stop_event.is_set():
            with lock:
                frame = shared["frame"]
            if frame is None:
                time.sleep(0.01)
                continue
            t0    = time.perf_counter()
            depth = estimator.estimate(frame)
            elapsed = time.perf_counter() - t0
            with lock:
                shared["fps"] = 1.0 / elapsed if elapsed > 0 else 0.0
            print(f"[ar] depth {elapsed*1000:.0f}ms  {shared['fps']:.1f}fps")

    stop_event = threading.Event()
    worker = threading.Thread(target=inference_loop, daemon=True)
    worker.start()

    frame_delay = max(1, int(1000 / vid_fps))
    source_label = Path(args.video).name if use_video else "webcam"
    print(f"\n[AR] Running on {source_label}.  Type commands in terminal.  Q to quit.\n")

    try:
        while True:
            frame = read_frame()
            if frame is None:
                break

            with lock:
                shared["frame"] = frame.copy()
                fps = shared["fps"]

            out = renderer.composite(frame, shared["pose"])

            label = f"AR  {source_label}  {fps:.1f}fps  Q=quit"
            cv2.putText(out, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Scale up for display so portrait video fills the screen nicely
            dh = 720
            dw = int(out.shape[1] * dh / out.shape[0])
            display = cv2.resize(out, (dw, dh))

            cv2.imshow("Autorobo — AR Preview (Q to quit)", display)
            if cv2.waitKey(1) & 0xFF == ord("q") or quit_event.is_set():
                break

    finally:
        stop_event.set()
        release()
        renderer.close()
        cv2.destroyAllWindows()
        print("[ARRenderer] Stopped.")


if __name__ == "__main__":
    main()
