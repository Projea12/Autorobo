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
    xml_path:      Path             = ROOT / "robot" / "robot.xml"
    render_width:  int              = 640
    render_height: int              = 480
    intrinsics:    CameraIntrinsics = field(default_factory=CameraIntrinsics)

    # Where the robot stands in the AR world (OpenCV world frame, metres)
    # x=0 centre, y=1.2 below camera (floor level), z=1.2 in front
    robot_world_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.2, 1.2])
    )

    # Virtual camera for robot sprite
    sprite_lookat:    np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.35, 0.0])
    )
    sprite_distance:  float = 2.0
    sprite_azimuth:   float = -45.0
    sprite_elevation: float = -20.0

    # Rendering
    robot_height_m:   float = 0.80   # approximate robot height (metres)
    shadow_opacity:   float = 0.50
    robot_opacity:    float = 0.92   # blend factor for robot pixels
    keyframe:         str   = "home"
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

        # Hide ground plane in renders
        self._opt = mujoco.MjvOption()
        self._opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False

        # Cache the sprite (re-rendered on pose change)
        self._sprite_rgb:  Optional[np.ndarray] = None
        self._sprite_mask: Optional[np.ndarray] = None
        self._sprite_dirty = True

        print("[ARRenderer] Ready.")

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

        # Project robot world position onto screen
        u, v, depth_z = self._project(camera_pose)

        print(f"[ar] project → screen=({u},{v})  depth={depth_z:.2f}m")
        if depth_z <= 0.05:
            return frame_bgr

        # Scale sprite so robot occupies correct pixel height
        cfg = self.cfg
        K   = cfg.intrinsics
        sprite_h_px = int(K.fy * cfg.robot_height_m / depth_z)
        sprite_h_px = max(40, min(sprite_h_px, frame_bgr.shape[0]))

        rgb, mask = self._get_sprite(sprite_h_px)

        out = frame_bgr.copy()
        self._draw_shadow(out, u, v, rgb.shape[1])
        self._blit(out, rgb, mask, u - rgb.shape[1] // 2, v - rgb.shape[0])
        return out

    def close(self) -> None:
        self._renderer.close()

    # ── internals ─────────────────────────────────────────────────────────────

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
        u = int(np.clip(u, w // 5, w * 4 // 5))
        v = int(np.clip(v, h // 2, h - 10))   # always in lower half = on the floor
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
        self._renderer.update_scene(
            self._data, camera=self._sprite_cam, scene_option=self._opt
        )
        rgb = self._renderer.render().copy()   # H×W×3 uint8

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(
            self._data, camera=self._sprite_cam, scene_option=self._opt
        )
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
        # Boost brightness to match typical indoor webcam exposure
        bgr = cv2.convertScaleAbs(bgr, alpha=1.6, beta=30)
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
        description="Autorobo AR renderer — robot overlaid on webcam"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    from ar.depth_estimator import DepthConfig, DepthEstimator
    from ar.camera_tracker  import CameraTracker

    depth_cfg = DepthConfig(camera_index=args.camera)
    estimator = DepthEstimator(depth_cfg)
    tracker   = CameraTracker()
    renderer  = ARRenderer()

    if args.no_preview:
        print("ARRenderer ready.")
        renderer.close()
        return

    estimator.open_camera()

    lock     = threading.Lock()
    shared: dict = {
        "frame": None,
        "pose":  np.eye(4),
        "fps":   0.0,
        "feats": 0,
    }

    def inference_loop() -> None:
        while not stop_event.is_set():
            with lock:
                frame = shared["frame"]
            if frame is None:
                time.sleep(0.01)
                continue

            t0    = time.perf_counter()
            depth = estimator.estimate(frame)
            result = tracker.update(frame, depth)
            elapsed = time.perf_counter() - t0

            with lock:
                shared["pose"]  = result.pose.copy()
                shared["fps"]   = 1.0 / elapsed if elapsed > 0 else 0.0
                shared["feats"] = result.n_features

            print(f"[ar] {result.status:<8}  feats={result.n_features:3d}  "
                  f"{shared['fps']:.1f}fps")

    stop_event = threading.Event()
    worker = threading.Thread(target=inference_loop, daemon=True)
    worker.start()

    print("\n[ARRenderer] Robot overlaid on webcam.  Move camera slowly.  Q to quit.\n")

    try:
        while True:
            frame = estimator.read_frame()
            if frame is None:
                break

            with lock:
                shared["frame"] = frame.copy()
                pose  = shared["pose"].copy()
                fps   = shared["fps"]
                feats = shared["feats"]

            # AR composite
            out = renderer.composite(frame, pose)

            label = f"AR  feats={feats}  {fps:.1f}fps  Q=quit"
            cv2.putText(out, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Autorobo — AR Preview (Q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_event.set()
        estimator.close_camera()
        renderer.close()
        cv2.destroyAllWindows()
        print("[ARRenderer] Stopped.")


if __name__ == "__main__":
    main()
