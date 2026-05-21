"""
ar/slam.py — Lightweight monocular visual SLAM / visual odometry.

Builds a sparse 3D map from video frames using:
  - ORB feature extraction
  - BF matcher + ratio test
  - Essential matrix (RANSAC) → relative pose
  - Triangulation → 3D map points
  - PnP (RANSAC) → global localisation against map

Designed for use with video_ar.py where video IS the robot camera feed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class SLAMConfig:
    # Camera intrinsics (set from video dimensions in video_ar.py)
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0

    # Feature detection
    n_features: int   = 1000       # ORB max features per frame
    match_ratio: float = 0.75      # Lowe's ratio test
    min_matches: int  = 30         # minimum matches to attempt pose estimation

    # Map management
    keyframe_min_matches: int = 80  # add keyframe if tracked matches < this
    map_max_points: int = 5000      # cap map size

    # Minimap display (pixels)
    minimap_size: int  = 200
    minimap_scale: float = 40.0    # pixels per metre in minimap


# ── frame / keyframe data ──────────────────────────────────────────────────────

@dataclass
class Frame:
    kps:  np.ndarray       # Nx2 float32 keypoint coords
    desc: np.ndarray       # NxD uint8 descriptors
    pose: np.ndarray       # 4×4 world-to-camera (OpenCV: Y-down, Z-fwd)


# ── slam system ───────────────────────────────────────────────────────────────

class VisualSLAM:
    """
    Monocular visual odometry + sparse map.

    Usage
    -----
    slam = VisualSLAM(cfg)
    slam.process(frame_gray)   # call every frame; returns current pose 4×4

    pose = slam.pose           # world-to-camera 4×4
    pos  = slam.position_xz    # (x, z) in metres (floor plane)
    """

    def __init__(self, cfg: SLAMConfig = SLAMConfig()) -> None:
        self.cfg  = cfg
        self._K   = np.array([
            [cfg.fx,   0.0,  cfg.cx],
            [  0.0,  cfg.fy,  cfg.cy],
            [  0.0,    0.0,    1.0 ],
        ], dtype=np.float64)

        self._orb     = cv2.ORB_create(nfeatures=cfg.n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._lock       = threading.Lock()
        self._pose       = np.eye(4)          # world-to-camera
        self._trajectory: List[np.ndarray] = [np.zeros(3)]  # world-space positions

        # Map: 3D points (world frame) + descriptors
        self._map_pts:  Optional[np.ndarray] = None   # Mx3
        self._map_desc: Optional[np.ndarray] = None   # MxD

        # Previous keyframe
        self._prev: Optional[Frame] = None
        self._initialized = False
        self._scale = 1.0    # running scale estimate

    # ── public ────────────────────────────────────────────────────────────────

    def process(self, gray: np.ndarray) -> np.ndarray:
        """
        Process one greyscale frame. Returns current world-to-camera pose (4×4).
        """
        kps_cv, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kps_cv) < self.cfg.min_matches:
            with self._lock:
                return self._pose.copy()

        kps = np.array([kp.pt for kp in kps_cv], dtype=np.float32)
        cur = Frame(kps=kps, desc=desc, pose=self._pose.copy())

        with self._lock:
            if not self._initialized:
                self._prev = cur
                self._initialized = True
                return self._pose.copy()

            pose = self._track(cur)
            self._pose = pose
            cur.pose   = pose.copy()

            # Camera position in world space
            R, t = pose[:3, :3], pose[:3, 3]
            cam_pos = -(R.T @ t)              # world position of camera
            self._trajectory.append(cam_pos)

            # Expand map if needed
            if self._should_add_keyframe(cur):
                self._add_keyframe(cur)
            self._prev = cur

        return self._pose.copy()

    @property
    def pose(self) -> np.ndarray:
        with self._lock:
            return self._pose.copy()

    @property
    def position_xz(self) -> Tuple[float, float]:
        """(x, z) robot position in world metres — floor plane."""
        with self._lock:
            if len(self._trajectory) < 1:
                return 0.0, 0.0
            p = self._trajectory[-1]
            return float(p[0]), float(p[2])

    @property
    def trajectory(self) -> List[np.ndarray]:
        with self._lock:
            return list(self._trajectory)

    def draw_minimap(self, canvas: np.ndarray) -> None:
        """Draw a top-down minimap in the bottom-right corner of canvas."""
        cfg = self.cfg
        ms  = cfg.minimap_size
        s   = cfg.minimap_scale

        h, w = canvas.shape[:2]
        x0, y0 = w - ms - 10, h - ms - 10   # top-left corner of minimap

        # Dark background
        sub = canvas[y0:y0+ms, x0:x0+ms]
        sub[:] = (sub * 0.3).astype(np.uint8)
        cv2.rectangle(canvas, (x0, y0), (x0+ms, y0+ms), (80, 80, 80), 1)

        cx, cz = ms // 2, ms // 2   # minimap centre = robot origin

        with self._lock:
            traj = list(self._trajectory)

        # Draw trajectory
        for i in range(1, len(traj)):
            p0, p1 = traj[i-1], traj[i]
            u0 = int(cx + p0[0] * s)
            v0 = int(cz - p0[2] * s)
            u1 = int(cx + p1[0] * s)
            v1 = int(cz - p1[2] * s)
            if _in_map(u0, v0, ms) and _in_map(u1, v1, ms):
                cv2.line(canvas, (x0+u0, y0+v0), (x0+u1, y0+v1), (0, 220, 80), 1)

        # Map points
        if self._map_pts is not None:
            for pt in self._map_pts[::5]:   # subsample for speed
                u = int(cx + pt[0] * s)
                v = int(cz - pt[2] * s)
                if _in_map(u, v, ms):
                    cv2.circle(canvas, (x0+u, y0+v), 1, (200, 200, 80), -1)

        # Robot dot
        with self._lock:
            rx, rz = (float(self._trajectory[-1][0]), float(self._trajectory[-1][2])) \
                     if self._trajectory else (0.0, 0.0)
        ru = int(cx + rx * s)
        rv = int(cz - rz * s)
        if _in_map(ru, rv, ms):
            cv2.circle(canvas, (x0+ru, y0+rv), 5, (0, 100, 255), -1)

        # Label
        cv2.putText(canvas, "MAP", (x0+4, y0+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # ── internals ─────────────────────────────────────────────────────────────

    def _track(self, cur: Frame) -> np.ndarray:
        """Estimate cur.pose from matches against prev frame."""
        prev = self._prev

        matches = self._match(prev.desc, cur.desc)
        if len(matches) < self.cfg.min_matches:
            return prev.pose.copy()

        pts_prev = prev.kps[[m.queryIdx for m in matches]]
        pts_cur  = cur.kps [[m.trainIdx for m in matches]]

        # Essential matrix
        E, mask = cv2.findEssentialMat(
            pts_cur, pts_prev, self._K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            return prev.pose.copy()

        _, R, t, mask2 = cv2.recoverPose(E, pts_cur, pts_prev, self._K, mask=mask)

        # t is unit-norm — use map-based scale if available, else heuristic
        t_scaled = t.flatten() * self._scale

        # Compose: new_pose = [R | t_scaled] @ prev_pose
        dT      = np.eye(4)
        dT[:3, :3] = R
        dT[:3,  3] = t_scaled
        new_pose = dT @ prev.pose

        # Refine pose against map via PnP if map exists
        if self._map_pts is not None and len(self._map_pts) > 10:
            refined = self._pnp_refine(cur, new_pose)
            if refined is not None:
                new_pose = refined

        return new_pose

    def _pnp_refine(self, cur: Frame, init_pose: np.ndarray) -> Optional[np.ndarray]:
        """Use PnP against the 3D map to get an absolute pose estimate."""
        if self._map_pts is None or self._map_desc is None:
            return None

        matches = self._match(self._map_desc, cur.desc)
        if len(matches) < 6:
            return None

        obj_pts = self._map_pts[[m.queryIdx for m in matches]].astype(np.float64)
        img_pts = cur.kps[[m.trainIdx for m in matches]].astype(np.float64)

        R0 = init_pose[:3, :3]
        t0 = init_pose[:3, 3:4]
        rvec0, _ = cv2.Rodrigues(R0)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self._K, None,
            rvec0, t0, useExtrinsicGuess=True,
            iterationsCount=100, reprojectionError=4.0
        )
        if not ok or inliers is None or len(inliers) < 6:
            return None

        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3,  3] = tvec.flatten()
        return pose

    def _should_add_keyframe(self, cur: Frame) -> bool:
        if self._prev is None:
            return False
        if self._map_pts is None:
            return True
        matches = self._match(self._prev.desc, cur.desc)
        return len(matches) < self.cfg.keyframe_min_matches

    def _add_keyframe(self, cur: Frame) -> None:
        """Triangulate new 3D points between prev keyframe and cur."""
        prev = self._prev
        matches = self._match(prev.desc, cur.desc)
        if len(matches) < self.cfg.min_matches:
            return

        pts_prev = prev.kps[[m.queryIdx for m in matches]].T   # 2×N
        pts_cur  = cur.kps [[m.trainIdx for m in matches]].T

        P0 = self._K @ prev.pose[:3]
        P1 = self._K @ cur.pose[:3]

        pts4d = cv2.triangulatePoints(P0, P1, pts_prev, pts_cur)
        pts4d /= pts4d[3:4]
        pts3d = pts4d[:3].T.astype(np.float32)    # Nx3

        # Filter: keep points in front of both cameras and not too far
        good = (pts4d[2] > 0.01) & (pts4d[2] < 50.0)
        pts3d = pts3d[good]

        descs = cur.desc[[m.trainIdx for m in matches]][good]

        # Update running scale from median depth
        if len(pts3d) > 5:
            median_depth = float(np.median(pts3d[:, 2]))
            if 0.1 < median_depth < 20.0:
                self._scale = median_depth * 0.05  # keep scale small

        # Append to map
        if self._map_pts is None:
            self._map_pts  = pts3d
            self._map_desc = descs
        else:
            self._map_pts  = np.vstack([self._map_pts,  pts3d])
            self._map_desc = np.vstack([self._map_desc, descs])

        # Cap map size
        cap = self.cfg.map_max_points
        if len(self._map_pts) > cap:
            keep = np.random.choice(len(self._map_pts), cap, replace=False)
            self._map_pts  = self._map_pts[keep]
            self._map_desc = self._map_desc[keep]

    def _match(self, desc1: np.ndarray, desc2: np.ndarray) -> list:
        """BF match + Lowe ratio test."""
        if desc1 is None or desc2 is None:
            return []
        pairs = self._matcher.knnMatch(desc1, desc2, k=2)
        good  = []
        for pair in pairs:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.cfg.match_ratio * n.distance:
                    good.append(m)
        return good


# ── helpers ───────────────────────────────────────────────────────────────────

def _in_map(u: int, v: int, size: int) -> bool:
    return 0 <= u < size and 0 <= v < size
