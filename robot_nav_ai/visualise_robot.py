"""
Autorobo — full robot visualiser (faithful to robot/robot.xml).

Shows the complete robot as simulated in MuJoCo:
  • Differential-drive base  (0.50 × 0.40 × 0.15 m)
  • Two large drive wheels   (r = 0.10 m)
  • Two front ball casters
  • 6-DOF arm  (UR5 proportions) in "ready" pose
  • Parallel gripper with two fingers
  • LiDAR site, RGB-D camera, wrist camera, IMU, F/T site
  • LiDAR 360° scan fan visualisation

Usage:
    python visualise_robot.py            # interactive 3-D window
    python visualise_robot.py --save     # save robot_model.png
    python visualise_robot.py --pose home   # show home (arm-up) pose
    python visualise_robot.py --pose ready  # show ready (arm-forward) pose [default]
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ══════════════════════════════════════════════════════════════════════════════
# Colour palette  — matches robot.xml rgba / material style
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "chassis":   "#2e3133",   # dark grey body
    "deck":      "#e8e8e8",   # light top plate
    "bumper":    "#1a55bb",   # blue front/rear accent
    "wheel":     "#111111",   # rubber black
    "hub":       "#aaaaaa",   # aluminium hub
    "arm":       "#cc6600",   # arm links (UR5 orange-ish)
    "wrist":     "#dd7700",
    "gripper":   "#888888",   # gripper body
    "tip":       "#22bb22",   # rubber fingertips
    "lidar":     "#111111",   # LiDAR housing
    "scan":      "#00ddff",   # LiDAR ring + rays
    "camera":    "#1a55bb",   # camera body
    "imu":       "#dddddd",   # IMU chip
    "ft":        "#ff4444",   # F/T sensor site
    "ground":    "#1c1c1c",
    "label":     "#ffffff",
}


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _box(cx, cy, cz, lx, ly, lz):
    """6 quad faces of an axis-aligned box."""
    dx, dy, dz = lx/2, ly/2, lz/2
    x0, x1 = cx-dx, cx+dx
    y0, y1 = cy-dy, cy+dy
    z0, z1 = cz-dz, cz+dz
    return [
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
        [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
    ]

def _cyl(cx, cy, cz, r, h, axis="z", n=28):
    """Lateral faces of a cylinder."""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    faces = []
    for i in range(n):
        t0, t1 = t[i], t[(i+1)%n]
        if axis == "y":
            pts = lambda a: [cx+r*np.cos(a), cy, cz+r*np.sin(a)]
            a0=[cx+r*np.cos(t0),cy-h/2,cz+r*np.sin(t0)]
            a1=[cx+r*np.cos(t1),cy-h/2,cz+r*np.sin(t1)]
            b1=[cx+r*np.cos(t1),cy+h/2,cz+r*np.sin(t1)]
            b0=[cx+r*np.cos(t0),cy+h/2,cz+r*np.sin(t0)]
        elif axis == "x":
            a0=[cx-h/2,cy+r*np.cos(t0),cz+r*np.sin(t0)]
            a1=[cx-h/2,cy+r*np.cos(t1),cz+r*np.sin(t1)]
            b1=[cx+h/2,cy+r*np.cos(t1),cz+r*np.sin(t1)]
            b0=[cx+h/2,cy+r*np.cos(t0),cz+r*np.sin(t0)]
        else:
            a0=[cx+r*np.cos(t0),cy+r*np.sin(t0),cz-h/2]
            a1=[cx+r*np.cos(t1),cy+r*np.sin(t1),cz-h/2]
            b1=[cx+r*np.cos(t1),cy+r*np.sin(t1),cz+h/2]
            b0=[cx+r*np.cos(t0),cy+r*np.sin(t0),cz+h/2]
        faces.append([a0,a1,b1,b0])
    return faces

def _sphere(cx, cy, cz, r, n=12):
    faces = []
    lats = np.linspace(-np.pi/2, np.pi/2, n)
    lons = np.linspace(0, 2*np.pi, n*2, endpoint=False)
    for i in range(len(lats)-1):
        for j in range(len(lons)):
            la0,la1 = lats[i],lats[i+1]
            lo0,lo1 = lons[j],lons[(j+1)%len(lons)]
            def p(la,lo): return [cx+r*np.cos(la)*np.cos(lo),
                                   cy+r*np.cos(la)*np.sin(lo),
                                   cz+r*np.sin(la)]
            faces.append([p(la0,lo0),p(la0,lo1),p(la1,lo1),p(la1,lo0)])
    return faces

def _capsule(p0, p1, r, n=16):
    """Approximate capsule as a cylinder between two points."""
    p0, p1 = np.array(p0), np.array(p1)
    d = p1 - p0
    L = np.linalg.norm(d)
    mid = (p0 + p1) / 2
    # choose an axis perpendicular to d
    if abs(d[2]) < 0.9:
        perp = np.cross(d, [0,0,1])
    else:
        perp = np.cross(d, [1,0,0])
    perp = perp / np.linalg.norm(perp)
    perp2 = np.cross(d/L, perp)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    faces = []
    for i in range(n):
        t0, t1 = t[i], t[(i+1)%n]
        for t_pair in [(t0,t1)]:
            a = r*(np.cos(t_pair[0])*perp + np.sin(t_pair[0])*perp2)
            b = r*(np.cos(t_pair[1])*perp + np.sin(t_pair[1])*perp2)
            faces.append([p0+a, p0+b, p1+b, p1+a])
    return faces

def _add(ax, faces, color, alpha=0.88, lw=0.25):
    pc = Poly3DCollection(faces, alpha=alpha, linewidth=lw, edgecolor="#000000")
    pc.set_facecolor(color)
    ax.add_collection3d(pc)

def _lbl(ax, x, y, z, txt, color="white", fs=7.5):
    ax.text(x, y, z, txt, fontsize=fs, ha="center", va="bottom",
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#00000099", ec="none"))

def _site(ax, pos, color, s=0.012, label=""):
    x,y,z = pos
    ax.scatter([x],[y],[z], c=[color], s=80, marker="o", zorder=5, depthshade=False)
    if label:
        ax.text(x, y, z+0.025, label, fontsize=6, color=color, ha="center")

def _axes(ax, ox, oy, oz, L=0.08, lw=2.0):
    for d, c in zip([(L,0,0),(0,L,0),(0,0,L)],
                    ["#ff3333","#33ff33","#3399ff"]):
        ax.quiver(ox,oy,oz,*d,color=c,arrow_length_ratio=0.3,linewidth=lw)


# ══════════════════════════════════════════════════════════════════════════════
# Arm kinematics  (joints rotate around Y unless noted)
# ══════════════════════════════════════════════════════════════════════════════

def _ry(theta):
    """Rotation matrix around Y axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def _rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def _rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def arm_fk(j1, j2, j3, j4, j5, j6):
    """
    Forward kinematics for the 6-DOF arm.
    Returns world-frame positions (in base body frame):
      shoulder_base, shoulder_tip, elbow, wrist1, wrist2, wrist3, ee
    """
    # Offsets in local frames (from robot.xml)
    SHOULDER_BASE  = np.array([0.0, 0.0, 0.075])   # pan base on chassis top
    PAN_TO_LIFT    = np.array([0.0, 0.0, 0.040])    # shoulder_pan → shoulder_lift
    UPPER_ARM_LEN  = 0.30
    FOREARM_LEN    = 0.26
    W1_OFFSET      = 0.045
    W2_OFFSET      = 0.040
    W3_OFFSET      = 0.025 + 0.042                  # to EE site

    # Base of shoulder in base body coords
    sb = SHOULDER_BASE

    # Rotation from joint1 (pan, around Z)
    R = _rz(j1)

    # shoulder_lift origin
    sl = sb + R @ PAN_TO_LIFT
    # joint2: shoulder lift around Y
    R = R @ _ry(j2)

    # elbow
    el = sl + R @ np.array([UPPER_ARM_LEN, 0, 0])
    # joint3: elbow around Y
    R = R @ _ry(j3)

    # wrist1
    w1 = el + R @ np.array([FOREARM_LEN, 0, 0])
    # joint4: wrist1 around Y
    R = R @ _ry(j4)

    # wrist2
    w2 = w1 + R @ np.array([W1_OFFSET, 0, 0])
    # joint5: wrist2 around Z
    R = R @ _rz(j5)

    # wrist3
    w3 = w2 + R @ np.array([W2_OFFSET, 0, 0])
    # joint6: wrist3 around X
    R = R @ _rx(j6)

    # end-effector
    ee = w3 + R @ np.array([W3_OFFSET, 0, 0])

    # gripper finger tips
    fl = ee + R @ np.array([0.0,  0.025, 0])
    fr = ee + R @ np.array([0.0, -0.025, 0])

    return dict(sb=sb, sl=sl, el=el, w1=w1, w2=w2, w3=w3, ee=ee, fl=fl, fr=fr, R=R)


# ══════════════════════════════════════════════════════════════════════════════
# World-frame dimensions  (base body at z=0.15)
# ══════════════════════════════════════════════════════════════════════════════
BASE_Z   = 0.15   # base body centre in world frame

def to_world(local):
    """Shift local base-body coords to world frame."""
    return np.array(local) + np.array([0, 0, BASE_Z])


def draw_robot(ax, pose="ready"):
    """Draw the complete robot. pose = 'home' | 'ready' | 'grasp_open'"""

    # ── ground ────────────────────────────────────────────────────────────────
    gf = _box(0, 0, -0.006, 1.4, 1.0, 0.012)
    pc = Poly3DCollection(gf, alpha=0.25, linewidth=0)
    pc.set_facecolor(C["ground"]); ax.add_collection3d(pc)

    # ── chassis  (size 0.25×0.20×0.075 in MuJoCo = half-extents) ─────────────
    # Full dimensions: 0.50 × 0.40 × 0.15 m, centre at (0,0,BASE_Z)
    _add(ax, _box(0, 0, BASE_Z, 0.50, 0.40, 0.15), C["chassis"], alpha=0.93)

    # top deck plate
    _add(ax, _box(0, 0, BASE_Z+0.076, 0.46, 0.36, 0.010), C["deck"], alpha=0.80)

    # front bumper accent
    _add(ax, _box( 0.255, 0, BASE_Z, 0.010, 0.40, 0.10), C["bumper"], alpha=0.95)
    # rear bumper accent
    _add(ax, _box(-0.255, 0, BASE_Z, 0.010, 0.40, 0.10), C["bumper"], alpha=0.95)

    # branding label on side
    ax.text(0.0, -0.21, BASE_Z+0.02, "AUTOROBO", fontsize=6.5, color="#ffffff",
            ha="center", va="center", fontweight="bold", alpha=0.7)
    ax.text(0.0, -0.21, BASE_Z-0.02, "NAV_ROBOT", fontsize=5.5, color="#aaccff",
            ha="center", va="center", alpha=0.6)

    # ── drive wheels  (radius=0.10, half-length=0.030 → width=0.060) ─────────
    # wheel centre at y=±0.225, z = BASE_Z - 0.05
    WZ = BASE_Z - 0.05
    _add(ax, _cyl(0,  0.225, WZ, 0.100, 0.060, axis="y"), C["wheel"], alpha=0.97)
    _add(ax, _cyl(0, -0.225, WZ, 0.100, 0.060, axis="y"), C["wheel"], alpha=0.97)
    # hub discs
    _add(ax, _cyl(0,  0.256, WZ, 0.040, 0.004, axis="y"), C["hub"])
    _add(ax, _cyl(0, -0.256, WZ, 0.040, 0.004, axis="y"), C["hub"])
    # hub bolts cross (left)
    _add(ax, _box(0,  0.258, WZ, 0.008, 0.003, 0.08), "#555555")
    _add(ax, _box(0,  0.258, WZ, 0.08,  0.003, 0.008), "#555555")
    # hub bolts cross (right)
    _add(ax, _box(0, -0.258, WZ, 0.008, 0.003, 0.08), "#555555")
    _add(ax, _box(0, -0.258, WZ, 0.08,  0.003, 0.008), "#555555")

    # ── caster spheres  (r=0.030, at pos=(±0.20, ±0.10, -0.12) in base) ─────
    for cx, cy in [(0.20, 0.10), (0.20, -0.10)]:
        wz = BASE_Z - 0.12
        _add(ax, _sphere(cx, cy, wz, 0.030), "#222222", alpha=0.90)
        _add(ax, _cyl(cx, cy, wz, 0.031, 0.006, axis="z"), C["hub"], alpha=0.70)

    # ── LiDAR housing (mast + sensor) ────────────────────────────────────────
    # Mast column from chassis top to sensor
    MAST_BASE_Z = BASE_Z + 0.075   # top of chassis
    MAST_TOP_Z  = MAST_BASE_Z + 0.32
    _add(ax, _box(-0.05, 0, (MAST_BASE_Z+MAST_TOP_Z)/2, 0.06, 0.06,
                  MAST_TOP_Z - MAST_BASE_Z), "#3a3a3a", alpha=0.90)

    # Velodyne-style LiDAR sensor head
    LIDAR_Z = MAST_TOP_Z + 0.04
    _add(ax, _cyl(-0.05, 0, LIDAR_Z, 0.065, 0.085, axis="z"), "#111111", alpha=0.97)
    _add(ax, _cyl(-0.05, 0, LIDAR_Z+0.005, 0.068, 0.010, axis="z"), C["scan"], alpha=0.95)
    _add(ax, _cyl(-0.05, 0, LIDAR_Z+0.048, 0.030, 0.006, axis="z"), C["hub"])

    # 360° scan fan (cone of rays in XY plane)
    LIDAR_SITE = np.array([-0.05, 0, LIDAR_Z])
    n_rays = 36
    ray_len = 0.55
    for i in range(n_rays):
        angle = 2*np.pi*i/n_rays
        dx, dy = ray_len*np.cos(angle), ray_len*np.sin(angle)
        ax.plot([LIDAR_SITE[0], LIDAR_SITE[0]+dx],
                [LIDAR_SITE[1], LIDAR_SITE[1]+dy],
                [LIDAR_SITE[2], LIDAR_SITE[2]],
                color=C["scan"], alpha=0.12, linewidth=0.6)

    _lbl(ax, -0.05, 0, LIDAR_Z+0.08, "LiDAR 360°", C["scan"], fs=7)

    # ── Front RGB-D camera  (at 0.26, 0, 0.085 in base) ─────────────────────
    CAM = to_world([0.26, 0, 0.085])
    _add(ax, _box(*CAM, 0.030, 0.060, 0.030), C["camera"], alpha=0.95)
    _add(ax, _cyl(CAM[0]+0.018, CAM[1], CAM[2], 0.010, 0.006, axis="x"), "#000000")
    _lbl(ax, CAM[0]+0.06, CAM[1], CAM[2]+0.02, "RGB-D cam", C["camera"], fs=6.5)

    # ── IMU (at 0, 0, 0.075 in base = top of chassis) ────────────────────────
    IMU = to_world([0.10, 0, 0.078])
    _add(ax, _box(*IMU, 0.020, 0.020, 0.010), C["imu"], alpha=0.85)

    # ── 6-DOF Arm ─────────────────────────────────────────────────────────────
    POSES = {
        "home":       (0.0,  1.5708,  0.0,   0.0, 0.0, 0.0),
        "ready":      (0.0,  0.7854, -1.0472,0.0, 0.0, 0.0),
        "grasp_open": (0.0,  0.7854, -1.0472,0.0, 0.0, 0.0),
    }
    j1,j2,j3,j4,j5,j6 = POSES.get(pose, POSES["ready"])
    fk = arm_fk(j1, j2, j3, j4, j5, j6)

    # Convert all FK points to world frame
    def W(key): return to_world(fk[key])

    # Shoulder pan base → shoulder lift (vertical column, joint 1)
    _add(ax, _capsule(to_world(fk["sb"]), to_world(fk["sl"]), 0.065), C["arm"])

    # Upper arm: shoulder lift → elbow
    _add(ax, _capsule(W("sl"), W("el"), 0.035), C["arm"], alpha=0.93)

    # Forearm: elbow → wrist1
    _add(ax, _capsule(W("el"), W("w1"), 0.028), C["arm"], alpha=0.93)

    # Wrist segment 1→2
    _add(ax, _capsule(W("w1"), W("w2"), 0.025), C["wrist"], alpha=0.93)

    # Wrist segment 2→3
    _add(ax, _capsule(W("w2"), W("w3"), 0.022), C["wrist"], alpha=0.93)

    # Wrist segment 3→ee
    _add(ax, _capsule(W("w3"), W("ee"), 0.020), C["wrist"], alpha=0.90)

    # Gripper palm
    EE = W("ee"); R = fk["R"]
    palm_faces = _box(*EE, 0.035, 0.030, 0.015)
    _add(ax, palm_faces, C["gripper"])

    # Fingers
    FL, FR = W("fl"), W("fr")
    _add(ax, _capsule(EE, FL, 0.008), C["gripper"])
    _add(ax, _capsule(EE, FR, 0.008), C["gripper"])
    # Rubber tips
    _add(ax, _sphere(*FL, 0.010), C["tip"], alpha=0.97)
    _add(ax, _sphere(*FR, 0.010), C["tip"], alpha=0.97)

    # ── joint / sensor sites ──────────────────────────────────────────────────
    _site(ax, W("sl"), "#ffaa00", label="J2")
    _site(ax, W("el"), "#ffaa00", label="J3")
    _site(ax, W("w1"), "#ffaa00", label="J4")
    _site(ax, W("ee"), C["tip"],  label="TCP")

    # F/T sensor site (ft_site at 0.025m before ee along arm axis)
    ft_local = fk["w3"] + 0.025 * R @ np.array([1,0,0])
    _site(ax, to_world(ft_local), C["ft"], label="F/T")

    # Wrist camera site
    wcam_local = fk["w3"] + R @ np.array([-0.01, 0, 0.030])
    wcam = to_world(wcam_local)
    _add(ax, _box(*wcam, 0.012, 0.020, 0.012), C["camera"], alpha=0.90)

    # Arm label
    arm_mid = (W("sl") + W("el")) / 2
    _lbl(ax, arm_mid[0]+0.05, arm_mid[1]+0.10, arm_mid[2]+0.05,
         "6-DOF arm", C["arm"], fs=7)

    # ── world axes ────────────────────────────────────────────────────────────
    _axes(ax, 0, 0, 0, L=0.15, lw=1.5)
    ax.text(0.17,0,0,"X",color="#ff3333",fontsize=9,fontweight="bold")
    ax.text(0,0.17,0,"Y",color="#33ff33",fontsize=9,fontweight="bold")
    ax.text(0,0,0.17,"Z",color="#3399ff",fontsize=9,fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════

def build_legend():
    return [
        mpatches.Patch(color=C["chassis"], label="Chassis  0.50×0.40×0.15 m  8 kg"),
        mpatches.Patch(color=C["wheel"],   label="Drive wheels  r=0.10 m  ×2"),
        mpatches.Patch(color=C["arm"],     label="Arm links 1–3  (UR5 proportions)"),
        mpatches.Patch(color=C["wrist"],   label="Wrist joints 4–6"),
        mpatches.Patch(color=C["tip"],     label="Gripper fingertips"),
        mpatches.Patch(color=C["scan"],    label="LiDAR  360°  0.1–12 m"),
        mpatches.Patch(color=C["camera"],  label="RGB-D camera  +  wrist camera"),
        mpatches.Patch(color=C["ft"],      label="F/T sensor site"),
    ]


def print_summary(pose):
    print()
    print("═"*60)
    print(f"  Autorobo — full robot  (robot/robot.xml)  pose: {pose}")
    print("═"*60)
    rows = [
        ("chassis",     "0.50×0.40×0.15 m  8.0 kg  17 DoF total"),
        ("wheel ×2",    "r=0.10 m  w=0.06 m  velocity-controlled"),
        ("casters ×2",  "sphere r=0.03 m  passive rolling"),
        ("joint1",      "shoulder pan    Z-axis  ±180°"),
        ("joint2",      "shoulder lift   Y-axis  −90° to +180°"),
        ("joint3",      "elbow           Y-axis  ±180°"),
        ("joint4-6",    "wrist 3-axis    ±180°"),
        ("gripper",     "parallel jaw  open 0–40 mm  50 N tip force"),
        ("LiDAR",       "360°  0.1–12 m  mast-mounted"),
        ("RGB-D cam",   "58° FOV  0.3–6 m  front"),
        ("wrist cam",   "69° FOV  close-up grasp camera"),
        ("F/T sensor",  "6-axis  wrist_force + wrist_torque"),
        ("IMU",         "gyro + accel + quat  (6-axis)"),
    ]
    for name, desc in rows:
        print(f"  {name:<18} {desc}")
    print("═"*60)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--pose", default="ready",
                        choices=["home","ready","grasp_open"])
    args = parser.parse_args()

    print_summary(args.pose)

    fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
    ax  = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor("#0d1117")

    draw_robot(ax, pose=args.pose)

    # ── view bounds ──────────────────────────────────────────────────────────
    ax.set_xlim(-0.55, 0.85)
    ax.set_ylim(-0.55, 0.55)
    ax.set_zlim(-0.12, 0.95)
    ax.set_xlabel("X  (forward)", color="#666666", fontsize=9, labelpad=6)
    ax.set_ylabel("Y  (left)",    color="#666666", fontsize=9, labelpad=6)
    ax.set_zlabel("Z  (up)",      color="#666666", fontsize=9, labelpad=6)
    ax.tick_params(colors="#444444", labelsize=7)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#1e1e1e")
    ax.grid(True, color="#1a1a1a", linewidth=0.5)
    ax.view_init(elev=20, azim=-50)

    title = f"Autorobo — nav_robot  (robot/robot.xml)   pose: {args.pose}"
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)

    ax.legend(handles=build_legend(), loc="upper left",
              fontsize=7.5, framealpha=0.25,
              labelcolor="white", facecolor="#111111", edgecolor="#333333")

    fig.text(0.02, 0.01, "Drag to rotate  |  Scroll to zoom",
             color="#333333", fontsize=8)
    fig.text(0.98, 0.01, "Units: metres",
             color="#333333", fontsize=8, ha="right")

    plt.tight_layout()

    if args.save:
        out = "robot_model.png"
        plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#0d1117")
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
