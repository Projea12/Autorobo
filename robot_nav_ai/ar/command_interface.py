"""
ar/command_interface.py — Natural-language command interface for AR robot.

Type a command in the terminal while the AR window is running:
    "forward"       → robot drives forward
    "back"          → robot reverses
    "left"          → pivot left
    "right"         → pivot right
    "stop"          → halt all motion
    "arm up"        → raise arm to ready pose
    "arm down"      → lower arm to home pose
    "open"          → open gripper
    "close"         → close gripper
    "wave"          → wave arm animation
    "home"          → reset everything to home keyframe
    "quit" / "q"    → exit

Commands are fuzzy-matched — "go forward", "move ahead", "drive forward" all work.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np


# ── command types ─────────────────────────────────────────────────────────────

class Cmd(Enum):
    STOP          = auto()
    FORWARD       = auto()
    BACKWARD      = auto()
    TURN_LEFT     = auto()
    TURN_RIGHT    = auto()
    ARM_UP        = auto()
    ARM_DOWN      = auto()
    GRIPPER_OPEN  = auto()
    GRIPPER_CLOSE = auto()
    WAVE          = auto()
    HOME          = auto()
    QUIT          = auto()


# ── keyword → command map ─────────────────────────────────────────────────────

_KEYWORDS: list[tuple[list[str], Cmd]] = [
    (["forward", "ahead", "advance", "go"],           Cmd.FORWARD),
    (["back", "backward", "reverse", "retreat"],      Cmd.BACKWARD),
    (["left", "rotate left", "spin left"],            Cmd.TURN_LEFT),
    (["right", "rotate right", "spin right"],         Cmd.TURN_RIGHT),
    (["stop", "halt", "freeze", "wait", "stand"],     Cmd.STOP),
    (["arm up", "raise", "lift arm", "reach"],        Cmd.ARM_UP),
    (["arm down", "lower", "retract"],                Cmd.ARM_DOWN),
    (["open", "release", "drop"],                     Cmd.GRIPPER_OPEN),
    (["close", "grab", "grasp", "pick"],              Cmd.GRIPPER_CLOSE),
    (["wave", "hello", "hi"],                         Cmd.WAVE),
    (["home", "reset", "origin"],                     Cmd.HOME),
    (["quit", "exit", "bye", "q"],                    Cmd.QUIT),
]


def parse(text: str) -> Optional[Cmd]:
    """Map natural-language text to a Cmd. Returns None if unrecognised."""
    t = text.strip().lower()
    for keywords, cmd in _KEYWORDS:
        if any(kw in t for kw in keywords):
            return cmd
    return None


# ── robot ctrl applier ────────────────────────────────────────────────────────

@dataclass
class CtrlConfig:
    wheel_speed:   float = 3.0    # rad/s — differential-drive velocity
    turn_speed:    float = 2.5    # rad/s — pivot turn velocity
    arm_home:      list  = None   # joint1..6 home positions (radians)
    arm_ready:     list  = None   # joint1..6 ready positions
    gripper_open:  float = 0.04   # metres
    gripper_close: float = 0.0    # metres

    def __post_init__(self):
        if self.arm_home is None:
            self.arm_home  = [0.0, -1.5708, 0.0, 0.0, 0.0, 0.0]
        if self.arm_ready is None:
            self.arm_ready = [0.0, -0.7854, 1.0472, 0.0, 0.0, 0.0]


class RobotController:
    """
    Translates Cmd values into MuJoCo ctrl signals.

    ctrl layout (9 values):
        [0] drive_left   — velocity (rad/s)
        [1] drive_right  — velocity (rad/s)
        [2-7] arm_j1..6 — position (rad)
        [8]  gripper     — position (m)
    """

    def __init__(self, cfg: CtrlConfig = CtrlConfig()) -> None:
        self.cfg      = cfg
        self._cmd     = Cmd.STOP
        self._lock    = threading.Lock()
        self._t_wave  = 0.0     # wave animation timer

    def set_command(self, cmd: Cmd) -> None:
        with self._lock:
            self._cmd    = cmd
            self._t_wave = 0.0

    def current(self) -> Cmd:
        with self._lock:
            return self._cmd

    def apply(self, ctrl: np.ndarray, t_sim: float) -> None:
        """Write ctrl values for the current command. Called from physics thread."""
        with self._lock:
            cmd = self._cmd

        cfg = self.cfg

        # ── wheel control ──────────────────────────────────────────────────
        if cmd == Cmd.FORWARD:
            ctrl[0] =  cfg.wheel_speed
            ctrl[1] =  cfg.wheel_speed
        elif cmd == Cmd.BACKWARD:
            ctrl[0] = -cfg.wheel_speed
            ctrl[1] = -cfg.wheel_speed
        elif cmd == Cmd.TURN_LEFT:
            ctrl[0] = -cfg.turn_speed
            ctrl[1] =  cfg.turn_speed
        elif cmd == Cmd.TURN_RIGHT:
            ctrl[0] =  cfg.turn_speed
            ctrl[1] = -cfg.turn_speed
        else:
            ctrl[0] = 0.0
            ctrl[1] = 0.0

        # ── arm control ────────────────────────────────────────────────────
        if cmd == Cmd.HOME:
            for i, v in enumerate(cfg.arm_home):
                ctrl[2 + i] = v
        elif cmd == Cmd.ARM_UP:
            for i, v in enumerate(cfg.arm_ready):
                ctrl[2 + i] = v
        elif cmd == Cmd.ARM_DOWN:
            for i, v in enumerate(cfg.arm_home):
                ctrl[2 + i] = v
        elif cmd == Cmd.WAVE:
            self._t_wave += 0.002   # step with physics dt
            ctrl[2] = 0.0
            ctrl[3] = -0.8 + 0.4 * np.sin(self._t_wave * 3.0)
            ctrl[4] =  1.0
            ctrl[5] =  0.5 * np.sin(self._t_wave * 6.0)

        # ── gripper control ────────────────────────────────────────────────
        if cmd == Cmd.GRIPPER_OPEN:
            ctrl[8] = cfg.gripper_open
        elif cmd in (Cmd.GRIPPER_CLOSE, Cmd.HOME):
            ctrl[8] = cfg.gripper_close


# ── input thread ──────────────────────────────────────────────────────────────

class CommandInterface:
    """
    Reads terminal input in a background thread.
    Calls on_command(Cmd) when a command is recognised.
    Calls on_quit() when user types quit/q.
    """

    def __init__(
        self,
        on_command: Callable[[Cmd], None],
        on_quit:    Callable[[], None],
    ) -> None:
        self._on_command = on_command
        self._on_quit    = on_quit
        self._thread     = threading.Thread(
            target=self._loop, daemon=True
        )

    def start(self) -> None:
        self._thread.start()
        print("\n[cmd] Command interface ready.")
        print("[cmd] Commands: forward | back | left | right | stop |")
        print("[cmd]           arm up | arm down | open | close | wave | home | quit\n")

    def _loop(self) -> None:
        while True:
            try:
                text = input("robot> ").strip()
            except (EOFError, KeyboardInterrupt):
                self._on_quit()
                return

            if not text:
                continue

            cmd = parse(text)
            if cmd is None:
                print(f"[cmd] Unknown: '{text}'  — try: forward, stop, arm up, wave …")
                continue

            if cmd == Cmd.QUIT:
                print("[cmd] Quitting …")
                self._on_quit()
                return

            print(f"[cmd] → {cmd.name}")
            self._on_command(cmd)
