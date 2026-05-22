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
    PICK          = auto()   # "pick up <object>" — triggers full grasp pipeline
    QUIT          = auto()


# ── keyword → command map ─────────────────────────────────────────────────────

_KEYWORDS: list[tuple[list[str], Cmd]] = [
    (["forward", "ahead", "advance", "go"],                      Cmd.FORWARD),
    (["back", "backward", "reverse", "retreat"],                 Cmd.BACKWARD),
    (["left", "rotate left", "spin left"],                       Cmd.TURN_LEFT),
    (["right", "rotate right", "spin right"],                    Cmd.TURN_RIGHT),
    (["stop", "halt", "freeze", "wait", "stand"],                Cmd.STOP),
    (["arm up", "raise", "lift arm", "reach"],                   Cmd.ARM_UP),
    (["arm down", "lower", "retract"],                           Cmd.ARM_DOWN),
    (["open", "release", "drop"],                                Cmd.GRIPPER_OPEN),
    # PICK must come before GRIPPER_CLOSE so "pick up X" is not swallowed by "pick"
    (["pick up", "pick me", "get me", "fetch", "bring me",
      "grab the", "take the", "get the"],                        Cmd.PICK),
    (["close", "grab", "grasp", "pick"],                         Cmd.GRIPPER_CLOSE),
    (["wave", "hello", "hi"],                                    Cmd.WAVE),
    (["home", "reset", "origin"],                                Cmd.HOME),
    (["quit", "exit", "bye", "q"],                               Cmd.QUIT),
]


def parse(text: str) -> Optional[Cmd]:
    """Map natural-language text to a Cmd. Returns None if unrecognised."""
    t = text.strip().lower()
    for keywords, cmd in _KEYWORDS:
        if any(kw in t for kw in keywords):
            return cmd
    return None


def parse_pick_target(text: str) -> str:
    """
    Extract the object name from a PICK command.

    Examples
    --------
    "pick up the mug"   → "mug"
    "get me a cup"      → "cup"
    "fetch the bottle"  → "bottle"
    "grab the red mug"  → "red mug"
    """
    t = text.strip().lower()
    # Strip leading trigger phrases
    triggers = [
        "pick up the", "pick up a", "pick up",
        "get me the", "get me a", "get me", "get the",
        "fetch the", "fetch a", "fetch",
        "bring me the", "bring me a", "bring me",
        "grab the", "grab a", "take the", "take a",
    ]
    for trigger in sorted(triggers, key=len, reverse=True):
        if t.startswith(trigger):
            return t[len(trigger):].strip()
    return t


# ── robot ctrl applier ────────────────────────────────────────────────────────

@dataclass
class CtrlConfig:
    move_speed:    float = 0.05    # metres per physics step (base x/y position)
    turn_speed:    float = 0.03    # radians per physics step (base yaw)

    # TidyBot home arm positions (ctrl[3..9] = joint_1..7)
    arm_home:  list = None
    arm_ready: list = None   # raised / ready pose


    def __post_init__(self):
        # From TidyBot "home" keyframe
        if self.arm_home is None:
            self.arm_home  = [0.0, 0.26179939, 3.14159265, -2.26892803,
                              0.0, 0.95993109, 1.57079633]
        # Raised: lift joint_2 up, extend slightly
        if self.arm_ready is None:
            self.arm_ready = [0.0, -0.34906585, 3.14159265, -2.54818071,
                              0.0, -0.87266463, 1.57079633]


class RobotController:
    """
    Translates Cmd values into TidyBot MuJoCo ctrl signals.

    TidyBot ctrl layout (11 values):
        [0]  joint_x    — base x position (m)
        [1]  joint_y    — base y position (m)
        [2]  joint_th   — base yaw (rad)
        [3]  joint_1    — Kinova arm joint 1 (rad)
        [4]  joint_2    — Kinova arm joint 2 (rad)
        [5]  joint_3    — Kinova arm joint 3 (rad)
        [6]  joint_4    — Kinova arm joint 4 (rad)
        [7]  joint_5    — Kinova arm joint 5 (rad)
        [8]  joint_6    — Kinova arm joint 6 (rad)
        [9]  joint_7    — Kinova arm joint 7 (rad)
        [10] fingers    — gripper (0=open, 255=closed)
    """

    def __init__(self, cfg: CtrlConfig = CtrlConfig()) -> None:
        self.cfg     = cfg
        self._cmd    = Cmd.STOP
        self._lock   = threading.Lock()
        self._t_wave = 0.0

        # Accumulated base pose (position-controlled base)
        self._base_x  = 0.0
        self._base_y  = 0.0
        self._base_th = 0.0

        # Last arm positions — held between arm commands so arm never collapses
        self._arm_pos = list(cfg.arm_home)

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

        # ── base motion (position accumulation) ───────────────────────────
        if cmd == Cmd.FORWARD:
            self._base_x += cfg.move_speed * np.sin(self._base_th)
            self._base_y += cfg.move_speed * np.cos(self._base_th)
        elif cmd == Cmd.BACKWARD:
            self._base_x -= cfg.move_speed * np.sin(self._base_th)
            self._base_y -= cfg.move_speed * np.cos(self._base_th)
        elif cmd == Cmd.TURN_LEFT:
            self._base_th -= cfg.turn_speed
        elif cmd == Cmd.TURN_RIGHT:
            self._base_th += cfg.turn_speed
        elif cmd == Cmd.HOME:
            self._base_x  = 0.0
            self._base_y  = 0.0
            self._base_th = 0.0

        ctrl[0] = self._base_x
        ctrl[1] = self._base_y
        ctrl[2] = self._base_th

        # ── arm control — always write arm positions so joints never collapse ──
        if cmd in (Cmd.HOME, Cmd.ARM_DOWN):
            self._arm_pos = list(cfg.arm_home)
        elif cmd == Cmd.ARM_UP:
            self._arm_pos = list(cfg.arm_ready)
        elif cmd == Cmd.WAVE:
            self._t_wave += 0.002
            self._arm_pos = [
                0.0,
                0.26179939 + 0.5 * np.sin(self._t_wave * 3.0),
                3.14159265,
                -2.26892803 + 0.4 * np.sin(self._t_wave * 2.0),
                0.0,
                0.95993109,
                1.57079633,
            ]

        # Always write stored arm position — holds pose between commands
        for i, v in enumerate(self._arm_pos):
            ctrl[3 + i] = v

        # ── gripper control ────────────────────────────────────────────────
        if cmd == Cmd.GRIPPER_OPEN:
            ctrl[10] = 0.0
        elif cmd in (Cmd.GRIPPER_CLOSE, Cmd.HOME):
            ctrl[10] = 200.0


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
            # Pass raw text so PICK handler can extract object name
            try:
                self._on_command(cmd, text)
            except TypeError:
                self._on_command(cmd)
