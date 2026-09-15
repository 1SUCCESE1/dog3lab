"""Terminal keyboard velocity controller (termios, no omni.appwindow needed).

W/S = forward/back, A/D = strafe, Q/E = turn, Space = stop, Ctrl+C = quit.
Ported from DDT_Lab: scripts/keyboard_vel_controller.py, adapted for the
Instinct-RL VecEnv wrapper (`env.unwrapped` is the manager-based RL env).
"""

import select
import sys
import termios
import threading
import tty

import torch


class _TTYKeyboard:
    def __init__(self):
        self._running = False
        self._pressed: set[str] = set()
        self._lock = threading.Lock()

    @property
    def pressed(self) -> set[str]:
        with self._lock:
            return set(self._pressed)

    def _read(self):
        while self._running:
            if select.select([sys.stdin], [], [], 0.02)[0]:
                try:
                    ch = sys.stdin.read(1)
                except (OSError, ValueError):
                    continue
                key = None
                if ch == "\x1b":
                    try:
                        seq = sys.stdin.read(1) + sys.stdin.read(1)
                    except (OSError, ValueError):
                        continue
                    key = {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}.get(seq)
                elif ch.lower() in "wasdqezxl ":
                    key = ch.upper() if ch != " " else "SPACE"
                if key:
                    with self._lock:
                        self._pressed.add(key)

    def start(self):
        self._running = True
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        except Exception:
            pass


class VelocityKeyboardController:
    """Maps terminal keyboard input to base_velocity commands in the env."""

    def __init__(self, env_cfg, sim_device: str = "cuda:0"):
        ranges = env_cfg.commands.base_velocity.ranges
        self.vx = ranges.lin_vel_x[1]
        self.vy = ranges.lin_vel_y[1]
        self.wz = ranges.ang_vel_z[1]
        self._kb = _TTYKeyboard()
        self._kb.start()
        print("=" * 60)
        print("  键盘控制:  W/S=前进/后退  A/D=侧移  Q/E=旋转  Space=停止")
        print("=" * 60)

    def apply_to_env(self, env, command_name: str = "base_velocity") -> None:
        k = self._kb.pressed
        vx = (1.0 if "W" in k or "UP" in k else 0.0) + (-1.0 if "S" in k or "DOWN" in k else 0.0)
        vy = (1.0 if "A" in k or "LEFT" in k else 0.0) + (-1.0 if "D" in k or "RIGHT" in k else 0.0)
        wz = (1.0 if "Q" in k else 0.0) + (-1.0 if "E" in k else 0.0)
        cmd = torch.tensor(
            [[vx * self.vx, vy * self.vy, wz * self.wz]],
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        term = env.unwrapped.command_manager._terms[command_name]
        term.vel_command_b[:, 0] = cmd[0, 0]
        term.vel_command_b[:, 1] = cmd[0, 1]
        term.vel_command_b[:, 2] = cmd[0, 2]

    def reset(self):
        pass
