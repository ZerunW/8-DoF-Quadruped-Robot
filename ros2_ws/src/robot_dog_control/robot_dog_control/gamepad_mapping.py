"""Gamepad to behavior commands: the pure mapping (no ROS).

L1 held = deadman: the dog steps in place and the sticks drive it as a tank (v = (vL + vR) / 2,
w = (vR - vL) / (2 half_track)). Cross calls ~/estop, Options ~/stand; R1/R2 raise and lower the body."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import List, Optional, Sequence, Tuple


@dataclass
class MapperParams:
    """Button and axis indices (hid_playstation order) and scalings."""
    axis_left_y: int = 1
    axis_right_y: int = 4
    button_cross: int = 0
    button_options: int = 9
    button_l1: int = 4
    button_r1: int = 5
    button_r2: int = 7
    axis_r2: int = 5
    # sign applied to each stick axis
    sign_left_y: float = 1.0
    sign_right_y: float = 1.0
    deadzone: float = 0.1
    max_speed_mps: float = 0.08           # the node clamps it to the controller's max_speed_mps
    max_turn_rps: float = 0.5             # the node clamps it to the controller's max_turn_rps
    half_track_m: float = 0.0882          # the node reads it from the controller's leg_pos_y_mm
    joy_timeout_s: float = 0.5
    height_raise_rate_mmps: float = 80.0  # R1
    height_lower_rate_mmps: float = 120.0  # R2
    height_min_mm: float = 130.0          # the node reads these three from the controller
    height_max_mm: float = 175.0
    stand_height_mm: float = 155.0

    def __post_init__(self):
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError('deadzone must be in [0, 1)')
        if not (math.isfinite(self.half_track_m) and self.half_track_m > 0.0):
            raise ValueError('half_track_m must be finite and > 0')
        if not (self.height_min_mm <= self.stand_height_mm <= self.height_max_mm):
            raise ValueError('need height_min_mm <= stand_height_mm <= height_max_mm')
        for name in ('max_speed_mps', 'max_turn_rps', 'joy_timeout_s',
                     'height_raise_rate_mmps', 'height_lower_rate_mmps'):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f'{name} must be finite and >= 0')

    @classmethod
    def names(cls) -> List[str]:
        return [f.name for f in fields(cls)]


@dataclass
class Commands:
    vel: Optional[Tuple[float, float]] = None           # (linear.x m/s, angular.z rad/s)
    step: Optional[bool] = None                         # /cmd_step
    estop: bool = False                                 # call ~/estop
    stand: bool = False                                 # call ~/stand
    pose: Optional[Tuple[float, float, float]] = None   # /cmd_pose (height m, pitch, roll rad)

    def merge(self, other: 'Commands') -> 'Commands':
        return Commands(other.vel if other.vel is not None else self.vel,
                        other.step if other.step is not None else self.step,
                        self.estop or other.estop, self.stand or other.stand,
                        other.pose if other.pose is not None else self.pose)


def apply_deadzone(value: float, deadzone: float) -> float:
    """Zero inside the deadzone, else rescaled linearly from 0 at its edge to +-1 at full scale."""
    if not math.isfinite(value):
        return 0.0
    value = max(-1.0, min(1.0, value))
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)


class GamepadMapper:
    def __init__(self, params: MapperParams):
        self.p = params
        self._axes: Sequence[float] = ()
        self._buttons: Sequence[int] = ()
        self._prev: Sequence[int] = ()
        self._last_joy: Optional[float] = None
        self._need_release = True      # L1 must be seen up before it arms
        self._moving = False           # a non-zero / step command was sent since the last zero
        self._timed_out = True
        self._height_mm = params.stand_height_mm
        self._last_pose_tick: Optional[float] = None
        self._r2_rest: Optional[float] = None

    def _btn(self, idx: int) -> bool:
        return 0 <= idx < len(self._buttons) and bool(self._buttons[idx])

    def _edge(self, idx: int) -> bool:
        was = 0 <= idx < len(self._prev) and bool(self._prev[idx])
        return self._btn(idx) and not was

    def _axis(self, idx: int, sign: float) -> float:
        v = self._axes[idx] if 0 <= idx < len(self._axes) else 0.0
        return apply_deadzone(sign * v, self.p.deadzone)

    def _r2_pressed(self) -> bool:
        if self._btn(self.p.button_r2):
            return True
        i = self.p.axis_r2
        if self._r2_rest is None or not 0 <= i < len(self._axes):
            return False
        return abs(self._axes[i] - self._r2_rest) > 1.5

    def _height_command(self, now: float,
                        last: Optional[float]) -> Optional[Tuple[float, float, float]]:
        dt = 0.0 if last is None else max(0.0, min(now - last, 0.1))
        direction = int(self._btn(self.p.button_r1)) - int(self._r2_pressed())
        if direction == 0 or dt == 0.0:
            return None
        p = self.p
        rate = p.height_raise_rate_mmps if direction > 0 else p.height_lower_rate_mmps
        new = max(p.height_min_mm, min(p.height_max_mm,
                                       self._height_mm + direction * rate * dt))
        if new == self._height_mm:
            return None
        self._height_mm = new
        return (new / 1000.0, 0.0, 0.0)

    @property
    def deadman(self) -> bool:
        return self._btn(self.p.button_l1) and not self._need_release

    def body_command(self, left: float, right: float) -> Tuple[float, float]:
        """Tank mixing: stick values (-1..1, up +1) -> (v m/s, w rad/s, +w CCW), clamped."""
        p = self.p
        vl, vr = left * p.max_speed_mps, right * p.max_speed_mps
        v = (vl + vr) / 2.0
        w = (vr - vl) / (2.0 * p.half_track_m)
        return (max(-p.max_speed_mps, min(p.max_speed_mps, v)),
                max(-p.max_turn_rps, min(p.max_turn_rps, w)))

    def _stop(self) -> Commands:
        """One zero command if something non-zero may be active."""
        if not self._moving:
            return Commands()
        self._moving = False
        return Commands(vel=(0.0, 0.0), step=False)

    def on_joy(self, axes: Sequence[float], buttons: Sequence[int], now: float) -> Commands:
        """Process one Joy message: estop / stand edges and the deadman."""
        p = self.p
        self._axes, self._buttons = axes, buttons
        # learn the R2 trigger axis rest value (-1 or +1, depending on the driver)
        if self._r2_rest is None and 0 <= p.axis_r2 < len(axes) and abs(axes[p.axis_r2]) >= 0.9:
            self._r2_rest = float(axes[p.axis_r2])
        if len(self._prev) == 0:
            self._prev = [0] * len(buttons)
        self._last_joy = now
        self._timed_out = False
        out = Commands()
        if not self._btn(p.button_l1):
            self._need_release = False
        if self._edge(p.button_cross):
            out.estop = True
            self._need_release = True
            out = out.merge(self._stop())
        if self._edge(p.button_options):
            out.stand = True
            self._height_mm = p.stand_height_mm
        if not self.deadman:
            out = out.merge(self._stop())
        self._prev = list(buttons)
        return out

    def tick(self, now: float) -> Commands:
        """20 Hz: stream the held walking commands, detect a lost controller."""
        if self._last_joy is None:
            return Commands()
        if now - self._last_joy > self.p.joy_timeout_s:
            if not self._timed_out:
                self._timed_out = True
                self._need_release = True
                return self._stop()
            return Commands()
        if not self.deadman:
            return self._stop()
        left = self._axis(self.p.axis_left_y, self.p.sign_left_y)
        right = self._axis(self.p.axis_right_y, self.p.sign_right_y)
        self._moving = True
        return Commands(vel=self.body_command(left, right), step=True)

    def pose_tick(self, now: float) -> Commands:
        """Height command while R1 or R2 is held; call it at the controller's rate."""
        last, self._last_pose_tick = self._last_pose_tick, now
        if self._last_joy is None or now - self._last_joy > self.p.joy_timeout_s:
            return Commands()
        return Commands(pose=self._height_command(now, last))
