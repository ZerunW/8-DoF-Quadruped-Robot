"""Periodic gait generator (no ROS): a GaitPreset (per-leg phase offsets, duty, period_s, step_height_mm,
Bezier swing) is selected by the body speed.

Leg i has phase p_i = (phi + offset_i) mod 1, lifts off at p_i = duty and touches down when p_i wraps to 0;
planted feet move with the ground at v_side = v -+ k w b (left -, right +). Feet are planned in the support
frame (see pose.py)."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from robot_dog_control.pose import BodyPose, PoseLimits, foot_in_leg

LEFT_LEGS = (0, 2)      # lf, lb
V_EPS = 1e-3            # m/s: below this the command counts as "stop"
W_EPS = 1e-2            # rad/s
HOME_MM = 1e-6          # a foot this close to its neutral foothold is home
_PHASE_EPS = 1e-9

#: keys of one preset in robot_dog.yaml (gait_presets.<name>.<key>)
PRESET_KEYS: Tuple[str, ...] = ('offsets', 'duty', 'period_s', 'step_height_mm', 'swing_x',
                                'swing_z')
#: default swing: cubic Bezier, smoothstep fore/aft (zero speed at both ends) and a
#: parabolic lift peaking at the step height mid-swing
DEFAULT_SWING_X: Tuple[float, ...] = (0.0, 0.0, 1.0, 1.0)
DEFAULT_SWING_Z: Tuple[float, ...] = (0.0, 4.0 / 3.0, 4.0 / 3.0, 0.0)


def bezier(ctrl: Sequence[float], s: float) -> float:
    """1-D Bezier curve through the control values ``ctrl`` at s in [0, 1] (de Casteljau)."""
    pts = list(ctrl)
    for r in range(1, len(pts)):
        for i in range(len(pts) - r):
            pts[i] += s * (pts[i + 1] - pts[i])
    return pts[0]


def bezier_slope_bound(ctrl: Sequence[float]) -> float:
    """Upper bound of |d bezier / ds| on [0, 1]: degree times the largest control step."""
    return (len(ctrl) - 1) * max(abs(b - a) for a, b in zip(ctrl, ctrl[1:]))


def _clip(x: float, half: float) -> float:
    return max(-half, min(half, x))


def _retard(current: float, target: float) -> float:
    """How far (cycle fraction, 0..1) a phase offset must be retarded to reach target."""
    d = (current - target) % 1.0
    return 0.0 if d < _PHASE_EPS or d > 1.0 - _PHASE_EPS else d


def shifted_pattern(current: Sequence[float], pattern: Sequence[float]) -> List[float]:
    """``pattern`` shifted by the common phase (same gait) that minimizes the largest
    retard from ``current``."""
    best: Tuple[float, List[float]] = (math.inf, list(pattern))
    for j in range(len(pattern)):
        c = current[j] - pattern[j]
        tgt = [(o + c) % 1.0 for o in pattern]
        tgt[j] = current[j]
        worst = max(_retard(a, b) for a, b in zip(current, tgt))
        if worst < best[0] - _PHASE_EPS:
            best = (worst, tgt)
    return best[1]


class GaitFoot(NamedTuple):
    dx: float           # mm, fore/aft offset from the neutral foothold (support frame)
    lift: float         # mm, height above the support plane
    in_stance: bool     # planned stance


@dataclass(frozen=True)
class GaitPreset:
    """One gait as data."""

    offsets: Tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)
    duty: float = 0.6
    period_s: float = 0.8
    step_height_mm: float = 25.0
    swing_x: Tuple[float, ...] = DEFAULT_SWING_X
    swing_z: Tuple[float, ...] = DEFAULT_SWING_Z

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any], name: str = '') -> 'GaitPreset':
        """From a robot_dog.yaml preset (every key required). Raises ValueError."""
        where = f'gait_presets.{name}' if name else 'gait preset'
        missing = [k for k in PRESET_KEYS if k not in m]
        unknown = sorted(set(m) - set(PRESET_KEYS))
        if missing or unknown:
            raise ValueError(f'{where}: missing keys {missing}, unknown keys {unknown} '
                             f'(a preset is exactly {list(PRESET_KEYS)})')
        try:
            preset = cls(tuple(float(v) for v in m['offsets']), float(m['duty']),
                         float(m['period_s']), float(m['step_height_mm']),
                         tuple(float(v) for v in m['swing_x']),
                         tuple(float(v) for v in m['swing_z']))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{where}: {exc}') from None
        problems = preset.validate()
        if problems:
            raise ValueError(f'{where}: ' + '; '.join(problems))
        return preset

    def as_dict(self) -> Dict[str, Any]:
        return {'offsets': list(self.offsets), 'duty': self.duty, 'period_s': self.period_s,
                'step_height_mm': self.step_height_mm, 'swing_x': list(self.swing_x),
                'swing_z': list(self.swing_z)}

    def validate(self) -> List[str]:
        p = []
        nums = (self.duty, self.period_s, self.step_height_mm) + tuple(self.offsets) \
            + tuple(self.swing_x) + tuple(self.swing_z)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in nums):
            return ['all values must be finite numbers']
        if len(self.offsets) != 4 or not all(0.0 <= o < 1.0 for o in self.offsets):
            p.append('offsets must be 4 phase fractions in [0, 1) (lf, rf, lb, rb)')
        if not 0.1 <= self.duty <= 0.95:
            p.append('duty must be in 0.1..0.95 (< 0.5 has flight phases)')
        if not 0.2 <= self.period_s <= 5.0:
            p.append('period_s must be in 0.2..5 s')
        if not 0.0 <= self.step_height_mm <= 100.0:
            p.append('step_height_mm must be in 0..100')
        sx, sz = self.swing_x, self.swing_z
        if len(sx) != len(sz) or len(sx) < 2:
            p.append('swing_x and swing_z must have the same length >= 2')
        elif not (sx[0] == 0.0 and sx[-1] == 1.0 and sz[0] == 0.0 and sz[-1] == 0.0):
            p.append('the swing must start at (0, 0) and end at (1, 0): swing_x[0] = 0, '
                     'swing_x[-1] = 1, swing_z[0] = swing_z[-1] = 0')
        elif not (all(-1.0 <= v <= 2.0 for v in sx) and all(0.0 <= v <= 3.0 for v in sz)):
            p.append('swing_x control points must be in -1..2 (outside 0..1: the foot moves '
                     'backward at lift-off / touch-down), swing_z in 0..3')
        return p

    @property
    def swing_time_s(self) -> float:
        return (1.0 - self.duty) * self.period_s

    def swing(self, s: float) -> Tuple[float, float]:
        """(fore/aft progress 0..1, lift fraction) at swing progress s."""
        return bezier(self.swing_x, s), bezier(self.swing_z, s)

    def shape_key(self) -> Tuple:
        """What the planned foot positions depend on (offsets/duty/period do not)."""
        return (self.step_height_mm, tuple(self.swing_x), tuple(self.swing_z))

    def mix(self, other: 'GaitPreset', t: float) -> 'GaitPreset':
        """Interpolate scalars and swing control points toward ``other`` (t in 0..1),
        keeping this preset's offsets."""
        lerp = lambda a, b: a + t * (b - a)  # noqa: E731
        return GaitPreset(self.offsets, lerp(self.duty, other.duty),
                          lerp(self.period_s, other.period_s),
                          lerp(self.step_height_mm, other.step_height_mm),
                          tuple(lerp(a, b) for a, b in zip(self.swing_x, other.swing_x)),
                          tuple(lerp(a, b) for a, b in zip(self.swing_z, other.swing_z)))


class Knot(NamedTuple):
    speed: float            # m/s gait speed at which this preset applies
    name: str
    preset: GaitPreset


def knot_index(knots: Sequence[Knot], s: float) -> int:
    """Index of the last knot whose speed is <= s (0 below the first)."""
    j = 0
    while j + 1 < len(knots) and s >= knots[j + 1].speed:
        j += 1
    return j


def scheduled_scalars(knots: Sequence[Knot], s: float) -> GaitPreset:
    """Duty, period, step height and swing at gait speed s: linear between knots, constant
    beyond the ends. The offsets are the lower knot's and are not used."""
    j = knot_index(knots, s)
    if j + 1 >= len(knots) or s <= knots[0].speed:
        return knots[j].preset
    a, b = knots[j], knots[j + 1]
    return a.preset.mix(b.preset, (s - a.speed) / (b.speed - a.speed))


@dataclass
class GaitParams:
    schedule: Tuple[Knot, ...] = field(default_factory=lambda: (Knot(0.0, 'trot', GaitPreset()),))
    hysteresis_mps: float = 0.01
    switch_periods: float = 1.0
    max_stride_mm: float = 40.0
    max_speed_mps: float = 0.08
    max_turn_rps: float = 0.6
    turn_stride_gain: float = 1.0
    ramp_time_s: float = 0.5
    half_track_mm: float = 50.0
    #: yaw rate (rad/s) the stride keeps in reserve at the speed limit
    turn_headroom_rps: float = 0.0

    @classmethod
    def single(cls, preset: GaitPreset, **kw) -> 'GaitParams':
        """One preset at every speed."""
        return cls(schedule=(Knot(0.0, 'preset', preset),), **kw)

    def validate(self) -> List[str]:
        p = []
        if not self.schedule:
            return ['the gait schedule needs at least one knot']
        for k in self.schedule:
            p += [f'gait_presets.{k.name}: {m}' for m in k.preset.validate()]
        speeds = [k.speed for k in self.schedule]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in speeds):
            p.append('gait_schedule_speeds_mps must be finite')
        elif speeds[0] != 0.0 or any(b <= a for a, b in zip(speeds, speeds[1:])):
            p.append('gait_schedule_speeds_mps must start at 0 and increase strictly')
        if len({len(k.preset.swing_x) for k in self.schedule}) > 1:
            p.append('the scheduled presets must have swing curves with the same number of '
                     'control points (they are interpolated)')
        vals = dict(max_stride_mm=self.max_stride_mm, max_speed_mps=self.max_speed_mps,
                    max_turn_rps=self.max_turn_rps, turn_stride_gain=self.turn_stride_gain,
                    ramp_time_s=self.ramp_time_s, half_track_mm=self.half_track_mm,
                    hysteresis_mps=self.hysteresis_mps, switch_periods=self.switch_periods,
                    turn_headroom_rps=self.turn_headroom_rps)
        for k, v in vals.items():
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                p.append(f'{k} must be finite')
        if p:
            return p
        if not 0.0 <= self.max_stride_mm <= 400.0:
            p.append('max_stride_mm must be in 0..400')
        if not 0.0 < self.max_speed_mps <= 1.0:
            p.append('max_speed_mps must be in (0, 1]')
        if not 0.0 <= self.max_turn_rps <= 5.0:
            p.append('max_turn_rps must be in 0..5')
        if self.turn_stride_gain < 0.0 or self.ramp_time_s < 0.0 or self.half_track_mm <= 0.0:
            p.append('turn_stride_gain, ramp_time_s must be >= 0 and half_track_mm > 0')
        if self.hysteresis_mps < 0.0:
            p.append('gait_switch_hysteresis_mps must be >= 0')
        if not 0.5 <= self.switch_periods <= 10.0:
            p.append('gait_switch_periods must be in 0.5..10 (below 0.5 a blend would run a '
                     'leg phase backward)')
        if self.turn_headroom_rps < 0.0:
            p.append('the turning headroom (heading_max_rps) must be >= 0')
        return p

    def stride_at(self, s: float) -> float:
        """Stride (mm) the schedule needs at gait speed s, plus the turning headroom."""
        g = scheduled_scalars(self.schedule, s)
        head = self.turn_stride_gain * self.turn_headroom_rps * self.half_track_mm / 1000.0
        return (s + head) * g.duty * g.period_s * 1000.0

    def speed_limit(self) -> float:
        """Highest body speed (m/s): ``max_speed_mps``, or the lowest speed at which the
        scheduled stride ``s D(s) T(s)`` plus the turning headroom reaches ``max_stride_mm``."""
        cap = self.max_speed_mps
        smax = self.max_stride_mm
        if self.stride_at(0.0) > smax:
            return 0.0
        pts = sorted({cap * i / 400.0 for i in range(1, 401)}
                     | {k.speed for k in self.schedule if 0.0 < k.speed < cap})
        prev = 0.0
        for s in pts:
            if self.stride_at(s) > smax:
                lo, hi = prev, s
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    if self.stride_at(mid) > smax:
                        hi = mid
                    else:
                        lo = mid
                return lo
            prev = s
        return cap


class GaitGenerator:
    IDLE, ACTIVE = 'idle', 'active'

    def __init__(self, params: GaitParams):
        problems = params.validate()
        if problems:
            raise ValueError('invalid gait parameters: ' + '; '.join(problems))
        self.params = params
        self.v_max = params.speed_limit()
        self.knots = tuple(params.schedule)
        self.knot = 0                                   # knot whose phase pattern is used
        self.offsets = list(self.knots[0].preset.offsets)
        self.target = list(self.offsets)                # pattern being blended to
        self.preset = self.knots[0].preset              # scheduled scalars in use
        self.reset()

    @property
    def duty(self) -> float:
        return self.preset.duty

    def leg_phase(self, i: int) -> float:
        return (self.phase + self.offsets[i]) % 1.0

    def in_stance(self, i: int) -> bool:
        return self.mode == self.IDLE or not self.swinging[i]

    def is_idle(self) -> bool:
        return self.mode == self.IDLE

    def reset(self) -> None:
        """Return to idle at the neutral footholds at once; use only when not walking."""
        self.phase = 0.0
        self.mode = self.IDLE
        self.amp = 0.0          # gait amplitude A (0..1): body speed and landing point
        self.lift_amp = 0.0     # lift amplitude L (0..1)
        self.v = 0.0            # ramped body speed (m/s)
        self.body_speed = 0.0   # planned body speed A (v_left + v_right) / 2 (m/s)
        self.w = 0.0            # ramped yaw rate (rad/s)
        self.speed = 0.0        # gait speed s (m/s)
        self.strides = [0.0] * 4        # A * S per leg (mm)
        self.x = [0.0] * 4              # foot fore/aft offset per leg (mm, support frame)
        self.lift = [0.0] * 4           # foot height per leg (mm)
        self.x0 = [0.0] * 4             # lift-off point of the current / last swing (mm)
        self.land = [0.0] * 4           # landing point of the current / last swing (mm)
        self.swinging = [False] * 4
        self._ph_lift = [0.0] * 4       # leg phase at lift-off

    def set_turn_headroom(self, rps: float) -> None:
        """Change the speed limit's turning headroom (rad/s) live; applies from the next
        update."""
        self.params = replace(self.params, turn_headroom_rps=float(rps))
        self.v_max = self.params.speed_limit()

    @staticmethod
    def _ramp(cur: float, target: float, rate: float, dt: float) -> float:
        if rate <= 0.0:
            return target
        step = rate * dt
        d = target - cur
        return target if abs(d) <= step else cur + math.copysign(step, d)

    def _landing(self, i: int, half: float) -> float:
        """Landing point of leg i's swing: A S/2, kept inside the validated swings
        (``|L| <= half`` and ``|x0 + L| <= half``, see ``gait_patch``)."""
        x0 = self.x0[i]
        self.land[i] = min(half, half - x0, max(-half, -half - x0, 0.5 * self.strides[i]))
        return self.land[i]

    def _start(self, knot: int) -> None:
        self.reset()
        self.mode = self.ACTIVE
        self.knot = knot
        self.offsets = list(self.knots[knot].preset.offsets)
        self.target = list(self.offsets)

    def update(self, dt: float, v_cmd: float, w_cmd: float, step: bool = False,
               w_steer: float = 0.0) -> List[GaitFoot]:
        """Advance by dt (s) and return the 4 feet (lf, rf, lb, rb).

        v_cmd (m/s), w_cmd (rad/s) and step request the gait. The heading hold's ``w_steer``
        (rad/s) only corrects a running gait; it never starts it or keeps it running."""
        p = self.params
        knots = self.knots
        dt = max(0.0, float(dt))
        v_max = self.v_max
        v_cmd = max(-v_max, min(v_max, v_cmd))
        moving = abs(v_cmd) > V_EPS or abs(w_cmd) > W_EPS
        if not moving:
            v_cmd = w_cmd = 0.0
        requested = moving or bool(step)
        w_cmd = max(-p.max_turn_rps, min(p.max_turn_rps, w_cmd + w_steer))

        if self.mode == self.IDLE:
            if not requested:
                return [GaitFoot(0.0, 0.0, True)] * 4
            self._start(knot_index(knots, abs(v_cmd)))

        R = p.ramp_time_s
        rate = 1.0 / R if R > 0 else 0.0
        self.v = self._ramp(self.v, v_cmd, v_max * rate, dt)
        self.w = self._ramp(self.w, w_cmd, p.max_turn_rps * rate, dt)
        self.amp = self._ramp(self.amp, 1.0 if requested else 0.0, rate, dt)
        home = all(abs(x) <= HOME_MM for x in self.x)
        self.lift_amp = self._ramp(self.lift_amp, 1.0 if requested else (
            0.0 if home else self.lift_amp), rate, dt)

        # speed schedule: scalars interpolated; the phase pattern switches with hysteresis,
        # only at full amplitude (start and stop ramps keep their pattern)
        s = self.speed = abs(self.v)
        if self.amp >= 1.0:
            k0 = self.knot
            while self.knot + 1 < len(knots) and s >= knots[self.knot + 1].speed:
                self.knot += 1
            while self.knot > 0 and s < knots[self.knot].speed - p.hysteresis_mps:
                self.knot -= 1
            if self.knot != k0:
                self.target = shifted_pattern(self.offsets, knots[self.knot].preset.offsets)
        g = self.preset = scheduled_scalars(knots, s)
        T, D = g.period_s, g.duty

        # planted-foot speed per side (m/s), limited to a full stride
        b = p.half_track_mm / 1000.0
        k = p.turn_stride_gain
        v_lim = p.max_stride_mm / (D * T * 1000.0)
        side_v = (max(-v_lim, min(v_lim, self.v - k * self.w * b)),
                  max(-v_lim, min(v_lim, self.v + k * self.w * b)))
        a = self.amp
        self.body_speed = a * 0.5 * (side_v[0] + side_v[1])

        # Advance the phase, then blend toward a new pattern by retarding planted legs only,
        # at most half a cycle per switch_periods periods and inside the validated stride.
        ph0 = [self.leg_phase(i) for i in range(4)]
        self.phase = (self.phase + dt / T) % 1.0
        r_max = 0.5 / (p.switch_periods * T) * dt
        half = 0.5 * p.max_stride_mm
        for i in range(4):
            back = _retard(self.offsets[i], self.target[i])
            if back == 0.0:
                self.offsets[i] = self.target[i]
                continue
            if self.swinging[i]:
                continue
            vb = a * side_v[0 if i in LEFT_LEGS else 1] * 1000.0      # mm/s, planted foot: -vb
            d = min(r_max, back)
            if abs(vb) > 1e-9:
                # lift-off point if the stance is held d longer: inside the validated stride
                room = (half + math.copysign(1.0, vb) * self.x[i]) / (abs(vb) * T) - (D - ph0[i])
                d = min(d, max(0.0, room))
            if d > 0.0:
                self.offsets[i] = (self.offsets[i] - d) % 1.0
                if _retard(self.offsets[i], self.target[i]) == 0.0:
                    self.offsets[i] = self.target[i]

        h = g.step_height_mm
        la = self.lift_amp
        out = []
        for i in range(4):
            sv = side_v[0 if i in LEFT_LEGS else 1]
            vb = a * sv * 1000.0                    # mm/s the planted foot moves backward
            self.strides[i] = a * sv * D * T * 1000.0       # A S (mm)
            dph = (self.leg_phase(i) - ph0[i]) % 1.0   # phase progress this tick, >= 0
            if dph > 0.5:
                dph = 0.0
            end = ph0[i] + dph                      # unwrapped leg phase after this tick
            if not self.swinging[i]:
                if end >= D:
                    # lift-off during this tick: stance up to the lift-off phase, then swing
                    lift_ph = max(D, ph0[i])
                    frac = (lift_ph - ph0[i]) / dph if dph > 0.0 else 0.0
                    self.x0[i] = _clip(self.x[i] - vb * frac * dt, half)
                    self._ph_lift[i] = lift_ph
                    self.swinging[i] = True
                else:
                    # planted: moves with the ground, clipped to the validated stride
                    self.x[i] = _clip(self.x[i] - vb * dt, half)
                    self.lift[i] = 0.0
            elif end >= 1.0:
                # touch-down (the leg phase wrapped): land, then planted
                frac = (end - 1.0) / dph if dph > 0.0 else 0.0
                self.x[i] = _clip(self._landing(i, half) - vb * frac * dt, half)
                self.lift[i] = 0.0
                self.swinging[i] = False
            if self.swinging[i]:
                sp = (end - self._ph_lift[i]) / (1.0 - self._ph_lift[i])
                sx, sz = g.swing(min(1.0, max(0.0, sp)))
                x0 = self.x0[i]
                self.x[i] = x0 + (self._landing(i, half) - x0) * sx
                self.lift[i] = la * h * sz
            out.append(GaitFoot(self.x[i], self.lift[i], not self.swinging[i]))

        if (not requested and a == 0.0 and all(abs(x) <= HOME_MM for x in self.x)
                and all(z == 0.0 for z in self.lift)):
            self.reset()
            return [GaitFoot(0.0, 0.0, True)] * 4
        return out


# Envelope: every foot pose the controller can plan
STAND_CHECK = 'stand_height_mm/stand_fx_mm'
REST_CHECK = 'rest_height_mm'
POSE_CHECK = 'height_min_mm/height_max_mm/pose_pitch_max_deg/pose_roll_max_deg/pose_x_max_mm'
GAIT_CHECK_PREFIX = 'max_stride_mm/gait_presets.'

class Scenario(tuple):
    """Planned feet at one body pose: per leg (lf, rf, lb, rb), a tuple of leg-frame
    (fx, fz). ``pose`` names that body pose in envelope messages (None if hand-made)."""

    pose: Optional[BodyPose] = None

#: (lift-off x0, landing L) pairs of the envelope patch as fractions of the stride: the
#: corners of the set |x0|, |L|, |x0 + L| <= 1/2 that every swing stays in, plus the swing
#: in place
PATCH_SWINGS: Tuple[Tuple[float, float], ...] = ((-0.5, 0.5), (0.5, -0.5), (-0.5, 0.0),
                                                 (0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
                                                 (0.0, 0.0))


def gait_patch(preset: GaitPreset, max_stride_mm: float, n: int = 7
               ) -> List[Tuple[float, float]]:
    """Support-frame (dx, lift) points that bound every foot position of the gait.

    The stance line at full stride plus full-lift swings between the ``PATCH_SWINGS``
    corners."""
    S, h = max_stride_mm, preset.step_height_mm
    lin = [k / (n - 1) for k in range(n)]
    pts = [(S * (0.5 - s), 0.0) for s in lin]
    for x0, xl in PATCH_SWINGS:
        for s in lin:
            sx, sz = preset.swing(s)
            pts.append((S * (x0 + (xl - x0) * sx), h * sz))
    return pts


def scenario(pose: BodyPose, hips: Sequence[Tuple[float, float]], stand_fx_mm: float,
             patch: Sequence[Tuple[float, float]]) -> Scenario:
    """Leg-frame points of every leg for the support-frame patch at this body pose."""
    sc = Scenario(tuple(foot_in_leg(pose, hx, hy, hx + stand_fx_mm + dx, z) for dx, z in patch)
                  for hx, hy in hips)
    sc.pose = pose
    return sc


def schedule_shapes(knots: Sequence[Knot]) -> 'OrderedDict[str, GaitPreset]':
    """Gait shapes a schedule can produce: every knot and the midpoint between adjacent
    knots, labeled a~b."""
    out: 'OrderedDict[str, GaitPreset]' = OrderedDict()
    for k in knots:
        out[k.name] = k.preset
    for a, b in zip(knots, knots[1:]):
        out[f'{a.name}~{b.name}'] = a.preset.mix(b.preset, 0.5)
    return out


def envelope_checks(shapes: Mapping[str, GaitPreset], max_stride_mm: float,
                    limits: PoseLimits, stand_fx_mm: float, stand_height_mm: float,
                    rest_height_mm: float, n: int = 7
                    ) -> Tuple['OrderedDict[str, List[Scenario]]', Dict[str, str]]:
    """Scenario groups for ``kinematics.validate_envelope`` and each gait shape's check name:
    the stand and rest poses, the neutral footholds at every pose sample of ``limits``, and
    each swing shape's patch at every sample of the walking set."""
    hips = limits.hips
    neutral = [(0.0, 0.0)]
    checks: 'OrderedDict[str, List[Scenario]]' = OrderedDict()
    checks[STAND_CHECK] = [scenario(BodyPose(stand_height_mm), hips, stand_fx_mm, neutral)]
    checks[REST_CHECK] = [scenario(BodyPose(rest_height_mm), hips, stand_fx_mm, neutral)]
    checks[POSE_CHECK] = [scenario(q, hips, stand_fx_mm, neutral) for q in limits.samples()]
    poses = limits.walking().samples()
    groups: 'OrderedDict[Tuple, List[str]]' = OrderedDict()
    for name in shapes:
        groups.setdefault(shapes[name].shape_key(), []).append(name)
    names: Dict[str, str] = {}
    for members in groups.values():
        cname = GAIT_CHECK_PREFIX + ','.join(members)
        patch = gait_patch(shapes[members[0]], max_stride_mm, n)
        checks[cname] = [scenario(q, hips, stand_fx_mm, patch) for q in poses]
        for m in members:
            names[m] = cname
    return checks, names


def same_for_all_legs(points: Sequence[Tuple[float, float]], n_legs: int = 4) -> List[Scenario]:
    """One scenario with the same leg-frame points on every leg."""
    pts = tuple((float(x), float(z)) for x, z in points)
    return [Scenario(pts for _ in range(n_legs))]


def all_presets(presets: Mapping[str, Mapping[str, Any]]) -> Dict[str, GaitPreset]:
    return {name: GaitPreset.from_mapping(m, name) for name, m in presets.items()}


def default_presets() -> Dict[str, Dict[str, Any]]:
    """Fallback presets: one default trot."""
    return {'trot': GaitPreset().as_dict()}


def stance_count(feet: Sequence[GaitFoot]) -> int:
    return sum(1 for f in feet if f.in_stance)
