"""Servo configuration and validation, the clamp, angle to pulse conversion and the rate
limiter, without ROS. Every pulse on a PCA9685 pin comes from here; ``servo_driver_node`` is
the only caller that writes its output to I2C.

Angles are servo degrees in ``[0, servo_range_deg)``, indexed by logical servo index ``i``
with PCA9685 output ``channel[i]``. Pulse math::

    a  = clamp(angle, min_i, max_i)
    a += offset_i
    a  = a mod servo_range
    us = pulse_min + a / servo_range * (pulse_max - pulse_min)
    us = clip(round(us), pulse_min, pulse_max)
    counts = us * 4096 // 20000         # 20 ms frame
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_log = logging.getLogger(__name__)

NUM_SERVOS = 8

#: Servo labels by index ("top" = thigh motor m1, "bottom" = calf motor m2).
SERVO_LABELS: Tuple[str, ...] = (
    'left_front_bottom',   # 0
    'left_front_top',      # 1
    'left_back_bottom',    # 2
    'left_back_top',       # 3
    'right_front_bottom',  # 4
    'right_front_top',     # 5
    'right_back_bottom',   # 6
    'right_back_top',      # 7
)

#: Fallback defaults.
DEFAULT_CHANNELS: Tuple[int, ...] = (3, 2, 12, 13, 4, 5, 11, 10)
DEFAULT_OFFSETS: Tuple[float, ...] = (15.0, 5.0, 15.0, 10.0, 3.0, -5.0, 3.0, 8.0)
DEFAULT_MINS: Tuple[float, ...] = (140.0, 150.0, 140.0, 150.0, 140.0, 150.0, 140.0, 150.0)
DEFAULT_MAXS: Tuple[float, ...] = (220.0, 210.0, 220.0, 210.0, 220.0, 210.0, 220.0, 210.0)
DEFAULT_STAND: Tuple[float, ...] = (172.0,) * NUM_SERVOS

#: Largest calibration offset accepted (deg).
MAX_ABS_OFFSET_DEG = 40.0
#: Largest offset change accepted in one runtime update (deg).
MAX_OFFSET_STEP_DEG = 10.0
#: Absolute pulse width window (us).
ABS_PULSE_MIN_US = 100.0
ABS_PULSE_MAX_US = 3000.0
#: Upper bound for the servo speed limit (deg/s).
MAX_SERVO_SPEED_DPS = 2000.0
PCA9685_FRAME_US = 20000
PCA9685_COUNTS = 4096


class ServoConfigError(ValueError):
    """Raised for an invalid servo configuration; the driver does not start."""


def _is_real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite(x) -> bool:
    return _is_real(x) and math.isfinite(x)


@dataclass(frozen=True)
class ServoConfig:
    """Validated servo table, indexed by logical servo index 0..7.

    Construction raises ServoConfigError on any problem.
    """

    channels: Tuple[int, ...] = DEFAULT_CHANNELS
    offsets: Tuple[float, ...] = DEFAULT_OFFSETS
    mins: Tuple[float, ...] = DEFAULT_MINS
    maxs: Tuple[float, ...] = DEFAULT_MAXS
    stand: Tuple[float, ...] = DEFAULT_STAND
    pulse_min_us: float = 500.0
    pulse_max_us: float = 2500.0
    servo_range_deg: float = 360.0
    labels: Tuple[str, ...] = SERVO_LABELS

    def __post_init__(self):
        for name in ('channels', 'offsets', 'mins', 'maxs', 'stand', 'labels'):
            value = getattr(self, name)
            try:
                object.__setattr__(self, name, tuple(value))
            except TypeError:
                raise ServoConfigError(f'{name}: expected a sequence, got {value!r}')
        problems = validate_servo_config(self)
        if problems:
            raise ServoConfigError('invalid servo configuration: ' + '; '.join(problems))

    def with_offsets(self, offsets: Sequence[float]) -> 'ServoConfig':
        """Return a validated copy with new calibration offsets."""
        return replace(self, offsets=tuple(offsets))

    def as_dict(self) -> Dict[str, list]:
        return {
            'servo_channel': list(self.channels),
            'servo_offset': list(self.offsets),
            'servo_min': list(self.mins),
            'servo_max': list(self.maxs),
            'stand_angles': list(self.stand),
        }


def validate_servo_config(cfg: ServoConfig) -> List[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    problems: List[str] = []
    n = NUM_SERVOS
    for name in ('channels', 'offsets', 'mins', 'maxs', 'stand', 'labels'):
        if len(getattr(cfg, name)) != n:
            problems.append(f'{name} must have exactly {n} entries (got {len(getattr(cfg, name))})')
    if problems:
        return problems

    for name in ('pulse_min_us', 'pulse_max_us', 'servo_range_deg'):
        if not _finite(getattr(cfg, name)):
            problems.append(f'{name} must be a finite number (got {getattr(cfg, name)!r})')
    if problems:
        return problems
    if not (ABS_PULSE_MIN_US <= cfg.pulse_min_us < cfg.pulse_max_us <= ABS_PULSE_MAX_US):
        problems.append(
            f'pulse range must satisfy {ABS_PULSE_MIN_US} <= pulse_min_us < pulse_max_us <= '
            f'{ABS_PULSE_MAX_US} (got {cfg.pulse_min_us}..{cfg.pulse_max_us})')
    if not (0.0 < cfg.servo_range_deg <= 360.0):
        problems.append(f'servo_range_deg must be in (0, 360] (got {cfg.servo_range_deg})')

    seen = {}
    for i, ch in enumerate(cfg.channels):
        if not isinstance(ch, int) or isinstance(ch, bool):
            problems.append(f'servo_channel[{i}] must be an integer (got {ch!r})')
            continue
        if not 0 <= ch <= 15:
            problems.append(f'servo_channel[{i}]={ch} outside PCA9685 channels 0..15')
        if ch in seen:
            problems.append(f'servo_channel[{i}]={ch} duplicates servo_channel[{seen[ch]}]')
        seen[ch] = i

    if len(set(cfg.labels)) != n or not all(isinstance(s, str) and s for s in cfg.labels):
        problems.append('labels must be 8 unique non-empty strings')

    rng = cfg.servo_range_deg
    for i in range(n):
        lo, hi, off, st = cfg.mins[i], cfg.maxs[i], cfg.offsets[i], cfg.stand[i]
        bad = [k for k, v in (('servo_min', lo), ('servo_max', hi), ('servo_offset', off),
                              ('stand_angles', st)) if not _finite(v)]
        if bad:
            problems.append(f'servo {i}: {", ".join(bad)} must be finite numbers')
            continue
        if not (0.0 <= lo < hi <= rng):
            problems.append(f'servo {i}: need 0 <= servo_min < servo_max <= {rng} (got {lo}..{hi})')
        if abs(off) > MAX_ABS_OFFSET_DEG:
            problems.append(f'servo {i}: |servo_offset|={abs(off)} exceeds {MAX_ABS_OFFSET_DEG} deg')
        if lo + off < 0.0 or hi + off >= rng:
            problems.append(
                f'servo {i}: limits+offset ({lo + off}..{hi + off}) must stay inside [0, {rng}) '
                'so the pulse never wraps around')
        if not (lo <= st <= hi):
            problems.append(f'servo {i}: stand_angles={st} outside limits {lo}..{hi}')
    return problems


def check_offset_update(old: ServoConfig, new_offsets: Sequence[float]) -> ServoConfig:
    """Validate a runtime offset update; return the new config or raise ServoConfigError.

    Besides the normal validation, each offset may change by at most MAX_OFFSET_STEP_DEG.
    """
    new_offsets = tuple(new_offsets)
    if len(new_offsets) != NUM_SERVOS:
        raise ServoConfigError(f'servo_offset must have {NUM_SERVOS} entries')
    for i, (a, b) in enumerate(zip(old.offsets, new_offsets)):
        if not _finite(b):
            raise ServoConfigError(f'servo_offset[{i}] must be finite (got {b!r})')
        if abs(b - a) > MAX_OFFSET_STEP_DEG:
            raise ServoConfigError(
                f'servo_offset[{i}] changes by {b - a:+.2f} deg; at most '
                f'{MAX_OFFSET_STEP_DEG} deg per update (offsets are not rate limited)')
    return old.with_offsets(new_offsets)


@dataclass(frozen=True)
class PulseFrame:
    """Result of ServoLimiter.process(): everything the driver needs for one write."""

    angles: Tuple[float, ...]          # clamped commanded angles (servo deg, before offset)
    clamped: Tuple[bool, ...]          # True where the request was outside [min, max]
    nonfinite: Tuple[bool, ...]        # True where the request was NaN/inf, replaced by stand
    pulses_us: Tuple[int, ...]         # per logical index
    by_channel: Dict[int, int]         # {pca_channel: pulse_us}


class ServoLimiter:
    """The clamp: every angle that becomes a pulse goes through angle_to_us()."""

    def __init__(self, config: ServoConfig):
        if not isinstance(config, ServoConfig):
            raise TypeError('ServoLimiter needs a validated ServoConfig')
        self.config = config

    def clamp_one(self, i: int, angle) -> Tuple[float, bool, bool]:
        """Clamp one angle to [min_i, max_i]; NaN, inf and non-numbers become stand_i.

        Returns (clamped_angle, was_clamped, was_nonfinite).
        """
        cfg = self.config
        try:
            a = float(angle)
        except (TypeError, ValueError):
            a = math.nan
        if not math.isfinite(a):
            return float(cfg.stand[i]), True, True
        lo, hi = cfg.mins[i], cfg.maxs[i]
        if a < lo:
            return float(lo), True, False
        if a > hi:
            return float(hi), True, False
        return a, False, False

    def clamp(self, angles: Sequence[float]) -> Tuple[List[float], List[bool]]:
        """Clamp all 8 angles. Returns (clamped, was_clamped_mask)."""
        if len(angles) != NUM_SERVOS:
            raise ValueError(f'expected {NUM_SERVOS} angles, got {len(angles)}')
        out, mask = [], []
        for i, a in enumerate(angles):
            c, was, _ = self.clamp_one(i, a)
            out.append(c)
            mask.append(was)
        return out, mask

    def angle_to_us(self, i: int, angle) -> int:
        """Servo angle -> pulse width in us (always inside [pulse_min, pulse_max])."""
        cfg = self.config
        a, _, nonfinite = self.clamp_one(i, angle)
        if nonfinite:
            _log.warning('servo %d: non-finite angle %r replaced by stand angle %.1f',
                         i, angle, cfg.stand[i])
        a = a + cfg.offsets[i]
        a = math.fmod(a, cfg.servo_range_deg)
        if a < 0.0:
            a += cfg.servo_range_deg
        us = cfg.pulse_min_us + a / cfg.servo_range_deg * (cfg.pulse_max_us - cfg.pulse_min_us)
        us = int(math.floor(us + 0.5))   # round half up
        lo, hi = int(math.ceil(cfg.pulse_min_us)), int(math.floor(cfg.pulse_max_us))
        return min(max(us, lo), hi)

    def process(self, angles: Sequence[float]) -> PulseFrame:
        """Clamp all 8 angles and compute their pulses as one frame."""
        if len(angles) != NUM_SERVOS:
            raise ValueError(f'expected {NUM_SERVOS} angles, got {len(angles)}')
        cl, mask, nf, pulses = [], [], [], []
        for i, a in enumerate(angles):
            c, was, nonfinite = self.clamp_one(i, a)
            cl.append(c)
            mask.append(was)
            nf.append(nonfinite)
            pulses.append(self.angle_to_us(i, c))
        by_channel = {self.config.channels[i]: pulses[i] for i in range(NUM_SERVOS)}
        return PulseFrame(tuple(cl), tuple(mask), tuple(nf), tuple(pulses), by_channel)


def us_to_counts(us: float, frame_us: int = PCA9685_FRAME_US) -> int:
    """Pulse width (us) to PCA9685 OFF count (integer math, 20 ms frame)."""
    return int(us) * PCA9685_COUNTS // frame_us


class RateLimiter:
    """Per-servo speed limit: each step moves at most ``max_speed_dps * dt`` degrees.

    The first ``step()`` without a prior ``reset()`` snaps to the target.
    """

    def __init__(self, max_speed_dps: float, n: int = NUM_SERVOS, max_dt: float = 0.1):
        self.n = n
        self.max_dt = max_dt
        self.set_speed(max_speed_dps)
        self._state: Optional[List[float]] = None

    def set_speed(self, max_speed_dps: float) -> None:
        if not _finite(max_speed_dps) or not 0.0 < max_speed_dps <= MAX_SERVO_SPEED_DPS:
            raise ServoConfigError(
                f'max_servo_speed_dps must be in (0, {MAX_SERVO_SPEED_DPS}] (got {max_speed_dps!r})')
        self.max_speed_dps = float(max_speed_dps)

    @property
    def state(self) -> Optional[List[float]]:
        return None if self._state is None else list(self._state)

    def reset(self, pose: Iterable[float]) -> None:
        pose = [float(p) for p in pose]
        if len(pose) != self.n or not all(math.isfinite(p) for p in pose):
            raise ValueError('reset pose must be %d finite values' % self.n)
        self._state = pose

    def step(self, target: Sequence[float], dt: float) -> List[float]:
        if len(target) != self.n:
            raise ValueError(f'expected {self.n} targets, got {len(target)}')
        if self._state is None:
            finite = [float(t) if _finite(t) else None for t in target]
            if any(t is None for t in finite):
                raise ValueError('first command must be finite')
            self._state = finite
            return list(self._state)
        if not _finite(dt) or dt < 0.0:
            dt = 0.0
        max_step = self.max_speed_dps * min(dt, self.max_dt)
        for i, t in enumerate(target):
            if not _finite(t):
                continue  # hold this servo
            d = t - self._state[i]
            if d > max_step:
                d = max_step
            elif d < -max_step:
                d = -max_step
            self._state[i] += d
        return list(self._state)

    def at_target(self, target: Sequence[float], tol: float = 1e-6) -> bool:
        if self._state is None:
            return False
        return all(abs(s - t) <= tol for s, t in zip(self._state, target) if _finite(t))
