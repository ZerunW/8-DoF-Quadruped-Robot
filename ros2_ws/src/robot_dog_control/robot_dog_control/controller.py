"""GaitControllerCore: the gait_controller node without ROS.

Per tick: heading hold -> gait -> body pose + leveling -> foot_in_leg -> IK -> startup/ESTOP blend
-> front/back leg guard (CAD unsafe-pair set) -> LegMap -> 8 servo angles."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass, field, fields, replace
from enum import IntEnum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from robot_dog_control.gait import (POSE_CHECK, PRESET_KEYS, REST_CHECK, STAND_CHECK, V_EPS,
                                    W_EPS, GaitGenerator, GaitParams, GaitPreset, Knot,
                                    all_presets, default_presets, envelope_checks,
                                    schedule_shapes, stance_count)
from robot_dog_control.joint_region import JointRegion
from robot_dog_control.kinematics import (EnvelopeReport, LegGeometry, LegKinematics,
                                          validate_envelope)
from robot_dog_control.leg_map import LEG_NAMES, LegMap
from robot_dog_control.pid import PID
from robot_dog_control.pose import BodyPose, PoseLimits, foot_in_leg
from robot_dog_control.unsafe_pairs import SidePairs, load_sides

_R2D = 180.0 / math.pi
_D2R = math.pi / 180.0
#: Slew of the leveling authority between STAND and WALK (deg/s).
LEVEL_LIMIT_SLEW_DPS = 30.0
#: Slew of the leveling gain scale (per s).
LEVEL_SCALE_SLEW_PER_S = 2.0


#: Leveling deadband hysteresis: once the tilt error leaves level_deadband_deg, the PIDs
#: run until it is back under this fraction of it.
LEVEL_DEADBAND_RELEASE = 0.25


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class State(IntEnum):
    """STAND and WALK level the body; REST crouches without leveling; ESTOP freezes the output."""
    STAND = 0
    WALK = 1
    REST = 2
    ESTOP = 3


# /gait/debug (std_msgs/Float64MultiArray) layout. Angles in deg, lengths in mm.
DEBUG_FIELDS: Tuple[str, ...] = (
    'state', 'phase', 'v_cmd', 'w_cmd', 'roll_deg', 'pitch_deg', 'yaw_deg',
    'u_roll', 'u_pitch', 'u_yaw', 'i_roll', 'i_pitch', 'kin_height_mm',
    'walk_allowed', 'level_active', 'heading_active', 'ik_max_err_mm', 'ws_clamped_legs',
) + tuple(f'{leg}_{k}' for leg in LEG_NAMES for k in ('fx', 'fz', 'm1', 'm2', 'stance')) + (
    'i_yaw',            # heading-hold integrator (rad/s)
    'pose_height_mm',   # applied body pose (command + leveling), support frame
    'pose_pitch_deg',
    'pose_roll_deg',
    'pose_x_mm',
    'stance_legs',      # planned stance feet (0 = flight phase)
    'gait_amp',         # gait amplitude A (0..1)
    'step_cmd',         # /cmd_step active
    'gait_speed',       # ramped body speed |v| (m/s); selects the gait
    'gait_knot',        # schedule knot whose phase pattern is used
    'gait_duty',        # scheduled duty
    'gait_period_s',    # scheduled period
    'heading_deg',      # IMU yaw relative to ~/zero_yaw, the /cmd_heading frame
    'interleg_holds',   # ticks a side's front/back legs were held by the pair guard
    'body_speed_mps',   # planned body speed along the heading (m/s)
)
DEBUG_INDEX = {name: i for i, name in enumerate(DEBUG_FIELDS)}
DEBUG_LAYOUT_LABEL = 'gait_debug_v6'


@dataclass
class ControllerParams:
    """All gait_controller parameters under their ROS names; ``geom`` and ``gait_presets``
    flatten to ``geom.<key>`` and ``gait_presets.<name>.<key>``."""

    rate_hz: float = 50.0
    imu_topic: str = '/bno055/imu'
    leg_names: List[str] = field(default_factory=lambda: list(LEG_NAMES))
    leg_thigh_idx: List[int] = field(default_factory=lambda: [1, 5, 3, 7])
    leg_calf_idx: List[int] = field(default_factory=lambda: [0, 4, 2, 6])
    leg_mirror_servo: List[bool] = field(default_factory=lambda: [False, True, False, True])
    # -1: knees point backward, canonical +x toward the tail.
    leg_x_sign: List[int] = field(default_factory=lambda: [-1, -1, -1, -1])
    # Hip (thigh axis) positions from the CAD.
    leg_pos_x_mm: List[float] = field(default_factory=lambda: [91.5, 91.5, -91.5, -91.5])
    leg_pos_y_mm: List[float] = field(default_factory=lambda: [88.2, -88.2, 88.2, -88.2])
    geom: Dict[str, float] = field(default_factory=lambda: LegGeometry().as_dict())
    # Feasible joint region, an m2 interval per m1 knot (see joint_region.py).
    ik_region_m1: List[float] = field(default_factory=lambda: [60.0, 120.0])
    ik_region_m2_lo: List[float] = field(default_factory=lambda: [60.0, 60.0])
    ik_region_m2_hi: List[float] = field(default_factory=lambda: [100.0, 100.0])
    stand_height_mm: float = 155.0
    stand_fx_mm: float = 20.0      # neutral foothold ahead of the hip (support frame)
    rest_height_mm: float = 145.0
    # Body pose limits (pose.PoseLimits): every hip between height_min/max above the
    # support plane, |pitch|/pitch_max + |roll|/roll_max <= 1, |x| <= x_max.
    height_min_mm: float = 145.0
    height_max_mm: float = 165.0
    # lowest hip height while the gait runs; 0 means height_min_mm
    walk_height_min_mm: float = 0.0
    pose_pitch_max_deg: float = 8.0
    pose_roll_max_deg: float = 8.0
    pose_x_max_mm: float = 0.0
    pose_rate_dps: float = 20.0     # slew of the commanded pitch / roll
    # Gait presets (see gait.py) and the speed schedule: gait_schedule[k] applies at gait
    # speed gait_schedule_speeds_mps[k].
    gait_presets: Dict[str, Dict[str, Any]] = field(default_factory=default_presets)
    gait_schedule: List[str] = field(default_factory=lambda: ['trot'])
    gait_schedule_speeds_mps: List[float] = field(default_factory=lambda: [0.0])
    gait_switch_hysteresis_mps: float = 0.01
    gait_switch_periods: float = 1.0
    max_stride_mm: float = 40.0
    max_speed_mps: float = 0.08
    max_turn_rps: float = 0.6
    turn_stride_gain: float = 1.0
    ramp_time_s: float = 0.5
    cmd_vel_timeout_s: float = 1.0
    level_enable: bool = True
    level_kp: float = 1.25          # deg of correction per deg of tilt error
    level_ki: float = 0.31          # deg per deg*s
    level_kd: float = 0.031         # deg per deg/s (gyro)
    level_lpf_hz: float = 3.0
    # Below this tilt error (deg) the leveling output is held.
    level_deadband_deg: float = 0.0
    level_max_deg: float = 15.0     # bound on |u_roll| + |u_pitch|
    level_walk_gain_scale: float = 0.5
    level_target_roll_deg: float = 0.0
    level_target_pitch_deg: float = 0.0
    heading_hold_enable: bool = True
    heading_kp: float = 1.5
    heading_ki: float = 0.0          # rad/s per rad*s
    heading_kd: float = 0.1
    heading_max_rps: float = 0.1  # output limit (rad/s)
    heading_use_lateral_accel: bool = False
    heading_lateral_gain: float = 2.0
    heading_lateral_leak_s: float = 1.0
    # Per-leg foot trims (mm): height (+ toward the hip) and fore/aft (+ toward the nose).
    leg_trim_mm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    leg_trim_x_mm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    servo_min: List[float] = field(
        default_factory=lambda: [140.0, 150.0, 140.0, 150.0, 140.0, 150.0, 140.0, 150.0])
    servo_max: List[float] = field(
        default_factory=lambda: [220.0, 210.0, 220.0, 210.0, 220.0, 210.0, 220.0, 210.0])
    startup_blend_s: float = 2.0
    height_rate_mmps: float = 30.0
    imu_timeout_s: float = 0.5

    def to_flat(self) -> Dict[str, Any]:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == 'geom':
                for k, gv in v.items():
                    out[f'geom.{k}'] = float(gv)
            elif f.name == 'gait_presets':
                for name, preset in v.items():
                    for k, pv in preset.items():
                        out[f'gait_presets.{name}.{k}'] = (
                            [float(x) for x in pv] if isinstance(pv, (list, tuple))
                            else float(pv))
            else:
                out[f.name] = list(v) if isinstance(v, list) else v
        return out

    @classmethod
    def from_flat(cls, flat: Dict[str, Any], base: Optional['ControllerParams'] = None
                  ) -> 'ControllerParams':
        """Build from ROS-style names, coercing to the declared types. Raises ValueError."""
        base = base or cls()
        kwargs = {}
        geom = dict(base.geom)
        presets = copy.deepcopy(base.gait_presets)
        for key, value in flat.items():
            if key.startswith('geom.'):
                g = key[5:]
                if g not in geom:
                    raise ValueError(f'unknown geometry parameter {key}')
                geom[g] = _coerce(value, 0.0, key)
                continue
            if key.startswith('gait_presets.'):
                parts = key.split('.')
                if len(parts) != 3 or parts[2] not in PRESET_KEYS or not parts[1]:
                    raise ValueError(f'unknown gait preset parameter {key} (expected '
                                     f'gait_presets.<name>.<{"|".join(PRESET_KEYS)}>)')
                like = [0.0] if parts[2] in ('offsets', 'swing_x', 'swing_z') else 0.0
                presets.setdefault(parts[1], {})[parts[2]] = _coerce(value, like, key)
                continue
            if not hasattr(base, key) or key in ('geom', 'gait_presets'):
                raise ValueError(f'unknown parameter {key}')
            kwargs[key] = _coerce(value, getattr(base, key), key)
        kwargs['geom'] = geom
        kwargs['gait_presets'] = presets
        return replace(base, **kwargs)

    def presets(self) -> Dict[str, GaitPreset]:
        """All presets, validated; raises ValueError naming the bad preset."""
        return all_presets(self.gait_presets)

    def knots(self) -> Tuple[Knot, ...]:
        if len(self.gait_schedule) != len(self.gait_schedule_speeds_mps):
            raise ValueError('gait_schedule and gait_schedule_speeds_mps must have the same '
                             'length')
        if not self.gait_schedule:
            raise ValueError('gait_schedule needs at least one preset')
        missing = [n for n in self.gait_schedule if n not in self.gait_presets]
        if missing:
            raise ValueError(f'gait_schedule names unknown presets {missing} (gait_presets: '
                             f'{sorted(self.gait_presets)})')
        return tuple(Knot(float(v), n, GaitPreset.from_mapping(self.gait_presets[n], n))
                     for n, v in zip(self.gait_schedule, self.gait_schedule_speeds_mps))

    def gait_params(self) -> GaitParams:
        return GaitParams(schedule=self.knots(), hysteresis_mps=self.gait_switch_hysteresis_mps,
                          switch_periods=self.gait_switch_periods,
                          max_stride_mm=self.max_stride_mm, max_speed_mps=self.max_speed_mps,
                          max_turn_rps=self.max_turn_rps,
                          turn_stride_gain=self.turn_stride_gain, ramp_time_s=self.ramp_time_s,
                          half_track_mm=sum(abs(y) for y in self.leg_pos_y_mm) / 4.0,
                          turn_headroom_rps=self.heading_max_rps)

    def pose_limits(self) -> PoseLimits:
        return PoseLimits(self.height_min_mm, self.height_max_mm,
                          self.pose_pitch_max_deg * _D2R, self.pose_roll_max_deg * _D2R,
                          self.pose_x_max_mm,
                          tuple((float(x), float(y)) for x, y in zip(self.leg_pos_x_mm,
                                                                     self.leg_pos_y_mm)),
                          self.walk_height_min_mm)

    def validate(self) -> List[str]:
        p = []
        pos = dict(rate_hz=self.rate_hz, stand_height_mm=self.stand_height_mm,
                   rest_height_mm=self.rest_height_mm, cmd_vel_timeout_s=self.cmd_vel_timeout_s,
                   height_rate_mmps=self.height_rate_mmps, imu_timeout_s=self.imu_timeout_s,
                   heading_lateral_leak_s=self.heading_lateral_leak_s,
                   pose_rate_dps=self.pose_rate_dps)
        for k, v in pos.items():
            if not (math.isfinite(v) and v > 0):
                p.append(f'{k} must be > 0')
        nonneg = dict(level_kp=self.level_kp, level_ki=self.level_ki, level_kd=self.level_kd,
                      level_lpf_hz=self.level_lpf_hz, level_max_deg=self.level_max_deg,
                      level_walk_gain_scale=self.level_walk_gain_scale,
                      heading_kp=self.heading_kp, heading_ki=self.heading_ki,
                      heading_kd=self.heading_kd,
                      heading_max_rps=self.heading_max_rps,
                      heading_lateral_gain=self.heading_lateral_gain,
                      startup_blend_s=self.startup_blend_s)
        for k, v in nonneg.items():
            if not (math.isfinite(v) and v >= 0):
                p.append(f'{k} must be >= 0')
        if not 10.0 <= self.rate_hz <= 200.0:
            p.append('rate_hz must be in 10..200')
        if self.level_max_deg > 30.0:
            p.append('level_max_deg must be <= 30')
        if not (math.isfinite(self.level_deadband_deg) and 0.0 <= self.level_deadband_deg <= 5.0):
            p.append('level_deadband_deg must be in 0..5 deg')
        if self.level_walk_gain_scale > 1.0:
            p.append('level_walk_gain_scale must be <= 1')
        if self.rest_height_mm > self.stand_height_mm:
            p.append('rest_height_mm must be <= stand_height_mm')
        lim_problems = self.pose_limits().validate()
        p += lim_problems
        if not lim_problems:
            for k in ('stand_height_mm', 'rest_height_mm'):
                if not self.height_min_mm <= getattr(self, k) <= self.height_max_mm:
                    p.append(f'{k} must be inside height_min_mm..height_max_mm')
        for k in ('level_target_roll_deg', 'level_target_pitch_deg'):
            if not abs(getattr(self, k)) <= 15.0:
                p.append(f'{k} must be within +-15 deg')
        if len(self.leg_trim_mm) != 4 or not all(math.isfinite(t) and abs(t) <= 30.0
                                                 for t in self.leg_trim_mm):
            p.append('leg_trim_mm must be 4 values within +-30 mm')
        if len(self.leg_trim_x_mm) != 4 or not all(math.isfinite(t) and abs(t) <= 30.0
                                                   for t in self.leg_trim_x_mm):
            p.append('leg_trim_x_mm must be 4 values within +-30 mm')
        if len(self.servo_min) != 8 or len(self.servo_max) != 8 or not all(
                lo < hi for lo, hi in zip(self.servo_min, self.servo_max)):
            p.append('servo_min/servo_max must be 8 values with min < max')
        try:
            self.joint_region()
        except ValueError as exc:
            p.append(str(exc))
        try:
            self.presets()
            p += self.gait_params().validate()
        except ValueError as exc:
            p.append(str(exc))
        return p

    def joint_region(self) -> JointRegion:
        return JointRegion(self.ik_region_m1, self.ik_region_m2_lo, self.ik_region_m2_hi)


#: Parameters that need a restart (wiring and geometry).
READ_ONLY_PARAMS = frozenset({
    'rate_hz', 'imu_topic', 'leg_names', 'leg_thigh_idx', 'leg_calf_idx', 'leg_mirror_servo',
    'leg_x_sign', 'leg_pos_x_mm', 'leg_pos_y_mm', 'ik_region_m1', 'ik_region_m2_lo',
    'ik_region_m2_hi', 'servo_min', 'servo_max'}
    | {f'geom.{k}' for k in LegGeometry().as_dict()})
#: Gait parameters, accepted only while the gait is idle. ``gait_presets`` covers every
#: ``gait_presets.<name>.<key>``.
GAIT_PARAMS = frozenset({'gait_presets', 'gait_schedule', 'gait_schedule_speeds_mps',
                         'gait_switch_hysteresis_mps', 'gait_switch_periods', 'max_stride_mm',
                         'max_speed_mps', 'max_turn_rps', 'turn_stride_gain', 'ramp_time_s'})
#: Pose limits, accepted only while the gait is idle and the current pose fits the new set.
POSE_LIMIT_PARAMS = frozenset({'height_min_mm', 'height_max_mm', 'walk_height_min_mm',
                               'pose_pitch_max_deg', 'pose_roll_max_deg', 'pose_x_max_mm'})
#: Parameters the envelope check depends on; its result is cached on them.
_ENVELOPE_FIELDS = ('leg_names', 'leg_thigh_idx', 'leg_calf_idx', 'leg_mirror_servo',
                    'leg_x_sign', 'leg_pos_x_mm', 'leg_pos_y_mm', 'geom', 'ik_region_m1',
                    'ik_region_m2_lo', 'ik_region_m2_hi', 'stand_height_mm', 'stand_fx_mm',
                    'rest_height_mm', 'height_min_mm', 'height_max_mm', 'walk_height_min_mm',
                    'pose_pitch_max_deg',
                    'pose_roll_max_deg', 'pose_x_max_mm', 'gait_presets', 'gait_schedule',
                    'gait_schedule_speeds_mps', 'max_stride_mm',
                    'leg_trim_mm', 'leg_trim_x_mm', 'servo_min', 'servo_max')


def _coerce(value, like, name):
    if isinstance(like, bool):
        if not isinstance(value, bool):
            raise ValueError(f'{name}: expected true/false')
        return value
    if isinstance(like, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name}: expected a number')
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f'{name}: must be finite')
        return v
    if isinstance(like, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                (isinstance(value, float) and not value.is_integer()):
            raise ValueError(f'{name}: expected an integer')
        return int(value)
    if isinstance(like, str):
        if not isinstance(value, str):
            raise ValueError(f'{name}: expected a string')
        return value
    if isinstance(like, list):
        if isinstance(value, (str, bytes)) or not hasattr(value, '__len__'):
            raise ValueError(f'{name}: expected a list')
        proto = like[0] if like else 0.0
        return [_coerce(v, proto, f'{name}[{i}]') for i, v in enumerate(list(value))]
    raise ValueError(f'{name}: unsupported type')


@dataclass
class ImuSample:
    roll: float          # rad (REP-103, + = right side down)
    pitch: float         # rad (+ = nose down)
    yaw: float           # rad (+ = CCW / left)
    wx: float            # rad/s body rates (gyro)
    wy: float
    wz: float
    ay: float            # m/s^2 body lateral linear acceleration (gravity removed)
    t: float             # monotonic s when received


class StepOutput(NamedTuple):
    servo: List[float]   # 8 servo angles (deg), logical index order
    debug: List[float]   # DEBUG_FIELDS


class _Components(NamedTuple):
    params: ControllerParams
    leg_map: LegMap
    kin: LegKinematics
    limits: PoseLimits
    interleg: Tuple[SidePairs, ...]   # per side: rear and front leg and their unsafe pairs
    report: EnvelopeReport
    gait_checks: Dict[str, str]       # gait shape (preset or a~b midpoint) -> its check
    walkable: Tuple[str, ...]         # gait shapes whose envelope fits
    walk_allowed: bool                # the pose check and every scheduled shape fit
    level_allowed: bool               # pose check passed: pose commands and leveling allowed
    stand_ok: bool


_ENVELOPE_CACHE: Dict[Any, Tuple[EnvelopeReport, Dict[str, str]]] = {}


def _hashable(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def _envelope(params: ControllerParams, leg_map: LegMap, kin: LegKinematics,
              limits: PoseLimits, interleg: Tuple[SidePairs, ...]
              ) -> Tuple[EnvelopeReport, Dict[str, str]]:
    """Run validate_envelope on everything the controller can plan, cached on the
    parameters that move a foot and on the pair-sweep data."""
    key = tuple(_hashable(getattr(params, k)) for k in _ENVELOPE_FIELDS) + tuple(
        (s.side, s.rear, s.front, s.pairs.key if s.pairs is not None else s.problem)
        for s in interleg)
    hit = _ENVELOPE_CACHE.get(key)
    if hit is not None:
        return hit
    shapes = OrderedDict(schedule_shapes(params.knots()))
    for name, preset in sorted(params.presets().items()):
        shapes.setdefault(name, preset)
    checks, names = envelope_checks(shapes, params.max_stride_mm, limits,
                                    params.stand_fx_mm, params.stand_height_mm,
                                    params.rest_height_mm)
    report = validate_envelope(kin, leg_map, params.servo_min, params.servo_max, checks,
                               params.leg_trim_mm, params.leg_trim_x_mm, interleg=interleg)
    if len(_ENVELOPE_CACHE) > 16:
        _ENVELOPE_CACHE.clear()
    _ENVELOPE_CACHE[key] = (report, names)
    return report, names


def build_components(params: ControllerParams) -> _Components:
    """Validate params, build the mapping/kinematics and run the envelope check."""
    problems = params.validate()
    if problems:
        raise ValueError('; '.join(problems))
    leg_map = LegMap.from_params(params.leg_names, params.leg_thigh_idx, params.leg_calf_idx,
                                 params.leg_mirror_servo, params.leg_x_sign,
                                 params.leg_pos_x_mm, params.leg_pos_y_mm)
    kin = LegKinematics(LegGeometry.from_mapping(params.geom), region=params.joint_region())
    limits = params.pose_limits()
    interleg = load_sides(leg_map)
    report, names = _envelope(params, leg_map, kin, limits, interleg)
    # a check fails on its own points or on an unsafe front/back pair
    blocked = set(report.failed) | set(report.interleg_failed)
    stand_ok = not ({STAND_CHECK, REST_CHECK} & blocked)
    level_allowed = stand_ok and POSE_CHECK not in blocked
    walkable = tuple(n for n in names if names[n] not in blocked)
    walk_allowed = level_allowed and all(n in walkable
                                         for n in schedule_shapes(params.knots()))
    return _Components(params, leg_map, kin, limits, interleg, report, names, walkable,
                       walk_allowed, level_allowed, stand_ok)


class _Log:
    """Default logger: collects (level, message) lines."""

    def __init__(self):
        self.lines: List[Tuple[str, str]] = []

    def info(self, msg):
        self.lines.append(('info', msg))

    def warn(self, msg):
        self.lines.append(('warn', msg))

    def error(self, msg):
        self.lines.append(('error', msg))


def _slew(cur: float, target: float, step: float) -> float:
    d = target - cur
    return target if abs(d) <= step else cur + math.copysign(step, d)


class GaitControllerCore:
    """The controller state machine. All times are monotonic seconds."""

    def __init__(self, params: ControllerParams, log=None):
        self.log = log or _Log()
        self._throttle: Dict[str, float] = {}
        self._install(build_components(params))
        p = self.p
        self.state = State.STAND
        self.gait = GaitGenerator(p.gait_params())
        self.pid_roll = self._make_level_pid()
        self.pid_pitch = self._make_level_pid()
        self.pid_yaw = PID(p.heading_kp, p.heading_ki, p.heading_kd, p.heading_max_rps)
        self._pose_cmd = self._neutral_pose()          # commanded (clamped) pose
        self._pose_ramp: Optional[BodyPose] = None     # target of a ~/rest or ~/stand ramp
        self._pose = self._pose_cmd                    # applied: command + leveling, clamped
        self._center = p.stand_fx_mm
        self._u_roll = self._u_pitch = self._u_yaw = 0.0
        self._level_lim = p.level_max_deg
        self._level_scale = 1.0      # leveling gain scale, slewed like the bound
        self._level_active = False
        self._heading_active = False
        self._yaw_ref = 0.0          # heading the hold drives to (IMU yaw, rad)
        self._yaw_zero = 0.0         # IMU yaw that counts as heading 0 (~/zero_yaw)
        self._yaw_target: Optional[float] = None   # /cmd_heading, relative to ~/zero_yaw
        self._lat_v = 0.0
        self._cmd = (0.0, 0.0)
        self._cmd_t: Optional[float] = None
        self._cmd_inhibit_t = -math.inf
        self._step_t: Optional[float] = None    # last /cmd_step true (expires like /cmd_vel)
        self._step_active = False
        self._n_stance = 4
        self._imu: Optional[ImuSample] = None
        self._measured: Optional[List[float]] = None
        self._pending_blend = True
        self._blend: Optional[Dict[str, Any]] = None
        self._frozen: Optional[List[float]] = None
        self._last_servo: Optional[List[float]] = None
        self._last_t: Optional[float] = None
        self._driver_enabled = True
        self._ik_guess: List[Optional[Tuple[float, float]]] = [None] * 4
        # front/back leg guard: last output of each leg while its side's pair was safe
        self._safe_motor: List[Optional[Tuple[float, float]]] = [None] * 4
        self._interleg_holds = 0

    def _install(self, comp: _Components) -> None:
        self.p = comp.params
        self.leg_map = comp.leg_map
        self.kin = comp.kin
        self.interleg = comp.interleg
        self.report = comp.report
        self.gait_checks = comp.gait_checks
        self.walkable = comp.walkable
        self.walk_allowed = comp.walk_allowed
        self.level_allowed = comp.level_allowed
        self.stand_ok = comp.stand_ok
        p = comp.params
        # Pose commands and leveling move the body inside the validated pose set; if that
        # check failed, only the neutral stand and rest heights are used.
        self.limits = comp.limits if comp.level_allowed else replace(
            comp.limits, height_min_mm=min(p.rest_height_mm, p.stand_height_mm),
            height_max_mm=max(p.rest_height_mm, p.stand_height_mm), pitch_max=0.0,
            roll_max=0.0, x_max_mm=0.0)
        # while walking the height floor rises to walk_height_min_mm
        self.limits_walk = self.limits.walking()
        self._floor = max(getattr(self, '_floor', -math.inf), self.limits.height_min_mm)
        self.hips = tuple((leg.hip_x_mm, leg.hip_y_mm) for leg in comp.leg_map.legs)
        if comp.report.ok:
            self.log.info(comp.report.summary())
        else:
            self.log.warn(comp.report.summary())
            if not comp.stand_ok:
                self.log.error('stand/rest pose outside the envelope (workspace or front/back '
                               'legs): fix stand_height_mm / stand_fx_mm / rest_height_mm / '
                               'leg_trim_mm, or the CAD pair sweep (pair_sweep_<side>_final.json)')
            if not comp.level_allowed:
                self.log.warn('pose commands and leveling DISABLED (pose limits do not fit)')
            others = [n for n in comp.gait_checks if n not in comp.walkable]
            if others:
                self.log.warn(f'gait shapes outside the envelope: {others}')
        if not comp.walk_allowed:
            self.log.warn(f'walking DISABLED: the gait schedule {p.gait_schedule} does not fit '
                          '(STAND only)')

    def _make_level_pid(self) -> PID:
        p = self.p
        return PID(p.level_kp, p.level_ki, p.level_kd, p.level_max_deg, lpf_hz=p.level_lpf_hz,
                   deadband=p.level_deadband_deg, deadband_release=LEVEL_DEADBAND_RELEASE)

    def _neutral_pose(self) -> BodyPose:
        return self.limits.clamp(BodyPose(self.p.stand_height_mm))

    def _limits_now(self) -> PoseLimits:
        """Pose limits in force now: the full set with its height floor at ``_floor``, which
        moves at height_rate_mmps between the standing and walking minimums."""
        if self._floor <= self.limits.height_min_mm:
            return self.limits
        return replace(self.limits, height_min_mm=min(self._floor, self.limits.height_max_mm))

    def update_params(self, new: ControllerParams) -> Tuple[bool, str]:
        """Live parameter update (validated like startup). Returns (ok, reason)."""
        old = self.p
        changed = {f.name for f in fields(ControllerParams)
                   if getattr(old, f.name) != getattr(new, f.name)}
        if not changed:
            return True, ''
        ro = changed & {n.split('.')[0] for n in READ_ONLY_PARAMS}
        if ro:
            return False, f'read-only (restart required): {sorted(ro)}'
        if changed & (GAIT_PARAMS | POSE_LIMIT_PARAMS) and not self.gait.is_idle():
            return False, 'stop walking before changing gait parameters or pose limits'
        try:
            comp = build_components(new)
        except ValueError as exc:
            return False, str(exc)
        # Reject an update that would disable walking, leveling or standing.
        if (self.walk_allowed and not comp.walk_allowed) or \
                (self.level_allowed and not comp.level_allowed) or \
                (self.stand_ok and not comp.stand_ok):
            return False, 'rejected, ' + comp.report.summary()
        if changed & POSE_LIMIT_PARAMS and comp.level_allowed and \
                not comp.limits.contains(self._pose, tol=1e-3):
            return False, ('the current body pose is outside the new pose limits: '
                           'command a pose inside them first (/cmd_pose)')
        self._install(comp)
        p = self.p
        for pid in (self.pid_roll, self.pid_pitch):
            pid.kp, pid.ki, pid.kd = p.level_kp, p.level_ki, p.level_kd
            pid.out_limit = pid.i_limit = p.level_max_deg
            pid.lpf.cutoff_hz = p.level_lpf_hz
            pid.deadband = p.level_deadband_deg
        yaw = self.pid_yaw
        yaw.kp, yaw.ki, yaw.kd = p.heading_kp, p.heading_ki, p.heading_kd
        yaw.out_limit = yaw.i_limit = p.heading_max_rps
        yaw.integrator = max(-p.heading_max_rps, min(p.heading_max_rps, yaw.integrator))
        if changed & GAIT_PARAMS:
            self.gait = GaitGenerator(p.gait_params())
        elif 'heading_max_rps' in changed:
            self.gait.set_turn_headroom(p.heading_max_rps)
        self._pose_cmd = self.limits.clamp(self._pose_cmd)
        if self._pose_ramp is not None:
            self._pose_ramp = self.limits.clamp(self._pose_ramp)
        return True, ''

    def set_cmd_vel(self, v: float, w: float, now: float) -> None:
        if math.isfinite(v) and math.isfinite(w):
            self._cmd = (v, w)
            self._cmd_t = now

    def set_cmd_step(self, step: bool, now: float) -> None:
        """True keeps the gait stepping, in place at zero velocity. Like /cmd_vel it expires
        after cmd_vel_timeout_s unless repeated; False stops at once."""
        self._step_t = now if step else None

    def set_cmd_pose(self, height_mm: float, pitch: float, roll: float, x_mm: float = 0.0,
                     y_mm: float = 0.0, now: float = 0.0) -> None:
        """Commanded body pose (support frame; rad, mm), not rate limited. ``height_mm <= 0``
        means stand_height_mm; ``y_mm`` is ignored. Clamped into the pose limits; ignored in
        REST and ESTOP."""
        vals = (height_mm, pitch, roll, x_mm)
        if not all(math.isfinite(v) for v in vals) or self.state in (State.REST, State.ESTOP):
            return
        if not self.level_allowed:
            self._warn_throttled('pose', '/cmd_pose ignored: the pose limits do not fit the '
                                 'envelope (' + self.report.summary() + ')', now)
            return
        if y_mm and math.isfinite(y_mm):
            self._warn_throttled('pose_y', '/cmd_pose linear.y ignored: the legs are planar '
                                 '(no abduction), the body only moves sideways with roll', now)
        h = height_mm if height_mm > 0.0 else self.p.stand_height_mm
        want = BodyPose(h, pitch, roll, x_mm)
        lim = self.limits
        self._pose_cmd = lim.clamp(want)
        self._pose_ramp = None
        if not lim.contains(want, tol=1e-6):
            t = self._pose_cmd
            self._warn_throttled('pose_clamp', f'/cmd_pose clamped into the pose limits: height '
                                 f'{t.height_mm:.0f} mm, pitch {t.pitch_deg:+.1f}, roll '
                                 f'{t.roll_deg:+.1f} deg, x {t.x_mm:+.0f} mm', now)

    def set_imu(self, sample: ImuSample) -> None:
        vals = (sample.roll, sample.pitch, sample.yaw, sample.wx, sample.wy, sample.wz)
        if all(math.isfinite(v) for v in vals):
            self._imu = sample

    def set_measured_pose(self, servo: List[float]) -> None:
        if len(servo) == 8 and all(math.isfinite(s) for s in servo):
            self._measured = list(servo)

    def set_heading_target(self, heading_rad: Optional[float]) -> None:
        """Heading (rad, relative to ~/zero_yaw) to hold while walking straight. None or NaN
        holds the heading each segment starts with."""
        if heading_rad is None or not math.isfinite(heading_rad):
            self._yaw_target = None
            return
        self._yaw_target = _wrap_pi(heading_rad)
        if self._heading_active:
            self._yaw_ref = self._yaw_zero + self._yaw_target

    def set_driver_enabled(self, enabled: bool) -> None:
        if enabled and not self._driver_enabled:
            self._pending_blend = True
            self._measured = None    # wait for the re-armed pose
            self.log.info('servo driver re-armed: blending from its stand pose')
        elif not enabled and self._driver_enabled:
            self.log.warn('servo driver relaxed: controller paused')
            self.gait.reset()
            self._step_t = None
            if self.state == State.WALK:
                self.state = State.STAND
        self._driver_enabled = enabled

    def request_stand(self, now: float) -> Tuple[bool, str]:
        self._step_t = None
        self._pose_ramp = self._neutral_pose()
        if self.state == State.ESTOP:
            self.state = State.STAND
            self.gait.reset()
            self._start_blend(self._frozen or self._last_servo or self._measured)
            self._frozen = None
            return True, 'leaving ESTOP: blending to stand'
        self._cmd_inhibit_t = now   # ignore cmd_vel sent before this request
        if self.state == State.REST:
            self.state = State.STAND
            return True, 'standing up'
        return True, 'stopping / standing' if self.state == State.WALK else 'standing'

    def request_rest(self, now: float) -> Tuple[bool, str]:
        if self.state == State.ESTOP:
            return False, 'in ESTOP: call ~/stand first'
        self._step_t = None
        self.state = State.REST
        self._pose_ramp = self.limits.clamp(BodyPose(self.p.rest_height_mm))
        return True, f'crouching to rest_height_mm={self.p.rest_height_mm}'

    def request_estop(self) -> Tuple[bool, str]:
        if self._last_servo is None:
            return False, 'no output yet'
        self.state = State.ESTOP
        self._step_t = None
        self._frozen = list(self._last_servo)
        self._blend = None
        return True, 'ESTOP: output frozen; call ~/stand to resume'

    def set_level_enable(self, enable: bool) -> Tuple[bool, str]:
        self.p = replace(self.p, level_enable=bool(enable))
        if enable and not self.level_allowed:
            return False, 'leveling disabled by envelope validation: ' + self.report.summary()
        return True, f'leveling {"enabled" if enable else "disabled"}'

    def zero_yaw(self) -> Tuple[bool, str]:
        if self._imu is None:
            return False, 'no IMU data'
        self._yaw_zero = self._imu.yaw
        self._yaw_ref = self._yaw_zero + (self._yaw_target or 0.0)
        self.pid_yaw.reset()
        return True, f'heading zero set to IMU yaw {self._imu.yaw * _R2D:.1f} deg'

    def _warn_throttled(self, key: str, msg: str, now: float, period: float = 2.0) -> None:
        last = self._throttle.get(key)
        if last is None or now - last >= period:
            self._throttle[key] = now
            self.log.warn(msg)

    def _start_blend(self, pose: Optional[List[float]]) -> None:
        if pose is None:
            self._blend = None
            return
        self._blend = {'from': list(pose), 't': 0.0, 'T': max(self.p.startup_blend_s, 1e-3)}
        self._u_roll = self._u_pitch = 0.0
        self._level_active = False

    def _imu_fresh(self, now: float) -> Optional[ImuSample]:
        s = self._imu
        if s is None or now - s.t > self.p.imu_timeout_s:
            return None
        return s

    def _warn_cmd_in_rest_estop(self, now: float) -> None:
        """Warn when a live /cmd_vel or /cmd_step is dropped because of REST or ESTOP."""
        if self.state not in (State.REST, State.ESTOP):
            return
        p = self.p
        live = self._cmd_t is not None and now - self._cmd_t <= p.cmd_vel_timeout_s
        step = self._step_t is not None and now - self._step_t <= p.cmd_vel_timeout_s
        if (live and (abs(self._cmd[0]) > V_EPS or abs(self._cmd[1]) > W_EPS)) or step:
            self._warn_throttled('cmd_state', f'cmd_vel / cmd_step ignored: the controller is in '
                                 f'{self.state.name}, call ~/stand first', now, 5.0)

    def _heading(self, w_cmd: float, imu: Optional[ImuSample], dt: float) -> float:
        p = self.p
        # active while the gait runs, including its stop ramp, with no commanded turn
        straight = (p.heading_hold_enable and self.state == State.WALK and abs(w_cmd) < W_EPS
                    and imu is not None)
        if not straight:
            self._heading_active = False
            self._u_yaw = 0.0
            return 0.0
        if not self._heading_active:
            # new straight segment: hold the commanded or current heading with a fresh PID
            self._heading_active = True
            self._yaw_ref = (self._yaw_zero + self._yaw_target if self._yaw_target is not None
                             else imu.yaw)
            self.pid_yaw.reset()
            self._lat_v = 0.0
        pid = self.pid_yaw
        u = pid.update(_wrap_pi(self._yaw_ref - imu.yaw), imu.wz, dt)
        if p.heading_use_lateral_accel:
            # Leaky-integrated lateral acceleration.
            self._lat_v += (imu.ay - self._lat_v / p.heading_lateral_leak_s) * dt
            u -= p.heading_lateral_gain * self._lat_v
            u = max(-pid.out_limit, min(pid.out_limit, u))
        self._u_yaw = u
        return u

    def _level(self, imu: Optional[ImuSample], dt: float, now: float) -> Tuple[float, float]:
        """Leveling correction (deg) added to the commanded attitude: u = PID(target -
        measured), so a nose-down tilt gives u_pitch < 0 and raises the front hips."""
        p = self.p
        # Authority |u_r| + |u_p| and the gains scale down while walking; both slew.
        target = p.level_walk_gain_scale if self.state == State.WALK else 1.0
        self._level_lim = _slew(self._level_lim, p.level_max_deg * target,
                                LEVEL_LIMIT_SLEW_DPS * dt)
        self._level_scale = _slew(self._level_scale, target, LEVEL_SCALE_SLEW_PER_S * dt)
        want = (p.level_enable and self.level_allowed and self._blend is None
                and self.state in (State.STAND, State.WALK) and self._driver_enabled)
        if want and imu is not None and self._n_stance == 0:
            # Flight phase: hold the correction.
            pass
        elif want and imu is not None:
            scale = self._level_scale
            cmd = self._pose_cmd
            e_r = p.level_target_roll_deg + cmd.roll_deg - imu.roll * _R2D
            e_p = p.level_target_pitch_deg + cmd.pitch_deg - imu.pitch * _R2D
            r_r, r_p = imu.wx * _R2D, imu.wy * _R2D
            if not self._level_active:
                self.pid_roll.preset(self._u_roll, e_r, r_r, scale)
                self.pid_pitch.preset(self._u_pitch, e_p, r_p, scale)
                self._level_active = True
            u_r = self.pid_roll.update(e_r, r_r, dt, scale)
            u_p = self.pid_pitch.update(e_p, r_p, dt, scale)
            # Bound |u_r| + |u_p| by the current authority, back-calculating the integrators.
            total = abs(u_r) + abs(u_p)
            if total > self._level_lim:
                k = self._level_lim / total
                u_r = self._back_calc(self.pid_roll, u_r, u_r * k)
                u_p = self._back_calc(self.pid_pitch, u_p, u_p * k)
            self._u_roll, self._u_pitch = u_r, u_p
        elif want:
            # No fresh IMU data: fade to the commanded pose.
            self._warn_throttled('imu', 'IMU data stale: fading the leveling correction out', now)
            self._level_active = False
            step = p.pose_rate_dps * dt
            self._u_roll = _slew(self._u_roll, 0.0, step)
            self._u_pitch = _slew(self._u_pitch, 0.0, step)
        else:
            self._level_active = False
            step = p.pose_rate_dps * dt
            self._u_roll = _slew(self._u_roll, 0.0, step)
            self._u_pitch = _slew(self._u_pitch, 0.0, step)
        total = abs(self._u_roll) + abs(self._u_pitch)
        if total > self._level_lim:
            k = self._level_lim / total
            self._u_roll *= k
            self._u_pitch *= k
        return self._u_roll, self._u_pitch

    @staticmethod
    def _back_calc(pid: PID, u: float, applied: float) -> float:
        """Unwind the PID integrator by the difference between asked and applied output, keeping its sign."""
        excess = u - applied
        if abs(excess) <= 1e-12:
            return u
        i_new = pid.integrator - excess
        if i_new * pid.integrator < 0.0:
            i_new = 0.0
        pid.integrator = i_new
        pid.output = applied
        return applied

    def _ramp_pose(self, dt: float) -> None:
        """The controller's own pose transitions (~/rest, ~/stand) ramp at height_rate_mmps
        and pose_rate_dps; /cmd_pose setpoints are applied as they come."""
        target = self._pose_ramp
        if target is None:
            return
        p = self.p
        c, dh, da = self._pose_cmd, p.height_rate_mmps * dt, p.pose_rate_dps * _D2R * dt
        self._pose_cmd = BodyPose(_slew(c.height_mm, target.height_mm, dh),
                                  _slew(c.pitch, target.pitch, da),
                                  _slew(c.roll, target.roll, da),
                                  _slew(c.x_mm, target.x_mm, dh))
        if self._pose_cmd == target:
            self._pose_ramp = None

    def _apply_pose(self) -> BodyPose:
        """Commanded pose plus leveling inside the pose limits. The command keeps priority
        (PoseLimits.add_within); the integrators unwind what did not fit."""
        lim = self._limits_now()
        cmd = lim.clamp(self._pose_cmd)
        pose, dp, dr = lim.add_within(cmd, self._u_pitch * _D2R, self._u_roll * _D2R)
        up, ur = dp * _R2D, dr * _R2D
        if self._level_active:
            up = self._back_calc(self.pid_pitch, self._u_pitch, up)
            ur = self._back_calc(self.pid_roll, self._u_roll, ur)
        self._u_pitch, self._u_roll = up, ur
        return pose

    def step(self, now: float) -> Optional[StepOutput]:
        """One control tick. Returns None while waiting for /servo_state."""
        p = self.p
        if not self._driver_enabled:
            return None
        if self._pending_blend:
            if self._measured is None:
                return None
            self._start_blend(self._measured)
            self._pending_blend = False
        dt = 1.0 / p.rate_hz if self._last_t is None else now - self._last_t
        dt = min(max(dt, 0.0), 0.1)
        self._last_t = now

        self._warn_cmd_in_rest_estop(now)
        if self.state == State.ESTOP and self._frozen is not None:
            return StepOutput(list(self._frozen),
                              self._debug(self._frozen, self._imu_fresh(now), [],
                                          float('nan'), 0, 0.0))

        imu = self._imu_fresh(now)
        cmd_ok = (self._cmd_t is not None and now - self._cmd_t <= p.cmd_vel_timeout_s
                  and self._cmd_t > self._cmd_inhibit_t)
        v_cmd, w_cmd = self._cmd if cmd_ok else (0.0, 0.0)
        step_cmd = (self._step_t is not None and now - self._step_t <= p.cmd_vel_timeout_s
                    and self._step_t > self._cmd_inhibit_t
                    and self.state in (State.STAND, State.WALK))
        gait_req = abs(v_cmd) > V_EPS or abs(w_cmd) > W_EPS or step_cmd
        if gait_req and not (self.walk_allowed and self._blend is None
                             and self.state in (State.STAND, State.WALK)):
            if self.state in (State.STAND, State.WALK):
                why = (self.report.summary() if not self.walk_allowed
                       else 'still blending to the stand pose')
                self._warn_throttled('nowalk', f'cmd_vel / cmd_step ignored, walking not '
                                     f'allowed: {why}', now, 5.0)
            v_cmd = w_cmd = 0.0
            step_cmd = gait_req = False
        # On a gait request the height floor rises to the walking minimum, and the gait
        # starts once it is there.
        walk_floor = gait_req or not self.gait.is_idle()
        target = self.limits_walk.height_min_mm if walk_floor else self.limits.height_min_mm
        if self._pose_cmd.height_mm >= target and self._floor < target:
            self._floor = target            # body already above the walking floor: no wait
        else:
            self._floor = _slew(self._floor, target, p.height_rate_mmps * dt)
        if gait_req and self.gait.is_idle() and self._floor < target - 1e-9:
            v_cmd = w_cmd = 0.0
            step_cmd = gait_req = False
        v_cmd = max(-p.max_speed_mps, min(p.max_speed_mps, v_cmd))
        w_cmd = max(-p.max_turn_rps, min(p.max_turn_rps, w_cmd))
        if self.state == State.STAND and gait_req:
            self.state = State.WALK

        u_yaw = self._heading(w_cmd, imu, dt)
        self._ramp_pose(dt)
        self._center = _slew(self._center, p.stand_fx_mm, p.height_rate_mmps * dt)
        feet = self.gait.update(dt, v_cmd, w_cmd, step_cmd, u_yaw)
        self._step_active = step_cmd
        self._n_stance = stance_count(feet)
        if self.state == State.WALK and self.gait.is_idle() and not gait_req:
            self.state = State.STAND

        self._level(imu, dt, now)
        pose = self._pose = self._apply_pose()

        motor, legs_dbg = [], []
        ik_err = 0.0
        n_clamped = 0
        stance_h = []
        c = self._center
        for i in range(4):
            hx, hy = self.hips[i]
            fx, fz = foot_in_leg(pose, hx, hy, hx + c + feet[i].dx, feet[i].lift)
            cx, cz = foot_in_leg(pose, hx, hy, hx + c, 0.0)
            tx, tz = p.leg_trim_x_mm[i], p.leg_trim_mm[i]
            xc, yc = self.leg_map.body_to_canonical(i, fx + tx, fz + tz)
            center = self.leg_map.body_to_canonical(i, cx + tx, cz + tz)
            x2, y2, r, was = self.kin.clamp_to_workspace(xc, yc, center, self._ik_guess[i])
            if not math.isfinite(r.m1):
                # Center unreachable and IK failed: hold the leg.
                r_prev = self.kin.region.clip(*(self._ik_guess[i] or (90.0, 90.0)))
                r = r._replace(m1=r_prev[0], m2=r_prev[1])
            self._ik_guess[i] = (r.m1, r.m2)
            n_clamped += int(was)
            ik_err = max(ik_err, r.err_mm if math.isfinite(r.err_mm) else 0.0)
            motor.append((r.m1, r.m2))
            bfx, bfz = self.leg_map.canonical_to_body(i, x2, y2)
            legs_dbg.append((bfx, bfz, r.m1, r.m2, 1.0 if feet[i].in_stance else 0.0))
            if feet[i].in_stance:
                stance_h.append(-bfz)
        if n_clamped:
            self._warn_throttled('ws', f'foot target outside the workspace on {n_clamped} '
                                 'leg(s): shrunk toward the stand point', now)
        kin_height = sum(stance_h) / len(stance_h) if stance_h else float('nan')

        if self._blend is not None:
            b = self._blend
            b['t'] += dt
            a = min(1.0, b['t'] / b['T'])
            a = a * a * (3.0 - 2.0 * a)
            servo = [f + (t - f) * a for f, t in zip(b['from'], self.leg_map.servo_vector(motor))]
            # Clip the blend point back into the joint region.
            clip = self.kin.region.clip
            motor = [clip(m1, m2) for m1, m2 in self.leg_map.motor_angles(servo)]
            if b['t'] >= b['T']:
                self._blend = None
                self.log.info('stand pose reached')
        motor = self._guard_legs(motor, now)
        servo = self.leg_map.servo_vector(motor)
        self._last_servo = list(servo)
        debug = self._debug(servo, imu, legs_dbg, kin_height, n_clamped, ik_err)
        return StepOutput(servo, debug)

    def _guard_legs(self, motor: List[Tuple[float, float]], now: float
                    ) -> List[Tuple[float, float]]:
        """Front/back leg guard: on a side whose (rear, front) joint pair is in the CAD
        unsafe-pair set, hold both legs at their last safe output."""
        held = []
        for side in self.interleg:
            pairs = side.pairs
            if pairs is None:
                continue
            r, f = side.rear, side.front
            if pairs.marked(motor[r], motor[f]):
                sr, sf = self._safe_motor[r], self._safe_motor[f]
                if sr is not None and sf is not None:
                    motor[r], motor[f] = sr, sf
                held.append(side.side)
            else:
                self._safe_motor[r], self._safe_motor[f] = motor[r], motor[f]
        if held:
            self._interleg_holds += 1
            self._warn_throttled('interleg', f'front/back legs of the {" and ".join(held)} '
                                 'side would come within the CAD pair sweep\'s margin of a '
                                 'contact: holding them at their last safe pose', now)
        return motor

    def _debug(self, servo, imu, legs_dbg, kin_height, n_clamped, ik_err) -> List[float]:
        nan = float('nan')
        # roll/pitch: the attitude error the leveling sees (IMU minus level target minus
        # commanded attitude); yaw: relative to the heading reference. 0 means on target.
        cmd = self._pose_cmd
        roll = (imu.roll * _R2D - self.p.level_target_roll_deg - cmd.roll_deg) if imu else nan
        pitch = (imu.pitch * _R2D - self.p.level_target_pitch_deg - cmd.pitch_deg) if imu else nan
        yaw = _wrap_pi(imu.yaw - self._yaw_ref) * _R2D if imu else nan
        d = [float(self.state), self.gait.phase, self.gait.v, self.gait.w, roll, pitch, yaw,
             self._u_roll, self._u_pitch, self._u_yaw, self.pid_roll.integrator,
             self.pid_pitch.integrator, kin_height, float(self.walk_allowed),
             float(self._level_active), float(self._heading_active), ik_err, float(n_clamped)]
        if legs_dbg:
            for leg in legs_dbg:
                d.extend(leg)
        else:
            d.extend([nan] * 20)
        q = self._pose
        d += [self.pid_yaw.integrator, q.height_mm, q.pitch_deg, q.roll_deg, q.x_mm,
              float(self._n_stance), self.gait.amp, float(self._step_active),
              self.gait.speed, float(self.gait.knot), self.gait.preset.duty,
              self.gait.preset.period_s,
              _wrap_pi(imu.yaw - self._yaw_zero) * _R2D if imu else nan,
              float(self._interleg_holds), self.gait.body_speed]
        return d
