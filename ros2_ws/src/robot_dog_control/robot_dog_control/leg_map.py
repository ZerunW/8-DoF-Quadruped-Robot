"""Leg <-> servo mapping (no ROS).

Legs are ordered ``("lf", "rf", "lb", "rb")``. IK motor angle m maps to servo
``s = m + 90``, or ``s = 360 - (m + 90)`` on mirrored servos (right legs). Leg coordinates
``(fx, fz)`` (mm) are body-aligned, fx forward and fz up from the hip axis; the canonical
IK frame is ``x_c = x_sign * fx``, ``y_c = fz``. Hip positions are in the body frame
(REP-103).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

LEG_NAMES: Tuple[str, ...] = ('lf', 'rf', 'lb', 'rb')
DIAGONAL_PAIRS = (('lf', 'rb'), ('rf', 'lb'))
NUM_LEGS = 4
NUM_SERVOS = 8

#: +1 front legs, -1 back legs
FRONT_SIGN = (1, 1, -1, -1)
#: -1 left legs, +1 right legs
RIGHT_SIGN = (-1, 1, -1, 1)


@dataclass(frozen=True)
class Leg:
    name: str
    thigh_idx: int
    calf_idx: int
    mirror_servo: bool
    x_sign: int
    hip_x_mm: float
    hip_y_mm: float


class LegMap:
    """Validated mapping for the four legs (order lf, rf, lb, rb)."""

    def __init__(self, legs: Sequence[Leg]):
        legs = tuple(legs)
        problems = []
        if tuple(l.name for l in legs) != LEG_NAMES:
            problems.append(f'leg_names must be {list(LEG_NAMES)} in this order')
        idx = [l.thigh_idx for l in legs] + [l.calf_idx for l in legs]
        if sorted(idx) != list(range(NUM_SERVOS)):
            problems.append(f'leg_thigh_idx + leg_calf_idx must use every servo 0..7 '
                            f'exactly once (got {idx})')
        for l in legs:
            if l.x_sign not in (1, -1):
                problems.append(f'leg {l.name}: x_sign must be +1 or -1')
        if not problems and len(legs) == NUM_LEGS:
            for i, l in enumerate(legs):
                if FRONT_SIGN[i] * l.hip_x_mm <= 0:
                    problems.append(f'leg {l.name}: hip_x_mm sign does not match front/back')
                if RIGHT_SIGN[i] * l.hip_y_mm >= 0:
                    problems.append(f'leg {l.name}: hip_y_mm sign does not match left/right '
                                    '(left legs y>0)')
        if problems:
            raise ValueError('invalid leg map: ' + '; '.join(problems))
        self.legs = legs

    @classmethod
    def from_params(cls, names, thigh_idx, calf_idx, mirror_servo, x_sign,
                    pos_x_mm, pos_y_mm) -> 'LegMap':
        n = len(names)
        for label, seq in (('leg_thigh_idx', thigh_idx), ('leg_calf_idx', calf_idx),
                           ('leg_mirror_servo', mirror_servo), ('leg_x_sign', x_sign),
                           ('leg_pos_x_mm', pos_x_mm), ('leg_pos_y_mm', pos_y_mm)):
            if len(seq) != n:
                raise ValueError(f'{label} must have {n} entries')
        return cls(Leg(str(names[i]), int(thigh_idx[i]), int(calf_idx[i]),
                       bool(mirror_servo[i]), int(x_sign[i]), float(pos_x_mm[i]),
                       float(pos_y_mm[i])) for i in range(n))

    @classmethod
    def default(cls) -> 'LegMap':
        """Default wiring: right legs mirrored, knees backward, hip positions from the CAD."""
        return cls.from_params(list(LEG_NAMES), [1, 5, 3, 7], [0, 4, 2, 6],
                               [False, True, False, True], [-1, -1, -1, -1],
                               [91.5, 91.5, -91.5, -91.5], [88.2, -88.2, 88.2, -88.2])

    def to_servo(self, i: int, m1: float, m2: float) -> Tuple[float, float]:
        """IK motor angles (deg) of leg i -> (thigh servo, calf servo) in servo deg."""
        s1, s2 = m1 + 90.0, m2 + 90.0
        if self.legs[i].mirror_servo:
            return 360.0 - s1, 360.0 - s2
        return s1, s2

    def from_servo(self, i: int, s_thigh: float, s_calf: float) -> Tuple[float, float]:
        """Inverse of to_servo()."""
        if self.legs[i].mirror_servo:
            s_thigh, s_calf = 360.0 - s_thigh, 360.0 - s_calf
        return s_thigh - 90.0, s_calf - 90.0

    def servo_vector(self, motor_angles: Sequence[Tuple[float, float]]) -> List[float]:
        """[(m1, m2) per leg] -> 8 servo angles in logical index order."""
        out = [0.0] * NUM_SERVOS
        for i, (m1, m2) in enumerate(motor_angles):
            st, sc = self.to_servo(i, m1, m2)
            out[self.legs[i].thigh_idx] = st
            out[self.legs[i].calf_idx] = sc
        return out

    def motor_angles(self, servo: Sequence[float]) -> List[Tuple[float, float]]:
        """8 servo angles -> [(m1, m2) per leg]."""
        return [self.from_servo(i, servo[l.thigh_idx], servo[l.calf_idx])
                for i, l in enumerate(self.legs)]

    def body_to_canonical(self, i: int, fx: float, fz: float) -> Tuple[float, float]:
        return self.legs[i].x_sign * fx, fz

    def canonical_to_body(self, i: int, x_c: float, y_c: float) -> Tuple[float, float]:
        return self.legs[i].x_sign * x_c, y_c

    @property
    def half_track_mm(self) -> float:
        """Mean |hip_y| (half the lateral distance between left and right feet)."""
        return sum(abs(l.hip_y_mm) for l in self.legs) / len(self.legs)
