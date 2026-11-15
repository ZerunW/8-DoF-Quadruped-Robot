"""IK motor angles -> joint angles of the open-tree leg model (pure ``math``, no ROS).

The URDF and the MJCF cut each closed 5-bar leg into an open tree of six revolute joints::

    base -> thigh (O, motor m1) -> calf (K)        [-> foot, fixed at E]
    base -> fan   (O, passive)  -> ab_link (A)     loop closes at B on the calf
    base -> rod   (M, motor m2) -> cd_link (C)     loop closes at D on the fan

All joints are zero at the CAD pose m1 = m2 = 90 deg and positive counter-clockwise in the
canonical leg frame of ``robot_dog_control.kinematics``, which is about body +Y. The angles
come from ``LegKinematics.fk_points`` and close both loops. Motor joints:
q_thigh = m1 - 90 deg, q_rod = m2 - 90 deg.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

from robot_dog_control.kinematics import LegKinematics
from robot_dog_control.leg_map import LEG_NAMES

#: joint order within a leg, also used for /joint_states
JOINTS: Tuple[str, ...] = ('thigh', 'calf', 'fan', 'rod', 'cd_link', 'ab_link')
#: actuated joints: (m1, m2)
MOTOR_JOINTS: Tuple[str, str] = ('thigh', 'rod')
#: IK pose of the zero joint angles (deg)
ZERO_POSE_DEG: Tuple[float, float] = (90.0, 90.0)


def joint_name(leg: str, joint: str) -> str:
    return f'{leg}_{joint}_joint'


def all_joint_names(legs: Iterable[str] = LEG_NAMES) -> List[str]:
    return [joint_name(leg, j) for leg in legs for j in JOINTS]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _ang(p, q) -> float:
    return math.atan2(q[1] - p[1], q[0] - p[0])


class LegJoints:
    """(m1, m2) in deg -> the six joint angles (rad) in ``JOINTS`` order."""

    def __init__(self, kin: LegKinematics, zero_deg: Tuple[float, float] = ZERO_POSE_DEG):
        self.kin = kin
        self.zero_deg = (float(zero_deg[0]), float(zero_deg[1]))
        self._zero = self._relative(*self.zero_deg)

    def _relative(self, m1: float, m2: float) -> Tuple[float, ...]:
        p = self.kin.fk_points(m1, m2)
        th = _ang(p.O, p.K)
        calf = _ang(p.K, p.E)
        fan = _ang(p.O, p.D)
        rod = _ang(p.M, p.C)
        cd = _ang(p.C, p.D)
        ab = _ang(p.A, p.B)
        # each angle relative to its parent link (absolute for links on the base)
        return (th, calf - th, fan, rod, cd - rod, ab - fan)

    def angles(self, m1: float, m2: float) -> Tuple[float, ...]:
        rel = self._relative(m1, m2)
        return tuple(_wrap(r - z) for r, z in zip(rel, self._zero))

    def as_dict(self, leg: str, m1: float, m2: float) -> Dict[str, float]:
        return {joint_name(leg, j): q for j, q in zip(JOINTS, self.angles(m1, m2))}

    @staticmethod
    def motor_deg_from_joints(q_thigh: float, q_rod: float) -> Tuple[float, float]:
        """(q_thigh, q_rod) in rad -> (m1, m2) in deg."""
        return (ZERO_POSE_DEG[0] + math.degrees(q_thigh), ZERO_POSE_DEG[1] + math.degrees(q_rod))

    @staticmethod
    def motor_joints_from_deg(m1: float, m2: float) -> Tuple[float, float]:
        return (math.radians(m1 - ZERO_POSE_DEG[0]), math.radians(m2 - ZERO_POSE_DEG[1]))


def all_joint_angles(leg_joints: LegJoints, motor_deg: Sequence[Tuple[float, float]],
                     legs: Sequence[str] = LEG_NAMES) -> List[float]:
    """[(m1, m2) per leg] -> 24 joint angles in ``all_joint_names()`` order."""
    out: List[float] = []
    for leg, (m1, m2) in zip(legs, motor_deg):
        out.extend(leg_joints.angles(m1, m2))
    return out
