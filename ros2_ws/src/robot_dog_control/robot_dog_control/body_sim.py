"""Kinematic body plant (no ROS, no dynamics).

Commanded servo angles pass a first-order lag and a speed limit; FK gives the four feet.
When the feet are coplanar within ``coplanar_tol_mm`` the body attitude is the tilt of their
plane, else it is held. Signs follow REP-103: +pitch nose down, +roll right side down.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from robot_dog_control.kinematics import LegKinematics
from robot_dog_control.leg_map import LegMap


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _q_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def _rpy_q(q):
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def fit_plane(points: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    """Least-squares z = a + b x + c y; returns (a, b, c, max |residual|)."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sz = sum(p[2] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    syy = sum(p[1] * p[1] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    sxz = sum(p[0] * p[2] for p in points)
    syz = sum(p[1] * p[2] for p in points)
    m = [[n, sx, sy], [sx, sxx, sxy], [sy, sxy, syy]]
    r = [sz, sxz, syz]

    def det3(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))

    d = det3(m)
    sol = []
    for col in range(3):
        mc = [row[:] for row in m]
        for k in range(3):
            mc[k][col] = r[k]
        sol.append(det3(mc) / d)
    a, b, c = sol
    res = max(abs(p[2] - (a + b * p[0] + c * p[1])) for p in points)
    return a, b, c, res


class BodySim:
    """Attitude of a rigid body standing on its four feet."""

    def __init__(self, kin: LegKinematics, leg_map: LegMap, ground_roll_deg: float = 0.0,
                 ground_pitch_deg: float = 0.0, coplanar_tol_mm: float = 4.0,
                 servo_tau_s: float = 0.08, servo_speed_dps: float = 500.0):
        self.servo_tau_s = servo_tau_s
        self.servo_speed_dps = servo_speed_dps
        self.servo_actual: Optional[List[float]] = None
        self.kin = kin
        self.leg_map = leg_map
        self.ground_roll = math.radians(ground_roll_deg)
        self.ground_pitch = math.radians(ground_pitch_deg)
        self.tol = coplanar_tol_mm
        self.rel_roll = 0.0
        self.rel_pitch = 0.0
        self.yaw = 0.0
        self.roll = self.pitch = 0.0
        self.rates = (0.0, 0.0, 0.0)
        self._prev: Optional[Tuple[float, float, float]] = None
        self.feet: List[Tuple[float, float, float]] = []

    def feet_body(self, servo: Sequence[float]) -> List[Tuple[float, float, float]]:
        pts = []
        for i, (m1, m2) in enumerate(self.leg_map.motor_angles(servo)):
            x_c, y_c, _ = self.kin.fk(m1, m2)
            fx, fz = self.leg_map.canonical_to_body(i, x_c, y_c)
            leg = self.leg_map.legs[i]
            pts.append((leg.hip_x_mm + fx, leg.hip_y_mm, fz))
        return pts

    def update(self, servo: Sequence[float], dt: float, yaw_rate: float = 0.0
               ) -> Tuple[float, float, float]:
        """Advance by dt with the commanded servo angles; returns (roll, pitch, yaw) rad."""
        if self.servo_actual is None:
            self.servo_actual = list(servo)
        elif dt > 0:
            a = dt / (self.servo_tau_s + dt) if self.servo_tau_s > 0 else 1.0
            vmax = self.servo_speed_dps * dt
            self.servo_actual = [s + max(-vmax, min(vmax, a * (c - s)))
                                 for s, c in zip(self.servo_actual, servo)]
        self.feet = self.feet_body(self.servo_actual)
        a, b, c, res = fit_plane(self.feet)
        if res <= self.tol:
            n = math.sqrt(1.0 + b * b + c * c)
            self.rel_pitch = math.asin(b / n)
            self.rel_roll = -math.atan(c)
        self.yaw += yaw_rate * dt
        q = _qmul(_q_rpy(0.0, 0.0, self.yaw),
                  _qmul(_q_rpy(self.ground_roll, self.ground_pitch, 0.0),
                        _q_rpy(self.rel_roll, self.rel_pitch, 0.0)))
        roll, pitch, yaw = _rpy_q(q)
        if self._prev is not None and dt > 0:
            pr, pp, py = self._prev
            dy = (yaw - py + math.pi) % (2 * math.pi) - math.pi
            self.rates = ((roll - pr) / dt, (pitch - pp) / dt, dy / dt)
        self._prev = (roll, pitch, yaw)
        self.roll, self.pitch = roll, pitch
        return roll, pitch, yaw

    def quaternion(self) -> Tuple[float, float, float, float]:
        return _q_rpy(self.roll, self.pitch, self._prev[2] if self._prev else 0.0)
