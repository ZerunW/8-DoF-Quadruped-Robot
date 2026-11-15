"""Quaternion and roll/pitch/yaw conversions shared by all nodes and tools.

REP-103 body frame (x forward, y left, z up), intrinsic Z-Y-X (yaw, pitch, roll):

* roll  > 0: rotation about +x, right side down
* pitch > 0: rotation about +y, nose down
* yaw   > 0: rotation about +z, counterclockwise seen from above (turn left)

Angles in radians; quaternions are (x, y, z, w) like geometry_msgs.
"""

from __future__ import annotations

import math
from typing import Tuple


def quat_to_rpy(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """Quaternion -> (roll, pitch, yaw) in rad; the input is normalized first."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError('invalid quaternion')
    x, y, z, w = x / n, y / n, z / n, w / n
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, s)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """(roll, pitch, yaw) in rad -> quaternion (x, y, z, w), R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def wrap_pi(angle: float) -> float:
    """Wrap an angle (rad) to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
