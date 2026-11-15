"""Body pose layer (no ROS): the rigid pose of the body over the support frame.

Frames (REP-103, mm, rad): the support frame S lies on the ground under the feet, x along
the heading; the body frame B has its origin at the center of the four hip axes. A
``BodyPose`` is B in S: rotation ``R = Ry(pitch) Rx(roll)`` (+pitch nose down, +roll right
side down, as the IMU reports them) about the support point under the body, with the body
origin at ``p_body = (x_mm, 0, 0) + R (0, 0, height_mm)``. ``foot_in_leg`` matches each
foot's fore/aft position and height; the lateral position follows from the leg plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

_D2R = math.pi / 180.0
_R2D = 180.0 / math.pi
#: envelope samples: heights at most this far apart within a pose's height interval (mm)
SAMPLE_HEIGHT_STEP_MM = 30.0


@dataclass(frozen=True)
class BodyPose:
    """Body frame B in the support frame S."""

    height_mm: float
    pitch: float = 0.0       # rad, + nose down
    roll: float = 0.0        # rad, + right side down
    x_mm: float = 0.0        # mm, body ahead of the feet's neutral center

    @classmethod
    def from_deg(cls, height_mm: float, pitch_deg: float = 0.0, roll_deg: float = 0.0,
                 x_mm: float = 0.0) -> 'BodyPose':
        return cls(float(height_mm), pitch_deg * _D2R, roll_deg * _D2R, float(x_mm))

    @property
    def pitch_deg(self) -> float:
        return self.pitch * _R2D

    @property
    def roll_deg(self) -> float:
        return self.roll * _R2D

    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.height_mm, self.pitch, self.roll,
                                              self.x_mm))


def foot_in_leg(pose: BodyPose, hip_x: float, hip_y: float, ground_x: float,
                ground_z: float) -> Tuple[float, float]:
    """Leg-frame (fx, fz) of a foot planned at support-frame fore/aft position
    ``ground_x`` (mm, same origin as ``x_mm``) and height ``ground_z`` above the ground,
    for the hip axis at (hip_x, hip_y) in the body frame. Solves
    ``p_S = p_body + R (hip + (fx, 0, fz))`` for its x and z components."""
    ct, st = math.cos(pose.pitch), math.sin(pose.pitch)
    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    bx = pose.x_mm + pose.height_mm * st * cr       # p_body = (x, 0, 0) + R (0, 0, height)
    bz = pose.height_mm * ct * cr
    p = ground_x - bx - st * sr * hip_y
    q = ground_z - bz - ct * sr * hip_y
    a = ct * p - st * q
    u = st * p + ct * q
    return a - hip_x, u / cr


def _inward_part(cmd: float, d: float) -> float:
    """Part of the correction d that moves cmd toward 0, at most to -cmd."""
    if cmd > 0.0 and d < 0.0:
        return max(d, -2.0 * cmd)
    if cmd < 0.0 and d > 0.0:
        return min(d, -2.0 * cmd)
    return 0.0


def hip_offsets(pitch: float, roll: float, hips: Sequence[Tuple[float, float]]) -> List[float]:
    """Height of each hip axis above the body origin, along the support normal (mm); the
    body origin itself is at ``height cos(pitch) cos(roll)``."""
    sp, cp, sr = math.sin(pitch), math.cos(pitch), math.sin(roll)
    return [-hx * sp + hy * cp * sr for hx, hy in hips]


@dataclass(frozen=True)
class PoseLimits:
    """The set every commanded and applied pose is kept in: every hip between height_min_mm
    and height_max_mm above the ground, |pitch|/pitch_max + |roll|/roll_max <= 1 and
    |x_mm| <= x_max_mm."""

    height_min_mm: float
    height_max_mm: float
    pitch_max: float                       # rad
    roll_max: float                        # rad
    x_max_mm: float
    hips: Tuple[Tuple[float, float], ...]  # (hip_x, hip_y) per leg, body frame (mm)
    #: lowest hip height while walking (see ``walking``); at or below height_min_mm it has
    #: no effect
    walk_height_min_mm: float = 0.0

    def walking(self) -> 'PoseLimits':
        """The set while walking: every hip at least ``walk_height_min_mm`` up, capped at
        height_max_mm."""
        if self.walk_height_min_mm <= self.height_min_mm:
            return self
        return replace(self, height_min_mm=min(self.walk_height_min_mm, self.height_max_mm))

    def validate(self) -> List[str]:
        p = []
        vals = (self.height_min_mm, self.height_max_mm, self.pitch_max, self.roll_max,
                self.x_max_mm, self.walk_height_min_mm)
        if not all(math.isfinite(v) for v in vals):
            return ['pose limits must be finite']
        if not 0.0 < self.height_min_mm <= self.height_max_mm:
            p.append('height_min_mm must be > 0 and <= height_max_mm')
        if self.walk_height_min_mm < 0.0:
            p.append('walk_height_min_mm must be >= 0')
        if not 0.0 <= self.pitch_max <= 30.0 * _D2R:
            p.append('pose_pitch_max_deg must be in 0..30')
        if not 0.0 <= self.roll_max <= 30.0 * _D2R:
            p.append('pose_roll_max_deg must be in 0..30')
        if not 0.0 <= self.x_max_mm <= 60.0:
            p.append('pose_x_max_mm must be in 0..60')
        return p

    def height_interval(self, pitch: float, roll: float) -> Tuple[float, float]:
        """Body heights at which every hip is inside the height band (lo > hi: none)."""
        d = hip_offsets(pitch, roll, self.hips)
        k = math.cos(pitch) * math.cos(roll)
        return (self.height_min_mm - min(d)) / k, (self.height_max_mm - max(d)) / k

    def _diamond(self, pitch: float, roll: float) -> float:
        n = 0.0
        for v, m in ((pitch, self.pitch_max), (roll, self.roll_max)):
            if m > 0.0:
                n += abs(v) / m
            elif v != 0.0:
                return math.inf
        return n

    def contains(self, pose: BodyPose, tol: float = 1e-6) -> bool:
        if not pose.is_finite() or abs(pose.x_mm) > self.x_max_mm + tol:
            return False
        if self._diamond(pose.pitch, pose.roll) > 1.0 + tol:
            return False
        lo, hi = self.height_interval(pose.pitch, pose.roll)
        return lo - tol <= pose.height_mm <= hi + tol

    def _attitude(self, pitch: float, roll: float) -> Tuple[float, float]:
        """Attitude part of ``clamp``: box, diamond, then shrink until the band holds it."""
        pitch = max(-self.pitch_max, min(self.pitch_max, pitch))
        roll = max(-self.roll_max, min(self.roll_max, roll))
        n = self._diamond(pitch, roll)
        if n > 1.0:
            pitch, roll = pitch / n, roll / n
        lo, hi = self.height_interval(pitch, roll)
        if lo > hi:
            k_ok, k_bad = 0.0, 1.0
            for _ in range(40):
                k = 0.5 * (k_ok + k_bad)
                lo, hi = self.height_interval(k * pitch, k * roll)
                if lo <= hi:
                    k_ok = k
                else:
                    k_bad = k
            pitch, roll = k_ok * pitch, k_ok * roll
        return pitch, roll

    def attitude_feasible(self, pitch: float, roll: float, tol: float = 1e-9) -> bool:
        """Inside the diamond and holdable by the height band at some height."""
        if self._diamond(pitch, roll) > 1.0 + tol:
            return False
        lo, hi = self.height_interval(pitch, roll)
        return lo <= hi + tol

    def _largest_share(self, pitch: float, roll: float, d_pitch: float,
                       d_roll: float) -> float:
        """Largest k in 0..1 with (pitch, roll) + k (d_pitch, d_roll) feasible, starting
        from a feasible attitude (bisection)."""
        if self.attitude_feasible(pitch + d_pitch, roll + d_roll):
            return 1.0
        k_ok, k_bad = 0.0, 1.0
        for _ in range(30):
            k = 0.5 * (k_ok + k_bad)
            if self.attitude_feasible(pitch + k * d_pitch, roll + k * d_roll):
                k_ok = k
            else:
                k_bad = k
        return k_ok

    def add_within(self, pose: BodyPose, d_pitch: float, d_roll: float
                   ) -> Tuple[BodyPose, float, float]:
        """Add as much of an attitude correction to ``pose`` as stays feasible, then clamp.

        Per axis the inward part (toward 0) is applied in full first; the outward parts get
        the largest common share that fits. Returns (pose, applied d_pitch, applied d_roll)."""
        p0, r0 = pose.pitch, pose.roll
        inward = [_inward_part(c, d) for c, d in ((p0, d_pitch), (r0, d_roll))]
        k_in = self._largest_share(p0, r0, inward[0], inward[1])
        p1, r1 = p0 + k_in * inward[0], r0 + k_in * inward[1]
        out_p, out_r = d_pitch - inward[0], d_roll - inward[1]
        k_out = self._largest_share(p1, r1, out_p, out_r) if (out_p or out_r) else 1.0
        dp = k_in * inward[0] + k_out * out_p
        dr = k_in * inward[1] + k_out * out_r
        out = BodyPose(pose.height_mm, p0 + dp, r0 + dr, pose.x_mm)
        return self.clamp(out), dp, dr

    def clamp(self, pose: BodyPose) -> BodyPose:
        """Project ``pose`` into the set: x and the attitude onto their limits (the attitude
        shrunk if no height can hold it), then the height into the band that attitude
        leaves. NaN components fall back to 0 or the middle of the band."""
        x = pose.x_mm if math.isfinite(pose.x_mm) else 0.0
        x = max(-self.x_max_mm, min(self.x_max_mm, x))
        pitch = pose.pitch if math.isfinite(pose.pitch) else 0.0
        roll = pose.roll if math.isfinite(pose.roll) else 0.0
        pitch, roll = self._attitude(pitch, roll)
        lo, hi = self.height_interval(pitch, roll)
        h = pose.height_mm if math.isfinite(pose.height_mm) else 0.5 * (lo + hi)
        return BodyPose(max(lo, min(hi, h)), pitch, roll, x)

    def samples(self) -> List[BodyPose]:
        """Poses the envelope check validates: the attitude diamond's vertices, edge
        midpoints and center, each over its height interval in steps of at most
        ``SAMPLE_HEIGHT_STEP_MM`` (at least bottom, middle and top) at x = -max, 0, +max."""
        pm, rm = self.pitch_max, self.roll_max
        att = [(0.0, 0.0), (pm, 0.0), (-pm, 0.0), (0.0, rm), (0.0, -rm)]
        att += [(sa * 0.5 * pm, sb * 0.5 * rm) for sa in (-1.0, 1.0) for sb in (-1.0, 1.0)]
        xs = sorted({-self.x_max_mm, 0.0, self.x_max_mm})
        out: List[BodyPose] = []
        seen = set()
        for pitch, roll in att:
            pitch, roll = self._attitude(pitch, roll)
            lo, hi = self.height_interval(pitch, roll)
            n = max(2, math.ceil((hi - lo) / SAMPLE_HEIGHT_STEP_MM - 1e-9))
            for h in (lo + (hi - lo) * k / n for k in range(n + 1)):
                for x in xs:
                    key = (round(h, 6), round(pitch, 9), round(roll, 9), round(x, 6))
                    if key not in seen:
                        seen.add(key)
                        out.append(BodyPose(h, pitch, roll, x))
        return out
