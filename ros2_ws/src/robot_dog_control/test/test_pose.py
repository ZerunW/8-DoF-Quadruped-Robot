"""Body pose layer: rigid transform with planar legs, pose limits, clamping."""

import math
import random
from dataclasses import astuple

import pytest

from robot_dog_control.pose import BodyPose, PoseLimits, foot_in_leg, hip_offsets

HIPS = ((91.5, 88.2), (91.5, -88.2), (-91.5, 88.2), (-91.5, -88.2))   # lf rf lb rb
C = 20.0
LIM = PoseLimits(130.0, 175.0, math.radians(10.0), math.radians(8.0), 10.0, HIPS)


def _to_support(pose, hip, fx, fz):
    """Leg-frame foot (planar leg at y = hip_y) in the support frame, R = Ry(pitch) Rx(roll)."""
    cp, sp, cr, sr = (math.cos(pose.pitch), math.sin(pose.pitch), math.cos(pose.roll),
                      math.sin(pose.roll))
    r = [[cp, sp * sr, sp * cr], [0.0, cr, -sr], [-sp, cp * sr, cp * cr]]
    p = (hip[0] + fx, hip[1], fz)
    body = [pose.x_mm * (i == 0) + r[i][2] * pose.height_mm for i in range(3)]
    return [body[i] + sum(r[i][k] * p[k] for k in range(3)) for i in range(3)]


@pytest.mark.parametrize('pose', [BodyPose(155.0), BodyPose.from_deg(150.0, 10.0, 0.0),
                                  BodyPose.from_deg(155.0, 0.0, 8.0),
                                  BodyPose.from_deg(140.0, 6.0, -5.0, 12.0)])
def test_foot_in_leg_inverts_the_rigid_transform(pose):
    for hip in HIPS:
        for gx, gz in ((hip[0] + C, 0.0), (hip[0] + C + 30.0, 0.0), (hip[0] + C - 20.0, 18.0)):
            fx, fz = foot_in_leg(pose, hip[0], hip[1], gx, gz)
            x, _, z = _to_support(pose, hip, fx, fz)
            assert x == pytest.approx(gx, abs=1e-9) and z == pytest.approx(gz, abs=1e-9)


def test_pitch_and_roll_signs():
    # nose down: front legs shorter, back legs longer
    pose = BodyPose.from_deg(150.0, 10.0, 0.0)
    s, c = math.sin(math.radians(10.0)), math.cos(math.radians(10.0))
    lf = foot_in_leg(pose, 91.5, 88.2, 91.5 + C, 0.0)
    lb = foot_in_leg(pose, -91.5, 88.2, -91.5 + C, 0.0)
    assert lf == pytest.approx((c * (91.5 + C) - 91.5, s * (91.5 + C) - 150.0))
    assert lf[1] > -150.0 > lb[1]
    # right side down: right legs shorter, left legs longer
    pose = BodyPose.from_deg(155.0, 0.0, 8.0)
    s, c = math.sin(math.radians(8.0)), math.cos(math.radians(8.0))
    lf = foot_in_leg(pose, 91.5, 88.2, 91.5 + C, 0.0)
    rf = foot_in_leg(pose, 91.5, -88.2, 91.5 + C, 0.0)
    assert rf == pytest.approx((C, (-155.0 * c + s * 88.2) / c))
    assert rf[1] > -155.0 > lf[1]
    # a body shift moves every foot back
    pose = BodyPose(155.0, 0.0, 0.0, 15.0)
    assert foot_in_leg(pose, 91.5, 88.2, 91.5 + C, 0.0) == pytest.approx((C - 15.0, -155.0))


def test_clamp_keeps_every_hip_in_the_band_and_the_attitude_in_the_diamond():
    rng = random.Random(3)
    for _ in range(2000):
        q = BodyPose(rng.uniform(80.0, 230.0), math.radians(rng.uniform(-30, 30)),
                     math.radians(rng.uniform(-30, 30)), rng.uniform(-40, 40))
        c = LIM.clamp(q)
        assert LIM.contains(c, tol=1e-6), (q, c)
        k = math.cos(c.pitch) * math.cos(c.roll)
        for d in hip_offsets(c.pitch, c.roll, HIPS):
            assert 130.0 - 1e-6 <= c.height_mm * k + d <= 175.0 + 1e-6
        assert abs(c.pitch) / LIM.pitch_max + abs(c.roll) / LIM.roll_max <= 1.0 + 1e-9
        assert astuple(LIM.clamp(c)) == pytest.approx(astuple(c), abs=1e-9)   # idempotent
    for q in (BodyPose(130.0), BodyPose.from_deg(155.0, -10.0, 0.0),
              BodyPose.from_deg(155.0, 0.0, 8.0), BodyPose(155.0, 0.0, 0.0, -10.0)):
        assert LIM.contains(q) and LIM.clamp(q) == q
    c = LIM.clamp(BodyPose.from_deg(155.0, 10.0, 8.0))           # the diamond scales a lean
    assert c.pitch_deg == pytest.approx(5.0) and c.roll_deg == pytest.approx(4.0)
    assert LIM.clamp(BodyPose(math.nan, math.nan, 0.0, math.nan)) == BodyPose(152.5)


def test_add_within_keeps_the_command_and_uses_the_room_left():
    d = math.radians
    cmd = BodyPose.from_deg(155.0, 0.0, 8.0)                     # roll at its limit
    pose, dp, dr = LIM.add_within(cmd, d(-3.0), d(1.0))
    assert abs(dp) < 1e-8 and abs(dr) < 1e-8                     # both outward: no room
    pose, dp, dr = LIM.add_within(cmd, d(-3.75), d(-2.5))
    assert dr == pytest.approx(d(-2.5), abs=1e-12)               # inward: in full
    assert dp == pytest.approx(d(-3.125), abs=1e-6)              # outward: the room it frees
    rng = random.Random(7)
    for _ in range(2000):
        q = LIM.clamp(BodyPose(rng.uniform(120, 185), d(rng.uniform(-12, 12)),
                               d(rng.uniform(-10, 10))))
        cp, cr = d(rng.uniform(-8, 8)), d(rng.uniform(-8, 8))
        out, dp, dr = LIM.add_within(q, cp, cr)
        assert LIM.contains(out, tol=1e-6)
        assert out.pitch == pytest.approx(q.pitch + dp, abs=1e-9)
        for c, want, got in ((q.pitch, cp, dp), (q.roll, cr, dr)):
            if abs(c + want) <= abs(c):                          # inward on this axis
                assert got == pytest.approx(want, abs=1e-9)


def test_samples_cover_the_set_and_limits_are_validated():
    s = LIM.samples()
    assert all(LIM.contains(q) for q in s)
    assert min(q.height_mm for q in s) == pytest.approx(130.0)
    assert max(q.height_mm for q in s) == pytest.approx(175.0)
    assert max(q.pitch_deg for q in s) == pytest.approx(10.0)
    assert min(q.roll_deg for q in s) == pytest.approx(-8.0)
    assert {q.x_mm for q in s} == {-10.0, 0.0, 10.0}
    assert PoseLimits(150.0, 150.0, 0.0, 0.0, 0.0, HIPS).samples() == [BodyPose(150.0)]
    assert LIM.validate() == []
    assert PoseLimits(160.0, 150.0, 0.1, 0.1, 0.0, HIPS).validate()
    assert PoseLimits(150.0, 160.0, math.radians(40.0), 0.1, 0.0, HIPS).validate()
    assert PoseLimits(150.0, 160.0, 0.1, 0.1, math.nan, HIPS).validate()
