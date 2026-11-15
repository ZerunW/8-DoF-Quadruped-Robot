"""Leg kinematics: FK, IK inside the joint region, workspace clamp and the envelope check."""

import math
import random
from dataclasses import replace

import pytest

from robot_dog_control.config import controller_params_from_config, load_robot_config
from robot_dog_control.controller import ControllerParams, build_components
from robot_dog_control.gait import GaitPreset, envelope_checks, same_for_all_legs
from robot_dog_control.joint_region import JointRegion
from robot_dog_control.kinematics import LegGeometry, LegKinematics, validate_envelope
from robot_dog_control.leg_map import LegMap
from robot_dog_control.pose import PoseLimits

CFG = load_robot_config()
PARAMS = controller_params_from_config(CFG)
REGION = PARAMS.joint_region()
KIN = LegKinematics(LegGeometry.from_mapping(PARAMS.geom), region=REGION)
SERVO_MIN = [140.0, 150.0] * 4
SERVO_MAX = [220.0, 210.0] * 4


def test_fk_reference_values():
    x, y, (kx, ky) = KIN.fk(90, 90)
    assert x == pytest.approx(0.2, abs=0.1) and y == pytest.approx(-155.8, abs=0.1)
    assert kx == pytest.approx(77.8, abs=0.1) and ky == pytest.approx(-77.8, abs=0.1)
    x, y, _ = KIN.fk(82, 82)
    assert x == pytest.approx(-21.5, abs=0.1) and y == pytest.approx(-154.3, abs=0.1)


def test_fk_ik_round_trip_over_the_joint_region():
    n = 0
    for m1, m2 in REGION.grid(4.0):
        if REGION.margin(m1, m2) < 0.5:
            continue
        x, y, _ = KIN.fk(m1, m2)
        for guess in (None, (m1 + 2.0, m2 - 2.0)):
            r = KIN.ik(x, y, guess)
            assert r.ok and abs(r.m1 - m1) < 0.05 and abs(r.m2 - m2) < 0.05, (m1, m2, r)
        n += 1
    assert n > 100


def test_ik_never_returns_a_pose_outside_the_region():
    rng = random.Random(7)
    n_ok = 0
    for _ in range(2000):
        x, y = rng.uniform(-260, 260), rng.uniform(-320, 60)
        guess = None if rng.random() < 0.3 else (rng.uniform(-90, 270), rng.uniform(-90, 270))
        r = KIN.ik(x, y, guess)
        if math.isfinite(r.m1):
            assert REGION.contains(r.m1, r.m2, tol=1e-9), (x, y, guess, r)
        if r.ok:
            n_ok += 1
            fx, fy, _ = KIN.fk(r.m1, r.m2)
            assert math.hypot(fx - x, fy - y) <= KIN.TOL_MM + 1e-9
    assert n_ok > 100


@pytest.mark.parametrize('target', [(math.nan, -150.0), (0.0, math.inf), (1e9, -1e9),
                                    (0.0, 0.0), (500.0, -150.0)])
def test_ik_garbage_targets_fail_safely(target):
    r = KIN.ik(*target)
    assert not r.ok
    if math.isfinite(r.m1):
        assert REGION.contains(r.m1, r.m2, tol=1e-9)


def test_clamp_to_workspace_shrinks_toward_the_center_inside_the_region():
    center = (-20.0, -155.0)
    for target in [(-20.0, -260.0), (-200.0, -155.0), (150.0, -60.0), (0.0, -20.0),
                   (-20.0, -150.0)]:
        x, y, r, clamped = KIN.clamp_to_workspace(*target, center)
        assert r.ok and REGION.contains(r.m1, r.m2)
        assert clamped == (target != (x, y))
        if clamped:
            dx, dy = target[0] - center[0], target[1] - center[1]
            assert abs(dx * (y - center[1]) - dy * (x - center[0])) < 1e-6   # on the segment
            d = math.hypot(target[0] - x, target[1] - y)
            assert not KIN.reachable(x + 0.5 * (target[0] - x) / d, y + 0.5 * (target[1] - y) / d)


def test_joint_region_interpolates_and_clips():
    r = JointRegion([0.0, 10.0, 20.0], [0.0, 5.0, 0.0], [10.0, 20.0, 10.0])
    assert r.interval(5.0) == pytest.approx((2.5, 15.0))
    assert r.contains(10.0, 5.0) and not r.contains(10.0, 4.9) and not r.contains(21.0, 5.0)
    assert r.clip(10.0, 0.0) == (10.0, 5.0) and r.clip(25.0, 50.0) == (20.0, 10.0)
    assert r.clip(5.0, 7.0) == (5.0, 7.0)
    assert r.margin(10.0, 12.0) > 0 > r.margin(10.0, 25.0)
    rng = random.Random(1)
    for _ in range(1000):
        assert r.contains(*r.clip(rng.uniform(-10, 30), rng.uniform(-10, 30)), tol=1e-9)
    assert JointRegion.from_params(*REGION.to_params().values()) == REGION


@pytest.mark.parametrize('m1,lo,hi', [([0.0], [0.0], [1.0]), ([0.0, 0.0], [0.0, 0.0], [1.0, 1.0]),
                                       ([0.0, 1.0], [0.0, 2.0], [1.0, 1.0]),
                                       ([0.0, math.nan], [0.0, 0.0], [1.0, 1.0])])
def test_joint_region_validation(m1, lo, hi):
    with pytest.raises(ValueError):
        JointRegion(m1, lo, hi)
    bad = replace(ControllerParams(), ik_region_m1=m1, ik_region_m2_lo=lo, ik_region_m2_hi=hi)
    assert bad.validate()


def test_servo_limits_contain_the_joint_region():
    lm = LegMap.default()
    lo, hi = CFG['servo_driver']['servo_min'], CFG['servo_driver']['servo_max']
    for i, leg in enumerate(lm.legs):
        for m1, m2 in REGION.polygon():
            st, sc = lm.to_servo(i, m1, m2)
            assert lo[leg.thigh_idx] <= st <= hi[leg.thigh_idx]
            assert lo[leg.calf_idx] <= sc <= hi[leg.calf_idx]


def test_committed_config_passes_the_envelope_inside_the_region():
    comp = build_components(PARAMS)
    assert comp.report.ok and comp.walk_allowed, comp.report.summary()
    assert comp.report.min_region_margin_deg > 0.0
    lm = LegMap.default()
    checks, _ = envelope_checks(PARAMS.presets(), PARAMS.max_stride_mm, PARAMS.pose_limits(),
                                PARAMS.stand_fx_mm, PARAMS.stand_height_mm,
                                PARAMS.rest_height_mm, n=5)
    for name, scenarios in checks.items():
        for sc in scenarios:
            for i in range(4):
                for fx, fz in sc[i]:
                    r = KIN.ik(*lm.body_to_canonical(i, fx, fz))
                    assert r.ok and REGION.contains(r.m1, r.m2), (name, i, fx, fz)


def _envelope(max_stride_mm=40.0, height=(145.0, 165.0), **preset_kw):
    box = LegKinematics()
    hips = tuple((leg.hip_x_mm, leg.hip_y_mm) for leg in LegMap.default().legs)
    lim = PoseLimits(height[0], height[1], math.radians(8.0), math.radians(8.0), 0.0, hips)
    preset = GaitPreset(**{**dict(duty=0.6), **preset_kw})
    checks, names = envelope_checks({'trot': preset}, max_stride_mm, lim, 20.0, 155.0, 145.0)
    return validate_envelope(box, LegMap.default(), SERVO_MIN, SERVO_MAX, checks), names


def test_validate_envelope_names_the_failing_parameter():
    rep, _ = _envelope()
    assert rep.ok and rep.points_checked > 400, rep.summary()
    rep, names = _envelope(max_stride_mm=200.0)
    assert rep.failed == [names['trot']] and names['trot'].startswith('max_stride_mm')
    rep, _ = _envelope(height=(80.0, 165.0))
    assert rep.failed[0].startswith('height_min_mm')
    rep = validate_envelope(LegKinematics(), LegMap.default(), [140.0, 170.0] * 4, SERVO_MAX,
                            {'stand': same_for_all_legs([(20.0, -155.0), (60.0, -140.0)])})
    assert not rep.ok and any('outside limits' in m for m in rep.messages)


def test_geometry_validation():
    with pytest.raises(ValueError):
        LegKinematics(LegGeometry(L1=-1.0))
    with pytest.raises(ValueError):
        LegGeometry.from_mapping({'L1': 110.0, 'bogus': 1.0})
