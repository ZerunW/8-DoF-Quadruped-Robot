"""Gait generator: phases, stride, speed schedule, the validated foot patch, the servo rate
of the configured gaits and distance walks on the planned body speed."""

import math
from dataclasses import replace

import pytest

from robot_dog_control.config import controller_params_from_config, load_robot_config
from robot_dog_control.controller import GaitControllerCore, State
from robot_dog_control.gait import (LEFT_LEGS, GaitGenerator, GaitParams, GaitPreset, Knot,
                                    stance_count)

DT = 0.02
PARAMS = controller_params_from_config(load_robot_config())
PRESETS = sorted(PARAMS.gait_presets)
#: largest scheduled servo step per 20 ms tick (deg)
MAX_SERVO_STEP_DEG = 5.6


def gen_for(name):
    preset = GaitPreset.from_mapping(PARAMS.gait_presets[name], name)
    return GaitGenerator(replace(PARAMS.gait_params(), schedule=(Knot(0.0, name, preset),)))


def run(gen, seconds, v=0.0, w=0.0, step=False):
    return [gen.update(DT, v, w, step) for _ in range(int(round(seconds / DT)))]


@pytest.mark.parametrize('name', PRESETS)
def test_duty_offsets_and_flight_phases(name):
    gen = gen_for(name)
    g = gen.preset
    T = g.period_s
    touch = {i: [] for i in range(4)}
    frames, prev = [], None
    for _ in range(int(round(20 * T / DT))):
        feet = gen.update(DT, 0.6 * gen.v_max, 0.0)
        for i in range(4):
            if prev is not None and feet[i].in_stance and not prev[i].in_stance:
                touch[i].append(gen.phase)
        frames.append(feet)
        prev = feet
    frames = frames[int(2 * T / DT):]
    for i in range(4):
        stance = sum(f[i].in_stance for f in frames) / len(frames)
        assert stance == pytest.approx(g.duty, abs=DT / T + 0.01), (name, i)
        assert touch[i]
        for ph in touch[i]:                     # touch-down at global phase 1 - offset_i
            d = (ph - (1.0 - g.offsets[i]) % 1.0 + 0.5) % 1.0 - 0.5
            assert abs(d) <= DT / T + 1e-9, (name, i, ph)
    flight = sum(1 for feet in frames if stance_count(feet) == 0) / len(frames)
    uncovered = sum(all((k / 1000.0 + o) % 1.0 >= g.duty for o in g.offsets)
                    for k in range(1000)) / 1000.0
    assert flight == pytest.approx(uncovered, abs=2 * DT / T + 1e-3)
    assert (flight > 0.1) == (g.duty < 0.5)


def test_planted_feet_move_with_the_body_inside_the_stride():
    gen = GaitGenerator(PARAMS.gait_params())
    p = gen.params
    b, k = p.half_track_mm / 1000.0, p.turn_stride_gain
    prev = gen.update(0.0, 0.0, 0.0)
    checked = 0
    for seconds, v, w in ((2.5, 0.05, 0.0), (2.5, gen.v_max, 0.2), (2.5, -0.04, -0.3),
                          (3.0, 0.0, 0.0)):
        for _ in range(int(round(seconds / DT))):
            feet = gen.update(DT, v, w)
            g = gen.preset
            lim = p.max_stride_mm / (g.duty * g.period_s * 1000.0)
            side = [gen.amp * max(-lim, min(lim, gen.v + s * k * gen.w * b)) for s in (-1, 1)]
            for i in range(4):
                if prev[i].in_stance and feet[i].in_stance and not gen.is_idle():
                    want = -side[0 if i in LEFT_LEGS else 1] * 1000.0 * DT
                    assert feet[i].dx - prev[i].dx == pytest.approx(want, abs=1e-9)
                    assert feet[i].lift == 0.0
                    checked += 1
            assert max(abs(s) for s in gen.strides) <= p.max_stride_mm + 1e-9
            prev = feet
    assert checked > 500 and gen.is_idle()


@pytest.mark.parametrize('name', PRESETS)
def test_stepping_in_place_has_zero_stride_and_full_lift(name):
    gen = gen_for(name)
    g = gen.preset
    late = run(gen, 4 * g.period_s, step=True)[int(2 * g.period_s / DT):]
    assert not gen.is_idle() and gen.amp == 1.0
    assert all(f.dx == 0.0 for feet in late for f in feet)
    assert max(f.lift for feet in late for f in feet) == pytest.approx(g.step_height_mm, rel=0.02)
    stop = run(gen, 1.0)                        # releasing the step stops at the footholds
    assert gen.is_idle() and all(f == (0.0, 0.0, True) for f in stop[-1])


def test_speed_limit_keeps_the_stride_and_turning_headroom():
    for name in PRESETS:
        gen = gen_for(name)
        g, p = gen.preset, gen.params
        head = p.turn_stride_gain * p.turn_headroom_rps * p.half_track_mm / 1000.0
        assert p.turn_headroom_rps == PARAMS.heading_max_rps > 0.0
        assert p.speed_limit() == pytest.approx(min(p.max_speed_mps, p.max_stride_mm / (
            g.duty * g.period_s) / 1000.0 - head), abs=1e-9)
        run(gen, 3.0, v=5.0)
        assert gen.v == pytest.approx(p.speed_limit())


def test_turning_direction_and_stride_limits():
    gen = gen_for('trot')
    run(gen, 2.0, 0.0, 0.6)                     # turn left (CCW) in place
    s_lf, s_rf, s_lb, s_rb = gen.strides
    assert s_rf > 0 > s_lf and s_rb > 0 > s_lb and s_rf == pytest.approx(-s_lf)
    run(gen, 2.0, 0.08, 0.6)
    assert all(abs(s) <= gen.params.max_stride_mm + 1e-9 for s in gen.strides)


@pytest.mark.parametrize('script', [[(3.0, 'top', 0.6)], [(3.0, 'top', 0.6), (3.0, '-top', -0.6)],
                                    [(3.0, 'top', 0.6), (0.3, 0.0, 0.0), (3.0, 'top', -0.6)]])
def test_every_foot_stays_in_the_validated_patch(script):
    """Lift-off x0 and landing L of every swing satisfy |x0|, |L|, |x0 + L| <= S/2."""
    gen = GaitGenerator(PARAMS.gait_params())
    half = 0.5 * gen.params.max_stride_mm
    for seconds, v, w in script + [(3.0, 0.0, 0.0)]:
        v = {'top': gen.v_max, '-top': -gen.v_max}.get(v, v)
        for _ in range(int(round(seconds / DT))):
            for i, f in enumerate(gen.update(DT, v, w)):
                assert abs(f.dx) <= half + 1e-9
                if not f.in_stance:
                    x0, land = gen.x0[i], gen.land[i]
                    assert max(abs(x0), abs(land), abs(x0 + land)) <= half + 1e-9
    assert gen.is_idle()


def test_speed_picks_the_phase_pattern_with_hysteresis():
    sched = replace(PARAMS.gait_params(), max_speed_mps=1.0, max_stride_mm=200.0)
    knots, h = sched.schedule, sched.hysteresis_mps
    gen = GaitGenerator(sched)
    profile = [k * 0.004 for k in range(int((knots[-1].speed + 0.05) / 0.004) + 1)]
    switches, prev = [], 0
    for v in profile + profile[::-1]:
        run(gen, 0.25, v=v)
        if gen.knot != prev:
            switches.append((gen.knot, prev, gen.speed))
        prev = gen.knot
    ups = [(j, s) for j, i, s in switches if j > i]
    downs = [(j, s) for j, i, s in switches if j < i]
    assert [j for j, _ in ups] == list(range(1, len(knots)))
    assert all(s >= knots[j].speed for j, s in ups)
    assert all(s < knots[j + 1].speed - h + 1e-9 for j, s in downs)


@pytest.mark.parametrize('kw', [dict(offsets=(0.0, 0.5, 1.0, 0.0)), dict(offsets=(0.0, 0.5)),
                                dict(duty=0.05), dict(period_s=0.0),
                                dict(step_height_mm=math.nan),
                                dict(swing_x=(0.0, 0.5, 1.0), swing_z=(0.0, 1.0))])
def test_preset_validation(kw):
    with pytest.raises(ValueError):
        GaitGenerator(GaitParams.single(GaitPreset(**kw)))


@pytest.mark.parametrize('kw', [dict(schedule=()), dict(schedule=(Knot(0.1, 'a', GaitPreset()),)),
                                dict(schedule=(Knot(0.0, 'a', GaitPreset()),
                                               Knot(0.0, 'b', GaitPreset()))),
                                dict(switch_periods=0.45), dict(turn_headroom_rps=-0.1)])
def test_schedule_validation(kw):
    with pytest.raises(ValueError):
        GaitGenerator(replace(GaitParams(), **kw))


def _servo_peak(params, segments):
    """Largest per-tick servo step (deg) of the controller over (seconds, v, w, step)
    segments and a stop."""
    core = GaitControllerCore(params)
    servo = [172.0] * 8
    core.set_measured_pose(servo)
    t, peak = 0.0, 0.0
    for _ in range(int(4.0 / DT)):
        t += DT
        servo = core.step(t).servo
    for seconds, v, w, step in list(segments) + [(3.0, 0.0, 0.0, False)]:
        for _ in range(int(round(seconds / DT))):
            t += DT
            core.set_cmd_step(step, t)
            core.set_cmd_vel(v, w, t)
            out = core.step(t)
            peak = max(peak, max(abs(a - b) for a, b in zip(out.servo, servo)))
            servo = out.servo
    assert core.state == State.STAND and core.gait.is_idle()
    return peak


@pytest.mark.parametrize('schedule', [None, (['walk', 'trot'], [0.0, 0.1])])
def test_configured_gaits_stay_within_the_servo_rate_limit(schedule):
    params = PARAMS if schedule is None else replace(
        PARAMS, gait_schedule=schedule[0], gait_schedule_speeds_mps=schedule[1])
    top = GaitGenerator(params.gait_params()).v_max
    for segments in ([(5.0, top, 0.0, False)], [(5.0, 0.0, 0.5, False)],
                     [(5.0, 0.0, 0.0, True)], [(5.0, top, params.max_turn_rps, False)],
                     [(3.0, top, 0.0, False), (3.0, -top, 0.0, False)],
                     [(3.0, 0.05, 0.0, False), (3.0, top, 0.0, False), (3.0, 0.05, 0.0, False)]):
        peak = _servo_peak(params, segments)
        assert peak <= MAX_SERVO_STEP_DEG, (segments, peak)


@pytest.mark.parametrize('speed,distance', [(0.1, 0.3), (-0.06, 0.1), (0.12, 0.02)])
def test_a_distance_walk_lands_on_the_planned_distance(speed, distance):
    """api._Approach on the integral of the gait's planned body speed, as walk(distance=...)
    runs it."""
    pytest.importorskip('rclpy')
    from robot_dog_control import api
    gen = GaitGenerator(PARAMS.gait_params())
    decel = api.BRAKE_FRACTION * PARAMS.max_speed_mps / PARAMS.ramp_time_s
    approach = api._Approach(math.copysign(distance, speed), speed, decel)
    x, ticks = 0.0, 0
    while True:
        v, done = approach.update(x)
        if done:
            break
        gen.update(DT, v, 0.0)
        x += gen.body_speed * DT
        ticks += 1
        assert ticks * DT < distance / abs(speed) * 2.0 + PARAMS.ramp_time_s + 5.0
    while not gen.is_idle():
        gen.update(DT, 0.0, 0.0)
        x += gen.body_speed * DT
    assert x * speed > 0.0 and abs(x) == pytest.approx(distance, abs=api.ARRIVE_TOL_M)
