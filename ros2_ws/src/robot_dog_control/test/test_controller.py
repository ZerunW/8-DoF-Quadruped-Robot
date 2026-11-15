"""Controller core in closed loop with a kinematic body plant (BodySim), plus the ROS node."""

import math
from dataclasses import replace

import pytest

from robot_dog_control.body_sim import BodySim
from robot_dog_control.config import (controller_params_from_config, load_robot_config,
                                      stand_servo_angles)
from robot_dog_control.controller import (DEBUG_INDEX, ControllerParams, GaitControllerCore,
                                          ImuSample, State)
from robot_dog_control.leg_map import LEG_NAMES

DT = 0.02
SERVO_MIN = [140.0, 150.0] * 4
SERVO_MAX = [220.0, 210.0] * 4


class Rig:
    """Controller core + BodySim plant stepped together at 50 Hz."""

    def __init__(self, params=None, ground_roll=0.0, ground_pitch=0.0, yaw_drift=0.0):
        self.core = GaitControllerCore(params or ControllerParams())
        self.sim = BodySim(self.core.kin, self.core.leg_map, ground_roll, ground_pitch)
        self.drift = yaw_drift                  # extra body yaw rate while walking (rad/s)
        self.servo = [172.0] * 8                # servo power-on pose
        self.core.set_measured_pose(self.servo)
        self.t = 0.0
        self.out = None
        self.history = []                       # (previous servo, servo, roll, pitch, yaw deg)

    def step(self, v=None, w=0.0):
        self.t += DT
        walking = self.core.state == State.WALK
        r, p, y = self.sim.update(self.servo, DT, self.core.gait.w + self.drift * walking)
        self.core.set_imu(ImuSample(r, p, y, *self.sim.rates, 0.0, self.t))
        if v is not None:
            self.core.set_cmd_vel(v, w, self.t)
        out = self.core.step(self.t)
        if out is not None:
            self.history.append((self.servo, out.servo, *map(math.degrees, (r, p, y))))
            self.servo, self.out = out.servo, out
        return out

    def run(self, seconds, v=None, w=0.0):
        for _ in range(int(round(seconds / DT))):
            self.step(v, w)

    def d(self, name):
        return self.out.debug[DEBUG_INDEX[name]]

    def max_servo_step(self, since=0):
        return max(max(abs(a - b) for a, b in zip(h[0], h[1])) for h in self.history[since:])

    def assert_servo_in_limits(self):
        for h in self.history:
            assert all(lo - 1e-9 <= s <= hi + 1e-9 for s, lo, hi in zip(h[1], SERVO_MIN, SERVO_MAX))


def test_startup_blend_is_a_straight_line_in_joint_space():
    rig = Rig()
    start, poses = list(rig.servo), []
    while rig.core._blend is not None or not poses:
        rig.step()
        poses.append(rig.servo)
    end = poses[-1]
    fractions = []
    for s in poses:
        a = [(x - x0) / (x1 - x0) for x, x0, x1 in zip(s, start, end) if abs(x1 - x0) > 1.0]
        assert max(a) - min(a) < 1e-6           # every servo at the same fraction of its move
        fractions.append(a[0])
    assert fractions == sorted(fractions) and fractions[-1] == pytest.approx(1.0)
    assert rig.core.state == State.STAND and rig.max_servo_step() < 0.5
    for left, right in ((0, 4), (1, 5), (2, 6), (3, 7)):      # right servos mirror the left
        assert end[left] + end[right] == pytest.approx(360.0, abs=0.05)
    assert all(rig.d(f'{leg}_fx') == pytest.approx(rig.core.p.stand_fx_mm) for leg in LEG_NAMES)


def test_the_driver_stand_pose_is_the_controller_stand_pose():
    cfg = load_robot_config()
    angles = stand_servo_angles(cfg)
    lo, hi = cfg['servo_driver']['servo_min'], cfg['servo_driver']['servo_max']
    assert all(lo[i] <= angles[i] <= hi[i] for i in range(8))
    core = GaitControllerCore(replace(controller_params_from_config(cfg), startup_blend_s=0.0))
    core.set_measured_pose(angles)
    assert max(abs(a - b) for a, b in zip(core.step(0.0).servo, angles)) < 0.01


@pytest.mark.parametrize('ground_roll,ground_pitch', [(0.0, 3.0), (2.0, 0.0), (-2.5, -4.0)])
def test_leveling_converges_on_a_slope(ground_roll, ground_pitch):
    rig = Rig(ground_roll=ground_roll, ground_pitch=ground_pitch)
    rig.run(30.0)
    assert abs(rig.history[-1][2]) < 0.3 and abs(rig.history[-1][3]) < 0.3
    fz = {leg: rig.d(f'{leg}_fz') for leg in LEG_NAMES}
    if ground_pitch > 0:                        # nose-down slope: front legs longer
        assert fz['lf'] < fz['lb'] and fz['rf'] < fz['rb']
    if ground_roll > 0:                         # right side down: right legs longer
        assert fz['rf'] < fz['lf'] and fz['rb'] < fz['lb']
    assert rig.d('ws_clamped_legs') == 0
    rig.assert_servo_in_limits()


def test_walks_and_stops_on_a_slope_with_leveling():
    rig = Rig(ground_roll=2.0, ground_pitch=2.0)
    rig.run(10.0)
    since = len(rig.history)
    rig.run(8.0, v=0.08, w=0.6)                 # top speed and turn
    assert rig.core.state == State.WALK
    assert rig.d('gait_speed') == pytest.approx(rig.core.gait.v_max)
    rig.run(4.0, v=0.0)
    assert rig.core.state == State.STAND and rig.core.gait.is_idle()
    assert rig.d('ws_clamped_legs') == 0
    assert rig.max_servo_step(since) < 8.0      # 400 deg/s per 20 ms tick
    rig.assert_servo_in_limits()


@pytest.mark.parametrize('hold', [True, False])
def test_heading_hold_corrects_drift_within_its_output_limit(hold):
    rig = Rig(replace(ControllerParams(), heading_hold_enable=hold), yaw_drift=0.03)
    rig.run(3.0)
    u_yaw = []
    for _ in range(int(12.0 / DT)):
        rig.step(v=0.05)
        u_yaw.append(abs(rig.d('u_yaw')))
    worst = max(abs(h[4]) for h in rig.history[-500:])
    assert worst < 5.0 if hold else worst > 10.0
    assert max(u_yaw) <= rig.core.p.heading_max_rps + 1e-9
    assert (max(u_yaw) > 0.0) == hold


def test_bad_parameters_are_refused():
    trot = ControllerParams().gait_presets['trot']
    for bad in (dict(gait_presets={'trot': {**trot, 'duty': 0.05}}), dict(rest_height_mm=200.0),
                dict(gait_schedule=['canter']), dict(gait_schedule_speeds_mps=[0.0, 0.1]),
                dict(leg_trim_mm=[0.0, 0.0, 0.0]), dict(leg_thigh_idx=[1, 5, 3, 3]),
                dict(level_deadband_deg=6.0), dict(heading_max_rps=-0.1),
                dict(geom={**ControllerParams().geom, 'L1': -5.0})):
        with pytest.raises(ValueError):
            GaitControllerCore(replace(ControllerParams(), **bad))
    # a gait outside the envelope disables walking, not standing
    rig = Rig(replace(ControllerParams(), max_stride_mm=200.0))
    assert not rig.core.walk_allowed and rig.core.level_allowed
    assert any('max_stride_mm' in msg for _, msg in rig.core.log.lines)
    rig.run(3.0)
    rig.run(2.0, v=0.08)
    assert rig.core.state == State.STAND and rig.core.gait.is_idle()
    # live updates are validated like startup, gait parameters only while idle
    rig = Rig()
    core = rig.core
    for bad in (dict(leg_x_sign=[1, 1, 1, 1]), dict(level_max_deg=40.0),
                dict(max_stride_mm=200.0)):
        assert not core.update_params(replace(core.p, **bad))[0]
    assert core.update_params(replace(core.p, level_kp=1.5, heading_max_rps=0.3))[0]
    assert core.pid_yaw.out_limit == 0.3 and core.gait.params.turn_headroom_rps == 0.3
    rig.run(3.0)
    rig.run(1.0, v=0.05)
    ok, why = core.update_params(replace(core.p, max_speed_mps=0.06))
    assert not ok and 'stop walking' in why


class _FrontM1Above:
    """Stand-in pair sweep that marks every pair whose front leg m1 is above ``m1_max``."""

    def __init__(self, m1_max):
        self.m1_max = m1_max

    def marked(self, rear, front):
        return front[0] > self.m1_max


def test_guard_holds_a_side_whose_front_back_pair_is_marked():
    rig = Rig()
    rig.run(3.0)
    core, lm = rig.core, rig.core.leg_map
    assert [(s.side, s.rear, s.front) for s in core.interleg] == [('left', 2, 0), ('right', 3, 1)]
    start = len(rig.history)
    rig.run(3.0, v=0.06)
    lf_m1 = [lm.motor_angles(h[1])[0][0] for h in rig.history[start:]]
    m1_max = 0.5 * (min(lf_m1) + max(lf_m1))     # marks half of the lf swing
    while lm.motor_angles(rig.servo)[0][0] > m1_max - 1.0:     # start from a clear pose
        rig.step(v=0.06)
    core.interleg = tuple(s._replace(pairs=_FrontM1Above(m1_max)) if s.side == 'left' else s
                          for s in core.interleg)
    before = rig.d('interleg_holds')
    left = [lm.legs[i].thigh_idx for i in (0, 2)] + [lm.legs[i].calf_idx for i in (0, 2)]
    n_held = 0
    for _ in range(150):
        holds = rig.d('interleg_holds')
        rig.step(v=0.06)
        prev, out = rig.history[-1][:2]
        assert lm.motor_angles(out)[0][0] <= m1_max + 1e-9        # never enters the set
        if rig.d('interleg_holds') > holds:
            n_held += 1
            assert all(out[i] == prev[i] for i in left)           # left legs held
    assert n_held > 10 and rig.d('interleg_holds') - before == n_held
    rf_m1 = [lm.motor_angles(h[1])[1][0] for h in rig.history[-150:]]
    assert max(rf_m1) - min(rf_m1) > 4.0                          # the right side walks on


def test_estop_freezes_and_stand_blends_back():
    rig = Rig()
    rig.run(3.0)
    rig.run(1.13, v=0.08)
    assert rig.core.request_estop()[0]
    frozen = list(rig.servo)
    rig.run(1.0, v=0.08)
    assert rig.core.state == State.ESTOP and rig.servo == frozen
    since = len(rig.history)
    assert rig.core.request_stand(rig.t)[0]
    rig.run(3.0)
    assert rig.core.state == State.STAND and rig.max_servo_step(since) < 1.0
    assert rig.core.request_rest(rig.t)[0]
    rig.run(4.0)
    assert rig.core.state == State.REST
    assert rig.d('kin_height_mm') == pytest.approx(rig.core.p.rest_height_mm, abs=0.01)


def test_controller_waits_for_servo_state_and_driver():
    core = GaitControllerCore(ControllerParams())
    assert core.step(0.02) is None                      # no servo state yet
    core.set_measured_pose([172.0] * 8)
    assert core.step(0.04) is not None
    core.set_driver_enabled(False)
    assert core.step(0.06) is None                      # paused while relaxed
    core.set_driver_enabled(True)
    assert core.step(0.08) is None                      # waits for a fresh pose
    core.set_measured_pose([172.0] * 8)
    out = core.step(0.10)
    assert out is not None and max(abs(s - 172.0) for s in out.servo) < 0.5


def test_gait_controller_node_runs_and_validates_params():
    rclpy = pytest.importorskip('rclpy')
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import JointState

    from robot_dog_control.gait_controller_node import GaitControllerNode
    rclpy.init()
    try:
        node = GaitControllerNode()
        sent = []
        node._pub_cmd.publish = lambda msg: sent.append(list(msg.position))
        node._on_servo_state(JointState(position=[172.0] * 8))
        for _ in range(5):
            node._on_timer()
        assert len(sent) == 5 and all(len(s) == 8 for s in sent)
        assert not node.set_parameters([Parameter('gait_presets.trot.step_height_mm',
                                                  value=95.0)])[0].successful     # envelope
        assert not node.set_parameters([Parameter('leg_mirror_servo',
                                                  value=[True] * 4)])[0].successful   # read-only
        assert node.set_parameters([Parameter('level_kp', value=1.0)])[0].successful
        node.destroy_node()
        with pytest.raises(ValueError):
            GaitControllerNode(parameter_overrides=[Parameter('gait_presets.trot.duty',
                                                              value=0.05)])
    finally:
        rclpy.try_shutdown()
