"""The only path to the PCA9685: clamp, rate limiter, config validation and servo_driver.

Invariant: for any request, the pulse on a pin stays inside [pulse_min, pulse_max] and
corresponds to an angle inside [min_i, max_i].
"""

import math
import random

import pytest

from robot_dog_hw.pca9685 import FULL_OFF, FakePCA9685
from robot_dog_hw.servo_map import (MAX_OFFSET_STEP_DEG, NUM_SERVOS, RateLimiter, ServoConfig,
                                    ServoConfigError, ServoLimiter, check_offset_update,
                                    us_to_counts)

CFG = ServoConfig()
SPECIALS = [math.nan, math.inf, -math.inf, 1e9, -1e9, 1e308, 5e-324, -0.0, 359.999, 360.0,
            720.0, -360.0, 139.999, 220.001, None, 'abc', object(), [1, 2]]


def reference_us(i, angle, cfg=CFG):
    """Independent float implementation of the angle-to-pulse math."""
    a = math.fmod(min(max(angle, cfg.mins[i]), cfg.maxs[i]) + cfg.offsets[i], 360.0)
    if a < 0:
        a += 360.0
    return min(max(int(math.floor(500.0 + a / 360.0 * 2000.0 + 0.5)), 500), 2500)


def evil(rng):
    r = rng.random()
    if r < 0.1:
        return rng.choice(SPECIALS)
    return rng.uniform(-1e4, 1e4) if r < 0.55 else rng.uniform(100.0, 260.0)


def test_fuzz_angle_to_us_never_leaves_limits():
    lim = ServoLimiter(CFG)
    rng = random.Random(1234)
    lo_us = [reference_us(i, CFG.mins[i]) for i in range(NUM_SERVOS)]
    hi_us = [reference_us(i, CFG.maxs[i]) for i in range(NUM_SERVOS)]
    for k in range(50_000):
        i = k % NUM_SERVOS
        a = evil(rng)
        us = lim.angle_to_us(i, a)
        assert isinstance(us, int) and lo_us[i] <= us <= hi_us[i], (i, a, us)
        c, was, nonfinite = lim.clamp_one(i, a)
        assert math.isfinite(c) and CFG.mins[i] <= c <= CFG.maxs[i]
        if nonfinite:
            assert c == CFG.stand[i] and was
    for i in range(NUM_SERVOS):
        for a in range(100, 261):
            assert lim.angle_to_us(i, a) == reference_us(i, a)


def test_fuzz_full_driver_pipeline_bytes():
    """Clamp, rate limiter, pulse conversion and PCA block: decode what reaches the bus."""
    lim = ServoLimiter(CFG)
    rate = RateLimiter(400.0)
    rate.reset(CFG.stand)
    pca = FakePCA9685()
    pca.begin()
    rng = random.Random(99)
    lo_c = {CFG.channels[i]: us_to_counts(reference_us(i, CFG.mins[i])) for i in range(8)}
    hi_c = {CFG.channels[i]: us_to_counts(reference_us(i, CFG.maxs[i])) for i in range(8)}
    for _ in range(10_000):
        safe, _ = lim.clamp([evil(rng) for _ in range(NUM_SERVOS)])
        pose = rate.step(safe, rng.choice([0.0, 0.02, 0.02, 0.5, -1.0, math.nan]))
        pca.write_us(lim.process(pose).by_channel)
        block = pca.blocks[-1]
        assert len(block) == 64
        for ch in range(16):
            assert block[4 * ch] | (block[4 * ch + 1] << 8) == 0
            off_l, off_h = block[4 * ch + 2], block[4 * ch + 3]
            if ch in lo_c:
                assert not off_h & FULL_OFF
                assert lo_c[ch] <= off_l | ((off_h & 0x0F) << 8) <= hi_c[ch]
            else:
                assert off_h == FULL_OFF and off_l == 0     # unused channels held off
    with pytest.raises(ValueError):
        lim.process([172.0] * 7)


def test_pca9685_refuses_out_of_range_pulses():
    pca = FakePCA9685()
    with pytest.raises(OSError):
        pca.write_us({3: 1500})                           # begin() not called
    pca.begin()
    for bad in ({3: 2600}, {3: 400}, {3: math.nan}, {16: 1500}, {-1: 1500}, {3: 'x'}):
        with pytest.raises(ValueError):
            pca.write_us(bad)
    assert pca.frames == []


def _set(seq, i, v):
    seq = list(seq)
    seq[i] = v
    return seq


BAD_CONFIGS = {
    'seven channels': dict(channels=list(CFG.channels)[:7]),
    'duplicate channel': dict(channels=_set(CFG.channels, 1, CFG.channels[0])),
    'channel 16': dict(channels=_set(CFG.channels, 2, 16)),
    'min > max': dict(mins=_set(CFG.mins, 3, 215.0)),
    'max > range': dict(maxs=_set(CFG.maxs, 4, 361.0)),
    'offset 41': dict(offsets=_set(CFG.offsets, 5, 41.0)),
    'offset nan': dict(offsets=_set(CFG.offsets, 5, math.nan)),
    'wraps above range': dict(maxs=_set(CFG.maxs, 6, 350.0), offsets=_set(CFG.offsets, 6, 15.0)),
    'stand below min': dict(stand=_set(CFG.stand, 0, 100.0)),
    'min string': dict(mins=_set(CFG.mins, 1, '150')),
}


@pytest.mark.parametrize('name', sorted(BAD_CONFIGS))
def test_config_validation_rejects(name):
    base = dict(channels=list(CFG.channels), offsets=list(CFG.offsets), mins=list(CFG.mins),
                maxs=list(CFG.maxs), stand=list(CFG.stand))
    with pytest.raises(ServoConfigError):
        ServoConfig(**{**base, **BAD_CONFIGS[name]})


@pytest.mark.parametrize('kw', [dict(pulse_min_us=2500.0, pulse_max_us=500.0),
                                dict(pulse_max_us=5000.0), dict(servo_range_deg=0.0)])
def test_config_validation_rejects_pulse_settings(kw):
    with pytest.raises(ServoConfigError):
        ServoConfig(**kw)


def test_dynamic_offset_update_rules():
    ok = check_offset_update(CFG, _set(CFG.offsets, 3, CFG.offsets[3] + 2.5))
    assert ok.offsets[3] == CFG.offsets[3] + 2.5
    with pytest.raises(ServoConfigError):   # step above MAX_OFFSET_STEP_DEG
        check_offset_update(CFG, _set(CFG.offsets, 3, CFG.offsets[3] + MAX_OFFSET_STEP_DEG + 0.1))
    with pytest.raises(ServoConfigError):   # beyond +-40 even in small steps
        cfg = CFG
        for _ in range(10):
            cfg = check_offset_update(cfg, _set(cfg.offsets, 0, cfg.offsets[0] + 9.0))
    with pytest.raises(ServoConfigError):
        check_offset_update(CFG, _set(CFG.offsets, 0, math.nan))


def test_rate_limiter_bound_and_convergence():
    rng = random.Random(7)
    rl = RateLimiter(400.0)
    rl.reset(CFG.stand)
    prev = rl.state
    for _ in range(5000):
        target = [rng.uniform(100, 260) for _ in range(8)]
        if rng.random() < 0.05:
            target[rng.randrange(8)] = math.nan
        dt = rng.choice([0.0, 0.001, 0.02, 0.05, 1.0, 10.0, -1.0, math.nan])
        cur = rl.step(target, dt)
        eff_dt = dt if math.isfinite(dt) and dt > 0 else 0.0
        bound = 400.0 * min(eff_dt, rl.max_dt) + 1e-9
        for a, b, t in zip(prev, cur, target):
            assert abs(b - a) <= bound
            if not math.isfinite(t):
                assert b == a                   # non-finite target: hold
        prev = cur
    for _ in range(200):
        rl.step([180.0] * 8, 0.02)
    assert rl.at_target([180.0] * 8)
    for bad in (0.0, -5.0, math.nan, math.inf, 1e6):
        with pytest.raises(ServoConfigError):
            RateLimiter(bad)


@pytest.fixture
def driver():
    rclpy = pytest.importorskip('rclpy')
    from rclpy.parameter import Parameter

    from robot_dog_hw.servo_driver_node import ServoDriverNode
    rclpy.init()
    nodes = []

    def make(**overrides):
        params = [Parameter('fake_hw', value=True)]
        params += [Parameter(k, value=v) for k, v in overrides.items()]
        nodes.append(ServoDriverNode(parameter_overrides=params))
        return nodes[-1]
    yield make
    for node in nodes:
        node.destroy_node()
    rclpy.try_shutdown()


@pytest.mark.parametrize('overrides', [dict(servo_min=[230.0] + list(CFG.mins[1:])),
                                       dict(servo_offset=[50.0] + list(CFG.offsets[1:])),
                                       dict(stand_angles=[100.0] * 8),
                                       dict(max_servo_speed_dps=0.0)])
def test_driver_refuses_to_start_on_bad_config(driver, overrides):
    with pytest.raises(ServoConfigError):
        driver(**overrides)


def test_driver_clamps_rate_limits_and_holds_on_timeout(driver):
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import JointState
    node = driver()
    lim = ServoLimiter(CFG)
    assert node._pca.frames[0] == {CFG.channels[i]: lim.angle_to_us(i, 172.0) for i in range(8)}
    msg = JointState(position=[1e9, -1e9, math.nan, math.inf, 300.0, 0.0, 175.0, 172.0])
    node._on_cmd(msg)
    assert max(abs(a - 172.0) for a in node._pose) <= 400.0 * 0.03 + 1e-9    # rate limited
    for _ in range(200):
        node._last_output -= 0.02
        node._on_cmd(msg)
    assert any(node._clamp_mask)
    for frame in node._pca.frames:
        for i in range(8):
            assert reference_us(i, CFG.mins[i]) <= frame[CFG.channels[i]] <= \
                reference_us(i, CFG.maxs[i])
    assert node._pose[0] == CFG.maxs[0] and node._pose[1] == CFG.mins[1]
    assert node._pose[2] == 172.0               # NaN entry held the previous target
    # watchdog: no /servo_cmd for cmd_timeout_s holds the pose
    n, pose = len(node._pca.frames), list(node._pose)
    node._last_cmd -= 1.0
    node._on_timer()
    assert node._watchdog_warned and node._pose == pose and len(node._pca.frames) == n
    node._on_cmd(msg)
    assert not node._watchdog_warned
    # offsets change in small steps only; limits are read-only
    big = [30.0] + list(CFG.offsets[1:])
    assert not node.set_parameters([Parameter('servo_offset', value=big)])[0].successful
    small = [16.0] + list(CFG.offsets[1:])
    assert node.set_parameters([Parameter('servo_offset', value=small)])[0].successful
    assert not node.set_parameters([Parameter('servo_max', value=[300.0] * 8)])[0].successful


def test_driver_relax_is_explicit_and_stand_rearms(driver):
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    node = driver()
    assert node._on_relax(Trigger.Request(), Trigger.Response()).success
    assert node._pca.all_off_count == 1 and not node._enabled
    n = len(node._pca.frames)
    node._on_cmd(JointState(position=[180.0] * 8))
    assert len(node._pca.frames) == n           # ignored while relaxed
    assert 'relaxed' in node.shutdown_to_stand()  # never re-energized implicitly
    node._pca.close = lambda: None
    assert node._on_stand(Trigger.Request(), Trigger.Response()).success and node._enabled
    node._on_cmd(JointState(position=[180.0] * 8))
    assert len(node._pca.frames) > n
    assert 'outputs left ON' in node.shutdown_to_stand() and node._pose == [172.0] * 8
