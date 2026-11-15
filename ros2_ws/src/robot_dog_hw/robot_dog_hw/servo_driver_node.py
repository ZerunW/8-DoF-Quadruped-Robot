"""servo_driver: the only process that writes to the PCA9685.

Each /servo_cmd is clamped to the servo limits, rate limited to max_servo_speed_dps, then
clamped again, offset and converted to pulses by servo_map.ServoLimiter, and written as one
64-byte block.

Topics
    sub  /servo_cmd              JointState  servo deg: 8 positions in index order without
                                             names, or any subset addressed by label
    pub  /servo_state            JointState  pose actually written (servo deg, before offset)
    pub  /servo_driver/clamped   Bool        the latest command needed clamping
    pub  /servo_driver/enabled   Bool        latched; false after ~/relax
Services (std_srvs/Trigger)
    /servo_driver/stand          move to stand_angles; re-arms the outputs after a relax
    /servo_driver/relax          all outputs off and the dog collapses; /servo_cmd is ignored
                                 until ~/stand

At startup the driver writes stand_angles. If /servo_cmd stops for cmd_timeout_s,
it warns once and holds the last pose. On shutdown it moves to stand_angles at the rate limit
and leaves the outputs on, or off if they were relaxed. servo_offset (at most 10 deg per
update), max_servo_speed_dps and cmd_timeout_s can be changed at runtime.
"""

from __future__ import annotations

import math
import sys
import time
from typing import List, Optional

from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from robot_dog_hw.node_runner import run_node
from robot_dog_hw.pca9685 import PCA9685, FakePCA9685
from robot_dog_hw.ros_params import (as_bool, as_float, as_float_list, as_int, as_int_list,
                                     declare)
from robot_dog_hw.servo_map import (NUM_SERVOS, RateLimiter, ServoConfig,
                                    ServoConfigError, ServoLimiter, check_offset_update)


class ServoDriverNode(Node):

    def __init__(self, **node_kwargs):
        super().__init__('servo_driver', **node_kwargs)
        self._declare_parameters()
        g = self.get_parameter
        try:
            self._cfg = ServoConfig(
                channels=as_int_list(g('servo_channel').value, 'servo_channel', NUM_SERVOS),
                offsets=as_float_list(g('servo_offset').value, 'servo_offset', NUM_SERVOS),
                mins=as_float_list(g('servo_min').value, 'servo_min', NUM_SERVOS),
                maxs=as_float_list(g('servo_max').value, 'servo_max', NUM_SERVOS),
                stand=as_float_list(g('stand_angles').value, 'stand_angles', NUM_SERVOS),
                pulse_min_us=as_float(g('pulse_min_us').value, 'pulse_min_us'),
                pulse_max_us=as_float(g('pulse_max_us').value, 'pulse_max_us'),
                servo_range_deg=as_float(g('servo_range_deg').value, 'servo_range_deg'),
            )
            pwm_hz = as_float(g('pwm_hz').value, 'pwm_hz')
            if not 40.0 <= pwm_hz <= 100.0:
                raise ServoConfigError(f'pwm_hz={pwm_hz} outside 40..100 Hz')
            self._rate = RateLimiter(as_float(g('max_servo_speed_dps').value,
                                              'max_servo_speed_dps'))
            self._cmd_timeout = self._check_timeout(g('cmd_timeout_s').value)
            fake = as_bool(g('fake_hw').value, 'fake_hw')
            bus = as_int(g('i2c_bus').value, 'i2c_bus')
            address = as_int(g('pca_address').value, 'pca_address')
            osc = as_float(g('oscillator_hz').value, 'oscillator_hz')
        except (ValueError, ServoConfigError) as exc:
            # Any invalid parameter is fatal: the node does not start.
            raise ServoConfigError(str(exc)) from None

        self._limiter = ServoLimiter(self._cfg)
        self._period = 1.0 / pwm_hz
        self._name_to_idx = {name: i for i, name in enumerate(self._cfg.labels)}

        pca_cls = FakePCA9685 if fake else PCA9685
        self._pca = pca_cls(bus=bus, address=address, oscillator_hz=osc, pwm_hz=pwm_hz,
                            min_us=self._cfg.pulse_min_us, max_us=self._cfg.pulse_max_us)
        self._fake = fake

        # -- state ---------------------------------------------------------------
        stand = [float(a) for a in self._cfg.stand]
        self._target: List[float] = list(stand)
        self._pose: List[float] = list(stand)
        self._clamp_mask = [False] * NUM_SERVOS
        self._enabled = True
        self._got_cmd = False
        self._watchdog_warned = False
        self._last_cmd = time.monotonic()
        self._last_output: Optional[float] = None
        self._write_failures = 0

        # Power-on: start the chip and immediately command the stand pose.
        self._pca.begin()
        self._rate.reset(stand)
        if not self._write_pose(stand, time.monotonic()):
            raise RuntimeError('could not write the stand pose to the PCA9685')
        self.get_logger().info(
            f'PCA9685 {"FAKE" if fake else f"bus {bus} addr {address:#04x}"} '
            f'prescale {self._pca.prescale} @ {pwm_hz:.0f} Hz; stand pose written: {stand}')

        # -- ROS interfaces -----------------------------------------------------
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._pub_state = self.create_publisher(JointState, '/servo_state', 10)
        self._pub_clamped = self.create_publisher(Bool, '~/clamped', 10)
        self._pub_enabled = self.create_publisher(Bool, '~/enabled', latched)
        self.create_subscription(JointState, '/servo_cmd', self._on_cmd,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.create_service(Trigger, '~/stand', self._on_stand)
        self.create_service(Trigger, '~/relax', self._on_relax)
        self.create_timer(self._period, self._on_timer)
        if not fake:
            self.create_timer(1.0, self._on_health_check)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._publish_enabled()

    # ------------------------------------------------------------------ params
    def _declare_parameters(self):
        d = declare
        d(self, 'i2c_bus', 1, 'Linux I2C bus number', read_only=True)
        d(self, 'pca_address', 0x40, 'PCA9685 7-bit address', read_only=True)
        d(self, 'oscillator_hz', 27000000, 'PCA9685 oscillator (Adafruit default)', read_only=True)
        d(self, 'pwm_hz', 50, 'servo frame rate', read_only=True)
        d(self, 'pulse_min_us', 500.0, 'pulse at servo angle 0', read_only=True)
        d(self, 'pulse_max_us', 2500.0, 'pulse at servo_range_deg', read_only=True)
        d(self, 'servo_range_deg', 360.0, 'servo travel for pulse_min..pulse_max', read_only=True)
        d(self, 'fake_hw', False, 'use FakePCA9685 (no I2C)', read_only=True)
        d(self, 'cmd_timeout_s', 0.5, 'watchdog: warn when /servo_cmd stops (pose is held)')
        d(self, 'max_servo_speed_dps', 400.0, 'rate limit per servo (deg/s)')
        d(self, 'servo_channel', list(ServoConfig().channels), 'PCA9685 channel per index',
          read_only=True)
        d(self, 'servo_offset', list(ServoConfig().offsets),
          'calibration offset per index (deg, added after the clamp)')
        d(self, 'servo_min', list(ServoConfig().mins), 'lower limit per index', read_only=True)
        d(self, 'servo_max', list(ServoConfig().maxs), 'upper limit per index', read_only=True)
        d(self, 'stand_angles', list(ServoConfig().stand), 'power-on / stand pose',
          read_only=True)

    @staticmethod
    def _check_timeout(value) -> float:
        t = as_float(value, 'cmd_timeout_s')
        if not 0.05 <= t <= 10.0:
            raise ValueError(f'cmd_timeout_s={t} outside 0.05..10 s')
        return t

    def _on_set_parameters(self, params):
        new_cfg = None
        new_speed = None
        new_timeout = None
        try:
            for p in params:
                if p.name == 'servo_offset':
                    offsets = as_float_list(p.value, 'servo_offset', NUM_SERVOS)
                    new_cfg = check_offset_update(self._cfg, offsets)
                elif p.name == 'max_servo_speed_dps':
                    new_speed = as_float(p.value, 'max_servo_speed_dps')
                    RateLimiter(new_speed)  # validates range
                elif p.name == 'cmd_timeout_s':
                    new_timeout = self._check_timeout(p.value)
        except (ValueError, ServoConfigError) as exc:
            self.get_logger().warn(f'rejected parameter update: {exc}')
            return SetParametersResult(successful=False, reason=str(exc))
        # All valid: apply atomically.
        if new_cfg is not None:
            self._cfg = new_cfg
            self._limiter = ServoLimiter(new_cfg)
            self.get_logger().info(f'servo_offset updated: {list(new_cfg.offsets)}')
            if self._enabled:
                self._write_pose(self._pose, time.monotonic())
        if new_speed is not None:
            self._rate.set_speed(new_speed)
        if new_timeout is not None:
            self._cmd_timeout = new_timeout
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ output
    def _write_pose(self, pose: List[float], now: float) -> bool:
        """Clamp, convert to pulses and write one block; False if the I2C write failed."""
        frame = self._limiter.process(pose)
        try:
            self._pca.write_us(frame.by_channel)
        except (OSError, ValueError) as exc:
            self._write_failures += 1
            self.get_logger().error(f'PCA9685 write failed ({self._write_failures}x): {exc}',
                                    throttle_duration_sec=1.0)
            self._rate.reset(self._pose)  # limiter restarts from the last written pose
            return False
        self._write_failures = 0
        self._pose = list(frame.angles)
        self._last_output = now
        return True

    def _service_output(self, now: float) -> None:
        dt = self._period if self._last_output is None else now - self._last_output
        dt = min(max(dt, 0.0), 1.5 * self._period)
        self._write_pose(self._rate.step(self._target, dt), now)

    # --------------------------------------------------------------- callbacks
    def _on_cmd(self, msg: JointState) -> None:
        log = self.get_logger()
        if not self._enabled:
            log.warn('outputs relaxed: ignoring /servo_cmd (call /servo_driver/stand to re-arm)',
                     throttle_duration_sec=2.0)
            return
        target = list(self._target)
        if msg.name:
            if len(msg.name) != len(msg.position):
                log.warn('/servo_cmd: name and position lengths differ, ignored',
                         throttle_duration_sec=2.0)
                return
            for name, pos in zip(msg.name, msg.position):
                idx = self._name_to_idx.get(name)
                if idx is None:
                    log.warn(f'/servo_cmd: unknown servo name {name!r}, message ignored '
                             f'(known: {list(self._cfg.labels)})', throttle_duration_sec=2.0)
                    return
                target[idx] = pos
        else:
            if len(msg.position) != NUM_SERVOS:
                log.warn(f'/servo_cmd: expected {NUM_SERVOS} positions, got '
                         f'{len(msg.position)}; ignored', throttle_duration_sec=2.0)
                return
            target = list(msg.position)
        for i, a in enumerate(target):
            if not math.isfinite(a):
                log.warn(f'/servo_cmd: servo {i} non-finite ({a}); holding previous target',
                         throttle_duration_sec=2.0)
                target[i] = self._target[i]
        clamped, mask = self._limiter.clamp(target)
        self._clamp_mask = mask
        if any(mask):
            detail = ', '.join(f'{self._cfg.labels[i]}={target[i]:.1f}->{clamped[i]:.1f}'
                               for i in range(NUM_SERVOS) if mask[i])
            log.warn(f'CLAMPED servo command: {detail}', throttle_duration_sec=1.0)
        self._target = clamped
        now = time.monotonic()
        self._last_cmd = now
        self._got_cmd = True
        if self._watchdog_warned:
            log.info('/servo_cmd resumed')
        self._watchdog_warned = False
        self._service_output(now)

    def _on_timer(self) -> None:
        now = time.monotonic()
        if self._enabled:
            if (self._got_cmd and not self._watchdog_warned
                    and now - self._last_cmd > self._cmd_timeout):
                self._watchdog_warned = True
                self.get_logger().warn(
                    f'no /servo_cmd for {self._cmd_timeout:.2f} s: holding last pose')
            converging = not self._rate.at_target(self._target, 1e-6)
            if converging and (self._last_output is None
                               or now - self._last_output >= 0.9 * self._period):
                self._service_output(now)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._cfg.labels)
        msg.position = [float(a) for a in self._pose]
        self._pub_state.publish(msg)
        self._pub_clamped.publish(Bool(data=bool(any(self._clamp_mask))))

    def _on_health_check(self) -> None:
        try:
            alive = self._pca.check_alive()
        except OSError as exc:
            self.get_logger().error(f'PCA9685 health check failed: {exc}',
                                    throttle_duration_sec=5.0)
            return
        if alive:
            return
        self.get_logger().error('PCA9685 lost its configuration (brown-out reset?): '
                                're-initialising and re-writing the current pose')
        try:
            self._pca.begin()
            if self._enabled:
                self._rate.reset(self._pose)
                self._write_pose(self._pose, time.monotonic())
            else:
                self._pca.all_off()
        except OSError as exc:
            self.get_logger().error(f'PCA9685 re-init failed: {exc}')

    def _on_stand(self, request, response):
        stand = [float(a) for a in self._cfg.stand]
        if not self._enabled:
            # Outputs were off: write the stand pose directly, without rate limiting.
            self.get_logger().warn('re-arming outputs at the stand pose (servos will jump)')
            self._rate.reset(stand)
            self._target = list(stand)
            self._clamp_mask = [False] * NUM_SERVOS
            if not self._write_pose(stand, time.monotonic()):
                response.success = False
                response.message = 'I2C write failed'
                return response
            self._enabled = True
            self._got_cmd = False
            self._publish_enabled()
        else:
            self._target = list(stand)
            self._clamp_mask = [False] * NUM_SERVOS
            self._service_output(time.monotonic())
        response.success = True
        response.message = 'moving to stand_angles (rate limited); a running controller ' \
                           'will take over again with its next /servo_cmd'
        return response

    def _on_relax(self, request, response):
        try:
            self._pca.all_off()
        except OSError as exc:
            response.success = False
            response.message = f'I2C error: {exc}'
            return response
        self._enabled = False
        self._publish_enabled()
        self.get_logger().warn('RELAX: all PWM outputs OFF - the dog collapses. /servo_cmd is '
                               'ignored until /servo_driver/stand is called.')
        response.success = True
        response.message = 'outputs off (dog is limp); call /servo_driver/stand to re-arm'
        return response

    def _publish_enabled(self) -> None:
        self._pub_enabled.publish(Bool(data=self._enabled))

    # ---------------------------------------------------------------- shutdown
    def shutdown_to_stand(self, timeout_s: float = 3.0) -> str:
        """Move to stand_angles at the rate limit, leaving the outputs on; returns a log line."""
        if not self._enabled:
            self._pca.close()
            return 'shutdown: outputs are relaxed, leaving them off'
        stand = [float(a) for a in self._cfg.stand]
        self._rate.reset(self._pose)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pose = self._rate.step(stand, self._period)
            self._write_pose(pose, time.monotonic())
            if self._rate.at_target(stand, 1e-3):
                break
            time.sleep(self._period)
        self._pca.close()
        return f'shutdown: stand pose written {self._pose}, outputs left ON'


def main(args=None):
    return run_node(ServoDriverNode, 'servo_driver',
                    on_exit=lambda node: node.shutdown_to_stand(),
                    fatal=(ServoConfigError, ValueError), args=args)


if __name__ == '__main__':
    sys.exit(main())
