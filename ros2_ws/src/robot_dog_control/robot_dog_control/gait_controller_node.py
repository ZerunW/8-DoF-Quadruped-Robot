"""gait_controller: ROS wrapper around controller.GaitControllerCore (50 Hz).

Subscribes
    /cmd_vel       Twist: linear.x m/s (+fwd), angular.z rad/s (+left); expires
    /cmd_step      Bool: true keeps stepping in place; expires like /cmd_vel
    /cmd_pose      Twist: linear.z height m (<= 0: stand height), angular.y pitch (+ nose
                   down) and angular.x roll (+ right down) in rad, linear.x shift m
    /cmd_heading   Float32: heading to hold (rad from ~/zero_yaw; NaN: the start heading)
    <imu_topic>, /servo_state, /servo_driver/enabled
Publishes /servo_cmd (8 servo angles, deg) and /gait/debug (controller.DEBUG_FIELDS).
Services ~/stand ~/rest ~/estop ~/zero_yaw (Trigger) and ~/level_enable (SetBool).
Parameters: robot_dog.yaml (gait_controller). Wiring, geometry, joint region and servo
limits are read-only; the rest is live and re-validated, gait and pose limits only while
the gait is idle.
"""

from __future__ import annotations

import math
import sys
import time

from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32, Float64MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool, Trigger

from robot_dog_control.controller import (DEBUG_FIELDS, DEBUG_LAYOUT_LABEL, READ_ONLY_PARAMS,
                                          ControllerParams, GaitControllerCore, ImuSample)
from robot_dog_hw.node_runner import run_node
from robot_dog_hw.orientation import quat_to_rpy
from robot_dog_hw.ros_params import declare
from robot_dog_hw.servo_map import SERVO_LABELS


class _RosLog:
    def __init__(self, logger):
        self._l = logger

    def info(self, msg):
        self._l.info(msg)

    def warn(self, msg):
        self._l.warn(msg)

    def error(self, msg):
        self._l.error(msg)


class GaitControllerNode(Node):

    def __init__(self, **node_kwargs):
        super().__init__('gait_controller', **node_kwargs)
        defaults = ControllerParams().to_flat()
        for name, value in self._preset_overrides().items():
            defaults.setdefault(name, value)      # presets defined only in the config
        for name, value in defaults.items():
            declare(self, name, value, read_only=name in READ_ONLY_PARAMS)
        self._warn_unknown_overrides(set(defaults))
        flat = {name: self.get_parameter(name).value for name in defaults}
        params = ControllerParams.from_flat(flat)          # ValueError -> refuse to start
        self.core = GaitControllerCore(params, log=_RosLog(self.get_logger()))
        p = self.core.p

        self._pub_cmd = self.create_publisher(JointState, '/servo_cmd', 10)
        self._pub_dbg = self.create_publisher(Float64MultiArray, '/gait/debug', 10)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        self.create_subscription(Bool, '/cmd_step', self._on_cmd_step, 10)
        self.create_subscription(Twist, '/cmd_pose', self._on_cmd_pose, 10)
        self.create_subscription(Imu, p.imu_topic, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Float32, '/cmd_heading', self._on_cmd_heading, 10)
        self.create_subscription(JointState, '/servo_state', self._on_servo_state,
                                 qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Bool, '/servo_driver/enabled', self._on_driver_enabled, latched)
        self.create_service(Trigger, '~/stand', self._srv(lambda: self.core.request_stand(
            time.monotonic())))
        self.create_service(Trigger, '~/rest', self._srv(lambda: self.core.request_rest(
            time.monotonic())))
        self.create_service(Trigger, '~/estop', self._srv(self.core.request_estop))
        self.create_service(Trigger, '~/zero_yaw', self._srv(self.core.zero_yaw))
        self.create_service(SetBool, '~/level_enable', self._on_level_enable)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.create_timer(1.0 / p.rate_hz, self._on_timer)
        self._waiting_since = time.monotonic()
        self._rearm_stamp = None
        self._layout = [MultiArrayDimension(label=DEBUG_LAYOUT_LABEL, size=len(DEBUG_FIELDS),
                                            stride=len(DEBUG_FIELDS))]
        self.get_logger().info(
            f'gait_controller up: gait schedule '
            f'{list(zip(p.gait_schedule, p.gait_schedule_speeds_mps))} (m/s), fitting the '
            f'envelope: {list(self.core.walkable)}, {p.rate_hz:.0f} Hz, walking '
            f'{"ENABLED" if self.core.walk_allowed else "DISABLED"}; waiting for /servo_state')

    def _preset_overrides(self):
        """{name: value} of every gait_presets.<name>.<key> given in the parameter files."""
        try:
            overrides = self._parameter_overrides
        except AttributeError:
            return {}
        return {name: param.value for name, param in overrides.items()
                if name.startswith('gait_presets.')}

    def _warn_unknown_overrides(self, declared):
        try:
            unknown = sorted(set(self._parameter_overrides) - declared - {'use_sim_time'})
        except AttributeError:
            return
        if unknown:
            self.get_logger().warn(f'ignoring unknown parameters (typo?): {unknown}')

    def _srv(self, fn):
        def handler(request, response):
            ok, msg = fn()
            response.success, response.message = bool(ok), msg
            (self.get_logger().info if ok else self.get_logger().warn)(msg)
            return response
        return handler

    def _on_level_enable(self, request, response):
        result = self.set_parameters([Parameter('level_enable', value=bool(request.data))])[0]
        response.success = result.successful
        response.message = result.reason or f'level_enable={bool(request.data)}'
        if result.successful and request.data and not self.core.level_allowed:
            response.success = False
            response.message = 'leveling blocked by envelope: ' + self.core.report.summary()
        return response

    def _on_set_parameters(self, params):
        flat = {p.name: p.value for p in params}
        try:
            new = ControllerParams.from_flat(flat, base=self.core.p)
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        ok, reason = self.core.update_params(new)
        if not ok:
            self.get_logger().warn(f'rejected parameter update: {reason}')
        return SetParametersResult(successful=ok, reason=reason)

    def _on_cmd_vel(self, msg: Twist):
        self.core.set_cmd_vel(msg.linear.x, msg.angular.z, time.monotonic())

    def _on_cmd_step(self, msg: Bool):
        self.core.set_cmd_step(bool(msg.data), time.monotonic())

    def _on_cmd_pose(self, msg: Twist):
        self.core.set_cmd_pose(msg.linear.z * 1000.0, msg.angular.y, msg.angular.x,
                               msg.linear.x * 1000.0, msg.linear.y * 1000.0, time.monotonic())

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        try:
            roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
        except ValueError:
            return
        w = msg.angular_velocity
        self.core.set_imu(ImuSample(roll, pitch, yaw, w.x, w.y, w.z,
                                    msg.linear_acceleration.y, time.monotonic()))

    def _on_cmd_heading(self, msg: Float32):
        self.core.set_heading_target(float(msg.data))   # rad relative to zero; NaN clears

    def _on_servo_state(self, msg: JointState):
        if self._rearm_stamp is not None:
            # After a re-arm, drop samples stamped before it.
            if Time.from_msg(msg.header.stamp) < self._rearm_stamp:
                return
            self._rearm_stamp = None
        if msg.name and list(msg.name) != list(SERVO_LABELS):
            index = {n: i for i, n in enumerate(msg.name)}
            try:
                pos = [msg.position[index[n]] for n in SERVO_LABELS]
            except (KeyError, IndexError):
                return
        else:
            pos = list(msg.position)
        self.core.set_measured_pose(pos)

    def _on_driver_enabled(self, msg: Bool):
        if msg.data and not self.core._driver_enabled:
            self._rearm_stamp = self.get_clock().now()
        self.core.set_driver_enabled(bool(msg.data))

    def _on_timer(self):
        now = time.monotonic()
        out = self.core.step(now)
        if out is None:
            if now - self._waiting_since > 2.0:
                self._waiting_since = now
                self.get_logger().warn('waiting for /servo_state (is servo_driver running?) '
                                       'or the driver is relaxed')
            return
        self._waiting_since = now
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = list(SERVO_LABELS)
        cmd.position = [float(a) for a in out.servo]
        self._pub_cmd.publish(cmd)
        dbg = Float64MultiArray()
        dbg.layout.dim = self._layout
        dbg.data = [float(v) if math.isfinite(v) else float('nan') for v in out.debug]
        self._pub_dbg.publish(dbg)


def main(args=None):
    return run_node(GaitControllerNode, 'gait_controller', fatal=(ValueError,), args=args)


if __name__ == '__main__':
    sys.exit(main())
