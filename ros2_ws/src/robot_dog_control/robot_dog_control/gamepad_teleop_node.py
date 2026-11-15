"""gamepad_teleop: PS5 DualSense (joy_node) to the controller's command topics.

Streams /cmd_vel and /cmd_step at 20 Hz while L1 is held and /cmd_pose (body height) at
pose_rate_hz while R1 or R2 is held; calls gait_controller ~/estop (Cross) and ~/stand
(Options). The mapping lives in gamepad_mapping.py; speed, turn and height limits come
from the running gait_controller, and Joy is ignored until they are read.
"""

from __future__ import annotations

import sys
import time

from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from robot_dog_control.gamepad_mapping import Commands, GamepadMapper, MapperParams
from robot_dog_hw.node_runner import run_node
from robot_dog_hw.ros_params import declare

CONTROLLER_NODE = 'gait_controller'
#: controller parameters the gamepad limits come from
CONTROLLER_PARAMS = ('max_speed_mps', 'max_turn_rps', 'leg_pos_y_mm', 'height_min_mm',
                     'height_max_mm', 'stand_height_mm')


def build_params(own: dict, ctrl: dict) -> MapperParams:
    """MapperParams from this node's yaml values and the controller's limits."""
    values = dict(own)
    # speed / turn: the yaml value, 0 = the controller's maximum, never above it
    for mine, theirs in (('max_speed_mps', 'max_speed_mps'), ('max_turn_rps', 'max_turn_rps')):
        cap = float(ctrl[theirs])
        v = float(values[mine])
        values[mine] = cap if v <= 0.0 else min(v, cap)
    # half track: mean |hip y| of the controller's legs
    ys = [abs(float(y)) for y in ctrl['leg_pos_y_mm']]
    for name in ('height_min_mm', 'height_max_mm', 'stand_height_mm'):
        if name in ctrl:
            values[name] = float(ctrl[name])
    values['half_track_m'] = sum(ys) / len(ys) / 1000.0
    return MapperParams(**values)


class GamepadTeleopNode(Node):

    def __init__(self):
        super().__init__('gamepad_teleop')
        defaults = MapperParams()
        self._own = {}
        for name in MapperParams.names():
            if name not in ('half_track_m', 'height_min_mm', 'height_max_mm',
                            'stand_height_mm'):      # read from the controller
                self._own[name] = declare(self, name, getattr(defaults, name)).value
        self._rate = float(declare(self, 'rate_hz', 20.0).value)
        self._pose_rate = float(declare(self, 'pose_rate_hz', 50.0).value)
        self.mapper = None
        self._pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self._pub_step = self.create_publisher(Bool, '/cmd_step', 10)
        self._pub_pose = self.create_publisher(Twist, '/cmd_pose', 10)
        self._estop = self.create_client(Trigger, f'{CONTROLLER_NODE}/estop')
        self._stand = self.create_client(Trigger, f'{CONTROLLER_NODE}/stand')
        self._get = self.create_client(GetParameters, f'{CONTROLLER_NODE}/get_parameters')
        self.create_subscription(Joy, '/joy', self._on_joy, 10)
        self._pending = None
        self.create_timer(1.0, self._load_limits)
        self.create_timer(1.0 / self._rate, self._on_tick)
        self.create_timer(1.0 / self._pose_rate, self._on_pose_tick)
        self.get_logger().info('waiting for gait_controller parameters; Joy is ignored until then')

    def _load_limits(self):
        if self.mapper is not None or self._pending is not None:
            return
        if not self._get.service_is_ready():
            return
        req = GetParameters.Request(names=list(CONTROLLER_PARAMS))
        self._pending = self._get.call_async(req)
        self._pending.add_done_callback(self._limits_done)

    def _limits_done(self, fut):
        self._pending = None
        try:
            res = fut.result()
            ctrl = {}
            for name, val in zip(CONTROLLER_PARAMS, res.values):
                if val.type == ParameterType.PARAMETER_NOT_SET:
                    raise ValueError(f'controller parameter {name} is not set')
                ctrl[name] = parameter_value_to_python(val)
            self.mapper = GamepadMapper(build_params(self._own, ctrl))
        except Exception as exc:  # retry on the next timer tick
            self.get_logger().warn(f'could not read controller limits: {exc}')
            return
        p = self.mapper.p
        self.get_logger().info(
            f'ready: speed <= {p.max_speed_mps:.3f} m/s, turn <= {p.max_turn_rps:.2f} rad/s, '
            f'half track {p.half_track_m * 1000:.1f} mm. Hold L1 to step, sticks drive; R1/R2 body up/down '
            f'({p.height_min_mm:.0f}..{p.height_max_mm:.0f} mm).')

    def _on_joy(self, msg: Joy):
        if self.mapper is None:
            return
        self._emit(self.mapper.on_joy(list(msg.axes), list(msg.buttons), time.monotonic()))

    def _on_tick(self):
        if self.mapper is None:
            return
        self._emit(self.mapper.tick(time.monotonic()))

    def _on_pose_tick(self):
        if self.mapper is None:
            return
        self._emit(self.mapper.pose_tick(time.monotonic()))

    def _emit(self, cmd: Commands):
        if cmd.vel is not None:
            m = Twist()
            m.linear.x, m.angular.z = cmd.vel
            self._pub_vel.publish(m)
        if cmd.step is not None:
            self._pub_step.publish(Bool(data=cmd.step))
        if cmd.pose is not None:
            m = Twist()
            m.linear.z, m.angular.y, m.angular.x = cmd.pose
            self._pub_pose.publish(m)
        if cmd.estop:
            self._call(self._estop, 'ESTOP')
        if cmd.stand:
            self._call(self._stand, 'STAND')

    def _call(self, client, label: str):
        if not client.service_is_ready():
            self.get_logger().warn(f'{label}: gait_controller service not available')
            return
        fut = client.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: self.get_logger().info(
            f'{label}: {f.result().message}' if f.result() is not None else f'{label}: no reply'))


def main(args=None):
    return run_node(GamepadTeleopNode, 'gamepad_teleop', args=args)


if __name__ == '__main__':
    sys.exit(main())
