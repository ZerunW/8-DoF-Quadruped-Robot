"""Helpers shared by the tools: a node spinning in a background thread, service calls with
timeouts, remote parameter get/set and /servo_cmd publishing.

Every hardware action goes through servo_driver; nothing here opens the I2C bus.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter, parameter_value_to_python
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

from robot_dog_hw.servo_map import SERVO_LABELS

DRIVER = '/servo_driver'
CONTROLLER = '/gait_controller'


class ToolNode:
    """rclpy node spun by an executor in a background thread."""

    def __init__(self, name: str):
        rclpy.init()
        self.node = rclpy.create_node(name)
        self.log = self.node.get_logger()
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        self._servo_state: Optional[List[float]] = None
        self._driver_enabled: Optional[bool] = None
        self._cmd_pub = None

    # ----------------------------------------------------------------- calls
    def call(self, srv_type, name: str, request, timeout: float = 3.0):
        cli = self.node.create_client(srv_type, name)
        try:
            if not cli.wait_for_service(timeout_sec=timeout):
                return None
            fut = cli.call_async(request)
            deadline = time.monotonic() + timeout
            while not fut.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            return fut.result() if fut.done() else None
        finally:
            self.node.destroy_client(cli)

    def service_exists(self, name: str, timeout: float = 1.0) -> bool:
        cli = self.node.create_client(Trigger, name)
        try:
            return cli.wait_for_service(timeout_sec=timeout)
        finally:
            self.node.destroy_client(cli)

    def trigger(self, name: str, timeout: float = 3.0) -> Tuple[bool, str]:
        res = self.call(Trigger, name, Trigger.Request(), timeout)
        if res is None:
            return False, f'service {name} not available'
        return res.success, res.message

    def set_bool(self, name: str, value: bool, timeout: float = 3.0) -> Tuple[bool, str]:
        res = self.call(SetBool, name, SetBool.Request(data=value), timeout)
        if res is None:
            return False, f'service {name} not available'
        return res.success, res.message

    def get_params(self, target: str, names: Sequence[str],
                   timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        res = self.call(GetParameters, f'{target}/get_parameters',
                        GetParameters.Request(names=list(names)), timeout)
        if res is None:
            return None
        return {n: parameter_value_to_python(v) for n, v in zip(names, res.values)}

    def set_params(self, target: str, values: Dict[str, Any],
                   timeout: float = 3.0) -> List[Tuple[bool, str]]:
        req = SetParameters.Request(
            parameters=[Parameter(k, value=v).to_parameter_msg() for k, v in values.items()])
        res = self.call(SetParameters, f'{target}/set_parameters', req, timeout)
        if res is None:
            return [(False, f'{target}/set_parameters not available')]
        return [(r.successful, r.reason) for r in res.results]

    # ---------------------------------------------------------------- topics
    def watch_servo_state(self) -> None:
        def cb(msg: JointState):
            if not msg.name or list(msg.name) == list(SERVO_LABELS):
                self._servo_state = list(msg.position)
        self.node.create_subscription(JointState, '/servo_state', cb, 10)

    def watch_driver_enabled(self) -> None:
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)

        def cb(msg: Bool):
            self._driver_enabled = bool(msg.data)
        self.node.create_subscription(Bool, f'{DRIVER}/enabled', cb, qos)

    def wait_servo_state(self, timeout: float = 5.0) -> Optional[List[float]]:
        deadline = time.monotonic() + timeout
        while self._servo_state is None and time.monotonic() < deadline:
            time.sleep(0.02)
        return None if self._servo_state is None else list(self._servo_state)

    @property
    def servo_state(self) -> Optional[List[float]]:
        return None if self._servo_state is None else list(self._servo_state)

    @property
    def driver_enabled(self) -> Optional[bool]:
        return self._driver_enabled

    def other_cmd_publishers(self, settle: float = 1.0) -> int:
        """Number of /servo_cmd publishers other than this tool, after ``settle`` seconds."""
        time.sleep(settle)
        n = self.node.count_publishers('/servo_cmd')
        return n - (1 if self._cmd_pub is not None else 0)

    def publish_cmd(self, pose: Sequence[float]) -> None:
        if self._cmd_pub is None:
            self._cmd_pub = self.node.create_publisher(JointState, '/servo_cmd', 10)
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(SERVO_LABELS)
        msg.position = [float(a) for a in pose]
        self._cmd_pub.publish(msg)

    def hold(self, pose: Sequence[float], seconds: float, rate_hz: float = 20.0) -> None:
        """Re-publish ``pose`` at ``rate_hz`` for ``seconds``."""
        end = time.monotonic() + seconds
        while True:
            self.publish_cmd(pose)
            if time.monotonic() >= end:
                break
            time.sleep(1.0 / rate_hz)

    def shutdown(self) -> None:
        try:
            self._executor.shutdown(timeout_sec=1.0)
            self.node.destroy_node()
        finally:
            rclpy.try_shutdown()


def require_no_controller(tool: ToolNode, what: str) -> bool:
    """False, with a message, if another node such as gait_controller publishes /servo_cmd."""
    n = tool.other_cmd_publishers()
    if n > 0:
        print(f'\n{what}: {n} other node(s) publish /servo_cmd (gait_controller?).\n'
              'Two publishers would fight over the servos. Stop them and run\n'
              '    pixi run launch_driver\nthen start this tool again.')
        return False
    return True
