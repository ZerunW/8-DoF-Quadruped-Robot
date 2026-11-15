"""Tiny rclpy client helper for the sim tools (service calls with a timeout)."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import rclpy
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

SIM = '/sim'


class Client:

    def __init__(self, name: str):
        rclpy.init()
        self.node = rclpy.create_node(name)

    def call(self, srv_type, service: str, request, timeout: float = 3.0):
        cli = self.node.create_client(srv_type, service)
        try:
            if not cli.wait_for_service(timeout_sec=timeout):
                return None
            fut = cli.call_async(request)
            rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
            return fut.result() if fut.done() else None
        finally:
            self.node.destroy_client(cli)

    def trigger(self, service: str) -> Tuple[bool, str]:
        res = self.call(Trigger, service, Trigger.Request())
        if res is None:
            return False, f'{service} not available (is `pixi run launch_sim` running?)'
        return bool(res.success), res.message

    def set_params(self, node: str, values: Dict[str, Any]) -> Tuple[bool, str]:
        req = SetParameters.Request(
            parameters=[Parameter(k, value=v).to_parameter_msg() for k, v in values.items()])
        res = self.call(SetParameters, f'{node}/set_parameters', req)
        if res is None:
            return False, f'{node}/set_parameters not available'
        bad = [r.reason for r in res.results if not r.successful]
        return (not bad), '; '.join(bad)

    def close(self) -> None:
        self.node.destroy_node()
        rclpy.try_shutdown()


def wait(seconds: float) -> None:
    time.sleep(seconds)
