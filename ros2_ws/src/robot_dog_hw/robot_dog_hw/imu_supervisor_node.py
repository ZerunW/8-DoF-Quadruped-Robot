"""imu_supervisor: restart the IMU driver when its topic goes silent.

After ``timeout_s`` without a message this node kills the driver with
``pkill -f process_pattern``; the launch file respawns it, which reinitializes the chip.
"""

from __future__ import annotations

import subprocess
import sys
import time

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from robot_dog_hw.node_runner import run_node
from robot_dog_hw.ros_params import as_float, declare


class ImuSupervisorNode(Node):

    def __init__(self, **kw):
        super().__init__('imu_supervisor', **kw)
        declare(self, 'topic', '/bno055/imu')
        declare(self, 'timeout_s', 2.0, 'silence on the topic before the driver is restarted')
        declare(self, 'process_pattern', 'lib/bno055/bno055', 'pkill -f pattern of the driver')
        declare(self, 'min_restart_interval_s', 10.0)
        g = self.get_parameter
        self._timeout = as_float(g('timeout_s').value, 'timeout_s')
        self._min_interval = as_float(g('min_restart_interval_s').value, 'min_restart_interval_s')
        self._pattern = str(g('process_pattern').value)
        self._last_msg = time.monotonic() + 10.0      # 10 s startup grace
        self._last_restart = 0.0
        self._restarts = 0
        self.create_subscription(Imu, str(g('topic').value), self._on_imu, qos_profile_sensor_data)
        self.create_timer(0.5, self._on_timer)
        self.get_logger().info(f'watching {g("topic").value}: restart the driver after '
                               f'{self._timeout:.1f} s of silence')

    def _on_imu(self, msg: Imu):
        if self._restarts and self._last_msg < time.monotonic() - self._timeout:
            self.get_logger().info('IMU data is back')
        self._last_msg = time.monotonic()

    def _on_timer(self):
        now = time.monotonic()
        if now - self._last_msg < self._timeout:
            return
        if now - self._last_restart < self._min_interval:
            return
        self._last_restart = now
        self._restarts += 1
        self.get_logger().warn(f'no IMU data for {now - self._last_msg:.1f} s: restarting the '
                               f'driver (restart #{self._restarts}); if it keeps failing at '
                               'connect, power-cycle the sensor rail')
        try:
            subprocess.run(['pkill', '-TERM', '-f', self._pattern], check=False, timeout=5)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'could not restart the IMU driver: {exc}')


def main(args=None):
    return run_node(ImuSupervisorNode, 'imu_supervisor', args=args)


if __name__ == '__main__':
    sys.exit(main())
