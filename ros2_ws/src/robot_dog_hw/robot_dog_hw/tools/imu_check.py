"""imu_check: print the BNO055 attitude, rates and calibration status.

Needs ``pixi run launch_bringup`` or the bno055 node alone. Sign conventions (REP-103,
placement_axis_remap P7 for the X-back / Y-left / Z-down mount):

    tilt nose down            -> pitch positive
    roll right side down      -> roll  positive
    turn left (CCW from top)  -> yaw   positive, gyro z positive while turning
"""

import json
import math
import sys
import time

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from robot_dog_hw.orientation import quat_to_rpy
from robot_dog_hw.tools.common import ToolNode


def main(argv=None):
    tool = ToolNode('imu_check_tool')
    state = {'imu': None, 'calib': None, 'n': 0}

    def on_imu(msg):
        state['imu'] = msg
        state['n'] += 1

    def on_calib(msg):
        try:
            state['calib'] = json.loads(msg.data)
        except ValueError:
            state['calib'] = msg.data

    tool.node.create_subscription(Imu, '/bno055/imu', on_imu, qos_profile_sensor_data)
    tool.node.create_subscription(String, '/bno055/calib_status', on_calib, 10)
    print(__doc__)
    try:
        last_n, last_t = 0, time.monotonic()
        while True:
            time.sleep(0.2)
            m = state['imu']
            if m is None:
                print('waiting for /bno055/imu ...')
                continue
            q = m.orientation
            r, p, y = (math.degrees(a) for a in quat_to_rpy(q.x, q.y, q.z, q.w))
            w = m.angular_velocity
            a = m.linear_acceleration
            now = time.monotonic()
            hz = (state['n'] - last_n) / (now - last_t)
            last_n, last_t = state['n'], now
            print(f'roll {r:+7.2f}  pitch {p:+7.2f}  yaw {y:+8.2f} deg | gyro '
                  f'{math.degrees(w.x):+7.1f} {math.degrees(w.y):+7.1f} '
                  f'{math.degrees(w.z):+7.1f} deg/s | lin acc {a.x:+5.2f} {a.y:+5.2f} '
                  f'{a.z:+5.2f} m/s2 | {hz:4.0f} Hz | calib {state["calib"]}')
    except KeyboardInterrupt:
        return 0
    finally:
        tool.shutdown()


if __name__ == '__main__':
    sys.exit(main())
