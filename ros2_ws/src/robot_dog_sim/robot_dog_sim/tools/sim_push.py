"""Push the simulated dog: a world-frame force / torque on the body for a while.

    ros2 run robot_dog_sim sim_push --fy 3 --duration 0.2     # 3 N to the left, 0.2 s
    ros2 run robot_dog_sim sim_push --fx -5 --duration 0.1    # 5 N backward
    ros2 run robot_dog_sim sim_push --tz 0.3 --duration 0.5   # yaw torque 0.3 N m

Sets the sim's push_force_n, push_torque_nm and push_duration_s parameters and calls
/sim/push. World frame = odom: x = the dog's heading at reset, y = its left, z up. The
force acts on the body's center of mass.
"""

from __future__ import annotations

import argparse
import sys

from rclpy.utilities import remove_ros_args

from robot_dog_sim.tools._client import SIM, Client


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    for k in ('fx', 'fy', 'fz'):
        ap.add_argument(f'--{k}', type=float, default=0.0, help='force (N), world frame')
    for k in ('tx', 'ty', 'tz'):
        ap.add_argument(f'--{k}', type=float, default=0.0, help='torque (N m), world frame')
    ap.add_argument('--duration', type=float, default=0.2, help='seconds')
    args = ap.parse_args(remove_ros_args(sys.argv)[1:] if argv is None else argv)
    if args.duration <= 0 or args.duration > 5.0:
        print('--duration must be in (0, 5] s', file=sys.stderr)
        return 2
    c = Client('sim_push')
    try:
        ok, msg = c.set_params(SIM, {'push_force_n': [args.fx, args.fy, args.fz],
                                     'push_torque_nm': [args.tx, args.ty, args.tz],
                                     'push_duration_s': float(args.duration)})
        if not ok:
            print(f'could not set the push parameters: {msg}', file=sys.stderr)
            return 1
        ok, msg = c.trigger(f'{SIM}/push')
        print(msg)
        return 0 if ok else 1
    finally:
        c.close()


if __name__ == '__main__':
    sys.exit(main())
