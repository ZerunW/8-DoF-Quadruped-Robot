"""stand: put the dog in its stand pose.

If servo_driver is relaxed or no controller runs, calls /servo_driver/stand, which writes
stand_angles and re-arms the outputs. If gait_controller runs, calls /gait_controller/stand,
which stops walking or leaves REST / ESTOP and blends to its IK stand pose.
"""

import sys
import time

from robot_dog_hw.tools.common import CONTROLLER, DRIVER, ToolNode


def main(argv=None):
    tool = ToolNode('stand_tool')
    try:
        tool.watch_driver_enabled()
        controller = tool.service_exists(f'{CONTROLLER}/stand', timeout=1.5)
        deadline = time.monotonic() + 1.0
        while tool.driver_enabled is None and time.monotonic() < deadline:
            time.sleep(0.02)
        rc = 0
        if tool.driver_enabled is False or not controller:
            ok, msg = tool.trigger(f'{DRIVER}/stand')
            print(f'servo_driver: {"OK" if ok else "FAILED"} - {msg}')
            rc |= 0 if ok else 1
        if controller:
            ok, msg = tool.trigger(f'{CONTROLLER}/stand')
            print(f'gait_controller: {"OK" if ok else "FAILED"} - {msg}')
            rc |= 0 if ok else 1
        return rc
    finally:
        tool.shutdown()


if __name__ == '__main__':
    sys.exit(main())
