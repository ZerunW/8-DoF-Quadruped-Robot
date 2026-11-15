"""Put the simulated dog back at the origin, standing in the current servo pose.

    ros2 run robot_dog_sim sim_reset          # = pixi run sim_reset
"""

from __future__ import annotations

import sys

from robot_dog_sim.tools._client import SIM, Client


def main(argv=None) -> int:
    c = Client('sim_reset')
    try:
        ok, msg = c.trigger(f'{SIM}/reset')
        print(msg)
        return 0 if ok else 1
    finally:
        c.close()


if __name__ == '__main__':
    sys.exit(main())
