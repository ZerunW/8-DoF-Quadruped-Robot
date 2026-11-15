"""relax: switch all servo outputs off (/servo_driver/relax).

The dog collapses. servo_driver then ignores /servo_cmd until ``pixi run stand`` or
/servo_driver/stand re-arms it.
"""

import argparse
import sys

from robot_dog_hw.tools.common import DRIVER, ToolNode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-y', '--yes', action='store_true', help='do not ask for confirmation')
    args, _ = ap.parse_known_args(argv)
    print('WARNING: all 8 servos go limp and the dog collapses. Hold it or put it down.')
    if not args.yes:
        try:
            if input("Type 'yes' to relax: ").strip().lower() != 'yes':
                print('aborted')
                return 1
        except EOFError:
            print('aborted (no tty; use --yes)')
            return 1
    tool = ToolNode('relax_tool')
    try:
        ok, msg = tool.trigger(f'{DRIVER}/relax')
        print(('OK: ' if ok else 'FAILED: ') + msg)
        if ok:
            print('Re-arm with: pixi run stand')
        return 0 if ok else 1
    finally:
        tool.shutdown()


if __name__ == '__main__':
    sys.exit(main())
