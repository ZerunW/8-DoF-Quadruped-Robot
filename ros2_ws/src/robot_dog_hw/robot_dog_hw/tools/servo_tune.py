"""servo_tune: interactive servo offset tuner over ROS parameters.

Needs ``pixi run launch_driver``. Commands (spaces are ignored):

    3+5        servo_offset[3] += 5
    7-0.25     servo_offset[7] -= 0.25
    show       print index / channel / label / offset
    stand      command the stand pose again (/servo_driver/stand)
    save       write servo_offset into robot_dog.yaml; the deploy terminal applies it to the
               laptop's copy
    q          quit; asks to save unsaved changes

Each change is sent as a servo_offset update to servo_driver and applies to the current pose.
"""

import sys

from robot_dog_hw.calibration import calibration_path, update_calibration
from robot_dog_hw.servo_map import NUM_SERVOS, SERVO_LABELS
from robot_dog_hw.tools.common import DRIVER, ToolNode


def parse_command(line: str):
    """'3+5' -> (3, 5.0), '7-0.25' -> (7, -0.25); None for anything else."""
    s = ''.join(line.split())
    for pos in range(1, len(s)):
        if s[pos] in '+-':
            try:
                idx = int(s[:pos])
                delta = float(s[pos + 1:])
            except ValueError:
                return None
            if pos == len(s) - 1:
                return None
            return idx, (delta if s[pos] == '+' else -delta)
    return None


def main(argv=None):
    tool = ToolNode('servo_tune_tool')
    try:
        p = tool.get_params(DRIVER, ['servo_offset', 'servo_channel'])
        if p is None:
            print('servo_driver not running: start `pixi run launch_driver` first')
            return 1
        offsets = list(p['servo_offset'])
        channels = list(p['servo_channel'])
        saved = list(offsets)
        print(__doc__)
        ok, msg = tool.trigger(f'{DRIVER}/stand')
        print(f'stand: {msg}')

        def show():
            for i in range(NUM_SERVOS):
                print(f'  [{i}] ch {channels[i]:2d} {SERVO_LABELS[i]:20s} offset {offsets[i]:+8.3f}')

        show()
        while True:
            try:
                line = input('tune> ').strip()
            except EOFError:
                line = 'q'
            if not line:
                continue
            if line in ('q', 'quit', 'exit'):
                if offsets != saved:
                    try:
                        ans = input('unsaved changes - save? [y/N] ').strip().lower()
                    except EOFError:
                        ans = 'n'
                    if ans == 'y':
                        path = update_calibration('servo_driver', {'servo_offset': offsets})
                        print(f'saved to {path}')
                return 0
            if line == 'show':
                show()
                continue
            if line == 'stand':
                print(tool.trigger(f'{DRIVER}/stand')[1])
                continue
            if line == 'save':
                path = update_calibration('servo_driver', {'servo_offset': offsets})
                saved = list(offsets)
                print(f'saved servo_offset to {path}')
                continue
            cmd = parse_command(line)
            if cmd is None:
                print('Invalid cmd. Use: <num>+<delta> or <num>-<delta> (e.g., 3+5, 7-0.25)')
                continue
            idx, delta = cmd
            if not 0 <= idx < NUM_SERVOS:
                print(f'servo index out of range: {idx}')
                continue
            new = list(offsets)
            new[idx] = round(new[idx] + delta, 6)
            (ok, reason), = tool.set_params(DRIVER, {'servo_offset': new})
            if ok:
                offsets = new
                print(f'servoIndex={idx} (pcaChannel={channels[idx]}) newOffset={offsets[idx]:.6f}')
            else:
                print(f'REJECTED by servo_driver: {reason}')
    finally:
        tool.shutdown()


if __name__ == '__main__':
    print(f'calibration file: {calibration_path()}')
    sys.exit(main())
