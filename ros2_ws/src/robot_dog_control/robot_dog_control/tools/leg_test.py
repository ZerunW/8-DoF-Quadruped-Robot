"""leg_test: move one foot through IK and LegMap to verify mirror_servo and x_sign.

    pixi run launch_driver                  # first; the gait controller must not run
    pixi run leg_test -- --leg rf --sweep-z -20 20

Increasing fz must lift the foot; increasing fx must move it toward the nose."""

import argparse
import sys
import time

from robot_dog_control.config import controller_params_from_config, load_robot_config
from robot_dog_control.controller import build_components
from robot_dog_control.leg_map import LEG_NAMES
from robot_dog_hw.tools.common import ToolNode, require_no_controller


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--leg', required=True, choices=LEG_NAMES)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--sweep-x', nargs=2, type=float, metavar=('FROM', 'TO'),
                   help='fx offsets from the stand point (mm)')
    g.add_argument('--sweep-z', nargs=2, type=float, metavar=('FROM', 'TO'),
                   help='fz offsets from the stand point (mm)')
    ap.add_argument('--speed', type=float, default=10.0, help='foot speed (mm/s)')
    ap.add_argument('--cycles', type=int, default=2)
    ap.add_argument('--blend', type=float, default=2.0, help='stand-up blend time (s)')
    args, _ = ap.parse_known_args(argv)
    if args.sweep_x is None and args.sweep_z is None:
        args.sweep_z = [-15.0, 15.0]

    comp = build_components(controller_params_from_config(load_robot_config()))
    p, kin, lm = comp.params, comp.kin, comp.leg_map
    li = LEG_NAMES.index(args.leg)
    leg = lm.legs[li]
    rate = 50.0

    def solve_all(offsets):
        motor, info = [], None
        for i in range(4):
            dfx, dfz = offsets if i == li else (0.0, 0.0)
            fx = p.stand_fx_mm + dfx
            fz = -p.stand_height_mm + p.leg_trim_mm[i] + dfz
            r = kin.ik(*lm.body_to_canonical(i, fx, fz), guesses[i])
            if not r.ok:
                raise RuntimeError(f'leg {LEG_NAMES[i]}: fx={fx:.1f} fz={fz:.1f} unreachable')
            guesses[i] = (r.m1, r.m2)
            motor.append((r.m1, r.m2))
            if i == li:
                info = (fx, fz, r.m1, r.m2)
        return lm.servo_vector(motor), info

    guesses = [None] * 4
    tool = ToolNode('leg_test_tool')
    try:
        if not require_no_controller(tool, 'leg_test'):
            return 1
        tool.watch_servo_state()
        start = tool.wait_servo_state(5.0)
        if start is None:
            print('no /servo_state: start `pixi run launch_driver` first')
            return 1
        stand, _ = solve_all((0.0, 0.0))
        print(__doc__)
        print(f'leg {leg.name}: thigh servo {leg.thigh_idx}, calf servo {leg.calf_idx}, '
              f'mirror_servo={leg.mirror_servo}, x_sign={leg.x_sign:+d}')
        print(f'blending to the IK stand pose over {args.blend:.1f} s ...')
        n = max(1, int(args.blend * rate))
        for k in range(n + 1):
            a = k / n
            a = a * a * (3 - 2 * a)
            tool.publish_cmd([s0 + (s1 - s0) * a for s0, s1 in zip(start, stand)])
            time.sleep(1.0 / rate)
        a0, a1 = args.sweep_x or args.sweep_z
        axis = 'fx' if args.sweep_x else 'fz'
        seg = [0.0, a0, a1, 0.0] if args.cycles > 0 else [0.0]
        path = seg + [a0, a1] * max(0, args.cycles - 1) + [0.0]
        print(f'sweeping {axis} of {leg.name}: {path} mm at {args.speed:.0f} mm/s')
        last_print = 0.0
        for frm, to in zip(path[:-1], path[1:]):
            steps = max(1, int(abs(to - frm) / args.speed * rate))
            for k in range(1, steps + 1):
                d = frm + (to - frm) * k / steps
                off = (d, 0.0) if axis == 'fx' else (0.0, d)
                servo, (fx, fz, m1, m2) = solve_all(off)
                tool.publish_cmd(servo)
                now = time.monotonic()
                if now - last_print > 0.25:
                    last_print = now
                    print(f'  {axis} offset {d:+6.1f} -> fx {fx:+6.1f} fz {fz:+7.1f} mm | '
                          f'm1 {m1:6.2f} m2 {m2:6.2f} | thigh s{leg.thigh_idx}='
                          f'{servo[leg.thigh_idx]:6.1f} calf s{leg.calf_idx}='
                          f'{servo[leg.calf_idx]:6.1f}')
                time.sleep(1.0 / rate)
        tool.hold(stand, 0.5)
        print('done: back at the IK stand pose (driver holds it).')
        return 0
    finally:
        tool.shutdown()


if __name__ == '__main__':
    sys.exit(main())
