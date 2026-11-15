"""walk_test: run one behavior of the API on the running stack and measure it (robot or sim).

    pixi run walk_test -- --speed 0.1 --distance 0.5 --heading 20  # 0.5 m at heading 20 deg
    pixi run walk_test -- --step --turn-to 90 --duration 6         # turn in place to 90 deg

Prints attitude and heading errors, corrections, the scheduled gait and IK clamps from /gait/debug, plus the
traveled distance from /sim/ground_truth in MuJoCo; ``--pin-gait`` restores the schedule afterwards
(exit code 2 if that fails)."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Tuple

from std_msgs.msg import Float64MultiArray

from robot_dog_control.api import RobotDog
from robot_dog_control.controller import DEBUG_FIELDS, DEBUG_INDEX, State
from robot_dog_hw.tools.common import CONTROLLER, ToolNode


def _rms(xs: List[float]) -> float:
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float('nan')


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--speed', type=float, default=0.03, help='m/s, + forward (walk)')
    ap.add_argument('--duration', type=float, default=5.0, help='s of walking / stepping')
    ap.add_argument('--distance', type=float, default=None,
                    help='m to walk instead of --duration (planned travel of the gait)')
    ap.add_argument('--heading', type=float, default=None,
                    help='heading to hold (deg, from where the dog faces at the start)')
    ap.add_argument('--step', action='store_true', help='step in place instead of walking')
    ap.add_argument('--turn-to', type=float, default=None, dest='turn_to',
                    help='with --step: heading to turn in place to (deg, from the start)')
    ap.add_argument('--pose', default='', help='height_mm,pitch_deg,roll_deg during the run')
    ap.add_argument('--pin-gait', default='', dest='pin_gait', metavar='PRESET',
                    help='tuning override: pin the speed schedule to this preset for the run '
                         '(not a behavior: restored afterwards)')
    ap.add_argument('--settle', type=float, default=2.0, help='s to log after the stop')
    ap.add_argument('--trace', default='', help='write every /gait/debug tick to this CSV')
    args, _ = ap.parse_known_args(argv)
    tool = ToolNode('walk_test_tool')
    samples: List[Tuple[float, List[float]]] = []
    gt: List[Tuple[float, float, float, float]] = []
    tool.node.create_subscription(Float64MultiArray, '/gait/debug',
                                  lambda m: samples.append((time.monotonic(), list(m.data))),
                                  10)
    try:
        from nav_msgs.msg import Odometry

        def on_gt(msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            yaw = math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y),
                                          1 - 2 * (q.y * q.y + q.z * q.z)))
            gt.append((time.monotonic(), p.x, p.y, yaw))
        tool.node.create_subscription(Odometry, '/sim/ground_truth', on_gt, 10)
    except ImportError:
        pass
    restore = None
    dog = None

    def measure() -> int:
        nonlocal restore, dog
        ok, msg = tool.trigger(f'{CONTROLLER}/stand')
        if not ok:
            print(f'gait_controller not available ({msg}); is the stack running?')
            return 1
        if args.pin_gait:
            restore = tool.get_params(CONTROLLER, ['gait_schedule', 'gait_schedule_speeds_mps'])
            if restore is None:
                print('gait_controller parameters not available')
                return 1
            if not _set_schedule(tool, [args.pin_gait], [0.0]):
                restore = None
                return 1
            print(f'speed schedule pinned to {args.pin_gait!r} (tuning override)')
        dog = RobotDog(node=tool.node)
        time.sleep(0.5)
        i = DEBUG_INDEX
        if args.pose:
            h, pitch, roll = (float(v) for v in args.pose.split(','))
            dog.set_height(h)
            dog.set_lean(pitch, roll)
            time.sleep(1.5)                            # let the leveling settle
            print(f'pose: height {h:.0f} mm, pitch {pitch:+.1f}, roll {roll:+.1f} deg -> applied '
                  f'{dog.debug("pose_height_mm"):.1f} mm, {dog.debug("pose_pitch_deg"):+.1f} / '
                  f'{dog.debug("pose_roll_deg"):+.1f} deg')
        d0 = samples[-1][1]
        gt0 = gt[-1] if gt else None
        print(f'start: attitude error roll {d0[i["roll_deg"]]:+.2f} pitch {d0[i["pitch_deg"]]:+.2f}'
              f' deg | walking {"ALLOWED" if d0[i["walk_allowed"]] > 0.5 else "BLOCKED"}')
        n_start = len(samples)
        t_cmd = time.monotonic()
        if args.step:
            what = 'step in place' + (f', turn to {args.turn_to:+.0f} deg' if args.turn_to
                                      is not None else '')
            print(f'--- {what} for {args.duration:.1f} s')
            res = dog.step_in_place(duration=args.duration, heading=args.turn_to)
        else:
            print(f'--- walk v={args.speed:+.3f} m/s for '
                  + (f'{args.distance:.2f} m' if args.distance is not None
                     else f'{args.duration:.1f} s')
                  + (f', heading {args.heading:+.1f} deg' if args.heading is not None else ''))
            res = dog.walk(speed=args.speed, heading=args.heading,
                           duration=None if args.distance is not None else args.duration,
                           distance=args.distance)
        t_stop = time.monotonic()
        time.sleep(args.settle)
        walk = [d for t, d in samples[n_start:] if int(d[i['state']]) == State.WALK]
        after = samples[-1][1]
        if not walk:
            print('the controller never entered WALK (walking not allowed? see its log)')
            return 1
        rolls = [d[i['roll_deg']] for d in walk if math.isfinite(d[i['roll_deg']])]
        pitches = [d[i['pitch_deg']] for d in walk if math.isfinite(d[i['pitch_deg']])]
        knots = sorted({int(d[i['gait_knot']]) for d in walk})
        print(f'WALK for {len(walk) / 50.0:.1f} s; gait speed up to '
              f'{max(d[i["gait_speed"]] for d in walk):.3f} m/s, schedule knots used {knots}, '
              f'duty {min(d[i["gait_duty"]] for d in walk):.2f}..'
              f'{max(d[i["gait_duty"]] for d in walk):.2f}, flight ticks '
              f'{sum(1 for d in walk if d[i["stance_legs"]] == 0)}')
        print(f'attitude error roll : peak {max(abs(r) for r in rolls):.2f} deg, rms '
              f'{_rms(rolls):.2f} deg')
        print(f'attitude error pitch: peak {max(abs(p) for p in pitches):.2f} deg, rms '
              f'{_rms(pitches):.2f} deg')
        print(f'heading error at the stop: {res["heading_error_deg"]:+.2f} deg (to the hold '
              f'reference), end {after[i["yaw_deg"]]:+.2f} deg')
        print(f'leveling u_roll/u_pitch: peak {max(abs(d[i["u_roll"]]) for d in walk):.2f}/'
              f'{max(abs(d[i["u_pitch"]]) for d in walk):.2f} deg; heading hold u_yaw peak '
              f'{max(abs(d[i["u_yaw"]]) for d in walk):.2f} rad/s')
        print(f'workspace clamps: {sum(int(d[i["ws_clamped_legs"]]) for d in walk)} leg-ticks, '
              f'IK err max {max(d[i["ik_max_err_mm"]] for d in walk):.2f} mm')
        print(f'planned distance {res["distance_m"]:+.3f} m in {res["time_s"]:.1f} s'
              + ('' if res['complete'] else f' (ended early: {res["reason"]})'))
        if gt0 and gt:
            seg = [g for g in gt if t_cmd <= g[0] <= t_stop + args.settle]
            dx, dy = gt[-1][1] - gt0[1], gt[-1][2] - gt0[2]
            yaw_change = _wrap(gt[-1][3] - gt0[3])
            print(f'sim ground truth: moved {math.hypot(dx, dy):.3f} m, yaw change '
                  f'{yaw_change:+.1f} deg' + (f' (mean rate {math.radians(yaw_change) / max(1e-6, t_stop - t_cmd):+.2f} rad/s)'
                                              if args.step else '') + f', {len(seg)} samples')
        if args.pose:
            dog.set_lean(0.0, 0.0)
            dog.set_height(dog.stand_height_mm)
        if args.trace:
            from pathlib import Path
            path = Path(args.trace)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write('t,' + ','.join(DEBUG_FIELDS) + '\n')
                for ts, d in samples:
                    f.write(f'{ts - t_cmd:.3f},' + ','.join(f'{v:.4f}' for v in d) + '\n')
            print(f'trace: {path} ({len(samples)} ticks, t=0 at the start of the behavior)')
        return 0

    restore_rc = 0
    try:
        code = measure()
    finally:
        if restore:
            restore_rc = _restore_schedule(tool, dog, restore)
        tool.shutdown()
    return restore_rc or code


def _restore_schedule(tool: ToolNode, dog, restore) -> int:
    """Stop the dog, wait for STAND, write the configured schedule back. 0, or 2 if it
    failed."""
    if dog is not None:
        dog.stop()
    standing = _wait_standing(tool, 5.0)
    names = list(restore['gait_schedule'])
    speeds = [float(v) for v in restore['gait_schedule_speeds_mps']]
    if standing and _set_schedule(tool, names, speeds):
        print(f'speed schedule restored to {list(zip(names, speeds))}')
        return 0
    print(f'!!! could not restore the speed schedule {list(zip(names, speeds))}'
          f'{"" if standing else " (the controller did not stand)"}: it is still pinned to one '
          'preset; restart the gait_controller (or set gait_schedule / '
          'gait_schedule_speeds_mps back atomically)')
    return 2


def _wait_standing(tool: ToolNode, timeout: float) -> bool:
    from robot_dog_control.controller import DEBUG_INDEX as I
    got = []
    sub = tool.node.create_subscription(Float64MultiArray, '/gait/debug',
                                        lambda m: got.append(int(m.data[I['state']])), 10)
    try:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if got and got[-1] == State.STAND:
                return True
            time.sleep(0.02)
        return False
    finally:
        tool.node.destroy_subscription(sub)


def _set_schedule(tool: ToolNode, names, speeds) -> bool:
    """Both schedule lists in one atomic update (their lengths must match)."""
    from rcl_interfaces.srv import SetParametersAtomically
    from rclpy.parameter import Parameter
    req = SetParametersAtomically.Request(parameters=[
        Parameter('gait_schedule', value=list(names)).to_parameter_msg(),
        Parameter('gait_schedule_speeds_mps', value=list(speeds)).to_parameter_msg()])
    res = tool.call(SetParametersAtomically, f'{CONTROLLER}/set_parameters_atomically', req)
    if res is None or not res.result.successful:
        print(f'could not set the gait schedule: {res.result.reason if res else "no reply"}')
        return False
    return True


if __name__ == '__main__':
    sys.exit(main())
