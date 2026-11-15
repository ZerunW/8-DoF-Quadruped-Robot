"""calib_offsets: turn the leveled stand pose into servo offsets.

Run after launch_bringup with the dog on a level floor: ``pixi run calib_offsets [-- --align right=+5]``.
Aligns the feet fore/aft (``lf +5`` moves the left-front foot 5 mm toward the nose), folds the servo deltas
into the offsets in steps of at most --step-deg until the residual leveling correction is under 0.6 deg,
then saves with zero trims."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

from std_msgs.msg import Float64MultiArray

from robot_dog_control.config import controller_params_from_config, load_robot_config
from robot_dog_control.controller import DEBUG_INDEX, State, build_components
from robot_dog_control.leg_map import LEG_NAMES
from robot_dog_hw.calibration import update_calibration
from robot_dog_hw.servo_map import MAX_ABS_OFFSET_DEG, MAX_OFFSET_STEP_DEG, SERVO_LABELS
from robot_dog_hw.tools.common import CONTROLLER, DRIVER, ToolNode

#: Parameters shrunk to the stand pose while nudging by hand, listed in restore order.
MANUAL_ENVELOPE = ('height_min_mm', 'height_max_mm', 'rest_height_mm', 'pose_pitch_max_deg',
                   'pose_roll_max_deg', 'pose_x_max_mm', 'max_stride_mm')


def manual_envelope(p) -> Dict[str, float]:
    """MANUAL_ENVELOPE values that allow only standing at stand_height_mm, rest height
    first."""
    h = p.stand_height_mm
    return {'rest_height_mm': h, 'height_min_mm': h, 'height_max_mm': h,
            'pose_pitch_max_deg': 0.0, 'pose_roll_max_deg': 0.0, 'pose_x_max_mm': 0.0,
            'max_stride_mm': 0.0}

GROUPS = {
    'lf': (0,), 'rf': (1,), 'lb': (2,), 'rb': (3,),
    'left': (0, 2), 'right': (1, 3), 'front': (0, 1), 'back': (2, 3), 'all': (0, 1, 2, 3),
}


class Session:

    def __init__(self, tool: ToolNode, args):
        self.tool = tool
        self.args = args
        self.latest: Optional[List[float]] = None
        tool.node.create_subscription(Float64MultiArray, '/gait/debug', self._on_dbg, 10)
        tool.watch_servo_state()
        cfg = load_robot_config()
        self.comp = build_components(controller_params_from_config(cfg))

    def _on_dbg(self, msg):
        self.latest = list(msg.data)

    def dbg(self, key: str) -> float:
        return float('nan') if self.latest is None else self.latest[DEBUG_INDEX[key]]

    def wait_level(self, label: str) -> bool:
        a = self.args
        print(f'{label}: waiting for |roll|,|pitch| < {a.tol_deg} deg for {a.hold_s} s ...')
        t0 = time.monotonic()
        ok_since = None
        last_print = 0.0
        while True:
            now = time.monotonic()
            if now - t0 > a.timeout_s:
                print('TIMEOUT: the body did not settle (floor level? imu_check signs? gains?)')
                return False
            time.sleep(0.05)
            if self.latest is None:
                continue
            roll, pitch = self.dbg('roll_deg'), self.dbg('pitch_deg')
            active = self.dbg('level_active') > 0.5
            state = int(self.dbg('state'))
            settled = (state == State.STAND and active and abs(roll) < a.tol_deg
                       and abs(pitch) < a.tol_deg)
            ok_since = (ok_since or now) if settled else None
            if now - last_print > 1.0:
                last_print = now
                print(f'  roll {roll:+6.2f} pitch {pitch:+6.2f} yaw {self.dbg("yaw_deg"):+7.2f} deg'
                      f' | u_roll {self.dbg("u_roll"):+6.2f} u_pitch {self.dbg("u_pitch"):+6.2f} deg'
                      f' | leveling {"on" if active else "OFF"}')
            if ok_since is not None and now - ok_since >= a.hold_s:
                return True

    def wait_stand_pose(self, timeout: float = 5.0) -> bool:
        """With leveling off, wait until the correction has faded out and the controller
        holds the level stand pose."""
        stand = float(self.controller_params()['stand_height_mm'])
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            time.sleep(0.05)
            vals = [self.dbg(k) for k in ('u_roll', 'u_pitch', 'pose_pitch_deg',
                                          'pose_roll_deg')]
            if all(math.isfinite(v) and abs(v) < 1e-3 for v in vals) and \
                    abs(self.dbg('pose_height_mm') - stand) < 1e-3:
                return True
        return False

    def restore_envelope(self) -> bool:
        """Restore the configured MANUAL_ENVELOPE values, then enable leveling. False if
        the controller rejected any."""
        restore = {k: getattr(self.comp.params, k) for k in MANUAL_ENVELOPE}
        results = self.tool.set_params(CONTROLLER, restore)
        bad = [r for ok, r in results if not ok]
        if bad:
            print(f'could not restore {restore}: {bad} (restart the controller to get the '
                  'configured values back)')
        ok, msg = self.tool.set_bool(f'{CONTROLLER}/level_enable', True)
        print(f'level_enable: {msg}')
        return not bad and ok

    def controller_params(self) -> Dict[str, float]:
        p = self.tool.get_params(CONTROLLER, ['stand_fx_mm', 'stand_height_mm', 'leg_trim_mm',
                                              'leg_trim_x_mm'])
        if p is None:
            raise RuntimeError('gait_controller parameters not available')
        return p

    def nominal_servo(self, stand_fx: float, height: float) -> List[float]:
        """Servo angles of the trim-free, leveling-free stand pose."""
        kin, leg_map = self.comp.kin, self.comp.leg_map
        current = self.tool.servo_state
        guesses = leg_map.motor_angles(current) if current else [None] * 4
        motor = []
        for i in range(4):
            x, y = leg_map.body_to_canonical(i, stand_fx, -height)
            r = kin.ik(x, y, guesses[i])
            if not r.ok:
                raise RuntimeError(f'nominal stand pose unreachable for leg {LEG_NAMES[i]}')
            motor.append((r.m1, r.m2))
        return leg_map.servo_vector(motor)

    def _nudge(self, specs: List[str], param: str, sign: float, what: str) -> bool:
        trims = list(self.controller_params()[param])
        for spec in specs:
            try:
                group, mm = spec.split('=')
                delta = float(mm)
                idx = GROUPS[group.strip().lower()]
            except (ValueError, KeyError):
                print(f'bad {what} {spec!r}: use <lf|rf|lb|rb|left|right|front|back|all>=<mm>')
                return False
            for i in idx:
                trims[i] = round(trims[i] + sign * delta, 2)
        (ok, reason), = self.tool.set_params(CONTROLLER, {param: trims})
        if not ok:
            print(f'controller REJECTED {param} {trims}: {reason}')
            return False
        print(f'{param}: ' + ', '.join(f'{n} {t:+.1f}' for n, t in zip(LEG_NAMES, trims)) + ' mm')
        return True

    def apply_align(self, specs: List[str]) -> bool:
        """--align right=+5 ...: feet toward the nose (leg_trim_x_mm += mm)."""
        return self._nudge(specs, 'leg_trim_x_mm', +1.0, '--align')

    def apply_raise(self, specs: List[str]) -> bool:
        """--raise right=+5 ...: body corner(s) up (legs extend: leg_trim_mm -= mm)."""
        return self._nudge(specs, 'leg_trim_mm', -1.0, '--raise')

    def record_level_here(self) -> bool:
        """Record the current body attitude (leveling off) as the level target."""
        cur = self.tool.get_params(CONTROLLER, ['level_target_roll_deg',
                                                'level_target_pitch_deg'])
        if cur is None:
            print('gait_controller parameters not available')
            return False
        # /gait/debug roll/pitch are relative to the current targets
        roll = round(float(cur['level_target_roll_deg']) + self.dbg('roll_deg'), 2)
        pitch = round(float(cur['level_target_pitch_deg']) + self.dbg('pitch_deg'), 2)
        if not (math.isfinite(roll) and math.isfinite(pitch)):
            print('no IMU data')
            return False
        axes = self.args.level_axes
        new = {}
        if axes in ('roll', 'both'):
            new['level_target_roll_deg'] = roll
        if axes in ('pitch', 'both'):
            new['level_target_pitch_deg'] = pitch
        (ok, reason), = self.tool.set_params(CONTROLLER, new)
        if not ok:
            print(f'controller REJECTED the level targets {new}: {reason}')
            return False
        path = update_calibration('gait_controller', new)
        print(f'level target recorded ({axes}): ' +
              ', '.join(f'{k} {v:+.2f} deg' for k, v in new.items()) +
              f' = body level (saved to {path})')
        return True

    def align_feet(self) -> None:
        if not sys.stdin.isatty():
            print('\n(no terminal for the interactive alignment; use --align right=+5 etc.)')
            return
        print('\nFore/aft alignment. Left and right feet of the same end must stand on one '
              'line.\n  <leg|group> <mm>   e.g. "lf +5" (left-front foot 5 mm toward the nose),'
              ' "right -3", "front +2", "all -4"\n  show / undo / done\n')
        history: List[List[float]] = []
        while True:
            try:
                line = input('align> ').strip().lower()
            except EOFError:
                line = 'done'
            if not line:
                continue
            if line == 'done':
                return
            trims = list(self.controller_params()['leg_trim_x_mm'])
            if line == 'show':
                print('  leg_trim_x_mm: ' + ', '.join(f'{n} {t:+.1f}' for n, t in zip(LEG_NAMES, trims)))
                continue
            if line == 'undo':
                if not history:
                    print('  nothing to undo')
                    continue
                new = history.pop()
            else:
                parts = line.split()
                if len(parts) != 2 or parts[0] not in GROUPS:
                    print('  usage: <lf|rf|lb|rb|left|right|front|back|all> <mm>')
                    continue
                try:
                    delta = float(parts[1])
                except ValueError:
                    print('  mm must be a number')
                    continue
                new = list(trims)
                for i in GROUPS[parts[0]]:
                    new[i] = round(new[i] + delta, 2)
            (ok, reason), = self.tool.set_params(CONTROLLER, {'leg_trim_x_mm': new})
            if not ok:
                print(f'  REJECTED: {reason}')
                continue
            if line != 'undo':
                history.append(trims)
            print('  leg_trim_x_mm: ' + ', '.join(f'{n} {t:+.1f}' for n, t in zip(LEG_NAMES, new)))

    def fold_into_offsets(self) -> Optional[List[float]]:
        tool, a = self.tool, self.args
        cp = self.controller_params()
        current = tool.servo_state
        if current is None:
            print('no /servo_state')
            return None
        old = tool.get_params(DRIVER, ['servo_offset'])
        if old is None:
            print('servo_driver parameters not available')
            return None
        old = [float(v) for v in old['servo_offset']]
        nominal = self.nominal_servo(cp['stand_fx_mm'], cp['stand_height_mm'])
        delta = [c - n for c, n in zip(current, nominal)]
        new = [round(o + d, 3) for o, d in zip(old, delta)]
        print(f'\nstand_fx_mm {cp["stand_fx_mm"]:.1f}  stand_height_mm {cp["stand_height_mm"]:.1f}'
              f'  leg_trim_mm {[round(t, 1) for t in cp["leg_trim_mm"]]}'
              f'  leg_trim_x_mm {[round(t, 1) for t in cp["leg_trim_x_mm"]]}')
        print(f'leveling correction: u_roll {self.dbg("u_roll"):+.2f} deg  u_pitch '
              f'{self.dbg("u_pitch"):+.2f} deg\n')
        print('idx label               servo now  nominal   delta   old offset  new offset')
        for i in range(8):
            print(f'{i:>3} {SERVO_LABELS[i]:<19} {current[i]:9.2f} {nominal[i]:8.2f} '
                  f'{delta[i]:+7.2f} {old[i]:+11.2f} {new[i]:+11.2f}')
        if any(abs(v) > MAX_ABS_OFFSET_DEG for v in new):
            print(f'\nABORT: an offset would exceed {MAX_ABS_OFFSET_DEG} deg; re-horn that servo.')
            return None
        if a.dry_run:
            print('(dry run: nothing applied or saved)')
            return None

        # Return the controller to its raw pose first: leveling off, trims zero.
        tool.set_bool(f'{CONTROLLER}/level_enable', False)
        tool.set_params(CONTROLLER, {'leg_trim_mm': [0.0] * 4, 'leg_trim_x_mm': [0.0] * 4})
        time.sleep(1.5)
        n_steps = max(1, math.ceil(max(abs(d) for d in delta) / a.step_deg))
        for k in range(1, n_steps + 1):
            step = [round(o + d * k / n_steps, 3) for o, d in zip(old, delta)]
            (ok, reason), = tool.set_params(DRIVER, {'servo_offset': step})
            if not ok:
                print(f'servo_driver REJECTED the offsets: {reason}')
                return None
            time.sleep(0.3)
        print(f'offsets applied in {n_steps} step(s)')
        # Trims are zero now: restore the configured envelope before enabling the loop again.
        self.restore_envelope()
        return new

    def run(self) -> int:
        tool, a = self.tool, self.args
        ok, msg = tool.trigger(f'{CONTROLLER}/stand')
        if not ok:
            print(f'gait_controller not available ({msg}); start `pixi run launch_bringup`')
            return 1
        if tool.wait_servo_state(5.0) is None:
            print('no /servo_state: is servo_driver running?')
            return 1
        manual = bool(a.raise_) or a.level_here or a.no_fold
        if manual:
            ok, msg = tool.set_bool(f'{CONTROLLER}/level_enable', False)
            print(f'level_enable: {msg}')
            # Wait for the correction to fade out and the ~/stand ramp to finish.
            if not self.wait_stand_pose():
                print('the leveling correction did not fade out / the stand pose was not '
                      'reached: leveling back on, nothing changed')
                self.restore_envelope()
                return 1
            # Shrink the envelope while nudging; restored by the fold or the next launch.
            results = tool.set_params(CONTROLLER, manual_envelope(self.comp.params))
            bad = [r for ok, r in results if not ok]
            if bad:
                print(f'could not shrink the envelope for the manual phase: {bad}')
                self.restore_envelope()
                return 1
            time.sleep(1.0)
            if a.raise_ and not self.apply_raise(a.raise_):
                return 1
            if a.align and not self.apply_align(a.align):
                return 1
            time.sleep(1.5)
            cp = self.controller_params()
            print(f'IMU now: roll {self.dbg("roll_deg"):+.2f} pitch {self.dbg("pitch_deg"):+.2f} '
                  f'yaw {self.dbg("yaw_deg"):+.2f} deg (relative to the level targets)\n'
                  f'leg_trim_mm {[round(t, 1) for t in cp["leg_trim_mm"]]}  '
                  f'leg_trim_x_mm {[round(t, 1) for t in cp["leg_trim_x_mm"]]}')
            if a.no_fold:
                print('(--no-fold: trims left in place, leveling OFF; look at the dog and run '
                      'again with more --raise/--align, or with --level-here to finish)')
                return 0
            if a.level_here and not self.record_level_here():
                return 1
        else:
            ok, msg = tool.set_bool(f'{CONTROLLER}/level_enable', True)
            print(f'level_enable: {msg}')
            if not ok:
                return 1
            if not self.wait_level('initial'):
                return 1
            if a.align:
                if not self.apply_align(a.align):
                    return 1
                time.sleep(1.0)
            elif not a.no_align:
                self.align_feet()
            if (a.align or not a.no_align) and not self.wait_level('after alignment'):
                return 1
        final = None
        for it in range(1, a.iterations + 1):
            print(f'\n=== iteration {it}/{a.iterations} ===')
            new = self.fold_into_offsets()
            if new is None:
                return 1
            final = new
            if not self.wait_level('after new offsets'):
                return 1
            residual = max(abs(self.dbg('u_roll')), abs(self.dbg('u_pitch')))
            print(f'residual leveling correction: {residual:.2f} deg')
            if residual <= 0.6:
                break
        ok, msg = tool.trigger(f'{CONTROLLER}/zero_yaw')
        print(f'zero_yaw: {msg}')
        time.sleep(0.5)
        path = update_calibration('servo_driver', {'servo_offset': final})
        update_calibration('gait_controller', {'leg_trim_mm': [0.0] * 4, 'leg_trim_x_mm': [0.0] * 4})
        print(f'\nsaved servo_offset {final} to {path}')
        print(f'final: roll {self.dbg("roll_deg"):+.2f} pitch {self.dbg("pitch_deg"):+.2f} '
              f'yaw {self.dbg("yaw_deg"):+.2f} deg, kinematic height '
              f'{self.dbg("kin_height_mm"):.1f} mm, correction u_roll {self.dbg("u_roll"):+.2f} '
              f'u_pitch {self.dbg("u_pitch"):+.2f} deg')
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tol-deg', type=float, default=0.3)
    ap.add_argument('--hold-s', type=float, default=2.0)
    ap.add_argument('--timeout-s', type=float, default=60.0)
    ap.add_argument('--step-deg', type=float, default=min(4.0, MAX_OFFSET_STEP_DEG),
                    help='largest offset change per driver update')
    ap.add_argument('--iterations', type=int, default=2)
    ap.add_argument('--align', action='append', default=[], metavar='GROUP=MM',
                    help='fore/aft nudge in mm (+ toward the nose), e.g. --align right=+5 '
                         '--align lf=-2; groups: lf rf lb rb left right front back all')
    ap.add_argument('--raise', dest='raise_', action='append', default=[], metavar='GROUP=MM',
                    help='MANUAL leveling: body corner(s) up by mm (legs extend), e.g. '
                         '--raise right=+10; switches the IMU leveling off')
    ap.add_argument('--no-fold', action='store_true',
                    help='apply the nudges, print the IMU, and stop (look, then run again)')
    ap.add_argument('--level-here', action='store_true',
                    help='the body is level NOW (by eye): record the IMU roll/pitch as the '
                         'level targets, then fold the trims into the offsets')
    ap.add_argument('--level-axes', choices=['roll', 'pitch', 'both'], default='both',
                    help='which IMU axes --level-here records as mount offsets (an axis the '
                         'IMU reads correctly keeps its target, normally 0)')
    ap.add_argument('--no-align', action='store_true', help='skip the fore/aft alignment prompt')
    ap.add_argument('--dry-run', action='store_true', help='compute and print only')
    args, _ = ap.parse_known_args(argv)
    tool = ToolNode('calib_offsets_tool')
    try:
        return Session(tool, args).run()
    except KeyboardInterrupt:
        print('\ninterrupted: nothing saved')
        return 1
    finally:
        tool.shutdown()


if __name__ == '__main__':
    sys.exit(main())
