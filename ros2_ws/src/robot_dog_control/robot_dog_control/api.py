"""Behavior API: walk, step_in_place, set_height, set_lean and jump as a client of the running gait_controller.

``walk`` covers walking and running (the speed picks the gait). Headings (deg) count from where the dog
faced when the client connected."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool, Float32, Float64MultiArray

from robot_dog_control.controller import DEBUG_INDEX, State
from robot_dog_control.gait import V_EPS

_I = DEBUG_INDEX
CONTROLLER = '/gait_controller'
#: a distance walk gives up after |distance| / |speed| * 2 + the ramp time + this (s)
DEADLINE_MARGIN_S = 5.0
#: how long a behavior waits for the controller to stand again after stopping (s)
STOP_TIMEOUT_S = 5.0
#: a distance walk is done when the planned travel is this close to the target (m)
ARRIVE_TOL_M = 0.005
#: braking deceleration as a fraction of the controller's speed ramp
BRAKE_FRACTION = 0.8
#: longest gap between two /gait/debug messages integrated into the planned travel (s)
MAX_DT_S = 0.1


def _slew(cur: float, target: float, step: float) -> float:
    d = target - cur
    return target if abs(d) <= step else cur + math.copysign(step, d)


class _Approach:
    """Speed profile for walk(distance=...): ``|speed|``, then ``sqrt(2 decel rem)`` with
    ``rem`` the remaining planned travel to ``target_m``. Done within ``tol_m`` of it."""

    def __init__(self, target_m: float, speed: float, decel_mps2: float,
                 tol_m: float = ARRIVE_TOL_M):
        self.target = float(target_m)
        self.sign = 1.0 if speed >= 0.0 else -1.0
        self.speed = abs(float(speed))
        self.decel = float(decel_mps2)
        self.tol = float(tol_m)

    def update(self, x_m: float) -> Tuple[float, bool]:
        """(speed to command, done) at planned travel ``x_m`` (m)."""
        rem = (self.target - x_m) * self.sign
        v = 0.0 if rem <= 0.0 else min(self.speed, math.sqrt(2.0 * self.decel * rem))
        return self.sign * v, rem <= self.tol


class RobotDog:
    """Client of the running gait_controller."""

    def __init__(self, node=None, stream_hz: float = 50.0, height_rate_mmps: float = 30.0,
                 lean_rate_dps: float = 20.0, wait_s: float = 10.0):
        self._own_node = node is None
        self._own_rclpy = self._own_node and not rclpy.ok()
        if self._own_rclpy:
            rclpy.init()
        self.node = node or rclpy.create_node('robot_dog_api')
        self.height_rate_mmps = float(height_rate_mmps)
        self.lean_rate_dps = float(lean_rate_dps)
        self._dt = 1.0 / float(stream_hz)
        n = self.node
        self._pub_vel = n.create_publisher(Twist, '/cmd_vel', 10)
        self._pub_step = n.create_publisher(Bool, '/cmd_step', 10)
        self._pub_pose = n.create_publisher(Twist, '/cmd_pose', 10)
        self._pub_heading = n.create_publisher(Float32, '/cmd_heading', 10)
        self._dbg: Optional[List[float]] = None
        self._dbg_t: Optional[float] = None
        self._travel = 0.0             # planned body travel along the heading (m)
        self._dbg_cv = threading.Condition()
        n.create_subscription(Float64MultiArray, '/gait/debug', self._on_dbg, 10)
        self._executor = None
        if self._own_node:
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(n)
            threading.Thread(target=self._executor.spin, daemon=True).start()
        self._lock = threading.Lock()
        self._speed = 0.0              # streamed /cmd_vel linear.x (m/s)
        self._stepping = False         # streamed /cmd_step
        self._active = False           # streaming /cmd_vel and /cmd_step
        self._pose: Optional[List[float]] = None      # commanded [height, pitch, roll]
        self._goal: Optional[List[float]] = None      # ramp target
        self._resend = 0
        self._running = True
        self._warned_speed = False
        try:
            self._connect(wait_s)
        except Exception:
            self._shutdown_node()
            raise
        self._thread = threading.Thread(target=self._stream, daemon=True)
        self._thread.start()

    def _connect(self, wait_s: float) -> None:
        if not self._wait_dbg(wait_s):
            raise RuntimeError('no /gait/debug: is the gait_controller running?')
        self._pose = self._controller_pose()
        params = self._controller_params({'stand_height_mm': self._pose[0],
                                          'max_speed_mps': math.inf, 'ramp_time_s': 0.5})
        #: the controller's nominal stand height (mm): set_height(dog.stand_height_mm)
        self.stand_height_mm = float(params['stand_height_mm'])
        #: the controller's speed cap (m/s): faster walk() commands run at it
        self.max_speed_mps = float(params['max_speed_mps'])
        self._ramp_s = float(params['ramp_time_s'])
        # heading 0 = where the dog faces now: walk(heading=...) counts from here
        ok, msg = self._trigger(f'{CONTROLLER}/zero_yaw')
        if not ok:
            raise RuntimeError(f'could not zero the heading reference (~/zero_yaw): {msg}')

    def _shutdown_node(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._own_node:
            self.node.destroy_node()
        if self._own_rclpy:
            rclpy.try_shutdown()

    def walk(self, height: Optional[float] = None, heading: Optional[float] = None,
             speed: float = 0.0, duration: Optional[float] = None,
             distance: Optional[float] = None, continuous: bool = False) -> Dict[str, Any]:
        """Walk or run at ``speed`` (m/s, negative walks backward at the same heading).

        height: body height (mm) during the walk; None keeps the current one.
        heading: heading-hold target (deg); None holds the start heading.
        duration (s) or distance (m): stop after that long or that far. A distance is measured
            by integrating the gait's planned body speed and gives up after twice the nominal
            time plus the ramp time and DEADLINE_MARGIN_S.
        continuous: return at once and walk until ``stop()`` or the next behavior call.

        Returns a dict: 'time_s', 'distance_m' (planned travel from start to stand),
        'heading_error_deg', 'complete' (False if the walk ended early) and 'reason'.
        """
        if not math.isfinite(speed):
            raise ValueError('walk(): speed must be finite')
        if distance is not None and abs(speed) <= V_EPS:
            raise ValueError(f'walk(distance=...) needs |speed| > {V_EPS} m/s (it would never '
                             'get there)')
        self._check_walk_allowed()
        if abs(speed) > self.max_speed_mps + 1e-9 and not self._warned_speed:
            self._warned_speed = True
            self.node.get_logger().warn(
                f'walk(speed={speed:+.3f}): above the controller speed cap max_speed_mps '
                f'{self.max_speed_mps:.3f} m/s: it walks at the cap, as the gait the speed '
                'schedule has there (warned once)')
        if height is not None:
            self._set_goal(height=height)
        self._pub_heading.publish(Float32(data=math.radians(heading) if heading is not None
                                          else float('nan')))
        if not continuous and duration is None and distance is None:
            raise ValueError('walk(): give duration, distance or continuous=True')
        x0 = self._travel
        with self._lock:
            self._speed, self._stepping, self._active = float(speed), False, True
        if continuous:
            return self._result(0.0, 0.0, True, '')
        return self._run_until(duration, distance, speed, x0)

    def step_in_place(self, duration: Optional[float] = None, continuous: bool = False,
                      height: Optional[float] = None, heading: Optional[float] = None
                      ) -> Dict[str, Any]:
        """Step in place for ``duration`` s, or until ``stop()`` with ``continuous``;
        ``heading`` (deg) turns in place. Returns like walk()."""
        self._check_walk_allowed()
        if height is not None:
            self._set_goal(height=height)
        self._pub_heading.publish(Float32(data=math.radians(heading) if heading is not None
                                          else float('nan')))
        x0 = self._travel
        with self._lock:
            self._speed, self._stepping, self._active = 0.0, True, True
        if continuous:
            return self._result(0.0, 0.0, True, '')
        if duration is None:
            raise ValueError('step_in_place(): give duration or continuous=True')
        return self._run_until(duration, None, 0.0, x0)

    def set_height(self, target_mm: float, wait: bool = True) -> None:
        """Ramp the body height (mm above the ground) at height_rate_mmps from the current
        pose; the controller clamps it into its pose limits. Raises in REST or ESTOP."""
        self._set_goal(height=target_mm)
        if wait:
            self._wait_goal()

    def set_lean(self, pitch_deg: float, roll_deg: float = 0.0, wait: bool = True) -> None:
        """Ramp the body lean (deg, +pitch nose down, +roll right side down) at
        lean_rate_dps; leveling holds it relative to gravity. Raises in REST or ESTOP."""
        self._set_goal(pitch=pitch_deg, roll=roll_deg)
        if wait:
            self._wait_goal()

    def jump(self, height: float, displacement: float = 0.0) -> None:
        """Jump with apex rise ``height`` (mm) and ``displacement`` (mm, + forward). Not
        implemented yet."""
        raise NotImplementedError('jump() is the next building block (see its docstring)')

    def stop(self, timeout: float = STOP_TIMEOUT_S) -> bool:
        """End a continuous walk / step_in_place; True when the controller stands."""
        self._halt()
        return self._wait_state(State.STAND, timeout)

    def close(self) -> None:
        self.stop()
        self._running = False
        self._thread.join(timeout=1.0)
        self._shutdown_node()

    def debug(self, name: str) -> float:
        """Latest value of a /gait/debug field (controller.DEBUG_FIELDS)."""
        return float('nan') if self._dbg is None else self._dbg[_I[name]]

    @property
    def commanded_pose(self) -> List[float]:
        """[height mm, pitch deg, roll deg] this client commands."""
        with self._lock:
            return list(self._pose)

    def _call(self, srv_type, name: str, request, timeout: float = 2.0):
        cli = self.node.create_client(srv_type, name)
        try:
            if not cli.wait_for_service(timeout_sec=timeout):
                return None
            fut = cli.call_async(request)
            end = time.monotonic() + timeout
            while not fut.done() and time.monotonic() < end:
                time.sleep(0.01)
            return fut.result() if fut.done() else None
        finally:
            self.node.destroy_client(cli)

    def _trigger(self, name: str):
        from std_srvs.srv import Trigger
        res = self._call(Trigger, name, Trigger.Request())
        if res is None:
            return False, f'service {name} not available'
        return bool(res.success), res.message

    def _controller_params(self, defaults: Dict[str, float]) -> Dict[str, float]:
        """The gait_controller's values of these parameters (defaults if unavailable)."""
        from rcl_interfaces.srv import GetParameters
        from rclpy.parameter import parameter_value_to_python
        names = list(defaults)
        res = self._call(GetParameters, f'{CONTROLLER}/get_parameters',
                         GetParameters.Request(names=names))
        out = dict(defaults)
        if res is not None and len(res.values) == len(names):
            for name, value in zip(names, res.values):
                try:
                    out[name] = float(parameter_value_to_python(value))
                except (TypeError, ValueError):
                    pass
        return out

    def _controller_pose(self) -> List[float]:
        """The controller's commanded pose = applied pose minus the leveling correction."""
        d = self._dbg
        return [d[_I['pose_height_mm']], d[_I['pose_pitch_deg']] - d[_I['u_pitch']],
                d[_I['pose_roll_deg']] - d[_I['u_roll']]]

    def _on_dbg(self, msg: Float64MultiArray) -> None:
        """Store the latest /gait/debug and integrate its planned body speed."""
        now = time.monotonic()
        d = list(msg.data)
        with self._dbg_cv:
            k = _I['body_speed_mps']
            if self._dbg_t is not None and k < len(d) and math.isfinite(d[k]):
                self._travel += d[k] * min(now - self._dbg_t, MAX_DT_S)
            self._dbg, self._dbg_t = d, now
            self._dbg_cv.notify_all()

    def _wait_dbg(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        with self._dbg_cv:
            while self._dbg is None and time.monotonic() < end:
                self._dbg_cv.wait(0.1)
        return self._dbg is not None

    def _state(self) -> Optional[int]:
        s = self.debug('state')
        return int(s) if math.isfinite(s) else None

    def _check_walk_allowed(self) -> None:
        if self._state() not in (State.STAND, State.WALK):
            raise RuntimeError('the controller is not standing (REST or ESTOP): call its '
                               '~/stand first')
        if self.debug('walk_allowed') < 0.5:
            raise RuntimeError('the controller does not allow walking (envelope check '
                               'failed, see its log)')

    def _set_goal(self, height: Optional[float] = None, pitch: Optional[float] = None,
                  roll: Optional[float] = None) -> None:
        for v in (height, pitch, roll):
            if v is not None and not math.isfinite(v):
                raise ValueError('pose values must be finite')
        with self._lock:
            if self._state() in (State.REST, State.ESTOP):
                raise RuntimeError('the controller ignores pose commands in REST / ESTOP: '
                                   'call its ~/stand first')
            if self._goal is None:
                # no ramp in progress: start from the pose the controller holds now
                self._pose = self._controller_pose()
            goal = list(self._goal or self._pose)
            for k, v in ((0, height), (1, pitch), (2, roll)):
                if v is not None:
                    goal[k] = float(v)
            self._goal = goal

    def _wait_goal(self, timeout: float = 30.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if self._goal is None:
                    return
            time.sleep(self._dt)

    def _publish_motion(self, speed: float, stepping: bool) -> None:
        t = Twist()
        t.linear.x = float(speed)
        self._pub_vel.publish(t)
        self._pub_step.publish(Bool(data=bool(stepping)))

    def _halt(self) -> None:
        with self._lock:
            self._speed, self._stepping, self._active = 0.0, False, False
        self._publish_motion(0.0, False)

    def _stream(self) -> None:
        """Publisher thread: the pose ramp at stream_hz, /cmd_vel and /cmd_step while a
        behavior runs."""
        next_t = time.monotonic()
        while self._running:
            with self._lock:
                pose_msg = None
                if self._goal is not None:
                    dh = self.height_rate_mmps * self._dt
                    da = self.lean_rate_dps * self._dt
                    p, g = self._pose, self._goal
                    self._pose = [_slew(p[0], g[0], dh), _slew(p[1], g[1], da),
                                  _slew(p[2], g[2], da)]
                    pose_msg = list(self._pose)
                    if self._pose == g:
                        self._goal = None
                        self._resend = 5            # repeat the final setpoint a few times
                elif self._resend > 0:
                    self._resend -= 1
                    pose_msg = list(self._pose)
                motion = (self._speed, self._stepping) if self._active else None
            if pose_msg is not None:
                t = Twist()
                t.linear.z = pose_msg[0] / 1000.0
                t.angular.y = math.radians(pose_msg[1])
                t.angular.x = math.radians(pose_msg[2])
                self._pub_pose.publish(t)
            if motion is not None:
                self._publish_motion(*motion)
            next_t += self._dt
            time.sleep(max(0.0, next_t - time.monotonic()))

    def _result(self, time_s: float, distance_m: float, complete: bool, reason: str,
                heading_err: float = float('nan')) -> Dict[str, Any]:
        return {'time_s': time_s, 'distance_m': distance_m, 'heading_error_deg': heading_err,
                'complete': complete, 'reason': reason}

    def _run_until(self, duration: Optional[float], distance: Optional[float],
                   speed: float, x0: float) -> Dict[str, Any]:
        t0 = time.monotonic()
        approach = None
        if distance is not None:
            deadline = t0 + abs(distance) / abs(speed) * 2.0 + self._ramp_s + DEADLINE_MARGIN_S
            decel = BRAKE_FRACTION * self.max_speed_mps / max(self._ramp_s, 1e-3)
            approach = _Approach(x0 + math.copysign(abs(distance), speed), speed, decel)
        complete, reason = True, ''
        while True:
            time.sleep(self._dt)
            now = time.monotonic()
            if self._state() not in (State.STAND, State.WALK):
                complete, reason = False, 'the controller left STAND/WALK (REST or ESTOP)'
                break
            if self.debug('walk_allowed') < 0.5:
                complete, reason = False, 'the controller disallowed walking'
                break
            if duration is not None and now - t0 >= duration:
                break
            if approach is not None:
                v, done = approach.update(self._travel)
                if done:
                    break
                with self._lock:
                    self._speed = v
                if now >= deadline:
                    complete, reason = False, (
                        f'deadline: {abs(self._travel - x0):.3f} of '
                        f'{abs(distance):.3f} m in {now - t0:.1f} s')
                    break
        heading_err = self.debug('yaw_deg')
        self._halt()
        end = time.monotonic() + STOP_TIMEOUT_S
        while time.monotonic() < end:
            time.sleep(self._dt)
            if self._state() != State.WALK:
                break
        return self._result(time.monotonic() - t0, self._travel - x0, complete, reason,
                            heading_err)

    def _wait_state(self, state: State, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._dbg is not None and self._state() == int(state):
                return True
            time.sleep(self._dt)
        return False
