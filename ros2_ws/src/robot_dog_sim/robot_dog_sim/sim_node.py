"""sim: MuJoCo physics in place of the servos and the BNO055.

Steps the physics in real time (500 Hz) and drives the motor hinges from /servo_state.

Subscribes  /servo_state (servo deg), /servo_driver/enabled (false: motors limp),
            /sim/apply_wrench (world-frame force N and torque N m for wrench_duration_s)
Publishes   /bno055/imu (body frame, acceleration without gravity), /bno055/imu_raw,
            /bno055/calib_status, /sim/ground_truth (twist in base_link),
            /sim/foot_forces (N, lf rf lb rb), /joint_states, TF odom -> base_link
Services    /sim/push (push_force_n, push_torque_nm for push_duration_s), /sim/reset
Parameters  ground_tilt_roll_deg (+: right side lower), ground_tilt_pitch_deg (+: nose
            lower); servo, payload, push and noise parameters are dynamic
"""

from __future__ import annotations

import json
import math
import random
import sys
import threading
import time
from typing import List, Optional

import numpy as np
from geometry_msgs.msg import TransformStamped, Wrench
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from robot_dog_control.config import controller_params_from_config, load_robot_config
from robot_dog_control.kinematics import LegGeometry, LegKinematics
from robot_dog_control.leg_map import LegMap
from robot_dog_description.model import config_mismatches, load_cad_frames
from robot_dog_hw.node_runner import run_node
from robot_dog_hw.ros_params import as_bool, as_float, as_float_list, declare
from robot_dog_hw.servo_map import SERVO_LABELS
from robot_dog_sim.sim_core import DogSim, ServoParams, mount_imu, quat_to_mat

#: most simulated time stepped per timer tick when the physics falls behind (s)
MAX_CATCHUP_S = 0.1


class SimNode(Node):

    def __init__(self, **kw):
        super().__init__('sim', **kw)
        cfg = load_robot_config()
        d = ServoParams.from_config(cfg)        # slew = servo_driver.max_servo_speed_dps
        declare(self, 'rate_hz', 50.0, 'publish rate (physics is stepped in batches)',
                read_only=True)
        declare(self, 'viewer', False, 'MuJoCo passive viewer window', read_only=True)
        declare(self, 'wait_for_servo_state', True, 'start the physics at the first /servo_state',
                read_only=True)
        declare(self, 'publish_tf', True, read_only=True)
        declare(self, 'publish_joint_states', True, read_only=True)
        declare(self, 'odom_frame', 'odom', read_only=True)
        declare(self, 'base_frame', 'base_link', read_only=True)
        declare(self, 'imu_frame', 'imu_link', read_only=True)
        declare(self, 'servo_kp', d.kp, 'N m / rad')
        declare(self, 'servo_kv', d.kv, 'N m s / rad')
        declare(self, 'servo_torque_nm', d.torque_nm, 'N m')
        declare(self, 'servo_speed_dps', d.speed_dps, 'set-point slew limit, deg/s')
        declare(self, 'ground_tilt_roll_deg', 0.0, '+ = right side lower')
        declare(self, 'ground_tilt_pitch_deg', 0.0, '+ = nose lower')
        declare(self, 'payload_mass_kg', 0.0)
        declare(self, 'payload_offset_m', [0.0, 0.0, 0.0], 'in base_link')
        declare(self, 'push_force_n', [0.0, 3.0, 0.0], 'world frame, /sim/push')
        declare(self, 'push_torque_nm', [0.0, 0.0, 0.0], 'world frame, /sim/push')
        declare(self, 'push_duration_s', 0.2)
        declare(self, 'wrench_duration_s', 0.2, 'duration of each /sim/apply_wrench')
        declare(self, 'imu_noise_deg', 0.0)
        declare(self, 'gyro_noise_rps', 0.0)
        declare(self, 'accel_noise_mps2', 0.0)

        params = controller_params_from_config(cfg)
        declare(self, 'imu_mount_roll_deg', float(params.level_target_roll_deg),
                'IMU board roll in the body (the real mount; level_target_roll_deg)',
                read_only=True)
        declare(self, 'imu_mount_pitch_deg', float(params.level_target_pitch_deg),
                'IMU board pitch in the body (level_target_pitch_deg)', read_only=True)
        self._mount = (math.radians(as_float(self.get_parameter('imu_mount_roll_deg').value,
                                             'imu_mount_roll_deg')),
                       math.radians(as_float(self.get_parameter('imu_mount_pitch_deg').value,
                                             'imu_mount_pitch_deg')))
        self._leg_map = LegMap.from_params(params.leg_names, params.leg_thigh_idx,
                                           params.leg_calf_idx, params.leg_mirror_servo,
                                           params.leg_x_sign, params.leg_pos_x_mm,
                                           params.leg_pos_y_mm)
        kin = LegKinematics(LegGeometry.from_mapping(params.geom), region=params.joint_region())
        for line in config_mismatches(load_cad_frames(), cfg.get('gait_controller', {})):
            self.get_logger().warn(f'robot_dog.yaml vs CAD: {line}')
        self.sim = DogSim(kin)
        self._apply_params(None)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self._rate = as_float(g('rate_hz'), 'rate_hz')
        self._odom, self._base = str(g('odom_frame')), str(g('base_frame'))
        self._imu_frame = str(g('imu_frame'))
        self._rng = random.Random(1)
        self._lock = threading.Lock()
        self._servo: Optional[List[float]] = None
        self._running = not bool(g('wait_for_servo_state'))
        self._t0 = None
        self._steps = 0
        self._names = list(self.sim.joint_names)
        if self._running:
            self._reset(None)

        sensor = qos_profile_sensor_data
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._pub_imu = self.create_publisher(Imu, '/bno055/imu', 10)
        self._pub_imu_raw = self.create_publisher(Imu, '/bno055/imu_raw', 10)
        self._pub_cal = self.create_publisher(String, '/bno055/calib_status', 10)
        self._pub_gt = self.create_publisher(Odometry, '~/ground_truth', 10)
        self._pub_feet = self.create_publisher(Float64MultiArray, '~/foot_forces', 10)
        self._pub_js = (self.create_publisher(JointState, '/joint_states', 10)
                        if as_bool(g('publish_joint_states'), 'publish_joint_states') else None)
        self._tf = TransformBroadcaster(self) if as_bool(g('publish_tf'), 'publish_tf') else None
        self.create_subscription(JointState, '/servo_state', self._on_servo, sensor)
        self.create_subscription(Bool, '/servo_driver/enabled', self._on_enabled, latched)
        self.create_subscription(Wrench, '~/apply_wrench', self._on_wrench, 10)
        self.create_service(Trigger, '~/push', self._on_push)
        self.create_service(Trigger, '~/reset', self._on_reset)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.create_timer(1.0 / self._rate, self._on_timer)
        self.create_timer(1.0, self._on_calib)
        self._viewer = None
        if as_bool(g('viewer'), 'viewer'):
            try:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
            except Exception as exc:
                self.get_logger().error(f'MuJoCo viewer unavailable ({exc}); running headless')
        self.get_logger().info(
            f'MuJoCo {self.sim.model.nbody} bodies, {self.sim.total_mass:.3f} kg, dt '
            f'{self.sim.dt * 1e3:.1f} ms; '
            + ('waiting for /servo_state' if not self._running else 'running'))

    # ------------------------------------------------------------ parameters
    def _apply_params(self, changed) -> None:
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        keys = changed or {'servo', 'tilt', 'payload'}
        if 'servo' in keys:
            self.sim.set_servo_params(ServoParams(
                as_float(g('servo_kp'), 'servo_kp'), as_float(g('servo_kv'), 'servo_kv'),
                as_float(g('servo_torque_nm'), 'servo_torque_nm'),
                as_float(g('servo_speed_dps'), 'servo_speed_dps')))
        if 'tilt' in keys:
            self.sim.set_ground_tilt(math.radians(as_float(g('ground_tilt_roll_deg'), 'roll')),
                                     math.radians(as_float(g('ground_tilt_pitch_deg'), 'pitch')))
            self._retilt = changed is not None
        if 'payload' in keys:
            self.sim.set_payload(as_float(g('payload_mass_kg'), 'payload_mass_kg'),
                                 as_float_list(g('payload_offset_m'), 'payload_offset_m', 3))

    def _on_set_parameters(self, params):
        groups = {'servo_kp': 'servo', 'servo_kv': 'servo', 'servo_torque_nm': 'servo',
                  'servo_speed_dps': 'servo', 'ground_tilt_roll_deg': 'tilt',
                  'ground_tilt_pitch_deg': 'tilt', 'payload_mass_kg': 'payload',
                  'payload_offset_m': 'payload'}
        try:
            for p in params:
                if p.name in ('payload_offset_m', 'push_force_n', 'push_torque_nm'):
                    as_float_list(p.value, p.name, 3)
                elif p.name in groups or p.name.endswith(('_s', '_deg', '_rps', '_mps2', '_m')):
                    v = as_float(p.value, p.name)
                    if p.name.startswith('servo_') and v < 0:
                        raise ValueError(f'{p.name} must be >= 0')
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        changed = {groups[p.name] for p in params if p.name in groups}
        if changed:
            # applied on the next timer tick, after the parameters are stored
            self._pending = (self._pending or set()) | changed
        return SetParametersResult(successful=True)

    _pending = None
    _retilt = False

    # ---------------------------------------------------------------- inputs
    def _on_servo(self, msg: JointState):
        if msg.name and list(msg.name) != list(SERVO_LABELS):
            index = {n: i for i, n in enumerate(msg.name)}
            try:
                servo = [msg.position[index[n]] for n in SERVO_LABELS]
            except (KeyError, IndexError):
                return
        else:
            servo = list(msg.position)
        if len(servo) != 8 or not all(math.isfinite(s) for s in servo):
            return
        with self._lock:
            self._servo = servo
            self.sim.set_targets_from_servo(self._leg_map, servo)
            if not self._running:
                self._reset(servo)
                self._running = True
                self.get_logger().info('first /servo_state: physics running')

    def _on_enabled(self, msg: Bool):
        with self._lock:
            self.sim.set_enabled(bool(msg.data))
        if not msg.data:
            self.get_logger().warn('servo driver relaxed: motors limp')

    def _on_wrench(self, msg: Wrench):
        dur = as_float(self.get_parameter('wrench_duration_s').value, 'wrench_duration_s')
        f, t = msg.force, msg.torque
        with self._lock:
            self.sim.apply_wrench((f.x, f.y, f.z), (t.x, t.y, t.z), dur)
        self.get_logger().info(f'wrench F=({f.x:.2f}, {f.y:.2f}, {f.z:.2f}) N, '
                               f'T=({t.x:.3f}, {t.y:.3f}, {t.z:.3f}) N m for {dur:.2f} s')

    def _on_push(self, request, response):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        f = as_float_list(g('push_force_n'), 'push_force_n', 3)
        t = as_float_list(g('push_torque_nm'), 'push_torque_nm', 3)
        dur = as_float(g('push_duration_s'), 'push_duration_s')
        with self._lock:
            self.sim.apply_wrench(f, t, dur)
        response.success = True
        response.message = f'push F={f} N, T={t} N m for {dur:.2f} s (world frame)'
        self.get_logger().info(response.message)
        return response

    def _on_reset(self, request, response):
        with self._lock:
            self._reset(self._servo)
        response.success = True
        response.message = 'reset to the origin in the current servo pose'
        self.get_logger().info(response.message)
        return response

    def _reset(self, servo: Optional[List[float]]) -> None:
        motor = (self._leg_map.motor_angles(servo) if servo is not None
                 else [(90.0, 90.0)] * 4)
        self.sim.reset(motor)
        self._t0 = time.monotonic()
        self._steps = 0

    # ---------------------------------------------------------------- output
    def _on_timer(self):
        if self._pending:
            changed, self._pending = self._pending, None
            with self._lock:
                self._apply_params(changed)
        if not self._running:
            return
        if self._retilt:
            # ground tilt changed: reset the dog onto the new slope at the origin
            self._retilt = False
            with self._lock:
                self._reset(self._servo)
            self.get_logger().info(f'ground tilt roll/pitch {np.degrees(self.sim.ground_rp)} '
                                   'deg: dog reset to the origin')
        now = time.monotonic()
        due = int((now - self._t0) / self.sim.dt) - self._steps
        n = min(max(due, 0), int(MAX_CATCHUP_S / self.sim.dt))
        with self._lock:
            if self._viewer is not None and self._viewer.is_running():
                with self._viewer.lock():
                    self.sim.step(n)
                self._viewer.sync()
            else:
                self.sim.step(n)
            self._steps += max(due, 0)          # drop any backlog beyond MAX_CATCHUP_S
            self._publish()

    def _noise(self, sigma: float) -> float:
        return self._rng.gauss(0.0, sigma) if sigma > 0 else 0.0

    def _publish(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        sim = self.sim
        stamp = self.get_clock().now().to_msg()
        q, gyro, lin, acc = sim.imu()
        q = mount_imu(q, *self._mount)
        n_att = math.radians(float(g('imu_noise_deg')))
        if n_att > 0:
            dq = np.array([self._noise(n_att) / 2, self._noise(n_att) / 2,
                           self._noise(n_att) / 2, 1.0])
            x1, y1, z1, w1 = q
            x2, y2, z2, w2 = dq / np.linalg.norm(dq)
            q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                          w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
        ng, na = float(g('gyro_noise_rps')), float(g('accel_noise_mps2'))
        gyro = gyro + np.array([self._noise(ng) for _ in range(3)])
        nacc = np.array([self._noise(na) for _ in range(3)])
        for pub, a in ((self._pub_imu, lin + nacc), (self._pub_imu_raw, acc + nacc)):
            m = Imu()
            m.header.stamp = stamp
            m.header.frame_id = self._imu_frame
            m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = (
                float(v) for v in q)
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (
                float(v) for v in gyro)
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (
                float(v) for v in a)
            pub.publish(m)

        pos, qw = sim.base_pose()
        v_world, w_body = sim.base_velocity()

        gt = Odometry()
        gt.header.stamp = stamp
        gt.header.frame_id = self._odom
        gt.child_frame_id = self._base
        pp = gt.pose.pose
        pp.position.x, pp.position.y, pp.position.z = (float(v) for v in pos)
        pp.orientation.w, pp.orientation.x, pp.orientation.y, pp.orientation.z = (
            float(v) for v in qw)
        v_body = quat_to_mat(qw).T @ v_world
        tw = gt.twist.twist
        tw.linear.x, tw.linear.y, tw.linear.z = (float(v) for v in v_body)
        tw.angular.x, tw.angular.y, tw.angular.z = (float(v) for v in w_body)
        self._pub_gt.publish(gt)

        feet = Float64MultiArray()
        feet.layout.dim = [MultiArrayDimension(label='lf_rf_lb_rb_N', size=4, stride=4)]
        feet.data = [float(f) for f in sim.foot_forces()]
        self._pub_feet.publish(feet)

        if self._pub_js is not None:
            js = JointState()
            js.header.stamp = stamp
            js.name = self._names
            js.position = [float(v) for v in sim.joint_positions()]
            js.velocity = [float(v) for v in sim.joint_velocities()]
            self._pub_js.publish(js)
        if self._tf is not None:
            t = TransformStamped()
            t.header = gt.header
            t.child_frame_id = self._base
            t.transform.translation.x = pp.position.x
            t.transform.translation.y = pp.position.y
            t.transform.translation.z = pp.position.z
            t.transform.rotation = pp.orientation
            self._tf.sendTransform(t)

    def _on_calib(self):
        self._pub_cal.publish(String(data=json.dumps({'sys': 3, 'gyro': 3, 'accel': 3,
                                                      'mag': 0})))

    def close(self) -> str:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        return f'sim stopped at t={self.sim.time:.1f} s'


def main(args=None):
    return run_node(SimNode, 'sim', on_exit=lambda node: node.close(),
                    fatal=(ValueError, FileNotFoundError), args=args)


if __name__ == '__main__':
    sys.exit(main())
