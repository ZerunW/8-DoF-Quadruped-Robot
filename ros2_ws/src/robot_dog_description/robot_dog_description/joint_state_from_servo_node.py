"""joint_state_from_servo: /servo_state -> /joint_states + TF odom -> base_link.

Publishes the commanded angles of all 24 leg joints (rad) for robot_state_publisher. Base
pose (``pose_source``): ``imu_dead_reckoning`` takes roll, pitch and yaw from the IMU, the
height from the stance feet's kinematics plus the foot radius, and x/y from the commanded
speed integrated along the yaw; ``sim_ground_truth`` takes the pose from /sim/ground_truth.
"""

from __future__ import annotations

import math
import sys
import time
from typing import List, Optional

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster

from robot_dog_control.config import controller_params_from_config, load_robot_config
from robot_dog_control.controller import DEBUG_INDEX
from robot_dog_control.kinematics import LegGeometry, LegKinematics
from robot_dog_control.leg_map import LEG_NAMES, LegMap
from robot_dog_description.leg_joints import LegJoints, all_joint_names
from robot_dog_description.model import BodyGeometry, config_mismatches, load_cad_frames
from robot_dog_hw.node_runner import run_node
from robot_dog_hw.orientation import quat_to_rpy, rpy_to_quat
from robot_dog_hw.ros_params import as_float, declare
from robot_dog_hw.servo_map import SERVO_LABELS

POSE_SOURCES = ('imu_dead_reckoning', 'sim_ground_truth')


class DeadReckoning:
    """Kinematic height and dead-reckoned x/y of the ``imu_dead_reckoning`` source (no ROS)."""

    def __init__(self, body: BodyGeometry, stance_tol_mm: float = 5.0):
        self.body = body
        self.stance_tol = stance_tol_mm * 1e-3
        self.x = self.y = 0.0
        self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0

    def feet_heights(self, feet_base) -> List[float]:
        """z of each foot center in a gravity-aligned frame at base_link (m)."""
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        out = []
        for x, y, z in feet_base:
            # R = Ry(pitch) Rx(roll); world z row = (-sp, cp sr, cp cr)
            out.append(-sp * x + cp * sr * y + cp * cr * z)
        return out

    def update_height(self, feet_base, stance: Optional[List[bool]] = None) -> float:
        h = self.feet_heights(feet_base)
        low = min(h)
        if stance is not None and any(stance):
            sel = [hz for hz, s in zip(h, stance) if s]
        else:
            sel = [hz for hz in h if hz - low <= self.stance_tol]
        self.z = -sum(sel) / len(sel) + self.body.foot_radius_m
        return self.z

    def integrate(self, v: float, dt: float) -> None:
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt


class JointStateFromServoNode(Node):

    def __init__(self, **kw):
        super().__init__('joint_state_from_servo', **kw)
        declare(self, 'pose_source', 'imu_dead_reckoning', 'imu_dead_reckoning | sim_ground_truth')
        declare(self, 'rate_hz', 50.0)
        declare(self, 'odom_frame', 'odom')
        declare(self, 'base_frame', 'base_link')
        declare(self, 'frame_prefix', '', 'prefix for base_frame (ghost model next to the sim)')
        declare(self, 'publish_tf', True)
        declare(self, 'imu_topic', '/bno055/imu')
        declare(self, 'stance_tol_mm', 5.0)
        declare(self, 'timeout_s', 0.5, '/gait/debug, /cmd_vel and IMU freshness')
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self._source = str(g('pose_source'))
        if self._source not in POSE_SOURCES:
            raise ValueError(f'pose_source must be one of {POSE_SOURCES}, got {self._source!r}')
        self._odom = str(g('odom_frame'))
        self._base = str(g('frame_prefix')) + str(g('base_frame'))
        self._tf_on = bool(g('publish_tf'))
        self._timeout = as_float(g('timeout_s'), 'timeout_s')
        rate = as_float(g('rate_hz'), 'rate_hz')

        cfg = load_robot_config()
        params = controller_params_from_config(cfg)
        self._leg_map = LegMap.from_params(params.leg_names, params.leg_thigh_idx,
                                           params.leg_calf_idx, params.leg_mirror_servo,
                                           params.leg_x_sign, params.leg_pos_x_mm,
                                           params.leg_pos_y_mm)
        self._kin = LegKinematics(LegGeometry.from_mapping(params.geom),
                                  region=params.joint_region())
        self._legj = LegJoints(self._kin)
        cad = load_cad_frames()
        self._body = BodyGeometry.from_cad(cad)
        for line in config_mismatches(cad, cfg.get('gait_controller', {})):
            self.get_logger().warn(f'robot_dog.yaml vs CAD: {line}')
        self._dr = DeadReckoning(self._body, as_float(g('stance_tol_mm'), 'stance_tol_mm'))
        self._names = all_joint_names()
        self._joints: Optional[List[float]] = None
        self._feet = None
        self._imu_t = -math.inf
        self._dbg = None
        self._dbg_t = -math.inf
        self._cmd_v = 0.0
        self._cmd_t = -math.inf
        self._gt: Optional[Odometry] = None
        self._last = None

        self._pub_js = self.create_publisher(JointState, 'joint_states', 10)
        self._tf = TransformBroadcaster(self) if self._tf_on else None
        self.create_subscription(JointState, '/servo_state', self._on_servo,
                                 qos_profile_sensor_data)
        if self._source == 'imu_dead_reckoning':
            self.create_subscription(Imu, str(g('imu_topic')), self._on_imu,
                                     qos_profile_sensor_data)
            self.create_subscription(Float64MultiArray, '/gait/debug', self._on_debug, 10)
            self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        else:
            self.create_subscription(Odometry, '/sim/ground_truth', self._on_gt, 10)
        self.create_timer(1.0 / rate, self._on_timer)
        self.get_logger().info(
            f'publishing {self._pub_js.topic_name} from /servo_state; '
            + (f'TF {self._odom} -> {self._base} from {self._source}' if self._tf_on
               else 'no TF'))

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
        joints, feet = [], []
        for i, (m1, m2) in enumerate(self._leg_map.motor_angles(servo)):
            joints.extend(self._legj.angles(m1, m2))
            xc, yc, _ = self._kin.fk(m1, m2)
            feet.append(self._body.foot_in_base(LEG_NAMES[i], xc, yc))
        self._joints, self._feet = joints, feet

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        try:
            self._dr.roll, self._dr.pitch, self._dr.yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
        except ValueError:
            return
        self._imu_t = time.monotonic()

    def _on_debug(self, msg: Float64MultiArray):
        if len(msg.data) > DEBUG_INDEX['rb_stance']:
            self._dbg = list(msg.data)
            self._dbg_t = time.monotonic()

    def _on_cmd_vel(self, msg: Twist):
        if math.isfinite(msg.linear.x):
            self._cmd_v = float(msg.linear.x)
            self._cmd_t = time.monotonic()

    def _on_gt(self, msg: Odometry):
        self._gt = msg

    # ---------------------------------------------------------------- output
    def _on_timer(self):
        if self._joints is None:
            return
        now = time.monotonic()
        dt = 0.0 if self._last is None else min(now - self._last, 0.2)
        self._last = now
        stamp = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.name = self._names
        js.position = [float(q) for q in self._joints]
        self._pub_js.publish(js)
        if not self._tf_on:
            return
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._odom
        t.child_frame_id = self._base
        if self._source == 'sim_ground_truth':
            if self._gt is None:
                return
            p, o = self._gt.pose.pose.position, self._gt.pose.pose.orientation
            t.transform.translation.x, t.transform.translation.y = p.x, p.y
            t.transform.translation.z = p.z
            t.transform.rotation = o
        else:
            dr = self._dr
            fresh_dbg = now - self._dbg_t <= self._timeout
            stance = None
            v = 0.0
            if fresh_dbg:
                d = self._dbg
                v = d[DEBUG_INDEX['v_cmd']]
                stance = [d[DEBUG_INDEX[f'{leg}_stance']] > 0.5 for leg in LEG_NAMES]
                if int(d[DEBUG_INDEX['state']]) != 1:      # not WALK: all feet down
                    stance = None
            elif now - self._cmd_t <= self._timeout:
                v = self._cmd_v
            if not math.isfinite(v):
                v = 0.0
            dr.update_height(self._feet, stance)
            dr.integrate(v, dt)
            x, y, z, w = rpy_to_quat(dr.roll, dr.pitch, dr.yaw)
            t.transform.translation.x, t.transform.translation.y = dr.x, dr.y
            t.transform.translation.z = dr.z
            t.transform.rotation.x, t.transform.rotation.y = x, y
            t.transform.rotation.z, t.transform.rotation.w = z, w
        self._tf.sendTransform(t)


def main(args=None):
    return run_node(JointStateFromServoNode, 'joint_state_from_servo',
                    fatal=(ValueError, FileNotFoundError), args=args)


if __name__ == '__main__':
    sys.exit(main())
