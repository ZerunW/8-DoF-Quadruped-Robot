"""MuJoCo model of the robot dog without ROS, used by sim_node, the SIL harness and the tests.

    sim = DogSim.from_config()                    # robot_dog.yaml geometry + mjcf
    sim.reset(stand_motor_deg)                    # IK motor angles per leg (deg)
    sim.set_motor_targets(motor_deg)              # or set_targets_from_servo(servo_deg)
    sim.step(10)                                  # 10 x 2 ms
    q_xyzw, gyro, lin_acc, acc = sim.imu()

Motor hinges are position servos with a slewed set-point; motor joint angle
q = radians(m - 90) on every leg. Frames: world = odom (z up, x = heading at reset),
base_link = REP-103 body frame; IMU outputs are in base_link like the BNO055 driver's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from robot_dog_control.kinematics import LegGeometry, LegKinematics
from robot_dog_control.leg_map import LEG_NAMES, LegMap
from robot_dog_description.leg_joints import JOINTS, LegJoints, all_joint_names
from robot_dog_hw.orientation import quat_to_rpy

GRAVITY = 9.81
PKG = 'robot_dog_sim'


def mjcf_path() -> Path:
    """share/robot_dog_sim/mjcf/robot_dog.xml if installed, else the source copy."""
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory(PKG)) / 'mjcf' / 'robot_dog.xml'
        if p.is_file():
            return p
    except Exception:  # workspace not built or not sourced, e.g. unit tests
        pass
    return Path(__file__).resolve().parents[1] / 'mjcf' / 'robot_dog.xml'


def load_model(path: Optional[Path] = None) -> mujoco.MjModel:
    """Compile the MJCF with its meshdir set to robot_dog_description's mesh directory."""
    from robot_dog_description.model import mesh_dir
    spec = mujoco.MjSpec.from_file(str(path or mjcf_path()))
    spec.meshdir = str(mesh_dir())
    return spec.compile()


def quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def mount_imu(q_xyzw: Sequence[float], roll: float, pitch: float) -> np.ndarray:
    """Body orientation (x, y, z, w) -> orientation reported by an IMU board tilted in the
    body by (roll, pitch) rad: q_imu = q_body * q_mount."""
    x, y, z, w = (float(v) for v in q_xyzw)
    q = quat_mul(np.array([w, x, y, z]), quat_wxyz_from_rpy(roll, pitch, 0.0))
    return np.array([q[1], q[2], q[3], q[0]])


def quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q_wxyz, float))
    return m.reshape(3, 3)


#: servo driver rate limit (deg/s), servo_driver.max_servo_speed_dps in robot_dog.yaml
DRIVER_SPEED_DPS = 400.0


@dataclass
class ServoParams:
    """Hobby-servo approximation, per motor hinge."""

    kp: float = 25.0           # N m / rad
    kv: float = 0.25           # N m s / rad
    torque_nm: float = 2.0     # N m, symmetric limit
    speed_dps: float = DRIVER_SPEED_DPS  # set-point slew limit, deg/s

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> 'ServoParams':
        """Defaults, with the slew limit from servo_driver in robot_dog.yaml."""
        if cfg is None:
            from robot_dog_control.config import load_robot_config
            cfg = load_robot_config()
        drv = cfg.get('servo_driver', {})
        return cls(speed_dps=float(drv.get('max_servo_speed_dps', DRIVER_SPEED_DPS)))


class DogSim:
    """The MuJoCo model plus servo set-points, disturbances, ground tilt and payload."""

    def __init__(self, kin: LegKinematics, mjcf: Optional[Path] = None,
                 servo: Optional[ServoParams] = None):
        self.model = load_model(mjcf)
        self.data = mujoco.MjData(self.model)
        self.kin = kin
        self.legj = LegJoints(kin)
        m = self.model
        self.dt = float(m.opt.timestep)
        self.base_id = m.body('base_link').id
        self.payload_id = m.body('payload').id
        self.floor_id = m.geom('floor').id
        root = m.joint('root').id
        self.qadr_root = int(m.jnt_qposadr[root])
        self.vadr_root = int(m.jnt_dofadr[root])
        self.joint_names = all_joint_names()
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.joint_names])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.joint_names])
        self.act = np.array([[m.actuator(f'{leg}_m1').id, m.actuator(f'{leg}_m2').id]
                             for leg in LEG_NAMES])
        self._sens = {}
        for name in ('imu_quat', 'imu_gyro', 'imu_acc', 'base_pos') + tuple(
                f'{leg}_touch' for leg in LEG_NAMES):
            s = m.sensor(name)
            self._sens[name] = (int(m.sensor_adr[s.id]), int(m.sensor_dim[s.id]))
        self.foot_site = [m.site(f'{leg}_foot_site').id for leg in LEG_NAMES]
        self.foot_radius = float(m.geom_size[m.geom('lf_foot_collision').id][0])
        self.target = np.zeros((4, 2))      # motor set-points (rad), per leg (m1, m2)
        self.ctrl = np.zeros((4, 2))        # slewed set-points sent to MuJoCo
        self.servo = servo or ServoParams()
        self.enabled = True
        self.ground_rp = (0.0, 0.0)
        self.payload = (0.0, (0.0, 0.0, 0.0))
        self._wrench: Optional[Tuple[np.ndarray, float]] = None
        self.set_servo_params(self.servo)

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, **kw) -> 'DogSim':
        """Geometry from robot_dog.yaml (gait_controller.geom)."""
        if cfg is None:
            from robot_dog_control.config import load_robot_config
            cfg = load_robot_config()
        gc = cfg.get('gait_controller', {})
        geom = LegGeometry.from_mapping(gc['geom']) if gc.get('geom') else LegGeometry()
        return cls(LegKinematics(geom), **kw)

    # ------------------------------------------------------------- parameters
    def set_servo_params(self, p: ServoParams) -> None:
        self.servo = p
        m = self.model
        for a in self.act.ravel():
            m.actuator_gainprm[a, 0] = p.kp if self.enabled else 0.0
            m.actuator_biasprm[a, 1] = -p.kp if self.enabled else 0.0
            m.actuator_biasprm[a, 2] = -p.kv if self.enabled else 0.0
            m.actuator_forcerange[a] = (-p.torque_nm, p.torque_nm)

    def set_enabled(self, enabled: bool) -> None:
        """False = servo outputs off (relax): the motor hinges go limp."""
        self.enabled = bool(enabled)
        self.set_servo_params(self.servo)

    def set_ground_tilt(self, roll: float, pitch: float) -> None:
        """Tilt the floor plane (rad). + roll: right side lower, + pitch: nose lower."""
        self.ground_rp = (float(roll), float(pitch))
        self.model.geom_quat[self.floor_id] = quat_wxyz_from_rpy(roll, pitch, 0.0)
        # recompute the world-body geom frames cached in the model
        self._set_const()

    def _set_const(self) -> None:
        """Recompute static geom frames and mass-dependent constants, keeping the state."""
        m, d = self.model, self.data
        state = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.act.copy(), d.time)
        mujoco.mj_setConst(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:], d.act[:], d.time = state
        mujoco.mj_forward(m, d)

    def set_payload(self, mass_kg: float, offset_m: Sequence[float]) -> None:
        """Extra mass, a 6 x 4 x 2 cm box, fixed to base_link at ``offset_m``."""
        m = self.model
        mass = max(float(mass_kg), 1e-6)
        a, b, c = 0.06, 0.04, 0.02
        m.body_mass[self.payload_id] = mass
        m.body_pos[self.payload_id] = np.asarray(offset_m, float)
        m.body_inertia[self.payload_id] = (mass * (b * b + c * c) / 12,
                                           mass * (a * a + c * c) / 12,
                                           mass * (a * a + b * b) / 12)
        m.geom_rgba[m.geom('payload_visual').id][3] = 0.8 if mass_kg > 1e-3 else 0.0
        self.payload = (float(mass_kg), tuple(float(v) for v in offset_m))
        self._set_const()

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.model.body_mass))

    # ------------------------------------------------------------------ state
    def reset(self, motor_deg: Sequence[Tuple[float, float]], x: float = 0.0, y: float = 0.0,
              yaw: float = 0.0, clearance: float = 0.0005) -> None:
        """Pose every leg at (m1, m2) [deg], body parallel to the floor, feet on it."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for i in range(4):
            q = self.legj.angles(*motor_deg[i])
            for k in range(len(JOINTS)):
                d.qpos[self.qadr[i * len(JOINTS) + k]] = q[k]
            self.target[i] = LegJoints.motor_joints_from_deg(*motor_deg[i])
        self.ctrl[:] = self.target
        d.ctrl[self.act.ravel()] = self.ctrl.ravel()
        q_ground = quat_wxyz_from_rpy(self.ground_rp[0], self.ground_rp[1], 0.0)
        q_base = quat_mul(q_ground, quat_wxyz_from_rpy(0.0, 0.0, yaw))
        r = self.qadr_root
        d.qpos[r:r + 3] = (x, y, 0.0)
        d.qpos[r + 3:r + 7] = q_base
        mujoco.mj_kinematics(m, d)
        n = quat_to_mat(q_ground)[:, 2]           # floor normal; the floor contains the origin
        lift = max((self.foot_radius - float(n @ d.site_xpos[s])) / n[2]
                   for s in self.foot_site) + clearance
        d.qpos[r + 2] += lift
        self._wrench = None
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)

    def set_motor_targets(self, motor_deg: Sequence[Tuple[float, float]]) -> None:
        for i in range(4):
            self.target[i] = LegJoints.motor_joints_from_deg(*motor_deg[i])

    def set_targets_from_servo(self, leg_map: LegMap, servo_deg: Sequence[float]) -> None:
        """8 servo angles (logical index order, servo deg) -> motor set-points."""
        self.set_motor_targets(leg_map.motor_angles(servo_deg))

    def apply_wrench(self, force: Sequence[float], torque: Sequence[float],
                     duration: float) -> None:
        """World-frame force (N) and torque (N m) on base_link's center of mass."""
        w = np.concatenate([np.asarray(force, float), np.asarray(torque, float)])
        self._wrench = (w, self.data.time + max(0.0, float(duration)))

    @property
    def wrench_active(self) -> bool:
        return self._wrench is not None and self.data.time < self._wrench[1]

    def step(self, n: int = 1) -> None:
        m, d = self.model, self.data
        vmax = math.radians(self.servo.speed_dps) * self.dt
        idx = self.act.ravel()
        for _ in range(int(n)):
            self.ctrl += np.clip(self.target - self.ctrl, -vmax, vmax)
            d.ctrl[idx] = self.ctrl.ravel()
            if self._wrench is not None:
                if d.time < self._wrench[1]:
                    d.xfrc_applied[self.base_id] = self._wrench[0]
                else:
                    d.xfrc_applied[self.base_id] = 0.0
                    self._wrench = None
            mujoco.mj_step(m, d)

    # ---------------------------------------------------------------- outputs
    @property
    def time(self) -> float:
        return float(self.data.time)

    def _sensor(self, name: str) -> np.ndarray:
        adr, dim = self._sens[name]
        return self.data.sensordata[adr:adr + dim].copy()

    def base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Position (m) and orientation quaternion (w, x, y, z) of base_link in world."""
        r = self.qadr_root
        return self.data.qpos[r:r + 3].copy(), self.data.qpos[r + 3:r + 7].copy()

    def base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Linear velocity (world frame) and angular velocity (body frame)."""
        v = self.vadr_root
        return self.data.qvel[v:v + 3].copy(), self.data.qvel[v + 3:v + 6].copy()

    def rpy(self) -> Tuple[float, float, float]:
        _, q = self.base_pose()
        return quat_to_rpy(q[1], q[2], q[3], q[0])

    def imu(self):
        """(quat xyzw, gyro rad/s, linear acc without gravity, raw specific force)."""
        qw = self._sensor('imu_quat')
        gyro = self._sensor('imu_gyro')
        acc = self._sensor('imu_acc')
        R = quat_to_mat(qw)
        lin = acc - R.T @ np.array([0.0, 0.0, GRAVITY])
        return np.array([qw[1], qw[2], qw[3], qw[0]]), gyro, lin, acc

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.vadr].copy()

    def foot_positions(self) -> np.ndarray:
        """Foot centers E (world, m), legs in LEG_NAMES order."""
        return np.array([self.data.site_xpos[s] for s in self.foot_site])

    def foot_forces(self) -> np.ndarray:
        """Normal contact force per foot (N)."""
        return np.array([self._sensor(f'{leg}_touch')[0] for leg in LEG_NAMES])

    def motor_angles_deg(self) -> List[Tuple[float, float]]:
        """Measured (m1, m2) per leg from the motor hinges."""
        q = self.joint_positions().reshape(4, len(JOINTS))
        i1, i2 = JOINTS.index('thigh'), JOINTS.index('rod')
        return [LegJoints.motor_deg_from_joints(q[i, i1], q[i, i2]) for i in range(4)]

    def loop_gap(self) -> float:
        """Largest loop-closure gap (m) over the 8 connect constraints."""
        m, d = self.model, self.data
        gap = 0.0
        for i in range(m.neq):
            b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
            a1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[i, 0:3]
            a2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[i, 3:6]
            gap = max(gap, float(np.linalg.norm(a1 - a2)))
        return gap

    def foot_in_hip_frame(self, i: int, hip_m: Sequence[float], x_sign: int
                          ) -> Tuple[float, float]:
        """Foot E of leg i in its canonical leg frame (mm) from the body-frame pose."""
        p, q = self.base_pose()
        R = quat_to_mat(q)
        e = R.T @ (self.data.site_xpos[self.foot_site[i]] - p)
        return x_sign * (e[0] - hip_m[0]) * 1e3, (e[2] - hip_m[2]) * 1e3
