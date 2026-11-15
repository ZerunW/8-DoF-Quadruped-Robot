"""Software in the loop without ROS, on simulated time: GaitControllerCore -> servo driver
clamp and rate limiter -> MuJoCo.

    loop = ClosedLoop.from_config()
    loop.run(3.0)                          # stand-up blend + settle
    loop.run(10.0, v=0.05)                 # walk
    print(loop.sim.base_pose()[0])
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from robot_dog_control.config import controller_params_from_config, load_robot_config
from robot_dog_control.controller import ControllerParams, GaitControllerCore, ImuSample
from robot_dog_hw.servo_map import RateLimiter, ServoConfig, ServoLimiter

from robot_dog_sim.sim_core import DogSim, ServoParams, mount_imu


class ClosedLoop:

    def __init__(self, params: ControllerParams, driver_cfg: dict,
                 sim: Optional[DogSim] = None, servo: Optional[ServoParams] = None):
        self.params = params
        self.core = GaitControllerCore(params)
        self.leg_map = self.core.leg_map
        speed = float(driver_cfg.get('max_servo_speed_dps', 400.0))
        self.sim = sim or DogSim(self.core.kin, servo=servo or ServoParams(speed_dps=speed))
        cfg = ServoConfig(channels=driver_cfg.get('servo_channel', ServoConfig().channels),
                          offsets=driver_cfg.get('servo_offset', ServoConfig().offsets),
                          mins=params.servo_min, maxs=params.servo_max,
                          stand=driver_cfg.get('stand_angles', ServoConfig().stand))
        self.limiter = ServoLimiter(cfg)
        self.rate = RateLimiter(speed)
        self.servo: List[float] = [float(a) for a in cfg.stand]
        self.rate.reset(self.servo)
        self.dt = 1.0 / params.rate_hz
        self.substeps = max(1, int(round(self.dt / self.sim.dt)))
        self.sim.reset(self.leg_map.motor_angles(self.servo))
        self.log: List[Dict[str, float]] = []
        self.driver_clamps = 0          # servo-ticks clamped by the driver's servo limits
        self.requested_step = 0.0       # largest |target - servo| of the last tick (deg)
        self.rate_limited_ticks = 0     # ticks in which the driver's rate limit cut a command

    @classmethod
    def from_config(cls, overrides: Optional[Dict] = None, **kw) -> 'ClosedLoop':
        """Parameters from robot_dog.yaml, as the launch files load them, plus overrides."""
        cfg = load_robot_config()
        params = controller_params_from_config(cfg)
        if overrides:
            params = ControllerParams.from_flat(overrides, base=params)
        return cls(params, cfg.get('servo_driver', {}), **kw)

    def imu_sample(self) -> ImuSample:
        q, gyro, lin, _ = self.sim.imu()
        from robot_dog_hw.orientation import quat_to_rpy
        # IMU board tilted in the body by the level targets
        q = mount_imu(q, math.radians(self.params.level_target_roll_deg),
                      math.radians(self.params.level_target_pitch_deg))
        roll, pitch, yaw = quat_to_rpy(q[0], q[1], q[2], q[3])
        return ImuSample(roll, pitch, yaw, float(gyro[0]), float(gyro[1]), float(gyro[2]),
                         float(lin[1]), self.sim.time)

    def tick(self, v: float = 0.0, w: float = 0.0, step: bool = False) -> None:
        t = self.sim.time
        core = self.core
        core.set_imu(self.imu_sample())
        core.set_measured_pose(self.servo)
        core.set_cmd_vel(v, w, t)              # sent every tick, zeros included, like a client
        core.set_cmd_step(step, t)
        out = core.step(t)
        self.out = out
        if out is not None:
            target, mask = self.limiter.clamp(out.servo)
            self.driver_clamps += sum(1 for m in mask if m)
            self.requested_step = max(abs(t - s) for t, s in zip(target, self.servo))
            self.servo = self.rate.step(target, self.dt)
            if any(abs(a - t) > 1e-9 for a, t in zip(self.servo, target)):
                self.rate_limited_ticks += 1
        self.sim.set_targets_from_servo(self.leg_map, self.servo)
        self.sim.step(self.substeps)

    def run(self, duration: float, v: float = 0.0, w: float = 0.0, record: bool = False,
            step: bool = False) -> None:
        n = int(round(duration / self.dt))
        for _ in range(n):
            self.tick(v, w, step)
            if record:
                self.record()

    def record(self) -> None:
        p, _ = self.sim.base_pose()
        r, pi, y = self.sim.rpy()
        self.log.append({'t': self.sim.time, 'x': float(p[0]), 'y': float(p[1]),
                         'z': float(p[2]), 'roll': math.degrees(r), 'pitch': math.degrees(pi),
                         'yaw': math.degrees(y), 'state': float(self.core.state)})

    def fallen(self, max_tilt_deg: float = 45.0) -> bool:
        r, p, _ = self.sim.rpy()
        return abs(math.degrees(r)) > max_tilt_deg or abs(math.degrees(p)) > max_tilt_deg


def stand_motor_angles(params: ControllerParams) -> List[Sequence[float]]:
    """IK motor angles of the controller's stand pose, per leg."""
    core = GaitControllerCore(replace(params))
    out = []
    for i in range(4):
        xc, yc = core.leg_map.body_to_canonical(i, params.stand_fx_mm,
                                                -params.stand_height_mm + params.leg_trim_mm[i])
        r = core.kin.ik(xc, yc)
        out.append((r.m1, r.m2))
    return out
