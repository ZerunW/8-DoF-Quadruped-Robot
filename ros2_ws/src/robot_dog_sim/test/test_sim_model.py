"""MuJoCo model: loads, matches the URDF and kinematics.fk, stands, and runs the controller
and the servo driver model in closed loop (robot_dog_sim.sil)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG.parent / 'robot_dog_description'))
sys.path.insert(0, str(_PKG))

mujoco = pytest.importorskip('mujoco')

from robot_dog_control.config import controller_params_from_config, load_robot_config  # noqa: E402
from robot_dog_control.controller import State  # noqa: E402
from robot_dog_control.leg_map import LEG_NAMES  # noqa: E402
from robot_dog_description.model import BodyGeometry  # noqa: E402
from robot_dog_sim.sil import ClosedLoop, stand_motor_angles  # noqa: E402
from robot_dog_sim.sim_core import DogSim, mjcf_path  # noqa: E402


def test_mjcf_loads_and_is_generated_from_the_current_urdf(tmp_path):
    sim = DogSim.from_config()
    m = sim.model
    assert sum(1 for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE) == 24
    assert m.nu == 8 and m.neq == 8 and sim.loop_gap() < 1e-6
    assert 1.2 < sim.total_mass < 1.8 and m.opt.timestep == pytest.approx(0.002)
    pytest.importorskip('xacro')
    sys.path.insert(0, str(_PKG / 'scripts'))
    import build_mjcf
    assert build_mjcf.build(tmp_path / 'robot_dog.xml') == mjcf_path().read_text(), \
        'mjcf/robot_dog.xml is stale: run `pixi run build_mjcf`'


@pytest.mark.parametrize('m1,m2', [(90.0, 90.0), (70.0, 95.0), (110.0, 65.0), (65.0, 70.0)])
def test_foot_site_matches_fk(m1, m2):
    """Motors held at (m1, m2) without gravity: the closed loops put the foot where
    kinematics.fk does."""
    sim = DogSim.from_config()
    body = BodyGeometry.from_cad()
    sim.model.opt.gravity[:] = 0.0
    sim.reset([(90.0, 90.0)] * 4)
    sim.data.qpos[sim.qadr_root + 2] += 0.3          # in the air
    sim.set_motor_targets([(m1, m2)] * 4)
    sim.step(int(1.5 / sim.dt))
    fx, fy, _ = sim.kin.fk(m1, m2)
    for i, leg in enumerate(LEG_NAMES):
        xc, yc = sim.foot_in_hip_frame(i, body.hips_m[leg], body.x_sign)
        assert math.hypot(xc - fx, yc - fy) < 2.0, leg
    assert sim.loop_gap() < 1e-4


def test_stands_holding_the_stand_pose():
    params = controller_params_from_config(load_robot_config())
    sim = DogSim.from_config()
    sim.reset(stand_motor_angles(params))
    nominal = params.stand_height_mm * 1e-3 + sim.foot_radius
    sim.step(int(5.0 / sim.dt))
    pos, _ = sim.base_pose()
    roll, pitch, _ = sim.rpy()
    assert abs(pos[2] - nominal) < 0.2 * nominal
    assert abs(math.degrees(roll)) < 5.0 and abs(math.degrees(pitch)) < 5.0
    assert np.all(sim.foot_forces() > 0.5)          # all four feet carry load


def test_closed_loop_levels_on_a_slope():
    pitch = {}
    for level in (False, True):
        loop = ClosedLoop.from_config({'level_enable': level})
        loop.sim.set_ground_tilt(0.0, math.radians(4.0))
        loop.sim.reset(loop.leg_map.motor_angles(loop.servo))
        loop.run(15.0)
        assert not loop.fallen()
        pitch[level] = math.degrees(loop.sim.rpy()[1])
    assert abs(pitch[False]) > 2.0 and abs(pitch[True]) < 1.0


def test_closed_loop_stands_up_and_walks_forward():
    loop = ClosedLoop.from_config()
    loop.run(4.0)
    pos, _ = loop.sim.base_pose()
    assert loop.core.state == State.STAND and not loop.fallen()
    assert abs(pos[2] - (loop.params.stand_height_mm * 1e-3 + loop.sim.foot_radius)) < 0.03
    loop.run(10.0, v=0.05)
    end, _ = loop.sim.base_pose()
    roll, pitch, _ = loop.sim.rpy()
    assert end[0] - pos[0] > 0.2 and abs(end[1] - pos[1]) < 0.15
    assert abs(math.degrees(roll)) < 5.0 and abs(math.degrees(pitch)) < 5.0
    assert loop.driver_clamps == 0 and not loop.fallen()
