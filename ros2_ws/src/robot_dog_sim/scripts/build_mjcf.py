#!/usr/bin/env python3
"""Generate mjcf/robot_dog.xml from the robot_dog_description URDF (committed output).

    pixi run build_mjcf

Writes the URDF tree as MuJoCo bodies and adds a free joint on base_link, two connect
constraints per leg closing the five-bar loops at D and B, position actuators on the motor
hinges (thigh = m1, rod = m2, ctrl = radians(m - 90)), IMU and foot touch sensors and a
ground plane. URDF cylinders become capsules; collision primitives only collide with the
ground.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

HERE = Path(__file__).resolve()
SIM_DIR = HERE.parents[1]
SRC_DIR = SIM_DIR.parent
DESC_DIR = SRC_DIR / 'robot_dog_description'
OUT = SIM_DIR / 'mjcf' / 'robot_dog.xml'
MESH_REL = '../../robot_dog_description/meshes'

LEGS = ('lf', 'rf', 'lb', 'rb')
MOTOR_SUFFIX = ('_thigh_joint', '_rod_joint')

# ------------------------------------------------------------------------------
# Servo and contact model defaults. sim_node can override the actuator gains at runtime
# (servo_kp, servo_kv, ...).
# ------------------------------------------------------------------------------
TIMESTEP = 0.002              # s (500 Hz physics)
SERVO_KP = 25.0               # N m / rad
SERVO_KV = 0.25               # N m s / rad
SERVO_TORQUE = 2.0            # N m, stall torque
SERVO_ARMATURE = 1.0e-3       # kg m^2, reflected rotor inertia
SERVO_FRICTION = 0.01         # N m gear friction
PASSIVE_DAMPING = 2.0e-3      # N m s / rad, printed pin joints
PASSIVE_ARMATURE = 1.0e-5
CTRL_RANGE = math.radians(105.0)  # m = -15..195 deg
FOOT_FRICTION = (1.0, 0.005, 0.0001)   # TPU foot on a hard floor
LOOP_SOLREF = (0.004, 1.0)    # loop closure: time constant (s), damping ratio
LOOP_SOLIMP = (0.95, 0.99, 0.001)


def fmt(v, nd=6) -> str:
    if isinstance(v, (list, tuple, np.ndarray)):
        return ' '.join(fmt(x, nd) for x in v)
    x = float(v)
    if abs(x) < 10 ** (-nd):
        x = 0.0
    s = f'{x:.{nd}g}' if abs(x) >= 1e-3 or x == 0.0 else f'{x:.{nd}e}'
    return s


def rpy_to_mat(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def mat_to_quat(R) -> List[float]:
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                (R[1, 0] - R[0, 1]) / (4 * w)]
    raise ValueError('rotation too close to 180 deg for this helper')


def vec3(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()])


def run_xacro() -> ET.Element:
    import xacro
    doc = xacro.process_file(str(DESC_DIR / 'urdf' / 'robot_dog.urdf.xacro'))
    return ET.fromstring(doc.toxml())


def build(out: Path = OUT) -> str:
    urdf = run_xacro()
    cad = yaml.safe_load((DESC_DIR / 'config' / 'cad_frames.yaml').read_text())
    x_sign = int(cad['canonical_frame']['x_sign'])
    zero = {k: np.array(v) for k, v in cad['canonical_frame']['zero_pose_points_mm'].items()}

    def canon_vec(c) -> np.ndarray:
        return np.array([x_sign * c[0], 0.0, c[1]]) * 1e-3

    links = {link.get('name'): link for link in urdf.findall('link')}
    materials = {m.get('name'): m.find('color').get('rgba')
                 for m in urdf.findall('material') if m.find('color') is not None}
    children: Dict[str, List[ET.Element]] = {}
    for j in urdf.findall('joint'):
        children.setdefault(j.find('parent').get('link'), []).append(j)

    mj = ET.Element('mujoco', model='robot_dog')
    ET.SubElement(mj, 'compiler', angle='radian', meshdir=MESH_REL, autolimits='true',
                  inertiafromgeom='false')
    ET.SubElement(mj, 'option', timestep=fmt(TIMESTEP), integrator='implicitfast',
                  cone='elliptic', impratio='10')
    vis = ET.SubElement(mj, 'visual')
    ET.SubElement(vis, 'global', offwidth='1600', offheight='1200', elevation='-20',
                  azimuth='120')
    ET.SubElement(vis, 'quality', shadowsize='4096')
    ET.SubElement(vis, 'headlight', ambient='0.35 0.35 0.35', diffuse='0.5 0.5 0.5',
                  specular='0.1 0.1 0.1')
    ET.SubElement(vis, 'map', znear='0.01')

    default = ET.SubElement(mj, 'default')
    d = ET.SubElement(default, 'default', {'class': 'visual'})
    ET.SubElement(d, 'geom', type='mesh', contype='0', conaffinity='0', group='2')
    d = ET.SubElement(default, 'default', {'class': 'collision'})
    ET.SubElement(d, 'geom', contype='2', conaffinity='1', group='3', rgba='0.9 0.2 0.2 0.4')
    d2 = ET.SubElement(d, 'default', {'class': 'foot'})
    ET.SubElement(d2, 'geom', friction=fmt(FOOT_FRICTION), condim='3', priority='1')
    d = ET.SubElement(default, 'default', {'class': 'motor'})
    ET.SubElement(d, 'joint', type='hinge', armature=fmt(SERVO_ARMATURE),
                  frictionloss=fmt(SERVO_FRICTION), damping='0')
    ET.SubElement(d, 'position', kp=fmt(SERVO_KP), kv=fmt(SERVO_KV),
                  forcerange=fmt([-SERVO_TORQUE, SERVO_TORQUE]),
                  ctrlrange=fmt([-CTRL_RANGE, CTRL_RANGE]))
    d = ET.SubElement(default, 'default', {'class': 'passive'})
    ET.SubElement(d, 'joint', type='hinge', armature=fmt(PASSIVE_ARMATURE),
                  damping=fmt(PASSIVE_DAMPING))
    d = ET.SubElement(default, 'default', {'class': 'loop'})
    ET.SubElement(d, 'equality', solref=fmt(LOOP_SOLREF), solimp=fmt(LOOP_SOLIMP))

    asset = ET.SubElement(mj, 'asset')
    ET.SubElement(asset, 'texture', name='grid', type='2d', builtin='checker',
                  rgb1='0.82 0.84 0.86', rgb2='0.70 0.72 0.75', width='512', height='512',
                  mark='edge', markrgb='0.6 0.6 0.6')
    ET.SubElement(asset, 'material', name='grid', texture='grid', texrepeat='8 8',
                  texuniform='true', reflectance='0.05')
    ET.SubElement(asset, 'texture', name='sky', type='skybox', builtin='gradient',
                  rgb1='0.92 0.94 0.97', rgb2='0.75 0.8 0.86', width='256', height='256')
    meshes_used = set()
    for link in links.values():
        for v in link.findall('visual'):
            m = v.find('geometry/mesh')
            if m is not None:
                meshes_used.add(Path(m.get('filename')).name)
    for name in sorted(meshes_used):
        ET.SubElement(asset, 'mesh', name=Path(name).stem, file=name)

    world = ET.SubElement(mj, 'worldbody')
    ET.SubElement(world, 'light', name='sun', pos='0.3 -0.6 1.5', dir='-0.2 0.4 -1',
                  diffuse='0.7 0.7 0.7', castshadow='true')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 0.05',
                  material='grid', contype='1', conaffinity='2', condim='3',
                  friction=fmt(FOOT_FRICTION))
    foot_r = float(cad['collisions']['left_foot']['radius'])
    e_z = zero['E'][1] * 1e-3
    base = ET.SubElement(world, 'body', name='base_link', pos=fmt([0, 0, foot_r - e_z + 0.002]))
    ET.SubElement(base, 'freejoint', name='root')
    ET.SubElement(base, 'site', name='imu', pos='0 0 0', size='0.008', rgba='0.1 0.8 0.1 1',
                  group='5')
    # payload body; the sim sets its mass and position at runtime
    payload = ET.SubElement(base, 'body', name='payload', pos='0 0 0')
    ET.SubElement(payload, 'inertial', pos='0 0 0', mass='1e-6', diaginertia='1e-10 1e-10 1e-10')
    ET.SubElement(payload, 'geom', {'class': 'visual', 'type': 'box', 'size': '0.03 0.02 0.01',
                                    'rgba': '0.9 0.8 0.2 0', 'name': 'payload_visual'})

    def identity_fixed_child(link_name):
        """Fixed child in the same frame, such as base_link -> body, or None."""
        for j in children.get(link_name, []):
            o = j.find('origin')
            if (j.get('type') == 'fixed' and o is not None
                    and np.allclose(vec3(o.get('xyz')), 0)
                    and np.allclose(vec3(o.get('rpy') or '0 0 0'), 0)
                    and links[j.find('child').get('link')].find('inertial') is not None):
                return j
        return None

    def add_link(body: ET.Element, link_name: str, skip=()):
        link = links[link_name]
        inert = link.find('inertial')
        merged = None
        if inert is None:
            # massless frame: fold its same-frame child into it
            merged = identity_fixed_child(link_name)
            if merged is not None:
                child = merged.find('child').get('link')
                inert = links[child].find('inertial')
                for el in links[child].findall('visual') + links[child].findall('collision'):
                    link.append(el)
        if inert is not None:
            o = inert.find('origin')
            xyz = vec3(o.get('xyz')) if o is not None else np.zeros(3)
            ie = inert.find('inertia')
            full = [float(ie.get(k)) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')]
            ET.SubElement(body, 'inertial', pos=fmt(xyz), mass=fmt(float(
                inert.find('mass').get('value'))), fullinertia=fmt(full, 8))
        for v in link.findall('visual'):
            m = v.find('geometry/mesh')
            if m is None:
                continue
            mat = v.find('material')
            rgba = materials.get(mat.get('name')) if mat is not None else None
            mesh = Path(m.get('filename')).stem
            g = ET.SubElement(body, 'geom', {'class': 'visual', 'mesh': mesh,
                                             'name': f'{link_name}_{mesh}_visual'})
            if rgba:
                g.set('rgba', rgba)
        for i, c in enumerate(link.findall('collision')):
            o = c.find('origin')
            xyz = vec3(o.get('xyz')) if o is not None else np.zeros(3)
            rpy = vec3(o.get('rpy')) if o is not None and o.get('rpy') else np.zeros(3)
            R = rpy_to_mat(*rpy)
            geo = c.find('geometry')[0]
            name = f'{link_name}_collision' + (f'{i}' if i else '')
            if geo.tag == 'cylinder':
                half = 0.5 * float(geo.get('length'))
                a = xyz - R @ np.array([0, 0, half])
                b = xyz + R @ np.array([0, 0, half])
                ET.SubElement(body, 'geom', {'class': 'collision', 'name': name,
                                             'type': 'capsule', 'fromto': fmt(list(a) + list(b)),
                                             'size': fmt(float(geo.get('radius')))})
            elif geo.tag == 'sphere':
                g = ET.SubElement(body, 'geom', {'class': 'foot' if link_name.endswith('_foot')
                                                 else 'collision', 'name': name, 'type': 'sphere',
                                                 'pos': fmt(xyz),
                                                 'size': fmt(float(geo.get('radius')))})
            elif geo.tag == 'box':
                ET.SubElement(body, 'geom', {'class': 'collision', 'name': name, 'type': 'box',
                                             'pos': fmt(xyz), 'quat': fmt(mat_to_quat(R)),
                                             'size': fmt(0.5 * vec3(geo.get('size')))})
        if link_name.endswith('_foot'):
            # touch-sensor volume around the foot sphere; group 5 is hidden by default
            ET.SubElement(body, 'site', name=f'{link_name}_site', pos='0 0 0', size='0.021',
                          rgba='1 0.2 0.2 0.3', group='5')
        for j in sorted(children.get(link_name, []), key=lambda e: e.get('name')):
            if j is merged:
                continue
            child = j.find('child').get('link')
            o = j.find('origin')
            xyz = vec3(o.get('xyz')) if o is not None else np.zeros(3)
            rpy = vec3(o.get('rpy')) if o is not None and o.get('rpy') else np.zeros(3)
            attrs = {'name': child, 'pos': fmt(xyz)}
            if np.any(np.abs(rpy) > 1e-12):
                attrs['quat'] = fmt(mat_to_quat(rpy_to_mat(*rpy)))
            cb = ET.SubElement(body, 'body', attrs)
            if j.get('type') in ('revolute', 'continuous'):
                jn = j.get('name')
                cls = 'motor' if jn.endswith(MOTOR_SUFFIX) else 'passive'
                ET.SubElement(cb, 'joint', {'name': jn, 'class': cls,
                                            'axis': fmt(vec3(j.find('axis').get('xyz')))})
            elif j.get('type') != 'fixed':
                raise ValueError(f'unsupported joint type {j.get("type")}')
            add_link(cb, child)

    add_link(base, 'base_link')

    # hidden marker sites (group 4) at the loop joints, on both sides of each cut
    bodies = {b.get('name'): b for b in mj.iter('body')}
    marks = (('cd_link', 'D', 'C', 'fan', 'O'), ('ab_link', 'B', 'A', 'calf', 'K'))
    for leg in LEGS:
        for b1, P, J1, b2, J2 in marks:
            ET.SubElement(bodies[f'{leg}_{b1}'], 'site', name=f'{leg}_{P}_{b1}',
                          pos=fmt(canon_vec(zero[P] - zero[J1])), size='0.003',
                          rgba='1 0.1 0.1 1', group='4')
            ET.SubElement(bodies[f'{leg}_{b2}'], 'site', name=f'{leg}_{P}_{b2}',
                          pos=fmt(canon_vec(zero[P] - zero[J2])), size='0.0045',
                          rgba='0.1 0.4 1 0.6', group='4')

    # loop closures, anchored in body1's frame; joint frames are body-aligned at qpos0
    eq = ET.SubElement(mj, 'equality')
    for leg in LEGS:
        ET.SubElement(eq, 'connect', {'class': 'loop', 'name': f'{leg}_loop_D',
                                      'body1': f'{leg}_cd_link', 'body2': f'{leg}_fan',
                                      'anchor': fmt(canon_vec(zero['D'] - zero['C']))})
        ET.SubElement(eq, 'connect', {'class': 'loop', 'name': f'{leg}_loop_B',
                                      'body1': f'{leg}_ab_link', 'body2': f'{leg}_calf',
                                      'anchor': fmt(canon_vec(zero['B'] - zero['A']))})
    act = ET.SubElement(mj, 'actuator')
    for leg in LEGS:
        ET.SubElement(act, 'position', {'class': 'motor', 'name': f'{leg}_m1',
                                        'joint': f'{leg}_thigh_joint'})
        ET.SubElement(act, 'position', {'class': 'motor', 'name': f'{leg}_m2',
                                        'joint': f'{leg}_rod_joint'})
    sens = ET.SubElement(mj, 'sensor')
    ET.SubElement(sens, 'framequat', name='imu_quat', objtype='site', objname='imu')
    ET.SubElement(sens, 'gyro', name='imu_gyro', site='imu')
    ET.SubElement(sens, 'accelerometer', name='imu_acc', site='imu')
    ET.SubElement(sens, 'framepos', name='base_pos', objtype='site', objname='imu')
    for leg in LEGS:
        ET.SubElement(sens, 'touch', name=f'{leg}_touch', site=f'{leg}_foot_site')

    ET.indent(mj, space='  ')
    header = ('<!-- GENERATED by src/robot_dog_sim/scripts/build_mjcf.py from the '
              'robot_dog_description URDF (pixi run build_mjcf). Do not edit by hand. -->\n')
    text = header + ET.tostring(mj, encoding='unicode') + '\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text


def check(path: Path = OUT) -> None:
    """Load with MuJoCo and report the mass and the loop-closure gap at qpos0."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    total = sum(m.body_mass)
    print(f'MJCF: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, {m.neq} loop '
          f'constraints, total mass {total:.3f} kg')
    viol = 0.0
    for i in range(m.neq):
        b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
        a1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[i, 0:3]
        a2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ m.eq_data[i, 3:6]
        viol = max(viol, float(np.linalg.norm(a1 - a2)))
    print(f'max loop-closure gap at qpos0: {viol * 1e3:.4f} mm')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--no-check', action='store_true')
    args = ap.parse_args(argv)
    build(args.out)
    print(f'wrote {args.out}')
    if not args.no_check:
        check(args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
