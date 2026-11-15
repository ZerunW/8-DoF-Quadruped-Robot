"""Paths of the description files and the measured CAD frames, without ROS.

Works from the colcon install space (ament index) and from the source tree (tests, scripts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

PKG = 'robot_dog_description'
_SRC_DIR = Path(__file__).resolve().parents[1]


def package_dir() -> Path:
    """share/robot_dog_description if installed, else the source package directory."""
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory(PKG))
        if (p / 'config' / 'cad_frames.yaml').is_file():
            return p
    except Exception:  # not built / not sourced
        pass
    return _SRC_DIR


def cad_frames_path() -> Path:
    return package_dir() / 'config' / 'cad_frames.yaml'


def xacro_path() -> Path:
    return package_dir() / 'urdf' / 'robot_dog.urdf.xacro'


def mesh_dir() -> Path:
    return package_dir() / 'meshes'


def load_cad_frames(path: Optional[Path] = None) -> dict:
    with open(path or cad_frames_path()) as f:
        return yaml.safe_load(f)


def robot_description(mesh_uri: Optional[str] = None) -> str:
    """URDF XML string (runs xacro in-process)."""
    import xacro
    mappings = {'mesh_uri': mesh_uri} if mesh_uri else {}
    doc = xacro.process_file(str(xacro_path()), mappings=mappings)
    return doc.toprettyxml(indent='  ')


@dataclass(frozen=True)
class BodyGeometry:
    """Placement of the legs in base_link, measured in the CAD."""

    hips_m: Dict[str, Tuple[float, float, float]]   # thigh axis O per leg (m)
    x_sign: int                                      # +1 if canonical +x_c is body +X, else -1
    foot_radius_m: float                             # foot sphere radius around E (m)

    @classmethod
    def from_cad(cls, cad: Optional[dict] = None) -> 'BodyGeometry':
        cad = cad or load_cad_frames()
        hips = {leg: tuple(float(v) for v in d['hip_xyz']) for leg, d in cad['legs'].items()}
        return cls(hips, int(cad['canonical_frame']['x_sign']),
                   float(cad['collisions']['left_foot']['radius']))

    def foot_in_base(self, leg: str, x_c_mm: float, y_c_mm: float) -> Tuple[float, float, float]:
        """Foot point E (canonical mm) -> base_link (m)."""
        hx, hy, hz = self.hips_m[leg]
        return (hx + self.x_sign * x_c_mm * 1e-3, hy, hz + y_c_mm * 1e-3)


def config_mismatches(cad: dict, gait_params: dict, tol_mm: float = 2.0) -> list:
    """Human-readable differences between robot_dog.yaml (gait_controller) and the CAD."""
    out = []
    xs = gait_params.get('leg_x_sign')
    cad_x = int(cad['canonical_frame']['x_sign'])
    if xs is not None and any(int(v) != cad_x for v in xs):
        out.append(f'leg_x_sign {list(xs)} but the CAD says {[cad_x] * 4} (canonical +x_c = '
                   f'knee direction = body {"+X (forward)" if cad_x > 0 else "-X (backward)"})')
    hip = cad['hip_positions_mm']
    for key in ('leg_pos_x_mm', 'leg_pos_y_mm'):
        cfg = gait_params.get(key)
        if cfg is not None and any(abs(float(a) - float(b)) > tol_mm
                                   for a, b in zip(cfg, hip[key])):
            out.append(f'{key} {list(cfg)} but the CAD says {hip[key]}')
    return out


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
