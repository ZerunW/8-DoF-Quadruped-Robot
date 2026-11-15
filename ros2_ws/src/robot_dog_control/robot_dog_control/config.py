"""Load ``robot_dog_bringup/config/robot_dog.yaml`` without ROS parameters.

``$ROBOT_DOG_CALIBRATION`` overrides the file location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from robot_dog_control.controller import ControllerParams
from robot_dog_hw.calibration import CALIBRATION_ENV


def bringup_config_path(name: str = 'robot_dog.yaml') -> Path:
    """Installed share/robot_dog_bringup/config/<name>, or the source tree copy."""
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory('robot_dog_bringup')) / 'config' / name
        if p.is_file():
            return p
    except Exception:  # not built / not sourced
        pass
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / 'robot_dog_bringup' / 'config' / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f'cannot find robot_dog_bringup/config/{name}')


def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


def load_robot_config(path: Optional[Path] = None, with_calibration: bool = True,
                      calibration: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """{node_name: ros__parameters dict} from the config file.

    Reads ``path``, else ``$ROBOT_DOG_CALIBRATION``, else the bringup config.
    ``with_calibration`` and ``calibration`` are ignored.
    """
    import os
    env = os.environ.get(CALIBRATION_ENV)
    path = Path(path) if path else (Path(env).expanduser() if env else bringup_config_path())
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {node: dict((sec or {}).get('ros__parameters') or {}) for node, sec in data.items()}


def flatten(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Nested mappings -> dotted ROS parameter names (geom: {d_x: ..} -> geom.d_x)."""
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def controller_params_from_config(cfg: Dict[str, Dict[str, Any]]) -> ControllerParams:
    """ControllerParams from the gait_controller section + servo_driver limits."""
    flat = flatten(cfg.get('gait_controller', {}))
    drv = cfg.get('servo_driver', {})
    for key in ('servo_min', 'servo_max'):
        if key in drv:
            flat[key] = drv[key]
    return ControllerParams.from_flat(flat)


def stand_servo_angles(cfg: Dict[str, Dict[str, Any]]) -> list:
    """The controller's stand pose as 8 servo angles (deg, logical index order)."""
    from robot_dog_control.controller import build_components
    comp = build_components(controller_params_from_config(cfg))
    p, kin, lm = comp.params, comp.kin, comp.leg_map
    motor = []
    for i in range(4):
        x, y = lm.body_to_canonical(i, p.stand_fx_mm + p.leg_trim_x_mm[i],
                                    -p.stand_height_mm + p.leg_trim_mm[i])
        r = kin.ik(x, y, None)
        if not r.ok:
            raise ValueError(f'stand pose unreachable for leg {lm.legs[i].name}: {r.err_mm:.1f} mm')
        motor.append((r.m1, r.m2))
    return [round(a, 3) for a in lm.servo_vector(motor)]
