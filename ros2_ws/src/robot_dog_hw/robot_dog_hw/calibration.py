"""In-place editing of the config file ``robot_dog_bringup/config/robot_dog.yaml``, without ROS.

The calibration tools servo_tune and calib_offsets run on the Pi and call
``update_calibration``. It edits the keys in place in the workspace copy, keeping comments
and layout, and prints a line

    ROBOT_DOG_CONFIG_UPDATE {"node": "...", "values": {...}}

which the deploy terminal applies to the laptop's copy with the same function; the next
deploy pushes that file to the Pi. The tools may only write ALLOWED_KEYS.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CALIBRATION_ENV = 'ROBOT_DOG_CALIBRATION'
CONFIG_FILE = 'robot_dog.yaml'
CONFIG_UPDATE_MARKER = 'ROBOT_DOG_CONFIG_UPDATE'
ALLOWED_KEYS = {
    'servo_driver': {'servo_offset'},
    'gait_controller': {'leg_trim_mm', 'leg_trim_x_mm', 'level_target_roll_deg',
                        'level_target_pitch_deg'},
}


def calibration_path() -> Path:
    """$ROBOT_DOG_CALIBRATION if set, else robot_dog.yaml in the workspace source tree.

    With --symlink-install the installed share path is a symlink, resolved to the source."""
    env = os.environ.get(CALIBRATION_ENV)
    if env:
        return Path(env).expanduser()
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory('robot_dog_bringup')) / 'config' / CONFIG_FILE
        if share.exists():
            return share.resolve()
    except Exception:
        pass
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / 'src' / 'robot_dog_bringup' / 'config' / CONFIG_FILE
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'cannot find robot_dog_bringup/config/{CONFIG_FILE}')


def load_calibration(path: Optional[Path] = None) -> Dict[str, Any]:
    """Parsed config file ({} if missing or empty)."""
    path = Path(path) if path else calibration_path()
    if not path.is_file():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'{path}: top level must be a mapping')
    return data


def calibration_params(node_name: str, path: Optional[Path] = None) -> Dict[str, Any]:
    """The ros__parameters dict of one node ({} if none)."""
    section = load_calibration(path).get(node_name) or {}
    return dict(section.get('ros__parameters') or {})


# ---------------------------------------------------------------- line editing
def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_fmt(v) for v in value) + ']'
    if isinstance(value, float):
        return repr(round(value, 3))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f'unsupported value type {type(value).__name__}')


def _norm(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return value


def _set_key(lines: List[str], node: str, key: str, value_text: str) -> List[str]:
    """Replace ``key: ...`` inside ``node: / ros__parameters:``; insert it if missing."""
    node_re = re.compile(rf'^{re.escape(node)}:\s*(#.*)?$')
    node_idx = next((i for i, l in enumerate(lines) if node_re.match(l)), None)
    if node_idx is None:
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        lines += [f'{node}:\n', '  ros__parameters:\n', f'    {key}: {value_text}\n']
        return lines
    # section = following lines until the next top-level key
    end = len(lines)
    for i in range(node_idx + 1, len(lines)):
        if re.match(r'^[^\s#]', lines[i]):
            end = i
            break
    rp_idx = next((i for i in range(node_idx + 1, end)
                   if re.match(r'^\s+ros__parameters:\s*(#.*)?$', lines[i])), None)
    if rp_idx is None:
        lines.insert(node_idx + 1, '  ros__parameters:\n')
        rp_idx = node_idx + 1
        end += 1
    key_re = re.compile(rf'^(\s+{re.escape(key)}:\s*)(.*?)(\s+#.*)?\s*$')
    for i in range(rp_idx + 1, end):
        m = key_re.match(lines[i])
        if m:
            comment = m.group(3) or ''
            lines[i] = f'{m.group(1)}{value_text}{comment}\n'
            return lines
    lines.insert(rp_idx + 1, f'    {key}: {value_text}\n')
    return lines


def update_calibration(node_name: str, values: Dict[str, Any], path: Optional[Path] = None,
                       announce: bool = True) -> Path:
    """Write ``values`` into the node's section in place, verify by re-parsing, and print
    the CONFIG_UPDATE line for the deploy terminal. Returns the file path."""
    allowed = ALLOWED_KEYS.get(node_name, set())
    bad = set(values) - allowed
    if bad:
        raise KeyError(f'not a calibration key for {node_name}: {sorted(bad)}')
    path = _write_values(node_name, values, path)
    if announce:
        print(CONFIG_UPDATE_MARKER + ' ' + json.dumps({'node': node_name, 'values': values}),
              flush=True)
    return path


def _write_values(node_name: str, values: Dict[str, Any], path: Optional[Path]) -> Path:
    path = Path(path) if path else calibration_path()
    text = path.read_text() if path.is_file() else ''
    lines = text.splitlines(keepends=True)
    for key, value in values.items():
        lines = _set_key(lines, node_name, key, _fmt(value))
    new_text = ''.join(lines)
    data = yaml.safe_load(new_text) or {}
    for key, value in values.items():
        got = ((data.get(node_name) or {}).get('ros__parameters') or {}).get(key)
        if _norm(got) != _norm(value):
            raise RuntimeError(f'{path}: could not set {node_name}.{key} (got {got!r})')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.robot_dog-', suffix='.yaml')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(new_text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def parse_config_update(line: str) -> Optional[Dict[str, Any]]:
    """{'node', 'values'} if ``line`` carries a CONFIG_UPDATE marker, else None."""
    clean = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line).strip()
    idx = clean.find(CONFIG_UPDATE_MARKER)
    if idx < 0:
        return None
    try:
        payload = json.loads(clean[idx + len(CONFIG_UPDATE_MARKER):].strip())
    except ValueError:
        return None
    if not isinstance(payload, dict) or 'node' not in payload or 'values' not in payload:
        return None
    return payload
