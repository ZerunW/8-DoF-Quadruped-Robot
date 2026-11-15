"""In-place config editing (robot_dog.yaml) and the CONFIG_UPDATE round trip."""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_dog_hw.calibration import (CONFIG_UPDATE_MARKER, calibration_params,  # noqa: E402
                                      parse_config_update, update_calibration)

SAMPLE = """# header comment
servo_driver:
  ros__parameters:
    i2c_bus: 1
    servo_offset:  [15.0, 5.0, 15.0, 10.0, 3.0, -5.0, 3.0, 8.0]   # Arduino values
    servo_min:     [140.0, 150.0, 140.0, 150.0, 140.0, 150.0, 140.0, 150.0]

gait_controller:
  ros__parameters:
    level_target_roll_deg: 0.0       # mount
    leg_trim_mm: [0.0, 0.0, 0.0, 0.0]
"""


def test_update_keeps_layout_and_comments(tmp_path, capsys):
    path = tmp_path / 'robot_dog.yaml'
    path.write_text(SAMPLE)
    out = update_calibration('servo_driver', {'servo_offset': [14.523, 12.2871, 20.4, 12.5,
                                                               9.52, 5.365, 4.879, 21.827]},
                             path=path)
    assert out == path
    text = path.read_text()
    assert text.startswith('# header comment\n')
    assert '# Arduino values' in text            # inline comment kept
    assert 'servo_min:     [140.0' in text       # untouched lines identical
    data = yaml.safe_load(text)
    assert data['servo_driver']['ros__parameters']['servo_offset'][1] == pytest.approx(12.287)
    assert data['servo_driver']['ros__parameters']['i2c_bus'] == 1
    update_calibration('gait_controller', {'level_target_roll_deg': -3.3,
                                           'leg_trim_x_mm': [0.0, 1.5, 0.0, 0.0]}, path=path)
    text = path.read_text()
    assert 'level_target_roll_deg: -3.3       # mount' in text
    assert calibration_params('gait_controller', path)['leg_trim_x_mm'] == [0.0, 1.5, 0.0, 0.0]
    printed = capsys.readouterr().out
    lines = [l for l in printed.splitlines() if l.startswith(CONFIG_UPDATE_MARKER)]
    assert len(lines) == 2
    upd = parse_config_update('\x1b[32m' + lines[0] + '\r')
    assert upd['node'] == 'servo_driver' and upd['values']['servo_offset'][0] == 14.523


def test_update_rejects_unknown_keys_and_missing_section_is_created(tmp_path):
    path = tmp_path / 'robot_dog.yaml'
    path.write_text('servo_driver:\n  ros__parameters:\n    i2c_bus: 1\n')
    with pytest.raises(KeyError):
        update_calibration('servo_driver', {'servo_min': [0.0] * 8}, path=path, announce=False)
    update_calibration('gait_controller', {'leg_trim_mm': [1.0, 2.0, 3.0, 4.0]}, path=path,
                       announce=False)
    data = yaml.safe_load(path.read_text())
    assert data['gait_controller']['ros__parameters']['leg_trim_mm'] == [1.0, 2.0, 3.0, 4.0]
    assert data['servo_driver']['ros__parameters']['i2c_bus'] == 1


def test_parse_config_update_ignores_other_lines():
    assert parse_config_update('roll +0.1 pitch -0.2') is None
    assert parse_config_update(CONFIG_UPDATE_MARKER + ' not json') is None
    payload = json.dumps({'node': 'gait_controller', 'values': {'leg_trim_mm': [0, 0, 0, 0]}})
    assert parse_config_update(f'[tool] {CONFIG_UPDATE_MARKER} {payload}')['node'] == 'gait_controller'
