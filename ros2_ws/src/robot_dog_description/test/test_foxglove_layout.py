"""The Foxglove layout: every panel in the tree has a config, and every /gait/debug series
is labeled with the controller.DEBUG_FIELDS name at its index."""

import json
import re
from pathlib import Path

from robot_dog_control.controller import DEBUG_FIELDS

LAYOUT = Path(__file__).resolve().parents[1] / 'foxglove' / 'robot_dog_layout.json'


def _leaves(node):
    if isinstance(node, str):
        return [node]
    return _leaves(node['first']) + _leaves(node['second'])


def test_layout_tree_matches_panel_configs():
    layout = json.loads(LAYOUT.read_text())
    assert sorted(_leaves(layout['layout'])) == sorted(layout['configById'])


def test_debug_series_match_the_debug_fields():
    layout = json.loads(LAYOUT.read_text())
    series = [p for cfg in layout['configById'].values() for p in cfg.get('paths', [])
              if p['value'].startswith('/gait/debug')]
    assert len(series) > 10
    for p in series:
        index = int(re.fullmatch(r'/gait/debug\.data\[(\d+)\]', p['value']).group(1))
        assert p['label'].split()[0] == DEBUG_FIELDS[index], p
