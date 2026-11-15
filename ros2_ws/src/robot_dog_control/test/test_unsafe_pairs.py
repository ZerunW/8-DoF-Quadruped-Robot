"""Front/rear leg pair sweep (doc/pair_sweep_<side>_final.json), its lookup and its use in
the startup envelope check."""

import json
from pathlib import Path

import pytest

import robot_dog_control.controller as ctl
from robot_dog_control.config import controller_params_from_config, load_robot_config
from robot_dog_control.gait import STAND_CHECK, envelope_checks, schedule_shapes
from robot_dog_control.kinematics import (INTERLEG_CHECK, LegGeometry, LegKinematics,
                                          validate_envelope)
from robot_dog_control.leg_map import LEG_NAMES, LegMap
from robot_dog_control.unsafe_pairs import (DEFAULT_MARGIN_STEPS, FINAL_JSON, SidePairs,
                                            UnsafePairs, load_sides)

DOC = Path(__file__).resolve().parents[2] / 'robot_dog_description' / 'doc'
SIDES = {'left': ('lb', 'lf'), 'right': ('rb', 'rf')}
PARAMS = controller_params_from_config(load_robot_config())
REGION = PARAMS.joint_region()
KIN = LegKinematics(LegGeometry.from_mapping(PARAMS.geom), region=REGION)


@pytest.fixture(scope='module', params=sorted(SIDES))
def final(request):
    return request.param, json.loads((DOC / FINAL_JSON.format(side=request.param)).read_text())


def test_final_sweep_belongs_to_the_committed_region(final):
    side, d = final
    assert (d['rear_leg'], d['front_leg']) == SIDES[side]
    assert UnsafePairs(d).matches_region(REGION.to_params()), 'joint region changed'
    step = d['grid_step_deg']
    assert [tuple(p) for p in d['poses']] == [tuple(map(float, p)) for p in REGION.grid(step)]
    assert d['n_unsafe'] == len(d['unsafe']) > 1000 and d['settings']['margin_steps'] == 1


def test_lookup_is_the_sweep_grown_by_its_margin(final):
    side, d = final
    u = UnsafePairs(d)
    step = d['grid_step_deg']
    lat = [(round(m1 / step), round(m2 / step)) for m1, m2 in d['poses']]
    index = {q: k for k, q in enumerate(lat)}
    nb = [[index[(a + da, b + db)] for da in (-1, 0, 1) for db in (-1, 0, 1)
           if (a + da, b + db) in index] for a, b in lat]
    grown = {(rn, fn) for r, f, *_ in d['unsafe'] for rn in nb[r] for fn in nb[f]}
    assert u.n_marked() == len(grown) > len(d['unsafe']) == u.n_marked(0)
    n = len(u.poses)
    sample = [(r, f) for r in range(0, n, 7) for f in range(3, n, 11)]
    assert sum((r, f) in grown for r, f in sample) > 100
    assert all(u.marked(u.poses[r], u.poses[f]) == ((r, f) in grown) for r, f in sample)


def test_the_controller_loads_both_sides():
    sides = load_sides(LegMap.default())
    assert [(s.side, LEG_NAMES[s.rear], LEG_NAMES[s.front]) for s in sides] == \
        [('left', 'lb', 'lf'), ('right', 'rb', 'rf')]
    for s in sides:
        assert s.pairs is not None and not s.problem
        assert s.pairs is UnsafePairs.load(s.pairs.path)          # cached per file


def _toy(**kw):
    data = {'grid_step_deg': 2.0, 'rear_leg': 'lb', 'front_leg': 'lf',
            'poses': [[40.0, 60.0], [40.0, 62.0], [42.0, 60.0], [42.0, 62.0]],
            'unsafe': [[0, 3, 0.5, 'foot/foot']], 'n_pairs': 16,
            'region': {'ik_region_m1': [40.0, 42.0], 'ik_region_m2_lo': [60.0, 60.0],
                       'ik_region_m2_hi': [62.0, 62.0]}}
    data.update(kw)
    return data


def test_corner_rule_on_the_grid_between_nodes_and_outside():
    u = UnsafePairs(_toy(), margin_steps=0)
    assert u.marked((40.0, 60.0), (42.0, 62.0))                 # the unsafe pair itself
    assert not u.marked((42.0, 62.0), (40.0, 60.0))             # (rear, front) order
    assert u.marked((41.0, 61.0), (42.0, 62.0))                 # one cell corner is unsafe
    assert not u.marked((42.0, 61.0), (42.0, 62.0))             # both cell corners clear
    assert not u.marked((50.0, 60.0), (42.0, 62.0))             # outside the grid
    assert u.covers(41.0, 61.0) and not u.covers(43.0, 61.0)


def test_margin_grows_the_set_in_every_combination_of_the_four_angles():
    poses = [[float(a), float(b)] for a in range(40, 52, 2) for b in range(60, 72, 2)]
    k = {tuple(p): i for i, p in enumerate(poses)}
    data = _toy(poses=poses, unsafe=[[k[(44.0, 64.0)], k[(46.0, 66.0)], 0.0, 'foot/foot']],
                n_pairs=len(poses) ** 2, region=None)
    one = UnsafePairs(data)
    assert one.margin_steps == DEFAULT_MARGIN_STEPS == 1
    assert one.n_marked() == 9 * 9 and one.n_marked(0) == 1
    assert one.marked((42.0, 66.0), (48.0, 64.0))                # every angle one step off
    assert not one.marked((42.0, 66.0), (50.0, 64.0))            # front m1 two steps off
    assert UnsafePairs(data, margin_steps=2).marked((42.0, 66.0), (50.0, 64.0))
    assert one.margin_steps_of([(42.0, 62.0)], [(48.0, 68.0)], 8) == 0
    assert one.margin_steps_of([(40.0, 60.0)], [(50.0, 70.0)], 8) == 1
    assert one.margin_steps_of([(44.0, 64.0)], [(46.0, 66.0)], 8) == -1


@pytest.mark.parametrize('bad', [{'poses': [[40.0, 60.0], [41.0, 62.0]]},
                                 {'poses': [[40.0, 60.0], [40.0, 60.0]]},
                                 {'unsafe': [[0, 7, 0.5, 'foot/foot']]},
                                 {'grid_step_deg': 0.0}])
def test_lookup_rejects_malformed_files(bad):
    with pytest.raises(ValueError):
        UnsafePairs(_toy(**bad))


def _stand_joints(lm):
    out = []
    for i in range(4):
        r = KIN.ik(*lm.body_to_canonical(i, PARAMS.stand_fx_mm, -PARAMS.stand_height_mm))
        assert r.ok
        out.append((r.m1, r.m2))
    return out


def _side_marking(rear_q, front_q, rear=2, front=0):
    """A sweep of the region's 2 deg grid with one unsafe pair nearest (rear_q, front_q)."""
    poses = [list(q) for q in REGION.grid(2.0)]
    index = {tuple(q): k for k, q in enumerate(poses)}

    def node(q):
        return index[(2.0 * round(q[0] / 2.0), 2.0 * round(q[1] / 2.0))]

    data = {'grid_step_deg': 2.0, 'rear_leg': LEG_NAMES[rear], 'front_leg': LEG_NAMES[front],
            'poses': poses, 'unsafe': [[node(rear_q), node(front_q), 0.0, 'foot/foot']],
            'region': REGION.to_params(), 'settings': {'margin_steps': 1}}
    return SidePairs('left', rear, front, UnsafePairs(data))


def test_envelope_checks_every_front_back_pair_against_the_sweep(monkeypatch):
    lm = LegMap.default()
    shapes = dict(schedule_shapes(PARAMS.knots()))
    checks, names = envelope_checks(shapes, PARAMS.max_stride_mm, PARAMS.pose_limits(),
                                    PARAMS.stand_fx_mm, PARAMS.stand_height_mm,
                                    PARAMS.rest_height_mm, n=5)
    sides = load_sides(lm)
    rep = validate_envelope(KIN, lm, PARAMS.servo_min, PARAMS.servo_max, checks, interleg=sides)
    assert rep.ok and rep.min_interleg_margin_deg > sides[0].pairs.margin_steps * 2.0
    # a sweep that marks the left stand pose fails every check that plans it
    q = _stand_joints(lm)
    toy = _side_marking(q[2], q[0])
    rep = validate_envelope(KIN, lm, PARAMS.servo_min, PARAMS.servo_max, checks,
                            interleg=[toy, sides[1]])
    assert rep.failed == [INTERLEG_CHECK]
    assert STAND_CHECK in rep.interleg_failed and names['trot'] in rep.interleg_failed
    # without a sweep nothing can be verified
    missing = SidePairs('left', 2, 0, None, 'no CAD pair sweep (test)')
    rep = validate_envelope(KIN, lm, PARAMS.servo_min, PARAMS.servo_max, checks,
                            interleg=[missing, sides[1]])
    assert set(rep.interleg_failed) == set(checks)
    # the controller then refuses to stand, level or walk
    monkeypatch.setattr(ctl, 'load_sides', lambda leg_map: (toy, sides[1]))
    comp = ctl.build_components(PARAMS)
    assert not comp.walk_allowed and not comp.stand_ok and not comp.level_allowed
