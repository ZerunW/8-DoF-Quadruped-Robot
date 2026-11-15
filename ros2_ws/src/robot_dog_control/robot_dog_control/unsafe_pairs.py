"""Unsafe rear x front leg pose pairs from the CAD pair sweep, with O(1) lookups.

The CAD pair sweep is one JSON file per side (robot_dog_description/doc/) listing the
joint-region grid pose pairs where the rear and front legs come within the clearance.
``UnsafePairs.marked`` grows that set by ``margin_steps`` grid steps in all four angles and
marks a pose between grid nodes if any corner of its grid cell is marked. Nodes outside the
swept region carry no data; ``covers()`` tells whether a pose is inside it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from robot_dog_control.joint_region import JointRegion

#: file name of the final sweep of one side (robot_dog_description/doc/)
FINAL_JSON = 'pair_sweep_{side}_final.json'
#: grid steps the set is grown by when the file has no settings.margin_steps
DEFAULT_MARGIN_STEPS = 1
_EPS = 1e-9

Pose = Tuple[float, float]


class UnsafePairs:
    """Unsafe rear x front pose pairs of one side.

    Grid nodes are numbered densely over their bounding box (row = m1 index, column = m2
    index); the set is one bitset over the front leg's nodes per rear node."""

    def __init__(self, data: Mapping, path: Optional[Path] = None,
                 margin_steps: Optional[int] = None):
        step = float(data['grid_step_deg'])
        if not (math.isfinite(step) and step > 0.0):
            raise ValueError(f'grid_step_deg must be > 0 (got {step})')
        self.step = step
        self.rear_leg = str(data['rear_leg'])
        self.front_leg = str(data['front_leg'])
        reg = data.get('region')
        self.region_params = dict(reg) if reg else None
        self.region: Optional[JointRegion] = (
            JointRegion(reg['ik_region_m1'], reg['ik_region_m2_lo'], reg['ik_region_m2_hi'])
            if reg else None)
        if margin_steps is None:
            margin_steps = int((data.get('settings') or {}).get('margin_steps',
                                                                DEFAULT_MARGIN_STEPS))
        if margin_steps < 0:
            raise ValueError(f'margin_steps must be >= 0 (got {margin_steps})')
        self.margin_steps = int(margin_steps)
        self.path = path
        self.key = None                     # set by load(): (path, size, mtime)
        # map the poses to lattice nodes, numbered densely over their bounding box
        lat = []
        for k, (m1, m2) in enumerate(data['poses']):
            lat.append(self._lattice(float(m1), float(m2)))
        if not lat:
            raise ValueError('no poses')
        if len(set(lat)) != len(lat):
            seen: Dict[Tuple[int, int], int] = {}
            for k, q in enumerate(lat):
                if q in seen:
                    raise ValueError(f'pose {k} {data["poses"][k]} repeats pose {seen[q]}')
                seen[q] = k
        self.poses: Tuple[Pose, ...] = tuple((a * step, b * step) for a, b in lat)
        self.i0 = min(a for a, _ in lat)
        self.j0 = min(b for _, b in lat)
        self.ni = max(a for a, _ in lat) - self.i0 + 1
        self.nj = max(b for _, b in lat) - self.j0 + 1
        n_dense = self.ni * self.nj
        self._dense_of_pose = [(a - self.i0) * self.nj + (b - self.j0) for a, b in lat]
        self._pose_of_dense = [-1] * n_dense
        pose_bits = 0
        for k, d in enumerate(self._dense_of_pose):
            self._pose_of_dense[d] = k
            pose_bits |= 1 << d
        self._pose_bits = pose_bits
        n = len(lat)
        rows = [0] * n_dense
        unsafe = set()
        for row in data['unsafe']:
            r, f = int(row[0]), int(row[1])
            if not (0 <= r < n and 0 <= f < n):
                raise ValueError(f'unsafe pair {list(row[:2])} indexes outside the {n} poses')
            unsafe.add((r, f))
            rows[self._dense_of_pose[r]] |= 1 << self._dense_of_pose[f]
        self.unsafe = frozenset(unsafe)
        self.n_poses = n
        self.n_pairs = int(data.get('n_pairs', n * n))
        # level k = the unsafe set grown by k grid steps. _grown holds the levels over the
        # whole box (growth may pass through non-pose nodes); _levels restricts them to poses.
        self._grown: List[List[int]] = [rows]
        self._levels: Dict[int, List[int]] = {}
        self.rows = self.level(self.margin_steps)

    @classmethod
    def load(cls, path, margin_steps: Optional[int] = None) -> 'UnsafePairs':
        """The file's set, cached per (path, size, mtime, margin) for the process."""
        path = Path(path).resolve()
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns, margin_steps)
        hit = _CACHE.get(key)
        if hit is None:
            hit = cls(json.loads(path.read_text()), path, margin_steps)
            hit.key = key
            if len(_CACHE) > 8:
                _CACHE.clear()
            _CACHE[key] = hit
        return hit

    def _lattice(self, m1: float, m2: float) -> Tuple[int, int]:
        a, b = m1 / self.step, m2 / self.step
        ka, kb = round(a), round(b)
        if abs(a - ka) > 1e-6 or abs(b - kb) > 1e-6:
            raise ValueError(f'pose ({m1}, {m2}) is not on the {self.step:g} deg grid')
        return int(ka), int(kb)

    def _corner_nodes(self, m1: float, m2: float) -> List[int]:
        """Dense indices of the grid cell corners around (m1, m2) inside the lattice box;
        nodes that are not poses mark nothing."""
        a, b = m1 / self.step - self.i0, m2 / self.step - self.j0
        ia, ja = math.floor(a + _EPS), math.floor(b + _EPS)
        ib = ia + 1 if a - ia > _EPS else ia
        jb = ja + 1 if b - ja > _EPS else ja
        ni, nj = self.ni, self.nj
        out = []
        for i in ((ia,) if ib == ia else (ia, ib)):
            if 0 <= i < ni:
                for j in ((ja,) if jb == ja else (ja, jb)):
                    if 0 <= j < nj:
                        out.append(i * nj + j)
        return out

    def corners(self, m1: float, m2: float) -> List[int]:
        """Pose indices of the grid cell around (m1, m2): the pose itself on the grid, up
        to 4 corners between nodes, none outside the swept grid."""
        pod = self._pose_of_dense
        return [pod[d] for d in self._corner_nodes(m1, m2) if pod[d] >= 0]

    def _grow(self, rows: List[int]) -> List[int]:
        """One grid step in every combination of the four angles (unrestricted)."""
        ni, nj = self.ni, self.nj
        n = ni * nj
        full = (1 << n) - 1
        first = last = 0                    # bits in the first / last m2 column
        for i in range(ni):
            first |= 1 << (i * nj)
            last |= 1 << (i * nj + nj - 1)
        not_first, not_last = full & ~first, full & ~last
        front = []
        for m in rows:                      # the front leg's two angles: bit shifts
            if m:
                m |= ((m << 1) & not_first) | ((m >> 1) & not_last)
                m = (m | (m << nj) | (m >> nj)) & full
            front.append(m)
        tmp = list(front)                   # the rear leg's m2: neighboring rows
        for d in range(n):
            j = d % nj
            if j > 0:
                tmp[d] |= front[d - 1]
            if j < nj - 1:
                tmp[d] |= front[d + 1]
        out = list(tmp)                     # the rear leg's m1
        for d in range(n):
            if d >= nj:
                out[d] |= tmp[d - nj]
            if d + nj < n:
                out[d] |= tmp[d + nj]
        return out

    def level(self, k: int) -> List[int]:
        """Rows (bitset per dense rear node) of the set grown by ``k`` grid steps,
        restricted to the grid's poses (computed once, cached)."""
        k = int(k)
        rows = self._levels.get(k)
        if rows is None:
            if k < 0:
                raise ValueError('level must be >= 0')
            while len(self._grown) <= k:
                self._grown.append(self._grow(self._grown[-1]))
            bits = self._pose_bits
            pod = self._pose_of_dense
            rows = [(m & bits) if pod[d] >= 0 else 0 for d, m in enumerate(self._grown[k])]
            self._levels[k] = rows
        return rows

    def marked(self, rear: Sequence[float], front: Sequence[float],
               rows: Optional[List[int]] = None) -> bool:
        """Is the pair of rear and front (m1, m2) poses marked? Tests at most 4 x 4 corner
        bits; ``rows`` selects another level (default: the margin's)."""
        fm = 0
        for d in self._corner_nodes(front[0], front[1]):
            fm |= 1 << d
        if not fm:
            return False
        rows = self.rows if rows is None else rows
        for d in self._corner_nodes(rear[0], rear[1]):
            if rows[d] & fm:
                return True
        return False

    def _masks(self, rear: Sequence[Pose], front: Sequence[Pose]):
        rn = set()
        for q in rear:
            rn.update(self._corner_nodes(q[0], q[1]))
        fm = 0
        for q in front:
            for d in self._corner_nodes(q[0], q[1]):
                fm |= 1 << d
        return rn, fm

    def marked_any(self, rear: Sequence[Pose], front: Sequence[Pose], level: Optional[int] = None,
                   _masks=None) -> bool:
        """Is any ``rear`` x ``front`` pair marked at ``level`` (default: the margin's)?"""
        rn, fm = _masks or self._masks(rear, front)
        if not fm:
            return False
        rows = self.rows if level is None else self.level(level)
        return any(rows[d] & fm for d in rn)

    def first_marked(self, rear: Sequence[Pose], front: Sequence[Pose]
                     ) -> Optional[Tuple[Pose, Pose]]:
        """A marked (rear, front) pair of the two pose lists, None if there is none."""
        for a in rear:
            for b in front:
                if self.marked(a, b):
                    return tuple(a), tuple(b)
        return None

    def margin_steps_of(self, rear: Sequence[Pose], front: Sequence[Pose],
                        max_steps: int) -> int:
        """Largest k <= ``max_steps`` such that the set grown by k steps marks no pair of
        the two pose lists; -1 if even the unsafe pairs themselves (k = 0) mark one."""
        masks = self._masks(rear, front)
        for k in range(max_steps + 1):
            if self.marked_any(rear, front, k, masks):
                return k - 1
        return max_steps

    def covers(self, m1: float, m2: float, tol: float = 1e-6) -> bool:
        """Is (m1, m2) inside the joint region the sweep covered?"""
        if self.region is not None:
            return self.region.contains(m1, m2, tol)
        return bool(self.corners(m1, m2))

    def matches_region(self, region_params: Mapping[str, Sequence[float]]) -> bool:
        """Was the sweep computed on this joint region (``JointRegion.to_params()``)?"""
        if not self.region_params:
            return False
        try:
            return all(len(self.region_params[k]) == len(v)
                       and all(abs(float(a) - float(b)) < 1e-6
                               for a, b in zip(self.region_params[k], v))
                       for k, v in region_params.items())
        except (KeyError, TypeError):
            return False

    def n_marked(self, level: Optional[int] = None) -> int:
        """Marked pairs of the grid (pose x pose) at ``level`` (default: the margin's)."""
        rows = self.rows if level is None else self.level(level)
        return sum(m.bit_count() for m in rows)

    def __len__(self) -> int:
        return len(self.unsafe)

    def __repr__(self) -> str:
        return (f'UnsafePairs({self.rear_leg} x {self.front_leg}: {len(self.unsafe)} unsafe of '
                f'{self.n_pairs} pairs, {self.n_poses} poses on a {self.step:g} deg grid, '
                f'grown by {self.margin_steps} step(s))')


_CACHE: Dict[tuple, UnsafePairs] = {}


def find_final_json(side: str) -> Optional[Path]:
    """The final sweep of ``side`` ('left' / 'right'): the installed
    share/robot_dog_description/doc copy, else the source tree's; None if neither exists."""
    name = FINAL_JSON.format(side=side)
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory('robot_dog_description')) / 'doc' / name
        if p.is_file():
            return p
    except Exception:  # noqa: BLE001
        pass
    for parent in Path(__file__).resolve().parents:
        cand = parent / 'robot_dog_description' / 'doc' / name
        if cand.is_file():
            return cand
    return None


class SidePairs(NamedTuple):
    """The rear and front leg of one side and their unsafe pairs; ``pairs`` is None when
    missing, and ``problem`` says why."""

    side: str                       # 'left' / 'right'
    rear: int                       # leg index (leg map order lf, rf, lb, rb)
    front: int
    pairs: Optional[UnsafePairs]
    problem: str = ''


def side_legs(leg_map) -> List[Tuple[str, int, int]]:
    """(side, rear leg index, front leg index) of each side, from the leg map's signs."""
    from robot_dog_control.leg_map import FRONT_SIGN, RIGHT_SIGN
    out = []
    n = len(leg_map.legs)
    for f in range(n):
        for b in range(n):
            if FRONT_SIGN[f] > 0 > FRONT_SIGN[b] and RIGHT_SIGN[f] == RIGHT_SIGN[b]:
                out.append(('left' if RIGHT_SIGN[f] < 0 else 'right', b, f))
    return out


def load_sides(leg_map, margin_steps: Optional[int] = None) -> Tuple[SidePairs, ...]:
    """The final pair sweep of every side of ``leg_map`` (cached per file)."""
    out = []
    for side, rear, front in side_legs(leg_map):
        names = (leg_map.legs[rear].name, leg_map.legs[front].name)
        path = find_final_json(side)
        if path is None:
            out.append(SidePairs(side, rear, front, None,
                                 f'no CAD pair sweep for the {side} legs '
                                 f'(robot_dog_description/doc/{FINAL_JSON.format(side=side)}): '
                                 'run `pixi run build`'))
            continue
        try:
            pairs = UnsafePairs.load(path, margin_steps)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            out.append(SidePairs(side, rear, front, None, f'cannot read {path} ({exc})'))
            continue
        if (pairs.rear_leg, pairs.front_leg) != names:
            out.append(SidePairs(side, rear, front, None,
                                 f'{path.name} is for {pairs.rear_leg} x {pairs.front_leg}, not '
                                 f'{names[0]} x {names[1]}'))
            continue
        out.append(SidePairs(side, rear, front, pairs))
    return tuple(out)
