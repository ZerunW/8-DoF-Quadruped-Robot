"""Feasible joint region of a leg in IK motor angles (m1, m2), in pure ``math``.

For each m1 knot the calf motor m2 has a feasible interval (linkage closure and part contact from the CAD),
linearly interpolated between knots; all four legs share one region (robot_dog.yaml ``ik_region_*``)."""

from __future__ import annotations

import bisect
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: parameter names in robot_dog.yaml (gait_controller)
PARAM_KEYS: Tuple[str, str, str] = ('ik_region_m1', 'ik_region_m2_lo', 'ik_region_m2_hi')


class JointRegion:
    """m1-monotone polygon in (m1, m2) degrees."""

    __slots__ = ('m1', 'm2_lo', 'm2_hi', '_bounds')

    def __init__(self, m1: Sequence[float], m2_lo: Sequence[float], m2_hi: Sequence[float]):
        m1 = tuple(float(v) for v in m1)
        lo = tuple(float(v) for v in m2_lo)
        hi = tuple(float(v) for v in m2_hi)
        problems = []
        if not (len(m1) == len(lo) == len(hi)):
            problems.append(f'{PARAM_KEYS[0]}, {PARAM_KEYS[1]}, {PARAM_KEYS[2]} must have the '
                            f'same length (got {len(m1)}, {len(lo)}, {len(hi)})')
        elif len(m1) < 2:
            problems.append(f'{PARAM_KEYS[0]} needs at least 2 knots')
        elif not all(math.isfinite(v) for v in m1 + lo + hi):
            problems.append('joint region values must be finite')
        else:
            if any(b <= a for a, b in zip(m1, m1[1:])):
                problems.append(f'{PARAM_KEYS[0]} must be strictly increasing')
            bad = [k for k in range(len(m1)) if lo[k] > hi[k]]
            if bad:
                problems.append(f'{PARAM_KEYS[1]} > {PARAM_KEYS[2]} at knot(s) {bad}')
        if problems:
            raise ValueError('invalid joint region: ' + '; '.join(problems))
        self.m1, self.m2_lo, self.m2_hi = m1, lo, hi
        self._bounds = (m1[0], m1[-1], min(lo), max(hi))

    @classmethod
    def box(cls, m1_range: Sequence[float], m2_range: Sequence[float]) -> 'JointRegion':
        """The rectangle m1_range x m2_range."""
        (a, b), (c, d) = m1_range, m2_range
        if not (a < b and c < d):
            raise ValueError('IK ranges must be (lo, hi) with lo < hi')
        return cls((a, b), (c, c), (d, d))

    @classmethod
    def from_params(cls, m1: Sequence[float], m2_lo: Sequence[float],
                    m2_hi: Sequence[float]) -> 'JointRegion':
        return cls(m1, m2_lo, m2_hi)

    def to_params(self, ndigits: int = 2) -> Dict[str, List[float]]:
        """{ik_region_m1: [...], ik_region_m2_lo: [...], ik_region_m2_hi: [...]}."""
        r = lambda seq: [round(v, ndigits) for v in seq]  # noqa: E731
        return {PARAM_KEYS[0]: r(self.m1), PARAM_KEYS[1]: r(self.m2_lo),
                PARAM_KEYS[2]: r(self.m2_hi)}

    def bounds(self) -> Tuple[float, float, float, float]:
        """(m1_min, m1_max, m2_min, m2_max): the bounding box."""
        return self._bounds

    def interval(self, m1: float) -> Tuple[float, float]:
        """m2 interval at m1 (m1 clamped into the region's m1 range)."""
        x = self.m1
        if m1 <= x[0]:
            return self.m2_lo[0], self.m2_hi[0]
        if m1 >= x[-1]:
            return self.m2_lo[-1], self.m2_hi[-1]
        k = bisect.bisect_right(x, m1) - 1
        t = (m1 - x[k]) / (x[k + 1] - x[k])
        lo, hi = self.m2_lo, self.m2_hi
        return lo[k] + t * (lo[k + 1] - lo[k]), hi[k] + t * (hi[k + 1] - hi[k])

    def contains(self, m1: float, m2: float, tol: float = 1e-9) -> bool:
        if not (self.m1[0] - tol <= m1 <= self.m1[-1] + tol):
            return False
        lo, hi = self.interval(m1)
        return lo - tol <= m2 <= hi + tol

    def clip(self, m1: float, m2: float) -> Tuple[float, float]:
        """Retraction into the region: m1 into [m1_min, m1_max], then m2 into the
        interval at that m1. Identity inside; NaN stays NaN."""
        a, b = self.m1[0], self.m1[-1]
        m1 = a if m1 < a else b if m1 > b else m1
        lo, hi = self.interval(m1)
        return m1, (lo if m2 < lo else hi if m2 > hi else m2)

    def polygon(self) -> List[Tuple[float, float]]:
        """Closed boundary, counter-clockwise: lower chain left to right, upper back."""
        low = list(zip(self.m1, self.m2_lo))
        up = list(zip(self.m1, self.m2_hi))[::-1]
        return low + up

    def margin(self, m1: float, m2: float) -> float:
        """Signed distance (deg) to the boundary: > 0 inside, < 0 outside."""
        poly = self.polygon()
        best = math.inf
        n = len(poly)
        for i in range(n):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            dd = dx * dx + dy * dy
            t = 0.0 if dd == 0.0 else max(0.0, min(1.0, ((m1 - x1) * dx + (m2 - y1) * dy) / dd))
            best = min(best, math.hypot(m1 - x1 - t * dx, m2 - y1 - t * dy))
        return best if self.contains(m1, m2, tol=0.0) else -best

    def area(self) -> float:
        """Area in deg^2."""
        return sum((self.m1[k + 1] - self.m1[k])
                   * ((self.m2_hi[k] - self.m2_lo[k]) + (self.m2_hi[k + 1] - self.m2_lo[k + 1]))
                   * 0.5 for k in range(len(self.m1) - 1))

    def grid(self, step: float = 1.0) -> Iterable[Tuple[float, float]]:
        """Points of the step grid (aligned to multiples of ``step``) inside the region."""
        m1_min, m1_max, m2_min, m2_max = self._bounds
        i = math.ceil(m1_min / step - 1e-9)
        while i * step <= m1_max + 1e-9:
            m1 = i * step
            lo, hi = self.interval(m1)
            j = math.ceil(lo / step - 1e-9)
            while j * step <= hi + 1e-9:
                yield m1, j * step
                j += 1
            i += 1

    def is_box(self) -> bool:
        return (len(set(self.m2_lo)) == 1 and len(set(self.m2_hi)) == 1)

    def __eq__(self, other) -> bool:
        return (isinstance(other, JointRegion) and self.m1 == other.m1
                and self.m2_lo == other.m2_lo and self.m2_hi == other.m2_hi)

    def __hash__(self) -> int:
        return hash((self.m1, self.m2_lo, self.m2_hi))

    def __repr__(self) -> str:
        b = self._bounds
        return (f'JointRegion({len(self.m1)} knots, m1 {b[0]:.1f}..{b[1]:.1f}, '
                f'm2 {b[2]:.1f}..{b[3]:.1f}, {self.area():.0f} deg^2)')


def region_from_config(gc: Dict, default: Optional[JointRegion] = None) -> Optional[JointRegion]:
    """JointRegion from a gait_controller parameter dict, ``default`` if the keys are absent."""
    if all(k in gc for k in PARAM_KEYS):
        return JointRegion(*(gc[k] for k in PARAM_KEYS))
    return default
