"""Kinematics of the 5-bar (parallelogram) leg in pure ``math``, no numpy.

Canonical leg frame: origin O at the thigh motor axis, +y up (foot at y < 0); lengths in mm, motor angles
in deg (m1 thigh motor at O, m2 calf motor at M = (d_x, d_y)). IK stays inside a JointRegion; FK returns a
pose for any (m1, m2); closes() says whether the linkage assembles there."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from robot_dog_control.joint_region import JointRegion

_D2R = math.pi / 180.0
_R2D = 180.0 / math.pi

# Reference foot trajectory in servo units (IK motor angle + 90): *_1 = 20 swing samples, *_2 = 20 stance samples.
LUT_THIGH_1 = (164.99, 167.58, 170.03, 172.33, 174.47, 176.46, 178.28, 179.93, 181.4, 182.65,
               183.69, 184.46, 184.97, 185.16, 185.03, 184.58, 183.83, 182.78, 181.48, 179.95)
LUT_THIGH_2 = (179.95, 179.19, 178.42, 177.64, 176.85, 176.05, 175.25, 174.44, 173.62, 172.79,
               171.95, 171.1, 170.24, 169.39, 168.52, 167.65, 166.77, 165.89, 165.0, 164.09)
LUT_CALF_1 = (166.18, 163.9, 161.84, 160.04, 158.56, 157.43, 156.7, 156.39, 156.52, 157.07,
              158.06, 159.44, 161.18, 163.23, 165.56, 168.12, 170.88, 173.77, 176.79, 179.9)
LUT_CALF_2 = (179.89, 179.15, 178.41, 177.68, 176.97, 176.26, 175.55, 174.88, 174.2, 173.54,
              172.89, 172.26, 171.63, 171.02, 170.42, 169.84, 169.26, 168.71, 168.17, 167.64)


@dataclass(frozen=True)
class LegGeometry:
    """Link lengths (mm) and bell-crank angle phi (deg) of the leg."""

    d_x: float = -20.0
    d_y: float = -20.0
    L1: float = 110.0    # thigh O-K
    L2: float = 110.0    # calf K-E
    L2a: float = 30.0    # calf lever K-B
    L3: float = 24.0     # calf motor crank M-C
    L41: float = 24.0    # bell-crank arm O-D
    L42: float = 30.0    # bell-crank arm O-A
    phi: float = 90.05   # angle D-O-A (deg)
    L51: float = 28.2    # link C-D
    L52: float = 110.0   # link A-B

    @classmethod
    def from_mapping(cls, m: Mapping[str, float]) -> 'LegGeometry':
        names = {f.name for f in fields(cls)}
        unknown = set(m) - names
        if unknown:
            raise ValueError(f'unknown geometry keys: {sorted(unknown)}')
        return cls(**{k: float(v) for k, v in m.items()})

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def validate(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if not math.isfinite(v):
                raise ValueError(f'geometry {f.name} must be finite')
            if f.name not in ('d_x', 'd_y', 'phi') and v <= 0.0:
                raise ValueError(f'geometry {f.name} must be > 0')


class IKResult(NamedTuple):
    m1: float        # thigh motor angle (deg)
    m2: float        # calf motor angle (deg)
    err_mm: float    # |FK(m1, m2) - target|
    ok: bool         # err_mm <= tolerance and target finite


class Closure(NamedTuple):
    """Assembly state of the two loops at (m1, m2), see ``LegKinematics.closure``."""

    loop1: float        # arccos argument of loop 1 (M-C-D-O, k1/|OC|); |.| < 1: closes
    loop2: float        # arccos argument of loop 2 (O-A-B-K, k2/|AK|); |.| < 1: closes
    cad_branch: bool    # the FK picks the CAD assembly in both loops
    mu1_deg: float      # transmission angle at D (D->O vs D->C); 0 / 180 = loop 1 flat
    mu2_deg: float      # transmission angle at B (B->K vs B->A) = knee angle O-K-E

    @property
    def closes(self) -> bool:
        return abs(self.loop1) < 1.0 and abs(self.loop2) < 1.0 and self.cad_branch


def _angle_deg(ux: float, uy: float, vx: float, vy: float) -> float:
    n = math.hypot(ux, uy) * math.hypot(vx, vy)
    if n == 0.0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (ux * vx + uy * vy) / n))))


class FKPoints(NamedTuple):
    """All joint points of the linkage (canonical frame, mm)."""

    O: Tuple[float, float]
    M: Tuple[float, float]
    K: Tuple[float, float]
    C: Tuple[float, float]
    D: Tuple[float, float]
    A: Tuple[float, float]
    B: Tuple[float, float]
    E: Tuple[float, float]


def _allclose(px: float, py: float, qx: float, qy: float) -> bool:
    """numpy.allclose(p, q) with its defaults (rtol=1e-5, atol=1e-8)."""
    return abs(px - qx) <= 1e-8 + 1e-5 * abs(qx) and abs(py - qy) <= 1e-8 + 1e-5 * abs(qy)


def _ccw(ax, ay, bx, by, cx, cy) -> bool:
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def _crosses(ax, ay, bx, by, kx, ky) -> bool:
    """Does segment A-B cross the thigh O-K (O at the origin)?"""
    if (_allclose(ax, ay, 0.0, 0.0) or _allclose(ax, ay, kx, ky)
            or _allclose(bx, by, 0.0, 0.0) or _allclose(bx, by, kx, ky)):
        return False
    return (_ccw(ax, ay, 0.0, 0.0, kx, ky) != _ccw(bx, by, 0.0, 0.0, kx, ky)
            and _ccw(ax, ay, bx, by, 0.0, 0.0) != _ccw(ax, ay, bx, by, kx, ky))


class LegKinematics:
    """FK / IK for one leg geometry; the IK stays inside a joint region of (m1, m2).

    ``region`` (a ``JointRegion``) replaces the rectangle ``m1_range x m2_range`` when
    given; ``m1_range`` / ``m2_range`` are then its bounding box.
    """

    #: IK is "ok" when the solution reproduces the target within this distance.
    TOL_MM = 0.5

    def __init__(self, geom: LegGeometry = LegGeometry(),
                 m1_range: Tuple[float, float] = (60.0, 120.0),
                 m2_range: Tuple[float, float] = (60.0, 100.0),
                 region: Optional[JointRegion] = None):
        geom.validate()
        if region is None:
            region = JointRegion.box(m1_range, m2_range)
        self.geom = geom
        self.region = region
        b = region.bounds()
        self.m1_range = (b[0], b[1])
        self.m2_range = (b[2], b[3])
        g = geom
        self._alpha0 = 5.0 * math.pi / 4.0
        self._gamma0 = math.pi / 4.0
        self._phi = math.radians(g.phi)
        self._k1c = g.L41 * g.L41 - g.L51 * g.L51
        self._k2c = g.L2a * g.L2a - g.L52 * g.L52
        self._seed_table: Optional[List[Tuple[float, float, float, float]]] = None
        self._seed_buckets: Optional[Dict[Tuple[int, int], List[int]]] = None
        self._build_seed_table()

    def _knee(self, m1: float) -> Tuple[float, float]:
        a = self._alpha0 + m1 * _D2R
        L1 = self.geom.L1
        return L1 * math.cos(a), L1 * math.sin(a)

    def _rocker(self, m2: float) -> Tuple[float, float, float, float, float, float]:
        """Bell-crank point A, plus C and D; depends on m2 only."""
        g = self.geom
        gam = self._gamma0 + m2 * _D2R
        cx = g.d_x + g.L3 * math.cos(gam)
        cy = g.d_y + g.L3 * math.sin(gam)
        c_len = math.sqrt(cx * cx + cy * cy)
        k1 = (c_len * c_len + self._k1c) / (2.0 * g.L41)
        if k1 > c_len:
            k1 = c_len
        elif k1 < -c_len:
            k1 = -c_len
        ang_c = math.atan2(cy, cx)
        w = math.acos(k1 / c_len)
        d1 = ang_c + w
        d2 = ang_c - w
        D1y = g.L41 * math.sin(d1)
        D2y = g.L41 * math.sin(d2)
        delta = d1 if D1y > D2y else d2
        da = delta - self._phi
        return (g.L42 * math.cos(da), g.L42 * math.sin(da), cx, cy,
                g.L41 * math.cos(delta), g.L41 * math.sin(delta))

    def _foot(self, kx: float, ky: float, ax: float, ay: float):
        """Foot E and point B from knee K and bell-crank A. B is the candidate whose link
        A-B does not cross the thigh, else the one closest to the link length L52."""
        g = self.geom
        vx = ax - kx
        vy = ay - ky
        v_len = math.sqrt(vx * vx + vy * vy)
        k2 = (v_len * v_len + self._k2c) / (2.0 * g.L2a)
        if k2 > v_len:
            k2 = v_len
        elif k2 < -v_len:
            k2 = -v_len
        ang_v = math.atan2(vy, vx)
        w = math.acos(k2 / v_len)
        bb1 = ang_v + w
        bb2 = ang_v - w
        b1x = kx + g.L2a * math.cos(bb1)
        b1y = ky + g.L2a * math.sin(bb1)
        b2x = kx + g.L2a * math.cos(bb2)
        b2y = ky + g.L2a * math.sin(bb2)
        c1 = _crosses(ax, ay, b1x, b1y, kx, ky)
        c2 = _crosses(ax, ay, b2x, b2y, kx, ky)
        if c1 and not c2:
            bb, bx, by = bb2, b2x, b2y
        elif c2 and not c1:
            bb, bx, by = bb1, b1x, b1y
        else:
            e1 = abs(math.hypot(ax - b1x, ay - b1y) - g.L52)
            e2 = abs(math.hypot(ax - b2x, ay - b2y) - g.L52)
            if e1 < e2:
                bb, bx, by = bb1, b1x, b1y
            else:
                bb, bx, by = bb2, b2x, b2y
        bp = bb + math.pi
        return kx + g.L2 * math.cos(bp), ky + g.L2 * math.sin(bp), bx, by

    def fk(self, m1: float, m2: float) -> Tuple[float, float, Tuple[float, float]]:
        """Foot position (x, y) and knee point for motor angles (deg)."""
        kx, ky = self._knee(m1)
        ax, ay = self._rocker(m2)[:2]
        ex, ey = self._foot(kx, ky, ax, ay)[:2]
        return ex, ey, (kx, ky)

    def fk_points(self, m1: float, m2: float) -> FKPoints:
        """All linkage points (canonical frame, mm)."""
        g = self.geom
        kx, ky = self._knee(m1)
        ax, ay, cx, cy, dx, dy = self._rocker(m2)
        ex, ey, bx, by = self._foot(kx, ky, ax, ay)
        return FKPoints((0.0, 0.0), (g.d_x, g.d_y), (kx, ky), (cx, cy), (dx, dy),
                        (ax, ay), (bx, by), (ex, ey))

    def closure(self, m1: float, m2: float) -> Closure:
        """Can the 5-bar assemble at (m1, m2), and does fk_points() return that assembly?

        Loop 1 closes iff ``|k1| < |OC|`` and loop 2 iff ``|k2| < |AK|``. The CAD assembly
        has D clockwise of O->C and the parallelogram O-A-B-K. Fields are NaN when a loop
        does not close.
        """
        g = self.geom
        nan = math.nan
        kx, ky = self._knee(m1)
        gam = self._gamma0 + m2 * _D2R
        cx = g.d_x + g.L3 * math.cos(gam)
        cy = g.d_y + g.L3 * math.sin(gam)
        c_len = math.sqrt(cx * cx + cy * cy)
        q1 = (c_len * c_len + self._k1c) / (2.0 * g.L41) / c_len if c_len > 0.0 else math.inf
        if not abs(q1) < 1.0:
            return Closure(q1, nan, False, nan, nan)
        ang_c = math.atan2(cy, cx)
        w = math.acos(q1)
        d1, d2 = ang_c + w, ang_c - w
        branch1 = math.sin(d2) - math.sin(d1) > 1e-12     # the FK takes d2, the CAD assembly
        dx, dy = g.L41 * math.cos(d2), g.L41 * math.sin(d2)
        da = d2 - self._phi
        ax, ay = g.L42 * math.cos(da), g.L42 * math.sin(da)
        mu1 = _angle_deg(-dx, -dy, cx - dx, cy - dy)
        vx, vy = ax - kx, ay - ky
        v_len = math.sqrt(vx * vx + vy * vy)
        q2 = (v_len * v_len + self._k2c) / (2.0 * g.L2a) / v_len if v_len > 0.0 else math.inf
        if not abs(q2) < 1.0:
            return Closure(q1, q2, False, mu1, nan)
        ang_v = math.atan2(vy, vx)
        w2 = math.acos(q2)
        b1x, b1y = kx + g.L2a * math.cos(ang_v + w2), ky + g.L2a * math.sin(ang_v + w2)
        b2x, b2y = kx + g.L2a * math.cos(ang_v - w2), ky + g.L2a * math.sin(ang_v - w2)
        branch2 = (_crosses(ax, ay, b1x, b1y, kx, ky)
                   and not _crosses(ax, ay, b2x, b2y, kx, ky))
        mu2 = _angle_deg(kx - b2x, ky - b2y, ax - b2x, ay - b2y)
        return Closure(q1, q2, branch1 and branch2, mu1, mu2)

    def closes(self, m1: float, m2: float, min_transmission_deg: float = 0.0) -> bool:
        """True iff the linkage closes in the CAD assembly at (m1, m2) and both
        transmission angles lie within [min_transmission_deg, 180 - min_transmission_deg]."""
        c = self.closure(m1, m2)
        if not c.closes:
            return False
        lo, hi = min_transmission_deg, 180.0 - min_transmission_deg
        return lo <= c.mu1_deg <= hi and lo <= c.mu2_deg <= hi

    def jacobian(self, m1: float, m2: float, h: float = 1e-3
                 ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """d(foot x, y)/d(m1, m2) in mm/deg (central differences): ((dx/dm1, dx/dm2),
        (dy/dm1, dy/dm2))."""
        xp, yp, _ = self.fk(m1 + h, m2)
        xm, ym, _ = self.fk(m1 - h, m2)
        xq, yq, _ = self.fk(m1, m2 + h)
        xr, yr, _ = self.fk(m1, m2 - h)
        s = 0.5 / h
        return ((xp - xm) * s, (xq - xr) * s), ((yp - ym) * s, (yq - yr) * s)

    def sigma_min(self, m1: float, m2: float) -> float:
        """Smallest singular value of the foot Jacobian (mm/deg): the foot moves at least
        this much per degree in every direction; 0 at a singular pose."""
        (a, b), (c, d) = self.jacobian(m1, m2)
        s = a * a + b * b + c * c + d * d
        det = a * d - b * c
        return math.sqrt(max(0.0, 0.5 * (s - math.sqrt(max(0.0, s * s - 4.0 * det * det)))))

    def _clip(self, m1: float, m2: float) -> Tuple[float, float]:
        return self.region.clip(m1, m2)

    def _gauss_newton(self, x: float, y: float, m1: float, m2: float,
                      max_iter: int = 6, tol: float = 1e-4):
        """Damped Gauss-Newton with a central-difference Jacobian; every iterate is clipped
        into the joint region. Returns (m1, m2, err)."""
        h = 1e-3              # deg, central-difference step
        inv2h = 1.0 / (2.0 * h)
        lam = 1e-6            # Levenberg damping
        max_step = 15.0       # deg per iteration
        knee = self._knee
        rocker = self._rocker
        foot = self._foot
        clip = self.region.clip
        kx, ky = knee(m1)
        r0 = rocker(m2)
        ex, ey = foot(kx, ky, r0[0], r0[1])[:2]
        rx = x - ex
        ry = y - ey
        err = math.sqrt(rx * rx + ry * ry)
        for _ in range(max_iter):
            if err <= tol:
                break
            kpx, kpy = knee(m1 + h)
            kmx, kmy = knee(m1 - h)
            rp = rocker(m2 + h)
            rm = rocker(m2 - h)
            e1p = foot(kpx, kpy, r0[0], r0[1])
            e1m = foot(kmx, kmy, r0[0], r0[1])
            e2p = foot(kx, ky, rp[0], rp[1])
            e2m = foot(kx, ky, rm[0], rm[1])
            j11 = (e1p[0] - e1m[0]) * inv2h
            j21 = (e1p[1] - e1m[1]) * inv2h
            j12 = (e2p[0] - e2m[0]) * inv2h
            j22 = (e2p[1] - e2m[1]) * inv2h
            a = j11 * j11 + j21 * j21 + lam
            b = j11 * j12 + j21 * j22
            d = j12 * j12 + j22 * j22 + lam
            g1 = j11 * rx + j21 * ry
            g2 = j12 * rx + j22 * ry
            det = a * d - b * b
            if not det > 1e-12:
                break
            d1 = (d * g1 - b * g2) / det
            d2 = (a * g2 - b * g1) / det
            big = max(abs(d1), abs(d2))
            if big > max_step:
                d1 *= max_step / big
                d2 *= max_step / big
            n1, n2 = clip(m1 + d1, m2 + d2)
            nkx, nky = knee(n1)
            nr0 = rocker(n2)
            nex, ney = foot(nkx, nky, nr0[0], nr0[1])[:2]
            nrx = x - nex
            nry = y - ney
            nerr = math.sqrt(nrx * nrx + nry * nry)
            if not nerr < err:
                if nerr == nerr and nerr <= err + 1e-9:
                    m1, m2, err = n1, n2, nerr
                break
            m1, m2, kx, ky, r0, rx, ry, err = n1, n2, nkx, nky, nr0, nrx, nry, nerr
        return m1, m2, err

    def _build_seed_table(self) -> None:
        """1-degree grid of FK samples inside the joint region, bucketed by 10 mm cells."""
        table = []
        for m1, m2 in self.region.grid(1.0):
            ex, ey, _ = self.fk(m1, m2)
            table.append((ex, ey, m1, m2))
        if not table:   # region thinner than the grid: use its knots
            for m1, lo, hi in zip(self.region.m1, self.region.m2_lo, self.region.m2_hi):
                m2 = 0.5 * (lo + hi)
                ex, ey, _ = self.fk(m1, m2)
                table.append((ex, ey, m1, m2))
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for i, (ex, ey, _, _) in enumerate(table):
            buckets.setdefault((int(math.floor(ex / 10.0)), int(math.floor(ey / 10.0))),
                               []).append(i)
        self._seed_table = table
        self._seed_buckets = buckets

    def seed(self, x: float, y: float) -> Tuple[float, float]:
        """Nearest grid sample (m1, m2) to the target (fallback IK seed)."""
        if self._seed_table is None:
            self._build_seed_table()
        table, buckets = self._seed_table, self._seed_buckets
        cx, cy = int(math.floor(x / 10.0)), int(math.floor(y / 10.0))
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(buckets.get((cx + dx, cy + dy), ()))
        if not cand:
            cand = range(len(table))
        best, best_d = None, math.inf
        for i in cand:
            ex, ey, _, _ = table[i]
            dd = (ex - x) * (ex - x) + (ey - y) * (ey - y)
            if dd < best_d:
                best, best_d = i, dd
        return table[best][2], table[best][3]

    def ik(self, x: float, y: float, guess: Optional[Tuple[float, float]] = None) -> IKResult:
        """Motor angles that put the foot at (x, y); never outside the joint region.

        Warm start from ``guess`` (e.g. last tick's solution); falls back to the
        1-degree seed table when the warm start fails or leaves err > 1 mm.
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            return IKResult(math.nan, math.nan, math.inf, False)
        if guess is not None and math.isfinite(guess[0]) and math.isfinite(guess[1]):
            m1, m2 = self._clip(float(guess[0]), float(guess[1]))
        else:
            m1, m2 = self.seed(x, y)
        m1, m2, err = self._gauss_newton(x, y, m1, m2)
        if not err <= 1.0:
            s1, s2 = self.seed(x, y)
            n1, n2, nerr = self._gauss_newton(x, y, s1, s2, max_iter=12)
            if nerr < err or not err == err:
                m1, m2, err = n1, n2, nerr
        if not err == err:  # NaN
            return IKResult(math.nan, math.nan, math.inf, False)
        return IKResult(m1, m2, err, err <= self.TOL_MM)

    def reachable(self, x: float, y: float, guess=None) -> bool:
        return self.ik(x, y, guess).ok

    def clamp_to_workspace(self, x: float, y: float, center: Tuple[float, float],
                           guess: Optional[Tuple[float, float]] = None,
                           iterations: int = 12) -> Tuple[float, float, IKResult, bool]:
        """Shrink (x, y) radially toward ``center`` until reachable.

        Returns (x', y', ik_result, was_clamped). ``center`` (the stand point) must be
        reachable; if it is not, the center and its best-effort IK (ok=False) are returned.
        """
        res = self.ik(x, y, guess)
        if res.ok:
            return x, y, res, False
        cx, cy = center
        cres = self.ik(cx, cy, guess)
        if not cres.ok or not (math.isfinite(x) and math.isfinite(y)):
            return cx, cy, cres, True
        lo, hi = 0.0, 1.0
        best = cres
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            px, py = cx + mid * (x - cx), cy + mid * (y - cy)
            r = self.ik(px, py, (best.m1, best.m2))
            if r.ok:
                lo, best = mid, r
            else:
                hi = mid
        return cx + lo * (x - cx), cy + lo * (y - cy), best, True

    def workspace_outline(self, step_deg: float = 1.0) -> List[Tuple[float, float]]:
        """Foot points along the joint region's boundary as a closed polygon: the m1 low
        edge, the m2 high edge, the m1 high edge, then the m2 low edge."""
        reg = self.region
        lo1, hi1 = self.m1_range
        pts = []

        def frange(a, b, s):
            n = max(1, int(round(abs(b - a) / s)))
            return [a + (b - a) * k / n for k in range(n)]

        lo_a, hi_a = reg.interval(lo1)
        lo_b, hi_b = reg.interval(hi1)
        for m2 in frange(lo_a, hi_a, step_deg):
            pts.append(self.fk(lo1, m2)[:2])
        for m1 in frange(lo1, hi1, step_deg):
            pts.append(self.fk(m1, reg.interval(m1)[1])[:2])
        for m2 in frange(hi_b, lo_b, step_deg):
            pts.append(self.fk(hi1, m2)[:2])
        for m1 in frange(hi1, lo1, step_deg):
            pts.append(self.fk(m1, reg.interval(m1)[0])[:2])
        return pts


# Envelope validation
@dataclass
class EnvelopeReport:
    """Outcome of validate_envelope(): which named checks failed and why."""

    ok: bool
    failed: List[str]                 # names of failed checks (parameter names)
    messages: List[str]               # one human-readable line per failure
    worst_err_mm: float               # largest IK error seen (passing points)
    min_servo_margin_deg: float       # smallest distance of any servo angle to a limit
    points_checked: int
    min_region_margin_deg: float = math.inf    # smallest distance to the joint region edge
    #: front/back legs: how far (deg, on all four joints) the unsafe pairs could grow and
    #: still mark no planned pair, capped at INTERLEG_MARGIN_MAX_STEPS grid steps
    min_interleg_margin_deg: float = math.inf
    interleg_margin_capped: bool = False       # the margin reached that cap
    interleg_failed: List[str] = field(default_factory=list)  # checks with an unsafe pair

    def summary(self) -> str:
        if self.ok:
            extra = ''
            if math.isfinite(self.min_region_margin_deg):
                extra += f', min region margin {self.min_region_margin_deg:.1f} deg'
            if math.isfinite(self.min_interleg_margin_deg):
                extra += (f', front/back legs {">= " if self.interleg_margin_capped else ""}'
                          f'{self.min_interleg_margin_deg:.0f} deg from a CAD contact')
            return (f'envelope OK ({self.points_checked} poses, worst IK err '
                    f'{self.worst_err_mm:.3f} mm, min servo margin '
                    f'{self.min_servo_margin_deg:.1f} deg{extra})')
        return 'envelope FAILED: ' + ' | '.join(self.messages)


#: envelope check name for the front/back leg pairs
INTERLEG_CHECK = 'front/back legs (CAD pair sweep)'
#: the front/back margin is reported up to this many grid steps of the pair sweep
INTERLEG_MARGIN_MAX_STEPS = 8


def validate_envelope(kin: LegKinematics, leg_map, servo_min: Sequence[float],
                      servo_max: Sequence[float],
                      checks: Mapping[str, Iterable[Sequence[Sequence[Tuple[float, float]]]]],
                      leg_trim_mm: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
                      leg_trim_x_mm: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
                      interleg: Optional[Sequence] = None) -> EnvelopeReport:
    """Check every planned foot target of every named check on every leg.

    ``checks`` maps a check name (the parameter that controls it) to scenarios, each the
    leg-frame points ``(fx, fz)`` (mm) per leg planned at one body pose. Every trimmed point
    must be reachable inside the joint region and within the servo limits. With
    ``interleg``, no rear x front joint pose pair of a side within a scenario may be in the
    CAD unsafe-pair set; a side without pair data fails every check.
    """
    failed: List[str] = []
    messages: List[str] = []
    worst = 0.0
    margin = math.inf
    region_margin = math.inf
    il_steps = INTERLEG_MARGIN_MAX_STEPS        # smallest margin (grid steps) seen
    il_step_deg = math.inf
    il_seen = False
    il_problem = None
    il_failed: List[str] = []
    n = 0
    legs = leg_map.legs
    for name, scenarios in checks.items():
        problem = None
        for sc in scenarios:
            joints: List[List[Tuple[float, float]]] = []
            for li, leg in enumerate(legs):
                guess = None
                qs = []
                for fx, fz in sc[li]:
                    fz_t = fz + leg_trim_mm[li]
                    fx_t = fx + leg_trim_x_mm[li]
                    x, y = leg_map.body_to_canonical(li, fx_t, fz_t)
                    r = kin.ik(x, y, guess)
                    n += 1
                    if not r.ok:
                        problem = (f'{name}: leg {leg.name} cannot reach fx={fx_t:.1f} '
                                   f'fz={fz_t:.1f} mm (IK error {r.err_mm:.1f} mm)')
                        break
                    guess = (r.m1, r.m2)
                    worst = max(worst, r.err_mm)
                    qs.append((r.m1, r.m2))
                    if not kin.region.is_box():
                        region_margin = min(region_margin, kin.region.margin(r.m1, r.m2))
                    s_t, s_c = leg_map.to_servo(li, r.m1, r.m2)
                    for idx, s in ((leg.thigh_idx, s_t), (leg.calf_idx, s_c)):
                        lo, hi = servo_min[idx], servo_max[idx]
                        if not lo <= s <= hi:
                            problem = (f'{name}: leg {leg.name} servo {idx} needs {s:.1f} deg '
                                       f'outside limits {lo:.0f}..{hi:.0f} at fx={fx_t:.1f} '
                                       f'fz={fz_t:.1f}')
                            break
                        margin = min(margin, s - lo, hi - s)
                    if problem:
                        break
                if problem:
                    break
                joints.append(qs)
            if problem:
                break
            for side in interleg or ():
                p, k = _interleg_check(legs, side, joints, getattr(sc, 'pose', None), il_steps)
                if side.pairs is not None and k is not None:
                    il_seen = True
                    il_step_deg = side.pairs.step
                    il_steps = min(il_steps, k)
                if p:
                    if name not in il_failed:
                        il_failed.append(name)
                    if il_problem is None:
                        il_problem = f'{name}: {p}'
        if problem:
            failed.append(name)
            messages.append(problem)
    if il_problem:
        failed.append(INTERLEG_CHECK)
        messages.append(f'{INTERLEG_CHECK}: {il_problem}')
    il_margin = max(0, il_steps) * il_step_deg if il_seen else math.inf
    return EnvelopeReport(not failed, failed, messages, worst, margin, n, region_margin,
                          il_margin, il_seen and il_steps >= INTERLEG_MARGIN_MAX_STEPS,
                          il_failed)


def _pose_text(pose) -> str:
    if pose is None:
        return ''
    return (f' at body height {pose.height_mm:.0f} mm, pitch {pose.pitch_deg:+.1f}, roll '
            f'{pose.roll_deg:+.1f} deg')


def _interleg_check(legs, side, joints: Sequence[Sequence[Tuple[float, float]]], pose,
                    cap_steps: int):
    """(problem or None, margin in grid steps or None) of one side in one scenario."""
    rear, front = legs[side.rear].name, legs[side.front].name
    pairs = side.pairs
    if pairs is None:
        return f'legs {rear} / {front}: {side.problem}', None
    rq, fq = joints[side.rear], joints[side.front]
    for leg, qs in ((rear, rq), (front, fq)):
        for q in qs:
            if not pairs.covers(q[0], q[1]):
                where = pairs.path.name if pairs.path else 'the pair sweep'
                return (f'leg {leg} (m1 {q[0]:.1f}, m2 {q[1]:.1f}){_pose_text(pose)} is outside '
                        f'the joint region {where} covers: re-run `pixi run pair_sweep_final` '
                        'for this joint region'), None
    k = pairs.margin_steps_of(rq, fq, max(cap_steps, pairs.margin_steps))
    if k >= pairs.margin_steps:
        return None, k
    hit = pairs.first_marked(rq, fq)
    (a, b), (c, d) = hit if hit else ((math.nan, math.nan), (math.nan, math.nan))
    return (f'legs {rear} / {front} may collide{_pose_text(pose)}: {rear} (m1 {a:.1f}, m2 '
            f'{b:.1f}) with {front} (m1 {c:.1f}, m2 {d:.1f}) is within '
            f'{pairs.margin_steps * pairs.step:g} deg of a contact of the CAD pair sweep'), k
