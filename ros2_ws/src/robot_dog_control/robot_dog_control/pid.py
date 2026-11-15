"""PID with filtered P, clamped integrator and derivative on measurement (no ROS).

Sign convention: ``error = target - measured`` and ``rate`` = d(measured)/dt (e.g. the
gyro), so the derivative term is ``-kd * rate``.
"""

from __future__ import annotations

import math
from typing import Optional


class LowPass:
    """First-order IIR low-pass, ``y += a (x - y)`` with ``a = dt / (tau + dt)``."""

    def __init__(self, cutoff_hz: float):
        self.cutoff_hz = float(cutoff_hz)
        self.y: Optional[float] = None

    def reset(self, value: Optional[float] = None) -> None:
        self.y = value

    def update(self, x: float, dt: float) -> float:
        if self.y is None or self.cutoff_hz <= 0.0:
            self.y = x
            return x
        tau = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        self.y += (dt / (tau + dt)) * (x - self.y)
        return self.y


class PID:
    """u = kp*LPF(e) + I - kd*rate, clamped to +-out_limit.

    ``I += ki * LPF(e) * dt`` within +-i_limit, with clamping anti-windup. ``scale``
    multiplies the gains for one update.
    """

    def __init__(self, kp: float, ki: float, kd: float, out_limit: float,
                 i_limit: Optional[float] = None, lpf_hz: float = 0.0,
                 deadband: float = 0.0, deadband_release: float = 1.0):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.out_limit = float(out_limit)
        self.i_limit = float(out_limit if i_limit is None else i_limit)
        self.lpf = LowPass(lpf_hz)
        #: below this |filtered error| the output is held (no update)
        self.deadband = float(deadband)
        #: hysteresis: once the error leaves the deadband, the loop runs until
        #: |filtered error| < deadband * deadband_release (1 gives a plain deadband)
        self.deadband_release = float(deadband_release)
        self.holding = True
        self.integrator = 0.0
        self.last_p = 0.0
        self.last_d = 0.0
        self.output = 0.0

    def reset(self, integrator: float = 0.0) -> None:
        self.integrator = max(-self.i_limit, min(self.i_limit, integrator))
        self.lpf.reset()
        self.last_p = self.last_d = 0.0
        self.output = self.integrator

    def preset(self, output: float, error: float, rate: float, scale: float = 1.0) -> None:
        """Bumpless start: choose the integrator so the next output equals ``output``."""
        self.lpf.reset(error)
        self.holding = abs(error) < self.deadband
        i = output - scale * self.kp * error + scale * self.kd * rate
        self.integrator = max(-self.i_limit, min(self.i_limit, i))
        self.output = output

    def update(self, error: float, rate: float, dt: float, scale: float = 1.0) -> float:
        if not (math.isfinite(error) and math.isfinite(rate) and dt > 0.0):
            return self.output
        e = self.lpf.update(error, dt)
        if self.holding:
            if abs(e) >= self.deadband:
                self.holding = False
        elif abs(e) < self.deadband * self.deadband_release:
            self.holding = True
        if self.holding:
            return self.output          # inside the deadband: hold, do not integrate
        p = scale * self.kp * e
        d = -scale * self.kd * rate
        lim = self.out_limit
        i_new = self.integrator + scale * self.ki * e * dt
        i_new = max(-self.i_limit, min(self.i_limit, i_new))
        u_unsat = p + i_new + d
        if not ((u_unsat > lim and e > 0) or (u_unsat < -lim and e < 0)):
            self.integrator = i_new
        u = p + self.integrator + d
        self.output = max(-lim, min(lim, u))
        self.last_p, self.last_d = p, d
        return self.output

