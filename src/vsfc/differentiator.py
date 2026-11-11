"""Filtered finite-difference differentiator for the observer's e_dot input [Eng] (A4, D15).

The paper does not say how e_dot is obtained. D15: reproduce with the exact e_dot from simulator truth,
and re-run with this realistic source as a check. Discrete first-order "dirty derivative":

    d_k = (e_k - e_{k-1}) / dt                (raw backward difference of sampled e)
    y_k = a y_{k-1} + (1 - a) d_k,  a = tau / (tau + dt)

i.e. a backward-Euler discretisation of y = s/(tau s + 1) e. tau = 0 gives the raw difference.
"""

from __future__ import annotations

import numpy as np


class FilteredDifferentiator:
    """Stateful differentiator. dt: sample period (s); tau: low-pass time constant (s), >= 0."""

    def __init__(self, dt: float, tau: float):
        if not dt > 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if tau < 0.0:
            raise ValueError(f"tau must be >= 0, got {tau}")
        self.dt = float(dt)
        self.a = tau / (tau + dt)
        self._e_prev: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def reset(self, e0: np.ndarray, e_dot0: np.ndarray | None = None) -> np.ndarray:
        """Start from sample e0; initial derivative estimate e_dot0 (default 0). Returns the estimate."""
        self._e_prev = np.array(e0, float)
        self._y = np.zeros_like(self._e_prev) if e_dot0 is None else np.array(e_dot0, float)
        return self._y.copy()

    def update(self, e: np.ndarray) -> np.ndarray:
        """Feed the next sample (dt after the previous one); return the filtered derivative estimate."""
        if self._e_prev is None:
            raise RuntimeError("call reset() before update()")
        e = np.asarray(e, float)
        d = (e - self._e_prev) / self.dt
        self._y = self.a * self._y + (1.0 - self.a) * d
        self._e_prev = e.copy()
        return self._y.copy()
