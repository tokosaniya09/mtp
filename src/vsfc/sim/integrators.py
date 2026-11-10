"""Fixed-step classical Runge-Kutta 4 (decision D17, A18).

Inputs are held constant over a step (zero-order hold), as for a sampled controller.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Deriv = Callable[[float, np.ndarray], np.ndarray]
PostStep = Callable[[np.ndarray], np.ndarray]


def rk4_step(f: Deriv, t: float, x: np.ndarray, h: float, post_step: PostStep | None = None) -> np.ndarray:
    """One RK4 step of x_dot = f(t, x); optional post_step (e.g. SO(3) projection)."""
    k1 = f(t, x)
    k2 = f(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, x + 0.5 * h * k2)
    k4 = f(t + h, x + h * k3)
    x_next = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return post_step(x_next) if post_step is not None else x_next


def integrate(f: Deriv, x0: np.ndarray, t0: float, t1: float, h: float,
              post_step: PostStep | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Integrate from t0 to t1 with step h. Returns (t (N+1,), X (N+1, n))."""
    n_steps = int(round((t1 - t0) / h))
    if not np.isclose(n_steps * h, t1 - t0, rtol=0, atol=1e-9 * max(1.0, abs(t1 - t0))):
        raise ValueError(f"(t1 - t0) = {t1 - t0} is not an integer multiple of h = {h}")
    t = t0 + h * np.arange(n_steps + 1)
    X = np.empty((n_steps + 1, x0.size))
    X[0] = x0
    for k in range(n_steps):
        X[k + 1] = rk4_step(f, t[k], X[k], h, post_step)
    return t, X
