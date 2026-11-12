"""Off-policy data for integral RL, paper Alg. 3 Step 1 (and Alg. 4 Step 1).

[Paper] "Collect the sampled data for the follower quadrotor j by using an admissible control policy mu_j."
The Bellman rows (27)-(28) need integrals over windows [t, t + dT] of functions of the state and of the
APPLIED (behaviour) input. Because those integrands depend on the current actor weights, they are
re-integrated from the stored trajectory at every iteration, so the fine grid is stored, not just window
ends (docs/equation_map.md §5.6, A18).

[Eng] A18: the input is held constant over each integration step h (zero-order hold, as in
`sim/integrators.py`). A trajectory is stored as
    delta: (K + 1, n)  learning state at the grid points t_0, ..., t_K
    mu:    (K, m)      input applied on [t_i, t_{i+1})
and cut into M consecutive windows of W = dT / h steps (K = M W).

This module is plant-agnostic: `rollout` takes any x_dot = f(t, x, u), an `observe(x) -> delta` map
(identity for the tests; [e, v_hat] from the observer for the quadrotor) and a behaviour policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from vsfc.sim.integrators import rk4_step


@dataclass(frozen=True)
class OffPolicyDataset:
    """Windowed trajectory data.

    delta: (M, W + 1, n)  state on each window's grid (window ends shared with neighbours)
    mu:    (M, W, m)      applied input on each sub-step (ZOH)
    h:     sub-step length, s;  dT = W h.
    """

    delta: np.ndarray
    mu: np.ndarray
    h: float

    def __post_init__(self):
        if self.delta.ndim != 3 or self.mu.ndim != 3:
            raise ValueError("delta must be (M, W+1, n) and mu (M, W, m)")
        M, W1, _ = self.delta.shape
        if self.mu.shape[:2] != (M, W1 - 1):
            raise ValueError(f"shape mismatch: delta {self.delta.shape}, mu {self.mu.shape}")

    @property
    def n_windows(self) -> int:
        return self.delta.shape[0]

    @property
    def steps_per_window(self) -> int:
        return self.mu.shape[1]

    @property
    def dT(self) -> float:
        return self.steps_per_window * self.h

    @classmethod
    def from_trajectory(cls, delta: np.ndarray, mu: np.ndarray, h: float, steps_per_window: int):
        """Cut a trajectory (delta (K+1, n), mu (K, m)) into windows of `steps_per_window` steps.

        Trailing steps that do not fill a whole window are dropped.
        """
        delta = np.asarray(delta, float)
        mu = np.asarray(mu, float)
        if delta.shape[0] != mu.shape[0] + 1:
            raise ValueError(f"need len(delta) = len(mu) + 1, got {delta.shape[0]} and {mu.shape[0]}")
        W = int(steps_per_window)
        M = mu.shape[0] // W
        if M < 1:
            raise ValueError("trajectory shorter than one window")
        idx = np.arange(M)[:, None] * W + np.arange(W + 1)[None, :]           # (M, W+1)
        return cls(delta=delta[idx], mu=mu[: M * W].reshape(M, W, -1), h=float(h))

    @staticmethod
    def concatenate(datasets: list["OffPolicyDataset"]) -> "OffPolicyDataset":
        """Stack windows from several runs (e.g. several initial conditions). All must share h and W."""
        h = datasets[0].h
        if any(not np.isclose(d.h, h) or d.steps_per_window != datasets[0].steps_per_window for d in datasets):
            raise ValueError("datasets must share h and steps_per_window")
        return OffPolicyDataset(delta=np.concatenate([d.delta for d in datasets]),
                                mu=np.concatenate([d.mu for d in datasets]), h=h)


def rollout(f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
            x0: np.ndarray,
            behaviour: Callable[[float, np.ndarray], np.ndarray],
            T: float, h: float,
            observe: Callable[[np.ndarray], np.ndarray] | None = None,
            post_step: Callable[[np.ndarray], np.ndarray] | None = None):
    """Simulate x_dot = f(t, x, u) with RK4 and ZOH input u_i = behaviour(t_i, x_i) on [t_i, t_{i+1}).

    Returns (t (K+1,), x (K+1, n_x), delta (K+1, n), mu (K, m)), with delta = observe(x) (identity if None).
    """
    K = int(round(T / h))
    if not np.isclose(K * h, T, rtol=0, atol=1e-9 * max(1.0, T)):
        raise ValueError(f"T = {T} is not an integer multiple of h = {h}")
    observe = observe or (lambda x: x)
    x = np.asarray(x0, float).copy()
    t = h * np.arange(K + 1)
    X = np.empty((K + 1, x.size))
    X[0] = x
    U = []
    for i in range(K):
        u = np.atleast_1d(np.asarray(behaviour(t[i], x), float))
        x = rk4_step(lambda tt, xx: f(tt, xx, u), t[i], x, h, post_step)
        X[i + 1] = x
        U.append(u)
    delta = np.array([observe(xi) for xi in X])
    return t, X, delta, np.array(U)


def sum_of_sinusoids(amplitude, frequencies, m: int, seed: int = 0):
    """[Eng] A17 exploration signal: e(t)_k = sum_q A_{q,k} sin(omega_q t + phi_{q,k}), random phases.

    amplitude: scalar, (m,) per channel, or (Q, m) per frequency and channel (D26).
    Returns a function t -> (m,). Used as the additive excitation of the behaviour policy.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(frequencies, float)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(w.size, m))
    A = np.broadcast_to(np.asarray(amplitude, float), (w.size, m))
    return lambda t: (A * np.sin(w[:, None] * t + phase)).sum(axis=0)
