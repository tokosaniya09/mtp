"""Adaptive velocity observer, paper Sec. III, eqs. (6)-(9), Theorem 1.

[Paper] (6), with e_tilde = e - e_hat, v_tilde = v_ij - v_hat:
    e_hat_dot    = v_hat / x* + alpha e_tilde
    v_hat_dot    = -g e3 + R_j f_Tj e3 / m_j + zeta_bar - e_tilde / x* + kappa rho
    zeta_bar     = zeta_hat sat(rho, eps)                    (elementwise)
    zeta_hat_dot = x* ||rho||
    rho          = e_tilde_dot + alpha e_tilde
    sat(r, eps)  = r / max(|r|, eps)

[Paper] (7) error dynamics (zeta_tilde = zeta_i - zeta_hat):
    e_tilde_dot    = v_tilde / x* - alpha e_tilde
    v_tilde_dot    = v_dot_i - zeta_bar + e_tilde / x* - kappa v_tilde / x*
    zeta_tilde_dot = -||v_tilde||
[Paper] (8) L_o = 1/2 |e_tilde|^2 + 1/2 |v_tilde|^2 + 1/2 zeta_tilde^2
[Paper] (9) L_o_dot <= -alpha |e_tilde|^2 - kappa/x* |v_tilde|^2 + 2/x* e_tilde.v_tilde
[Paper] Theorem 1: (9) is negative definite in (e_tilde, v_tilde) iff alpha kappa > 1/x*.

Identity used throughout (docs/equation_map.md §4): rho = e_dot - v_hat/x* = v_tilde/x*.
Caveats: (9) holds only outside the boundary layer |rho_k| < eps, so convergence is to a neighbourhood
(A5); zeta_hat only grows (no leakage, A5); the observer needs e_dot, whose source is pluggable (A4, D15);
it uses m_j, g and the physical thrust vector R_j f_Tj e3 (A22). SI units, implicit unit gains (A21).

The observer is plant-agnostic: it takes measurements and the follower's thrust vector and returns the
time derivative of its own state. Observer state layout [Eng]: [e_hat (3), v_hat (3), zeta_hat (1)].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vsfc.dynamics import E3

STATE_DIM = 7


def sat(rho: np.ndarray, eps: float) -> np.ndarray:
    """Elementwise saturation of eq. (6): rho / max(|rho|, eps). sign(rho) if |rho| >= eps, rho/eps inside."""
    rho = np.asarray(rho, float)
    return rho / np.maximum(np.abs(rho), eps)


def gain_condition(alpha: float, kappa: float, x_star: float) -> bool:
    """Theorem 1: alpha kappa > 1/x*."""
    return alpha * kappa > 1.0 / x_star


def pack_observer_state(e_hat, v_hat, zeta_hat) -> np.ndarray:
    return np.concatenate([np.asarray(e_hat, float), np.asarray(v_hat, float), [float(zeta_hat)]])


def unpack_observer_state(z: np.ndarray):
    """Return (e_hat (3,), v_hat (3,), zeta_hat scalar)."""
    return z[0:3], z[3:6], z[6]


@dataclass(frozen=True)
class AdaptiveVelocityObserver:
    """Observer of eq. (6) for one leader-follower pair (i, j).

    alpha: 1/s; kappa: m/s (A21); eps: 1/s (units of rho); x_star: m; m_j: kg; g: m/s^2.
    """

    alpha: float
    kappa: float
    eps: float
    x_star: float
    m_j: float
    g: float

    @classmethod
    def from_config(cls, follower, g: float) -> "AdaptiveVelocityObserver":
        """Build from a `vsfc.config.FollowerConfig` (gains from paper Sec. VI)."""
        o = follower.observer
        return cls(alpha=o.alpha, kappa=o.kappa, eps=o.eps, x_star=follower.x_star, m_j=follower.mass, g=g)

    def signals(self, z: np.ndarray, e_meas: np.ndarray, e_dot_meas: np.ndarray):
        """Internal signals of eq. (6): (e_tilde, rho, zeta_bar), each (3,).

        rho is computed literally as e_tilde_dot + alpha e_tilde with e_tilde_dot = e_dot - e_hat_dot;
        algebraically this equals e_dot - v_hat/x*.
        """
        e_hat, v_hat, zeta_hat = unpack_observer_state(z)
        e_tilde = np.asarray(e_meas, float) - e_hat
        e_hat_dot = v_hat / self.x_star + self.alpha * e_tilde
        e_tilde_dot = np.asarray(e_dot_meas, float) - e_hat_dot
        rho = e_tilde_dot + self.alpha * e_tilde
        zeta_bar = zeta_hat * sat(rho, self.eps)
        return e_tilde, rho, zeta_bar

    def derivatives(self, z: np.ndarray, e_meas: np.ndarray, e_dot_meas: np.ndarray,
                    thrust_vec: np.ndarray) -> np.ndarray:
        """Eq. (6): time derivative of the observer state z = [e_hat, v_hat, zeta_hat], shape (7,).

        e_meas: measured feature error (3,); e_dot_meas: its time derivative (3,) 1/s, source pluggable
        (D15); thrust_vec: physical thrust vector R_j f_Tj e3 of follower j, (3,) N.
        """
        _, v_hat, _ = unpack_observer_state(z)
        e_tilde, rho, zeta_bar = self.signals(z, e_meas, e_dot_meas)
        e_hat_dot = v_hat / self.x_star + self.alpha * e_tilde
        v_hat_dot = (-self.g * E3 + np.asarray(thrust_vec, float) / self.m_j + zeta_bar
                     - e_tilde / self.x_star + self.kappa * rho)
        zeta_hat_dot = self.x_star * np.linalg.norm(rho)
        return np.concatenate([e_hat_dot, v_hat_dot, [zeta_hat_dot]])

    def gain_condition(self) -> bool:
        return gain_condition(self.alpha, self.kappa, self.x_star)


# ---------- eqs. (7)-(9): analysis helpers (used by tests) ----------

def error_dynamics(e_tilde, v_tilde, zeta_hat, a_leader, alpha, kappa, eps, x_star):
    """Eq. (7): (e_tilde_dot (3,), v_tilde_dot (3,), zeta_tilde_dot scalar).

    zeta_bar = zeta_hat sat(v_tilde/x*, eps), using rho = v_tilde/x*. zeta_i is constant, so
    zeta_tilde_dot = -zeta_hat_dot = -||v_tilde||.
    """
    e_tilde, v_tilde = np.asarray(e_tilde, float), np.asarray(v_tilde, float)
    zeta_bar = zeta_hat * sat(v_tilde / x_star, eps)
    e_tilde_dot = v_tilde / x_star - alpha * e_tilde
    v_tilde_dot = np.asarray(a_leader, float) - zeta_bar + e_tilde / x_star - kappa * v_tilde / x_star
    return e_tilde_dot, v_tilde_dot, -np.linalg.norm(v_tilde)


def lyapunov(e_tilde, v_tilde, zeta_tilde) -> float:
    """Eq. (8): L_o."""
    return 0.5 * float(np.dot(e_tilde, e_tilde) + np.dot(v_tilde, v_tilde) + zeta_tilde ** 2)


def lyapunov_rate_bound(e_tilde, v_tilde, alpha, kappa, x_star) -> float:
    """Right-hand side of eq. (9) (valid outside the boundary layer, A5)."""
    e_tilde, v_tilde = np.asarray(e_tilde, float), np.asarray(v_tilde, float)
    return float(-alpha * e_tilde @ e_tilde - (kappa / x_star) * v_tilde @ v_tilde
                 + (2.0 / x_star) * e_tilde @ v_tilde)
