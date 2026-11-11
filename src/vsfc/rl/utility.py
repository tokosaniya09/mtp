"""Input-constrained utility and tanh policy, paper Sec. IV eqs. (11)-(13), (15), (18), (25)-(26).

Generic in the input bound and weight so the attitude loop (Sec. V, eqs. (34)-(36), (42)) reuses it:
`bound` is beta_j (thrust vector, N) or gamma_j (attitude input); `r` is the diagonal of R_j or D_j.

[Paper] (11)/(12)  U(mu) = 2 int_0^mu beta (tanh^{-1}(upsilon/beta))^T R d upsilon
[Paper] (13)       U(mu) = 2 beta tanh^{-1}(mu/beta)^T R mu + beta^2 Rbar^T ln(1 - (mu/beta)^2)
[Paper] (15)/(18)  mu = -beta tanh(xi),  xi = (1/(2 beta)) R^{-1} B^T grad V
[Paper] (25)       mu_hat = -beta tanh(xi_hat)
[Paper] (26)       U(mu_hat) = 2 beta^2 xi_hat^T R tanh(xi_hat) + beta^2 Rbar^T ln(1 - tanh^2(xi_hat))

R = diag(r) must be diagonal: only then is the integrand of (11) a gradient field, so the integral is
path-independent and (13) holds (the paper states R_j = diag{r_j1, r_j2, r_j3}). Rbar = r (printed as a
column vector; Rbar^T ln(.) is meant, see docs/equation_map.md §5.2). ln and squares are elementwise.

Constraint note (A2): tanh acts per component, so |mu_k| < beta for each k (an infinity-norm bound).

[Eng] numerical stability: ln(1 - tanh^2 xi) = -2 ln cosh xi, evaluated as
-2 (|xi| + log1p(exp(-2|xi|)) - ln 2). Identical mathematically; finite for any xi (the literal form
underflows to ln 0 = -inf for |xi| > ~19).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def _check_inside(mu: np.ndarray, bound: float) -> np.ndarray:
    mu = np.asarray(mu, float)
    if np.any(np.abs(mu) >= bound):
        raise ValueError(f"utility defined only for |mu_k| < bound = {bound}; got mu = {mu}")
    return mu


def utility_closed_form(mu: np.ndarray, bound: float, r: np.ndarray) -> float:
    """Eq. (13). mu, r: (m,); bound: scalar > 0; defined for |mu_k| < bound.

    Per component U_k = beta^2 r_k f(x), x = mu_k/beta, f(x) = (1+x)ln(1+x) + (1-x)ln(1-x) >= 0, so U(0) = 0
    and U stays BOUNDED (f -> 2 ln 2 as |x| -> 1). The bound on mu is enforced by the gradient
    dU/dmu = 2 beta R tanh^{-1}(mu/beta), which diverges at the boundary, not by U itself.
    """
    mu = _check_inside(mu, bound)
    r = np.asarray(r, float)
    x = mu / bound
    return float(2.0 * bound * np.arctanh(x) @ (r * mu) + bound ** 2 * r @ np.log1p(-x ** 2))


def utility_numeric(mu: np.ndarray, bound: float, r: np.ndarray) -> float:
    """Eq. (11)/(12) by quadrature along the straight path upsilon = s mu, s in [0, 1] (test reference).

    U = 2 int_0^1 beta tanh^{-1}(s mu / beta)^T R mu ds.
    """
    mu = _check_inside(mu, bound)
    r = np.asarray(r, float)
    val, _ = quad(lambda s: bound * np.arctanh(s * mu / bound) @ (r * mu), 0.0, 1.0, epsabs=1e-13, epsrel=1e-12)
    return 2.0 * val


def utility_gradient(mu: np.ndarray, bound: float, r: np.ndarray) -> np.ndarray:
    """dU/dmu = 2 beta R tanh^{-1}(mu/beta), (m,). Used in the stationarity condition behind (15) and in (20)."""
    mu = _check_inside(mu, bound)
    return 2.0 * bound * np.asarray(r, float) * np.arctanh(mu / bound)


def policy_from_xi(xi: np.ndarray, bound: float) -> np.ndarray:
    """Eqs. (15), (18), (25): mu = -beta tanh(xi), (m,) or (N, m)."""
    return -bound * np.tanh(np.asarray(xi, float))


def xi_from_value_gradient(grad_V: np.ndarray, B: np.ndarray, bound: float, r: np.ndarray) -> np.ndarray:
    """Eqs. (15), (18), model-based: xi = (1/(2 beta)) R^{-1} B^T grad V, (m,). grad_V: (n,); B: (n, m).

    Not used by the off-policy learner (which never sees B); used for tests and the LQR cross-check.
    """
    return (np.asarray(B, float).T @ np.asarray(grad_V, float)) / (2.0 * bound * np.asarray(r, float))


def log_one_minus_tanh_sq(xi: np.ndarray) -> np.ndarray:
    """ln(1 - tanh^2 xi) = -2 ln cosh xi, numerically stable for any xi [Eng]. Elementwise."""
    a = np.abs(np.asarray(xi, float))
    return -2.0 * (a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0))


def utility_from_xi(xi: np.ndarray, bound: float, r: np.ndarray) -> np.ndarray:
    """Eq. (26): U(-beta tanh xi) = 2 beta^2 xi^T R tanh(xi) + beta^2 Rbar^T ln(1 - tanh^2 xi).

    xi: (m,) -> scalar, or (N, m) -> (N,) (batch over time samples, used for the integrals in (27)).
    """
    xi = np.asarray(xi, float)
    r = np.asarray(r, float)
    return (2.0 * bound ** 2 * (xi * np.tanh(xi)) @ r + bound ** 2 * log_one_minus_tanh_sq(xi) @ r)
