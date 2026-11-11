"""Tests for eqs. (11)-(13), (15), (25)-(26): src/vsfc/rl/utility.py."""

import numpy as np
import pytest
from scipy.optimize import minimize

from vsfc.rl.utility import (log_one_minus_tanh_sq, policy_from_xi, utility_closed_form, utility_from_xi,
                             utility_gradient, utility_numeric, xi_from_value_gradient)

BETA = 5.0                       # paper Sec. VI, beta_2 = beta_3
R_DIAG = np.full(3, 8.0)         # paper Sec. VI, R_j = 8 I_3


def test_hand_worked_value():
    # One component: beta = 5, r = 8, mu = 2.5 (x = 0.5):
    # 2*5*8*2.5*atanh(0.5) + 25*8*ln(0.75) = 109.861 - 57.536 = 52.325
    # = beta^2 r [(1+x)ln(1+x) + (1-x)ln(1-x)] = 200 * 0.261624
    assert utility_closed_form([2.5], BETA, [8.0]) == pytest.approx(52.3248, abs=1e-4)
    assert utility_closed_form([2.5, 0, 0], BETA, R_DIAG) == pytest.approx(52.3248, abs=1e-4)


def test_closed_form_matches_quadrature_of_eq11():
    rng = np.random.default_rng(0)
    for _ in range(100):
        r = rng.uniform(0.5, 10, size=3)
        mu = rng.uniform(-0.999, 0.999, size=3) * BETA
        assert utility_closed_form(mu, BETA, r) == pytest.approx(utility_numeric(mu, BETA, r), rel=1e-9, abs=1e-12)


def test_positive_definite_and_bounded():
    rng = np.random.default_rng(1)
    assert utility_closed_form(np.zeros(3), BETA, R_DIAG) == 0.0
    for _ in range(100):
        mu = rng.uniform(-0.99, 0.99, size=3) * BETA
        assert utility_closed_form(mu, BETA, R_DIAG) > 0.0
    # U is bounded: per component -> beta^2 r 2 ln 2 as |mu| -> beta (gradient diverges instead).
    limit = BETA ** 2 * 8.0 * 2 * np.log(2)
    near = utility_closed_form([BETA * (1 - 1e-12)], BETA, [8.0])
    assert near < limit and near == pytest.approx(limit, rel=1e-9)
    assert utility_gradient([BETA * (1 - 1e-12)], BETA, [8.0])[0] > 1e3


def test_rejects_outside_bound():
    for mu in ([BETA, 0, 0], [0, -BETA - 1, 0]):
        with pytest.raises(ValueError, match="bound"):
            utility_closed_form(mu, BETA, R_DIAG)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(2)
    h = 1e-6
    for _ in range(20):
        mu = rng.uniform(-0.95, 0.95, size=3) * BETA
        fd = np.array([(utility_closed_form(mu + h * e, BETA, R_DIAG) - utility_closed_form(mu - h * e, BETA, R_DIAG))
                       / (2 * h) for e in np.eye(3)])
        np.testing.assert_allclose(utility_gradient(mu, BETA, R_DIAG), fd, rtol=1e-6, atol=1e-8)


def test_small_input_limit_is_quadratic():
    # U = r mu^2 (1 + x^2/6 + O(x^4)), so U -> mu^T R mu and the relative error is ~ x^2/6.
    for x in (1e-1, 1e-2, 1e-3):
        mu = np.array([x, -x, 0.5 * x]) * BETA
        U, quad = utility_closed_form(mu, BETA, R_DIAG), mu @ (R_DIAG * mu)
        assert (U - quad) / quad == pytest.approx(x ** 2 * (1 + 1 + 0.5 ** 4) / (6 * 2.25), rel=0.02)


def test_eq15_minimises_hamiltonian():
    """mu* = -beta tanh((1/2beta) R^-1 B^T gradV) minimises U(mu) + gradV^T B mu over the box |mu_k| < beta.

    (The other terms of H in (14) do not depend on mu.)"""
    rng = np.random.default_rng(3)
    B = np.vstack([np.zeros((3, 3)), np.eye(3) / 1.3])            # B_j of eq. (10), m_2 = 1.3 kg
    for _ in range(10):
        gradV = rng.normal(size=6) * 30
        mu_star = policy_from_xi(xi_from_value_gradient(gradV, B, BETA, R_DIAG), BETA)
        assert np.all(np.abs(mu_star) < BETA)

        def H(mu):
            return utility_closed_form(mu, BETA, R_DIAG) + gradV @ B @ mu

        res = minimize(H, np.zeros(3), method="L-BFGS-B", bounds=[(-BETA * (1 - 1e-9), BETA * (1 - 1e-9))] * 3,
                       options={"ftol": 1e-15, "gtol": 1e-12})
        np.testing.assert_allclose(res.x, mu_star, atol=1e-5)
        assert H(mu_star) <= res.fun + 1e-9


def test_eq26_equals_eq13_of_tanh_policy():
    rng = np.random.default_rng(4)
    for _ in range(100):
        xi = rng.normal(size=3) * 2
        U26 = utility_from_xi(xi, BETA, R_DIAG)
        U13 = utility_closed_form(policy_from_xi(xi, BETA), BETA, R_DIAG)
        assert U26 == pytest.approx(U13, rel=1e-9)


def test_eq26_batch_and_large_xi_stable():
    xi = np.array([[0.1, -0.2, 0.3], [30.0, -50.0, 800.0]])
    U = utility_from_xi(xi, BETA, R_DIAG)
    assert U.shape == (2,) and np.all(np.isfinite(U))
    assert U[0] == pytest.approx(utility_from_xi(xi[0], BETA, R_DIAG))
    # literal ln(1 - tanh^2) underflows here; stable form equals -2 ln cosh
    with np.errstate(divide="ignore"):
        assert np.isneginf(np.log(1 - np.tanh(30.0) ** 2))
    assert log_one_minus_tanh_sq(30.0) == pytest.approx(-2 * (30 - np.log(2)), rel=1e-12)
    np.testing.assert_allclose(log_one_minus_tanh_sq(np.array([0.0, 0.5, -2.0])),
                               np.log(1 - np.tanh([0.0, 0.5, -2.0]) ** 2), rtol=1e-12, atol=1e-15)
