"""Tests for the polynomial basis, eqs. (23)-(24): src/vsfc/rl/basis.py."""

import numpy as np
import pytest

from vsfc.rl.basis import (MonomialBasis, PolynomialBasis, axis_groups, grouped_basis,
                           weights_from_quadratic_form)


def test_sizes():
    assert PolynomialBasis(6, [2]).size == 21                    # quadratic critic on delta in R^6
    assert PolynomialBasis(6, [1]).size == 6
    assert PolynomialBasis(6, [1, 3]).size == 6 + 56
    assert PolynomialBasis(6, [2, 4]).size == 21 + 126
    assert PolynomialBasis.count(12, [2]) == 78                  # attitude state z in R^12
    for dim, deg in [(3, [0, 1, 2]), (6, [2, 4]), (12, [1, 3])]:
        assert PolynomialBasis(dim, deg).size == PolynomialBasis.count(dim, deg)


def test_hand_example_and_order():
    b = PolynomialBasis(2, [2])
    assert b.labels() == ["x1^2", "x1*x2", "x2^2"]
    np.testing.assert_allclose(b([2.0, 3.0]), [4.0, 6.0, 9.0])
    b = PolynomialBasis(2, [1, 3])
    assert b.labels() == ["x1", "x2", "x1^3", "x1^2*x2", "x1*x2^2", "x2^3"]
    np.testing.assert_allclose(b([2.0, 3.0]), [2, 3, 8, 12, 18, 27])
    assert PolynomialBasis(2, [0]).labels() == ["1"]


def test_batch_equals_loop():
    b = PolynomialBasis(6, [1, 2, 3])
    X = np.random.default_rng(0).normal(size=(50, 6))
    np.testing.assert_allclose(b(X), np.array([b(x) for x in X]))
    np.testing.assert_allclose(b.jacobian(X), np.array([b.jacobian(x) for x in X]))


@pytest.mark.parametrize("degrees", [[1], [2], [1, 3], [2, 4], [0, 1, 2]])
def test_jacobian_matches_finite_difference(degrees):
    b = PolynomialBasis(6, degrees)
    rng = np.random.default_rng(1)
    h = 1e-6
    for x in [rng.normal(size=6), np.array([0.0, 1.2, 0.0, -0.7, 0.3, 0.0])]:   # includes exact zeros
        fd = np.stack([(b(x + h * e) - b(x - h * e)) / (2 * h) for e in np.eye(6)], axis=1)
        np.testing.assert_allclose(b.jacobian(x), fd, rtol=1e-6, atol=1e-8)


def test_quadratic_form_weights():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 6))
    P = A @ A.T
    b = PolynomialBasis(6, [2, 4])
    w = weights_from_quadratic_form(b, P)
    for x in rng.normal(size=(20, 6)):
        assert w @ b(x) == pytest.approx(x @ P @ x, rel=1e-12)
        np.testing.assert_allclose(w @ b.jacobian(x), 2 * P @ x, rtol=1e-10)   # grad V = 2 P x
    with pytest.raises(ValueError):
        weights_from_quadratic_form(PolynomialBasis(6, [1, 3]), P)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        PolynomialBasis(0, [1])
    with pytest.raises(ValueError):
        PolynomialBasis(3, [2, 2])
    with pytest.raises(ValueError):
        PolynomialBasis(3, [2])(np.zeros(4))

# ---- per-axis (grouped) bases, Phase 1.5 (D25) -----------------------------------------------------------

def test_polynomial_is_monomial_with_same_exponents():
    b = PolynomialBasis(4, [1, 3])
    m = MonomialBasis(4, b.exponents)
    X = np.random.default_rng(3).normal(size=(10, 4))
    np.testing.assert_array_equal(b(X), m(X))
    np.testing.assert_array_equal(b.jacobian(X), m.jacobian(X))


def test_axis_groups():
    assert axis_groups() == [(0, 3), (1, 4), (2, 5)]


def test_per_axis_critic_and_actor_sizes_and_labels():
    names = ["e1", "e2", "e3", "v1", "v2", "v3"]
    critic = grouped_basis(6, axis_groups(), (2, 4))
    assert critic.size == 3 * (3 + 5) == 24
    assert critic.labels(names)[:8] == ["e1^2", "e1*v1", "v1^2", "e1^4", "e1^3*v1", "e1^2*v1^2", "e1*v1^3", "v1^4"]
    assert critic.labels(names)[8] == "e2^2"
    actors = [grouped_basis(6, [g], (1, 3)) for g in axis_groups()]
    assert [a.size for a in actors] == [6, 6, 6]
    assert actors[1].labels(names) == ["e2", "v2", "e2^3", "e2^2*v2", "e2*v2^2", "v2^3"]
    # no monomial mixes variables of two axes
    for row in critic.exponents:
        used = {i % 3 for i in np.nonzero(row)[0]}
        assert len(used) == 1


def test_grouped_basis_values_and_jacobian():
    b = grouped_basis(6, axis_groups(), (2, 4))
    x = np.array([0.3, -0.7, 0.2, 1.1, -0.4, 0.9])
    v = b(x)
    assert v[0] == pytest.approx(0.3 ** 2) and v[1] == pytest.approx(0.3 * 1.1) and v[7] == pytest.approx(1.1 ** 4)
    J = b.jacobian(x)
    eps = 1e-6
    Jfd = np.stack([(b(x + eps * np.eye(6)[k]) - b(x - eps * np.eye(6)[k])) / (2 * eps) for k in range(6)], axis=-1)
    np.testing.assert_allclose(J, Jfd, atol=1e-8)


def test_grouped_basis_validation():
    with pytest.raises(ValueError):
        grouped_basis(6, [(0, 3), (3, 4)], (2,))        # overlapping groups
    with pytest.raises(ValueError):
        grouped_basis(6, [(0, 6)], (2,))                # index out of range
    with pytest.raises(ValueError):
        grouped_basis(6, axis_groups(), (0, 2))         # constant would repeat per group
    with pytest.raises(ValueError):
        MonomialBasis(2, [[1, 0], [1, 0]])              # duplicate rows


def test_quadratic_form_weights_for_block_diagonal_P():
    """Per-axis basis represents x^T P x exactly when P only couples (e_k, v_k); rejects cross-axis P."""
    b = grouped_basis(6, axis_groups(), (2, 4))
    P = np.zeros((6, 6))
    for k in range(3):
        P[np.ix_([k, k + 3], [k, k + 3])] = [[50.3, 11.6], [11.6, 14.6]]
    w = weights_from_quadratic_form(b, P)
    X = np.random.default_rng(4).normal(size=(20, 6))
    np.testing.assert_allclose(b(X) @ w, np.einsum("ni,ij,nj->n", X, P, X), rtol=1e-12)
    P[0, 1] = P[1, 0] = 1.0                             # e1-e2 coupling: not representable
    with pytest.raises(ValueError):
        weights_from_quadratic_form(b, P)
