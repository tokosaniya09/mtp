"""Tests for the polynomial basis, eqs. (23)-(24): src/vsfc/rl/basis.py."""

import numpy as np
import pytest

from vsfc.rl.basis import PolynomialBasis, weights_from_quadratic_form


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
