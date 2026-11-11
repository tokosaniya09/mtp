"""Polynomial basis functions for the critic and actor networks, paper eqs. (23)-(24) (and (40)-(41)).

[Paper] (23) V_hat(delta) = w_c^T sigma_c(delta),   (24) xi_hat_k(delta) = w_a,k^T sigma_a,k(delta).
Sec. VI says only that the bases are "polynomial combinations" of the state; the exact sets, l_c and l_a are
not given (A16). This module provides a generic monomial basis; WHICH degrees are used is an [Eng] decision
made when the learner is trained (Phase 1.4b/1.5) and logged there.

A basis with degree set D on R^n contains every monomial prod_i x_i^{p_i} with sum_i p_i in D, in a fixed
order: by degree (ascending, in the order given), then by `itertools.combinations_with_replacement` order of
variable indices. Example n = 2, D = (2,): [x1^2, x1 x2, x2^2]. Count per degree d: C(n + d - 1, d).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb

import numpy as np


class PolynomialBasis:
    """Monomial basis sigma: R^n -> R^l.

    dim: n; degrees: iterable of total degrees (each >= 0; 0 gives a constant term).
    """

    def __init__(self, dim: int, degrees):
        degrees = tuple(int(d) for d in degrees)
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if not degrees or min(degrees) < 0 or len(set(degrees)) != len(degrees):
            raise ValueError(f"degrees must be distinct non-negative integers, got {degrees}")
        self.dim = dim
        self.degrees = degrees
        rows = []
        for d in degrees:
            for idx in combinations_with_replacement(range(dim), d):
                p = np.zeros(dim, dtype=int)
                for i in idx:
                    p[i] += 1
                rows.append(p)
        self.exponents = np.array(rows, dtype=int).reshape(-1, dim)   # (l, n)

    @property
    def size(self) -> int:
        """l = sum over degrees of C(n + d - 1, d)."""
        return self.exponents.shape[0]

    @staticmethod
    def count(dim: int, degrees) -> int:
        return sum(comb(dim + d - 1, d) for d in degrees)

    def labels(self, names=None) -> list[str]:
        names = names or [f"x{i + 1}" for i in range(self.dim)]
        out = []
        for p in self.exponents:
            terms = [n if k == 1 else f"{n}^{k}" for n, k in zip(names, p) if k > 0]
            out.append("*".join(terms) or "1")
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """sigma(x): x (n,) -> (l,), or x (N, n) -> (N, l)."""
        x = np.asarray(x, float)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got shape {x.shape}")
        # x[..., None, :] ** E -> (..., l, n); 0**0 = 1 as required.
        return np.prod(x[..., None, :] ** self.exponents, axis=-1)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """d sigma / dx: x (n,) -> (l, n), or (N, n) -> (N, l, n)."""
        x = np.asarray(x, float)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got shape {x.shape}")
        E = self.exponents
        powers = x[..., None, :] ** E                                    # (..., l, n)
        J = np.empty(x.shape[:-1] + E.shape)
        for k in range(self.dim):
            Ek = E[:, k]
            # derivative of x_k^p: p x_k^(p-1); written to avoid 0**(-1) when p = 0
            dk = np.where(Ek > 0, Ek * x[..., None, k] ** np.maximum(Ek - 1, 0), 0.0)
            others = np.prod(np.delete(powers, k, axis=-1), axis=-1)     # (..., l)
            J[..., k] = dk * others
        return J


def weights_from_quadratic_form(basis: PolynomialBasis, P: np.ndarray) -> np.ndarray:
    """Weights w with w^T sigma(x) = x^T P x (P symmetric), for a basis containing all degree-2 monomials.

    Used to compare a learned critic with an LQR/Riccati solution (Phase 1.4b known-answer test).
    """
    P = np.asarray(P, float)
    if 2 not in basis.degrees:
        raise ValueError("basis has no degree-2 monomials")
    if not np.allclose(P, P.T):
        raise ValueError("P must be symmetric")
    w = np.zeros(basis.size)
    for row, p in enumerate(basis.exponents):
        if p.sum() != 2:
            continue
        idx = np.nonzero(p)[0]
        w[row] = P[idx[0], idx[0]] if idx.size == 1 else 2.0 * P[idx[0], idx[1]]
    return w
