"""Polynomial basis functions for the critic and actor networks, paper eqs. (23)-(24) (and (40)-(41)).

[Paper] (23) V_hat(delta) = w_c^T sigma_c(delta),   (24) xi_hat_k(delta) = w_a,k^T sigma_a,k(delta).
Sec. VI says only that the bases are "polynomial combinations" of the state; the exact sets, l_c and l_a are
not given (A16). This module provides monomial bases; WHICH monomials are used is an [Eng] decision logged
where the learner is trained (Phase 1.4b: full bases for the known-answer tests; Phase 1.5: per-axis bases, D25).

`MonomialBasis(dim, exponents)` is the general form: row i of `exponents` (l, n) gives sigma_i(x) = prod_j x_j^E_ij.

`PolynomialBasis(dim, degrees)` contains every monomial with total degree in `degrees`, in a fixed order: by
degree (ascending, in the order given), then by `itertools.combinations_with_replacement` order of variable
indices. Example n = 2, D = (2,): [x1^2, x1 x2, x2^2]. Count per degree d: C(n + d - 1, d).

`grouped_basis(dim, groups, degrees)` [Eng, D25] contains, for each group of variable indices in turn, every
monomial of total degree in `degrees` that uses ONLY variables of that group (no cross-group products).
Used in Phase 1.5 because the outer-loop optimal value separates by axis (docs/equation_map.md §5.8):
groups (e_k, v_hat_k) = (k, k + 3).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb

import numpy as np


def _monomials(variables, degrees, dim: int) -> list[np.ndarray]:
    """Exponent rows (dim,) of all monomials in `variables` with total degree in `degrees` (ordered as above)."""
    rows = []
    for d in degrees:
        for idx in combinations_with_replacement(tuple(variables), d):
            p = np.zeros(dim, dtype=int)
            for i in idx:
                p[i] += 1
            rows.append(p)
    return rows


def _check_degrees(degrees) -> tuple[int, ...]:
    degrees = tuple(int(d) for d in degrees)
    if not degrees or min(degrees) < 0 or len(set(degrees)) != len(degrees):
        raise ValueError(f"degrees must be distinct non-negative integers, got {degrees}")
    return degrees


class MonomialBasis:
    """Monomial basis sigma: R^n -> R^l with explicit exponent matrix E (l, n), sigma_i(x) = prod_j x_j^E_ij."""

    def __init__(self, dim: int, exponents):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        E = np.asarray(exponents, dtype=int).reshape(-1, dim)
        if E.shape[0] == 0 or np.any(E < 0):
            raise ValueError("exponents must be a non-empty array of non-negative integers")
        if len({tuple(row) for row in E}) != E.shape[0]:
            raise ValueError("exponent rows must be distinct")
        self.dim = dim
        self.exponents = E                                              # (l, n)

    @property
    def size(self) -> int:
        """l, the number of basis functions."""
        return self.exponents.shape[0]

    @property
    def total_degrees(self) -> np.ndarray:
        """Total degree of each basis function, (l,)."""
        return self.exponents.sum(axis=1)

    def labels(self, names=None) -> list[str]:
        names = names or [f"x{i + 1}" for i in range(self.dim)]
        out = []
        for p in self.exponents:
            terms = [n if k == 1 else f"{n}^{k}" for n, k in zip(names, p) if k > 0]
            out.append("*".join(terms) or "1")
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """sigma(x): x (..., n) -> (..., l)."""
        x = np.asarray(x, float)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got shape {x.shape}")
        # x[..., None, :] ** E -> (..., l, n); 0**0 = 1 as required.
        return np.prod(x[..., None, :] ** self.exponents, axis=-1)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """d sigma / dx: x (..., n) -> (..., l, n)."""
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


class PolynomialBasis(MonomialBasis):
    """All monomials on R^n whose total degree is in `degrees` (0 gives a constant term)."""

    def __init__(self, dim: int, degrees):
        degrees = _check_degrees(degrees)
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        super().__init__(dim, _monomials(range(dim), degrees, dim))
        self.degrees = degrees

    @staticmethod
    def count(dim: int, degrees) -> int:
        return sum(comb(dim + d - 1, d) for d in degrees)


def grouped_basis(dim: int, groups, degrees) -> MonomialBasis:
    """[Eng, D25] Union over `groups` (sequence of index tuples) of all monomials of total degree in `degrees`
    using only the variables of that group. Groups must be disjoint. Count = sum_g sum_d C(|g| + d - 1, d)."""
    degrees = _check_degrees(degrees)
    groups = [tuple(int(i) for i in g) for g in groups]
    flat = [i for g in groups for i in g]
    if not groups or any(len(g) == 0 for g in groups):
        raise ValueError("groups must be non-empty")
    if len(set(flat)) != len(flat) or min(flat) < 0 or max(flat) >= dim:
        raise ValueError(f"groups must be disjoint indices in [0, {dim}), got {groups}")
    if 0 in degrees and len(groups) > 1:
        raise ValueError("degree 0 would repeat the constant term in every group")
    rows = []
    for g in groups:
        rows += _monomials(g, degrees, dim)
    return MonomialBasis(dim, rows)


def axis_groups(n_axes: int = 3) -> list[tuple[int, int]]:
    """Groups (e_k, v_hat_k) = (k, k + n_axes) for delta = [e (n_axes,), v_hat (n_axes,)]  (Phase 1.5, D25)."""
    return [(k, k + n_axes) for k in range(n_axes)]


def weights_from_quadratic_form(basis: MonomialBasis, P: np.ndarray) -> np.ndarray:
    """Weights w with w^T sigma(x) = x^T P x (P symmetric).

    The basis must contain a degree-2 monomial for every nonzero entry of P (raises otherwise).
    Used to compare a learned critic with an LQR/Riccati solution (known-answer tests).
    """
    P = np.asarray(P, float)
    if P.shape != (basis.dim, basis.dim) or not np.allclose(P, P.T):
        raise ValueError("P must be symmetric with the basis dimension")
    w = np.zeros(basis.size)
    covered = np.zeros_like(P, dtype=bool)
    for row, p in enumerate(basis.exponents):
        if p.sum() != 2:
            continue
        idx = np.nonzero(p)[0]
        i, j = (idx[0], idx[0]) if idx.size == 1 else (idx[0], idx[1])
        w[row] = P[i, i] if i == j else 2.0 * P[i, j]
        covered[i, j] = covered[j, i] = True
    if not covered.any():
        raise ValueError("basis has no degree-2 monomials")
    if np.any(~covered & (np.abs(P) > 1e-12 * max(1.0, np.abs(P).max()))):
        raise ValueError("P has nonzero entries with no matching degree-2 monomial in the basis")
    return w
