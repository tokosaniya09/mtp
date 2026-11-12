"""Off-policy integral RL for input-constrained optimal control, paper eqs. (22), (27)-(29), Algorithms 2-3.

Same code serves the attitude loop, eqs. (39)-(41) and Alg. 4, through (bound, r, Q) = (gamma, diag D, S).

[Paper] Critic (23): V_hat(delta) = w_c^T sigma_c(delta).  Actor (24): xi_hat_k(delta) = w_a,k^T sigma_a,k(delta).
[Paper] Policy (25): mu_hat = -beta tanh(xi_hat).  Utility (26), see `rl/utility.py`.
[Paper] Bellman row (27)-(28), written out with R = diag(r) (A3; derivation in docs/equation_map.md §5.6):

    eta_v^T w_c^(n+1) + sum_k eta_bar_k w_a,k^(n+1) = T^(n)

    eta_v     = sigma_c(delta(t)) - sigma_c(delta(t + dT))                                   (l_c,)
    eta_bar_k = 2 beta r_k (eta_w,k + beta eta_u1,k)                                         (l_a,)
    eta_w,k   = int mu_k sigma_a^T dtau                       (applied/behaviour input)       (l_a,)
    eta_u1,k  = int tanh(xi_hat_k^(n)) sigma_a^T dtau                                         (l_a,)
    eta_u2,k  = int ln(1 - tanh^2 xi_hat_k^(n)) dtau                                          scalar
    eta_delta = int delta^T Q delta dtau                                                      scalar
    T^(n)     = eta_delta + 2 beta^2 sum_k r_k eta_u1,k w_a,k^(n) + beta^2 sum_k r_k eta_u2,k

    (the T^(n) terms after eta_delta are exactly int U(mu_hat^(n)) dtau, by (26)).
[Paper] (29): w^(n+1) = (N^T N)^{-1} N^T T, stacked over M windows; w = [w_c; w_a,1; ...; w_a,m].
[Paper] Alg. 3 Step 4: stop when ||w^(n+1) - w^(n)|| < varrho_3.

[Eng] decisions in this module (logged as D23, D25):
  - actor: either one basis shared by all m channels (D23) or one basis sigma_a,k per channel k as in (24)
    (D25); all channels must have the same size l_a (the paper's l_aj carries no k index), so the actor
    weights are always an (m, l_a) array;
  - (29) solved with `np.linalg.lstsq` (SVD) instead of forming (N^T N)^{-1}: same minimiser when N has
    full column rank, better conditioned; rank and condition number are recorded every iteration;
  - quadrature (A18): trapezoid on the stored grid for state-only integrands; for eta_w the ZOH input
    is exact on each sub-step, so int mu sigma = sum_i h mu_i (sigma_i + sigma_{i+1}) / 2;
  - eta_v, eta_delta, eta_w and sigma_a on the grid do not depend on the iteration and are computed once;
    eta_u1, eta_u2 are re-integrated each iteration from the stored grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vsfc.rl.basis import MonomialBasis
from vsfc.rl.data import OffPolicyDataset
from vsfc.rl.utility import log_one_minus_tanh_sq, policy_from_xi


@dataclass(frozen=True)
class IRLProblem:
    """critic: sigma_c on R^n; actor: one basis on R^n shared by all m channels (D23), or a sequence of m bases
    sigma_a,k (eq. (24), D25) of equal size l_a; bound: beta (or gamma); r: diagonal of R (m,); Q: (n, n)."""

    critic: MonomialBasis
    actor: MonomialBasis | tuple
    bound: float
    r: np.ndarray
    Q: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "r", np.asarray(self.r, float).ravel())
        object.__setattr__(self, "Q", np.asarray(self.Q, float))
        n = self.critic.dim
        if isinstance(self.actor, MonomialBasis):
            actors = (self.actor,) * self.r.size
        else:
            actors = tuple(self.actor)
            if len(actors) != self.r.size:
                raise ValueError(f"need one actor basis per input channel: got {len(actors)}, m = {self.r.size}")
            object.__setattr__(self, "actor", actors)
        object.__setattr__(self, "actors", actors)
        if any(a.dim != n for a in actors) or self.Q.shape != (n, n):
            raise ValueError("critic, actor and Q must share the state dimension")
        if len({a.size for a in actors}) != 1:
            raise ValueError("all actor bases must have the same size l_a (paper's l_aj)")
        if np.any(self.r <= 0) or self.bound <= 0:
            raise ValueError("r and bound must be positive")

    @property
    def m(self) -> int:
        return self.r.size

    @property
    def l_a(self) -> int:
        return self.actors[0].size

    @property
    def n_weights(self) -> int:
        return self.critic.size + self.m * self.l_a

    def actor_features(self, delta: np.ndarray) -> np.ndarray:
        """sigma_a,k(delta) for every channel: delta (..., n) -> (..., m, l_a)."""
        if isinstance(self.actor, MonomialBasis):                       # shared: evaluate once
            s = self.actor(delta)
            return np.broadcast_to(s[..., None, :], s.shape[:-1] + (self.m, s.shape[-1]))
        return np.stack([a(delta) for a in self.actors], axis=-2)

    def split(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """w (l_c + m l_a,) -> (w_c (l_c,), W_a (m, l_a))."""
        lc = self.critic.size
        return w[:lc], w[lc:].reshape(self.m, self.l_a)

    def join(self, w_c: np.ndarray, W_a: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(w_c, float).ravel(), np.asarray(W_a, float).ravel()])


# ---- evaluation of the learned networks --------------------------------------------------------------

def critic_value(problem: IRLProblem, w_c: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Eq. (23): V_hat = w_c^T sigma_c(delta). delta (n,) -> scalar or (N, n) -> (N,)."""
    return problem.critic(delta) @ w_c


def actor_xi(problem: IRLProblem, W_a: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Eq. (24): xi_hat_k = w_a,k^T sigma_a,k(delta). delta (n,) -> (m,) or (N, n) -> (N, m)."""
    return np.einsum("...kl,kl->...k", problem.actor_features(delta), np.asarray(W_a, float))


def policy(problem: IRLProblem, W_a: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Eq. (25) / Alg. 3 Step 5: mu_hat = -beta tanh(xi_hat(delta))."""
    return policy_from_xi(actor_xi(problem, W_a, delta), problem.bound)


# ---- Bellman rows (27)-(28) ------------------------------------------------------------------------------

def _trapz_windows(y: np.ndarray, h: float) -> np.ndarray:
    """Trapezoid rule along axis 1 (the sub-step grid of each window)."""
    return 0.5 * h * (y[:, :-1] + y[:, 1:]).sum(axis=1)


@dataclass
class _FixedTerms:
    """Iteration-independent parts of the rows."""

    sig_a: np.ndarray       # (M, W+1, m, l_a)
    eta_v: np.ndarray       # (M, l_c)
    eta_delta: np.ndarray   # (M,)
    eta_w: np.ndarray       # (M, m, l_a)


def precompute(data: OffPolicyDataset, problem: IRLProblem) -> _FixedTerms:
    d, mu, h = data.delta, data.mu, data.h
    if d.shape[2] != problem.critic.dim or mu.shape[2] != problem.m:
        raise ValueError(f"data dims (n={d.shape[2]}, m={mu.shape[2]}) do not match the problem")
    sig_a = np.ascontiguousarray(problem.actor_features(d))                     # (M, W+1, m, l_a)
    eta_v = problem.critic(d[:, 0]) - problem.critic(d[:, -1])                  # (M, l_c)
    eta_delta = _trapz_windows(np.einsum("mwi,ij,mwj->mw", d, problem.Q, d), h)  # (M,)
    sig_mid = 0.5 * (sig_a[:, :-1] + sig_a[:, 1:])                              # (M, W, m, l_a)
    eta_w = h * np.einsum("mwk,mwkl->mkl", mu, sig_mid)                         # (M, m, l_a)
    return _FixedTerms(sig_a, eta_v, eta_delta, eta_w)


def bellman_rows(data: OffPolicyDataset | _FixedTerms, problem: IRLProblem, W_a_n: np.ndarray,
                 h: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Eqs. (27)-(28): stacked rows N_bar (M, l_c + m l_a) and targets T_bar (M,) for the current actor W_a^(n).

    `data` may be a dataset or its precomputed fixed terms (then h must be given).
    """
    if isinstance(data, OffPolicyDataset):
        h = data.h
        fixed = precompute(data, problem)
    else:
        fixed = data
        if h is None:
            raise ValueError("h is required with precomputed terms")
    beta, r = problem.bound, problem.r
    W_a_n = np.asarray(W_a_n, float).reshape(problem.m, problem.l_a)

    xi = np.einsum("mwkl,kl->mwk", fixed.sig_a, W_a_n)                          # (M, W+1, m)
    th = np.tanh(xi)
    eta_u1 = _trapz_windows(th[..., None] * fixed.sig_a, h)                     # (M, m, l_a)
    eta_u2 = _trapz_windows(log_one_minus_tanh_sq(xi), h)                       # (M, m)

    eta_bar = 2.0 * beta * r[None, :, None] * (fixed.eta_w + beta * eta_u1)      # (M, m, l_a)
    N_bar = np.concatenate([fixed.eta_v, eta_bar.reshape(eta_bar.shape[0], -1)], axis=1)
    T_bar = (fixed.eta_delta
             + 2.0 * beta ** 2 * np.einsum("k,mkl,kl->m", r, eta_u1, W_a_n)
             + beta ** 2 * eta_u2 @ r)
    return N_bar, T_bar


def solve_weights(N_bar: np.ndarray, T_bar: np.ndarray) -> tuple[np.ndarray, int, float, float]:
    """Eq. (29) by SVD least squares. Returns (w, rank, condition number, RMS residual)."""
    w, _, rank, sv = np.linalg.lstsq(N_bar, T_bar, rcond=None)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    rms = float(np.sqrt(np.mean((N_bar @ w - T_bar) ** 2)))
    return w, int(rank), cond, rms


# ---- Algorithm 3, Steps 2-4 --------------------------------------------------------------------------------

@dataclass
class TrainResult:
    w_c: np.ndarray                 # (l_c,)
    W_a: np.ndarray                 # (m, l_a)
    converged: bool
    iterations: int
    history: list = field(default_factory=list)   # per iteration: dict(w, change, rank, cond, rms)


def train(data: OffPolicyDataset, problem: IRLProblem, W_a0: np.ndarray, w_c0: np.ndarray | None = None,
          tol: float = 1e-5, max_iter: int = 100) -> TrainResult:
    """Alg. 3 Steps 2-4: from initial weights, iterate (29) on the same data until ||w^(n+1) - w^(n)|| < tol.

    W_a0: (m, l_a) initial actor (defines mu^(0)); w_c0: initial critic (only enters the stopping test).
    Raises if the stacked matrix is rank-deficient (insufficient excitation, A17).
    """
    fixed = precompute(data, problem)
    W_a0 = np.asarray(W_a0, float).reshape(problem.m, problem.l_a)
    w_c0 = np.zeros(problem.critic.size) if w_c0 is None else np.asarray(w_c0, float)
    w = problem.join(w_c0, W_a0)
    history = []
    for it in range(1, max_iter + 1):
        _, W_a_n = problem.split(w)
        N_bar, T_bar = bellman_rows(fixed, problem, W_a_n, h=data.h)
        w_new, rank, cond, rms = solve_weights(N_bar, T_bar)
        if rank < problem.n_weights:
            raise np.linalg.LinAlgError(
                f"N_bar rank {rank} < {problem.n_weights} unknowns: data not exciting enough (A17)")
        change = float(np.linalg.norm(w_new - w))
        history.append(dict(w=w_new, change=change, rank=rank, cond=cond, rms=rms))
        w = w_new
        if change < tol:
            w_c, W_a = problem.split(w)
            return TrainResult(w_c, W_a, True, it, history)
    w_c, W_a = problem.split(w)
    return TrainResult(w_c, W_a, False, max_iter, history)
