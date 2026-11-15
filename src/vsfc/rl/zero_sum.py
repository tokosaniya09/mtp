"""Baseline learner: off-policy RL for the discounted zero-sum (disturbance-attenuation) optimal controller of
ref. [16] of the base paper.

[16] M. Cheng, H. Liu, Q. Gao, J. Lu, X. Xia, "Optimal Containment Control of a Quadrotor Team With Active
Leaders via Reinforcement Learning," IEEE Trans. Cybern., 54(8):4502-4512, Aug. 2024, DOI 10.1109/TCYB.2023.3284648.
PDF: papers/ref16_cheng2024.pdf. Equation numbers below are [16]'s.

System x_dot = f(x) + G u + D d, with applied input u (m,) and RECORDED disturbance d (q,).
[16] (17)  J = int_t^inf e^{-alpha (tau - t)} r dtau,  r = x^T Q x + u^T R u - gamma^2 Delta^T Delta
[16] (20)  u* = -(1/2) R^-1 G^T grad J*,  Delta* = D^T grad J* / (2 gamma^2)       (unconstrained, no tanh)
[16] (32)  integral Bellman equation over [t, t + T] (the T_pi mu_pi term of [16] comes from its containment
           observer and is absent here: one leader, no observer input in the learning model):

    e^{-alpha T} J(x(t+T)) - J(x(t)) = int e^{-alpha (tau - t)} [ -r(x, u^n, Delta^n)
                                          - 2 (u^{n+1})^T R (u - u^n) + 2 gamma^2 (Delta^{n+1})^T (d - Delta^n) ] dtau

[16] (38)  J_hat = w_c^T sigma_c(x),  u_hat_k = w_a,k^T sigma_a,k(x),  Delta_hat_j = w_d,j^T sigma_d,j(x);
           with R = diag(r) the equation is linear in (w_c, w_a, w_d) and is solved by least squares.

Row for one window (derivation: differentiate e^{-alpha t} J^n along the data and use (20)):

    [e^{-alpha T} sigma_c(x(t+T)) - sigma_c(x(t))]^T w_c
      + sum_k 2 r_k  int e (u_k - u_k^n) sigma_a,k^T dtau  w_a,k
      - sum_j 2 gamma^2 int e (d_j - Delta_j^n) sigma_d,j^T dtau  w_d,j   =   - int e r(x, u^n, Delta^n) dtau

Identifiability: the w_d columns need d - Delta^n to be excited independently of u, so the data must contain a
nonzero recorded disturbance ([16]: "Record the historical data of ... disturbance input").

Known answer (linear f = A x): J* = x^T P x with the discounted game Riccati equation
    (A - alpha/2 I)^T P + P (A - alpha/2 I) + Q - P (B R^-1 B^T - D D^T / gamma^2) P = 0,
    u* = -R^-1 B^T P x,  Delta* = D^T P x / gamma^2.
A stabilising P >= 0 exists only if the control outweighs the disturbance (for D = B: gamma^2 > r).

[Eng] quadrature as in `offpolicy_irl.py` (A18): ZOH u and d on each sub-step, trapezoid for state integrands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vsfc.rl.basis import MonomialBasis


def game_riccati(A, B, D, Q, R, gamma2: float, alpha: float = 0.0) -> np.ndarray | None:
    """Stabilising solution P of the discounted game Riccati equation above, or None if there is none.

    Stable invariant subspace of the Hamiltonian [[Aa, -M], [-Q, -Aa^T]], Aa = A - alpha/2 I,
    M = B R^-1 B^T - D D^T / gamma^2. Checks the residual and P > 0."""
    A, B, D, Q, R = (np.atleast_2d(np.asarray(x, float)) for x in (A, B, D, Q, R))
    n = A.shape[0]
    Aa = A - 0.5 * alpha * np.eye(n)
    M = B @ np.linalg.solve(R, B.T) - D @ D.T / gamma2
    w, V = np.linalg.eig(np.block([[Aa, -M], [-Q, -Aa.T]]))
    order = np.argsort(w.real)
    if w.real[order[n - 1]] > -1e-9 or w.real[order[n]] < 1e-9:           # need n strictly stable eigenvalues
        return None
    X = V[:, order[:n]]
    try:
        P = np.real(X[n:] @ np.linalg.inv(X[:n]))
    except np.linalg.LinAlgError:
        return None
    P = 0.5 * (P + P.T)
    res = Aa.T @ P + P @ Aa + Q - P @ M @ P
    if np.abs(res).max() > 1e-6 * max(1.0, np.abs(P).max()) or np.linalg.eigvalsh(P).min() <= 0.0:
        return None
    return P


@dataclass(frozen=True)
class GameDataset:
    """Windowed data: x (M, W+1, n) state grid; u (M, W, m) applied input; d (M, W, q) recorded disturbance
    (both ZOH on each sub-step); h sub-step, s."""

    x: np.ndarray
    u: np.ndarray
    d: np.ndarray
    h: float

    @classmethod
    def from_trajectory(cls, x, u, d, h: float, steps_per_window: int) -> "GameDataset":
        x, u, d = np.asarray(x, float), np.asarray(u, float), np.asarray(d, float)
        if not (x.shape[0] == u.shape[0] + 1 == d.shape[0] + 1):
            raise ValueError("need len(x) = len(u) + 1 = len(d) + 1")
        W = int(steps_per_window)
        M = u.shape[0] // W
        if M < 1:
            raise ValueError("trajectory shorter than one window")
        idx = np.arange(M)[:, None] * W + np.arange(W + 1)[None, :]
        return cls(x[idx], u[: M * W].reshape(M, W, -1), d[: M * W].reshape(M, W, -1), float(h))

    @staticmethod
    def concatenate(sets: list["GameDataset"]) -> "GameDataset":
        return GameDataset(np.concatenate([s.x for s in sets]), np.concatenate([s.u for s in sets]),
                           np.concatenate([s.d for s in sets]), sets[0].h)

    @property
    def n_windows(self) -> int:
        return self.x.shape[0]


def _features(bases, x: np.ndarray) -> np.ndarray:
    """Per-channel bases evaluated on x (..., n) -> (..., channels, l)."""
    return np.stack([b(x) for b in bases], axis=-2)


@dataclass(frozen=True)
class GameProblem:
    """critic: sigma_c; actors: m bases sigma_a,k (equal size l_a); dists: q bases sigma_d,j (equal size l_d);
    r: diagonal of R (m,); Q (n, n); gamma2 = gamma^2; alpha: discount rate, 1/s."""

    critic: MonomialBasis
    actors: tuple
    dists: tuple
    r: np.ndarray
    Q: np.ndarray
    gamma2: float
    alpha: float

    def __post_init__(self):
        object.__setattr__(self, "actors", tuple(self.actors))
        object.__setattr__(self, "dists", tuple(self.dists))
        object.__setattr__(self, "r", np.asarray(self.r, float).ravel())
        object.__setattr__(self, "Q", np.asarray(self.Q, float))
        n = self.critic.dim
        if any(b.dim != n for b in self.actors + self.dists) or self.Q.shape != (n, n):
            raise ValueError("critic, actor, disturbance bases and Q must share the state dimension")
        if len(self.actors) != self.r.size or len({b.size for b in self.actors}) != 1 \
                or len({b.size for b in self.dists}) != 1:
            raise ValueError("need one actor basis per input channel; equal sizes within actors and within dists")
        if np.any(self.r <= 0) or self.gamma2 <= 0 or self.alpha < 0:
            raise ValueError("r, gamma2 must be positive and alpha non-negative")

    @property
    def m(self) -> int:
        return len(self.actors)

    @property
    def q(self) -> int:
        return len(self.dists)

    @property
    def l_a(self) -> int:
        return self.actors[0].size

    @property
    def l_d(self) -> int:
        return self.dists[0].size

    @property
    def n_weights(self) -> int:
        return self.critic.size + self.m * self.l_a + self.q * self.l_d

    def split(self, w: np.ndarray):
        """w -> (w_c (l_c,), W_a (m, l_a), W_d (q, l_d))."""
        lc, na = self.critic.size, self.m * self.l_a
        return w[:lc], w[lc:lc + na].reshape(self.m, self.l_a), w[lc + na:].reshape(self.q, self.l_d)

    def join(self, w_c, W_a, W_d) -> np.ndarray:
        return np.concatenate([np.ravel(w_c), np.ravel(W_a), np.ravel(W_d)]).astype(float)

    def control(self, W_a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """[16] (38): u_hat = W_a sigma_a(x). x (n,) -> (m,) or (N, n) -> (N, m)."""
        return np.einsum("...kl,kl->...k", _features(self.actors, x), np.asarray(W_a, float))

    def disturbance(self, W_d: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.einsum("...jl,jl->...j", _features(self.dists, x), np.asarray(W_d, float))


def _trapz(y: np.ndarray, h: float) -> np.ndarray:
    return 0.5 * h * (y[:, :-1] + y[:, 1:]).sum(axis=1)


def game_rows(data: GameDataset, prob: GameProblem, W_a_n: np.ndarray, W_d_n: np.ndarray):
    """[16] (32) with (38): stacked rows N (M, n_weights) and targets T (M,) for the current (u^n, Delta^n)."""
    x, u, d, h = data.x, data.u, data.d, data.h
    Wn = x.shape[1] - 1
    disc = np.exp(-prob.alpha * h * np.arange(Wn + 1))                             # e^{-alpha (tau - t)}, (W+1,)
    sa = _features(prob.actors, x) * disc[None, :, None, None]                     # (M, W+1, m, l_a), discounted
    sd = _features(prob.dists, x) * disc[None, :, None, None]                      # (M, W+1, q, l_d)
    un = prob.control(W_a_n, x)                                                    # (M, W+1, m)
    dn = prob.disturbance(W_d_n, x)                                                # (M, W+1, q)

    col_c = disc[-1] * prob.critic(x[:, -1]) - prob.critic(x[:, 0])                # (M, l_c)
    col_a = h * np.einsum("mwk,mwkl->mkl", u, 0.5 * (sa[:, :-1] + sa[:, 1:])) - _trapz(un[..., None] * sa, h)
    col_d = h * np.einsum("mwj,mwjl->mjl", d, 0.5 * (sd[:, :-1] + sd[:, 1:])) - _trapz(dn[..., None] * sd, h)
    col_a = 2.0 * prob.r[None, :, None] * col_a
    col_d = -2.0 * prob.gamma2 * col_d
    M = x.shape[0]
    N = np.concatenate([col_c, col_a.reshape(M, -1), col_d.reshape(M, -1)], axis=1)
    cost = (np.einsum("mwi,ij,mwj->mw", x, prob.Q, x) + (un ** 2) @ prob.r - prob.gamma2 * (dn ** 2).sum(-1))
    return N, -_trapz(cost * disc[None, :], h)


@dataclass
class GameResult:
    w_c: np.ndarray
    W_a: np.ndarray                 # (m, l_a): u = W_a sigma_a
    W_d: np.ndarray                 # (q, l_d): Delta = W_d sigma_d
    converged: bool
    iterations: int
    history: list = field(default_factory=list)


def train_game(data: GameDataset, prob: GameProblem, W_a0: np.ndarray, W_d0: np.ndarray | None = None,
               tol: float = 1e-5, max_iter: int = 100) -> GameResult:
    """[16] Sec. III-A learning steps: from an admissible u^0 (and Delta^0, default 0) iterate (32) on the same
    data until ||w^{n+1} - w^n|| < tol. Raises if the stacked matrix is rank-deficient."""
    W_d0 = np.zeros((prob.q, prob.l_d)) if W_d0 is None else W_d0
    w = prob.join(np.zeros(prob.critic.size), W_a0, W_d0)
    history = []
    for it in range(1, max_iter + 1):
        _, W_a_n, W_d_n = prob.split(w)
        N, T = game_rows(data, prob, W_a_n, W_d_n)
        w_new, _, rank, sv = np.linalg.lstsq(N, T, rcond=None)
        if rank < prob.n_weights:
            raise np.linalg.LinAlgError(f"rank {rank} < {prob.n_weights} unknowns: data not exciting enough "
                                        "(is a recorded disturbance present?)")
        change = float(np.linalg.norm(w_new - w))
        history.append(dict(w=w_new, change=change, rank=int(rank), cond=float(sv[0] / sv[-1]),
                            rms=float(np.sqrt(np.mean((N @ w_new - T) ** 2)))))
        w = w_new
        if change < tol:
            return GameResult(*prob.split(w), True, it, history)
    return GameResult(*prob.split(w), False, max_iter, history)
