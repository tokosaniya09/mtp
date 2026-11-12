"""Tests for rl/offpolicy_irl.py: Bellman rows (27)-(28), least squares (29), Algorithm 3.

Known-answer tests (no quadrotor yet):
  1. LQR: linear plant, beta = 1000 (tanh ~ linear, U ~ mu^T R mu). Critic -> Riccati P, actor -> K / beta,
     and iteration 1 -> Kleinman's first Lyapunov iterate from the same initial gain.
  2. Constrained scalar plant x_dot = a x + b u, beta = 1 (input saturates). The HJB (16) reduces pointwise to
     q x^2 + a x p - 2 beta^2 r ln cosh(b p / (2 beta r)) = 0 for p = V'(x) (derivation in `_V_prime`),
     so u*(x) = -beta tanh(b V'(x) / (2 beta r)) is computable by root-finding.
"""

import numpy as np
import pytest
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov
from scipy.optimize import brentq

from vsfc.rl.basis import PolynomialBasis, axis_groups, grouped_basis, weights_from_quadratic_form
from vsfc.rl.data import OffPolicyDataset, rollout, sum_of_sinusoids
from vsfc.rl.offpolicy_irl import (IRLProblem, actor_xi, bellman_rows, critic_value, policy, solve_weights,
                                   train)

H, W = 1e-3, 50          # A18: 1 ms sub-steps, dT = 0.05 s (Sec. VI)

# ---- LQR fixture ------------------------------------------------------------------------------------------
A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, -1.0, -0.2]])     # open-loop unstable
B = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
Q = 10.0 * np.eye(3)
r = np.array([8.0, 4.0])
R = np.diag(r)
BETA = 1000.0
K0 = np.array([[3.0, 4.0, 0.0], [2.0, 2.0, 4.0]])                          # stabilising initial gain


def _lin(t, x, u):
    return A @ x + B @ u


@pytest.fixture(scope="module")
def lqr():
    rng = np.random.default_rng(1)
    runs = []
    for s in range(4):
        e = sum_of_sinusoids(1.0, [0.7, 1.9, 3.1, 5.3, 8.9], 2, seed=s)
        _, _, d, mu = rollout(_lin, rng.uniform(-2, 2, 3), lambda t, x, e=e: -K0 @ x + e(t), 5.0, H)
        runs.append(OffPolicyDataset.from_trajectory(d, mu, H, W))
    data = OffPolicyDataset.concatenate(runs)
    prob = IRLProblem(PolynomialBasis(3, (2,)), PolynomialBasis(3, (1,)), BETA, r, Q)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return data, prob, P, K


def _kleinman_step(Kn):
    Acl = A - B @ Kn
    Pn1 = solve_continuous_lyapunov(Acl.T, -(Q + Kn.T @ R @ Kn))
    return Pn1, np.linalg.solve(R, B.T @ Pn1)


def test_problem_validation():
    with pytest.raises(ValueError):
        IRLProblem(PolynomialBasis(3, (2,)), PolynomialBasis(2, (1,)), 1.0, [1.0], np.eye(3))
    with pytest.raises(ValueError):
        IRLProblem(PolynomialBasis(1, (2,)), PolynomialBasis(1, (1,)), 1.0, [-1.0], [[1.0]])
    p = IRLProblem(PolynomialBasis(3, (2,)), PolynomialBasis(3, (1,)), 1.0, [1.0, 2.0], np.eye(3))
    w = np.arange(p.n_weights, dtype=float)
    wc, Wa = p.split(w)
    assert wc.shape == (6,) and Wa.shape == (2, 3)
    np.testing.assert_array_equal(p.join(wc, Wa), w)


def test_rows_shape(lqr):
    data, prob, _, _ = lqr
    N, T = bellman_rows(data, prob, K0 / BETA)
    assert N.shape == (data.n_windows, prob.n_weights) and T.shape == (data.n_windows,)


def test_exact_kleinman_iterate_satisfies_rows(lqr):
    """Rows (27)-(28) checked independently of least squares: the exact (P1, K1) from K0 zero the residual."""
    data, prob, _, _ = lqr
    P1, K1 = _kleinman_step(K0)
    N, T = bellman_rows(data, prob, K0 / BETA)
    w1 = prob.join(weights_from_quadratic_form(prob.critic, P1), K1 / BETA)
    rel = np.linalg.norm(N @ w1 - T) / np.linalg.norm(T)
    assert rel < 1e-5          # residual from quadrature O(h^2) and tanh ~ linear, O((mu/beta)^2)


def test_lqr_known_answer(lqr):
    data, prob, P, K = lqr
    res = train(data, prob, K0 / BETA, tol=1e-5)
    assert res.converged and res.iterations <= 10
    wP = weights_from_quadratic_form(prob.critic, P)
    assert np.linalg.norm(res.w_c - wP) / np.linalg.norm(wP) < 1e-5
    assert np.linalg.norm(res.W_a * BETA - K) / np.linalg.norm(K) < 1e-5
    assert res.history[-1]["rank"] == prob.n_weights


def test_first_iteration_matches_kleinman(lqr):
    data, prob, _, _ = lqr
    P1, K1 = _kleinman_step(K0)
    res = train(data, prob, K0 / BETA, max_iter=1)
    wc1, Wa1 = prob.split(res.history[0]["w"])
    wP1 = weights_from_quadratic_form(prob.critic, P1)
    assert np.linalg.norm(wc1 - wP1) / np.linalg.norm(wP1) < 1e-4
    assert np.linalg.norm(Wa1 * BETA - K1) / np.linalg.norm(K1) < 1e-4


def test_quadratic_convergence(lqr):
    """Policy iteration = Newton's method on the Riccati equation (Kleinman): change_{n+1} ~ C change_n^2."""
    data, prob, _, _ = lqr
    ch = [h["change"] for h in train(data, prob, K0 / BETA).history]
    # last three steps: each change well below the square of the previous (times a modest constant)
    for a_, b_ in zip(ch[-4:-1], ch[-3:]):
        assert b_ < 0.1 * a_


def test_held_out_bellman_residual(lqr):
    """Converged weights are a fixed point of (28) on data not used for training."""
    data, prob, _, _ = lqr
    res = train(data, prob, K0 / BETA)
    e = sum_of_sinusoids(1.0, [1.3, 2.7, 6.1], 2, seed=99)
    _, _, d, mu = rollout(_lin, np.array([1.0, -1.5, 0.5]), lambda t, x: -K0 @ x + e(t), 3.0, H)
    test = OffPolicyDataset.from_trajectory(d, mu, H, W)
    N, T = bellman_rows(test, prob, res.W_a)
    rel = np.linalg.norm(N @ prob.join(res.w_c, res.W_a) - T) / np.linalg.norm(T)
    assert rel < 1e-5


def test_on_policy_data_cannot_identify_actor():
    """With mu = mu^(n) exactly (no exploration), eta_bar = 0 (mu + beta tanh xi^(n) = 0): N_bar is rank-
    deficient. This is why the behaviour policy needs excitation (A17)."""
    prob = IRLProblem(PolynomialBasis(1, (2,)), PolynomialBasis(1, (1,)), 1.0, [1.0], [[1.0]])
    Wa = np.array([[0.8]])
    runs = []
    for x0 in (-1.0, 0.5, 1.0):
        _, _, d, mu = rollout(lambda t, x, u: 0.3 * x + u, np.array([x0]),
                              lambda t, x: policy(prob, Wa, x), 1.0, H)
        runs.append(OffPolicyDataset.from_trajectory(d, mu, H, W))
    data = OffPolicyDataset.concatenate(runs)
    N, _ = bellman_rows(data, prob, Wa)
    # ZOH makes the cancellation exact only to O(h); still orders of magnitude below the critic column
    assert np.abs(N[:, 1]).max() < 1e-3 * np.abs(N[:, 0]).max()


def test_train_raises_without_excitation():
    """Data resting at the origin gives N_bar = 0: train must refuse rather than return arbitrary weights."""
    prob = IRLProblem(PolynomialBasis(1, (2,)), PolynomialBasis(1, (1,)), 1.0, [1.0], [[1.0]])
    _, _, d, mu = rollout(lambda t, x, u: -x + u, np.zeros(1), lambda t, x: np.zeros(1), 0.5, H)
    data = OffPolicyDataset.from_trajectory(d, mu, H, W)
    with pytest.raises(np.linalg.LinAlgError):
        train(data, prob, np.array([[1.0]]))
    w, rank, cond, _ = solve_weights(np.zeros((5, 2)), np.zeros(5))
    assert rank == 0 and cond == np.inf


# ---- constrained scalar known answer --------------------------------------------------------------------
a_s, b_s, q_s, r_s, beta_s = 0.5, 1.0, 1.0, 1.0, 1.0      # null-controllable region |x| < beta b / a = 2


def _V_prime(x):
    """Stabilising root p = V'(x) of the scalar HJB.

    With u* = -beta tanh(xi), xi = b p / (2 beta r) (eq. 15), U(u*) + p b u* = -2 beta^2 r ln cosh(xi) (from (26)),
    so (16) becomes g(p) = q x^2 + a x p - 2 beta^2 r ln cosh(b p / (2 beta r)) = 0. g is concave, g(0) > 0, and
    g -> -inf along sign(p) = sign(x) when |x| < beta b / a, so there is exactly one root with p x > 0.
    """
    if x == 0.0:
        return 0.0
    s = np.sign(x)
    g = lambda p: q_s * x * x + a_s * x * (s * p) - 2 * beta_s ** 2 * r_s * np.log(np.cosh(b_s * p / (2 * beta_s * r_s)))
    return s * brentq(g, 0.0, 200.0)


def _u_star(x):
    return -beta_s * np.tanh(b_s * _V_prime(x) / (2 * beta_s * r_s))


@pytest.fixture(scope="module")
def scalar_data():
    runs = []
    for s, x0 in enumerate(np.linspace(-1.2, 1.2, 9)):
        e = sum_of_sinusoids(0.1, [0.9, 2.3, 4.1, 7.7], 1, seed=10 + s)
        _, _, d, mu = rollout(lambda t, x, u: a_s * x + b_s * u, np.array([x0]),
                              lambda t, x, e=e: -0.9 * np.tanh(2.0 * x) + e(t), 4.0, H)
        runs.append(OffPolicyDataset.from_trajectory(d, mu, H, W))
    return OffPolicyDataset.concatenate(runs)


@pytest.mark.parametrize("cdeg, adeg, tol", [((2, 4, 6), (1, 3, 5), 1e-2), ((2, 4, 6, 8), (1, 3, 5, 7), 2e-3)])
def test_constrained_scalar_known_answer(scalar_data, cdeg, adeg, tol):
    prob = IRLProblem(PolynomialBasis(1, cdeg), PolynomialBasis(1, adeg), beta_s, [r_s], [[q_s]])
    W0 = np.zeros((1, len(adeg)))
    W0[0, 0] = 1.0                                   # mu^(0) = -tanh(x): admissible on |x| <= 1.2
    res = train(scalar_data, prob, W0)
    assert res.converged
    xs = np.linspace(-1.0, 1.0, 41)
    u_learned = policy(prob, res.W_a, xs[:, None])[:, 0]
    u_true = np.array([_u_star(x) for x in xs])
    assert abs(u_true[-1]) > 0.95                    # the constraint is active in the test range
    assert np.abs(u_learned - u_true).max() < tol
    assert np.all(np.abs(u_learned) < beta_s)        # tanh policy respects the bound (A2)
    # critic: V(0) = 0 and V increasing in |x|
    v = critic_value(prob, res.w_c, xs[:, None])
    assert np.all(np.diff(v[20:]) > 0) and np.all(np.diff(v[:21]) < 0)


def test_actor_xi_and_policy_shapes(lqr):
    _, prob, _, K = lqr
    x = np.array([[0.1, 0.2, 0.3], [1.0, 0.0, -1.0]])
    assert actor_xi(prob, K / BETA, x).shape == (2, 2)
    np.testing.assert_allclose(policy(prob, K / BETA, x[0]), -K @ x[0], rtol=1e-6)

# ---- per-channel actor bases, eq. (24) (D25) -------------------------------------------------------------

def test_per_channel_identical_bases_equal_shared(lqr):
    data, prob, _, _ = lqr
    per = IRLProblem(prob.critic, [PolynomialBasis(3, (1,)), PolynomialBasis(3, (1,))], BETA, r, Q)
    N1, T1 = bellman_rows(data, prob, K0 / BETA)
    N2, T2 = bellman_rows(data, per, K0 / BETA)
    np.testing.assert_allclose(N2, N1, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(T2, T1, rtol=1e-13)
    x = np.array([0.4, -1.0, 2.0])
    np.testing.assert_allclose(actor_xi(per, K0 / BETA, x), actor_xi(prob, K0 / BETA, x))


def test_per_channel_validation():
    with pytest.raises(ValueError):          # wrong number of actor bases
        IRLProblem(PolynomialBasis(3, (2,)), [PolynomialBasis(3, (1,))], 1.0, [1.0, 1.0], np.eye(3))
    with pytest.raises(ValueError):          # unequal l_a
        IRLProblem(PolynomialBasis(3, (2,)), [PolynomialBasis(3, (1,)), PolynomialBasis(3, (1, 2))],
                   1.0, [1.0, 1.0], np.eye(3))


# Outer-loop linearisation of Phase 1.5 (docs/equation_map.md §5.8): per axis e_dot = v/x*, v_dot = mu/m.
XS, MASS, QS, RS = 4.0, 1.3, 10.0, 8.0
A6 = np.zeros((6, 6)); A6[:3, 3:] = np.eye(3) / XS
B6 = np.zeros((6, 3)); B6[3:] = np.eye(3) / MASS
K06 = np.hstack([1.0 * np.eye(3), 1.5 * np.eye(3)])                       # stabilising PD-like initial gain


@pytest.fixture(scope="module")
def double_integrators():
    rng = np.random.default_rng(7)
    runs = []
    for s_ in range(6):
        e = sum_of_sinusoids(0.5, [0.5, 1.3, 2.9, 4.7, 7.1], 3, seed=10 + s_)
        _, _, d, mu = rollout(lambda t, x, u: A6 @ x + B6 @ u, rng.uniform(-1, 1, 6),
                              lambda t, x, e=e: -K06 @ x + e(t), 5.0, H)
        runs.append(OffPolicyDataset.from_trajectory(d, mu, H, W))
    P = solve_continuous_are(A6, B6, QS * np.eye(6), RS * np.eye(3))
    K = B6.T @ P / RS
    return OffPolicyDataset.concatenate(runs), P, K


@pytest.mark.parametrize("cdeg, adeg", [((2,), (1,)), ((2, 4), (1, 3))])
def test_lqr_known_answer_per_axis_bases(double_integrators, cdeg, adeg):
    """Phase 1.5 bases (D25) on the linearised outer loop with beta = 1000 (unconstrained limit): the critic must
    equal the Riccati P (block-diagonal by axis) and the actor K / beta; quartic critic and cubic actor weights
    must vanish because the exact value is quadratic."""
    data, P, K = double_integrators
    critic = grouped_basis(6, axis_groups(), cdeg)
    actors = [grouped_basis(6, [g], adeg) for g in axis_groups()]
    prob = IRLProblem(critic, actors, BETA, [RS] * 3, QS * np.eye(6))
    W0 = np.zeros((3, prob.l_a))
    W0[:, :2] = np.array([[1.0, 1.5]] * 3) / BETA                         # same initial gain as the data
    res = train(data, prob, W0, tol=1e-5)
    assert res.converged and res.iterations <= 10
    assert res.history[-1]["rank"] == prob.n_weights
    wP = weights_from_quadratic_form(critic, P)
    assert np.linalg.norm(res.w_c - wP) / np.linalg.norm(wP) < 1e-4
    K_lin = res.W_a[:, :2] * BETA                                          # coefficients of (e_k, v_k)
    K_ref = np.array([[K[k, k], K[k, k + 3]] for k in range(3)])
    np.testing.assert_allclose(K_lin, K_ref, rtol=1e-4)
    hi_c = critic.total_degrees > 2
    if hi_c.any():
        assert np.abs(res.w_c[hi_c]).max() < 1e-4 * np.abs(wP).max()
        assert np.abs(res.W_a[:, 2:]).max() * BETA < 1e-4 * np.abs(K_ref).max()
