"""Tests for rl/zero_sum.py: ref. [16]'s discounted zero-sum off-policy learner (baseline of Fig. 10).

Known answer: for linear dynamics the optimum is the discounted game Riccati equation. Test plant = the outer-loop
learning model of docs/equation_map.md §5.8 (three double integrators e_dot = v/x*, v_dot = (mu + d)/m), with the
disturbance d entering as a force in the input channel (D = B), Q = 10 I, R = 8 I, gamma^2 = 16, alpha = 0.06.
"""

import numpy as np
import pytest

from vsfc.rl.basis import axis_groups, grouped_basis, weights_from_quadratic_form
from vsfc.rl.data import rollout, sum_of_sinusoids
from vsfc.rl.zero_sum import GameDataset, GameProblem, game_riccati, game_rows, train_game

H, W = 1e-3, 50
XS, MASS, QS, RS, G2, ALPHA = 4.0, 1.3, 10.0, 8.0, 16.0, 0.06
A = np.zeros((6, 6)); A[:3, 3:] = np.eye(3) / XS
B = np.zeros((6, 3)); B[3:] = np.eye(3) / MASS
K0 = np.hstack([1.0 * np.eye(3), 1.5 * np.eye(3)])


def test_game_riccati_residual_and_limits():
    P = game_riccati(A, B, B, QS * np.eye(6), RS * np.eye(3), G2, ALPHA)
    Aa = A - 0.5 * ALPHA * np.eye(6)
    M = B @ B.T / RS - B @ B.T / G2
    np.testing.assert_allclose(Aa.T @ P + P @ Aa + QS * np.eye(6) - P @ M @ P, 0.0, atol=1e-8)
    K = B.T @ P / RS
    assert np.linalg.eigvals(A - B @ K).real.max() < 0                       # stabilising
    np.testing.assert_allclose([K[0, 0], K[0, 3]], [1.345, 1.985], atol=2e-3)  # values logged in the research log
    # gamma -> infinity, alpha = 0 recovers the ordinary LQR gain of §5.8 (1.118, 1.406)
    P_lqr = game_riccati(A, B, B, QS * np.eye(6), RS * np.eye(3), 1e12, 0.0)
    np.testing.assert_allclose([(B.T @ P_lqr / RS)[0, 0], (B.T @ P_lqr / RS)[0, 3]], [1.118, 1.406], atol=1e-3)


def test_game_riccati_has_no_solution_when_disturbance_outweighs_control():
    """gamma^2 = 4 < r = 8 with D = B ([16]'s gamma^2 with the base paper's R): no stabilising solution."""
    assert game_riccati(A, B, B, QS * np.eye(6), RS * np.eye(3), 4.0, ALPHA) is None


@pytest.fixture(scope="module")
def game_data():
    rng = np.random.default_rng(5)
    sets = []
    for s in range(6):
        ex = sum_of_sinusoids(0.5, [0.5, 1.3, 2.9, 4.7, 7.1], 3, seed=20 + s)       # exploration in u
        dist = sum_of_sinusoids(0.4, [0.3, 0.9, 1.9, 3.7], 3, seed=40 + s)          # recorded disturbance (N)
        beh = lambda t, x, ex=ex, dist=dist: np.concatenate([-K0 @ x + ex(t), dist(t)])
        _, _, xs, v = rollout(lambda t, x, v: A @ x + B @ (v[:3] + v[3:]), rng.uniform(-1, 1, 6), beh, 5.0, H)
        sets.append(GameDataset.from_trajectory(xs, v[:, :3], v[:, 3:], H, W))
    return GameDataset.concatenate(sets)


def _problem(cdeg, adeg):
    critic = grouped_basis(6, axis_groups(), cdeg)
    acts = [grouped_basis(6, [g], adeg) for g in axis_groups()]
    return GameProblem(critic, acts, acts, [RS] * 3, QS * np.eye(6), G2, ALPHA)


def test_exact_solution_satisfies_rows(game_data):
    """The Riccati solution (value, control, disturbance law) satisfies [16] (32) on the data."""
    prob = _problem((2,), (1,))
    P = game_riccati(A, B, B, prob.Q, RS * np.eye(3), G2, ALPHA)
    K, L = B.T @ P / RS, B.T @ P / G2
    Wa = np.array([[-K[k, k], -K[k, k + 3]] for k in range(3)])
    Wd = np.array([[L[k, k], L[k, k + 3]] for k in range(3)])
    N, T = game_rows(game_data, prob, Wa, Wd)
    w = prob.join(weights_from_quadratic_form(prob.critic, P), Wa, Wd)
    assert np.abs(N @ w - T).max() < 1e-6 * np.abs(T).max()


@pytest.mark.parametrize("cdeg, adeg, tol", [((2,), (1,), 1e-4), ((2, 4), (1, 3), 1e-3)])
def test_game_known_answer(game_data, cdeg, adeg, tol):
    prob = _problem(cdeg, adeg)
    Wa0 = np.zeros((3, prob.l_a)); Wa0[:, :2] = [-1.0, -1.5]                 # u^0 = -K0 x (admissible)
    res = train_game(game_data, prob, Wa0, tol=1e-6, max_iter=60)
    assert res.converged and res.history[-1]["rank"] == prob.n_weights
    P = game_riccati(A, B, B, prob.Q, RS * np.eye(3), G2, ALPHA)
    K, L = B.T @ P / RS, B.T @ P / G2
    wP = weights_from_quadratic_form(prob.critic, P)
    assert np.linalg.norm(res.w_c - wP) / np.linalg.norm(wP) < tol
    np.testing.assert_allclose(res.W_a[:, :2], [[-K[k, k], -K[k, k + 3]] for k in range(3)], rtol=tol)
    np.testing.assert_allclose(res.W_d[:, :2], [[L[k, k], L[k, k + 3]] for k in range(3)], rtol=tol)


def test_no_recorded_disturbance_is_rank_deficient():
    """Without disturbance excitation the Delta weights are not identifiable (needs recorded d, [16])."""
    rng = np.random.default_rng(6)
    ex = sum_of_sinusoids(0.5, [0.5, 1.3, 2.9, 4.7, 7.1], 3, seed=1)
    _, _, xs, u = rollout(lambda t, x, u: A @ x + B @ u, rng.uniform(-1, 1, 6), lambda t, x: -K0 @ x + ex(t), 5.0, H)
    data = GameDataset.from_trajectory(xs, u, np.zeros_like(u), H, W)
    prob = _problem((2,), (1,))
    Wa0 = np.zeros((3, 2)); Wa0[:] = [-1.0, -1.5]
    with pytest.raises(np.linalg.LinAlgError):
        train_game(data, prob, Wa0)
