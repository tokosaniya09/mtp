"""Tests for sim/baseline16.py: the ref. [16] baseline (D35) set up on the case study."""

import numpy as np
import pytest

from vsfc.config import load_case_study
from vsfc.controllers.cascade import CascadeController
from vsfc.controllers.thrust_mapping import ReferenceGenerator
from vsfc.rl.zero_sum import train_game
from vsfc.sim.baseline16 import (attitude_game_problem, attitude_riccati, clipped_policy, collect_outer_game_data,
                                 outer_game_problem, outer_riccati)
from vsfc.sim.cascade import CascadeSystem, simulate_cascade


@pytest.fixture(scope="module")
def case():
    return load_case_study()


def test_weights_follow_d35(case):
    """Base paper's Q, R; gamma^2 = ratio * R with [16]'s ratios; alpha from [16]."""
    for fol in case.followers:
        po, pa = outer_game_problem(case, fol), attitude_game_problem(case, fol)
        assert po.gamma2 == 16.0 and pa.gamma2 == 48.0 and po.alpha == pa.alpha == 0.06
        np.testing.assert_array_equal(po.Q, fol.Q)
        np.testing.assert_array_equal(po.r, [8.0] * 3)
        np.testing.assert_array_equal(pa.Q, 10.0 * np.eye(6))
        assert po.critic.size == 24 and po.l_a == 6 and pa.critic.size == 9 and pa.l_a == 2


def test_riccati_gains(case):
    """Values logged in the research log (D35)."""
    f2, f3 = case.followers
    for fol, ref in ((f2, [1.345, 1.985]), (f3, [1.320, 2.150])):
        _, K, L = outer_riccati(case, fol)
        np.testing.assert_allclose([K[0, 0], K[0, 3]], ref, atol=2e-3)
        np.testing.assert_allclose(L, K * 8.0 / 16.0, rtol=1e-12)          # D = B: L = (r / gamma^2) K
    _, Ka, _ = attitude_riccati(case, f2)
    np.testing.assert_allclose([Ka[0, 0], Ka[0, 3]], [1.188, 1.234], atol=2e-3)


def test_outer_baseline_learns_the_riccati_gain(case):
    fol = case.followers[0]
    prob = outer_game_problem(case, fol)
    data = collect_outer_game_data(case, fol, 1.0, 1.5)
    W0 = np.zeros((3, prob.l_a)); W0[:, :2] = [-1.0, -1.5]
    res = train_game(data, prob, W0, tol=fol.tol_vs, max_iter=case.baseline16.max_iter)
    _, K, _ = outer_riccati(case, fol)
    assert res.converged and res.iterations <= 10
    np.testing.assert_allclose(-res.W_a[:, :2], [[K[k, k], K[k, k + 3]] for k in range(3)], rtol=1e-3)


def test_clipped_policy(case):
    fol = case.followers[0]
    prob = outer_game_problem(case, fol)
    W = np.zeros((3, prob.l_a)); W[:, :2] = [-1.345, -1.985]
    pol = clipped_policy(prob, W, fol.beta)
    np.testing.assert_allclose(pol(np.array([1.0, 0, 0, 0, 0, 0])), [-1.345, 0, 0])
    np.testing.assert_array_equal(pol(np.array([10.0, -10.0, 0, 0, 0, 0])), [-5.0, 5.0, 0.0])   # direct saturation


def test_baseline_cascade_converges_from_paper_initial_conditions(case):
    """Riccati-gain laws with clipping in the cascade (no weight files needed): bounded inputs, error shrinks."""
    fol = case.followers[0]
    _, Ko, _ = outer_riccati(case, fol)
    _, Ki, _ = attitude_riccati(case, fol)
    ctrl = CascadeController(lambda d: np.clip(-Ko @ d, -fol.beta, fol.beta),
                             lambda xe: np.clip(-Ki @ xe, -fol.gamma, fol.gamma),
                             ReferenceGenerator.from_case_study(case, fol))
    system = CascadeSystem.from_case_study(case, fol)
    tr = simulate_cascade(system, ctrl, system.initial_state_from_config(fol), 10.0, case.step)
    assert np.abs(tr.mu).max() <= fol.beta and np.abs(tr.u).max() <= fol.gamma
    assert np.linalg.norm(tr.e[-1]) < 0.3 * np.linalg.norm(tr.e[0])
    assert np.abs(tr.x_e[-1, :3]).max() < 5e-3
