"""Tests for sim/outer_loop.py: Phase 1.5 outer-loop system (ideal attitude D24, D14, observer (6)) and data
collection for Alg. 3 (D26). Checks the claims of docs/equation_map.md §5.8."""

import dataclasses

import numpy as np
import pytest

from vsfc.config import load_case_study
from vsfc.rl.data import sum_of_sinusoids
from vsfc.sim.outer_loop import (OuterLoopSystem, behaviour_policy, collect_dataset, per_axis_gain, simulate)

H = 1e-3


@pytest.fixture(scope="module")
def case():
    return load_case_study()


@pytest.fixture(scope="module")
def sys2(case):
    return OuterLoopSystem.from_case_study(case, case.followers[0])


def test_initial_state_reproduces_requested_relative_state(sys2):
    e0, v0 = np.array([1.2, -0.4, 0.3]), np.array([0.5, -1.0, 0.2])
    x0 = sys2.initial_state(e0, v0)
    e, v = sys2.relative(0.0, x0)
    np.testing.assert_allclose(e, e0, atol=1e-12)
    np.testing.assert_allclose(v, v0, atol=1e-12)
    np.testing.assert_allclose(sys2.delta(0.0, x0), np.concatenate([e0, np.zeros(3)]))   # A19: v_hat(0) = 0


def test_paper_initial_condition(case, sys2):
    """Sec. VI: p2(0) = [0.4, -0.3, 0] gives e(0) = [-0.475, -0.425, 0] (Phase 1.2 hand value)."""
    x0 = sys2.initial_state_from_config(case.followers[0])
    np.testing.assert_allclose(sys2.relative(0.0, x0)[0], [-0.475, -0.425, 0.0], atol=1e-12)


def test_zero_mu_is_hover(sys2):
    """D14: mu = 0 gives thrust m g e3, so the follower's acceleration is zero."""
    x0 = sys2.initial_state(np.array([0.5, 0.0, 0.0]), np.zeros(3))
    xd = sys2.derivatives(0.0, x0, np.zeros(3))
    np.testing.assert_allclose(xd[3:6], 0.0, atol=1e-12)
    np.testing.assert_allclose(sys2.thrust_vector(np.zeros(3)), [0, 0, sys2.mass * sys2.g])


def test_behaviour_policy_inside_bound(sys2):
    pol = behaviour_policy(per_axis_gain(1.0, 1.5), sys2.beta, lambda t: np.array([50.0, -50.0, 0.0]))
    mu = pol(0.0, np.array([10.0, -10.0, 3.0, 5.0, 5.0, 5.0]))
    assert np.all(np.abs(mu) < sys2.beta)
    # without exploration, small delta: mu ~ -K0 delta
    d = 1e-4 * np.arange(1, 7)
    np.testing.assert_allclose(behaviour_policy(per_axis_gain(1.0, 1.5), 5.0)(0.0, d),
                               -per_axis_gain(1.0, 1.5) @ d, rtol=1e-6)


def test_delta_follows_double_integrators_after_burn_in(sys2):
    """§5.8: once the observer has converged, e_dot = v_hat / x* and v_hat_dot = mu / m (checked by central
    differences on the simulated trajectory, 6-10 s, exploring behaviour policy)."""
    expl = sum_of_sinusoids(1.0, [0.4, 1.1, 2.3], 3, seed=3)
    pol = behaviour_policy(per_axis_gain(1.0, 1.5), sys2.beta, expl)
    tr = simulate(sys2, pol, sys2.initial_state(np.array([1.0, 0.5, -0.5]), np.array([0.3, -0.2, 0.4])), 10.0, H)
    i = np.arange(int(6 / H), int(10 / H) - 1)
    d = tr.delta
    e_dot = (d[i + 1, :3] - d[i - 1, :3]) / (2 * H)
    vh_dot = (d[i + 1, 3:] - d[i, 3:]) / H                    # forward difference: mu is constant on [t_i, t_i+1)
    err_e = np.abs(e_dot - 0.5 * (d[i + 1, 3:] + d[i - 1, 3:]) / sys2.x_star).max()
    err_v = np.abs(vh_dot - tr.mu[i] / sys2.mass).max()
    assert err_e < 1e-4                    # residual = v_tilde / x*, observer converged
    assert err_v < 1e-2 * np.abs(tr.mu[i] / sys2.mass).max()
    # and the observer error is small, which is why the burn-in is dropped (D26)
    assert np.abs(tr.v_true[i] - d[i, 3:]).max() < 1e-3
    early = np.abs(tr.v_true[:int(1 / H)] - d[:int(1 / H), 3:]).max()
    assert early > 0.1                     # during the first second the model above does NOT hold


def test_collect_dataset_shapes_and_burn_in(case, sys2):
    spec = dataclasses.replace(case.learning_outer.data, n_runs=2, run_length=2.0, burn_in=1.0)
    data, trajs = collect_dataset(sys2, spec, H, case.dT)
    per_run = int(round((2.0 - 1.0) / case.dT))
    assert data.n_windows == 2 * per_run and data.steps_per_window == 50
    # first window of run 0 starts at t = burn_in
    np.testing.assert_allclose(data.delta[0, 0], trajs[0].delta[int(1.0 / H)])
    np.testing.assert_allclose(data.mu[0], trajs[0].mu[int(1.0 / H):int(1.0 / H) + 50])
    assert np.all(np.abs(data.mu) < sys2.beta)
    with pytest.raises(ValueError):
        collect_dataset(sys2, dataclasses.replace(spec, burn_in=1.01), H, case.dT)


def test_sum_of_sinusoids_per_frequency_amplitude():
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    f = sum_of_sinusoids(A, [1.0, 3.0], 2, seed=5)
    g1 = sum_of_sinusoids(1.0, [1.0], 2, seed=5)               # same RNG draw for the first frequency's phases
    t = 0.7
    assert f(t)[0] == pytest.approx(g1(t)[0])                  # channel 0 only has the 1 rad/s term
    rng = np.random.default_rng(5)
    ph = rng.uniform(0, 2 * np.pi, size=(2, 2))
    assert f(t)[1] == pytest.approx(2.0 * np.sin(3.0 * t + ph[1, 1]))
