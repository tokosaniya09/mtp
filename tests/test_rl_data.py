"""Tests for rl/data.py: windowing of off-policy trajectories (Alg. 3 Step 1, A18)."""

import numpy as np
import pytest

from vsfc.rl.data import OffPolicyDataset, rollout, sum_of_sinusoids


def test_from_trajectory_shapes_and_shared_window_ends():
    K, n, m, W = 23, 2, 3, 5
    delta = np.arange((K + 1) * n, dtype=float).reshape(K + 1, n)
    mu = np.arange(K * m, dtype=float).reshape(K, m)
    d = OffPolicyDataset.from_trajectory(delta, mu, h=0.01, steps_per_window=W)
    assert d.n_windows == 4                      # 23 // 5, trailing 3 steps dropped
    assert d.delta.shape == (4, W + 1, n) and d.mu.shape == (4, W, m)
    np.testing.assert_array_equal(d.delta[1, 0], d.delta[0, -1])   # consecutive windows share an end
    np.testing.assert_array_equal(d.delta[2, 3], delta[13])
    np.testing.assert_array_equal(d.mu[3, 4], mu[19])
    assert d.dT == pytest.approx(0.05)


def test_from_trajectory_rejects_bad_lengths():
    with pytest.raises(ValueError):
        OffPolicyDataset.from_trajectory(np.zeros((10, 1)), np.zeros((10, 1)), 0.1, 2)
    with pytest.raises(ValueError):
        OffPolicyDataset.from_trajectory(np.zeros((3, 1)), np.zeros((2, 1)), 0.1, 5)


def test_concatenate_requires_same_grid():
    a = OffPolicyDataset.from_trajectory(np.zeros((11, 1)), np.zeros((10, 1)), 0.1, 5)
    b = OffPolicyDataset.from_trajectory(np.zeros((11, 1)), np.zeros((10, 1)), 0.2, 5)
    assert OffPolicyDataset.concatenate([a, a]).n_windows == 4
    with pytest.raises(ValueError):
        OffPolicyDataset.concatenate([a, b])


def test_rollout_zero_order_hold_is_exact_for_integrator():
    # x_dot = u, u_i = t_i held on [t_i, t_i + h): x(t_K) = sum_i h t_i exactly.
    h, T = 0.1, 1.0
    t, X, delta, mu = rollout(lambda t, x, u: u, np.zeros(1), lambda t, x: np.array([t]), T, h)
    assert X.shape == (11, 1) and mu.shape == (10, 1)
    np.testing.assert_allclose(X[-1, 0], h * t[:-1].sum(), atol=1e-14)
    np.testing.assert_array_equal(delta, X)      # identity observe


def test_rollout_observe_map_and_bad_T():
    _, X, delta, _ = rollout(lambda t, x, u: -x, np.array([1.0, 2.0]), lambda t, x: np.zeros(1), 0.5, 0.01,
                             observe=lambda x: x[:1])
    assert delta.shape == (51, 1)
    np.testing.assert_allclose(delta[-1, 0], np.exp(-0.5), rtol=1e-10)
    with pytest.raises(ValueError):
        rollout(lambda t, x, u: u, np.zeros(1), lambda t, x: np.zeros(1), 0.55, 0.1)


def test_sum_of_sinusoids_shape_and_amplitude():
    e = sum_of_sinusoids(0.5, [1.0, 2.0, 3.0], m=2, seed=3)
    vals = np.array([e(t) for t in np.linspace(0, 20, 2001)])
    assert vals.shape == (2001, 2)
    assert np.abs(vals).max() <= 1.5 + 1e-12
    assert not np.allclose(vals[:, 0], vals[:, 1])   # independent phases per channel
