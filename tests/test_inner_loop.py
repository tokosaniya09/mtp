"""Tests for sim/inner_loop.py: attitude learning problem (Phase 1.6 Step 2, D29, D31; docs/equation_map.md §6.1).

Known answer: the linearised attitude error (three double integrators e_Theta_dot = e_omega, e_omega_dot = u/J_k)
learned by Alg. 4 with gamma = 1000 (tanh ~ linear) must give the Riccati P and K of §6.1 Consequence 3, for both
the main basis B and the check basis A (D31). Quartic/cubic weights of A must vanish (exact value is quadratic).
"""

import numpy as np
import pytest

from vsfc.config import load_case_study
from vsfc.rl.basis import weights_from_quadratic_form
from vsfc.rl.data import OffPolicyDataset, rollout, sum_of_sinusoids
from vsfc.rl.offpolicy_irl import train
from vsfc.sim.inner_loop import (ERROR_NAMES, build_inner_problem, error_cost_weight, initial_actor_from_gain,
                                 linearised_attitude, lqr_reference_inner, per_axis_lqr_gains)

H, W = 1e-3, 50                       # A18: 1 ms sub-steps; dT = 0.05 s (Sec. VI)
GAMMA_LIN = 1000.0                    # unconstrained limit for the known-answer test
K_THETA0, K_OMEGA0 = 0.5, 0.1         # behaviour/initial PD gains (N m/rad, N m s/rad): stabilising


@pytest.fixture(scope="module")
def case():
    return load_case_study()


@pytest.fixture(scope="module")
def fol(case):
    return next(f for f in case.followers if f.id == 2)


def test_config_bases(case):
    li = case.learning_inner
    assert li.critic_degrees == (2, 4) and li.actor_degrees == (1, 3)              # main basis A (D31, D33)
    assert li.check_critic_degrees == (2,) and li.check_actor_degrees == (1,)      # check basis B (D31, D33)
    assert li.max_iter >= 10


def test_both_followers_share_the_attitude_problem(case):
    """§6.1: J, S, D, gamma are the same for both followers, so one attitude problem serves both."""
    f2, f3 = case.followers
    for name in ("inertia", "S", "D"):
        np.testing.assert_array_equal(getattr(f2, name), getattr(f3, name))
    assert f2.gamma == f3.gamma == 1.0


@pytest.mark.parametrize("which, l_c, l_a", [("main", 24, 6), ("check", 9, 2)])
def test_basis_sizes_and_labels(case, fol, which, l_c, l_a):
    li = case.learning_inner
    degs = ((li.critic_degrees, li.actor_degrees) if which == "main"
            else (li.check_critic_degrees, li.check_actor_degrees))
    prob = build_inner_problem(fol, *degs)
    assert prob.critic.size == l_c and prob.l_a == l_a and prob.m == 3
    assert prob.bound == fol.gamma
    np.testing.assert_array_equal(prob.r, [8.0, 8.0, 8.0])
    np.testing.assert_array_equal(prob.Q, 10.0 * np.eye(6))
    np.testing.assert_array_equal(error_cost_weight(fol), fol.S[:6, :6])
    assert prob.critic.labels(ERROR_NAMES)[:3] == ["eT1^2", "eT1*ew1", "ew1^2"]
    for k, a in enumerate(prob.actors):                                  # channel k uses only its own axis
        assert a.labels(ERROR_NAMES)[:2] == [f"eT{k + 1}", f"ew{k + 1}"]


def test_lqr_reference_matches_equation_map(fol):
    """Values of the §6.1 Consequence 3 table."""
    P, K = lqr_reference_inner(fol)
    np.testing.assert_allclose(per_axis_lqr_gains(K),
                               [[1.1180, 1.1262], [1.1180, 1.1260], [1.1180, 1.1324]], atol=6e-5)
    np.testing.assert_allclose([P[0, 0], 2 * P[0, 3], P[3, 3]], [10.073, 0.147, 0.0739], atol=6e-4)
    np.testing.assert_allclose([P[2, 2], 2 * P[2, 5], P[5, 5]], [10.129, 0.259, 0.1314], atol=6e-4)
    A, B = linearised_attitude(fol)
    poles = np.sort(np.linalg.eigvals(A - B @ K).real)
    np.testing.assert_allclose(poles, [-139.75, -136.34, -77.10, -1.0, -1.0, -1.0], atol=0.01)


def test_check_critic_can_represent_lqr_value(fol, case):
    """P is block-diagonal by axis, so even the smaller basis B (check) represents x^T P x exactly
    (weights_from_quadratic_form would raise otherwise)."""
    li = case.learning_inner
    prob = build_inner_problem(fol, li.check_critic_degrees, li.check_actor_degrees)
    P, _ = lqr_reference_inner(fol)
    x = np.random.default_rng(0).normal(size=(20, 6))
    wP = weights_from_quadratic_form(prob.critic, P)
    np.testing.assert_allclose(prob.critic(x) @ wP, np.einsum("ni,ij,nj->n", x, P, x), rtol=1e-12)


def test_initial_actor_layout(fol):
    prob = build_inner_problem(fol, (2,), (1,))
    W0 = initial_actor_from_gain(prob, 0.5, 0.1)
    x = np.zeros(6); x[1], x[4] = 0.2, -0.3                               # axis 2 only
    xi = np.einsum("kl,kl->k", prob.actor_features(x), W0)
    np.testing.assert_allclose(xi, [0.0, (0.5 * 0.2 - 0.1 * 0.3) / fol.gamma, 0.0])


@pytest.fixture(scope="module")
def linear_attitude_data(fol):
    """6 runs x 5 s of the linearised attitude error under u = -K0 x + sum of sinusoids (amplitude 0.05 N m)."""
    A, B = linearised_attitude(fol)
    K0 = np.hstack([K_THETA0 * np.eye(3), K_OMEGA0 * np.eye(3)])
    rng = np.random.default_rng(7)
    runs = []
    for s in range(6):
        e = sum_of_sinusoids(0.05, [0.5, 1.3, 2.9, 4.7, 7.1, 13.0], 3, seed=10 + s)
        _, _, d, u = rollout(lambda t, x, u: A @ x + B @ u, rng.uniform(-0.5, 0.5, 6),
                             lambda t, x, e=e: -K0 @ x + e(t), 5.0, H)
        runs.append(OffPolicyDataset.from_trajectory(d, u, H, W))
    return OffPolicyDataset.concatenate(runs)


@pytest.mark.parametrize("cdeg, adeg, tol", [((2,), (1,), 1e-5), ((2, 4), (1, 3), 1e-4)])
def test_lqr_known_answer_attitude_bases(fol, linear_attitude_data, cdeg, adeg, tol):
    """Basis B (main) and A (check), gamma = 1000: critic -> Riccati P, actor x gamma -> K (§6.1)."""
    prob = build_inner_problem(fol, cdeg, adeg)
    prob = type(prob)(prob.critic, prob.actors, GAMMA_LIN, prob.r, prob.Q)
    res = train(linear_attitude_data, prob, initial_actor_from_gain(prob, K_THETA0, K_OMEGA0), tol=fol.tol_att)
    assert res.converged and res.iterations <= 12
    assert res.history[-1]["rank"] == prob.n_weights
    P, K = lqr_reference_inner(fol)
    wP = weights_from_quadratic_form(prob.critic, P)
    assert np.linalg.norm(res.w_c - wP) / np.linalg.norm(wP) < tol
    np.testing.assert_allclose(res.W_a[:, :2] * GAMMA_LIN, per_axis_lqr_gains(K), rtol=tol)
    hi = prob.critic.total_degrees > 2
    if hi.any():
        assert np.abs(res.w_c[hi]).max() < tol * np.abs(wP).max()
        assert np.abs(res.W_a[:, 2:]).max() * GAMMA_LIN < tol * np.abs(K).max()


# ---- data system (Phase 1.6 Step 3, D32, D33) ---------------------------------------------------------------

from vsfc.config import InnerDataConfig                                          # noqa: E402
from vsfc.controllers.thrust_mapping import tracking_error                       # noqa: E402
from vsfc.dynamics import (QuadrotorParams, pack_state, quadrotor_derivatives,  # noqa: E402
                           rotation_to_euler)
from vsfc.sim.inner_loop import (AttitudeSystem, SinusoidalAttitudeReference,    # noqa: E402
                                 behaviour_torque, collect_attitude_dataset, per_axis_gain, simulate_attitude)


@pytest.fixture(scope="module")
def ref():
    return SinusoidalAttitudeReference.random([[0.05, 0.15, 0.0], [0.05, 0.15, 0.0]], [0.4, 0.9],
                                              np.random.default_rng(3))


def test_rotational_dynamics_match_eq1(fol, ref):
    """AttitudeSystem.derivatives = the R and omega rows of quadrotor_derivatives (eq. (1))."""
    rng = np.random.default_rng(4)
    sysm = AttitudeSystem(fol.inertia, ref)
    params = QuadrotorParams(m=fol.mass, J=fol.inertia, g=9.81)
    for _ in range(10):
        x = sysm.initial_state(rng.uniform(-0.5, 0.5, 3), rng.uniform(-1, 1, 3), t0=rng.uniform(0, 5))
        u = rng.uniform(-1, 1, 3)
        full = quadrotor_derivatives(pack_state(np.zeros(3), np.zeros(3), x[:9].reshape(3, 3), x[9:]), 10.0, u,
                                     params)
        np.testing.assert_allclose(sysm.derivatives(0.0, x, u), full[6:], rtol=1e-12, atol=1e-12)


def test_reference_rates(ref):
    """Theta_r_dot = finite difference of Theta_r; omega_r = W(Theta_r)^-1 Theta_r_dot (D30 definition)."""
    from vsfc.dynamics import euler_rate_matrix
    for t in [0.0, 1.3, 4.7]:
        th, th_dot, om = ref.state(t)
        fd = (ref.state(t + 1e-6)[0] - ref.state(t - 1e-6)[0]) / 2e-6
        np.testing.assert_allclose(th_dot, fd, atol=1e-8)
        np.testing.assert_allclose(euler_rate_matrix(th) @ om, th_dot, atol=1e-12)
        assert th[2] == 0.0 and np.abs(th).max() <= 0.3


def test_initial_state_has_requested_error(fol, ref):
    sysm = AttitudeSystem(fol.inertia, ref)
    e0, w0 = np.array([0.1, -0.5, 0.2]), np.array([0.3, -1.0, 0.4])
    np.testing.assert_allclose(sysm.error(2.0, sysm.initial_state(e0, w0, t0=2.0)), np.r_[e0, w0], atol=1e-12)


def test_euler_rate_error_is_derivative_of_angle_error(fol, ref):
    """D33: along a trajectory of the exact plant, d/dt e_Theta = e_omega (Euler-rate error) to integration accuracy;
    the D29 body-rate error omega - omega_r does not satisfy this at large angles (E13)."""
    sysm = AttitudeSystem(fol.inertia, ref)
    pol = behaviour_torque(per_axis_gain(0.5, 0.1), fol.gamma, lambda t: 0.08 * np.sin([9 * t, 17 * t, 29 * t]))
    tr = simulate_attitude(sysm, pol, sysm.initial_state([0.2, 0.6, 0.2], [0.5, -1.0, 0.5]), 1.0, 1e-3)
    de = np.gradient(tr.x_e[:, :3], tr.t, axis=0)[5:-5]
    np.testing.assert_allclose(de, tr.x_e[5:-5, 3:], atol=2e-3 * np.abs(tr.x_e[:, 3:]).max())
    body = np.array([tracking_error(rotation_to_euler(x[:9].reshape(3, 3)), x[9:], th, om)[1]
                     for x, th, om in zip(tr.x, tr.theta_r, tr.omega_r)])[5:-5]
    assert np.abs(de - body).max() > 0.1                                   # body-rate error is not e_Theta_dot


def test_behaviour_torque_bounded_and_stabilising(fol):
    sysm = AttitudeSystem(fol.inertia, SinusoidalAttitudeReference.zero())
    tr = simulate_attitude(sysm, behaviour_torque(per_axis_gain(0.5, 0.1), fol.gamma),
                           sysm.initial_state([0.3, -0.6, 0.3], [1.0, 1.0, -1.0]), 8.0, 1e-3)
    assert np.abs(tr.u).max() < fol.gamma
    assert np.abs(tr.x_e[-1]).max() < 1e-3


def test_collect_attitude_dataset_small(fol):
    spec = InnerDataConfig(n_runs=2, run_length=1.0, k_theta=0.5, k_omega=0.1, explore_freqs=np.array([0.7, 17.0]),
                           explore_amplitude=np.array([[0.05, 0.1, 0.05], [0.08, 0.08, 0.08]]),
                           e_theta0_max=np.array([0.25, 0.6, 0.25]), e_omega0_max=np.array([0.6, 1.5, 0.6]),
                           ref_freqs=np.array([0.4]), ref_amplitude=np.array([[0.05, 0.15, 0.0]]), seed=1)
    data, trajs = collect_attitude_dataset(fol, spec, 1e-3, 0.05)
    assert data.delta.shape == (40, 51, 6) and data.mu.shape == (40, 50, 3)
    assert np.abs(data.mu).max() < fol.gamma
    assert np.abs(data.delta[..., :3]).max() < 1.0                       # no wrapping (E12)
    np.testing.assert_array_equal(data.delta[0, 0], trajs[0].x_e[0])


def test_data_config_rejects_yaw_reference():
    from vsfc.config import _inner_data
    d = dict(n_runs=1, run_length=1.0, k_theta=0.5, k_omega=0.1, explore_freqs=[1.0], explore_amplitude=[[0.1] * 3],
             e_theta0_max=[0.1] * 3, e_omega0_max=[0.1] * 3, ref_freqs=[0.4], ref_amplitude=[[0.1, 0.1, 0.1]], seed=1)
    with pytest.raises(ValueError):
        _inner_data(d)
