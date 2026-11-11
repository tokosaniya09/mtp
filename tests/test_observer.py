"""Tests for eqs. (6)-(9), Theorem 1: src/vsfc/observer.py (and the open-loop harness)."""

import numpy as np
import pytest

from vsfc import dynamics as dyn
from vsfc.config import load_case_study
from vsfc.leader_follower import relative_dynamics
from vsfc.observer import (AdaptiveVelocityObserver, error_dynamics, gain_condition, lyapunov_rate_bound,
                           pack_observer_state, sat)
from vsfc.sim.leaders import ConstantVelocityLeader, SinusoidalLeader
from vsfc.sim.observer_openloop import simulate_observer_open_loop

H = 2e-3   # [Eng] test step; RK4 stable with margin (fastest observer eigenvalue ~ -14 1/s)


@pytest.fixture(scope="module")
def cs():
    return load_case_study()


def _norms(a):
    return np.linalg.norm(a, axis=1)


# ---------- sat and Theorem 1 ----------

def test_sat_cases():
    eps = 0.5
    np.testing.assert_allclose(sat([2.0, -3.0, 0.5], eps), [1.0, -1.0, 1.0])     # outside / on boundary
    np.testing.assert_allclose(sat([0.25, -0.1, 0.0], eps), [0.5, -0.2, 0.0])    # inside: rho/eps
    np.testing.assert_allclose(sat([-0.5, 0.4999], eps), [-1.0, 0.9998])         # continuous at |rho| = eps
    assert np.all(np.abs(sat(np.random.default_rng(0).normal(size=100) * 5, eps)) <= 1.0)


def test_gain_condition(cs):
    # Sec. VI: alpha kappa = 25 (pair 12) and 275 (pair 13) vs 1/x* = 0.25.
    for f in cs.followers:
        assert AdaptiveVelocityObserver.from_config(f, cs.g).gain_condition()
    assert gain_condition(1.0, 0.26, 4.0) and not gain_condition(1.0, 0.24, 4.0)


def test_eq9_quadratic_form_definite_iff_gain_condition():
    # (9) = -[e;v]^T M [e;v] with M = [[alpha, -1/x*], [-1/x*, kappa/x*]] (per axis).
    for alpha, kappa, x_star in [(5, 5, 4), (5, 55, 4), (1.0, 0.26, 4), (1.0, 0.24, 4)]:
        M = np.array([[alpha, -1 / x_star], [-1 / x_star, kappa / x_star]])
        assert (np.linalg.eigvalsh(M).min() > 0) == gain_condition(alpha, kappa, x_star)


# ---------- algebra of (6) -> (7) ----------

def _random_case(rng, f, g):
    obs = AdaptiveVelocityObserver.from_config(f, g)
    e, v = rng.normal(size=3), rng.normal(size=3) * 2
    z = pack_observer_state(rng.normal(size=3), rng.normal(size=3) * 2, abs(rng.normal()) * 3)
    thrust = rng.normal(size=3) * 5 + f.mass * g * dyn.E3
    a_leader = rng.normal(size=3)
    return obs, e, v, z, thrust, a_leader


@pytest.mark.parametrize("idx", [0, 1])
def test_rho_identity(cs, idx):
    """rho = e_tilde_dot + alpha e_tilde = e_dot - v_hat/x* = v_tilde/x* (with exact e_dot)."""
    rng = np.random.default_rng(10 + idx)
    f = cs.followers[idx]
    for _ in range(50):
        obs, e, v, z, _, _ = _random_case(rng, f, cs.g)
        _, rho, zeta_bar = obs.signals(z, e, v / f.x_star)
        np.testing.assert_allclose(rho, (v - z[3:6]) / f.x_star, atol=1e-12)
        np.testing.assert_allclose(zeta_bar, z[6] * sat(rho, f.observer.eps), atol=1e-12)


@pytest.mark.parametrize("idx", [0, 1])
def test_eq7_follows_from_eq5_and_eq6(cs, idx):
    """d/dt(truth) from (5) minus d/dt(estimate) from (6) equals the error dynamics (7)."""
    rng = np.random.default_rng(20 + idx)
    f = cs.followers[idx]
    o = f.observer
    for _ in range(50):
        obs, e, v, z, thrust, a_leader = _random_case(rng, f, cs.g)
        e_dot, v_dot = relative_dynamics(v, thrust, a_leader, f.mass, cs.g, f.x_star)
        dz = obs.derivatives(z, e, e_dot, thrust)
        de, dv, dzeta = error_dynamics(e - z[0:3], v - z[3:6], z[6], a_leader, o.alpha, o.kappa, o.eps, f.x_star)
        np.testing.assert_allclose(e_dot - dz[0:3], de, atol=1e-12)
        np.testing.assert_allclose(v_dot - dz[3:6], dv, atol=1e-12)
        assert -dz[6] == pytest.approx(dzeta, abs=1e-12)          # zeta_i constant


@pytest.mark.parametrize("idx", [0, 1])
def test_lyapunov_rate_bound_outside_boundary_layer(cs, idx):
    """Eq. (9): with ||a_leader|| <= zeta_i, zeta_hat >= 0 and every |rho_k| >= eps, L_o_dot <= RHS(9) < 0."""
    rng = np.random.default_rng(30 + idx)
    f = cs.followers[idx]
    o, xs = f.observer, f.x_star
    for _ in range(500):
        e_t = rng.normal(size=3)
        v_t = rng.choice([-1, 1], size=3) * xs * o.eps * (1 + rng.exponential(size=3))   # |rho_k| >= eps
        zeta_i = rng.uniform(0, 3)
        a = rng.normal(size=3)
        a *= zeta_i * rng.uniform() / np.linalg.norm(a)
        zeta_hat = rng.uniform(0, 6)
        de, dv, dzeta = error_dynamics(e_t, v_t, zeta_hat, a, o.alpha, o.kappa, o.eps, xs)
        L_dot = e_t @ de + v_t @ dv + (zeta_i - zeta_hat) * dzeta
        bound = lyapunov_rate_bound(e_t, v_t, o.alpha, o.kappa, xs)
        assert L_dot <= bound + 1e-9
        assert bound < 0.0


# ---------- closed simulations (follower = eq. (1), prescribed excitation) ----------

@pytest.fixture(scope="module")
def paper_runs(cs):
    """[Paper] Sec. VI scenario for the leader (constant velocity), exact e_dot (D15), 10 s."""
    leader = ConstantVelocityLeader.from_config(cs.leader)
    return [simulate_observer_open_loop(f, leader, cs.leader.sphere_offset, cs.g, 10.0, H) for f in cs.followers]


def test_converges_paper_scenario(paper_runs):
    """Fig. 5 behaviour: e_tilde, v_tilde -> 0 within about 5-10 s."""
    for run in paper_runs:
        k5 = int(round(5.0 / H))
        assert _norms(run.v_tilde)[0] > 0.7                       # v_hat(0) = 0, |v_ij(0)| = 0.755
        assert _norms(run.v_tilde)[-1] < 1e-4 and _norms(run.e_tilde)[-1] < 1e-5
        assert _norms(run.v_tilde)[k5] < 1e-2
    k1 = int(round(1.0 / H))
    # kappa_13 = 55 >> kappa_12 = 5, so pair 13 converges much faster.
    assert _norms(paper_runs[1].v_tilde)[k1] < 0.01 * _norms(paper_runs[0].v_tilde)[k1]


def test_zeta_hat_nondecreasing(paper_runs):
    for run in paper_runs:
        assert np.all(np.diff(run.zeta_hat) >= -1e-12)


def test_stays_exact_when_started_at_truth(cs):
    f = cs.followers[0]
    leader = ConstantVelocityLeader.from_config(cs.leader)
    run = simulate_observer_open_loop(f, leader, cs.leader.sphere_offset, cs.g, 2.0, H,
                                      v_hat0=cs.leader.velocity - f.v0)
    assert _norms(run.v_tilde).max() < 1e-9 and _norms(run.e_tilde).max() < 1e-9
    assert run.zeta_hat[-1] < 1e-9


def test_bounded_with_accelerating_leader(cs):
    """[Eng] extension (A23): sinusoidal leader acceleration. Convergence is to a neighbourhood (A5); the
    adaptive term must do better than the kappa-only linear bound zeta_i x*/kappa."""
    f = cs.followers[0]
    leader = SinusoidalLeader(cs.leader.p0, cs.leader.velocity, np.array([0.5, 0.5, 0.2]), 1.0)
    run = simulate_observer_open_loop(f, leader, cs.leader.sphere_offset, cs.g, 30.0, H)
    late = _norms(run.v_tilde)[int(round(20.0 / H)):]
    linear_bound = leader.accel_bound * f.x_star / f.observer.kappa      # 0.59 m/s for pair 12
    assert late.max() < 0.5 * linear_bound
    assert np.all(np.isfinite(run.v_hat))
    assert run.zeta_hat[-1] > run.zeta_hat[int(round(10.0 / H))]          # keeps adapting (no leakage, A5)


def test_finite_difference_error_is_filter_lag(cs):
    """D15 check: with a filtered finite difference, |v_tilde| ~ (tau + h) |a_ij| (lag), not divergence.

    Pointwise the ratio oscillates (the observer's slow mode, ~1.2 rad/s for pair 12, filters the lag
    error, and a_ij oscillates at the 1.3 rad/s excitation), so the check uses the mean over 4-10 s and
    the scaling with tau."""
    f = cs.followers[0]
    leader = ConstantVelocityLeader.from_config(cs.leader)
    mean_ratio = {}
    for tau in (0.02, 0.05):
        run = simulate_observer_open_loop(f, leader, cs.leader.sphere_offset, cs.g, 10.0, H,
                                          e_dot_source="finite_difference", fd_tau=tau)
        a_ij = _norms(np.gradient(run.v_ij, run.t, axis=0))
        window = slice(int(round(4.0 / H)), -1)
        mean_ratio[tau] = np.mean(_norms(run.v_tilde)[window] / a_ij[window])
        assert 0.75 * (tau + H) < mean_ratio[tau] < 1.25 * (tau + H)
        assert np.all(np.isfinite(run.v_hat))
    assert mean_ratio[0.05] / mean_ratio[0.02] == pytest.approx((0.05 + H) / (0.02 + H), rel=0.15)
