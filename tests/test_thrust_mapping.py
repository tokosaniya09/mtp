"""Tests for the [Eng] thrust-vector decomposition (A15, D30): src/vsfc/controllers/thrust_mapping.py."""

import numpy as np
import pytest

from vsfc.config import load_case_study
from vsfc.controllers.thrust_mapping import (ReferenceGenerator, decompose_thrust, thrust_mapping,
                                             thrust_vector, tracking_error, wrap_angle)
from vsfc.dynamics import E3, euler_to_rotation

G = 9.81
MASSES = (1.3, 2.0)       # [Paper] Sec. VI, m2 and m3
BETA = 5.0                # [Paper] Sec. VI


@pytest.fixture(scope="module")
def cs():
    return load_case_study()


@pytest.mark.parametrize("m", MASSES)
def test_hover_is_level(m):
    f_T, theta_r = thrust_mapping(np.zeros(3), m, G)
    assert f_T == pytest.approx(m * G, rel=1e-15)
    np.testing.assert_allclose(theta_r, 0.0, atol=1e-15)


@pytest.mark.parametrize("psi_r", [0.0, 0.7, -2.5])
@pytest.mark.parametrize("m", MASSES)
def test_round_trip_reproduces_thrust_vector(m, psi_r):
    rng = np.random.default_rng(0)
    for _ in range(2000):
        mu = rng.normal(size=3)
        mu *= BETA * rng.uniform() / np.linalg.norm(mu)
        F = thrust_vector(mu, m, G)
        f_T, theta_r = decompose_thrust(F, psi_r)
        assert theta_r[2] == psi_r
        np.testing.assert_allclose(f_T * euler_to_rotation(theta_r) @ E3, F, atol=1e-12)


def test_thrust_vector_is_gravity_compensated():
    np.testing.assert_allclose(thrust_vector([1.0, -2.0, 0.5], 1.3, G), [1.0, -2.0, 0.5 + 1.3 * G])


def test_tilt_direction_signs():
    # e3 points down (A12): +x force needs a positive pitch, +y force a negative roll (ZYX).
    _, th = thrust_mapping([2.0, 0.0, 0.0], 1.3, G)
    assert th[1] > 0 and th[0] == pytest.approx(0.0, abs=1e-15)
    _, th = thrust_mapping([0.0, 2.0, 0.0], 1.3, G)
    assert th[0] < 0 and th[1] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("m", MASSES)
def test_tilt_bound(m):
    """E11: over ||mu|| <= beta the tilt is at most arcsin(beta/(m g)), attained for mu tangent to the sphere."""
    bound = np.arcsin(BETA / (m * G))
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(5000):
        mu = rng.normal(size=3)
        mu *= BETA / np.linalg.norm(mu)
        f_T, theta_r = thrust_mapping(mu, m, G)
        worst = max(worst, np.arccos((euler_to_rotation(theta_r) @ E3)[2]))
    assert worst <= bound + 1e-12
    # Tangent mu: F perpendicular to mu, so the tilt equals the bound.
    a = np.pi / 2 + bound
    _, theta_r = thrust_mapping(BETA * np.array([np.sin(a), 0.0, np.cos(a)]), m, G)
    assert np.arccos((euler_to_rotation(theta_r) @ E3)[2]) == pytest.approx(bound, abs=1e-12)
    assert np.degrees(bound) == pytest.approx({1.3: 23.08, 2.0: 14.76}[m], abs=0.01)


def test_rejects_degenerate_or_inverted_thrust():
    with pytest.raises(ValueError):
        decompose_thrust(np.zeros(3))
    with pytest.raises(ValueError):
        decompose_thrust(np.array([0.1, 0.0, -1.0]))


def test_wrap_angle():
    np.testing.assert_allclose(wrap_angle([2 * np.pi - 0.1, -2 * np.pi + 0.1, 0.3, np.pi, -np.pi]),
                               [-0.1, 0.1, 0.3, np.pi, np.pi], atol=1e-15)


def test_tracking_error_wraps_yaw():
    e_th, e_om = tracking_error([0.1, -0.2, 3.1], [1.0, 2.0, 3.0], [0.0, 0.0, -3.1], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(e_th, [0.1, -0.2, 6.2 - 2 * np.pi], atol=1e-15)
    np.testing.assert_allclose(e_om, [0.5, 1.5, 2.5])


# --- reference generator: omega_r -----------------------------------------------------------------------------

AMP = np.array([3.0, 2.5, 1.5])      # N, keeps ||mu|| < m g
FREQ = np.array([1.1, 0.8, 1.7])     # rad/s


def _mu(t):
    return AMP * np.sin(FREQ * t + np.array([0.0, 1.0, 2.0]))


def _true_body_rate(t, m, psi_r, h=1e-6):
    """Independent truth: sk(omega) = R^T R_dot, with R(t) = R(Theta_r(t)) and R_dot by central difference."""
    R = lambda s: euler_to_rotation(thrust_mapping(_mu(s), m, G, psi_r)[1])
    S = R(t).T @ (R(t + h) - R(t - h)) / (2 * h)
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def _run(gen, m, psi_r, dt, T):
    gen.reset(_mu(0.0))
    out = []
    for k in range(1, int(round(T / dt)) + 1):
        t = k * dt
        out.append((t, gen.update(_mu(t)).omega_r, _true_body_rate(t, m, psi_r)))
    return out


@pytest.mark.parametrize("psi_r", [0.0, 0.7])
def test_omega_r_matches_body_rate_raw_difference(psi_r):
    m, dt = 1.3, 1e-3
    out = _run(ReferenceGenerator(m, G, dt, 0.0, psi_r), m, psi_r, dt, 6.0)
    err = max(np.abs(w - w_true).max() for t, w, w_true in out[1:])
    scale = max(np.abs(w_true).max() for _, _, w_true in out)
    assert scale > 0.1                     # the test signal really moves the reference
    assert err < 2e-3 * scale              # backward-difference error O(dt/2 * Theta_ddot)


def test_omega_r_filter_lag_is_bounded():
    m, dt, tau = 1.3, 1e-3, 0.02
    out = _run(ReferenceGenerator(m, G, dt, tau), m, 0.0, dt, 8.0)
    after = [(t, w, w_true) for t, w, w_true in out if t > 10 * tau]
    err = max(np.abs(w - w_true).max() for _, w, w_true in after)
    w_true = np.array([x[2] for x in out])
    accel = np.abs(np.diff(w_true, axis=0) / dt).max()
    assert err <= 1.25 * (tau + dt) * accel     # first-order lag: error ~ (tau + dt) * |omega_dot|
    assert err < 0.1 * np.abs(w_true).max()


def test_constant_mu_gives_zero_rates():
    gen = ReferenceGenerator(2.0, G, 1e-3, 0.02)
    gen.reset([1.0, -0.5, 0.3])
    for _ in range(100):
        ref = gen.update([1.0, -0.5, 0.3])
    np.testing.assert_allclose(ref.omega_r, 0.0, atol=1e-15)
    f_T, theta_r = thrust_mapping([1.0, -0.5, 0.3], 2.0, G)
    assert ref.f_T == f_T
    np.testing.assert_allclose(ref.theta_r, theta_r)


def test_from_case_study_and_reset_required(cs):
    gen = ReferenceGenerator.from_case_study(cs, cs.followers[0])
    assert gen.mass == cs.followers[0].mass and gen.psi_r == cs.psi_ref == 0.0
    assert gen._diff.dt == cs.step and cs.ref_rate_tau == 0.02
    with pytest.raises(RuntimeError):
        gen.update(np.zeros(3))
