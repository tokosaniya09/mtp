"""Tests for eq. (1): src/vsfc/dynamics.py and the RK4 integrator."""

import numpy as np
import pytest

from vsfc.dynamics import (E3, QuadrotorParams, euler_rate_matrix, euler_to_rotation, normalize_state,
                           pack_state, quadrotor_derivatives, rotation_to_euler, skew, unpack_state)
from vsfc.sim.integrators import integrate

G = 9.81
PARAMS = QuadrotorParams(m=1.3, J=np.diag([0.0082, 0.0080, 0.0145]), g=G)  # follower 2, Sec. VI


def Rx(a):
    return euler_to_rotation([a, 0.0, 0.0])


def Rz(a):
    return euler_to_rotation([0.0, 0.0, a])


def run(x0, f_T, tau, T, h, post=normalize_state):
    return integrate(lambda t, x: quadrotor_derivatives(x, f_T, tau, PARAMS), x0, 0.0, T, h, post)


# ---------- helpers ----------

def test_skew_matches_cross():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_euler_roundtrip_and_orthonormal():
    rng = np.random.default_rng(1)
    for _ in range(100):
        th = rng.uniform([-np.pi, -1.4, -np.pi], [np.pi, 1.4, np.pi])
        R = euler_to_rotation(th)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-14)
        assert np.linalg.det(R) == pytest.approx(1.0)
        np.testing.assert_allclose(rotation_to_euler(R), th, atol=1e-12)


def test_euler_rate_matrix_matches_finite_difference():
    # Theta_dot from W(Theta) omega must equal d/dt rotation_to_euler(R(t)) with R_dot = R sk(omega).
    th = np.array([0.3, -0.2, 0.5])
    w = np.array([0.4, -0.7, 0.9])
    R = euler_to_rotation(th)
    dt = 1e-6
    R_plus = R @ (np.eye(3) + skew(w) * dt + 0.5 * (skew(w) @ skew(w)) * dt**2)
    fd = (rotation_to_euler(R_plus) - th) / dt
    np.testing.assert_allclose(euler_rate_matrix(th) @ w, fd, atol=1e-5)


# ---------- eq. (1) right-hand side ----------

def test_hover_is_equilibrium():
    x = pack_state([1, 2, 3], np.zeros(3), np.eye(3), np.zeros(3))
    xd = quadrotor_derivatives(x, PARAMS.m * G, np.zeros(3), PARAMS)
    np.testing.assert_allclose(xd, np.zeros(18), atol=1e-14)


def test_free_fall_exact():
    # f_T = 0 -> p(t) = p0 + 0.5 g t^2 e3 ; RK4 is exact for this quadratic.
    x0 = pack_state(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    t, X = run(x0, 0.0, np.zeros(3), T=2.0, h=0.01)
    p, v, _, _ = unpack_state(X[-1])
    np.testing.assert_allclose(p, 0.5 * G * 2.0**2 * E3, atol=1e-12)
    np.testing.assert_allclose(v, G * 2.0 * E3, atol=1e-12)


def test_tilted_thrust_sign_convention():
    # Positive roll phi, f_T = m g / cos(phi): no vertical acceleration, +y acceleration g tan(phi).
    phi = 0.2
    x = pack_state(np.zeros(3), np.zeros(3), Rx(phi), np.zeros(3))
    xd = quadrotor_derivatives(x, PARAMS.m * G / np.cos(phi), np.zeros(3), PARAMS)
    np.testing.assert_allclose(xd[3:6], [0.0, G * np.tan(phi), 0.0], atol=1e-12)


def test_torque_gives_angular_acceleration():
    x = pack_state(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    tau = np.array([0.01, -0.02, 0.03])
    xd = quadrotor_derivatives(x, 0.0, tau, PARAMS)
    np.testing.assert_allclose(xd[15:18], tau / np.diag(PARAMS.J))


# ---------- integration ----------

def test_constant_yaw_rate_matches_closed_form():
    wz = 0.8
    x0 = pack_state(np.zeros(3), np.zeros(3), np.eye(3), [0, 0, wz])
    _, X = run(x0, PARAMS.m * G, np.zeros(3), T=5.0, h=0.001)
    _, _, R, w = unpack_state(X[-1])
    np.testing.assert_allclose(R, Rz(wz * 5.0), atol=1e-10)
    np.testing.assert_allclose(w, [0, 0, wz], atol=1e-12)


def test_torque_free_invariants():
    # Energy 0.5 w^T J w and inertial angular momentum R J w are conserved; R stays in SO(3).
    w0 = np.array([0.5, 1.5, -0.7])  # non-principal spin, asymmetric J -> nontrivial motion
    x0 = pack_state(np.zeros(3), np.zeros(3), euler_to_rotation([0.1, -0.2, 0.3]), w0)
    _, X = run(x0, 0.0, np.zeros(3), T=20.0, h=0.001)
    J = PARAMS.J
    _, _, R0, _ = unpack_state(X[0])
    E0, L0 = 0.5 * w0 @ J @ w0, R0 @ J @ w0
    for x in X[::1000]:
        _, _, R, w = unpack_state(x)
        assert 0.5 * w @ J @ w == pytest.approx(E0, rel=1e-8)
        np.testing.assert_allclose(R @ J @ w, L0, atol=1e-9)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)


def test_rk4_fourth_order_convergence():
    w0 = np.array([0.5, 1.5, -0.7])
    x0 = pack_state(np.zeros(3), np.zeros(3), np.eye(3), w0)
    _, X_ref = run(x0, 0.0, np.zeros(3), T=2.0, h=0.0005, post=None)
    errs = []
    for h in (0.04, 0.02):
        _, X = run(x0, 0.0, np.zeros(3), T=2.0, h=h, post=None)
        errs.append(np.linalg.norm(X[-1] - X_ref[-1]))
    assert 12.0 < errs[0] / errs[1] < 20.0  # ideal ratio 2^4 = 16


def test_integrate_rejects_non_multiple_step():
    x0 = pack_state(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError):
        run(x0, 0.0, np.zeros(3), T=1.0, h=0.3)
