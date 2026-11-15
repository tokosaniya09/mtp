"""Tests for controllers/cascade.py and sim/cascade.py (Phase 1.6 Step 6): plant (1) + observer (6) + dual loop.

The policies here are the LQR-limit tanh policies of both loops (docs/equation_map.md §5.8, §6.1), so the tests
do not depend on trained weight files.
"""

import numpy as np
import pytest
from scipy.linalg import solve_continuous_are

from vsfc.config import load_case_study
from vsfc.controllers.cascade import CascadeController
from vsfc.controllers.thrust_mapping import ReferenceGenerator
from vsfc.dynamics import E3, euler_rate_matrix, euler_to_rotation, quadrotor_derivatives
from vsfc.sim.cascade import STATE_DIM, CascadeSystem, simulate_cascade
from vsfc.sim.inner_loop import lqr_reference_inner

H = 1e-3


@pytest.fixture(scope="module")
def case():
    return load_case_study()


@pytest.fixture(scope="module")
def fol(case):
    return next(f for f in case.followers if f.id == 2)


def lqr_policies(case, fol):
    """outer: mu = -beta tanh(K delta / beta) (§5.8); inner: u = -gamma tanh(K x_e / gamma) (§6.1)."""
    A = np.zeros((6, 6)); A[:3, 3:] = np.eye(3) / fol.x_star
    B = np.zeros((6, 3)); B[3:] = np.eye(3) / fol.mass
    Ko = np.linalg.solve(fol.R, B.T @ solve_continuous_are(A, B, fol.Q, fol.R))
    _, Ki = lqr_reference_inner(fol)
    return (lambda d: -fol.beta * np.tanh(Ko @ d / fol.beta)), (lambda xe: -fol.gamma * np.tanh(Ki @ xe / fol.gamma))


def make_controller(case, fol):
    outer, inner = lqr_policies(case, fol)
    return CascadeController(outer, inner, ReferenceGenerator.from_case_study(case, fol))


def test_command_is_consistent(case, fol):
    """f_T R(Theta_r) e3 = mu + m g e3 (D14, D30); x_e follows D29/D33; inputs within their bounds."""
    ctrl = make_controller(case, fol)
    theta, omega = np.array([0.05, -0.1, 0.174]), np.array([0.2, -0.3, 0.1])
    c = ctrl.step(np.array([0.5, -0.4, 0.2, 0.1, 0.0, -0.2]), theta, omega)
    np.testing.assert_allclose(c.thrust_commanded, c.mu + fol.mass * case.g * E3, atol=1e-12)
    np.testing.assert_allclose(c.f_T * (euler_to_rotation(c.theta_r) @ E3), c.thrust_commanded, atol=1e-12)
    assert c.theta_r[2] == case.psi_ref
    np.testing.assert_allclose(c.x_e[:3], theta - c.theta_r, atol=1e-12)
    np.testing.assert_allclose(c.x_e[3:], euler_rate_matrix(theta) @ omega - c.theta_r_dot, atol=1e-12)
    assert np.all(np.abs(c.mu) < fol.beta) and np.all(np.abs(c.u) < fol.gamma)


def test_derivatives_match_plant_and_observer(case, fol):
    sysm = CascadeSystem.from_case_study(case, fol)
    x = sysm.initial_state([0.3, -0.2, 0.1], [0.2, 0.1, -0.1], [0.05, -0.1, 0.174], [0.1, 0.2, -0.1])
    assert x.shape == (STATE_DIM,)
    f_T, tau, cmd = 14.0, np.array([0.1, -0.2, 0.05]), np.array([1.0, -0.5, 12.0])
    d = sysm.derivatives(0.0, x, f_T, tau, cmd)
    np.testing.assert_allclose(d[:18], quadrotor_derivatives(x[:18], f_T, tau, sysm.params), atol=1e-14)
    e, v_ij = sysm.relative(0.0, x)
    actual = f_T * (x[6:15].reshape(3, 3) @ E3)
    np.testing.assert_allclose(d[18:], sysm.observer.derivatives(x[18:], e, v_ij / sysm.x_star, actual), atol=1e-14)
    sysm_c = CascadeSystem.from_case_study(case, fol, observer_thrust="commanded")
    np.testing.assert_allclose(sysm_c.derivatives(0.0, x, f_T, tau, cmd)[18:],
                               sysm.observer.derivatives(x[18:], e, v_ij / sysm.x_star, cmd), atol=1e-14)
    with pytest.raises(ValueError):
        CascadeSystem.from_case_study(case, fol, observer_thrust="other")


def test_initial_state_from_config(case, fol):
    sysm = CascadeSystem.from_case_study(case, fol)
    x0 = sysm.initial_state_from_config(fol)
    np.testing.assert_allclose(sysm.relative(0.0, x0)[0], [-0.475, -0.425, 0.0], atol=1e-12)   # Phase 1.2 hand value
    theta, omega = sysm.attitude(x0)
    np.testing.assert_allclose(theta, fol.euler0, atol=1e-12)
    np.testing.assert_allclose(omega, 0.0)
    np.testing.assert_allclose(x0[18:21], sysm.relative(0.0, x0)[0])                           # A19: e_hat(0) = e(0)


def test_formation_equilibrium_is_held(case, fol):
    """At e = 0, v_ij = 0, level attitude and a converged observer, the cascade commands hover and stays there."""
    sysm = CascadeSystem.from_case_study(case, fol)
    x0 = sysm.initial_state(np.zeros(3), np.zeros(3), np.zeros(3))
    tr = simulate_cascade(sysm, make_controller(case, fol), x0, 0.5, H)
    assert np.abs(tr.e).max() < 1e-9 and np.abs(tr.x_e).max() < 1e-9
    np.testing.assert_allclose(tr.f_T, fol.mass * case.g, rtol=1e-9)
    assert np.abs(tr.mu).max() < 1e-8 and np.abs(tr.u).max() < 1e-8


@pytest.mark.parametrize("mode", ["actual", "commanded"])
def test_closed_loop_from_paper_initial_conditions(case, fol, mode):
    """10 s from the Sec. VI initial conditions with LQR-limit policies: inputs bounded, depth positive, the yaw
    error of Fig. 8 decays, the observer converges and the feature error shrinks."""
    sysm = CascadeSystem.from_case_study(case, fol, observer_thrust=mode)
    tr = simulate_cascade(sysm, make_controller(case, fol), sysm.initial_state_from_config(fol), 10.0, H)
    assert np.abs(tr.mu).max() < fol.beta and np.abs(tr.u).max() < fol.gamma
    assert tr.e[:, 0].min() > -1.0                                        # sphere stays in front of the camera
    assert tr.x_e[0, 2] == pytest.approx(0.174, abs=1e-3)                 # Fig. 8: e_psi(0) = psi(0)
    assert np.abs(tr.x_e[-1, :3]).max() < 2e-3
    assert np.linalg.norm(tr.v_tilde[-1]) < 1e-2
    assert np.linalg.norm(tr.e[-1]) < 0.15 * np.linalg.norm(tr.e[0])


def test_finite_difference_e_dot_source(case, fol):
    """D15 check in the cascade: with e sampled per step and e_dot from the filtered difference (tau = 0.02 s) the
    loop still converges; the velocity-estimate error stays small (filter lag only, no leader acceleration)."""
    sysm = CascadeSystem.from_case_study(case, fol)
    x0 = sysm.initial_state_from_config(fol)
    fd = simulate_cascade(sysm, make_controller(case, fol), x0, 10.0, H, e_dot_source="finite_difference")
    ex = simulate_cascade(sysm, make_controller(case, fol), x0, 10.0, H)
    assert np.abs(fd.mu).max() < fol.beta and np.abs(fd.u).max() < fol.gamma
    assert np.linalg.norm(fd.v_tilde[-1]) < 2e-2
    assert np.abs(fd.e - ex.e).max() < 0.05                                  # close to the exact-e_dot run
    assert np.linalg.norm(fd.e[-1]) < 0.15 * np.linalg.norm(fd.e[0])
    with pytest.raises(ValueError):
        simulate_cascade(sysm, make_controller(case, fol), x0, 0.01, H, e_dot_source="other")
