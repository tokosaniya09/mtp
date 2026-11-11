"""Tests for eq. (5): src/vsfc/leader_follower.py."""

import numpy as np
import pytest

from vsfc import dynamics as dyn
from vsfc.config import load_case_study
from vsfc.leader_follower import relative_dynamics, relative_state
from vsfc.sim.integrators import integrate
from vsfc.sim.leaders import ConstantVelocityLeader, SinusoidalLeader
from vsfc.sim.observer_openloop import default_excitation


@pytest.fixture(scope="module")
def cs():
    return load_case_study()


def test_relative_state_worked_example_follower2_t0(cs):
    # Same hand example as Phase 1.2: e = [-0.475, -0.425, 0]; v_12 = v_1 - v_2 = [0.4, 0.5, 0.4] - 0.
    f2 = cs.followers[0]
    p1, v1, _ = ConstantVelocityLeader.from_config(cs.leader).state(0.0)
    e, v = relative_state(p1, v1, cs.leader.sphere_offset, f2.p0, f2.v0, f2.x_star, f2.s_star)
    np.testing.assert_allclose(e, [-0.475, -0.425, 0.0], atol=1e-15)
    np.testing.assert_allclose(v, [0.4, 0.5, 0.4])


def test_hover_thrust_cancels_gravity(cs):
    f = cs.followers[1]
    e_dot, v_dot = relative_dynamics(np.ones(3), f.mass * cs.g * dyn.E3, np.zeros(3), f.mass, cs.g, f.x_star)
    np.testing.assert_allclose(e_dot, np.ones(3) / f.x_star)
    np.testing.assert_allclose(v_dot, 0.0, atol=1e-15)


@pytest.mark.parametrize("idx", [0, 1])
def test_eq5_consistent_with_eq1(cs, idx):
    """Simulate the follower with eq. (1) and a moving, accelerating leader; build (e, v_ij) from the two
    trajectories; their central-difference derivatives must match eq. (5)."""
    f = cs.followers[idx]
    params = dyn.QuadrotorParams(m=f.mass, J=f.inertia, g=cs.g)
    exc = default_excitation(f.mass, cs.g)
    leader = SinusoidalLeader(cs.leader.p0, cs.leader.velocity, np.array([0.5, 0.5, 0.2]), 1.0)
    x0 = dyn.pack_state(f.p0, f.v0, dyn.euler_to_rotation(f.euler0), f.omega0)
    h = 1e-3
    t, X = integrate(lambda tt, x: dyn.quadrotor_derivatives(x, *exc(tt), params), x0, 0.0, 3.0, h,
                     post_step=dyn.normalize_state)

    E, V = np.empty((t.size, 3)), np.empty((t.size, 3))
    for k, (tk, xk) in enumerate(zip(t, X)):
        p_i, v_i, _ = leader.state(tk)
        p_j, v_j, _, _ = dyn.unpack_state(xk)
        E[k], V[k] = relative_state(p_i, v_i, cs.leader.sphere_offset, p_j, v_j, f.x_star, f.s_star)

    for k in range(1, t.size - 1, 97):
        f_T, _ = exc(t[k])
        _, _, a_i = leader.state(t[k])
        e_dot, v_dot = relative_dynamics(V[k], dyn.thrust_vector(X[k], f_T), a_i, f.mass, cs.g, f.x_star)
        np.testing.assert_allclose((E[k + 1] - E[k - 1]) / (2 * h), e_dot, atol=1e-7)
        np.testing.assert_allclose((V[k + 1] - V[k - 1]) / (2 * h), v_dot, atol=1e-5)
