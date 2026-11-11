"""Open-loop test harness for the adaptive velocity observer (Phase 1.3) [Eng].

The follower is the full plant of eq. (1), driven by a PRESCRIBED thrust/torque excitation (no controller
yet: the visual-servoing controller comes in Phase 1.4+). The leader is kinematic. The observer (6) runs
alongside, receiving e_ij, e_dot_ij and the follower's physical thrust vector R_j f_Tj e3.

Combined state [Eng]: x = [follower (18, dynamics.STATE_DIM), observer (7)], integrated with RK4 (D17).

e_dot source (D15):
- "exact": e and e_dot evaluated from simulator truth at every RK4 stage.
- "finite_difference": e is sampled once per step and held over the step (zero-order hold, like a camera
  frame); e_dot comes from `FilteredDifferentiator` on the sampled e.

Because the observer receives the same thrust vector the plant uses, the error dynamics (7) do not depend
on the excitation; the excitation only makes e and v_ij non-trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from vsfc import dynamics as dyn
from vsfc.camera import feature_rate
from vsfc.differentiator import FilteredDifferentiator
from vsfc.leader_follower import relative_state
from vsfc.observer import AdaptiveVelocityObserver, pack_observer_state
from vsfc.sim.integrators import rk4_step

NF = dyn.STATE_DIM
Excitation = Callable[[float], tuple[float, np.ndarray]]   # t -> (f_T [N], tau [N m] (3,))


def default_excitation(m: float, g: float) -> Excitation:
    """[Eng] near-hover excitation: f_T = m g (1 + 0.05 sin 1.3t), small zero-mean sinusoidal torques.

    Keeps tilt within a few degrees over tens of seconds so the leader stays in front of the camera.
    """
    def exc(t: float):
        f_T = m * g * (1.0 + 0.05 * np.sin(1.3 * t))
        tau = 1e-4 * np.array([np.sin(2.1 * t), np.cos(1.7 * t), 0.5 * np.sin(0.9 * t)])
        return f_T, tau
    return exc


@dataclass
class ObserverRun:
    """Time histories, each (N+1, ...)."""

    t: np.ndarray
    e: np.ndarray
    v_ij: np.ndarray
    e_hat: np.ndarray
    v_hat: np.ndarray
    zeta_hat: np.ndarray
    e_dot_used: np.ndarray       # e_dot fed to the observer (held value in FD mode)

    @property
    def e_tilde(self) -> np.ndarray:
        return self.e - self.e_hat

    @property
    def v_tilde(self) -> np.ndarray:
        return self.v_ij - self.v_hat


def simulate_observer_open_loop(follower, leader, h_i: np.ndarray, g: float, duration: float, h: float,
                                excitation: Excitation | None = None, e_dot_source: str = "exact",
                                fd_tau: float = 0.02, observer: AdaptiveVelocityObserver | None = None,
                                e_hat0: np.ndarray | None = None, v_hat0: np.ndarray | None = None,
                                zeta_hat0: float = 0.0) -> ObserverRun:
    """Run follower (eq. 1) + observer (eq. 6) for `duration` s with fixed step h.

    follower: vsfc.config.FollowerConfig; leader: object with state(t) -> (p, v, a).
    Observer initial state defaults (A19): e_hat(0) = e(0), v_hat(0) = 0, zeta_hat(0) = 0.
    """
    if e_dot_source not in ("exact", "finite_difference"):
        raise ValueError(f"unknown e_dot_source {e_dot_source!r}")
    params = dyn.QuadrotorParams(m=follower.mass, J=follower.inertia, g=g)
    obs = observer or AdaptiveVelocityObserver.from_config(follower, g)
    exc = excitation or default_excitation(follower.mass, g)
    x_star, s_star = follower.x_star, follower.s_star

    def truth(t, xf):
        p_i, v_i, _ = leader.state(t)
        p_j, v_j, _, _ = dyn.unpack_state(xf)
        return relative_state(p_i, v_i, h_i, p_j, v_j, x_star, s_star)

    xf0 = dyn.pack_state(follower.p0, follower.v0, dyn.euler_to_rotation(follower.euler0), follower.omega0)
    e0, v0 = truth(0.0, xf0)
    z0 = pack_observer_state(e0 if e_hat0 is None else e_hat0,
                             np.zeros(3) if v_hat0 is None else v_hat0, zeta_hat0)
    x = np.concatenate([xf0, z0])

    n = int(round(duration / h))
    t_hist = h * np.arange(n + 1)
    E, V, Eh, Vh, Ed = (np.empty((n + 1, 3)) for _ in range(5))
    Zh = np.empty(n + 1)

    held = {"e": e0, "e_dot": feature_rate(v0, x_star)}
    diff = None
    if e_dot_source == "finite_difference":
        diff = FilteredDifferentiator(h, fd_tau)
        held["e_dot"] = diff.reset(e0)   # zero initial derivative estimate

    def rhs_factory(f_T, tau):
        def rhs(t, xx):
            xf, z = xx[:NF], xx[NF:]
            if e_dot_source == "exact":
                e, v_ij = truth(t, xf)
                e_meas, e_dot = e, feature_rate(v_ij, x_star)
            else:
                e_meas, e_dot = held["e"], held["e_dot"]
            dxf = dyn.quadrotor_derivatives(xf, f_T, tau, params)
            dz = obs.derivatives(z, e_meas, e_dot, dyn.thrust_vector(xf, f_T))
            return np.concatenate([dxf, dz])
        return rhs

    def post(xx):
        xx = xx.copy()
        xx[:NF] = dyn.normalize_state(xx[:NF])
        return xx

    for k in range(n + 1):
        t = t_hist[k]
        e, v_ij = truth(t, x[:NF])
        if diff is not None and k > 0:
            held["e"], held["e_dot"] = e, diff.update(e)
        elif diff is None:
            held["e_dot"] = feature_rate(v_ij, x_star)
        E[k], V[k], Ed[k] = e, v_ij, held["e_dot"]
        Eh[k], Vh[k], Zh[k] = x[NF:NF + 3], x[NF + 3:NF + 6], x[NF + 6]
        if k == n:
            break
        f_T, tau = exc(t)                # zero-order hold of the input over the step
        x = rk4_step(rhs_factory(f_T, tau), t, x, h, post)

    return ObserverRun(t=t_hist, e=E, v_ij=V, e_hat=Eh, v_hat=Vh, zeta_hat=Zh, e_dot_used=Ed)
