"""Full closed loop for one leader-follower pair: plant (1) + observer (6) + cascade controller (Phase 1.6 Step 6).

- [Paper] leader: kinematic, Sec. VI (`sim/leaders.py`), sphere offset h_i.
- [Paper] follower: full eq. (1) (`dynamics.quadrotor_derivatives`), inputs (f_T, tau).
- [Paper] observer eq. (6) with exact e_dot (D15).
- [Eng] controller: `controllers/cascade.CascadeController`, evaluated once per step h and held (ZOH, A18), the
  same rate as Phase 1.5. (The attitude loop has poles near -140 rad/s, so it cannot be held over dT = 0.05 s.)
- [Eng] observer thrust input (D20, A26): "actual" = physical f_T R e3 (default); "commanded" = mu + m g e3,
  which differs from the actual vector while the attitude lags its reference.

Combined state [Eng]: x = [p (3), v (3), R (9, row-major), omega (3), z (7)], z = [e_hat, v_hat, zeta_hat].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vsfc.config import CaseStudy, FollowerConfig
from vsfc.controllers.cascade import CascadeController
from vsfc.dynamics import (E3, QuadrotorParams, euler_to_rotation, pack_state, project_to_so3,
                           quadrotor_derivatives, rotation_to_euler)
from vsfc.leader_follower import relative_state
from vsfc.observer import AdaptiveVelocityObserver, pack_observer_state, unpack_observer_state
from vsfc.sim.integrators import rk4_step
from vsfc.sim.leaders import ConstantVelocityLeader

STATE_DIM = 25
OBSERVER_THRUST = ("actual", "commanded")


@dataclass(frozen=True)
class CascadeSystem:
    leader: object                 # state(t) -> (p, v, a)
    h_i: np.ndarray                # (3,) m
    params: QuadrotorParams        # follower m, J, g
    x_star: float
    s_star: np.ndarray             # (3,)
    observer: AdaptiveVelocityObserver
    observer_thrust: str = "actual"

    def __post_init__(self):
        if self.observer_thrust not in OBSERVER_THRUST:
            raise ValueError(f"observer_thrust must be one of {OBSERVER_THRUST}")

    @classmethod
    def from_case_study(cls, case: CaseStudy, follower: FollowerConfig, leader=None,
                        observer_thrust: str = "actual") -> "CascadeSystem":
        return cls(leader=leader or ConstantVelocityLeader.from_config(case.leader),
                   h_i=np.asarray(case.leader.sphere_offset, float),
                   params=QuadrotorParams(m=follower.mass, J=follower.inertia, g=case.g),
                   x_star=follower.x_star, s_star=follower.s_star,
                   observer=AdaptiveVelocityObserver.from_config(follower, case.g), observer_thrust=observer_thrust)

    # ---- measurements ----
    def relative(self, t: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """True (e_ij, v_ij) at time t."""
        p_i, v_i, _ = self.leader.state(t)
        return relative_state(p_i, v_i, self.h_i, x[0:3], x[3:6], self.x_star, self.s_star)

    def delta(self, t: float, x: np.ndarray) -> np.ndarray:
        """delta = [e (measured), v_hat (observer)], (6,)."""
        e, _ = self.relative(t, x)
        return np.concatenate([e, unpack_observer_state(x[18:25])[1]])

    @staticmethod
    def attitude(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(Theta ZYX (3,), omega body (3,))."""
        return rotation_to_euler(x[6:15].reshape(3, 3)), x[15:18]

    # ---- initial states ----
    def _pack(self, p, v, euler, omega, e0) -> np.ndarray:
        z0 = pack_observer_state(np.asarray(e0, float), np.zeros(3), 0.0)           # A19
        return np.concatenate([pack_state(p, v, euler_to_rotation(euler), omega), z0])

    def initial_state_from_config(self, follower: FollowerConfig) -> np.ndarray:
        """Paper Sec. VI: p_j(0), Theta_j(0), v_j(0) = 0, omega_j(0) = 0 (A19); observer per A19."""
        p_i, v_i, _ = self.leader.state(0.0)
        e0, _ = relative_state(p_i, v_i, self.h_i, follower.p0, follower.v0, self.x_star, self.s_star)
        return self._pack(follower.p0, follower.v0, follower.euler0, follower.omega0, e0)

    def initial_state(self, e0, v_ij0, euler0, omega0=np.zeros(3), t0: float = 0.0) -> np.ndarray:
        """Follower placed so that e(t0) = e0, v_ij(t0) = v_ij0, with attitude euler0 and body rate omega0."""
        p_i, v_i, _ = self.leader.state(t0)
        p_j = p_i + self.h_i - self.x_star * (self.s_star + np.asarray(e0, float))
        return self._pack(p_j, v_i - np.asarray(v_ij0, float), euler0, omega0, e0)

    # ---- dynamics ----
    def derivatives(self, t: float, x: np.ndarray, f_T: float, tau: np.ndarray,
                    thrust_commanded: np.ndarray) -> np.ndarray:
        """x_dot. Plant: eq. (1) with (f_T, tau). Observer: eq. (6) with exact e_dot = v_ij / x* (D15) and the thrust
        vector selected by `observer_thrust`."""
        plant = quadrotor_derivatives(x[:18], f_T, tau, self.params)
        e, v_ij = self.relative(t, x)
        thrust = f_T * (x[6:15].reshape(3, 3) @ E3) if self.observer_thrust == "actual" else thrust_commanded
        z_dot = self.observer.derivatives(x[18:25], e, v_ij / self.x_star, thrust)
        return np.concatenate([plant, z_dot])

    @staticmethod
    def post_step(x: np.ndarray) -> np.ndarray:
        """Re-project R onto SO(3) (D17)."""
        x = x.copy()
        x[6:15] = project_to_so3(x[6:15].reshape(3, 3)).reshape(9)
        return x


@dataclass
class CascadeTrajectory:
    t: np.ndarray           # (K+1,)
    x: np.ndarray           # (K+1, 25)
    e: np.ndarray           # (K+1, 3) true feature error
    v_ij: np.ndarray        # (K+1, 3) true relative velocity, m/s
    e_hat: np.ndarray       # (K+1, 3)
    v_hat: np.ndarray       # (K+1, 3)
    theta: np.ndarray       # (K+1, 3) rad
    mu: np.ndarray          # (K, 3) N
    f_T: np.ndarray         # (K,) N
    u: np.ndarray           # (K, 3) N m
    theta_r: np.ndarray     # (K, 3) rad
    x_e: np.ndarray         # (K, 6) attitude tracking error (D29, D33)

    @property
    def e_tilde(self) -> np.ndarray:
        return self.e - self.e_hat

    @property
    def v_tilde(self) -> np.ndarray:
        return self.v_ij - self.v_hat


def simulate_cascade(system: CascadeSystem, controller: CascadeController, x0: np.ndarray, T: float,
                     h: float) -> CascadeTrajectory:
    """RK4 with the controller evaluated at each grid point and held over the step (ZOH)."""
    K = int(round(T / h))
    t = h * np.arange(K + 1)
    X = np.empty((K + 1, STATE_DIM))
    X[0] = x = np.asarray(x0, float).copy()
    cmds = []
    for i in range(K):
        theta, omega = system.attitude(x)
        c = controller.step(system.delta(t[i], x), theta, omega)
        thrust_cmd = c.thrust_commanded
        x = rk4_step(lambda tt, xx: system.derivatives(tt, xx, c.f_T, c.u, thrust_cmd), t[i], x, h,
                     system.post_step)
        X[i + 1] = x
        cmds.append(c)
    rel = [system.relative(ti, xi) for ti, xi in zip(t, X)]
    return CascadeTrajectory(
        t=t, x=X, e=np.array([r[0] for r in rel]), v_ij=np.array([r[1] for r in rel]),
        e_hat=X[:, 18:21], v_hat=X[:, 21:24], theta=np.array([rotation_to_euler(xi[6:15].reshape(3, 3)) for xi in X]),
        mu=np.array([c.mu for c in cmds]), f_T=np.array([c.f_T for c in cmds]), u=np.array([c.u for c in cmds]),
        theta_r=np.array([c.theta_r for c in cmds]), x_e=np.array([c.x_e for c in cmds]))
