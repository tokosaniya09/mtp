"""Dual-loop (cascade) controller: visual servoing (Sec. IV) -> thrust mapping -> attitude control (Sec. V).

Plant-agnostic (working rule 10): takes measurements, returns the plant inputs of eq. (1).

One control step, sample period dt:
    [Paper] (25)  mu  = -beta tanh(xi_hat(delta)),  delta = [e_ij, v_hat_ij]          outer policy (callable)
    [Eng]  D14    F   = mu + m g e3                                                    physical thrust vector
    [Eng]  D30    f_T = ||F||, Theta_r from F / ||F|| (psi_r = 0), Theta_r_dot by the filtered difference (D20)
    [Eng]  D29/D33  x_e = [wrap(Theta - Theta_r); W(Theta) omega - Theta_r_dot]
    [Paper] (42)  u   = -gamma tanh(varsigma_hat(x_e))                                 attitude policy (callable)

The paper says only that the attitude controller "is integrated with the visual servoing controller to construct
a dual-loop control architecture" (Sec. V); the link is D30. The policies are passed as callables so that learned
weights, the LQR-limit policy or a baseline can be used without changing this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from vsfc.controllers.thrust_mapping import ReferenceGenerator, tracking_error_euler_rate
from vsfc.dynamics import E3, euler_to_rotation


@dataclass(frozen=True)
class CascadeCommand:
    """Output of one control step."""

    mu: np.ndarray           # (3,) N, learned thrust-vector deviation (D14)
    f_T: float               # N, total thrust, eq. (1)
    u: np.ndarray            # (3,) N m, body torque, eq. (1)
    theta_r: np.ndarray      # (3,) rad, ZYX reference attitude (D30)
    theta_r_dot: np.ndarray  # (3,) rad/s, filtered
    x_e: np.ndarray          # (6,) attitude learning state (D29, D33)

    @property
    def thrust_commanded(self) -> np.ndarray:
        """Commanded physical thrust vector f_T R(Theta_r) e3 = mu + m g e3, (3,) N."""
        return self.f_T * (euler_to_rotation(self.theta_r) @ E3)


class CascadeController:
    """outer(delta (6,)) -> mu (3,);  inner(x_e (6,)) -> u (3,);  reference: `ReferenceGenerator` (dt = sample period)."""

    def __init__(self, outer: Callable[[np.ndarray], np.ndarray], inner: Callable[[np.ndarray], np.ndarray],
                 reference: ReferenceGenerator):
        self.outer, self.inner, self.reference = outer, inner, reference
        self._started = False

    def step(self, delta: np.ndarray, theta: np.ndarray, omega: np.ndarray) -> CascadeCommand:
        """delta = [e, v_hat] (6,); theta: ZYX Euler angles (3,) rad; omega: body rate (3,) rad/s.
        Must be called once per sample period, in time order (the reference-rate filter is stateful)."""
        mu = np.asarray(self.outer(np.asarray(delta, float)), float)
        ref = self.reference.update(mu) if self._started else self.reference.reset(mu)
        self._started = True
        e_theta, e_omega = tracking_error_euler_rate(theta, omega, ref.theta_r, ref.theta_r_dot)
        x_e = np.concatenate([e_theta, e_omega])
        u = np.asarray(self.inner(x_e), float)
        return CascadeCommand(mu, ref.f_T, u, ref.theta_r, ref.theta_r_dot, x_e)
