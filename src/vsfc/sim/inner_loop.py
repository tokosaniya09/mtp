"""Attitude (inner) loop learning problem, paper Sec. V eqs. (33)-(42), Alg. 4, in D29 error coordinates.

Derivation: docs/equation_map.md §6.1. Phase 1.6 Step 2 sets up the problem; Step 3 collects the data (D32).

[Paper] (33): z = [Theta; omega; Theta_r; omega_r], cost z^T S z + U(u) with S = 10 I_12, D = 8 I_3, |u_k| < gamma = 1.
[Eng] D29: z = [e_Theta; e_omega; Theta_r; omega_r] with e_Theta = wrap(Theta - Theta_r).
[Eng] D33 (revises D29's rate error): e_omega = Theta_dot - Theta_r_dot = W(Theta) omega - Theta_r_dot (Euler-rate
    error), so e_Theta_dot = e_omega exactly; the body-rate error omega - omega_r coupled the axes (E13).
[Eng] D31 (A16): the learner works on the ERROR block only, x = [e_Theta; e_omega] (6,), with Q = the error block of S.
    The reference block of z^T S z is not affected by u, so it adds a policy-independent term to V and does not
    change (36) (§6.1, Consequence 1). The reference evolves with the outer loop (exogenous), so a critic in
    (Theta_r, omega_r) could not represent that term anyway.
[Eng] D31 (A16) bases, grouped per axis k in (e_Theta_k, e_omega_k) = (k, k + 3) (§6.1, Consequence 2), roles
    swapped by D33: main basis A: critic degrees (2, 4) (l_c = 24), actor degrees (1, 3) (l_a = 6), as the outer
    loop (D25); check basis B: critic degree 2 (l_c = 9), actor degree 1 (l_a = 2, 6 actor weights). B cannot fit
    the value where the policy saturates (A30).

Known answer (§6.1, Consequence 3): with W = I and d(t) neglected, each axis is the double integrator
e_Theta_dot = e_omega, e_omega_dot = u / J_k. Its LQR (Q = 10 I_2, R = 8) is the small-error limit of the
constrained optimum; the learned linear actor coefficients times gamma must approach K.

Data system for Alg. 4 Step 1 [Eng, D32] ("collect the data ... by using an admissible control u_j into the
augmented attitude system (37)"; chi_j is "an unknown smooth function"):
- [Paper] plant: rotational part of eq. (1), R_dot = R sk(omega), J omega_dot = -omega x J omega + u (thrust does not
  enter it). State x = [R (9, row-major), omega (3)]; RK4 + ZOH input on the 1 ms grid, R re-projected (D17).
- [Eng] reference: Theta_r(t) = sum of sinusoids (smooth, known in closed form), psi_r = 0 (D30); exact
  Theta_r_dot and omega_r = W(Theta_r)^-1 Theta_r_dot (same definition as D30, without the filter).
- [Eng] behaviour (admissible): u = gamma tanh((-K0 x_e + b(t)) / gamma), K0 = per-axis PD gains (k_theta, k_omega);
  b(t) = low-frequency sinusoids (move the error through the region) + high-frequency dither (drives u into
  saturation), as D27. No burn-in: no observer in this loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import solve_continuous_are

from vsfc.config import FollowerConfig
from vsfc.controllers.thrust_mapping import tracking_error_euler_rate
from vsfc.dynamics import euler_rate_matrix_inv, euler_to_rotation, project_to_so3, rotation_to_euler, skew
from vsfc.rl.basis import axis_groups, grouped_basis
from vsfc.rl.data import OffPolicyDataset, rollout, sum_of_sinusoids
from vsfc.rl.offpolicy_irl import IRLProblem

STATE_DIM = 12                                                  # [R (9), omega (3)]
TorquePolicy = Callable[[float, np.ndarray], np.ndarray]       # (t, x_e (6,)) -> u (3,) N m

ERROR_NAMES = ["eT1", "eT2", "eT3", "ew1", "ew2", "ew3"]     # (e_phi, e_theta, e_psi, e_omega_x, _y, _z)


def error_cost_weight(fol: FollowerConfig) -> np.ndarray:
    """[Eng, D31] Q (6, 6): error block [e_Theta; e_omega] of S (paper's S is 12 x 12, Sec. VI)."""
    return np.asarray(fol.S, float)[:6, :6]


def build_inner_problem(fol: FollowerConfig, critic_degrees, actor_degrees) -> IRLProblem:
    """IRLProblem for Alg. 4 on x = [e_Theta; e_omega]: per-axis critic, one actor basis per torque channel (24)/(41).

    bound = gamma, r = diag(D), Q = error block of S (D31)."""
    critic = grouped_basis(6, axis_groups(), critic_degrees)
    actors = [grouped_basis(6, [g], actor_degrees) for g in axis_groups()]
    return IRLProblem(critic, actors, fol.gamma, np.diag(fol.D), error_cost_weight(fol))


def linearised_attitude(fol: FollowerConfig) -> tuple[np.ndarray, np.ndarray]:
    """§6.1 Consequence 2: x_dot = A x + B u with A = [[0, I], [0, 0]], B = [0; J^-1] (W = I, d neglected)."""
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    B = np.zeros((6, 3))
    B[3:] = np.linalg.inv(fol.inertia)
    return A, B


def lqr_reference_inner(fol: FollowerConfig) -> tuple[np.ndarray, np.ndarray]:
    """§6.1 Consequence 3: Riccati P (6, 6) and gain K = D^-1 B^T P (3, 6) of the linearised attitude problem."""
    A, B = linearised_attitude(fol)
    P = solve_continuous_are(A, B, error_cost_weight(fol), fol.D)
    return P, np.linalg.solve(fol.D, B.T @ P)


def per_axis_lqr_gains(K: np.ndarray) -> np.ndarray:
    """(3, 2) array [k_Theta, k_omega] per axis from K (3, 6)."""
    return np.array([[K[k, k], K[k, k + 3]] for k in range(3)])


def initial_actor_from_gain(prob: IRLProblem, k_theta: float, k_omega: float) -> np.ndarray:
    """varsigma^(0)_k = (k_theta e_Theta_k + k_omega e_omega_k) / gamma; the linear monomials of channel k are the
    first two entries of its basis (grouped_basis order). Positive gains are admissible for a double integrator."""
    W = np.zeros((prob.m, prob.l_a))
    W[:, 0], W[:, 1] = k_theta / prob.bound, k_omega / prob.bound
    return W


# ---- data system (Alg. 4 Step 1), D32 ---------------------------------------------------------------------

@dataclass(frozen=True)
class SinusoidalAttitudeReference:
    """[Eng, D32] Theta_r(t) = sum_q A_q * sin(w_q t + phase_q) per Euler angle (rad); yaw column must be 0 (D30).

    amplitude (Q, 3) rad, freqs (Q,) rad/s, phase (Q, 3) rad. state(t) -> (Theta_r, Theta_r_dot, omega_r)."""

    amplitude: np.ndarray
    freqs: np.ndarray
    phase: np.ndarray

    @classmethod
    def random(cls, amplitude, freqs, rng: np.random.Generator) -> "SinusoidalAttitudeReference":
        A = np.asarray(amplitude, float)
        w = np.asarray(freqs, float)
        return cls(A, w, rng.uniform(0.0, 2.0 * np.pi, A.shape))

    @classmethod
    def zero(cls) -> "SinusoidalAttitudeReference":
        return cls(np.zeros((1, 3)), np.zeros(1), np.zeros((1, 3)))

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arg = self.freqs[:, None] * t + self.phase
        theta_r = (self.amplitude * np.sin(arg)).sum(axis=0)
        theta_r_dot = (self.amplitude * self.freqs[:, None] * np.cos(arg)).sum(axis=0)
        omega_r = euler_rate_matrix_inv(theta_r) @ theta_r_dot
        return theta_r, theta_r_dot, omega_r


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a x b for (3,) vectors (np.cross has a large per-call overhead at this size)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@dataclass(frozen=True)
class AttitudeSystem:
    """Rotational part of eq. (1) for follower j with a reference (D32). State x = [R (9), omega (3)]."""

    inertia: np.ndarray            # (3, 3) kg m^2
    reference: object              # state(t) -> (Theta_r, Theta_r_dot, omega_r)

    def __post_init__(self):
        object.__setattr__(self, "inertia", np.asarray(self.inertia, float))
        object.__setattr__(self, "_J_inv", np.linalg.inv(self.inertia))

    def derivatives(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """eq. (1): R_dot = R sk(omega), omega_dot = J^-1 (-omega x J omega + u)."""
        R, omega = x[:9].reshape(3, 3), x[9:]
        omega_dot = self._J_inv @ (u - _cross(omega, self.inertia @ omega))
        return np.concatenate([(R @ skew(omega)).reshape(9), omega_dot])

    @staticmethod
    def post_step(x: np.ndarray) -> np.ndarray:
        """Re-project R onto SO(3) (D17)."""
        x = x.copy()
        x[:9] = project_to_so3(x[:9].reshape(3, 3)).reshape(9)
        return x

    def error(self, t: float, x: np.ndarray) -> np.ndarray:
        """Learning state x_e = [e_Theta; e_omega] (D29 angle error, D33 Euler-rate error), (6,)."""
        theta_r, theta_r_dot, _ = self.reference.state(t)
        e_theta, e_omega = tracking_error_euler_rate(rotation_to_euler(x[:9].reshape(3, 3)), x[9:], theta_r,
                                                     theta_r_dot)
        return np.concatenate([e_theta, e_omega])

    def initial_state(self, e_theta0: np.ndarray, e_omega0: np.ndarray, t0: float = 0.0) -> np.ndarray:
        """x with Theta = Theta_r(t0) + e_theta0 and Theta_dot = Theta_r_dot(t0) + e_omega0 (D33)."""
        theta_r, theta_r_dot, _ = self.reference.state(t0)
        theta = theta_r + np.asarray(e_theta0, float)
        omega = euler_rate_matrix_inv(theta) @ (theta_r_dot + np.asarray(e_omega0, float))
        return np.concatenate([euler_to_rotation(theta).reshape(9), omega])


@dataclass
class AttitudeTrajectory:
    t: np.ndarray        # (K+1,) s
    x: np.ndarray        # (K+1, 12)
    x_e: np.ndarray      # (K+1, 6) [e_Theta, e_omega]
    u: np.ndarray        # (K, 3) N m, applied on [t_i, t_{i+1})
    theta_r: np.ndarray  # (K+1, 3) rad
    omega_r: np.ndarray  # (K+1, 3) rad/s


def simulate_attitude(system: AttitudeSystem, policy: TorquePolicy, x0: np.ndarray, T: float,
                      h: float) -> AttitudeTrajectory:
    """RK4 with ZOH torque u_i = policy(t_i, x_e(t_i, x_i)) on the h grid (A18). x_e is computed once per grid
    point (recorded when the policy is evaluated; the last point after the run)."""
    errors = []

    def act(tt, xx):
        xe = system.error(tt, xx)
        errors.append(xe)
        return policy(tt, xe)

    t, X, _, U = rollout(system.derivatives, x0, act, T, h, post_step=system.post_step)
    errors.append(system.error(t[-1], X[-1]))
    ref = [system.reference.state(ti) for ti in t]
    return AttitudeTrajectory(t, X, np.array(errors), U, np.array([r[0] for r in ref]),
                              np.array([r[2] for r in ref]))


def per_axis_gain(k_theta: float, k_omega: float) -> np.ndarray:
    """(3, 6) gain on x_e = [e_Theta, e_omega]: u_k = -(k_theta e_Theta_k + k_omega e_omega_k)."""
    return np.hstack([k_theta * np.eye(3), k_omega * np.eye(3)])


def behaviour_torque(K0: np.ndarray, gamma: float, exploration: Callable[[float], np.ndarray] | None = None
                     ) -> TorquePolicy:
    """[Eng, D32] u = gamma tanh((-K0 x_e + b(t)) / gamma): saturated PD (admissible), |u_k| < gamma."""
    K0 = np.asarray(K0, float)
    b = exploration or (lambda t: np.zeros(3))
    return lambda t, xe: gamma * np.tanh((-(K0 @ xe) + b(t)) / gamma)


def collect_attitude_dataset(fol: FollowerConfig, spec, h: float, dT: float, with_reference: bool = True):
    """Alg. 4 Step 1 [Eng, D32]: `spec.n_runs` behaviour runs from random initial errors, each with its own random
    reference (or Theta_r = 0 if with_reference is False), cut into windows of dT.
    spec: `vsfc.config.InnerDataConfig`. Returns (OffPolicyDataset, list[AttitudeTrajectory])."""
    W = int(round(dT / h))
    rng = np.random.default_rng(spec.seed)
    K0 = per_axis_gain(spec.k_theta, spec.k_omega)
    sets, trajs = [], []
    for r in range(spec.n_runs):
        ref = (SinusoidalAttitudeReference.random(spec.ref_amplitude, spec.ref_freqs, rng) if with_reference
               else SinusoidalAttitudeReference.zero())
        system = AttitudeSystem(fol.inertia, ref)
        e_theta0 = rng.uniform(-1.0, 1.0, 3) * spec.e_theta0_max
        e_omega0 = rng.uniform(-1.0, 1.0, 3) * spec.e_omega0_max
        sines = sum_of_sinusoids(spec.explore_amplitude, spec.explore_freqs, 3, seed=spec.seed * 1000 + r)
        tr = simulate_attitude(system, behaviour_torque(K0, fol.gamma, sines),
                               system.initial_state(e_theta0, e_omega0), spec.run_length, h)
        trajs.append(tr)
        sets.append(OffPolicyDataset.from_trajectory(tr.x_e, tr.u, h, W))
    return OffPolicyDataset.concatenate(sets), trajs
