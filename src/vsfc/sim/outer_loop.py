"""Outer-loop (visual-servoing) simulation for Phase 1.5: data collection for Alg. 3 and closed-loop runs.

System (docs/equation_map.md §5.8):
- [Paper] leader: kinematic, Sec. VI (`sim/leaders.py`), sphere offset h_i.
- [Paper] follower: translational part of eq. (1), p_dot = v, v_dot = g e3 - R f_T e3 / m.
- [Eng, D24] ideal attitude: the physical thrust vector R f_T e3 is applied directly (no attitude dynamics yet).
- [Eng, D14] gravity compensation: R f_T e3 = mu + m g e3, so the learned input mu is the deviation from hover.
- [Paper] observer eq. (6) (`observer.py`), fed e, e_dot (exact, D15) and the physical thrust vector (D20).
- [Paper] learning state delta = [e_ij, v_hat_ij] (eq. (10)).

Combined state [Eng]: x = [p_j (3), v_j (3), z (7)] with z = [e_hat, v_hat, zeta_hat], integrated with RK4 and
a zero-order-hold input (`rl/data.rollout`, A18).

Data for Alg. 3 [Eng, D26]: behaviour mu = beta tanh((-K0 delta + exploration(t)) / beta); each run drops its
first `burn_in` seconds, while v_hat is still converging (eq. (10)'s e_dot = v_hat/x* does not hold yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from vsfc.config import CaseStudy, FollowerConfig, OuterDataConfig
from vsfc.dynamics import E3
from vsfc.leader_follower import relative_state
from vsfc.observer import AdaptiveVelocityObserver, pack_observer_state, unpack_observer_state
from vsfc.rl.data import OffPolicyDataset, rollout, sum_of_sinusoids
from vsfc.sim.leaders import ConstantVelocityLeader

STATE_DIM = 13
Policy = Callable[[float, np.ndarray], np.ndarray]     # (t, delta (6,)) -> mu (3,)


@dataclass(frozen=True)
class OuterLoopSystem:
    """One leader-follower pair (i, j) with ideal attitude (D24) and gravity compensation (D14)."""

    leader: object                 # has state(t) -> (p, v, a), inertial (3,) each
    h_i: np.ndarray                # (3,) m, sphere offset on the leader
    mass: float                    # m_j, kg
    g: float                       # m/s^2
    x_star: float                  # m
    s_star: np.ndarray             # (3,)
    observer: AdaptiveVelocityObserver
    beta: float                    # bound on |mu_k| (A2), N

    @classmethod
    def from_case_study(cls, case: CaseStudy, follower: FollowerConfig, leader=None) -> "OuterLoopSystem":
        leader = leader or ConstantVelocityLeader.from_config(case.leader)
        return cls(leader=leader, h_i=np.asarray(case.leader.sphere_offset, float), mass=follower.mass, g=case.g,
                  x_star=follower.x_star, s_star=follower.s_star,
                  observer=AdaptiveVelocityObserver.from_config(follower, case.g), beta=follower.beta)

    # ---- state helpers ----
    def relative(self, t: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """True (e_ij (3,), v_ij (3,) m/s) at time t."""
        p_i, v_i, _ = self.leader.state(t)
        return relative_state(p_i, v_i, self.h_i, x[0:3], x[3:6], self.x_star, self.s_star)

    def delta(self, t: float, x: np.ndarray) -> np.ndarray:
        """Learning state delta = [e_ij, v_hat_ij] (eq. (10)), (6,). e is measured, v_hat from the observer."""
        e, _ = self.relative(t, x)
        _, v_hat, _ = unpack_observer_state(x[6:13])
        return np.concatenate([e, v_hat])

    def thrust_vector(self, mu: np.ndarray) -> np.ndarray:
        """D14: physical R f_T e3 = mu + m g e3, (3,) N."""
        return np.asarray(mu, float) + self.mass * self.g * E3

    def initial_state(self, e0: np.ndarray, v_ij0: np.ndarray, t0: float = 0.0) -> np.ndarray:
        """Follower placed so that e(t0) = e0 and v_ij(t0) = v_ij0; observer init per A19
        (e_hat = e0, v_hat = 0, zeta_hat = 0). Returns x (13,)."""
        p_i, v_i, _ = self.leader.state(t0)
        p_j = p_i + self.h_i - self.x_star * (self.s_star + np.asarray(e0, float))
        v_j = v_i - np.asarray(v_ij0, float)
        z0 = pack_observer_state(np.asarray(e0, float), np.zeros(3), 0.0)
        return np.concatenate([p_j, v_j, z0])

    def initial_state_from_config(self, follower: FollowerConfig) -> np.ndarray:
        """Paper Sec. VI initial condition: p_j(0) = p0, v_j(0) = v0 (A19); observer per A19."""
        p_i, v_i, _ = self.leader.state(0.0)
        e0, _ = relative_state(p_i, v_i, self.h_i, follower.p0, follower.v0, self.x_star, self.s_star)
        z0 = pack_observer_state(e0, np.zeros(3), 0.0)
        return np.concatenate([follower.p0, follower.v0, z0])

    # ---- dynamics ----
    def derivatives(self, t: float, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """x_dot for x = [p_j, v_j, z] under learned input mu (3,) N."""
        thrust = self.thrust_vector(mu)
        e, v_ij = self.relative(t, x)
        p_dot = x[3:6]
        v_dot = self.g * E3 - thrust / self.mass                     # eq. (1), translational, D24
        z_dot = self.observer.derivatives(x[6:13], e, v_ij / self.x_star, thrust)   # eq. (6), exact e_dot (D15)
        return np.concatenate([p_dot, v_dot, z_dot])


@dataclass
class Trajectory:
    t: np.ndarray        # (K+1,) s
    x: np.ndarray        # (K+1, 13)
    delta: np.ndarray    # (K+1, 6)
    mu: np.ndarray       # (K, 3) applied on [t_i, t_{i+1})
    e_true: np.ndarray   # (K+1, 3)
    v_true: np.ndarray   # (K+1, 3) true v_ij, m/s


def simulate(system: OuterLoopSystem, policy: Policy, x0: np.ndarray, T: float, h: float) -> Trajectory:
    """RK4 with ZOH input mu_i = policy(t_i, delta(t_i, x_i)) (A18)."""
    t, X, _, U = rollout(system.derivatives, x0, lambda tt, xx: policy(tt, system.delta(tt, xx)), T, h)
    delta = np.array([system.delta(ti, xi) for ti, xi in zip(t, X)])
    rel = [system.relative(ti, xi) for ti, xi in zip(t, X)]
    return Trajectory(t, X, delta, U, np.array([r[0] for r in rel]), np.array([r[1] for r in rel]))


def behaviour_policy(K0: np.ndarray, beta: float, exploration: Callable[[float], np.ndarray] | None = None) -> Policy:
    """[Eng, D26] mu = beta tanh((-K0 delta + exploration(t)) / beta): admissible (saturated PD on a double
    integrator) and inside |mu_k| < beta. K0: (3, 6)."""
    K0 = np.asarray(K0, float)
    if exploration is None:
        return lambda t, d: beta * np.tanh(-(K0 @ d) / beta)
    return lambda t, d: beta * np.tanh((-(K0 @ d) + exploration(t)) / beta)


def per_axis_gain(k_e: float, k_v: float, n_axes: int = 3) -> np.ndarray:
    """(n_axes, 2 n_axes) gain acting on delta = [e, v_hat]: mu_k = -(k_e e_k + k_v v_hat_k)."""
    return np.hstack([k_e * np.eye(n_axes), k_v * np.eye(n_axes)])


# [Eng, D26] data settings: typed and validated in `vsfc.config` (YAML section `learning_outer.data`).
# Exploration e(t) = b + sum_q A_{q,k} sin(omega_q t + phi_{q,k}) per channel k: low frequencies move the state
# through the region; high frequencies ("dither") drive mu into saturation while barely moving e; the constant
# b (drawn per run, uniform in [bias_low, bias_high]) shifts the operating point by about b / k_e. A positive x
# bias keeps the depth error e_x > -1 (sphere in front of the camera).
DataSpec = OuterDataConfig


def collect_dataset(system: OuterLoopSystem, spec: DataSpec, h: float, dT: float):
    """Alg. 3 Step 1: run `spec.n_runs` behaviour-policy simulations from random initial relative states,
    drop each run's burn-in, cut into windows of dT. Returns (OffPolicyDataset, list[Trajectory])."""
    W = int(round(dT / h))
    burn = int(round(spec.burn_in / h))
    if burn % W:
        raise ValueError("burn_in must be a multiple of dT so windows stay on the dT grid")
    rng = np.random.default_rng(spec.seed)
    K0 = per_axis_gain(spec.k_e, spec.k_v)
    sets, trajs = [], []
    for r in range(spec.n_runs):
        e0 = rng.uniform(spec.e0_low, spec.e0_high)
        v0 = rng.uniform(-spec.v0_max, spec.v0_max, 3)
        sines = sum_of_sinusoids(np.asarray(spec.explore_amplitude, float), spec.explore_freqs, 3,
                                 seed=spec.seed * 1000 + r)
        bias = rng.uniform(spec.bias_low, spec.bias_high)
        expl = lambda t, sines=sines: bias + sines(t)
        tr = simulate(system, behaviour_policy(K0, system.beta, expl), system.initial_state(e0, v0),
                      spec.run_length, h)
        trajs.append(tr)
        sets.append(OffPolicyDataset.from_trajectory(tr.delta[burn:], tr.mu[burn:], h, W))
    return OffPolicyDataset.concatenate(sets), trajs
