"""Baseline of the base paper's Fig. 10: "the optimal controller presented in [16] ... implemented in the considered
IBVS formation control scenario", with "a direct input saturation strategy" (base paper, Sec. VI).

[16] = Cheng et al., IEEE Trans. Cybern. 54(8), 2024 (see `rl/zero_sum.py`). The base paper gives no weights,
discount or attenuation level for this baseline and does not say how [16] was adapted. [Eng] D35:

- Structure: [16]'s cascade (optimal position law -> attitude conversion -> optimal attitude law) on OUR signals:
  outer state delta = [e, v_hat] from the base paper's observer (6) (replaces [16]'s containment observer, which
  needs inter-vehicle communication); attitude state x_e (D29, D33); attitude conversion = D30 ([16] (2), (40)
  is the same decomposition).
- Laws: [16] (20), (46), learned by [16] (32), (48) (`rl/zero_sum.train_game`), then clipped component-wise to
  +-beta and +-gamma (direct saturation).
- Weights: the base paper's Q, R (outer) and S (error block), D (attitude), as for the proposed controller;
  discount alpha = 0.06 from [16]; gamma^2 = ratio * R with [16]'s ratios (2 and 6), because [16]'s gamma^2 = 4, 6
  with R = 8 gives no solution (needs gamma^2 > R, see `test_zero_sum.py`).
- Disturbance channel: a force (outer) / torque (attitude) entering like the input, D = G, as in [16] where the
  equivalent disturbance enters the input channel of (14), (41).
- Outer-loop data: the learning model of §5.8, delta_dot = A delta + B (mu + d) (eq. (5) with D14 and a converged
  observer), with a recorded sinusoidal force d. Attitude data: exact rotational plant (D32 settings) with a
  recorded sinusoidal torque d.
"""

from __future__ import annotations

import numpy as np

from vsfc.config import CaseStudy, FollowerConfig
from vsfc.rl.basis import axis_groups, grouped_basis
from vsfc.rl.data import rollout, sum_of_sinusoids
from vsfc.rl.zero_sum import GameDataset, GameProblem, game_riccati
from vsfc.sim.inner_loop import (AttitudeSystem, SinusoidalAttitudeReference, error_cost_weight, linearised_attitude,
                                 per_axis_gain, simulate_attitude)


def _bases(critic_degrees, actor_degrees):
    critic = grouped_basis(6, axis_groups(), critic_degrees)
    acts = [grouped_basis(6, [g], actor_degrees) for g in axis_groups()]
    return critic, acts


# ---- outer loop ---------------------------------------------------------------------------------------------

def outer_model(fol: FollowerConfig) -> tuple[np.ndarray, np.ndarray]:
    """§5.8: delta_dot = A delta + B mu, A = [[0, I/x*], [0, 0]], B = [0; I/m]."""
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3) / fol.x_star
    B = np.zeros((6, 3))
    B[3:] = np.eye(3) / fol.mass
    return A, B


def outer_game_problem(case: CaseStudy, fol: FollowerConfig) -> GameProblem:
    cfg = case.baseline16
    critic, acts = _bases(cfg.critic_degrees, cfg.actor_degrees)
    r = np.diag(fol.R)
    return GameProblem(critic, acts, acts, r, fol.Q, cfg.gamma2_ratio_pos * float(r[0]), cfg.alpha)


def outer_riccati(case: CaseStudy, fol: FollowerConfig):
    """Known answer: (P, K, L) with u* = -K delta, Delta* = L delta."""
    A, B = outer_model(fol)
    g2 = case.baseline16.gamma2_ratio_pos * float(fol.R[0, 0])
    P = game_riccati(A, B, B, fol.Q, fol.R, g2, case.baseline16.alpha)
    return P, np.linalg.solve(fol.R, B.T @ P), B.T @ P / g2


def collect_outer_game_data(case: CaseStudy, fol: FollowerConfig, k_e: float, k_v: float) -> GameDataset:
    """Behaviour mu = -K0 delta + exploration; recorded force disturbance d; D = B."""
    spec, h, W = case.baseline16.outer_data, case.step, int(round(case.dT / case.step))
    A, B = outer_model(fol)
    K0 = np.hstack([k_e * np.eye(3), k_v * np.eye(3)])
    rng = np.random.default_rng(spec.seed)
    sets = []
    for s in range(spec.n_runs):
        ex = sum_of_sinusoids(spec.explore_amplitude, spec.explore_freqs, 3, seed=spec.seed * 1000 + s)
        dist = sum_of_sinusoids(spec.dist_amplitude, spec.dist_freqs, 3, seed=spec.seed * 2000 + s)
        beh = lambda t, x, ex=ex, dist=dist: np.concatenate([-K0 @ x + ex(t), dist(t)])
        _, _, xs, v = rollout(lambda t, x, v: A @ x + B @ (v[:3] + v[3:]),
                              rng.uniform(-spec.x0_max, spec.x0_max, 6), beh, spec.run_length, h)
        sets.append(GameDataset.from_trajectory(xs, v[:, :3], v[:, 3:], h, W))
    return GameDataset.concatenate(sets)


# ---- attitude loop ------------------------------------------------------------------------------------------

def attitude_game_problem(case: CaseStudy, fol: FollowerConfig) -> GameProblem:
    cfg = case.baseline16
    critic, acts = _bases(cfg.att_critic_degrees, cfg.att_actor_degrees)   # quadratic / linear (D35, E18)
    r = np.diag(fol.D)
    return GameProblem(critic, acts, acts, r, error_cost_weight(fol), cfg.gamma2_ratio_att * float(r[0]), cfg.alpha)


def attitude_riccati(case: CaseStudy, fol: FollowerConfig):
    """Known answer of the linearised attitude error (§6.1): (P, K, L)."""
    A, B = linearised_attitude(fol)
    g2 = case.baseline16.gamma2_ratio_att * float(fol.D[0, 0])
    P = game_riccati(A, B, B, error_cost_weight(fol), fol.D, g2, case.baseline16.alpha)
    return P, np.linalg.solve(fol.D, B.T @ P), B.T @ P / g2


def collect_attitude_game_data(case: CaseStudy, fol: FollowerConfig) -> GameDataset:
    """D32 data system with a linear behaviour u = -K0 x_e + exploration (no tanh: the baseline is unconstrained)
    and a recorded torque disturbance d added to the applied torque (D = G)."""
    cfg, spec, h, W = case.baseline16, case.learning_inner.data, case.step, int(round(case.dT / case.step))
    rng = np.random.default_rng(spec.seed)
    K0 = per_axis_gain(spec.k_theta, spec.k_omega)
    sets = []
    for r in range(spec.n_runs):
        ref = SinusoidalAttitudeReference.random(spec.ref_amplitude, spec.ref_freqs, rng)
        system = AttitudeSystem(fol.inertia, ref)
        e0 = rng.uniform(-1.0, 1.0, 3) * spec.e_theta0_max
        w0 = rng.uniform(-1.0, 1.0, 3) * spec.e_omega0_max
        ex = sum_of_sinusoids(spec.explore_amplitude, spec.explore_freqs, 3, seed=spec.seed * 1000 + r)
        dist = sum_of_sinusoids(cfg.att_dist_amplitude, cfg.att_dist_freqs, 3, seed=spec.seed * 3000 + r)
        us, ds = [], []

        def total(t, xe, ex=ex, dist=dist):
            u, d = -K0 @ xe + ex(t), dist(t)
            us.append(u)
            ds.append(d)
            return u + d

        tr = simulate_attitude(system, total, system.initial_state(e0, w0), spec.run_length, h)
        sets.append(GameDataset.from_trajectory(tr.x_e, np.array(us), np.array(ds), h, W))
    return GameDataset.concatenate(sets)


# ---- the baseline control laws ---------------------------------------------------------------------------------

def clipped_policy(prob: GameProblem, W_a: np.ndarray, bound: float):
    """[16] (38) control law with direct input saturation (base paper Sec. VI): clip(W_a sigma_a(x), +-bound)."""
    W_a = np.asarray(W_a, float)
    return lambda x: np.clip(prob.control(W_a, x), -bound, bound)
