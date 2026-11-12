"""Phase 1.5 Step 4-5: train the visual-servoing (outer-loop) controller with Alg. 3, eqs. (27)-(29).

Uses the data saved by `collect_outer_data.py` (results/phase1_5/data_f<j>.npz). Bases: D25 (config
`learning_outer.critic_degrees`, `actor_degrees`). Stopping test: ||w^(n+1) - w^(n)|| < tol_vs (Sec. VI, 1e-5).

Runs:
  - init "K0": actor initialised from the behaviour PD gain, xi^(0) = K0 delta / beta (admissible);
  - init "seed s": random ADMISSIBLE actor (D28, A28): per-axis linear gains k_e, k_v ~ U(init_gain_range) > 0
    (stabilising for a double integrator) and cubic weights N(0, init_cubic_std^2). The paper says only "randomly
    initialized"; a non-admissible start converges to a non-stabilising Riccati/HJB solution (A28).
Checks (Step 5): rank / condition of N_bar; linear actor coefficients vs the LQR limit K / beta and quadratic critic
weights vs the Riccati P (docs/equation_map.md §5.8); spread of the final weights across initialisations.
Saves results/phase1_5/weights_f<j>.npz (K0 run) and the iteration history.

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\train_outer_loop.py 2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_outer_data import data_path                      # noqa: E402
from vsfc.config import load_case_study                       # noqa: E402
from vsfc.rl.basis import axis_groups, grouped_basis, weights_from_quadratic_form  # noqa: E402
from vsfc.rl.data import OffPolicyDataset                      # noqa: E402
from vsfc.rl.offpolicy_irl import IRLProblem, train            # noqa: E402

OUT = ROOT / "results" / "phase1_5"
NAMES = ["e1", "e2", "e3", "v1", "v2", "v3"]


def build_problem(case, fol) -> IRLProblem:
    lo = case.learning_outer
    critic = grouped_basis(6, axis_groups(), lo.critic_degrees)
    actors = [grouped_basis(6, [g], lo.actor_degrees) for g in axis_groups()]
    return IRLProblem(critic, actors, fol.beta, np.diag(fol.R), fol.Q)


def lqr_reference(fol):
    """§5.8: LQR of e_dot = v_hat/x*, v_hat_dot = mu/m (the unconstrained, small-delta limit)."""
    A = np.zeros((6, 6)); A[:3, 3:] = np.eye(3) / fol.x_star
    B = np.zeros((6, 3)); B[3:] = np.eye(3) / fol.mass
    P = solve_continuous_are(A, B, fol.Q, fol.R)
    K = np.linalg.solve(fol.R, B.T @ P)
    return P, K


def initial_actor_from_gain(prob: IRLProblem, k_e: float, k_v: float) -> np.ndarray:
    """xi^(0)_k = (k_e e_k + k_v v_hat_k) / beta: linear monomials of channel k are its first two entries."""
    W = np.zeros((prob.m, prob.l_a))
    W[:, 0], W[:, 1] = k_e / prob.bound, k_v / prob.bound
    return W


def random_admissible_actor(prob: IRLProblem, gain_range, cubic_std: float, seed: int) -> np.ndarray:
    """[Eng, D28] W (m, l_a): linear part (k_e, k_v)/beta with k ~ U(gain_range) > 0, cubic part N(0, cubic_std^2)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, cubic_std, (prob.m, prob.l_a))
    W[:, :2] = rng.uniform(*gain_range, (prob.m, 2)) / prob.bound
    return W


def load_data(follower_id: int) -> OffPolicyDataset:
    z = np.load(data_path(follower_id))
    return OffPolicyDataset(delta=z["delta"], mu=z["mu"], h=float(z["h"]))


def main(follower_id: int) -> None:
    case = load_case_study()
    fol = next(f for f in case.followers if f.id == follower_id)
    lo = case.learning_outer
    prob = build_problem(case, fol)
    data = load_data(follower_id)
    P, K = lqr_reference(fol)
    print(f"follower {follower_id}: {data.n_windows} windows; l_c = {prob.critic.size}, l_a = {prob.l_a} x {prob.m}; "
          f"unknowns {prob.n_weights}; tol {fol.tol_vs}")

    inits = [("K0", initial_actor_from_gain(prob, lo.data.k_e, lo.data.k_v))]
    for s in lo.init_seeds:
        inits.append((f"seed {s}", random_admissible_actor(prob, lo.init_gain_range, lo.init_cubic_std, s)))

    results = {}
    for name, W0 in inits:
        try:
            res = train(data, prob, W0, tol=fol.tol_vs, max_iter=lo.max_iter)
        except np.linalg.LinAlgError as exc:
            print(f"  [{name}] FAILED: {exc}")
            continue
        results[name] = res
        h = res.history
        print(f"  [{name}] converged={res.converged} iterations={res.iterations} "
              f"rank={h[-1]['rank']}/{prob.n_weights} cond={h[-1]['cond']:.2e} rms={h[-1]['rms']:.3e}")
        print("      changes " + ", ".join(f"{x['change']:.2e}" for x in h))

    ref = results.get("K0")
    if ref is None:
        return
    # ---- Step 5 checks against the LQR limit ----
    K_ref = np.array([[K[k, k], K[k, k + 3]] for k in range(3)])
    K_lin = ref.W_a[:, :2] * prob.bound
    wP = weights_from_quadratic_form(prob.critic, P)
    quad = prob.critic.total_degrees == 2
    print("  LQR limit (linear actor coefficients x beta, per axis [k_e, k_v]):")
    for k in range(3):
        print(f"      axis {k + 1}: learned {np.round(K_lin[k], 3)}   LQR {np.round(K_ref[k], 3)}   "
              f"rel. err {np.linalg.norm(K_lin[k] - K_ref[k]) / np.linalg.norm(K_ref[k]):.2%}")
    print("  quadratic critic weights [e^2, e*v, v^2] per axis (LQR: P_ee, 2 P_ev, P_vv):")
    for k in range(3):
        idx = np.nonzero(quad)[0][3 * k:3 * k + 3]
        print(f"      axis {k + 1}: learned {np.round(ref.w_c[idx], 2)}   LQR {np.round(wP[idx], 2)}")
    print("  higher-order weights: critic quartic max |w| = "
          f"{np.abs(ref.w_c[~quad]).max():.3f}, actor cubic max |w| = {np.abs(ref.W_a[:, 2:]).max():.4f}")
    # ---- spread across initialisations ----
    w_ref = prob.join(ref.w_c, ref.W_a)
    for name, res in results.items():
        if name != "K0":
            w = prob.join(res.w_c, res.W_a)
            print(f"  [{name}] ||w - w_K0|| / ||w_K0|| = {np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref):.2e}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"weights_f{follower_id}.npz", w_c=ref.w_c, W_a=ref.W_a,
             history_w=np.array([x["w"] for x in ref.history]),
             history_change=np.array([x["change"] for x in ref.history]),
             critic_labels=np.array(prob.critic.labels(NAMES)),
             actor_labels=np.array([a.labels(NAMES) for a in prob.actors]))
    print(f"  saved {(OUT / f'weights_f{follower_id}.npz').relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
