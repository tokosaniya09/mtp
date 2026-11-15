"""Phase 1.7: train the baseline of Fig. 10, the optimal controller of ref. [16] (D35), for both loops.

Outer loop (per follower): [16] (32) on data from the linear relative model with a recorded force disturbance;
checked against the discounted game Riccati solution (exact known answer).
Attitude loop (one problem for both followers): [16] (48) on the exact rotational plant with a recorded torque
disturbance; compared with the game Riccati solution of the linearised attitude error (small-error limit).
Saves results/phase1_7/baseline16_outer_f<j>.npz and baseline16_att.npz.

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\train_baseline16.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsfc.config import load_case_study                       # noqa: E402
from vsfc.rl.basis import weights_from_quadratic_form          # noqa: E402
from vsfc.rl.zero_sum import train_game                        # noqa: E402
from vsfc.sim.baseline16 import (attitude_game_problem, attitude_riccati, collect_attitude_game_data,  # noqa: E402
                                 collect_outer_game_data, outer_game_problem, outer_riccati)

OUT = ROOT / "results" / "phase1_7"


def initial_actor(prob, k1: float, k2: float) -> np.ndarray:
    """u^0_k = -(k1 x_k + k2 x_{k+3}): admissible PD start (linear monomials are the first two per channel)."""
    W = np.zeros((prob.m, prob.l_a))
    W[:, 0], W[:, 1] = -k1, -k2
    return W


def train_and_report(name, data, prob, W0, P, K, L, tol, max_iter, path):
    res = train_game(data, prob, W0, tol=tol, max_iter=max_iter)
    h = res.history[-1]
    print(f"{name}: {data.n_windows} windows, {prob.n_weights} unknowns, gamma^2 = {prob.gamma2:g}, alpha = "
          f"{prob.alpha:g}; converged={res.converged} iterations={res.iterations} rank={h['rank']} "
          f"cond={h['cond']:.2e} rms={h['rms']:.2e}")
    print("    changes " + ", ".join(f"{x['change']:.1e}" for x in res.history[:12]))
    K_ref = np.array([[K[k, k], K[k, k + 3]] for k in range(3)])
    L_ref = np.array([[L[k, k], L[k, k + 3]] for k in range(3)])
    K_l, L_l = -res.W_a[:, :2], res.W_d[:, :2]
    wP = weights_from_quadratic_form(prob.critic, P)
    quad = prob.critic.total_degrees == 2
    print(f"    control gain per axis: learned {np.round(K_l, 4).tolist()}")
    print(f"                           Riccati {np.round(K_ref, 4).tolist()}  max rel. err "
          f"{np.abs(K_l / K_ref - 1).max():.2%}")
    print(f"    worst-case disturbance gain: learned {np.round(L_l, 4).tolist()}, Riccati {np.round(L_ref, 4).tolist()}")
    print(f"    quadratic critic: rel. err {np.linalg.norm(res.w_c[quad] - wP[quad]) / np.linalg.norm(wP[quad]):.2e}")
    if (~quad).any():
        print(f"    higher-order |w| max: critic {np.abs(res.w_c[~quad]).max():.2e}, "
              f"control {np.abs(res.W_a[:, 2:]).max():.2e}")
    np.savez(path, w_c=res.w_c, W_a=res.W_a, W_d=res.W_d, history_w=np.array([x["w"] for x in res.history]))
    print(f"    saved {path.relative_to(ROOT)}")


def main() -> None:
    case = load_case_study()
    cfg, lo, li = case.baseline16, case.learning_outer, case.learning_inner
    OUT.mkdir(parents=True, exist_ok=True)
    for fol in case.followers:
        prob = outer_game_problem(case, fol)
        data = collect_outer_game_data(case, fol, lo.data.k_e, lo.data.k_v)
        P, K, L = outer_riccati(case, fol)
        train_and_report(f"outer loop, follower {fol.id}", data, prob, initial_actor(prob, lo.data.k_e, lo.data.k_v),
                         P, K, L, fol.tol_vs, cfg.max_iter, OUT / f"baseline16_outer_f{fol.id}.npz")
    fol = case.followers[0]
    prob = attitude_game_problem(case, fol)
    data = collect_attitude_game_data(case, fol)
    P, K, L = attitude_riccati(case, fol)
    train_and_report("attitude loop (both followers)", data, prob,
                     initial_actor(prob, li.data.k_theta, li.data.k_omega), P, K, L, fol.tol_att, cfg.max_iter,
                     OUT / "baseline16_att.npz")


if __name__ == "__main__":
    main()
