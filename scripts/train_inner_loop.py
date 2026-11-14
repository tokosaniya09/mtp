"""Phase 1.6 Step 4: train the attitude controller with Alg. 4, eqs. (39)-(42).

Uses results/phase1_6/data_att.npz (collect_inner_data.py). Bases: D31/D33, main A (config `critic_degrees`,
`actor_degrees`) and check B (`check_*`). Stopping test ||w^(n+1) - w^(n)|| < tol_att (Sec. VI, 1e-5).

Runs per basis: init "K0" (behaviour gains, admissible) and random ADMISSIBLE starts (D28 analogue: per-axis
k_theta, k_omega ~ U(init_gain_range), higher-order actor weights 0).
Checks: rank / condition of N_bar; linear actor coefficients x gamma vs the LQR K of §6.1 (small-error limit);
quadratic critic weights vs Riccati P; spread across initialisations.
Saves results/phase1_6/weights_att_{A,B}.npz (K0 run, with the iteration history for Fig. 2c,d).

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\train_inner_loop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_inner_data import DATA, OUT, attitude_follower   # noqa: E402
from vsfc.config import load_case_study                       # noqa: E402
from vsfc.rl.basis import weights_from_quadratic_form          # noqa: E402
from vsfc.rl.data import OffPolicyDataset                      # noqa: E402
from vsfc.rl.offpolicy_irl import train                        # noqa: E402
from vsfc.sim.inner_loop import (ERROR_NAMES, build_inner_problem, initial_actor_from_gain,  # noqa: E402
                                 lqr_reference_inner, per_axis_lqr_gains)


def bases(case) -> dict:
    li = case.learning_inner
    return {"A": (li.critic_degrees, li.actor_degrees), "B": (li.check_critic_degrees, li.check_actor_degrees)}


def weights_path(name: str) -> Path:
    return OUT / f"weights_att_{name}.npz"


def random_admissible_actor(prob, gain_range, seed: int) -> np.ndarray:
    """Per-axis (k_theta, k_omega) / gamma with k ~ U(gain_range) > 0; higher-order actor weights 0 (D28)."""
    rng = np.random.default_rng(seed)
    W = np.zeros((prob.m, prob.l_a))
    W[:, :2] = rng.uniform(*gain_range, (prob.m, 2)) / prob.bound
    return W


def main() -> None:
    case = load_case_study()
    fol = attitude_follower(case)
    li = case.learning_inner
    z = np.load(DATA)
    data = OffPolicyDataset(delta=z["delta"], mu=z["mu"], h=float(z["h"]))
    P, K = lqr_reference_inner(fol)
    K_ref = per_axis_lqr_gains(K)
    for name, (cd, ad) in bases(case).items():
        prob = build_inner_problem(fol, cd, ad)
        print(f"basis {name} ({'main' if name == 'A' else 'check'}): critic deg {cd}, actor deg {ad}; "
              f"l_c = {prob.critic.size}, l_a = {prob.l_a} x {prob.m}; {data.n_windows} windows; tol {fol.tol_att}")
        inits = [("K0", initial_actor_from_gain(prob, li.data.k_theta, li.data.k_omega))]
        inits += [(f"seed {s}", random_admissible_actor(prob, li.init_gain_range, s)) for s in li.init_seeds]
        results = {}
        for iname, W0 in inits:
            try:
                res = train(data, prob, W0, tol=fol.tol_att, max_iter=li.max_iter)
            except np.linalg.LinAlgError as exc:
                print(f"  [{iname}] FAILED: {exc}")
                continue
            results[iname] = res
            h = res.history[-1]
            print(f"  [{iname}] converged={res.converged} iterations={res.iterations} rank={h['rank']}/"
                  f"{prob.n_weights} cond={h['cond']:.2e} rms={h['rms']:.3e}")
            if iname == "K0":
                print("      changes " + ", ".join(f"{x['change']:.2e}" for x in res.history))
        ref = results.get("K0")
        if ref is None:
            continue
        K_lin = ref.W_a[:, :2] * prob.bound
        print("  small-error limit: linear actor coefficients x gamma per axis [k_theta, k_omega] vs LQR (§6.1)")
        for k, ax in enumerate(("roll", "pitch", "yaw")):
            print(f"      {ax:5s}: learned {np.round(K_lin[k], 4)}   LQR {np.round(K_ref[k], 4)}   "
                  f"rel. err {np.linalg.norm(K_lin[k] - K_ref[k]) / np.linalg.norm(K_ref[k]):.1%}")
        wP = weights_from_quadratic_form(prob.critic, P)
        quad = prob.critic.total_degrees == 2
        print(f"  quadratic critic weights (per axis e_T^2, e_T*e_w, e_w^2): learned "
              f"{np.round(ref.w_c[quad], 3).tolist()}\n      LQR {np.round(wP[quad], 3).tolist()}")
        if (~quad).any():
            print(f"  higher-order weights: critic max |w| = {np.abs(ref.w_c[~quad]).max():.3f}, "
                  f"actor max |w| = {np.abs(ref.W_a[:, 2:]).max():.4f}")
        w_ref = prob.join(ref.w_c, ref.W_a)
        for iname, res in results.items():
            if iname != "K0":
                w = prob.join(res.w_c, res.W_a)
                print(f"  [{iname}] ||w - w_K0|| / ||w_K0|| = {np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref):.2e}")
        np.savez(weights_path(name), w_c=ref.w_c, W_a=ref.W_a, critic_degrees=cd, actor_degrees=ad,
                 history_w=np.array([x["w"] for x in ref.history]),
                 history_change=np.array([x["change"] for x in ref.history]),
                 critic_labels=np.array(prob.critic.labels(ERROR_NAMES)),
                 actor_labels=np.array([a.labels(ERROR_NAMES) for a in prob.actors]))
        print(f"  saved {weights_path(name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
