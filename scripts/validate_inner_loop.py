"""Phase 1.6 Step 5: validate the learned attitude controller (stand-alone, before the cascade).

Needs results/phase1_6/data_att.npz and weights_att_{A,B}.npz (collect_inner_data.py, train_inner_loop.py).

Checks (exact rotational plant, eq. (1); learning state per D33):
  1. Held-out Bellman residual (39) on fresh data (new seed).
  2. Closed loop, 10 s, Theta_r = 0 unless stated, policies: learned A (main), learned B (check), LQR-limit tanh
     policy u = -gamma tanh(K x_e / gamma) (§6.1). Cases: (a) Fig. 8 yaw error e_psi(0) = 0.174 rad; (b) pitch
     transient e(0) = [0.02, 0.6, 0.174] rad (size of Fig. 8's pitch swing); (c) all axes 0.4 rad, rates 1 rad/s;
     (d) tracking a fresh sinusoidal reference from e(0) = 0. Reports |u_k| < gamma, convergence and the cost
     int (x_e^T Q x_e + U(u)) dt (U from xi, eq. (26), stable at saturation). The constrained optimum should not
     cost more than the LQR-limit policy.
  3. Critic V_hat(x_e(t)) non-increasing for basis A (Lyapunov property, Thm 2 analogue for Sec. V).
Figures (plain diagnostics, results/phase1_6/): weights vs iteration (cf. Fig. 2c,d); attitude errors (cf. Fig. 8,
stand-alone, not the cascade).

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\validate_inner_loop.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_inner_data import OUT, attitude_follower          # noqa: E402
from train_inner_loop import bases, weights_path               # noqa: E402
from vsfc.config import load_case_study                       # noqa: E402
from vsfc.rl.offpolicy_irl import actor_xi, bellman_rows, critic_value  # noqa: E402
from vsfc.rl.utility import policy_from_xi, utility_from_xi   # noqa: E402
from vsfc.sim.inner_loop import (AttitudeSystem, SinusoidalAttitudeReference, build_inner_problem,  # noqa: E402
                                 collect_attitude_dataset, error_cost_weight, initial_actor_from_gain,
                                 lqr_reference_inner, simulate_attitude)

T_END = 10.0
AXIS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # validated categorical slots 1-3 (light mode), as Phase 1.5
AXIS_STYLES = ["-", "--", ":"]
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e6e5e0"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def run(system, xi_fn, x0, fol, h):
    """Simulate with u = -gamma tanh(xi(x_e)); cost from xi (left Riemann sum on the ZOH grid)."""
    tr = simulate_attitude(system, lambda t, xe: policy_from_xi(xi_fn(xe), fol.gamma), x0, T_END, h)
    xi = xi_fn(tr.x_e[:-1])
    q = np.einsum("ni,ij,nj->n", tr.x_e[:-1], error_cost_weight(fol), tr.x_e[:-1])
    cost = float(h * (q + utility_from_xi(xi, fol.gamma, np.diag(fol.D))).sum())
    return tr, cost


def figure_weights(hists):
    fig, axes = plt.subplots(len(hists), 2, figsize=(9, 3.0 * len(hists)), squeeze=False)
    for row, (name, prob, hist) in enumerate(hists):
        lc = prob.critic.size
        it = np.arange(hist.shape[0])
        for ax, block, lab in [(axes[row, 0], hist[:, :lc], "critic"), (axes[row, 1], hist[:, lc:], "actor")]:
            for k in range(block.shape[1]):
                ax.plot(it, block[:, k], color=AXIS_COLORS[0], linewidth=1.2, marker="o", markersize=3, alpha=0.75)
            ax.set_xlabel("iteration")
            ax.set_ylabel(f"{lab} weights")
            ax.set_title(f"basis {name}: {block.shape[1]} {lab} weights (cf. Fig. 2{'c' if lab == 'critic' else 'd'})",
                         fontsize=9, color=INK, loc="left")
            if it.size <= 15:
                ax.set_xticks(it)
            _style(ax)
    fig.tight_layout()
    path = OUT / "fig2cd_weights_att.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_errors(trs):
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)
    names = [r"$e_\phi$", r"$e_\theta$", r"$e_\psi$"]
    for k, ax in enumerate(axes):
        for j, (label, tr) in enumerate(trs):
            ax.plot(tr.t, tr.x_e[:, k], color=AXIS_COLORS[j], linestyle=AXIS_STYLES[j], linewidth=1.5, label=label)
        ax.set_ylabel(names[k] + " (rad)")
        _style(ax)
    axes[0].set_title("stand-alone attitude loop, e(0) = [0.02, 0.6, 0.174] rad (cf. Fig. 8; not the cascade)",
                      fontsize=9, color=INK, loc="left")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    path = OUT / "fig8_attitude_standalone.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    case = load_case_study()
    fol = attitude_follower(case)
    li = case.learning_inner
    h = case.step
    _, K = lqr_reference_inner(fol)
    probs, weights = {}, {}
    for name, (cd, ad) in bases(case).items():
        probs[name] = build_inner_problem(fol, cd, ad)
        weights[name] = np.load(weights_path(name))

    # 1. held-out Bellman residual
    spec = dataclasses.replace(li.data, n_runs=3, seed=li.data.seed + 100)
    test, _ = collect_attitude_dataset(fol, spec, h, case.dT)
    for name, prob in probs.items():
        z = weights[name]
        N, T = bellman_rows(test, prob, z["W_a"])
        w = prob.join(z["w_c"], z["W_a"])
        print(f"basis {name}: held-out Bellman residual ||N w - T|| / ||T|| = "
              f"{np.linalg.norm(N @ w - T) / np.linalg.norm(T):.2e} on {test.n_windows} new windows")

    # 2. closed loop
    xi_fns = {f"learned {n}": (lambda p, W: (lambda xe: actor_xi(p, W, xe)))(probs[n], weights[n]["W_a"])
              for n in probs}
    xi_fns["LQR-tanh"] = lambda xe: np.asarray(xe) @ K.T / fol.gamma
    zero = AttitudeSystem(fol.inertia, SinusoidalAttitudeReference.zero())
    track = AttitudeSystem(fol.inertia, SinusoidalAttitudeReference.random(
        li.data.ref_amplitude, li.data.ref_freqs, np.random.default_rng(li.data.seed + 200)))
    cases = [("(a) Fig. 8 yaw", zero, [0.0, 0.0, 0.174], [0, 0, 0]),
             ("(b) pitch 0.6", zero, [0.02, 0.6, 0.174], [0, 0, 0]),
             ("(c) all 0.4, rate 1", zero, [0.4, -0.4, 0.4], [1.0, -1.0, 1.0]),
             ("(d) tracking", track, [0, 0, 0], [0, 0, 0])]
    sims = {}
    for cname, system, e0, w0 in cases:
        x0 = system.initial_state(np.array(e0, float), np.array(w0, float))
        print(f"  {cname}")
        for pname, xi_fn in xi_fns.items():
            tr, cost = run(system, xi_fn, x0, fol, h)
            sims[(cname, pname)] = tr
            conv = np.abs(tr.x_e[-1]).max()
            print(f"      {pname:10s} cost {cost:9.4f}  max|u_k| {np.abs(tr.u).max():.3f}  "
                  f"max|e_Theta| {np.abs(tr.x_e[:, :3]).max():.3f}  |x_e(10 s)|_max {conv:.1e}"
                  f"{'' if conv < 1e-2 else '  NOT CONVERGED'}")

    # 3. critic A as a Lyapunov function (regulation cases)
    pa, wa = probs["A"], weights["A"]
    for cname in ("(a) Fig. 8 yaw", "(b) pitch 0.6", "(c) all 0.4, rate 1"):
        V = critic_value(pa, wa["w_c"], sims[(cname, "learned A")].x_e)
        print(f"  basis A critic along {cname}: increases on {np.mean(np.diff(V) > 1e-12):.1%} of steps; "
              f"V {V[0]:.3f} -> {V[-1]:.2e}; min V {V.min():.2e}")

    OUT.mkdir(parents=True, exist_ok=True)
    hists = []
    for name, prob in probs.items():
        w0 = prob.join(np.zeros(prob.critic.size),
                       initial_actor_from_gain(prob, li.data.k_theta, li.data.k_omega))
        hists.append((name, prob, np.vstack([w0, weights[name]["history_w"]])))
    paths = [figure_weights(hists),
             figure_errors([(n, sims[("(b) pitch 0.6", n)]) for n in ("learned A", "LQR-tanh")])]
    for p in paths:
        print(f"  saved {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
