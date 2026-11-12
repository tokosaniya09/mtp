"""Phase 1.5 Steps 5-6: validate the learned visual-servoing controller and compare with Figs. 2(a,b), 4(a,b).

Needs results/phase1_5/data_f<j>.npz and weights_f<j>.npz (collect_outer_data.py, train_outer_loop.py).

Checks:
  1. Held-out Bellman residual (29) on fresh data (new seed), relative to the training residual.
  2. Closed loop, 50 s (paper's time axis), ideal attitude (D24), observer (6) with A19 init, learned policy (25):
     (a) paper's stated ICs (Sec. VI); (b) [Eng] large-error case e(0) = large_error_e0 (A27), follower at rest;
     (c) [Eng] near-saturation case: (b) with v_ij(0) = near_saturation_v0.
     Reports |mu_k| < beta, saturation time, ||mu||_2 (A2), settling of ||e||, depth margin e_x > -1, and the
     accumulated cost int (delta^T Q delta + U(mu)) dt compared with the LQR-limit tanh policy
     mu = -beta tanh(K delta / beta) (§5.8): the learned constrained controller should not cost more.
  3. Critic V_hat(delta(t)) non-increasing after the observer burn-in (Lyapunov check of Thm 2 / eq. (32)).
Figures (plain diagnostics, results/phase1_5/): weights vs iteration (cf. Fig. 2a,b), mu vs time (cf. Fig. 4a,b),
e vs time (cf. Fig. 7). Behaviour is compared, not values (A11, A16, A27).

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\validate_outer_loop.py 2
    .venv\\Scripts\\python.exe scripts\\validate_outer_loop.py 3
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

from train_outer_loop import build_problem, initial_actor_from_gain, lqr_reference  # noqa: E402
from vsfc.config import load_case_study                       # noqa: E402
from vsfc.rl.offpolicy_irl import bellman_rows, critic_value, policy  # noqa: E402
from vsfc.rl.utility import utility_closed_form               # noqa: E402
from vsfc.sim.outer_loop import OuterLoopSystem, collect_dataset, simulate  # noqa: E402

OUT = ROOT / "results" / "phase1_5"
T_END = 50.0
# validated categorical slots 1-3 (dataviz reference palette, light mode); identity also carried by line style
AXIS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
AXIS_STYLES = ["-", "--", ":"]
AXIS_LABELS = ["x", "y", "z"]
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e6e5e0"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def run_cost(fol, delta, mu, h):
    """int_0^T (delta^T Q delta + U(mu)) dt (left Riemann sum on the ZOH grid), eqs. (10)-(13)."""
    r = np.diag(fol.R)
    q = np.einsum("ni,ij,nj->n", delta[:-1], fol.Q, delta[:-1])
    u = np.array([utility_closed_form(m, fol.beta, r) for m in mu])
    return float(h * (q + u).sum())


def metrics(name, tr, fol, h):
    e_norm = np.linalg.norm(tr.e_true, axis=1)
    above = lambda thr: tr.t[np.nonzero(e_norm > thr)[0][-1]] if np.any(e_norm > thr) else 0.0
    sat = (np.abs(tr.mu) > 0.99 * fol.beta).any(axis=1)
    return dict(
        name=name, max_mu=float(np.abs(tr.mu).max()), max_mu2=float(np.linalg.norm(tr.mu, axis=1).max()),
        sat_time=float(sat.sum() * h), last_sat=float(tr.t[:-1][sat][-1]) if sat.any() else 0.0,
        t05=float(above(0.05)), t01=float(above(0.01)), e_end=float(e_norm[-1]),
        min_ex=float(tr.e_true[:, 0].min()), cost=run_cost(fol, tr.delta, tr.mu, h))


def print_metrics(m, beta):
    print(f"  [{m['name']}] max|mu_k| = {m['max_mu']:.3f} N (< beta = {beta}); max||mu||_2 = {m['max_mu2']:.3f}; "
          f"|mu_k| > 0.99 beta for {m['sat_time']:.2f} s (last at {m['last_sat']:.2f} s)")
    print(f"      ||e|| < 0.05 after {m['t05']:.1f} s, < 0.01 after {m['t01']:.1f} s, ||e(50)|| = {m['e_end']:.1e}; "
          f"min e_x = {m['min_ex']:.2f}; cost = {m['cost']:.2f}")


def weight_history(case, fol, prob, z):
    """Iteration 0 = K0 initial actor and zero critic (as trained), then each iterate (cf. Fig. 2)."""
    lo = case.learning_outer
    w0 = prob.join(np.zeros(prob.critic.size), initial_actor_from_gain(prob, lo.data.k_e, lo.data.k_v))
    return np.vstack([w0, z["history_w"]])


def figure_weights(hist, prob, fid):
    lc = prob.critic.size
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    it = np.arange(hist.shape[0])
    for ax, block, ylab, title in [(axes[0], hist[:, :lc], r"critic weights $\varpi_c$", "(a) critic"),
                                   (axes[1], hist[:, lc:], r"actor weights $\varpi_a$", "(b) actor")]:
        for k in range(block.shape[1]):
            ax.plot(it, block[:, k], color=AXIS_COLORS[0], linewidth=1.2, marker="o", markersize=3, alpha=0.75)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylab)
        ax.set_title(f"{title}: {block.shape[1]} weights, follower {fid}", fontsize=9, color=INK, loc="left")
        ax.set_xticks(it)
        _style(ax)
    fig.tight_layout()
    path = OUT / f"fig2ab_weights_f{fid}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_mu(runs, fol, fid):
    fig, axes = plt.subplots(len(runs), 1, figsize=(7, 2.6 * len(runs)), squeeze=False)
    for ax, (name, tr) in zip(axes[:, 0], runs):
        t = tr.t[:-1]
        for k in range(3):
            ax.plot(t, tr.mu[:, k], color=AXIS_COLORS[k], linestyle=AXIS_STYLES[k], linewidth=1.6,
                    label=rf"$\mu_{{{AXIS_LABELS[k]}}}$")
        for b in (fol.beta, -fol.beta):
            ax.axhline(b, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.text(T_END, fol.beta, r"  $\beta$", color=MUTED, va="center", fontsize=8)
        ax.set_xlim(0, T_END)
        ax.set_ylim(-1.15 * fol.beta, 1.15 * fol.beta)
        ax.set_ylabel(r"$\mu_%d$ (N)" % fid)
        ax.set_title(f"follower {fid}: {name}", fontsize=9, color=INK, loc="left")
        _style(ax)
        ins = ax.inset_axes([0.45, 0.52, 0.5, 0.42])
        n3 = int(3.0 / (t[1] - t[0]))
        for k in range(3):
            ins.plot(t[:n3], tr.mu[:n3, k], color=AXIS_COLORS[k], linestyle=AXIS_STYLES[k], linewidth=1.2)
        ins.set_xlim(0, 3)
        ins.set_ylim(-1.1 * fol.beta, 1.1 * fol.beta)
        ins.set_title("first 3 s", fontsize=7, color=MUTED)
        _style(ins)
        ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=3)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    path = OUT / f"fig4ab_mu_f{fid}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_e(runs, fid):
    fig, axes = plt.subplots(len(runs), 1, figsize=(7, 2.3 * len(runs)), squeeze=False)
    for ax, (name, tr) in zip(axes[:, 0], runs):
        for k in range(3):
            ax.plot(tr.t, tr.e_true[:, k], color=AXIS_COLORS[k], linestyle=AXIS_STYLES[k], linewidth=1.6,
                    label=rf"$e_{{{AXIS_LABELS[k]}}}$")
        ax.set_xlim(0, T_END)
        ax.set_ylabel("feature error e (-)")
        ax.set_title(f"follower {fid}: {name}", fontsize=9, color=INK, loc="left")
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
        _style(ax)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    path = OUT / f"fig7_e_f{fid}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(follower_id: int) -> None:
    case = load_case_study()
    fol = next(f for f in case.followers if f.id == follower_id)
    lo = case.learning_outer
    h = case.step
    prob = build_problem(case, fol)
    z = np.load(OUT / f"weights_f{follower_id}.npz")
    w_c, W_a = z["w_c"], z["W_a"]
    system = OuterLoopSystem.from_case_study(case, fol)
    print(f"follower {follower_id}")

    # 1. held-out Bellman residual
    spec = dataclasses.replace(lo.data, n_runs=3, seed=lo.data.seed + 100)
    test, _ = collect_dataset(system, spec, h, case.dT)
    w = prob.join(w_c, W_a)
    N, T = bellman_rows(test, prob, W_a)
    print(f"  held-out Bellman residual (29): ||N w - T|| / ||T|| = {np.linalg.norm(N @ w - T) / np.linalg.norm(T):.2e} "
          f"on {test.n_windows} new windows (rms {np.sqrt(np.mean((N @ w - T) ** 2)):.2e})")

    # 2. closed loop
    learned = lambda t, d: policy(prob, W_a, d)
    _, K = lqr_reference(fol)
    lqr_tanh = lambda t, d: -fol.beta * np.tanh(K @ d / fol.beta)
    x_paper = system.initial_state_from_config(fol)
    x_large = system.initial_state(lo.large_error_e0, system.leader.state(0.0)[1])   # follower at rest
    x_sat = system.initial_state(lo.large_error_e0, lo.near_saturation_v0)
    sims = {}
    cases = [("stated ICs", x_paper), ("large error", x_large), ("near saturation", x_sat)]
    for name, pol, x0 in [(f"{c}, {p}", f, x) for c, x in cases for p, f in (("learned", learned), ("LQR-tanh", lqr_tanh))]:
        sims[name] = simulate(system, pol, x0, T_END, h)
        print_metrics(metrics(name, sims[name], fol, h), fol.beta)

    # 3. critic as a Lyapunov function after the burn-in
    for name in ("stated ICs, learned", "large error, learned", "near saturation, learned"):
        tr = sims[name]
        i0 = int(lo.data.burn_in / h)
        V = critic_value(prob, w_c, tr.delta[i0:])
        dV = np.diff(V)
        print(f"  [{name}] V_hat(delta) after {lo.data.burn_in:.0f} s: increases on {np.mean(dV > 1e-12):.1%} of steps; "
              f"V {V[0]:.3f} -> {V[-1]:.2e}; min V over all t = {critic_value(prob, w_c, tr.delta).min():.2e}")

    OUT.mkdir(parents=True, exist_ok=True)
    paths = [figure_weights(weight_history(case, fol, prob, z), prob, follower_id),
             figure_mu([("paper's stated initial conditions", sims["stated ICs, learned"]),
                        ("large-error case e(0) = [3.2, -1.45, 0] (A27)", sims["large error, learned"]),
                        ("near-saturation case: same e(0), v(0) = [2, -2, 2] m/s", sims["near saturation, learned"])],
                       fol, follower_id),
             figure_e([("paper's stated initial conditions", sims["stated ICs, learned"]),
                       ("large-error case (A27)", sims["large error, learned"]),
                       ("near-saturation case", sims["near saturation, learned"])], follower_id)]
    for p in paths:
        print(f"  saved {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
