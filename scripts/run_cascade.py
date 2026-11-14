"""Phase 1.6 Step 6: full cascade (observer + learned outer loop + thrust mapping + learned attitude loop) on eq. (1).

Needs results/phase1_5/weights_f<j>.npz and results/phase1_6/weights_att_A.npz.

Runs, 50 s each (paper's time axis), control evaluated every simulation step (1 ms, ZOH):
  (a) paper's stated initial conditions (Sec. VI), observer fed the ACTUAL thrust vector f_T R e3 (D20);
  (b) the same with the observer fed the COMMANDED vector mu + m g e3 (A26 hypothesis for Fig. 5's oscillation);
  (c) [Eng] large-error case e(0) = large_error_e0 (A27), follower at rest;
  (d) [Eng] near-saturation case: (c) with v_ij(0) = near_saturation_v0;
  (e), (f) [Eng] exploratory (A11): e(0) as PLOTTED in Fig. 7 (fig7_plotted_e0, outside the outer loop's training
      region), follower at rest, observer fed the actual / the commanded thrust vector.
Each is compared with the IDEAL-attitude run of Phase 1.5 (D24) from the same relative state: settling of ||e||,
cost int (delta^T Q delta + U(mu)) dt, input bounds, depth margin (attitude lag, A29).
Figures (plain diagnostics, results/phase1_6/): mu and u (cf. Fig. 4), observer errors (cf. Fig. 5), feature errors
(cf. Fig. 7), attitude tracking errors (cf. Fig. 8), relative position errors x* e in metres (cf. Fig. 9). Behaviour is compared, not values (A11, A27).

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\run_cascade.py 2
    .venv\\Scripts\\python.exe scripts\\run_cascade.py 3
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_outer_loop import build_problem                    # noqa: E402
from vsfc.config import load_case_study                       # noqa: E402
from vsfc.controllers.cascade import CascadeController        # noqa: E402
from vsfc.controllers.thrust_mapping import ReferenceGenerator  # noqa: E402
from vsfc.rl.offpolicy_irl import actor_xi, policy            # noqa: E402
from vsfc.rl.utility import utility_from_xi                   # noqa: E402
from vsfc.sim.cascade import CascadeSystem, simulate_cascade  # noqa: E402
from vsfc.sim.inner_loop import build_inner_problem           # noqa: E402
from vsfc.sim.outer_loop import OuterLoopSystem, simulate     # noqa: E402

OUT5, OUT = ROOT / "results" / "phase1_5", ROOT / "results" / "phase1_6"
T_END, T_PLOT = 50.0, 10.0
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]      # validated categorical slots 1-3 (light mode), as Phase 1.5
STYLES = ["-", "--", ":"]
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e6e5e0"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def first_time_below(t, y, level):
    """First time after which y stays below `level` (inf if never)."""
    above = np.nonzero(y >= level)[0]
    if above.size == 0:
        return 0.0
    return float(t[above[-1] + 1]) if above[-1] + 1 < t.size else float("inf")


def outer_cost(prob, W_a, fol, delta, h):
    """int (delta^T Q delta + U(mu)) dt along a run, U from xi (eq. (26))."""
    xi = actor_xi(prob, W_a, delta[:-1])
    q = np.einsum("ni,ij,nj->n", delta[:-1], fol.Q, delta[:-1])
    return float(h * (q + utility_from_xi(xi, fol.beta, np.diag(fol.R))).sum())


def report(name, tr, fol, cost, ideal=None):
    t, ne = tr.t, np.linalg.norm(tr.e, axis=1)
    vt, et = np.linalg.norm(tr.v_tilde, axis=1), np.linalg.norm(tr.e_tilde, axis=1)
    i3 = int(3.0 / (t[1] - t[0]))
    peaks = [i for i in range(1, i3) if vt[i] > vt[i - 1] and vt[i] >= vt[i + 1] and vt[i] > 0.02]
    print(f"  [{name}]")
    print(f"      ||e||: {ne[0]:.3f} -> < 0.05 after {first_time_below(t, ne, 0.05):.1f} s, < 0.01 after "
          f"{first_time_below(t, ne, 0.01):.1f} s, ||e(50)|| = {ne[-1]:.1e}; min e_x = {tr.e[:, 0].min():.2f}; "
          f"cost = {cost:.3f}")
    sat = (np.abs(tr.mu) > 0.99 * fol.beta).any(axis=1)
    print(f"      |mu_k| > 0.99 beta for {sat.sum() * (t[1] - t[0]):.2f} s; "
          f"max |mu_k| = {np.abs(tr.mu).max():.2f} N (beta {fol.beta:g}), max ||mu||_2 = "
          f"{np.linalg.norm(tr.mu, axis=1).max():.2f}; max |u_k| = {np.abs(tr.u).max():.3f} N m (gamma {fol.gamma:g}); "
          f"f_T in [{tr.f_T.min():.2f}, {tr.f_T.max():.2f}] N")
    ea = np.abs(tr.x_e[:, :3])
    print(f"      attitude: max |e_Theta| = {np.round(ea.max(0), 3)} rad, all < 0.01 rad after "
          f"{first_time_below(t[:-1], ea.max(1), 0.01):.1f} s; max |Theta_r| = {np.round(np.abs(tr.theta_r).max(0), 3)}")
    print(f"      observer: max ||v~|| = {vt.max():.3f}, < 1e-3 after {first_time_below(t, vt, 1e-3):.1f} s; "
          f"max ||e~|| = {et.max():.4f}; local maxima of ||v~|| in the first 3 s: "
          f"{[(round(float(t[i]), 2), round(float(vt[i]), 3)) for i in peaks] or 'none (monotone)'}")
    if ideal is not None:
        tri, ci = ideal
        ni = np.linalg.norm(tri.e_true, axis=1)
        print(f"      ideal attitude (1.5): < 0.05 after {first_time_below(tri.t, ni, 0.05):.1f} s, < 0.01 after "
              f"{first_time_below(tri.t, ni, 0.01):.1f} s; cost = {ci:.3f}; max |e_cascade - e_ideal| = "
              f"{np.abs(tr.e - tri.e_true).max():.4f}")


def fig_inputs(runs, fol, fid):
    fig, axes = plt.subplots(2, len(runs), figsize=(4.6 * len(runs), 5.2), squeeze=False)
    for c, (name, tr) in enumerate(runs):
        n = int(T_PLOT / (tr.t[1] - tr.t[0]))
        for row, (y, bound, lab, sym) in enumerate([(tr.mu, fol.beta, r"$\mu$ (N)", r"\mu"),
                                                    (tr.u, fol.gamma, r"$u$ (N m)", "u")]):
            ax = axes[row, c]
            for k in range(3):
                ax.plot(tr.t[:n], y[:n, k], color=COLORS[k], linestyle=STYLES[k], linewidth=1.4,
                        label=rf"${sym}_{{{'xyz'[k]}}}$")
            for b in (bound, -bound):
                ax.axhline(b, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
            ax.set_ylim(-1.15 * bound, 1.15 * bound)
            ax.set_ylabel(lab)
            ax.set_title(f"follower {fid}: {name}" if row == 0 else "attitude torque", fontsize=9, color=INK, loc="left")
            ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=3)
            _style(ax)
        axes[1, c].set_xlabel("time (s)")
    fig.tight_layout()
    path = OUT / f"fig4_inputs_f{fid}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_observer(runs, fid):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for j, (name, tr) in enumerate(runs):
        n = int(T_PLOT / (tr.t[1] - tr.t[0]))
        axes[0].plot(tr.t[:n], np.linalg.norm(tr.e_tilde[:n], axis=1), color=COLORS[j], linestyle=STYLES[j], label=name)
        axes[1].plot(tr.t[:n], np.linalg.norm(tr.v_tilde[:n], axis=1), color=COLORS[j], linestyle=STYLES[j], label=name)
    axes[0].set_ylabel(r"$\|\tilde e\|$")
    axes[1].set_ylabel(r"$\|\tilde v\|$ (m/s)")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title(f"follower {fid}: observer errors (cf. Fig. 5)", fontsize=9, color=INK, loc="left")
    axes[0].legend(fontsize=8, frameon=False)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    path = OUT / f"fig5_observer_f{fid}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_errors(runs, fid, key, ylab, fname, title, t_plot=T_PLOT):
    fig, axes = plt.subplots(len(runs), 1, figsize=(7, 2.4 * len(runs)), squeeze=False, sharex=True)
    for ax, (name, tr) in zip(axes[:, 0], runs):
        y = getattr(tr, key)
        n = min(int(t_plot / (tr.t[1] - tr.t[0])), y.shape[0])
        for k in range(3):
            ax.plot(tr.t[:n], y[:n, k], color=COLORS[k], linestyle=STYLES[k], linewidth=1.4, label=ylab[k])
        ax.set_title(f"follower {fid}: {name} ({title})", fontsize=9, color=INK, loc="left")
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
        _style(ax)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    path = OUT / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(follower_id: int) -> None:
    case = load_case_study()
    fol = next(f for f in case.followers if f.id == follower_id)
    lo, li, h = case.learning_outer, case.learning_inner, case.step
    po, zo = build_problem(case, fol), np.load(OUT5 / f"weights_f{follower_id}.npz")
    pi_, zi = build_inner_problem(fol, li.critic_degrees, li.actor_degrees), np.load(OUT / "weights_att_A.npz")
    outer = lambda d: policy(po, zo["W_a"], d)
    inner = lambda xe: policy(pi_, zi["W_a"], xe)
    controller = lambda: CascadeController(outer, inner, ReferenceGenerator.from_case_study(case, fol))
    sys_a = CascadeSystem.from_case_study(case, fol)
    sys_c = CascadeSystem.from_case_study(case, fol, observer_thrust="commanded")
    ideal_sys = OuterLoopSystem.from_case_study(case, fol)
    v_lead = sys_a.leader.state(0.0)[1]
    print(f"follower {follower_id}: cascade, {T_END:.0f} s, step {h} s")

    def ideal(x0):
        tr = simulate(ideal_sys, lambda t, d: outer(d), x0, T_END, h)
        return tr, outer_cost(po, zo["W_a"], fol, tr.delta, h)

    def cascade(system, x0):
        tr = simulate_cascade(system, controller(), x0, T_END, h)
        delta = np.hstack([tr.e, tr.v_hat])
        return tr, outer_cost(po, zo["W_a"], fol, delta, h)

    runs = {}
    specs = [("stated ICs", sys_a, sys_a.initial_state_from_config(fol), ideal_sys.initial_state_from_config(fol)),
             ("stated ICs, observer fed commanded thrust", sys_c, sys_c.initial_state_from_config(fol), None),
             ("large error", sys_a, sys_a.initial_state(lo.large_error_e0, v_lead, fol.euler0),
              ideal_sys.initial_state(lo.large_error_e0, v_lead)),
             ("near saturation", sys_a, sys_a.initial_state(lo.large_error_e0, lo.near_saturation_v0, fol.euler0),
              ideal_sys.initial_state(lo.large_error_e0, lo.near_saturation_v0)),
             ("Fig. 7 plotted e(0)", sys_a, sys_a.initial_state(lo.fig7_plotted_e0, v_lead, fol.euler0), None),
             ("Fig. 7 plotted e(0), observer fed commanded thrust", sys_c,
              sys_c.initial_state(lo.fig7_plotted_e0, v_lead, fol.euler0), None)]
    for name, system, x0, x0_ideal in specs:
        tr, cost = cascade(system, x0)
        runs[name] = tr
        report(name, tr, fol, cost, ideal(x0_ideal) if x0_ideal is not None else None)

    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        fig_inputs([("stated ICs", runs["stated ICs"]), ("near saturation", runs["near saturation"]),
                    ("Fig. 7 plotted e(0)", runs["Fig. 7 plotted e(0)"])], fol, follower_id),
        fig_observer([("stated ICs, actual thrust", runs["stated ICs"]),
                      ("Fig. 7 e(0), actual thrust", runs["Fig. 7 plotted e(0)"]),
                      ("Fig. 7 e(0), commanded thrust", runs["Fig. 7 plotted e(0), observer fed commanded thrust"])],
                     follower_id),
        fig_errors([("stated ICs", runs["stated ICs"]), ("near saturation", runs["near saturation"])], follower_id, "e",
                   [r"$e_x$", r"$e_y$", r"$e_z$"], f"fig7_e_cascade_f{follower_id}.png", "cf. Fig. 7"),
        fig_errors([("stated ICs", runs["stated ICs"]), ("near saturation", runs["near saturation"])], follower_id, "x_e",
                   [r"$e_\phi$", r"$e_\theta$", r"$e_\psi$"], f"fig8_attitude_cascade_f{follower_id}.png", "cf. Fig. 8"),
        fig_errors([("stated ICs, relative position error x* e in metres",
                     SimpleNamespace(t=runs["stated ICs"].t, d=fol.x_star * runs["stated ICs"].e))], follower_id, "d",
                   [r"$\Delta_x$", r"$\Delta_y$", r"$\Delta_z$"], f"fig9_relpos_cascade_f{follower_id}.png",
                   "cf. Fig. 9", t_plot=T_END),
    ]
    for p in paths:
        print(f"  saved {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
