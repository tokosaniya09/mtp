"""Phase 1.7: proposed controller vs the ref. [16] baseline with input saturation (cf. Figs. 9 and 10).

Needs results/phase1_5/weights_f<j>.npz, results/phase1_6/weights_att_A.npz (proposed) and
results/phase1_7/baseline16_outer_f<j>.npz, baseline16_att.npz (train_baseline16.py).

Both controllers run in the SAME cascade (D34: plant (1), observer (6), thrust mapping D30, 1 ms control) from the
same states, 50 s:
  (a) paper's stated initial conditions; (b) [Eng] large error; (c) [Eng] near saturation;
  (d) [Eng] exploratory: e(0) as plotted in Fig. 7 (A11).
Baseline (D35): [16]'s optimal laws clipped component-wise to +-beta and +-gamma.
Reports settling of the relative position error x* e (m), input clipping time, peak values and the integral of
||x* e||^2. Figure: relative position errors of both controllers (cf. Figs. 9, 10).

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\run_baseline16.py 2
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_cascade import COLORS, INK, STYLES, _style, first_time_below  # noqa: E402
from train_outer_loop import build_problem                    # noqa: E402
from vsfc.config import load_case_study                       # noqa: E402
from vsfc.controllers.cascade import CascadeController        # noqa: E402
from vsfc.controllers.thrust_mapping import ReferenceGenerator  # noqa: E402
from vsfc.rl.offpolicy_irl import policy                      # noqa: E402
from vsfc.sim.baseline16 import attitude_game_problem, clipped_policy, outer_game_problem  # noqa: E402
from vsfc.sim.cascade import CascadeSystem, simulate_cascade  # noqa: E402
from vsfc.sim.inner_loop import build_inner_problem           # noqa: E402

OUT5, OUT6, OUT = (ROOT / "results" / p for p in ("phase1_5", "phase1_6", "phase1_7"))
T_END = 50.0


def report(name, tr, fol):
    t, h = tr.t, tr.t[1] - tr.t[0]
    d = fol.x_star * tr.e                                               # relative position error, m
    nd = np.linalg.norm(d, axis=1)
    clip_mu = (np.abs(tr.mu) >= fol.beta * (1 - 1e-9)).any(axis=1).sum() * h
    clip_u = (np.abs(tr.u) >= fol.gamma * (1 - 1e-9)).any(axis=1).sum() * h
    ea = np.abs(tr.x_e[:, :3])
    print(f"    {name:9s} ||x* e||: {nd[0]:.2f} m -> < 0.2 m after {first_time_below(t, nd, 0.2):5.1f} s, < 0.05 m after "
          f"{first_time_below(t, nd, 0.05):5.1f} s, at 50 s {nd[-1]:.1e} m; int ||x* e||^2 dt = {h * (nd ** 2).sum():8.2f}")
    print(f"              max |mu_k| {np.abs(tr.mu).max():.2f} N, at the bound for {clip_mu:5.2f} s; max |u_k| "
          f"{np.abs(tr.u).max():.3f} N m, at the bound for {clip_u:5.3f} s; max |e_Theta| {np.round(ea.max(0), 3)} rad; "
          f"min e_x {tr.e[:, 0].min():.2f}")
    return d


def figure(pairs, fid, fname):
    """pairs: list of (case name, {controller: (t, d)})."""
    fig, axes = plt.subplots(3, len(pairs), figsize=(6.2 * len(pairs), 6.6), squeeze=False, sharex=True)
    for c, (cname, runs) in enumerate(pairs):
        for k in range(3):
            ax = axes[k, c]
            for j, (label, (t, d)) in enumerate(runs.items()):
                ax.plot(t, d[:, k], color=COLORS[j], linestyle=STYLES[j], linewidth=1.4, label=label)
            ax.set_ylabel(rf"$\Delta_{'xyz'[k]}$ (m)")
            _style(ax)
        axes[0, c].set_title(f"follower {fid}: {cname}, relative position errors (cf. Figs. 9, 10)", fontsize=9,
                             color=INK, loc="left")
        axes[0, c].legend(fontsize=8, frameon=False, loc="upper right")
        axes[2, c].set_xlabel("time (s)")
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
    pi_, zi = build_inner_problem(fol, li.critic_degrees, li.actor_degrees), np.load(OUT6 / "weights_att_A.npz")
    bo, zbo = outer_game_problem(case, fol), np.load(OUT / f"baseline16_outer_f{follower_id}.npz")
    bi, zbi = attitude_game_problem(case, fol), np.load(OUT / "baseline16_att.npz")
    policies = {
        "proposed": (lambda d: policy(po, zo["W_a"], d), lambda xe: policy(pi_, zi["W_a"], xe)),
        "ref. [16]": (clipped_policy(bo, zbo["W_a"], fol.beta), clipped_policy(bi, zbi["W_a"], fol.gamma)),
    }
    system = CascadeSystem.from_case_study(case, fol)
    v_lead = system.leader.state(0.0)[1]
    cases = [("stated ICs", system.initial_state_from_config(fol)),
             ("large error", system.initial_state(lo.large_error_e0, v_lead, fol.euler0)),
             ("near saturation", system.initial_state(lo.large_error_e0, lo.near_saturation_v0, fol.euler0)),
             ("Fig. 7 plotted e(0)", system.initial_state(lo.fig7_plotted_e0, v_lead, fol.euler0))]
    print(f"follower {follower_id}: proposed vs ref. [16] baseline with input saturation, {T_END:.0f} s")
    results = {}
    for cname, x0 in cases:
        print(f"  [{cname}]")
        results[cname] = {}
        for pname, (outer, inner) in policies.items():
            ctrl = CascadeController(outer, inner, ReferenceGenerator.from_case_study(case, fol))
            try:
                tr = simulate_cascade(system, ctrl, x0, T_END, h)
            except ValueError as exc:                               # e.g. sphere behind the camera, thrust not upright
                print(f"    {pname:9s} FAILED: {exc}")
                continue
            results[cname][pname] = (tr.t, report(pname, tr, fol))
    OUT.mkdir(parents=True, exist_ok=True)
    path = figure([(c, results[c]) for c in ("stated ICs", "near saturation", "Fig. 7 plotted e(0)") if results[c]],
                  follower_id, f"fig9_10_relpos_f{follower_id}.png")
    print(f"  saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
