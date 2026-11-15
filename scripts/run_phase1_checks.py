"""Phase 1.7 wrap-up: remaining checks of the reproduction, all in the full cascade (D34) with the learned policies.

  1. Reference: paper's stated initial conditions, both followers, 50 s, exact e_dot. Gives the 3-D trajectories of
     the leader and both followers (cf. Fig. 3). Each follower tracks the leader only (Sec. VI), so the two pairs
     are simulated separately and drawn together.
  2. A6: observer gain kappa_13. The paper prints kappa_12 = 5, kappa_13 = 55 (A6: possible typo for 5). Follower 3
     is re-run with kappa_13 = 5.
  3. D15: e_dot from a filtered finite difference of the sampled feature error (tau = 0.02 s, D20) instead of the
     exact value, both followers.
Runs 2 and 3 last 30 s (the transients are over by then). Figures: results/phase1_7/fig3_trajectories.png.

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\run_phase1_checks.py
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

from run_cascade import COLORS, INK, MUTED, first_time_below  # noqa: E402
from train_outer_loop import build_problem                    # noqa: E402
from vsfc.config import load_case_study                       # noqa: E402
from vsfc.controllers.cascade import CascadeController        # noqa: E402
from vsfc.controllers.thrust_mapping import ReferenceGenerator  # noqa: E402
from vsfc.rl.offpolicy_irl import policy                      # noqa: E402
from vsfc.sim.cascade import CascadeSystem, simulate_cascade  # noqa: E402
from vsfc.sim.inner_loop import build_inner_problem           # noqa: E402

OUT5, OUT6, OUT = (ROOT / "results" / p for p in ("phase1_5", "phase1_6", "phase1_7"))
T_REF, T_CHECK = 50.0, 30.0


def summary(name, tr, ref=None):
    t = tr.t
    ne, vt = np.linalg.norm(tr.e, axis=1), np.linalg.norm(tr.v_tilde, axis=1)
    i5 = int(5.0 / (t[1] - t[0]))
    line = (f"  [{name}] ||e|| < 0.05 after {first_time_below(t, ne, 0.05):.1f} s, < 0.01 after "
            f"{first_time_below(t, ne, 0.01):.1f} s; ||v~|| < 1e-2 after {first_time_below(t, vt, 1e-2):.2f} s, "
            f"< 1e-3 after {first_time_below(t, vt, 1e-3):.2f} s, max after 5 s {vt[i5:].max():.1e}; "
            f"max |mu_k| {np.abs(tr.mu).max():.2f}, max |u_k| {np.abs(tr.u).max():.3f}")
    if ref is not None:
        n = min(tr.e.shape[0], ref.e.shape[0])
        line += f"; max |e - e_ref| {np.abs(tr.e[:n] - ref.e[:n]).max():.4f}, max |mu - mu_ref| " \
                f"{np.abs(tr.mu[:n - 1] - ref.mu[:n - 1]).max():.3f} N"
    print(line)


def figure_trajectories(case, runs):
    """Leader and followers in the inertial frame (cf. Fig. 3); axes as in the paper (p_x, p_y, p_z)."""
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(projection="3d")
    t = next(iter(runs.values())).t
    lead = np.array([case.leader.position(ti) for ti in t])
    series = [("Quadrotor 1 (leader)", lead, "-")] + [(f"Quadrotor {fid}", tr.x[:, 0:3], s)
                                                      for (fid, tr), s in zip(runs.items(), ("--", ":"))]
    for j, (label, p, style) in enumerate(series):
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=COLORS[j], linestyle=style, linewidth=1.6, label=label)
        ax.scatter(*p[0], color=COLORS[j], marker="o", s=18)
        ax.scatter(*p[-1], color=COLORS[j], marker="^", s=26)
    ax.set_xlabel(r"$p_x$ (m)")
    ax.set_ylabel(r"$p_y$ (m)")
    ax.set_zlabel(r"$p_z$ (m)")
    ax.set_title("trajectories, stated initial conditions, 50 s (cf. Fig. 3); circle = start, triangle = end",
                 fontsize=9, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    path = OUT / "fig3_trajectories.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    case = load_case_study()
    li, h = case.learning_inner, case.step
    zi = np.load(OUT6 / "weights_att_A.npz")

    def run(fol, T, weights_id=None, **kw):
        po, zo = build_problem(case, fol), np.load(OUT5 / f"weights_f{weights_id or fol.id}.npz")
        pi_ = build_inner_problem(fol, li.critic_degrees, li.actor_degrees)
        ctrl = CascadeController(lambda d: policy(po, zo["W_a"], d), lambda xe: policy(pi_, zi["W_a"], xe),
                                 ReferenceGenerator.from_case_study(case, fol))
        system = CascadeSystem.from_case_study(case, fol)
        return simulate_cascade(system, ctrl, system.initial_state_from_config(fol), T, h, **kw)

    print("1. reference runs (stated ICs, exact e_dot, 50 s)")
    refs = {}
    for fol in case.followers:
        refs[fol.id] = run(fol, T_REF)
        summary(f"follower {fol.id}, kappa = {fol.observer.kappa:g}", refs[fol.id])
        d_end = fol.x_star * refs[fol.id].e[-1]
        sep = case.leader.position(T_REF) + case.leader.sphere_offset - refs[fol.id].x[-1, 0:3]
        print(f"      final q_1 - p_{fol.id} = {np.round(sep, 4)} m (desired Delta = {fol.delta}), error {np.abs(d_end).max():.1e} m")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"  saved {figure_trajectories(case, refs).relative_to(ROOT)}")

    print("2. A6: follower 3 with kappa_13 = 5 instead of the printed 55 (30 s)")
    f3 = next(f for f in case.followers if f.id == 3)
    f3_k5 = dataclasses.replace(f3, observer=dataclasses.replace(f3.observer, kappa=5.0))
    summary("follower 3, kappa = 5", run(f3_k5, T_CHECK), refs[3])

    print("3. D15: finite-difference e_dot (tau = 0.02 s), 30 s")
    for fol in case.followers:
        summary(f"follower {fol.id}, finite-difference e_dot", run(fol, T_CHECK, e_dot_source="finite_difference"),
                refs[fol.id])


if __name__ == "__main__":
    main()
