"""Phase 1.3 numerical validation of the adaptive velocity observer, eq. (6).

Runs the open-loop harness (follower = eq. (1) with a prescribed near-hover excitation; no controller yet)
for both pairs of the paper's case study and prints convergence metrics. Saves a plain diagnostic figure
laid out like the paper's Fig. 5 (||e_tilde||, ||v_tilde|| vs time). NOT a reproduction of Fig. 5: the
paper's figure is closed-loop and its initial errors cannot be reproduced (see research log, A25/A26).

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\validate_observer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsfc.config import load_case_study                       # noqa: E402
from vsfc.sim.leaders import ConstantVelocityLeader, SinusoidalLeader  # noqa: E402
from vsfc.sim.observer_openloop import simulate_observer_open_loop     # noqa: E402

DURATION = 20.0
H = 1e-3                     # config simulation.step (A18)
FD_TAU = 0.02                # [Eng] D15 filter time constant


def settle_time(t, x, tol):
    """First time after which x stays below tol (nan if never)."""
    above = np.nonzero(x >= tol)[0]
    if above.size == 0:
        return t[0]
    return t[above[-1] + 1] if above[-1] + 1 < t.size else np.nan


def summarise(label, run):
    nv, ne = np.linalg.norm(run.v_tilde, axis=1), np.linalg.norm(run.e_tilde, axis=1)
    def fmt(ts):
        return " never" if np.isnan(ts) else f"{ts:5.2f}s"
    print(f"  {label:<34s} |v~|(0)={nv[0]:.3f}  max|v~|={nv.max():.3f}  "
          f"t(|v~|<1e-2)={fmt(settle_time(run.t, nv, 1e-2))}  t(|v~|<1e-3)={fmt(settle_time(run.t, nv, 1e-3))}  "
          f"|v~|(end)={nv[-1]:.2e}  |e~|(end)={ne[-1]:.2e}  zeta^(end)={run.zeta_hat[-1]:.3f}")


def main():
    cs = load_case_study()
    h_i = cs.leader.sphere_offset
    paper_leader = ConstantVelocityLeader.from_config(cs.leader)
    acc_leader = SinusoidalLeader(cs.leader.p0, cs.leader.velocity, np.array([0.5, 0.5, 0.2]), 1.0)

    runs = {}
    for f in cs.followers:
        o = f.observer
        print(f"Pair 1{f.id}: alpha={o.alpha}, kappa={o.kappa}, eps={o.eps}, x*={f.x_star}, "
              f"alpha*kappa={o.alpha * o.kappa} > 1/x*={1 / f.x_star}")
        runs[(f.id, "exact")] = r = simulate_observer_open_loop(f, paper_leader, h_i, cs.g, DURATION, H)
        summarise("paper leader, exact e_dot", r)
        r = simulate_observer_open_loop(f, paper_leader, h_i, cs.g, DURATION, H,
                                        e_dot_source="finite_difference", fd_tau=FD_TAU)
        summarise(f"paper leader, FD e_dot (tau={FD_TAU})", r)
        r = simulate_observer_open_loop(f, acc_leader, h_i, cs.g, DURATION, H)
        summarise(f"[Eng] accel. leader (zeta_i={acc_leader.accel_bound:.2f})", r)
        print(f"      kappa-only linear bound zeta_i x*/kappa = {acc_leader.accel_bound * f.x_star / o.kappa:.3f} m/s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for f in cs.followers:
        r = runs[(f.id, "exact")]
        axes[0].plot(r.t, np.linalg.norm(r.e_tilde, axis=1), label=f"Quadrotors 1, {f.id}")
        axes[1].plot(r.t, np.linalg.norm(r.v_tilde, axis=1), label=f"Quadrotors 1, {f.id}")
    axes[0].set_ylabel("||e~_ij||")
    axes[1].set_ylabel("||v~_ij|| (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend()
    fig.suptitle("Phase 1.3 diagnostic: observer errors (open loop, exact e_dot)")
    out = ROOT / "results" / "phase1_3_observer_errors.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
