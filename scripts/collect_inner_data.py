"""Phase 1.6 Step 3: collect off-policy data for Alg. 4 (attitude loop), paper Alg. 4 Step 1.

System [Eng, D32]: rotational part of eq. (1), smooth sinusoidal reference (psi_r = 0, D30), saturated-PD behaviour
torque with exploration; learning state x_e = [e_Theta; e_omega] with the Euler-rate error (D33). Settings: config
`learning_inner.data`. J, S, D, gamma are the same for both followers (Sec. VI), so ONE dataset serves both.
Saves results/phase1_6/data_att.npz and prints coverage statistics.

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\collect_inner_data.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsfc.config import load_case_study                       # noqa: E402
from vsfc.sim.inner_loop import collect_attitude_dataset      # noqa: E402

OUT = ROOT / "results" / "phase1_6"
DATA = OUT / "data_att.npz"


def attitude_follower(case):
    """The attitude problem is identical for all followers (checked here); return the first."""
    f0 = case.followers[0]
    for f in case.followers[1:]:
        if not (np.array_equal(f.inertia, f0.inertia) and np.array_equal(f.S, f0.S) and np.array_equal(f.D, f0.D)
                and f.gamma == f0.gamma and f.tol_att == f0.tol_att):
            raise ValueError("followers differ in J, S, D, gamma or tol_att: train the attitude loop per follower")
    return f0


def coverage_report(x_e: np.ndarray, u: np.ndarray, gamma: float) -> str:
    d = x_e.reshape(-1, 6)
    u = u.reshape(-1, 3)
    p = lambda q: np.round(np.percentile(d, q, axis=0), 2)
    return "\n".join([
        "  x_e = [e_phi, e_theta, e_psi, e_phidot, e_thetadot, e_psidot]  (rad, rad/s; D33 Euler-rate error)",
        f"  min   {np.round(d.min(0), 2)}",
        f"  p5    {p(5)}",
        f"  p95   {p(95)}",
        f"  max   {np.round(d.max(0), 2)}",
        f"  max |u_k| {np.round(np.abs(u).max(0), 3)} (gamma = {gamma}); fraction |u_k| > 0.9 gamma "
        f"{np.round((np.abs(u) > 0.9 * gamma).mean(0), 3)}",
    ])


def main() -> None:
    case = load_case_study()
    fol = attitude_follower(case)
    spec = case.learning_inner.data
    t0 = time.time()
    data, trajs = collect_attitude_dataset(fol, spec, case.step, case.dT)
    wall = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, delta=data.delta, mu=data.mu, h=data.h)
    print(f"attitude data (both followers): {spec.n_runs} runs x {spec.run_length} s, {data.n_windows} windows of "
          f"{data.dT:.3f} s, wall {wall:.0f} s")
    print(f"  max |Theta_r| {max(np.abs(t.theta_r).max() for t in trajs):.3f} rad, "
          f"max |omega| {max(np.abs(t.x[:, 9:]).max() for t in trajs):.2f} rad/s")
    print(coverage_report(data.delta, data.mu, fol.gamma))
    print(f"  saved {DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
