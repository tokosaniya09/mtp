"""Phase 1.5 Step 3: collect off-policy data for Alg. 3 (outer / visual-servoing loop), paper Alg. 2-3 Step 1.

System: paper leader, follower with ideal attitude (D24), gravity compensation (D14), observer (6) with exact
e_dot (D15); behaviour policy and data region per D26 (config `learning_outer.data`). The first `burn_in`
seconds of each run are dropped (D26). Saves results/phase1_5/data_f<j>.npz and prints coverage statistics.

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\collect_outer_data.py 2
    .venv\\Scripts\\python.exe scripts\\collect_outer_data.py 3
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vsfc.config import load_case_study                       # noqa: E402
from vsfc.sim.outer_loop import OuterLoopSystem, collect_dataset  # noqa: E402

OUT = ROOT / "results" / "phase1_5"


def data_path(follower_id: int) -> Path:
    return OUT / f"data_f{follower_id}.npz"


def coverage_report(delta: np.ndarray, mu: np.ndarray, beta: float) -> str:
    d = delta.reshape(-1, delta.shape[-1])
    u = mu.reshape(-1, mu.shape[-1])
    p = lambda q: np.round(np.percentile(d, q, axis=0), 2)
    lines = [
        f"  delta = [e_x, e_y, e_z, vh_x, vh_y, vh_z]",
        f"  min   {np.round(d.min(0), 2)}",
        f"  p5    {p(5)}",
        f"  p95   {p(95)}",
        f"  max   {np.round(d.max(0), 2)}",
        f"  fraction |e_k| < 0.2:            {np.round((np.abs(d[:, :3]) < 0.2).mean(0), 3)}",
        f"  fraction |mu_k| > 0.9 beta:      {np.round((np.abs(u) > 0.9 * beta).mean(0), 3)}",
    ]
    return "\n".join(lines)


def main(follower_id: int) -> None:
    case = load_case_study()
    fol = next(f for f in case.followers if f.id == follower_id)
    system = OuterLoopSystem.from_case_study(case, fol)
    spec = case.learning_outer.data
    t0 = time.time()
    data, trajs = collect_dataset(system, spec, case.step, case.dT)
    wall = time.time() - t0
    burn = int(round(spec.burn_in / case.step))
    v_tilde = max(np.abs(tr.v_true[burn:] - tr.delta[burn:, 3:]).max() for tr in trajs)
    min_ex = min(tr.e_true[:, 0].min() for tr in trajs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path(follower_id), delta=data.delta, mu=data.mu, h=data.h)
    print(f"follower {follower_id}: {spec.n_runs} runs x {spec.run_length} s (burn-in {spec.burn_in} s dropped), "
          f"{data.n_windows} windows of {data.dT:.3f} s, wall {wall:.0f} s")
    print(f"  max |v_hat - v| in kept data: {v_tilde:.2e} m/s;  min e_x over all runs: {min_ex:.2f} (must be > -1)")
    print(coverage_report(data.delta, data.mu, system.beta))
    print(f"  saved {data_path(follower_id).relative_to(ROOT)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
