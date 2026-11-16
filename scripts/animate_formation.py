"""Animation of the reproduced formation flight (presentation aid, not a validation step).

Simulates the full cascade (D34: plant (1), observer (6), learned outer and attitude policies, 1 ms control) for
both followers from the same initial case, then draws a GIF with two views:
  left:  3-D flight in the inertial frame (as in the paper's Fig. 3), with trails and each drone drawn as a
         quadrotor whose arms follow its attitude R; the leader carries the sphere marker (h_1 above it);
  right: the same flight seen from the leader's sphere marker: each follower's position relative to the sphere,
         with its desired formation slot -Delta_1j marked, so convergence to the formation is visible.
Each follower interacts only with the leader (Sec. VI), so the two pairs are simulated separately and drawn
together. Coordinates are plotted as stored (same convention as fig3_trajectories.png).

Cases: "stated" = the paper's Sec. VI initial conditions; "large" = [Eng] large-error case e(0) = large_error_e0
(A27), followers at rest, so the formation visibly forms.

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\animate_formation.py stated
    .venv\\Scripts\\python.exe scripts\\animate_formation.py large
Output: results/animations/formation_<case>.gif (simulation cached in formation_<case>.npz).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_outer_loop import build_problem                    # noqa: E402
from vsfc.config import load_case_study                       # noqa: E402
from vsfc.controllers.cascade import CascadeController        # noqa: E402
from vsfc.controllers.thrust_mapping import ReferenceGenerator  # noqa: E402
from vsfc.rl.offpolicy_irl import policy                      # noqa: E402
from vsfc.sim.cascade import CascadeSystem, simulate_cascade  # noqa: E402
from vsfc.sim.inner_loop import build_inner_problem           # noqa: E402

OUT = ROOT / "results" / "animations"
T_SIM = 30.0            # s simulated (the formation has settled by ~20 s)
SPEED = 2.0             # playback speed-up
FPS = 15
ARM = 0.7               # drawn arm length (m), enlarged for visibility
COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}    # validated categorical slots 1-3, as the other figures
INK, MUTED = "#2b2b2a", "#6b6a64"


def simulate(case_name: str) -> dict:
    """Run (or load) both followers' cascades for the chosen case."""
    cache = OUT / f"formation_{case_name}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    case = load_case_study()
    li, h = case.learning_inner, case.step
    zi = np.load(ROOT / "results" / "phase1_6" / "weights_att_A.npz")
    out = {}
    for fol in case.followers:
        po, zo = build_problem(case, fol), np.load(ROOT / "results" / "phase1_5" / f"weights_f{fol.id}.npz")
        pi_ = build_inner_problem(fol, li.critic_degrees, li.actor_degrees)
        ctrl = CascadeController(lambda d, po=po, zo=zo: policy(po, zo["W_a"], d),
                                 lambda xe: policy(pi_, zi["W_a"], xe), ReferenceGenerator.from_case_study(case, fol))
        system = CascadeSystem.from_case_study(case, fol)
        if case_name == "stated":
            x0 = system.initial_state_from_config(fol)
        else:
            x0 = system.initial_state(case.learning_outer.large_error_e0, system.leader.state(0.0)[1], fol.euler0)
        print(f"  simulating follower {fol.id} ({case_name}, {T_SIM:.0f} s) ...", flush=True)
        tr = simulate_cascade(system, ctrl, x0, T_SIM, h)
        out[f"t"] = tr.t
        out[f"p{fol.id}"] = tr.x[:, 0:3]
        out[f"R{fol.id}"] = tr.x[:, 6:15]
        out[f"delta{fol.id}"] = fol.delta
    out["p1"] = np.array([case.leader.position(ti) for ti in out["t"]])
    out["h1"] = case.leader.sphere_offset
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def arms(p: np.ndarray, R: np.ndarray) -> list[np.ndarray]:
    """Two crossing arm segments of a quadrotor at p with attitude R (body x and y axes)."""
    return [np.array([p - ARM * R[:, k], p + ARM * R[:, k]]) for k in (0, 1)]


def animate(case_name: str, data: dict) -> Path:
    t = data["t"]
    step = max(1, int(round(SPEED / (FPS * (t[1] - t[0])))))
    frames = np.arange(0, t.size, step)
    q1 = data["p1"] + data["h1"]                                    # leader's sphere marker
    ids = (2, 3)
    rel = {j: data[f"p{j}"] - q1 for j in ids}                       # follower relative to the sphere
    slot = {j: -data[f"delta{j}"] for j in ids}                      # desired: q_1 - p_j = Delta_1j

    fig = plt.figure(figsize=(11, 5))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    axr = fig.add_subplot(1, 2, 2, projection="3d")
    allp = np.vstack([data["p1"], q1] + [data[f"p{j}"] for j in ids])
    lo, hi = allp.min(0) - 1.5, allp.max(0) + 1.5
    allr = np.vstack([rel[j] for j in ids] + [slot[j][None] for j in ids] + [np.zeros((1, 3))])
    rlo, rhi = allr.min(0) - 1.0, allr.max(0) + 1.0
    for ax, (a, b), title in ((ax3, (lo, hi), "flight (inertial frame, cf. Fig. 3)"),
                              (axr, (rlo, rhi), "seen from the leader's marker: followers → formation slots")):
        ax.set_xlim(a[0], b[0]); ax.set_ylim(a[1], b[1]); ax.set_zlim(a[2], b[2])
        ax.set_xlabel(r"$x$ (m)"); ax.set_ylabel(r"$y$ (m)"); ax.set_zlabel(r"$z$ (m)")
        ax.set_title(title, fontsize=9, color=INK)
        ax.tick_params(colors=MUTED, labelsize=7)

    # static: formation slots (right) and the leader's marker at the origin
    axr.scatter(0, 0, 0, color=COLORS[1], s=60, marker="o", label="leader's sphere marker")
    for j in ids:
        axr.scatter(*slot[j], color=COLORS[j], s=110, marker="x", linewidths=2, label=f"slot of Quadrotor {j}")
    axr.legend(fontsize=7, frameon=False, loc="upper left")

    trails = {j: ax3.plot([], [], [], color=COLORS[j], linewidth=1.2, alpha=0.7)[0] for j in (1, 2, 3)}
    bodies = {j: [ax3.plot([], [], [], color=COLORS[j], linewidth=2.4)[0] for _ in range(2)] for j in (1, 2, 3)}
    sphere = ax3.plot([], [], [], marker="o", color=COLORS[1], markersize=7)[0]
    sightlines = {j: ax3.plot([], [], [], color=COLORS[j], linewidth=0.8, linestyle=":")[0] for j in ids}
    rtrails = {j: axr.plot([], [], [], color=COLORS[j], linewidth=1.2)[0] for j in ids}
    rdots = {j: axr.plot([], [], [], marker="o", color=COLORS[j], markersize=7)[0] for j in ids}
    clock = fig.text(0.5, 0.02, "", ha="center", fontsize=10, color=INK)
    labels = [ax3.plot([], [], [], color=COLORS[j], linewidth=2.4,
                       label=("Quadrotor 1 (leader)" if j == 1 else f"Quadrotor {j} (follower)"))[0] for j in (1, 2, 3)]
    ax3.legend(handles=labels, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"Reproduced formation flight, {'paper stated initial conditions' if case_name == 'stated' else 'large initial error case'}"
                 f" ({SPEED:.0f}x speed)", fontsize=11, color=INK)

    def setline(line, pts):
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])

    def update(i):
        for j in (1, 2, 3):
            p = data[f"p{j}"]
            setline(trails[j], p[: i + 1])
            R = np.eye(3) if j == 1 else data[f"R{j}"][i].reshape(3, 3)
            for line, seg in zip(bodies[j], arms(p[i], R)):
                setline(line, seg)
        setline(sphere, q1[i:i + 1])
        for j in ids:
            setline(sightlines[j], np.array([data[f"p{j}"][i], q1[i]]))
            setline(rtrails[j], rel[j][: i + 1])
            setline(rdots[j], rel[j][i:i + 1])
        err = max(np.linalg.norm(rel[j][i] - slot[j]) for j in ids)
        clock.set_text(f"t = {t[i]:5.1f} s     largest distance from formation slot: {err:5.2f} m")
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
    path = OUT / f"formation_{case_name}.gif"
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=80)
    plt.close(fig)
    return path


def main(case_name: str) -> None:
    if case_name not in ("stated", "large"):
        raise SystemExit("case must be 'stated' or 'large'")
    data = simulate(case_name)
    print(f"  saved {animate(case_name, data).relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stated")
