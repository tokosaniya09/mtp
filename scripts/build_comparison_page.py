"""Build a LOCAL side-by-side page: each figure of the base paper next to our reproduction.

The paper's figures are cropped from papers/base_paper.pdf (IEEE copyright): the page and the crops stay on this
computer (results/comparison/ is git-ignored) and must not be published or committed.

Needs: poppler's `pdftoppm` (found on PATH, or set POPPLER_BIN), Pillow, and our figures from run_cascade.py,
run_baseline16.py, run_phase1_checks.py, validate_inner_loop.py, validate_outer_loop.py.

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe scripts\\build_comparison_page.py
Then open results\\comparison\\index.html in a browser.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "comparison"
PDF = ROOT / "papers" / "base_paper.pdf"
DPI = 200

# Paper figure -> (PDF page, crop box as fractions of the page width/height: x0, y0, x1, y1). Measured from rendered
# pages of the PDF (letter size); boxes include the captions.
CROPS = {
    "2": (8, (0.498, 0.072, 0.940, 0.408)),
    "3": (8, (0.498, 0.448, 0.940, 0.722)),
    "4": (9, (0.064, 0.075, 0.500, 0.394)),
    "5": (9, (0.064, 0.421, 0.500, 0.674)),
    "6": (9, (0.505, 0.082, 0.916, 0.338)),
    "7": (9, (0.505, 0.362, 0.916, 0.617)),
    "8": (9, (0.505, 0.645, 0.916, 0.897)),
    "9": (10, (0.078, 0.081, 0.494, 0.341)),
    "10": (10, (0.078, 0.362, 0.494, 0.630)),
}

# What we show for each paper figure, and the verdict (from docs/RESEARCH_LOG.md, "Phase 1 reproduction summary").
ROWS = [
    ("2", "NN weights: (a,b) visual-servoing loop, (c,d) attitude loop",
     ["results/phase1_5/fig2ab_weights_f2.png", "results/phase1_6/fig2cd_weights_att.png"],
     "partly",
     "Outer loop: weights settle in 4 iterations (6–7 from random admissible starts), as in Fig. 2a,b. Attitude loop: "
     "converges in 9 iterations with a larger basis (24/18 weights) than the ~6–7 traces of Fig. 2c,d; the smaller "
     "basis does not converge on the exact plant. Individual weight values cannot be compared (basis unknown)."),
    ("3", "3-D trajectories of the formation",
     ["results/phase1_7/fig3_trajectories.png"],
     "reproduced",
     "Followers join the leader's path; final offsets equal the desired Delta to 4e-6 m. The paper's plotted "
     "starting points do not follow from its stated initial conditions (A11)."),
    ("4", "Control inputs: thrust deviation mu (a,b) and torque u (c,d)",
     ["results/phase1_6/fig4_inputs_f2.png", "results/phase1_6/fig4_inputs_f3.png"],
     "partly",
     "Inputs stay within ±5 N and ±1 N m and go to zero. From the stated initial conditions mu never saturates "
     "(max 0.53 / 0.83 N); from Fig. 7's plotted initial error it saturates at ±5 for 0.7–1.0 s like Fig. 4a,b. "
     "Torque spikes reach 0.43–0.46 N m, not ±1 (A32)."),
    ("5", "Observer errors",
     ["results/phase1_6/fig5_observer_f2.png", "results/phase1_3_observer_errors.png"],
     "partly",
     "Both observer errors go to zero (4.5 s for pair 1–2, 0.5 s for pair 1–3, consistent with kappa_13 = 55). "
     "The paper's two-peak oscillation is not reproduced: ours is monotone, with one hump only if the observer is "
     "fed the commanded thrust (A26). The paper's initial errors are not reproducible (A25)."),
    ("6", "Estimated relative velocities",
     ["results/phase1_6/fig5_observer_f2.png"],
     "reproduced",
     "v_hat converges to the true relative velocity (shown here through the observer error ||v~|| → 0)."),
    ("7", "Image feature errors",
     ["results/phase1_6/fig7_e_cascade_f2.png", "results/phase1_6/fig7_e_cascade_f3.png"],
     "partly",
     "Converge to ~1e-6; ||e|| < 0.05 after 10.4 / 12.7 s (paper: ~5–10 s). Fig. 7's e_z ≈ 12 → 0 in ~3 s cannot "
     "happen with |mu| ≤ 5 N (A11)."),
    ("8", "Attitude tracking errors",
     ["results/phase1_6/fig8_attitude_cascade_f2.png", "results/phase1_6/fig8_attitude_cascade_f3.png"],
     "partly",
     "e_psi 0.174 → 0.03 rad at 2 s as in Fig. 8; all errors < 0.01 rad after 3.3–3.7 s. The pitch oscillation up "
     "to ~0.5–0.8 rad is not reproduced (ours ≤ 0.39 rad, monotone; A32)."),
    ("9", "Relative position errors, proposed controller",
     ["results/phase1_6/fig9_relpos_cascade_f2.png"],
     "reproduced",
     "Errors converge to zero (< 0.05 m after 15.7 / 17.5 s). Our Delta_z transient is ~0.5 m vs ~0.03 m plotted "
     "(the leader climbs at 0.4 m/s while the follower starts at rest; A11)."),
    ("10", "Relative position errors, baseline [16] with input saturation",
     ["results/phase1_7/fig9_10_relpos_f2.png", "results/phase1_7/fig9_10_relpos_f3.png"],
     "partly",
     "As the paper claims, the baseline converges more slowly (22–30 % longer) with larger final errors (20–125×), "
     "and saturates for up to 2 s. Fig. 10's long oscillation is not reproduced; ours is smooth, also with [16]'s "
     "own weights (A33)."),
]

VERDICT_STYLE = {"reproduced": ("Reproduced", "#1baf7a"), "partly": ("Partly reproduced", "#eb6834")}


def find_pdftoppm() -> str:
    exe = shutil.which("pdftoppm")
    if exe:
        return exe
    env = os.environ.get("POPPLER_BIN")
    if env and (Path(env) / "pdftoppm.exe").exists():
        return str(Path(env) / "pdftoppm.exe")
    raise SystemExit("pdftoppm not found: put poppler's bin folder on PATH or set POPPLER_BIN")


def crop_paper_figures() -> dict[str, str]:
    exe = find_pdftoppm()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for page in sorted({p for p, _ in CROPS.values()}):
            subprocess.run([exe, "-r", str(DPI), "-f", str(page), "-l", str(page), "-png", str(PDF),
                            str(Path(tmp) / f"p{page}")], check=True, capture_output=True)
        for fig, (page, (x0, y0, x1, y1)) in CROPS.items():
            src = next(Path(tmp).glob(f"p{page}*.png"))
            im = Image.open(src)
            w, h = im.size
            name = f"paper_fig{fig}.png"
            im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).save(OUT / name)
            out[fig] = name
    return out


def build_page(paper: dict[str, str]) -> Path:
    rows = []
    for fig, title, ours, verdict, note in ROWS:
        label, color = VERDICT_STYLE[verdict]
        imgs = "".join(
            f'<a href="{html.escape(os.path.relpath(ROOT / p, OUT))}"><img src="{html.escape(os.path.relpath(ROOT / p, OUT))}" '
            f'alt="our reproduction of Fig. {fig}"></a>' if (ROOT / p).exists() else
            f'<p class="missing">missing: {html.escape(p)} (run the script that makes it)</p>' for p in ours)
        rows.append(f"""
<section id="fig{fig}">
  <h2>Fig. {fig} — {html.escape(title)} <span class="badge" style="background:{color}">{label}</span></h2>
  <div class="pair">
    <div class="col"><h3>Paper</h3><a href="{paper[fig]}"><img src="{paper[fig]}" alt="paper Fig. {fig}"></a></div>
    <div class="col"><h3>Our reproduction</h3>{imgs}</div>
  </div>
  <p class="note">{html.escape(note)}</p>
</section>""")
    anims = "".join(
        f'<div class="col"><h3>{t}</h3><img src="../animations/{f}" alt="{t}"></div>'
        for f, t in (("formation_stated.gif", "Paper's stated initial conditions"),
                     ("formation_large.gif", "Large initial error case")) if (ROOT / "results" / "animations" / f).exists())
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Phase 1 — reproduction vs base paper</title>
<style>
 body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 0 auto; max-width: 1400px; padding: 24px;
        color: #2b2b2a; background: #fbfbf9; line-height: 1.45; }}
 h1 {{ margin-bottom: 4px; }} .sub {{ color: #6b6a64; margin-top: 0; }}
 nav a {{ margin-right: 10px; }}
 section {{ border-top: 1px solid #e6e5e0; padding: 18px 0; }}
 h2 {{ font-size: 1.15rem; }} h3 {{ font-size: .9rem; color: #6b6a64; margin: 0 0 6px; text-transform: uppercase; letter-spacing: .04em; }}
 .badge {{ color: #fff; font-size: .75rem; padding: 2px 8px; border-radius: 10px; vertical-align: middle; margin-left: 8px; }}
 .pair {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr); gap: 20px; align-items: start; }}
 .col img {{ width: 100%; border: 1px solid #e6e5e0; background: #fff; margin-bottom: 8px; }}
 .note {{ max-width: 1100px; }} .missing {{ color: #b00; }}
 .anim {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
</style></head><body>
<h1>Phase 1 — our reproduction next to the base paper</h1>
<p class="sub">Yi et al., IEEE Trans. Cybern. 56(6), 2026. Local page only: the left column is cropped from the paper's PDF
(IEEE copyright), do not publish. "Reproduced" means the behaviour matches; plotted values often cannot match because
the paper's plotted initial values do not follow from its stated initial conditions (A11). Details:
docs/RESEARCH_LOG.md, "Phase 1 reproduction summary". Click any image to open it full size.</p>
<nav>{" ".join(f'<a href="#fig{f}">Fig. {f}</a>' for f, *_ in ROWS)} <a href="#anim">Animations</a></nav>
{"".join(rows)}
<section id="anim"><h2>Animations of the reproduced formation flight</h2>
<div class="anim">{anims or '<p class="missing">run scripts/animate_formation.py stated / large</p>'}</div>
<p class="note">Left view: actual flight; right view: positions relative to the leader's sphere marker, with each
follower's formation slot marked ×. Followers are simulated with the full model, observer and learned controllers.</p>
</section>
</body></html>"""
    path = OUT / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    if not PDF.exists():
        raise SystemExit(f"missing {PDF}")
    OUT.mkdir(parents=True, exist_ok=True)
    paper = crop_paper_figures()
    print(f"cropped {len(paper)} paper figures; page: {build_page(paper).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
