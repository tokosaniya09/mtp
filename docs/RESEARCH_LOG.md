# Research Log — Occlusion-Resilient Multi-Modal Perception for Vision-Based Quadrotor Formation Control

Last updated: 2026-11-11

This is the canonical research log (decision D12). It continues the earlier external notes (snapshot of 2026-11-10).
Labels: **[Paper]** base paper's method · **[Eng]** our engineering decision · **[Ours]** our contribution.

## Project overview

- **Degree:** M.Tech thesis, with one research paper as the end goal.
- **Thesis title:** "Occlusion-Resilient Multi-Modal Perception for Vision-Based Quadrotor Formation Control"
- **Core question:** how does a follower quadrotor keep a reliable estimate of its leader's state, and hold
  formation, when its visual measurement of the leader is intermittent?
- **Current phase:** Phase 0 (scoping and setup).

## Base paper

X. Yi, H. Liu, H. Duan, J. Qiu, "Input-Constrained Visual Servoing Formation Control for Quadrotors Using
Off-Policy Reinforcement Learning," IEEE Trans. Cybernetics, vol. 56, no. 6, pp. 3540–3550, June 2026.
DOI: 10.1109/TCYB.2026.3656290. Local copy: `papers/base_paper.pdf`.

### Summary

Follower quadrotors keep formation with a leader using only an onboard monocular camera that sees a sphere
marker on the leader. No inter-vehicle communication, no GNSS.

1. **Image model (Sec. II, eqs. 2–5):** a virtual camera frame removes the follower's rotation; sphere-based
   image moments give linear image dynamics s_dot = v_ij / x*, so the feature behaves like a scaled relative position.
2. **Adaptive velocity observer (Sec. III, eq. 6, Thm 1):** estimates relative velocity from image data; an
   adaptive smoothed-sign term rejects the leader's unknown acceleration. Converges if alpha*kappa > 1/x*.
3. **Visual-servoing controller (Sec. IV, Alg. 1–3, Thm 2):** thrust-vector input with ||mu|| <= beta via a
   tanh-based cost, mu = -beta*tanh(...). Learned by off-policy integral RL with polynomial actor–critic
   networks solved by least squares, without the drift model.
4. **Attitude controller (Sec. V, Alg. 4):** same method for torque, ||u|| <= gamma; cascade with the outer loop.
5. **Simulation (Sec. VI):** 1 leader + 2 followers, beta = 5, gamma = 1, dT = 0.05 s; errors converge in
   about 5–10 s; better than ref. [16] with input clipping.

### Limitations we identified

1. Constraints are input bounds, not state constraints (FOV and tilt are not directly guaranteed).
2. Gravity vs beta inconsistency: hover needs ~12.7 N / 19.6 N but beta = 5 and Fig. 4 shows mu -> 0.
   A nonzero steady input would also make the infinite-horizon cost diverge.
3. Observer not exercised as designed: the leader moves at constant velocity.
4. Observer proof gaps: needs derivative of a measured signal; boundary layer gives bounded, not asymptotic,
   convergence; zeta-estimate convergence not shown.
5. No combined observer + outer + inner loop stability proof; mu -> (thrust, reference angles) mapping not given.
6. Theorem 2 condition depends on learned approximation error; not checkable in advance.
7. Idealized evaluation: simulation only; no noise, latency, occlusion, wind, collision avoidance; one baseline;
   no quantitative table.
8. **The visual signal is assumed continuous.** During occlusion the observer loses its measurement and
   Theorem 1 no longer applies. **This is the gap our thesis targets.**

### Details missing for reproduction

Polynomial basis terms; exploration (behaviour) policy; number of samples; mu -> attitude-reference mapping;
unknown chi_j; sphere radius; desired area a*; integration method. Each will be filled by a logged [Eng] decision.

## Research goals

Treated as one problem: keep a reliable leader-state estimate and stable formation when the visual measurement
is intermittent. G1 supplies measurements; G2 covers the gaps.

- **G1 — Adaptive multi-modal perception hierarchy.** Reliably detect and track the leader using multiple
  modalities: vision-based detection, markerless tracking, monocular/stereo, Vision Transformers, UWB and IMU
  localization, adaptive sensor selection and fusion.
- **G2 — Predictive occlusion handling and re-acquisition.** Maintain leader-state estimates during temporary
  occlusion and re-acquire the leader: trajectory prediction, motion estimation without vision,
  Transformer-based prediction, re-detection, switching between visual and non-visual sensing.

Framing:
- Depth goes into estimation and control; the leader detector is an off-the-shelf [Eng] component (D4, proposed).
- Baselines first: base paper unchanged; observer prediction-only; Kalman filter with UWB/IMU. A Transformer
  predictor counts only if it beats a tuned Kalman/IMM baseline.
- Target claim shape: "Under occlusions up to T s, our framework keeps formation error below X, where the base
  observer diverges and a Kalman fallback reaches Y." T, X, Y come from experiments.
- Novelty not yet confirmed; the literature review must locate the exact gap.

## Decision log (newest first)

| # | Date | Decision | Why | Impact | Status |
|---|---|---|---|---|---|
| D22 | 2026-11-11 | [Eng] RL utility/basis (Phase 1.4a): R must be diagonal (r passed as a vector); ln(1 − tanh²ξ) in eq. (26) evaluated as −2 ln cosh ξ (stable form); `PolynomialBasis` is generic and the degree sets are chosen later (A16, Phase 1.4b/1.5) | The literal ln(1 − tanh²ξ) underflows to −inf for |ξ| > ~19; (13) is only path-independent for diagonal R; Fig. 2 must guide the basis choice | `rl/utility.py`, `rl/basis.py` | Decided |
| D21 | 2026-11-11 | [Eng] Phase 1.3 validated in open loop (follower = eq. (1) with a prescribed near-hover thrust/torque excitation, no controller yet); Fig. 5 is compared on behaviour only, and the closed-loop comparison is repeated once the controller exists | Fig. 5 is closed-loop; its initial errors (A25) and oscillations (A26) cannot come from eq. (6) alone | `sim/observer_openloop.py`, `scripts/validate_observer.py`; revisit Fig. 5 in the closed-loop phase | Decided |
| D20 | 2026-11-11 | [Eng] Observer implementation: takes the PHYSICAL thrust vector R_j f_Tj e3 (D14 mapping stays in the controller); state [e_hat, v_hat, zeta_hat]; init per A19 (e_hat(0)=e(0), v_hat(0)=0, zeta_hat(0)=0); kappa_13 = 55 as printed (A6); FD e_dot = backward difference + first-order filter, tau = 0.02 s; extra [Eng] sinusoidal-leader test (A23) | Keeps eq. (6) literal and plant-agnostic; FD filter is the simplest realistic source | `leader_follower.py`, `observer.py`, `differentiator.py`, `sim/leaders.py` | Decided |
| D19 | 2026-11-11 | [Eng] A8: use Way 1 features s = vq/x* everywhere; eq. (3) implemented with disc moments (equals Way 1 exactly); exact elliptical sphere image not modelled | User choice; it is the model the paper's dynamics assume | A8 becomes a stated assumption, not numerically quantified | Decided |
| D18 | 2026-11-10 | Code hosted at github.com/tokosaniya09/mtp (branch main). Paper PDF not committed; local instruction file excluded via `.git/info/exclude` | IEEE licence forbids redistributing the PDF; instruction file is local-only by user choice | README tells readers to obtain the PDF via DOI | Decided |
| D17 | 2026-11-10 | [Eng] Plant eq. (1): state [p, v, R(9), omega]; RK4 h = 1 ms; R re-projected to SO(3) by SVD each step; ZYX Euler for output only; g = 9.81 (adopts A13, A18, A24 for the plant) | Integrating R avoids gimbal lock; projection is cheap insurance (measured drift without it: 4e-14 over 50 s at h = 1 ms) | `dynamics.py`, `sim/integrators.py` | Decided |
| D16 | 2026-11-10 | Focus on reproducing the base paper only; literature review deferred until the code reproduces Figs. 2–9 behaviour. Every error is logged (symptom, cause, fix) | User direction: a working reproduction first | Lit-review tasks moved after Phase 1; new Error log section | Decided |
| D15 | 2026-11-10 | [Eng] A4: observer's e_dot is pluggable; reproduce with exact e_dot from simulator truth, then re-run with filtered finite difference of sampled e as a check | Paper does not say how e_dot is obtained; exact value isolates the paper's design, finite difference shows realistic behaviour | Observer takes e_dot as an input; both modes tested and reported | Decided |
| D14 | 2026-11-10 | [Eng] A1: learn mu as the gravity-compensated input; applied thrust vector = mu + m_j*g*e3, so C = 0 for learning and the beta bound applies to the deviation | As printed, delta = 0 is not reachable with beta = 5 < m*g and the cost diverges; Fig. 4 shows mu -> 0 | Thrust mapping uses m_j (already needed by observer); state as a reproduction assumption in thesis | Decided |
| D13 | 2026-11-10 | Equation map done (`docs/equation_map.md`); ambiguities tracked there as A1–A23; [Eng] resolutions stay "Proposed" until the component that needs them is built | One place to trace each eq. to code and each gap to a decision; avoids silently fixing gaps | Phase 0.2 scaffold follows the module/function names in the map | Decided |
| D12 | 2026-11-10 | Keep the research log in the repository; this file is the canonical log | Log lives next to the code and is updated with it | Earlier external notes are a snapshot as of 2026-11-10 | Decided |
| D11 | 2026-11-10 | Simulation-only validation | No access to real drones | High-fidelity sim (e.g. Gazebo + PX4 SITL) in Phase 3 + Monte Carlo statistics; stated as a limitation | Decided |
| D10 | 2026-11-10 | Implement full codebase from scratch | No existing code; authors released none | Phase 0 includes scaffold; structure follows equation map | Decided |
| D9 | 2026-11-10 | Reproduce in pure Python (NumPy/SciPy, pytest), not Gazebo | Paper results come from ideal equations; a physics sim adds unmodelled effects so mismatches could not be traced; Gazebo only replaces the plant | Plant-agnostic controller/observer/learning modules for later ROS 2 + Gazebo/PX4 | Decided |
| D8 | 2026-11-10 | Keep a running research log | One place for decisions and progress; enables resuming | Updated every session (location per D12) | Decided |
| D7 | 2026-11-10 | Lightweight Python sim first; rendering simulator later | Validate maths before visuals; rendering needed only for G1 detection | No simulator setup until Phase 1 verified | Decided |
| D6 | 2026-11-10 | Workflow: lit review → gap/hypothesis → baselines → method → experiments; write as we go | Reviewers accept specific, testable claims | Gap statement before method design | Decided |
| D5 | 2026-11-10 | Thesis title as above | Matches narrowed scope | Approved by guide | Decided |
| D4 | 2026-11-10 | Depth in estimation/control; detector is off-the-shelf [Eng] | Matches base paper; allows proofs; avoids dataset/GPU competition | ViT/stereo are options, not core claims | Proposed |
| D3 | 2026-11-10 | G1 + G2 as one problem: formation under intermittent visual measurement | Extends the observer's continuous-measurement assumption (Limitation 8) | Single clear paper claim | Decided |
| D2 | 2026-11-10 | Narrow from five goals to G1 and G2 | One sharp contribution; G3–G5 weakly linked; G4/G5 assume communication | G3, G4, G5 out of scope / future work | Decided |
| D1 | 2026-11-10 | Use Yi et al. (2026) as base paper; reproduce first | Validated core and baseline to beat | Phase 1 before any extension | Decided |

## Current status and next steps

**Done:** base paper analysed; scope fixed to G1 + G2; title approved; log created; VS Code handoff kit prepared.
Phase 0.1 equation map (`docs/equation_map.md`), 2026-11-10: eqs. (1)–(42) mapped to modules; (13), (15),
(20)–(22), (27)–(28) and observer (7)/(9) re-derived and consistent with the paper; 23 ambiguities logged (A1–A23).

**Key findings from the map (short):**
- Observer: rho = e_dot − v_hat/x*, i.e. rho = v_tilde/x*. The observer needs the derivative of the measured feature,
  and has no input during occlusion (A4). This links directly to our thesis gap.
- tanh policy bounds each component (|mu_k| < beta), not ||mu||_2 ≤ beta as stated (A2).
- Fig. 9 z-component suggests Delta_ij is the desired sphere-centre offset q_i − p_j, so s* = Delta/x* (A10);
  the x/y initial values in Figs. 7 and 9 do not match the stated initial positions (A11).

**Next up (in order):**
- [x] Phase 0.1 — Build `docs/equation_map.md`.
- [x] Phase 0.2 — Scaffold: `.venv` (Python 3.13.1), `requirements.txt` (pinned), `pyproject.toml` (pytest config),
      `config/paper_case_study.yaml` (every value sourced), `src/vsfc/config.py` loader + 5 tests. 2026-11-10.
- [x] Phase 1.1 — Quadrotor dynamics eq. (1): `src/vsfc/dynamics.py`, `src/vsfc/sim/integrators.py`, 11 tests
      (hover, exact free fall, tilt sign convention, closed-form yaw spin, energy + angular-momentum conservation,
      SO(3), Euler round-trip, Euler-rate matrix). RK4 order measured: error ratio 15.999 per step halving (ideal 16). 2026-11-10.
- [x] Phase 1.2 — Camera & features, eqs. (2)–(4): `src/vsfc/camera.py`, 7 tests (hand-worked example for follower 2
      at t = 0: e = [-0.475, -0.425, 0]; eq. (3) with disc moments = Way 1 to 1e-12; s_dot = v/x*; eq. (2) interaction
      matrix vs finite difference; sphere-behind-camera rejected). 2026-11-11.
- [x] Phase 1.3 — Leader–follower dynamics eq. (5) + adaptive velocity observer eqs. (6)–(9), 2026-11-11.
      `leader_follower.py`, `observer.py`, `differentiator.py`, `sim/leaders.py`, `sim/observer_openloop.py`,
      `scripts/validate_observer.py`; 22 new tests (45 total, all pass). Eqs. (5)–(7) re-checked against the PDF.
      Results (open loop, h = 1 ms, paper's constant-velocity leader, exact e_dot):
      pair 12 |v~| < 1e-3 at 4.5 s; pair 13 at 0.5 s (kappa = 55); both reach ~1e-14 by 20 s. Fig. 5 shows
      convergence within ~5 s, which is consistent (behaviour only, D21).
      Eq. (5) matches eq. (1) by central differences (accelerating leader); (6) − (5) gives (7) to 1e-12;
      (9) bound holds at 1000 random states outside the boundary layer.
      [Eng] accelerating leader (||a|| <= 0.73 m/s^2): |v~| bounded at 0.16–0.21 m/s (pair 12), under the kappa-only
      bound 0.59; zeta_hat grows without limit (A5 confirmed).
      FD e_dot (tau = 0.02 s): |v~| ≈ (tau + h)·|a_ij| on average (≈ 0.02 m/s here), scales with tau: filter lag,
      not instability.
- [x] Phase 1.4a — Constrained utility and tanh policy, eqs. (11)–(13), (15), (25)–(26) (`rl/utility.py`), and polynomial
      basis, eqs. (23)–(24) (`rl/basis.py`), 2026-11-11. Eqs. (10)–(18), (23)–(27) re-checked against the PDF. 19 new tests
      (64 total, all pass): (13) = quadrature of (11) to 1e-9; hand value U(2.5) = 52.3248 (beta = 5, r = 8); dU/dmu by finite
      difference; U ≈ mu^T R mu for small mu (rel. error x²/6 confirmed); (15) equals the numerical minimiser of H to 1e-5;
      (26) = (13)∘tanh policy to 1e-9 and stays finite for large xi; basis sizes, Jacobian vs finite difference, quadratic-form weights.
      Finding: U is BOUNDED (→ 2 ln 2 · beta² r per component); the bound on mu comes from dU/dmu → ∞ (E4).
- [ ] Phase 1.4b — Off-policy IRL: data collection, Bellman rows (27)–(28), least squares (29), Alg. 3; LQR known-answer test.
- [ ] Phase 1.5+ — Outer-loop training on the paper's followers (Figs. 2a,b, 4a,b), inner loop, closed loop, figures.
- [ ] *Deferred until Phase 1 is done (D16):* literature review (occlusion / intermittent measurements;
      vision–UWB–IMU fusion), then a one-page gap statement and hypothesis.

**Planned phases:** 0 Scoping/setup (current) → 1 Faithful reproduction → 2 Our contribution → 3 Evaluation and writing.

## Open questions and pending inputs

- [ ] Confirm D4: estimation/control-centric contribution, detector off-the-shelf?
- [ ] Target venue (conference or journal) and paper deadline.
- [x] Gravity vs beta (A1): resolved by D14.
- [x] Source of e_dot in the observer (A4): resolved by D15.
- [ ] A14/A15: attitude state coordinates (literal z vs tracking errors) and the mu -> (f_T, Theta_r, omega_r) mapping.
- [ ] A11: Figs. 7/9 initial values inconsistent with stated initial conditions; validation will compare behaviour, not t=0 values.
- [ ] Remaining A-items (A2, A3, A6, A8–A10, A12, A13, A16–A20) have proposed resolutions in `docs/equation_map.md` §9.
- [ ] A25: Fig. 5 starts at ||e~_12(0)|| ≈ 8.8 and ||e~_13(0)|| ≈ 1.7, while ||e_12(0)|| ≈ 0.64; the paper's e_hat(0) is unknown
      and not reproducible (like A11). We use e_hat(0) = e(0) (A19).
- [ ] A26: Fig. 5 ||v~_12|| oscillates (peaks ≈ 1.2, 0.8 at ~0.3 s and ~1 s). With exact e_dot and the true thrust, (7) is
      linear with real eigenvalues (−5.02, −1.23 for pair 12) and the sat term only adds damping, so (6) alone cannot
      oscillate. Hypotheses: the paper fed the observer the commanded mu (not the actual R f e3 during attitude lag), or a
      noisy/differentiated e_dot. Check in the closed-loop phase.
- [ ] A6: sensitivity check of kappa_13 = 55 vs 5 still to do (in closed loop).
- [ ] Is UWB acceptable under the "communication-free" premise? State the assumption explicitly.

## Error log

Every error hit while building or running code: environment, install, test failure, wrong numbers.
Newest first. Status: Fixed / Open / Workaround.

| # | Date | Where | Symptom | Cause | Fix | Status |
|---|---|---|---|---|---|---|
| E4 | 2026-11-11 | Phase 1.4a, `docs/equation_map.md` §5.2 | Map claimed U(mu) → ∞ as |mu_k| → beta (wrong maths in our own notes) | U_k = beta² r f(x), f(x) = (1+x)ln(1+x) + (1−x)ln(1−x) → 2 ln 2: bounded. Only dU/dmu diverges | §5.2 corrected; test `test_positive_definite_and_bounded` checks the limit and the diverging gradient | Fixed |
| E3 | 2026-11-11 | Phase 1.3, `test_finite_difference_error_is_filter_lag` | Assertion failed: pointwise ratio |v~|/|a_ij| = 0.0154 < 0.8·(tau+h) = 0.0176 at t = 6 s | Test model too crude: the lag error is filtered by the observer's slow mode (~1.2 rad/s) while a_ij oscillates at 1.3 rad/s, so the pointwise ratio swings 0.013–0.026 | Test now checks the 4–10 s mean ratio (within ±25 % of tau+h) and the scaling with tau (0.02 vs 0.05); code unchanged | Fixed |
| E2 | 2026-11-11 | Phase 1.3, PDF check | `import fitz` / `import pypdf` fail in `.venv`; `pdftotext` drops Greek symbols; `pdftoppm` warns "No display font for 'Symbol'" | No PDF library in the venv; text extraction cannot map the Symbol font | Rendered pages with `pdftoppm` (warnings harmless, equations render correctly) and read eqs. (5)–(7) and Fig. 5 from the images | Workaround |
| E1 | 2026-11-10 | Phase 0.1, PDF tooling | `python -I crop.py` failed: `ModuleNotFoundError: No module named 'fitz'` | `-I` (isolated mode) ignores user site-packages, where PyMuPDF is installed | Ran the helper script without `-I` (script is ours, in scratchpad) | Fixed |
