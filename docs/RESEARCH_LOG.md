# Research Log — Occlusion-Resilient Multi-Modal Perception for Vision-Based Quadrotor Formation Control

Last updated: 2026-11-13

This is the canonical research log (decision D12). It continues the earlier external notes (snapshot of 2026-11-10).
Labels: **[Paper]** base paper's method · **[Eng]** our engineering decision · **[Ours]** our contribution.

## Project overview

- **Degree:** M.Tech thesis, with one research paper as the end goal.
- **Thesis title:** "Occlusion-Resilient Multi-Modal Perception for Vision-Based Quadrotor Formation Control"
- **Core question:** how does a follower quadrotor keep a reliable estimate of its leader's state, and hold
  formation, when its visual measurement of the leader is intermittent?
- **Current phase:** Phase 1 (faithful reproduction); Phase 1.5 done (outer loop, ideal attitude), Phase 1.6 (attitude loop) in progress (D29, D30).

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
| D30 | 2026-11-13 | [Eng] A15: thrust-vector decomposition. F = mu + m_j g e3 (D14); f_T = \|F\|; body axis R e3 = F/\|F\|; psi_r = 0; (phi_r, theta_r) from R e3 under ZYX (A13); omega_r = body rate of the reference, W(Theta_r)^-1 times the filtered derivative of Theta_r (filter as D20); yaw error wrapped to (−pi, pi] | Paper gives no mapping; this is the standard cascade decomposition (a verified citation is still to be added). Fig. 8 e_psi(0) = psi(0) = 0.174 implies psi_r = 0; \|mu\| <= 5 < m g gives f_T > 0 and tilt <= arcsin(beta/(m g)) = 23.1° / 14.8° (corrected, E11); chi "unknown" fits a numerically generated reference | `controllers/thrust_mapping.py` (to build); reference rates feed the A14 error coordinates | Decided (user approved 2026-11-13) |
| D29 | 2026-11-13 | [Eng] A14: attitude state in tracking-error coordinates z = [Theta − Theta_r; omega − omega_r; Theta_r; omega_r] in R^12; G as printed; S = 10 I_12, D = 8 I_3, gamma = 1 unchanged | Literal z with block-diagonal S gives a separable V*, so u* cannot depend on Theta_r (it would level the vehicle and never track). This contradicts the Sec. V aim and Fig. 8 (tracking errors). Reference block of the cost is finite only because Theta_r, omega_r → 0 for the constant-velocity leader (revisit for accelerating leaders in Phase 2) | Attitude data/learner use error coordinates; Fig. 8 compared directly | Decided (user approved 2026-11-13) |
| D28 | 2026-11-12 | [Eng] A28: "randomly initialized" actor weights (Sec. VI) are drawn from ADMISSIBLE policies only: per-axis linear gains k_e, k_v ~ U(0.2, 2) > 0, cubic weights 0 (a tanh-saturated PD with positive gains stabilises a double integrator). Reference run starts from the behaviour gain K0 | Alg. 1 Step 1 requires an admissible mu^(0). Measured: unconstrained random N(0, 0.1²) starts converge per axis to the 3 non-stabilising Riccati solutions (E8); random cubic terms (std 0.01) still made 1–2 of 10 starts fail | `scripts/train_outer_loop.py`, config `init_gain_range`, `init_cubic_std` | Decided |
| D27 | 2026-11-12 | [Eng] Refines D26 exploration after prototyping: e(t) = b + low-frequency sines (0.2–1.1 rad/s; x 0.3 N, y/z 1.2 N each) + high-frequency dither (2.3, 3.7, 5.3 rad/s; 5 N each); b_x drawn per run in [−0.4, 3] N (b_y = b_z = 0); e_x(0) in [−0.8, 3.5], e_y,z(0) in [−3, 3]; 20 runs × 20 s, 5 s burn-in. All values in config `learning_outer` | With PD gain k_e the closed-loop DC gain from exploration force to e is 1/k_e, so low-frequency excitation of a few N drove e_x < −1 (sphere behind camera, E7); dither saturates mu while barely moving e; the bias spreads runs over depth errors −0.5…3.3 (covers stated ICs and the A27 case) | `sim/outer_loop.py`, `config.py` (`OuterLearningConfig`), `scripts/collect_outer_data.py`; `sum_of_sinusoids` accepts (Q, m) amplitudes | Decided |
| D26 | 2026-11-12 | [Eng] A17/A27 data for the outer loop (Phase 1.5 Step 3): data system = paper leader + ideal-attitude follower (D24) + observer (6) with exact e_dot + D14; each 20 s run drops its first 5 s (observer burn-in) from the training windows; behaviour mu = beta tanh((−K0 delta + sum of sinusoids)/beta), K0 = [1.0, 1.5] per axis; data region up to \|e_k\| ≈ 3.5, \|v_hat_k\| ≈ 3 m/s; ~20 runs (~6000 windows); actor init from K0 and from random seeds; extra large-error validation case e(0) ≈ [3.2, −1.45, 0] (pair 12, Fig. 7 x/y) | Learner assumes delta follows e_dot = v_hat/x*, which is false while v_hat converges (~4.5 s, Phase 1.3); dropping the burn-in keeps A19 and the observer unchanged; tanh keeps the behaviour input within ±beta; the region reaches saturation (A27) | `sim/outer_loop_data.py` (to build); rank/condition logged | Decided (user approved option A 2026-11-12) |
| D25 | 2026-11-12 | [Eng] A16 for the outer loop: per-axis bases. Critic = for each axis k, all degree-2 and degree-4 monomials in (e_k, v_hat_k) (8 per axis, l_c = 24); actor channel k = degree-1 and degree-3 monomials in (e_k, v_hat_k) (l_a = 6, 18 actor weights). Learner extended to per-channel actor bases sigma_a,k as in eq. (24), equal size l_a; shared basis (D23) still supported | Exact V* separates by axis and xi_k depends only on (e_k, v_hat_k) (§5.8); quartic/cubic terms let V depart from quadratic where mu saturates; counts 24/18 are close to Fig. 2a,b (~15–25 / ~15–20 traces) | `rl/basis.py` (`MonomialBasis`, `grouped_basis`, `axis_groups`), `rl/offpolicy_irl.py`; 10 new tests | Decided (user approved 2026-11-12) |
| D24 | 2026-11-12 | [Eng] Phase 1.5 trains and tests the outer loop with ideal attitude: applied force = mu + m_j g e3 enters the translational part of (1) directly; the inner loop joins in 1.6 | Isolates Alg. 3 (as the paper presents it) so mismatches trace to the outer loop; attitude lag would mix in | Fig. 4a,b comparison repeated with the full cascade in 1.6 | Decided (user confirmed 2026-11-12) |
| D23 | 2026-11-11 | [Eng] Off-policy learner (Phase 1.4b): one actor basis shared by the m input channels; (29) solved by SVD `lstsq` (same minimiser as (N^T N)^-1 N^T T at full rank), rank/condition/RMS logged per iteration and `train` raises on rank deficiency; input ZOH over h, so int mu sigma is integrated exactly per sub-step, other integrals by trapezoid on the 1 ms grid (A18 → Decided); A3 → Decided (r_{k1,k2} = [R]_{k1k2}, verified against the PDF and the LQR test) | Simplest faithful form of (27)–(29); SVD avoids squaring the condition number; ZOH matches the plant integrator | `rl/data.py`, `rl/offpolicy_irl.py`; quadrotor bases (A16) and behaviour policy (A17) still to choose in 1.5 | Decided |
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
      Committed as fae1cf4.
- [x] Phase 1.4b — Off-policy IRL, 2026-11-11: `rl/data.py` (windowed fine-grid data, RK4 + ZOH rollout, sum-of-sinusoids
      exploration), `rl/offpolicy_irl.py` (rows (27)–(28), least squares (29), Alg. 3 Steps 2–4). Eqs. (27)–(29) and Alg. 3
      re-checked against the PDF (p. 3545): the derived explicit row in equation_map §5.6 matches term by term. 18 new tests
      (82 total). Known-answer results:
      LQR (n = 3, m = 2, unstable A, Q = 10I, R = diag(8, 4), beta = 1000, 400 windows of 0.05 s): critic vs Riccati P and
      actor vs K/beta both 3e-7 relative; converged in 6 iterations with changes 44.5, 11.3, 2.5, 0.15, 5e-4, 8e-9
      (quadratic, as Kleinman/Newton predicts); iteration 1 = Kleinman's first Lyapunov iterate to 3e-6; held-out
      Bellman residual < 1e-5 relative; cond(N_bar) ≈ 1.1e5.
      Constrained scalar (x_dot = 0.5x + u, beta = 1, q = r = 1; exact u* by root-finding the pointwise HJB; u*(1) = −0.983,
      saturated): max |u_learned − u*| on |x| <= 1 is 5.7e-3 (actor degrees 1,3,5) and 6.0e-4 (1,3,5,7), so the error
      falls as the basis grows.
      Finding (A17): with on-policy data (mu = mu^(n)) the actor columns of N_bar vanish, so the behaviour policy must
      carry exploration; the paper's "admissible control policy" alone is not enough.
      Committed as 8831e0f.
- [x] Phase 1.5 — Outer loop on the paper's followers (done 2026-11-12, ideal attitude D24): choose bases (A16) and behaviour policy/M_j (A17) for delta = [e, v_hat],
      collect data with plant (1) + observer (6) + D14 gravity compensation, train, compare with Figs. 2a,b and 4a,b.
      Plan (2026-11-12): Step 0 re-read PDF → 1 learning problem and maths → 2 bases (A16) → 3 behaviour policy/M_j (A17)
      → 4 data + Alg. 3 training → 5 checks (LQR limit, |mu_k| < beta, held-out residual, seeds) → 6 closed loop vs
      Figs. 2a,b/4a,b → 7 log + commit. Attitude in 1.5: ideal (thrust vector applied directly), D24 (Decided).
  - [x] Step 0 (2026-11-12): Sec. IV (eqs. 10–27), Sec. VI, Figs. 2, 4 re-read from rendered pages (p. 3543–3544, 3547–3548).
        - Eq. (10) re-checked: A7 still stands (A_j depends on e~, zeta_bar, e_dot; first block uses v_hat).
        - Known answer: with the observer converged and D14, (10) becomes per axis e_dot = v_hat/x*, v_hat_dot = mu/m.
          LQR (x* = 4, Q = 10 I6, R = 8 I3) gives per-axis P = [[50.30, 11.63], [11.63, 14.62]], K = [1.118, 1.406] (m = 1.3)
          and P = [[55.06, 17.89], [17.89, 24.62]], K = [1.118, 1.539] (m = 2). Closed-loop poles −0.26, −0.82 and −0.29, −0.48.
          The learned policy must match this near delta = 0 (tanh policy ≈ linear, U ≈ mu^T R mu).
        - Fig. 2a: about 15–25 critic traces (overlapping; not countable exactly), converged by iteration ~5–6, plotted to
          iteration 11; final weights include large negative values (≈ −47, −33, −13). A diagonal-dominant quadratic
          critic (LQR above: all three per-axis weights positive) cannot produce these, so the paper's basis/ordering is not
          a plain quadratic in [e, v_hat]. Fig. 2 also does not say which follower it shows. → Compare iteration count and
          convergence shape, not individual weights (A16).
        - Fig. 2b: about 15–20 actor traces, final values within ±1.7; LQR-limit actor gains K/beta ≈ 0.22–0.31.
        - Fig. 4b inset: at t ≈ 0.1 s one component ≈ +5 while another ≈ +3, so ||mu_3||_2 ≈ 5.8 > beta = 5. Confirms the
          bound is per component (A2), as the tanh formula gives.
        - Fig. 4: early saturation at ±5 (purple component) for < 0.5 s, then mu → 0 within ~10 s (both followers).
  - [x] Step 1 (2026-11-12): learning problem written out in `docs/equation_map.md` §5.8 (variables, frames, units).
        - Once observer errors decay (they do not depend on mu), delta follows three decoupled double integrators
          e_dot = v_hat/x*, v_hat_dot = mu/m. Q, R, U also separate by axis, so V*(delta) = sum_k V1(e_k, v_hat_k) with one
          2-D function, and xi_k depends only on (e_k, v_hat_k). This guides the basis (A16, Step 2).
        - Near delta = 0 the learned controller must equal the LQR values above.
        - Finding (A27): scratch run with the LQR-limit tanh policy + real observer gives max |mu_k| = 0.53 N (j = 2) and
          0.83 N (j = 3) from the stated initial conditions, vs beta = 5; Fig. 4 saturates at ±5. Saturation in Fig. 4 cannot
          come from the stated ICs (consistent with A11). Validate on stated ICs and on an [Eng] large-error scenario.
  - [x] Step 2 (2026-11-12): bases (D25) + per-channel actor bases in the learner. 92 tests pass (10 new).
        - `MonomialBasis` (explicit exponents), `PolynomialBasis` now a subclass (unchanged behaviour), `grouped_basis`,
          `axis_groups`; `weights_from_quadratic_form` now rejects P entries the basis cannot represent.
        - Shared-basis path unchanged: all 82 earlier tests pass; per-channel identical bases reproduce the shared rows to 1e-13.
        - Known answer on the linearised outer loop (3 double integrators, x* = 4, m = 1.3, Q = 10 I6, R = 8 I3, beta = 1000,
          600 windows): quadratic/linear bases critic 1.3e-7, actor 9.9e-8 relative error vs Riccati; Phase 1.5 bases
          (2,4)/(1,3) critic 2.6e-7, actor 1.5e-7, quartic/cubic weights < 1e-4 of the largest; 4 iterations, changes
          1.0e2, 1.5, 1.1e-3, 1.2e-8 (quadratic convergence); cond(N_bar) 1.5e5 → 1.6e6 with the larger basis.
  - [x] Step 3 (2026-11-12): `sim/outer_loop.py` (system, simulate, behaviour policy, collect_dataset), config
        `learning_outer`, `scripts/collect_outer_data.py`; 7 new tests (99 total) incl. the §5.8 claim checked numerically
        (after burn-in e_dot = v_hat/x* to 1e-4 and v_hat_dot = mu/m; in the first second |v_hat − v| > 0.1, so the
        model does not hold there). Data (D26/D27), 6000 windows per follower (wall ~200 s each):
        follower 2: e_x −0.50…3.28, e_y −2.58…2.43, e_z −2.06…1.86, |v_hat| ≤ 4.3 m/s; |mu_k| > 0.9 beta in 23–25 % of samples;
        max |v_hat − v| in kept data 3.1e-4 m/s. Follower 3: e_x −0.58…3.24, similar; max |v_hat − v| 2e-14 (kappa = 55).
        Stated e_x(0) = −0.475 (j = 2) is inside the data; −0.75 (j = 3) slightly outside (min −0.58).
  - [x] Step 4 (2026-11-12): `scripts/train_outer_loop.py`, Alg. 3 on 6000 windows, 42 unknowns, rank 42, cond(N_bar)
        5.0e4 (j = 2) / 5.7e4 (j = 3). From K0: converged in 4 iterations (changes 101, 1.6, 1.7e-3, 5e-7 for j = 2).
        LQR limit: linear actor gains within 0.26 % (j = 2) and 0.19 % (j = 3) of K; quadratic critic weights within 0.4 % of P.
        Quartic critic weights ≤ 0.35, cubic actor ≤ 0.007 (the constrained correction away from delta = 0).
        5 random admissible starts (D28) per follower: same weights to ≤ 2e-11, 6–7 iterations. Fig. 2 shows convergence
        "within several iterations" (settled by ~5–6): consistent.
        Finding (A28): non-admissible random starts converge to non-stabilising Riccati/HJB solutions (E8).
  - [x] Steps 5–6 (2026-11-12): `scripts/validate_outer_loop.py`; figures in `results/phase1_5/` (fig2ab_weights_f*,
        fig4ab_mu_f*, fig7_e_f*; plain diagnostics, palette validated, git-ignored).
        - Held-out Bellman residual (900 new windows): 7.7e-4 (j = 2), 4.9e-4 (j = 3) relative.
        - Closed loop 50 s, learned policy, observer from A19 init:
          stated ICs: max |mu_k| 0.53 / 0.83 N; ||e|| < 0.05 after 8.9 / 11.4 s, < 0.01 after 15.0 / 17.0 s; ||e(50)|| ~1e-6.
          large error e(0) = [3.2, −1.45, 0]: max |mu_k| 3.2 / 3.3 N; ||e|| < 0.05 after 17.8 s (both).
          near saturation (same e(0), v_ij(0) = [2, −2, 2]): max |mu_k| 3.2 / 4.4 N, max ||mu||_2 4.1 / 6.2 N (> beta: A2).
        - Critic V_hat never increases after the 5 s burn-in (all runs): Lyapunov property of eq. (32) holds.
        - Learned policy cost ≤ LQR-limit tanh policy in every case (e.g. 1352.65 vs 1354.99, j = 3 near saturation):
          the quartic/cubic terms improve on the small-delta approximation, as the constrained optimum should.
        - Comparison with the paper: Fig. 2a,b — weights settle in ~5–6 iterations (ours: 4 from K0, 6–7 from random
          admissible starts), consistent; individual weights not comparable (A16). Fig. 4a,b — paper shows ±5 saturation
          for < 0.5 s; ours never exceeds 4.4 N. With the learned gain ≈ 1.118/beta = 0.224 on e near 0, mu reaches ±5
          (tanh ≥ 0.99) only for |e| ≳ 12; Fig. 7 shows e_z(0) ≈ 12 for pair 12, so Fig. 4's saturation matches the
          PLOTTED initial errors, not the stated positions (A27 explained via A11). Errors converge in ~9–11 s from the
          stated ICs vs "5–10 s" in the paper: consistent.
- [ ] Phase 1.6 — Inner (attitude) loop, current. A14 → D29 (error coordinates), A15 → D30 (thrust-vector
      decomposition) decided 2026-11-13.
      - [x] Step 1 (2026-11-13): `controllers/thrust_mapping.py` (D30) + `tests/test_thrust_mapping.py` (20 tests pass).
            Round trip f_T R(Theta_r) e3 = F to 1e-12; max tilt arcsin(beta/(m g)) = 23.1° / 14.8° (E11); omega_r vs an
            independent R^T R_dot truth: 0.07 % (raw difference), 2.8 % (tau = 0.02 s lag) of max |omega_r| = 0.29 rad/s.
            Config: `engineering.ref_rate_tau = 0.02`.
      - [ ] Step 2: attitude learning problem in D29 coordinates. Derivation done 2026-11-13 (equation map §6.1): reference
            block of S does not change the policy; neglecting the small d(t) the problem separates per axis into double
            integrators; LQR known answer K ≈ [1.118, 1.13] per axis, dominant pole −1 (consistent with Fig. 8); Fig. 2c,d show
            only ~6–7 weights each. New open question A29 (weak inner/outer time-scale separation). Basis choice (A16)
            awaiting user decision.
      - [ ] Step 3: attitude data (reference signals + behaviour torque, A17 analogue). Then Alg. 4 data/training like 1.5, cascade with the 1.5 outer loop, repeat Fig. 4 with attitude lag
      (A26 hypothesis), Figs. 2c,d, 4c,d, 8.
- [ ] Phase 1.7 — Full closed loop (observer + outer + inner), baseline [16] with input clipping, Figs. 2–10.
- [ ] *Deferred until Phase 1 is done (D16):* literature review (occlusion / intermittent measurements;
      vision–UWB–IMU fusion), then a one-page gap statement and hypothesis.

**Planned phases:** 0 Scoping/setup (done) → 1 Faithful reproduction (current) → 2 Our contribution → 3 Evaluation and writing.

## Open questions and pending inputs

- [ ] Confirm D4: estimation/control-centric contribution, detector off-the-shelf?
- [ ] Target venue (conference or journal) and paper deadline.
- [x] Gravity vs beta (A1): resolved by D14.
- [x] Source of e_dot in the observer (A4): resolved by D15.
- [x] A14/A15: resolved by D29 (tracking-error coordinates) and D30 (thrust-vector decomposition, psi_r = 0).
- [ ] A11: Figs. 7/9 initial values inconsistent with stated initial conditions; validation will compare behaviour, not t=0 values.
- [ ] Remaining A-items (A2, A3, A6, A8–A10, A12, A13, A16–A20) have proposed resolutions in `docs/equation_map.md` §9.
- [ ] A25: Fig. 5 starts at ||e~_12(0)|| ≈ 8.8 and ||e~_13(0)|| ≈ 1.7, while ||e_12(0)|| ≈ 0.64; the paper's e_hat(0) is unknown
      and not reproducible (like A11). We use e_hat(0) = e(0) (A19).
- [ ] A26: Fig. 5 ||v~_12|| oscillates (peaks ≈ 1.2, 0.8 at ~0.3 s and ~1 s). With exact e_dot and the true thrust, (7) is
      linear with real eigenvalues (−5.02, −1.23 for pair 12) and the sat term only adds damping, so (6) alone cannot
      oscillate. Hypotheses: the paper fed the observer the commanded mu (not the actual R f e3 during attitude lag), or a
      noisy/differentiated e_dot. Check in the closed-loop phase.
- [x] A27: stated ICs keep the outer-loop input at ≤ 0.83 N vs beta = 5, but Fig. 4 saturates. Explained (Phase 1.5): ±5
      needs |e| ≳ 12, which matches Fig. 7's plotted e_z(0) ≈ 12 (A11), not the stated positions. Validated on stated ICs,
      a large-error case and a near-saturation case.
- [ ] A28: "randomly initialized" weights (Sec. VI) vs Alg. 1's admissible initial policy: random non-admissible starts
      converge to non-stabilising solutions (E8). We use random admissible starts (D28); state in thesis.
- [ ] A6: sensitivity check of kappa_13 = 55 vs 5 still to do (in closed loop).
- [ ] Is UWB acceptable under the "communication-free" premise? State the assumption explicitly.

## Error log

Every error hit while building or running code: environment, install, test failure, wrong numbers.
Newest first. Status: Fixed / Open / Workaround.

| # | Date | Where | Symptom | Cause | Fix | Status |
|---|---|---|---|---|---|---|
| E10 | 2026-11-12 | Phase 1.5, docs edit | Scripted log edit aborted with `SyntaxError: (unicode error) 'unicodeescape' ... truncated \xXX escape`; nothing written | LaTeX (`\xi`, `\mu`) inside a non-raw Python string (same family as E5) | Re-ran from a file with raw strings; verified LaTeX and no control characters. Rule: LaTeX edits only via raw strings or the editor | Fixed |
| E11 | 2026-11-13 | Phase 1.6 Step 1, D30 wording | Max tilt claimed as arctan(beta/(m g)) = 21.4° / 14.3° (also told to the user) | That is the tilt for a horizontal mu only. Over the ball \|mu\| <= beta the tilt is largest when mu is tangent to the sphere around m g e3, giving sin(tilt) = beta/(m g) | Corrected to arcsin(beta/(m g)) = 23.1° (Quad 2), 14.8° (Quad 3), checked by a brute-force sweep; `test_tilt_bound` asserts both the bound and that it is attained | Fixed |
| E10 | 2026-11-13 | Phase 1.6, docs edit (D29/D30) | `git diff --stat` showed 565 changed lines in RESEARCH_LOG.md for a ~10-line edit; one `sed` status substitution in `equation_map.md` silently did not match | Repeat of E6: `sed -i` rewrote the CRLF log with LF. The second sed pattern's LaTeX escaping did not match the text | Restored CRLF from bytes (diff back to 15 lines); made the unmatched edit with the editor. Rule restated: no `sed -i` on `docs/`, use the editor | Fixed |
| E9 | 2026-11-12 | Phase 1.5, file edits | A scripted config edit failed its own match check; inspection showed line endings changed in 6 files (YAML CRLF → LF; `equation_map.md`, `config.py`, `outer_loop.py` LF → CRLF) | `sed -i` in Git Bash rewrote the YAML with LF; Python `open(p, 'w')` on Windows writes CRLF | Restored each file to its committed convention (new files LF, the repo majority); scripted edits now read/write bytes and keep the file's own ending | Fixed |
| E8 | 2026-11-12 | Phase 1.5 Step 4, training | Random initial actor N(0, 0.1²): 2 of 5 runs (follower 2) did not converge in 50 iterations; converged runs ended 60–175 % away from the K0-initialised weights | Each axis converges independently; axes whose initial gain is not stabilising converge to the other three solutions of the per-axis Riccati equation, e.g. K = (−1.118, ±0.723), (1.118, −1.406) for m = 1.3 (matched to 3 decimals by enumerating all ARE solutions). With random cubic weights (std 0.01) 1–2 of 10 admissible-gain starts still failed | Random starts restricted to admissible policies, cubic part 0 (D28, A28): 10/10 seeds per follower reach the K0 solution to 1e-11 | Fixed |
| E7 | 2026-11-12 | Phase 1.5 Step 3, data prototyping | (a) PD gains (1, 1.5) + 1 N sines: after burn-in |e| < 1 and 0 % of samples near the bound (region too small). (b) Weaker gains / larger low-frequency sines: every run crashed with `ValueError: sphere is not in front of the camera` | Closed-loop DC gain from exploration force to e is 1/k_e; low-frequency sines summing to several N push e_x below −1 (follower passes the leader) | Split exploration into low-frequency (small on x) + high-frequency dither + per-run x bias (D27); verified min e_x ≥ −0.79 over all 40 runs | Fixed |
| E6 | 2026-11-12 | Phase 1.5, docs edit | `git diff --stat` showed 426 changed lines in RESEARCH_LOG.md for a ~30-line edit | Scripted edit wrote LF; the file is stored with CRLF (other docs are LF) | Rewrote the file with CRLF; diff back to the real change. Scripted edits must keep the existing line endings | Fixed |
| E5 | 2026-11-11 | Phase 1.4b, docs edit | A scripted Python string replace wrote a backspace character into `equation_map.md` (`\bar` read as `\b` + "ar") and silently skipped one replacement; a `sed` repair then dropped the backslash | Escape sequences in a non-raw Python string; `sed` replacement also consumes `\\` | Fixed both lines with direct edits; checked the file has no control characters left. Use raw strings or the editor for LaTeX text | Fixed |
| E4 | 2026-11-11 | Phase 1.4a, `docs/equation_map.md` §5.2 | Map claimed U(mu) → ∞ as |mu_k| → beta (wrong maths in our own notes) | U_k = beta² r f(x), f(x) = (1+x)ln(1+x) + (1−x)ln(1−x) → 2 ln 2: bounded. Only dU/dmu diverges | §5.2 corrected; test `test_positive_definite_and_bounded` checks the limit and the diverging gradient | Fixed |
| E3 | 2026-11-11 | Phase 1.3, `test_finite_difference_error_is_filter_lag` | Assertion failed: pointwise ratio |v~|/|a_ij| = 0.0154 < 0.8·(tau+h) = 0.0176 at t = 6 s | Test model too crude: the lag error is filtered by the observer's slow mode (~1.2 rad/s) while a_ij oscillates at 1.3 rad/s, so the pointwise ratio swings 0.013–0.026 | Test now checks the 4–10 s mean ratio (within ±25 % of tau+h) and the scaling with tau (0.02 vs 0.05); code unchanged | Fixed |
| E2 | 2026-11-11 | Phase 1.3, PDF check | `import fitz` / `import pypdf` fail in `.venv`; `pdftotext` drops Greek symbols; `pdftoppm` warns "No display font for 'Symbol'" | No PDF library in the venv; text extraction cannot map the Symbol font | Rendered pages with `pdftoppm` (warnings harmless, equations render correctly) and read eqs. (5)–(7) and Fig. 5 from the images | Workaround |
| E1 | 2026-11-10 | Phase 0.1, PDF tooling | `python -I crop.py` failed: `ModuleNotFoundError: No module named 'fitz'` | `-I` (isolated mode) ignores user site-packages, where PyMuPDF is installed | Ran the helper script without `-I` (script is ours, in scratchpad) | Fixed |
