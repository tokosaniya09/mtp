# Equation-to-Code Map — Yi et al., IEEE TCYB 56(6):3540–3550, 2026

Phase 0.1 deliverable. Source of truth: `papers/base_paper.pdf`. Every equation below was transcribed from the PDF
(checked against 300-dpi crops). Labels: **[Paper]** = as printed · **[Eng]** = our implementation choice ·
**[Derived]** = algebra we did ourselves to check or expand a step the paper skips (each one marked so it can be
re-checked) · **[Ours]** = research contribution (none in this document).

Ambiguities are numbered **A1–A28** (Section 9). A proposed resolution is *not adopted* until the component that
needs it is implemented and the decision is logged in `docs/RESEARCH_LOG.md`.

Indices: leader `i`, follower `j`. Case study: leader 1, followers 2 and 3 (pairs 12, 13).
Shapes use NumPy convention: `(3,)` = vector, `(3,3)` = matrix.

---

## 0. Pipeline at a glance

```
leader trajectory p_i(t) ──► sphere centre q_i = p_i + h_i
                                   │
follower plant (1) ──► p_j, R_j ──►│ v q_ij = q_i − p_j  (virtual frame, rotation removed)
                                   ▼
                     image moments (2)-(3) ──► feature s_ij ──► error e_ij = s_ij − s*_ij  (4)
                                   │                                 │
                                   ▼                                 ▼
                     observer (6): ê, v̂, ζ̂  ◄──── follower's own thrust vector μ_j
                                   │
                     δ_j = [e_ij; v̂_ij] ──► outer loop  μ_j = −β tanh(ξ̂(δ_j))      (25)
                                   │
                     μ_j ──► (f_T, Θ_r, ω_r)   [missing in paper, A15]
                                   │
                     z_j = [Θ, ω, Θ_r, ω_r] ──► inner loop u_j = −γ tanh(ς̂(z_j))  (42)
                                   │
                     (f_T, u_j = τ_j) ──► follower plant (1)
```

Both loops are learned offline by off-policy integral RL (Alg. 3 and Alg. 4), from recorded data, without using
the drift model.

---

## 1. Quadrotor model — Sec. II, eq. (1) → `src/vsfc/dynamics.py`

**[Paper] (1)**

$$\dot p_j = v_j,\qquad m_j\dot v_j = -f_{Tj}R_je_3 + m_jge_3,\qquad \dot R_j = R_j\,sk(\omega_j),\qquad J_j\dot\omega_j = -\omega_j\times J_j\omega_j + f_{\tau j}$$

**Meaning.** Rigid body, thrust along the body z-axis, gravity along $+e_3$. Thrust enters as $-f_TRe_3$ and
gravity as $+mge_3$, so $e_3$ points **down** (NED-like convention, A12). Hover requires $f_T R e_3 = m g e_3$,
i.e. $f_T = mg$ with the body level.

| Symbol | Meaning | Shape | Units | Frame |
|---|---|---|---|---|
| $p_j$ | position | (3,) | m | ${}^e\mathcal F$ (inertial) |
| $v_j$ | linear velocity | (3,) | m/s | ${}^e\mathcal F$ |
| $\Theta_j=[\phi,\theta,\psi]$ | Euler angles (convention unstated, A13) | (3,) | rad | — |
| $R_j$ | rotation body→inertial | (3,3) | — | ${}^b\mathcal F\to{}^e\mathcal F$ |
| $\omega_j$ | body angular velocity | (3,) | rad/s | ${}^b\mathcal F$ |
| $m_j$ | mass | scalar | kg | — |
| $J_j$ | inertia (diagonal in Sec. VI) | (3,3) | kg·m² (paper prints kg·m²/rad², A21) | ${}^b\mathcal F$ |
| $f_{Tj}$ | total thrust | scalar | N | — |
| $f_{\tau j}$ | body torque | (3,) | N·m | ${}^b\mathcal F$ |
| $g$ | gravity | scalar | m/s² | — |
| $e_3$ | $[0,0,1]^T$ | (3,) | — | ${}^e\mathcal F$ |
| $sk(\cdot)$ | skew-symmetric matrix, $sk(a)b=a\times b$ | (3,3) | — | — |

**Code.**
- `skew(w) -> (3,3)`
- `euler_to_rotation(theta) -> (3,3)` and `rotation_to_euler(R) -> (3,)`. Convention [Eng] ZYX, see A13.
- `QuadrotorParams(m, J, g)` dataclass.
- `quadrotor_derivatives(x, f_T, tau, params) -> x_dot`, with state `x = [p(3), v(3), R.flatten()(9), ω(3)]`.
  [Eng] Integrate $R$ directly, as eq. (1) does, and re-orthonormalise after each step (SVD projection).
  Euler angles are output only.

**Tests.** Hover ($f_T=mg$, $R=I$ → $\dot v=0$); free fall ($f_T=0$ → $\dot v = ge_3$); $R$ stays orthonormal
with $\det R = 1$ over a long run; torque-free rotation about a principal axis keeps $\omega$ constant;
rotational kinetic energy is conserved when $\tau=0$.

---

## 2. Camera, sphere moments, image dynamics — Sec. II, eqs. (2)–(4) → `src/vsfc/camera.py`

**[Paper] Geometry (unnumbered).** Sphere centre $q_i = p_i + h_i$. The virtual frame ${}^v\mathcal F_j$ moves
with the follower but does not rotate with it (the rotation from ${}^c\mathcal F_j$ to ${}^v\mathcal F_j$ is
$R_j$), so its axes are parallel to the inertial axes and
$${}^vq_{ij} = q_i - p_j = [x_{ij},y_{ij},z_{ij}]^T,\qquad {}^v\dot q_{ij} = v_{ij} = v_i - v_j.$$
The optical axis is ${}^b\mathcal X_j$, so $x_{ij}$ is the depth.

**[Paper] (2) Perspective projection on the virtual image plane**
$$u_{ij} = \lambda_j y_{ij}/x_{ij},\qquad n_{ij} = \lambda_j z_{ij}/x_{ij}$$
Differentiating gives the point-feature interaction matrix (unnumbered)
$$\begin{bmatrix}\dot u\\\dot n\end{bmatrix} = \begin{bmatrix}-u/x & \lambda/x & 0\\ -n/x & 0 & \lambda/x\end{bmatrix} v_{ij}.$$

**[Paper] Moments.** $m^{pq}_{ij} = \iint u^p n^q\,du\,dn$ over the sphere's image, with area $a_{ij}=m^{00}_{ij}$.

**[Paper] (3) Features**
$$s_{xij} = \sqrt{a^*_{ij}/a_{ij}},\qquad s_{yij} = s_{xij}\,m^{10}_{ij}/(\lambda_ja_{ij}),\qquad s_{zij} = s_{xij}\,m^{01}_{ij}/(\lambda_ja_{ij})$$
and (unnumbered) $\dot s_{ij} = v_{ij}/x^*_{ij}$.

**[Paper] (4)** $e_{ij} = s_{ij}-s^*_{ij}$.

**Explanation [Derived — to verify numerically in Phase 1].** Approximate the sphere's image as a disc centred on
the projected centre, with area $a \approx \pi\lambda^2 r^2/x^2$, where $r$ is the sphere radius. Then
$\sqrt{a^*/a} = x/x^*$. The centroid is $m^{10}/a = u = \lambda y/x$, so
$s_y = (x/x^*)(y/x) = y/x^*$, and likewise $s_z = z/x^*$. Hence
$$s_{ij} \approx {}^vq_{ij}/x^*_{ij}\quad\Rightarrow\quad \dot s_{ij} = v_{ij}/x^*_{ij}.$$
The feature is the relative position scaled by the constant desired depth. That is why the image dynamics are
linear and do not depend on the unknown depth. The exact perspective image of a sphere is an ellipse whose area
is $\pi\lambda^2r^2/(x^2-r^2)$ on-axis, and whose centroid is not the projected centre off-axis. So the relation
is exact only as $r/x\to0$ near the optical axis (A8). Note that $\lambda$ and $r$ cancel from $s$: only $x^*$
matters for the dynamics.

| Symbol | Meaning | Shape | Units | Frame |
|---|---|---|---|---|
| $h_i$ | leader body → sphere centre offset; $[0,0,1]$ in Sec. VI | (3,) | m | inertial assumed, A19 |
| $q_i$ | sphere centre | (3,) | m | ${}^e\mathcal F$ |
| ${}^vq_{ij}=[x,y,z]$ | sphere centre relative to follower $j$ | (3,) | m | ${}^v\mathcal F_j$ (inertial-parallel) |
| $v_{ij}$ | relative velocity $v_i-v_j$ | (3,) | m/s | ${}^v\mathcal F_j$ |
| $\lambda_j$ | focal length, 3.2 mm | scalar | m | — |
| $u_{ij},n_{ij}$ | image coordinates | scalar | m (image plane) | virtual image plane |
| $m^{pq}_{ij}$, $a_{ij}$ | image moments, area | scalar | m^{2+p+q} | virtual image plane |
| $a^*_{ij}$ | desired area (not given, A9) | scalar | m² | — |
| $x^*_{ij}$ | desired depth, $>0$; 4 m inferred from $\Delta$ (A9, A10) | scalar | m | — |
| $s_{ij}$, $s^*_{ij}$, $e_{ij}$ | feature, desired feature, error | (3,) | dimensionless | ${}^v\mathcal F_j$ |

**Code.**
- `sphere_centre(p_i, h_i)`, `relative_position_virtual(q_i, p_j)`
- `project_point(vq, lam) -> (u, n)` (2), `point_interaction_matrix(u, n, x, lam)` (unnumbered)
- `sphere_image_moments(vq, radius, lam) -> (a, m10, m01)`: numerical moments of the true projected ellipse,
  used to quantify the error of A8
- `moment_features(a, m10, m01, a_star, lam) -> s` (3)
- `ideal_features(vq, x_star) -> vq / x_star`: the model the paper's dynamics assume ([Eng], A8)
- `feature_error(s, s_star)` (4)
- `in_field_of_view(vq, R_j, fov)`: diagnostic. The real camera sees $R_j^T\,{}^vq$, not ${}^vq$.

**Tests.** `moment_features` matches `ideal_features` as $r/x\to0$; finite-difference $\dot s$ matches $v/x^*$;
the interaction matrix matches finite differences of (2); a vector ${}^vq$ equal to $x^*s^*$ gives $e=0$.

---

## 3. Image-based leader–follower dynamics — eq. (5) → `src/vsfc/leader_follower.py`

**[Paper] (5)**
$$\dot e_{ij} = v_{ij}/x^*_{ij},\qquad \dot v_{ij} = -ge_3 + R_jf_{Tj}e_3/m_j + \dot v_i,\qquad \|\dot v_i\|\le\zeta_i$$

**Explanation.** Eq. (5) is the dynamics of the *relative* state, i.e. the leader as seen by the follower's camera. The leader moves freely: its acceleration $\dot v_i$ is an unknown, bounded disturbance. The follower's thrust vector is the only control input. $\dot v_{ij}=\dot v_i-\dot v_j$; substituting $\dot v_j$ from (1) gives the second line. With
the control input defined as $\mu_j = R_jf_{Tj}e_3$ (the thrust vector, used in Sec. IV), the system is a
double integrator in $(e, v_{ij})$ driven by $\mu_j/m_j$, by gravity, and by the unknown, bounded leader
acceleration.

| Symbol | Meaning | Shape | Units |
|---|---|---|---|
| $\dot v_i$ | leader acceleration (unknown to follower) | (3,) | m/s² |
| $\zeta_i$ | bound on $\|\dot v_i\|$ (unknown) | scalar | m/s² |

**Code.** `relative_dynamics(v_ij, thrust_vec, a_leader, m_j, g, x_star) -> (e_dot, v_ij_dot)` (does not depend on $e$; `thrust_vec` is the physical $R_jf_{Tj}e_3$, D20) and `relative_state(p_i, v_i, h_i, p_j, v_j, x_star, s_star) -> (e, v_ij)`. Used for tests and
as the reference model the observer is checked against. The plant itself is (1) for each vehicle.

**Tests.** Consistency with (1): simulate leader and follower separately, compute $e$ and $v_{ij}$, and check
that their derivatives match (5).

---

## 4. Adaptive velocity observer — Sec. III, eqs. (6)–(9), Theorem 1 → `src/vsfc/observer.py`

**[Paper] (6)** (with $\tilde e = e-\hat e$, $\tilde v = v_{ij}-\hat v$)
$$\begin{aligned}
\dot{\hat e}_{ij} &= \hat v_{ij}/x^*_{ij} + \alpha_{ij}\tilde e_{ij}\\
\dot{\hat v}_{ij} &= -ge_3 + R_jf_{Tj}e_3/m_j + \bar\zeta_{ij} - \tilde e_{ij}/x^*_{ij} + \kappa_{ij}\rho_{ij}\\
\bar\zeta_{ij} &= \hat\zeta_{ij}\,sat(\rho_{ij},\epsilon_{ij})\\
\dot{\hat\zeta}_{ij} &= x^*_{ij}\|\rho_{ij}\|
\end{aligned}
\qquad \rho_{ij} = \dot{\tilde e}_{ij}+\alpha_{ij}\tilde e_{ij}$$
Elementwise $sat(\rho,\epsilon) = \rho/\max(|\rho|,\epsilon)$: equal to $sign(\rho)$ outside the boundary layer
$|\rho|<\epsilon$ and linear inside it.

**[Paper] (7) Error dynamics** (with $\tilde\zeta = \zeta_i-\hat\zeta_{ij}$)
$$\dot{\tilde e} = \tilde v/x^* - \alpha\tilde e,\qquad \dot{\tilde v} = \dot v_i - \bar\zeta + \tilde e/x^* - \kappa\tilde v/x^*,\qquad \dot{\tilde\zeta} = -\|\tilde v\|$$

**[Paper] (8)–(9), Theorem 1.**
$L_o = \tfrac12\tilde e^T\tilde e + \tfrac12\tilde v^T\tilde v + \tfrac12\tilde\zeta^2$, with
$\dot L_o \le -\alpha\|\tilde e\|^2 - \tfrac{\kappa}{x^*}\|\tilde v\|^2 + \tfrac{2}{x^*}\tilde e^T\tilde v$.
This is negative definite in $(\tilde e,\tilde v)$ iff $\alpha\kappa > 1/x^*$.

**Explanation [Derived].**
1. Substitute the first line of (6): $\dot{\tilde e} = \dot e - \dot{\hat e} = \tilde v/x^* - \alpha\tilde e$, so
   $$\rho = \dot{\tilde e}+\alpha\tilde e = \tilde v/x^* \qquad(\text{equivalently } \rho = \dot e - \hat v/x^*).$$
   This identity is what makes (7) follow from (6): $\kappa\rho = \kappa\tilde v/x^*$ is a damping term on
   $\tilde v$, and $\dot{\hat\zeta} = x^*\|\rho\| = \|\tilde v\|$.
2. $\bar\zeta = \hat\zeta\,sat(\rho)$ is a sliding-mode-like term that cancels the unknown leader acceleration.
   Its amplitude $\hat\zeta$ adapts upward until it dominates $\|\dot v_i\|$. Outside the boundary layer,
   $\tilde v^T sign(\tilde v) = \|\tilde v\|_1 \ge \|\tilde v\|_2$, so
   $\tilde v^T(\dot v_i - \hat\zeta\,sign) - \tilde\zeta\|\tilde v\| \le 0$. Those terms are dropped, which gives (9).
3. The cross term $2\tilde e^T\tilde v/x^*$ comes from the $-\tilde e/x^*$ in (6). Bounding it with Young's
   inequality gives the gain condition $\alpha\kappa>1/x^*$. Sec. VI values: $\alpha\kappa = 25$ (pair 12) and 275
   (pair 13), against $1/x^* = 0.25$.
4. **Implementability (A4).** $\rho$ needs $\dot e$, the time derivative of the measured feature. If $\dot e$
   were available exactly, $x^*\dot e = v_{ij}$ would give the velocity directly. In practice $\dot e$ must
   come from differentiating a noisy, sampled $e$. This also means the observer has no input at all while
   the leader is occluded, which is the gap the thesis targets.

| Symbol | Meaning | Shape | Units | Sec. VI value |
|---|---|---|---|---|
| $\hat e,\hat v$ | estimates of $e$, $v_{ij}$ | (3,), (3,) | –, m/s | init. not given (A19) |
| $\hat\zeta_{ij}$ | estimate of $\zeta_i$ | scalar | m/s² (implicit unit gain, A21) | init. not given |
| $\bar\zeta_{ij}$ | robust term | (3,) | m/s² | — |
| $\rho_{ij}$ | filtered error $=\tilde v/x^*$ | (3,) | 1/s | — |
| $\alpha_{ij}$ | gain | scalar | 1/s | 5, 5 |
| $\kappa_{ij}$ | gain | scalar | m/s | 5 (12), **55** (13), A6 |
| $\epsilon_{ij}$ | boundary-layer width | scalar | 1/s | 0.5, 0.5 |

Note: the observer uses $m_j$, $g$, $R_j$ and $f_{Tj}$, i.e. it needs the follower's own mass and thrust. The
controller (Sec. IV) is model-free; the observer is not (A22).

**Code.**
- `sat(rho, eps)` (elementwise)
- `class AdaptiveVelocityObserver(alpha, kappa, eps, x_star, m_j, g)` with
  `derivatives(obs_state, e_meas, e_dot_meas, thrust_vec) -> obs_state_dot`, where `obs_state = [ê(3), v̂(3), ζ̂(1)]`
  and `thrust_vec` is the physical $R_jf_{Tj}e_3$ (D20); `signals(...) -> (ẽ, ρ, ζ̄)` (needed later by (10)).
- `differentiator.FilteredDifferentiator(dt, tau)`: the finite-difference source of $\dot e$ (D15, D20).
- `sim/observer_openloop.simulate_observer_open_loop(...)`: open-loop harness, follower = eq. (1) with prescribed excitation (D21).
  The source of `e_dot_meas` is pluggable (A4).
- `error_dynamics(...)` (7), `lyapunov(...)` (8), `lyapunov_rate_bound(...)` (9), `gain_condition(alpha, kappa, x_star)` (Thm 1): test helpers.

**Tests.** `sat` boundary cases; with exact $\dot e$ and a sinusoidally accelerating leader, $\tilde e,\tilde v\to$
a small neighbourhood of 0 (bounded rather than exact, because of the boundary layer, A5); $L_o$ is
non-increasing outside the boundary layer; the gain condition holds for the Sec. VI values.

---

## 5. Visual-servoing controller — Sec. IV, eqs. (10)–(32) → `src/vsfc/rl/`, `src/vsfc/controllers/visual_servo.py`

### 5.1 Learning system — (10)

**[Paper] (10)** $\dot\delta_j = A_j(\delta_j) + B_j\mu_j + C_j$, with $\delta_j=[e_{ij};\hat v_{ij}]\in\mathbb R^6$,
$\mu_j = R_jf_{Tj}e_3$, $\Omega_{\mu_j}=\{\mu:\|\mu\|\le\beta_j\}$, and
$$A_j = \begin{bmatrix}\hat v_{ij}/x^*\\ \bar\zeta_{ij}-\tilde e_{ij}/x^* + \kappa(\dot e_{ij}+\alpha\tilde e_{ij})\end{bmatrix},\quad B_j=\begin{bmatrix}0_{3\times3}\\ I_3/m_j\end{bmatrix},\quad C_j=\begin{bmatrix}0_{3\times3}\\ -ge_3\end{bmatrix}$$

**Explanation.** The state is built from things the follower has: its measured feature error and its own velocity
estimate. The second block is exactly the observer equation (6). Two observations (A7):
1. The first block uses $\hat v/x^*$, while the true $\dot e = v/x^*$. Eq. (10) equals the true dynamics only up
   to $\tilde v/x^*$.
2. $A_j$ also depends on $\tilde e$, $\bar\zeta$ and $\dot e$, which are not functions of $\delta_j$ alone.
The off-policy method never evaluates $A,B,C$ (it uses recorded trajectories), so this does not change the
algorithm. It does mean $V(\delta)$ is an approximation on a non-autonomous system. Note also that $C_j$ is
written as $(3\times3)$ blocks but must be $(6,)$.

### 5.2 Cost and constrained utility — (11)–(13)

**[Paper]** $V_j(\delta_j)=\int_t^\infty(\delta^TQ\delta + U(\mu))d\tau$ with
$$U(\mu) = 2\int_0^\mu\beta(\Upsilon^{-1}(\upsilon/\beta))^T\mathcal R\,d\upsilon \;\;(11),\qquad \Upsilon=\tanh\;\;(12)$$
$$U(\mu) = 2\beta\tanh^{-1}(\mu/\beta)^T\mathcal R\mu + \beta^2\bar{\mathcal R}\ln(\mathbf 1_3-(\mu/\beta)^2)\;\;(13)$$
with $\mathcal R=\mathrm{diag}(r_1,r_2,r_3)$, $\bar{\mathcal R}=[r_1,r_2,r_3]$, and $\ln$ and the squaring applied
elementwise.

**[Derived] check of (13).** $\int\tanh^{-1}x\,dx = x\tanh^{-1}x+\tfrac12\ln(1-x^2)$. With $\upsilon=\beta x$:
$2\beta r\int_0^{\mu}\tanh^{-1}(\upsilon/\beta)d\upsilon = 2\beta r\mu\tanh^{-1}(\mu/\beta) + \beta^2r\ln(1-(\mu/\beta)^2)$. ✓
Per component, $U_k=\beta^2r_kf(x)$ with $x=\mu_k/\beta$ and $f(x)=2x\tanh^{-1}x+\ln(1-x^2)=(1+x)\ln(1+x)+(1-x)\ln(1-x)$.
So $U\ge0$, $U(0)=0$, $U\approx\mu^T\mathcal R\mu$ for small $\mu$ (relative error $\approx x^2/6$), and **$U$ stays bounded**:
$f\to2\ln2$ as $|x|\to1$. *(Corrected 2026-11-11: an earlier version of this map said $U\to\infty$; see E4.)* What keeps
the optimal input inside the bound is the gradient $\partial U/\partial\mu=2\beta\mathcal R\tanh^{-1}(\mu/\beta)$, which
diverges at the boundary, so the minimiser of $H$ is always interior. (13) requires $\mathcal R$ diagonal: only then is the
integrand of (11) a gradient field, so the integral is path-independent.

### 5.3 Hamiltonian, optimal policy, HJB — (14)–(16)

**[Paper]** $H = \delta^TQ\delta + U(\mu) + \nabla V^T(A+B\mu+C)$ (14). Setting $\partial H/\partial\mu=0$ gives
$$\mu^* = -\beta\tanh(\xi^*),\qquad \xi^* = \tfrac{1}{2\beta}\mathcal R^{-1}B^T\nabla V^*\quad(15)$$
and $H(\delta,\nabla V^*,\mu^*)=0$ (16, HJB).

**[Derived] check of (15).** $\partial U/\partial\mu = 2\beta\mathcal R\tanh^{-1}(\mu/\beta)$, so stationarity gives
$2\beta\mathcal R\tanh^{-1}(\mu/\beta) + B^T\nabla V = 0$. ✓

**Constraint note (A2).** $\tanh$ acts per component, so (15) guarantees $|\mu_k|<\beta$ for each $k$, i.e.
$\|\mu\|_\infty<\beta$. That means $\|\mu\|_2<\sqrt3\beta$, which is weaker than the stated $\|\mu\|_2\le\beta$.
The same holds for $u$ and $\gamma$.

### 5.4 Policy iteration — Alg. 1, (17)–(18) [model-based, reference only]

**[Paper]** Evaluate: $\delta^TQ\delta + U(\mu^{(n)}) + \nabla V^{(n+1)T}(A+B\mu^{(n)}+C)=0$ (17).
Improve: $\mu^{(n+1)}=-\beta\tanh\big(\tfrac{1}{2\beta}\mathcal R^{-1}B^T\nabla V^{(n+1)}\big)$ (18).
Stop when $\|\mu^{(n+1)}-\mu^{(n)}\|<\varrho_1$.

**Code [Eng, optional].** `rl/model_based_pi.py`. Only as a cross-check on a simplified, known-model case, if
needed. It is not part of the reproduced pipeline.

### 5.5 Off-policy integral Bellman equation — (19)–(22)

**[Paper]** Write the dynamics under an arbitrary behaviour policy $\mu$ (19). Along those trajectories,
$\dot V^{(n+1)} = -2\beta\tanh^{-1}(\mu^{(n+1)}/\beta)^T\mathcal R(\mu-\mu^{(n)}) - \delta^TQ\delta - U(\mu^{(n)})$
(20–21). Integrating over $[t,t+\Delta T]$ gives
$$V^{(n+1)}(\delta(t+\Delta T)) - V^{(n+1)}(\delta(t)) = -\int_t^{t+\Delta T}\!\!2\beta\tanh^{-1}(\mu^{(n+1)}/\beta)^T\mathcal R(\mu-\mu^{(n)})d\tau - \int_t^{t+\Delta T}\!\!\big(\delta^TQ\delta+U(\mu^{(n)})\big)d\tau\quad(22)$$

**Explanation [Derived check].** From (18), $\nabla V^{(n+1)T}B = -2\beta\tanh^{-1}(\mu^{(n+1)}/\beta)^T\mathcal R$. This
is the step that removes $B$ (and with it $m_j$). Eq. (17) removes $A$ and $C$. So (22) contains no model
quantities: only measured $\delta(\tau)$, the applied $\mu(\tau)$, and the unknowns $V^{(n+1)}$ and $\mu^{(n+1)}$.
One batch of data, collected once with any sufficiently exciting behaviour policy, serves every iteration. That is
what "off-policy" means here.

### 5.6 Actor–critic parameterisation and least squares — (23)–(29), Alg. 2–3

**[Paper]**
- Critic: $\hat V^{(n)}(\delta)=\varpi_{c}^{(n)T}\sigma_c(\delta)$, $\sigma_c\in\mathbb R^{l_c}$ (23).
- Actor: $\hat\xi^{(n)}_{k}(\delta)=\varpi_{a,k}^{(n)T}\sigma_{a,k}(\delta)$, $k=1,2,3$, $\sigma_{a,k}\in\mathbb R^{l_a}$ (24).
- Policy: $\hat\mu^{(n)} = -\beta\tanh(\hat\xi^{(n)})$ (25).
- Utility: $U(\hat\mu^{(n)}) = 2\beta^2\hat\xi^{(n)T}\mathcal R\tanh\hat\xi^{(n)} + \beta^2\bar{\mathcal R}\ln(\mathbf 1_3-\tanh^2\hat\xi^{(n)})$ (26).
- Bellman residual (27), its linear form (28) $\varepsilon = \mathcal N^{(n)}\varpi^{(n+1)} - \mathcal T^{(n)}$, and
  least squares (29) $\varpi^{(n+1)} = (\bar{\mathcal N}^T\bar{\mathcal N})^{-1}\bar{\mathcal N}^T\bar{\mathcal T}$,
  stacked over $\mathcal M_j$ samples.

**[Derived] explicit form of (27)–(28).** The paper writes double sums with $r_{j,k_1,k_2}$, a symbol it never
defines (A3). We read it as the $(k_1,k_2)$ entry of $\mathcal R_j$. Since $\mathcal R_j$ is diagonal, only
$k_1=k_2=k$ survive. We substituted (23)–(26) into (22) ourselves, using
$\tanh^{-1}(\hat\mu^{(n+1)}/\beta)=-\hat\xi^{(n+1)}$ and $\hat\mu^{(n)}=-\beta\tanh\hat\xi^{(n)}$. The result matches
every term of (27) under that reading:
$$\underbrace{\eta_v^T}_{1\times l_c}\varpi_c^{(n+1)} + \sum_{k=1}^3\underbrace{2\beta r_k\!\int_t^{t+\Delta T}\!\!\big(\mu_k+\beta\tanh\hat\xi^{(n)}_k\big)\sigma_{a,k}^T d\tau}_{\bar\eta_k\;(1\times l_a)}\;\varpi_{a,k}^{(n+1)} \;=\; \underbrace{\eta_\delta + 2\beta^2\sum_k r_k\!\int\!\tanh(\hat\xi_k^{(n)})\,\sigma_{a,k}^T\varpi_{a,k}^{(n)}d\tau + \beta^2\sum_k r_k\!\int\!\ln(1-\tanh^2\hat\xi_k^{(n)})d\tau}_{\mathcal T^{(n)}\;(\text{scalar})}$$
with $\eta_v = \sigma_c(\delta(t))-\sigma_c(\delta(t+\Delta T))$ and $\eta_\delta=\int\delta^TQ\delta\,d\tau$. Each
data window gives one row of length $l_c+3l_a$. The unknown vector is
$\varpi^{(n+1)}=[\varpi_c;\varpi_{a,1};\varpi_{a,2};\varpi_{a,3}]$, which requires
$\mathcal M_j\ge l_c+3l_a$ windows and a full-rank $\bar{\mathcal N}$ (persistent excitation, A17). The integrands
of $\mathcal T$ and $\bar\eta_k$ depend on $\varpi^{(n)}$, so they must be **re-integrated from the stored
trajectory at every iteration**. Store $\delta$ and $\mu$ at the sub-step resolution, not just at window ends (A18).

| Symbol | Meaning | Shape | Sec. VI |
|---|---|---|---|
| $\sigma_c$, $\sigma_{a,k}$ | polynomial bases | $(l_c,)$, $(l_a,)$ | not given (A16) |
| $\varpi_c$, $\varpi_{a,k}$ | critic / actor weights | $(l_c,)$, $(l_a,)$ | random init (A17) |
| $\Delta T$ | integral window | scalar, s | 0.05 |
| $\mathcal M_j$ | number of windows | int | not given (A17) |
| $\varrho_{3j}$ | stop tolerance on $\|\varpi^{(n+1)}-\varpi^{(n)}\|$ | scalar | 1e-5 |
| $\beta_j$ | input bound | scalar, N | 5 (A1, A2) |
| $\mathcal Q_j$ | state weight | (6,6) | $10I_6$ |
| $\mathcal R_j$ | input weight | (3,3) | $8I_3$ |

**Code.** Written once and generic, so the same code serves (27)–(29) and (39)–(42):
- `rl/basis.py` (built, Phase 1.4a): `PolynomialBasis(dim, degrees)` with `__call__(x)` ((n,)→(l,) or (N,n)→(N,l)), `jacobian(x)`, `labels()`, `count(dim, degrees)`; `weights_from_quadratic_form(basis, P)` for the LQR check. The degree sets are chosen in 1.4b/1.5 ([Eng], A16): the critic should have no constant term (because $V(0)=0$), and the actor an odd/linear part. Phase 1.5 (D25): `MonomialBasis(dim, exponents)` base class, `grouped_basis(dim, groups, degrees)` (monomials within one variable group only), `axis_groups(3)` = [(0,3),(1,4),(2,5)]; `IRLProblem.actor` takes one shared basis or m per-channel bases of equal size, as in (24).
- `rl/utility.py` (built, Phase 1.4a): `utility_closed_form(mu, bound, r)` (13)/(34), `utility_numeric(...)` (11) for tests, `utility_gradient(mu, bound, r)` ($\partial U/\partial\mu$), `policy_from_xi(xi, bound)` (15)/(25)/(36)/(42), `xi_from_value_gradient(gradV, B, bound, r)` (15)/(18) model-based (tests only), `utility_from_xi(xi, bound, r)` (26), evaluated with the stable form $\ln(1-\tanh^2\xi)=-2\ln\cosh\xi$ (D22).
- `rl/data.py` (built, Phase 1.4b): `OffPolicyDataset(delta (M,W+1,n), mu (M,W,m), h)` with `from_trajectory(delta, mu, h, W)` and `concatenate`; `rollout(f, x0, behaviour, T, h, observe)` (RK4, ZOH input; plant-agnostic through `observe`); `sum_of_sinusoids(...)` exploration signal (A17). Stores the fine grid (A18).
- `rl/offpolicy_irl.py` (built, Phase 1.4b): `IRLProblem(critic, actor, bound, r, Q)`; `bellman_rows(data, problem, W_a_n) -> (N_bar, T_bar)` (27)–(28); `solve_weights` (29) by SVD `lstsq`, returning rank, condition number, RMS residual; `train(data, problem, W_a0, tol, max_iter)` (Alg. 3 Steps 2–4, raises if $\bar{\mathcal N}$ is rank-deficient); `critic_value`, `actor_xi`, `policy` (23)–(25). One actor basis shared by all channels (D23). The same functions serve the attitude loop through `(bound, r, Q)`.
- `controllers/visual_servo.py`: `VisualServoController(weights, basis, beta).mu(delta)` (Alg. 3 Step 5).

**Tests.** $U$ closed form matches numerical quadrature of (11); $\partial U/\partial\mu$ is consistent with (15)
by finite difference. **Known-answer test for the solver:** on a linear system with a quadratic cost and a very
large $\beta$ (so $\tanh$ is near-linear), the learned weights must reproduce the LQR solution from
`scipy.linalg.solve_continuous_are`. The Bellman residual of the converged weights must be small on
held-out windows.

### 5.7 Closed loop and Theorem 2 — (30)–(32)

**[Paper]** $\dot\delta = A + B\hat\mu^{(n)} + C$ (30). With $\varepsilon_\mu = \hat\mu^{(n)}-\mu^{(n)}$, the loop
is asymptotically stable if $\nabla V^{(n+1)T}B\varepsilon_\mu < \delta^TQ\delta + U(\mu^{(n)})$ (31), since then
$\dot V^{(n+1)} = \nabla V^{(n+1)T}B\varepsilon_\mu - \delta^TQ\delta - U(\mu^{(n)}) < 0$ (32).

**Code/tests.** The check is post hoc: evaluate (31) along simulated trajectories. Note that $\nabla V^T B$
requires $m_j$, which is fine for a diagnostic.

### 5.8 Phase 1.5 learning problem for the paper's followers (ideal attitude, D24) — derived 2026-11-12

**Setting.** [Paper] (10)–(13) with Sec. VI values; [Eng] D14 (gravity compensation), D15 (exact $\dot e$),
D19 (Way 1 features), D24 (ideal attitude: the applied thrust vector $R_jf_{Tj}e_3=\mu_j+m_jge_3$ enters the
translational part of (1) directly).

| Symbol | Meaning | Shape | Units | Frame |
|---|---|---|---|---|
| $e_{ij}=s_{ij}-s^*_{ij}$ | feature error, $s={}^vq_{ij}/x^*$, $s^*=\Delta_{ij}/x^*$ (A10) | (3,) | – | ${}^v\mathcal F_j$ (∥ inertial) |
| $\hat v_{ij}$ | observer (6) estimate of $v_{ij}=v_i-v_j$ | (3,) | m/s | ${}^v\mathcal F_j$ |
| $\delta_j=[e_{ij};\hat v_{ij}]$ | learning state | (6,) | mixed | — |
| $\mu_j$ | learned input = physical $R_jf_{Tj}e_3-m_jge_3$ (D14); $|\mu_{j,k}|<\beta_j=5$ (A2) | (3,) | N | inertial |
| $x^*$ | desired depth = 4 | scalar | m | — |
| $m_j$ | 1.3 (j = 2), 2.0 (j = 3) | scalar | kg | — |
| $\mathcal Q_j=10I_6$, $\mathcal R_j=8I_3$ | cost weights | (6,6), (3,3) | – | — |

**True dynamics of $\delta$ (from (1), (5), (6), D14, leader acceleration $\dot v_i=0$ in Sec. VI).**
$$\dot e = \frac{\hat v+\tilde v}{x^*},\qquad \dot{\hat v} = \frac{\mu}{m_j} + \underbrace{\bar\zeta - \tilde e/x^* + \kappa\rho}_{\text{observer correction}},\qquad \rho=\tilde v/x^*$$
This is (10) with $C$ cancelled by D14. The observer error dynamics (7) do not contain $\mu$ (the observer is
fed the true thrust), so $(\tilde e,\tilde v)$ decay on their own (Phase 1.3: $|\tilde v|<10^{-3}$ after 4.5 s,
pair 12). Once they have decayed, $\delta$ follows exactly
$$\dot e = \hat v/x^*,\qquad \dot{\hat v}=\mu/m_j \qquad\text{(three decoupled double integrators).}$$

**Consequence 1: the optimal value separates by axis.** Dynamics, $\mathcal Q$, $\mathcal R$ and $U$ (13) are all
sums over axes $k=1,2,3$ of terms in $(e_k,\hat v_k,\mu_k)$ only. Hence $V^*(\delta)=\sum_k V^*_1(e_k,\hat v_k)$ with
the same 2-D function $V^*_1$ for every axis, and $\xi^*_k = \frac{1}{2\beta r m_j}\,\partial V^*_1/\partial\hat v_k$
depends only on $(e_k,\hat v_k)$. This exact structure is the guide for the basis choice (A16, Step 2).

**Consequence 2: known answer near $\delta=0$.** There $U\approx\mu^T\mathcal R\mu$ and $\tanh\approx$ identity, so
the learned controller must approach the LQR solution of the double integrator (per axis, $A=[[0,1/x^*],[0,0]]$,
$B=[0;1/m_j]$, $Q=10I_2$, $R=8$):

| Follower | $P$ (per axis) | $K$ | $\xi\approx K\delta_k/\beta$ coefficients | closed-loop poles |
|---|---|---|---|---|
| 2 ($m=1.3$) | [[50.30, 11.63], [11.63, 14.62]] | [1.118, 1.406] | [0.224, 0.281] | −0.26, −0.82 |
| 3 ($m=2.0$) | [[55.06, 17.89], [17.89, 24.62]] | [1.118, 1.539] | [0.224, 0.308] | −0.29, −0.48 |

**Consequence 3: realisability (for 1.6).** Any $\mu$ with $|\mu_k|<5$ is a valid thrust vector: the physical
$z$-component $m_jg+\mu_z\ge12.75-5>0$, and the tilt is at most $\arctan(5\sqrt2/7.75)\approx42°$ (j = 2), $\approx27°$ (j = 3).

**Consequence 4: the stated initial conditions barely use the constraint (A27).** Scratch check (2026-11-12;
LQR-limit tanh policy $\mu=-\beta\tanh(K\delta/\beta)$, real observer (6), exact $\dot e$, RK2, $h=1$ ms):
$\max|\mu_k| = 0.53$ N (j = 2, $e(0)=[-0.475,-0.425,0]$) and 0.83 N (j = 3, $e(0)=[-0.75,-0.375,0]$), against
$\beta=5$; $\|e\|<0.05$ after 9.0 s / 11.4 s. Fig. 4 saturates at ±5. So the saturation in Fig. 4 cannot come from
the stated initial conditions (consistent with A11). Plan: train on a data region that includes large errors,
validate on (i) the stated initial conditions and (ii) an [Eng] large-error scenario that drives $\mu$ into saturation.
**Resolved (Phase 1.5):** near $\delta=0$ the learned $\xi_k\approx0.224\,e_k+\dots$, so $|\mu_k|\to\beta$ ($\tanh\ge0.99$) needs
$|e_k|\gtrsim12$; Fig. 7 plots $e_z(0)\approx12$ for pair 12. Fig. 4's saturation therefore matches the plotted (not the
stated) initial errors (A11).

---

## 6. Attitude controller — Sec. V, eqs. (33)–(42), Alg. 4 → `src/vsfc/controllers/attitude.py` (+ shared `rl/`)

**[Paper] (33)** $\dot z_j = F_j(z_j)+G_ju_j$, with $z_j=[\Theta_j;\omega_j;\Theta_{rj};\omega_{rj}]\in\mathbb R^{12}$,
$u_j=f_{\tau j}$, $\|u_j\|\le\gamma_j$,
$$F_j = \begin{bmatrix}\omega_jI_3\\ -J_j^{-1}(\omega_j\times J_j\omega_j)\\ \chi_j(\Theta_{rj},\omega_{rj})\end{bmatrix},\qquad G_j=\begin{bmatrix}0_{3\times3}\\J_j^{-1}\\0_{6\times3}\end{bmatrix}$$
Here $\chi_j\in\mathbb R^6$ is the unknown reference generator. The first row makes $\dot\Theta=\omega$, a
small-angle approximation of the Euler kinematics (A13).

**[Paper] (34)–(42)** are the same construction with the substitutions
$\delta\to z$, $\mu\to u$, $\beta\to\gamma$, $\mathcal Q\to\mathcal S$, $\mathcal R\to\mathcal D$,
$V\to\mathcal V$, $\sigma\to\varphi$, $\varpi\to\vartheta$, $\xi\to\varsigma$:
utility (34), Hamiltonian (35), optimal policy $u^*=-\gamma\tanh(\tfrac1{2\gamma}\mathcal D^{-1}G^T\nabla\mathcal V^*)$
(36), rewritten dynamics (37), derivative (38), integral Bellman equation (39), critic (40), actor (41),
policy (42), and the residual $o_j^{(n)}$ (unnumbered; its first $\tanh$ has a typo, A20). Algorithm 4 mirrors Algorithm 2.

| Symbol | Meaning | Shape | Units | Sec. VI |
|---|---|---|---|---|
| $\Theta_{rj},\omega_{rj}$ | reference attitude / rate from the outer loop | (3,), (3,) | rad, rad/s | mapping missing (A15) |
| $u_j=f_{\tau j}$ | torque | (3,) | N·m | $\gamma=1$ |
| $\mathcal S_j$ | state weight | (12,12) | — | $10I_{12}$ |
| $\mathcal D_j$ | input weight | (3,3) | — | $8I_3$ |
| $\varphi_c,\varphi_{a,k}$; $\vartheta_c,\vartheta_{a,k}$ | bases; weights | $(\iota_c,)$, $(\iota_a,)$ | — | not given (A16) |

**Code.** `AttitudeController(weights, basis, gamma).u(z)` (42); training uses `rl/offpolicy_irl.py` with
`bound=γ, Rdiag=diag(D), Q=S`.

**Tests.** Same structure as Sec. 5. Additionally, holding a constant $\Theta_r$ drives the attitude error to zero
on plant (1) with $|u_k|<\gamma$.

### 6.1 Phase 1.6 learning problem in D29 coordinates — derived 2026-11-13

**Setting.** [Paper] (33)–(42) with Sec. VI values; [Eng] D29 (error coordinates), D30 (reference from
`thrust_mapping.py`), A13 (plant integrates $R$ exactly; $\Theta$ ZYX).

| Symbol | Meaning | Shape | Units | Frame |
|---|---|---|---|---|
| $e_\Theta=\mathrm{wrap}(\Theta-\Theta_r)$ | attitude tracking error (Fig. 8) | (3,) | rad | Euler (ZYX) |
| $e_\omega=\omega-\omega_r$ | rate tracking error | (3,) | rad/s | body |
| $z=[e_\Theta;e_\omega;\Theta_r;\omega_r]$ | learning state (D29) | (12,) | mixed | — |
| $u=f_\tau$ | torque, $\|u_k\|<\gamma=1$ | (3,) | N·m | body |
| $J=\mathrm{diag}(0.0082,0.0080,0.0145)$ | inertia, **same for both followers** | (3,3) | kg·m² | body |
| $\mathcal S=10I_{12}$, $\mathcal D=8I_3$ | cost weights | (12,12), (3,3) | — | — |

**True dynamics of the error (from (1), A13, D30).**
$$\dot e_\Theta = W(\Theta)\omega - W(\Theta_r)\omega_r,\qquad
\dot e_\omega = J^{-1}u \;\underbrace{-\,J^{-1}(\omega\times J\omega) - \dot\omega_r}_{d(t)}.$$
With the paper's small-angle model ($W=I$ in (33)) the first equation is $\dot e_\Theta=e_\omega$. The
term $d$ does not contain $u$. Its size in the case study is small: $\|J\omega\times\omega\|\sim J\|\omega\|^2\approx10^{-3}$ N·m for
$\|\omega\|\approx0.3$ rad/s (Step 1 test signal), against $\gamma=1$. The reference block
$(\Theta_r,\omega_r)$ evolves as the outer loop dictates (the numerical $\chi_j$, D30) and is **not** influenced by $u$
($G$ has zeros there).

**Consequence 1: the reference block of the cost does not change the policy.** $z^T\mathcal Sz=10(\|e_\Theta\|^2+\|e_\omega\|^2)
+10(\|\Theta_r\|^2+\|\omega_r\|^2)$. The second part does not depend on $u$ and adds a policy-independent term to
$\mathcal V$, whose gradient along $G$ (the $e_\omega$ rows) is zero. So (36) is unchanged if that part is left out of the learning.

**Consequence 2: neglecting $d$ and $W\neq I$, the problem separates by axis** into three double integrators
$\dot e_{\Theta k}=e_{\omega k}$, $\dot e_{\omega k}=u_k/J_k$ with cost $10(e_{\Theta k}^2+e_{\omega k}^2)+U_k(u_k)$. Then
$\mathcal V^*=\sum_k\mathcal V^*_k(e_{\Theta k},e_{\omega k})$ and $\varsigma^*_k$ depends only on $(e_{\Theta k},e_{\omega k})$. This is the same
structure as §5.8, and it is the guide for the basis choice (A16, attitude).

**Consequence 3: known answer near $z=0$** (per axis, $A=[[0,1],[0,0]]$, $B=[0;1/J_k]$, $Q=10I_2$, $R=8$; $\varsigma\approx K x/\gamma$):

| Axis | $J_k$ | $K$ | $P_{11}$, $2P_{12}$, $P_{22}$ | closed-loop poles |
|---|---|---|---|---|
| roll | 0.0082 | [1.1180, 1.1262] | 10.073, 0.147, 0.0739 | −1.00, −136.3 |
| pitch | 0.0080 | [1.1180, 1.1260] | 10.071, 0.143, 0.0721 | −1.00, −139.8 |
| yaw | 0.0145 | [1.1180, 1.1324] | 10.129, 0.259, 0.1314 | −1.00, −77.1 |

The dominant pole is −1 rad/s on every axis (time constant 1 s), so $e_\psi$ goes from 0.174 rad to 0.024 rad in 2 s.
That is consistent with Fig. 8. $\|K x\|<1$ unless $|e_\Theta|$ or $|e_\omega|\gtrsim0.9$, so saturation happens only in brief transients (Fig. 4c,d).

**Consequence 4: Fig. 2(c,d) size.** About 6–7 critic traces (final ≈ 0.17, 0.03, ≈0 ×3–4, −1.65) and about 6–7 actor traces, all
positive (final ≈ 2.28, 1.35 ×2, 1.1, 0.75, 0.48 ×2), settled after about 4 iterations. A full polynomial in $\mathbb R^{12}$ would have
78 quadratic terms alone. The figure therefore suggests per-axis bases in $(e_{\Theta k},e_{\omega k})$: a linear actor gives exactly 6
actor weights. Individual values cannot be matched (basis ordering and scaling unknown). The critic's −1.65 is not a
diagonal quadratic term of a positive-definite $\mathcal V$, so their critic basis is not simply ours (A16).

**Consequence 5: time-scale separation is weak (A29).** The inner loop's dominant pole (−1) is only about 1.2–4× faster than
the outer loop's (−0.26, −0.82 for j = 2; −0.29, −0.48 for j = 3, §5.8). The attitude lag may visibly change Fig. 4a,b
and the observer behaviour (A26) once the loops are cascaded.

---

## 7. Cascade and case study — Sec. VI → `src/vsfc/controllers/thrust_mapping.py`, `src/vsfc/sim/`, `config/paper_case_study.yaml`

**Missing link (A15).** The paper does not give the map from $\mu_j$ to $(f_{Tj},\Theta_{rj},\omega_{rj})$.
**[Eng] D30 (built 2026-11-13, `controllers/thrust_mapping.py`).** $F=\mu+m_jge_3$ (D14), $f_T=\|F\|$,
$b=F/\|F\|$, $\bar b=R_z(\psi_r)^Tb=[\sin\theta\cos\phi,\,-\sin\phi,\,\cos\theta\cos\phi]$ (ZYX),
$\phi_r=\arcsin(-\bar b_y)$, $\theta_r=\operatorname{atan2}(\bar b_x,\bar b_z)$, $\psi_r=0$;
$\omega_r=W(\Theta_r)^{-1}\dot\Theta_r$ with $\dot\Theta_r$ from the D20 filter ($\tau=0.02$ s, config
`engineering.ref_rate_tau`). This is the numerical $\chi_j$. Error coordinates (D29): `tracking_error` gives
$e_\Theta=\mathrm{wrap}(\Theta-\Theta_r)$, $e_\omega=\omega-\omega_r$.
Functions: `thrust_vector`, `decompose_thrust`, `thrust_mapping`, `wrap_angle`, `tracking_error`,
`ReferenceGenerator(mass, g, dt, tau, psi_r).reset(mu0)/.update(mu) -> AttitudeReference(f_T, theta_r, theta_r_dot, omega_r)`.
Validation: round trip $f_TR(\Theta_r)e_3=F$ to $10^{-12}$; tilt bound $\arcsin(\beta/m_jg)$ reached; $\omega_r$ vs an
independent $R^T\dot R$ truth: 0.07 % error (raw difference), 2.8 % ($\tau=0.02$ s lag) of $\max\|\omega_r\|=0.29$ rad/s.

**Sec. VI parameters [Paper]** (to go into `config/paper_case_study.yaml`):

| Item | Value | Note |
|---|---|---|
| $m_1,m_2,m_3$ | 1, 1.3, 2 kg | |
| $J_1=J_2=J_3$ | diag(0.0082, 0.0080, 0.0145) | units, A21 |
| $\lambda_2=\lambda_3$ | 3.2 mm | |
| $h_1$ | [0,0,1] m | frame, A19 |
| $p_1(t)$ | [2.5+0.4t, −1+0.5t, 0.4t] m | constant velocity → $\dot v_1=0$ (A23) |
| $\Delta_{12},\Delta_{13}$ | [4,1,1], [4,−1,1] m | definition, A10 |
| $p_2(0),p_3(0)$ | [0.4,−0.3,0], [1.5,1.5,0] m | |
| $\Theta_2(0)=\Theta_3(0)$ | [0,0,0.174] rad | |
| $\beta$, $\gamma$ | 5, 1 | A1, A2 |
| $\alpha_{12},\alpha_{13}$ | 5, 5 | |
| $\kappa_{12},\kappa_{13}$ | 5, 55 | A6 |
| $\epsilon_{12},\epsilon_{13}$ | 0.5, 0.5 | |
| $\Delta T$ | 0.05 s | |
| $\varrho_{32},\varrho_{33},\varrho_{42},\varrho_{43}$ | 1e-5 | |
| $\mathcal Q$, $\mathcal R$, $\mathcal S$, $\mathcal D$ | $10I_6$, $8I_3$, $10I_{12}$, $8I_3$ | |
| Duration | 50 s | read from figure axes |

**What the figures say (targets for Phase 1 comparison).** Fig. 2: weights converge in about 4–6 iterations,
and the plots run to 11. Panels (a,b) have roughly 20 visible traces each; (c,d) have about 6 or fewer
(approximate counts). Fig. 4: $\mu$ components saturate at ±5 at first, then go to 0 within about 3 s; $u$
reaches ±1 briefly. Fig. 5: observer errors go to 0 within about 5 s. Figs. 7–9: feature, attitude and
relative-position errors settle within about 5–15 s. Fig. 8: $e_\psi(0)\approx0.17$, decaying to 0. Fig. 10
(baseline [16] with clipping) is slower and oscillatory.

**Code.**
- `sim/scenario.py`: `leader_trajectory(t)` (kinematic leader, [Eng]); `load_case_study(yaml)`.
- `sim/closed_loop.py`: RK4 at a fixed step $h$ ([Eng], A18). Plant (1) ×2, observer (6) ×2, controllers (25)
  and (42) ×2, mapping (A15); logs every signal shown in Figs. 3–9.
- `scripts/run_case_study.py`.

---

## 8. Equation index

| Eq. | Content | Module / function |
|---|---|---|
| (1) | quadrotor dynamics | `dynamics.quadrotor_derivatives` |
| unnum. | $q_i$, ${}^vq_{ij}$ | `camera.sphere_centre`, `camera.relative_position_virtual` |
| (2) + unnum. | projection, interaction matrix | `camera.project_point`, `camera.point_interaction_matrix` |
| unnum. | moments $m^{pq}$ | `camera.sphere_image_moments` |
| (3) + unnum. | features, $\dot s=v/x^*$ | `camera.moment_features`, `camera.ideal_features` |
| (4) | feature error | `camera.feature_error` |
| (5) | leader–follower image dynamics | `leader_follower.relative_dynamics` |
| (6) + sat | observer | `observer.AdaptiveVelocityObserver.derivatives`, `observer.sat` |
| (7)–(9), Thm 1 | error dynamics, Lyapunov | `observer.error_dynamics`, `.lyapunov`, `.gain_condition` (tests) |
| (10) | learning system | documented; not evaluated by the learner |
| (11)–(13) | constrained utility | `rl.utility.utility_closed_form`, `utility_numeric` |
| (14)–(16) | Hamiltonian, $\mu^*$, HJB | `rl.utility.policy_from_xi`; HJB residual diagnostic |
| (17)–(18), Alg. 1 | model-based PI | `rl.model_based_pi` (optional cross-check) |
| (19)–(22) | off-policy integral Bellman eq. | `rl.offpolicy_irl.bellman_rows` |
| (23)–(26) | critic, actor, policy, utility | `rl.basis`, `rl.utility.utility_from_xi`, `controllers.visual_servo` |
| (27)–(29), Alg. 2–3 | residual, regression, LS | `rl.offpolicy_irl.bellman_rows`, `solve_weights`, `train` |
| (30)–(32), Thm 2 | closed loop, stability condition | `sim.closed_loop`; Thm 2 check in `scripts/` |
| (33) | augmented attitude system | `controllers.attitude` (data from plant (1)) |
| (34)–(36) | attitude utility, Hamiltonian, $u^*$ | `rl.utility` (shared) |
| (37)–(39), Alg. 4 | attitude integral Bellman eq. | `rl.offpolicy_irl` (shared) |
| (40)–(42) + $o_j$ | attitude critic/actor/policy, residual | `rl.basis`, `controllers.attitude` |

---

## 9. Ambiguities and gaps

Status: **Open** = needs a decision; **Proposed** = we have a recommendation, pending sign-off when the component
is built; **Note** = no action, but it must be stated in the thesis.

| ID | Issue | Evidence | Proposed resolution | Status |
|---|---|---|---|---|
| A1 | Gravity vs $\beta$: hover needs $\|\mu\|=m_jg$ = 12.75 / 19.6 N > $\beta$=5; with $C=-ge_3$, $\delta=0$ is not an equilibrium inside $\Omega_\mu$ and the infinite-horizon cost diverges | (5), (10), Fig. 4 shows $\mu\to0$ | [Eng] Learn $\mu$ as the gravity-compensated input: applied thrust vector $=\mu+m_jge_3$ (so $C=0$ for learning). Uses $m_j$, which the observer already needs. Bound applies to the deviation. | **Decided (D14)** |
| A2 | Per-component $\tanh$ gives $\|\mu\|_\infty<\beta$, not $\|\mu\|_2\le\beta$ | (15), (25), Fig. 4 shows components up to ±5; Fig. 4b inset at t ≈ 0.1 s: one component ≈ +5 and another ≈ +3 at once, so $\|\mu_3\|_2\approx5.8>\beta$ (2026-11-12) | Implement the formula as printed; report both norms; note in thesis | Proposed (evidence: the paper's own bound is per component) |
| A3 | $r_{j,k_1,k_2}$ undefined in (27)–(29) | — | $r_{j,k_1,k_2}:=[\mathcal R_j]_{k_1k_2}$; diagonal ⇒ explicit form in §5.6, derived from (22); checked against the PDF and by the LQR known-answer test (D23) | Decided |
| A4 | Observer needs $\dot e$; $\rho=\dot e-\hat v/x^*$, and exact $\dot e$ already gives $v=x^*\dot e$ | (6), §4 derivation | Pluggable source: (a) exact $\dot e$ from simulator (idealised; possibly what the figures used), (b) filtered finite difference of sampled $e$. Reproduce with (a), report (b) as a check | **Decided (D15)** |
| A5 | Boundary layer ⇒ (9) holds only outside $|\rho_k|<\epsilon$; convergence is to a neighbourhood. $\hat\zeta$ only grows (no leakage). $\tilde\zeta\to0$ is not shown | (6), (9) | Note; test for bounded convergence, not exact | Note |
| A6 | $\kappa_{13}=55$ vs $\kappa_{12}=5$ (possible typo) | Sec. VI | Use as printed; sensitivity check | Proposed |
| A7 | (10) is not autonomous in $\delta$; first block uses $\hat v$ instead of $v$; $C_j$ written as a matrix | (10) | Note; learner is unaffected (data-driven) | Note |
| A8 | $\dot s=v/x^*$ exact only for a small sphere near the optical axis | §2 derivation | Way 1 `ideal_features` everywhere; `moment_features` uses disc moments; exact ellipse not modelled — stated assumption | **Decided (D19)** |
| A9 | Sphere radius, $a^*$ not given; $x^*$ not stated | Sec. VI | $x^*=4$ m from $\Delta_x$; radius and $a^*$ needed only for the moment check, e.g. $r=0.1$ m [Eng] | Proposed |
| A10 | Is $\Delta_{ij}$ desired $p_i-p_j$ or $q_i-p_j$? | Fig. 9: $\Delta_{z}(0)\approx0$ matches $q_1-p_2$ (z: 1−1=0), not $p_1-p_2$ (z: 0−1) | $\Delta_{ij}$ = desired ${}^vq_{ij}$ ⇒ $s^*_{ij}=\Delta_{ij}/x^*$ | Proposed |
| A11 | Initial values in Figs. 7 and 9 cannot be reproduced from stated positions. E.g. pair 12: expected $e(0)\approx[-0.48,-0.43,0]$ (under A10) vs plotted ≈[3.2, −1.45, ~0 then spike to 12]; expected $\Delta(0)=[-1.9,-1.7,0]$ vs plotted ≈[1.15, −0.7, 0] | Figs. 7, 9 | Validate on qualitative behaviour (bounds, convergence times, shapes), not on initial values; record in thesis | Open |
| A12 | Frame: (1) is z-down, but $p_{1z}=0.4t$ (descending) and Fig. 3 draws $p_z$ upward | (1), Fig. 3 | Follow (1) literally; plot with an explicit axis label | Proposed |
| A13 | Euler convention unstated; $F$ uses $\dot\Theta=\omega$ (small angle) | (1), (33) | Plant integrates $R$ exactly; Euler ZYX for reporting and $\Theta_r$; learner only sees data | Proposed |
| A14 | Attitude cost $z^T\mathcal Sz$ penalises $\Theta_r,\omega_r$ themselves (not controllable); finite only if references → 0. Fig. 8 plots tracking errors; Fig. 2(c,d) has only ~6 weights, far below a full polynomial in $\mathbb R^{12}$. **Added 2026-11-13:** in literal $z$ the $(\Theta,\omega)$ rows of (33) do not depend on $(\Theta_r,\omega_r)$, $\chi$ does not depend on $(\Theta,\omega,u)$, and $\mathcal S=10I_{12}$ is block-diagonal, so $\mathcal V^*$ splits into an attitude part plus a reference part and $u^*$ in (36) cannot depend on $\Theta_r$. The result would level the vehicle and never track, which contradicts the Sec. V aim ("attitude tracking of the reference commands") and the outer loop's need to tilt. With error coordinates $[\Theta-\Theta_r;\omega-\omega_r;\Theta_r;\omega_r]$, $G$ is unchanged, $\dot e_\omega$ contains $\dot\omega_r$, and the reference terms stay finite here because $\Theta_r,\omega_r\to0$ for a constant-velocity leader | (33)–(36), Sec. V intro, Figs. 2, 8 | Decide when implementing Sec. V: literal $z$ vs tracking-error coordinates $[\Theta-\Theta_r,\omega-\omega_r,\dots]$ | **Decided (D29)**: tracking-error coordinates |
| A15 | $\mu\to(f_T,\Theta_r,\omega_r)$ mapping, $\psi_r$, $\omega_r$, $\chi_j$ not given | Sec. V | Standard thrust-vector decomposition; $\psi_r=0$ (Fig. 8: $e_\psi(0)\approx0.174=\psi(0)$); $\omega_r$ from filtered derivative of $\Theta_r$. **Added 2026-11-13:** with D14 the commanded force $\mu+m_jge_3$ has $\|\mu\|\le\beta=5<m_jg$, so $f_T>0$ always and the tilt is at most $\arcsin(\beta/m_jg)$ = 23.1° (Quad 2), 14.8° (Quad 3), reached when $\mu$ is tangent to the sphere around $m_jge_3$ (a horizontal $\mu$ gives only $\arctan$ = 21.4°/14.3°; corrected, E11). This is consistent with the "attitude restrictions" in Sec. VI and with the small-angle $\dot\Theta=\omega$ in (33). The paper calls $\chi_j$ "unknown", which fits a numerically generated reference | **Decided (D30)** |
| A16 | Bases, $l_c$, $l_a$, $\iota_c$, $\iota_a$ not given | Sec. VI ("polynomial combinations"); Fig. 2a ≈ 15–25 critic traces with final values ≈ −47…+31 (large negatives, unlike the all-positive per-axis LQR weights 50.3, 23.3, 14.6 for $e^2, 2ev, v^2$); Fig. 2b ≈ 15–20 actor traces within ±1.7 | Start with quadratic critic / linear+cubic actor; weight-by-weight match with Fig. 2 impossible (basis ordering unknown), so compare iteration count and convergence shape. **Outer loop decided (D25):** per-axis critic deg (2,4) in $(e_k,\hat v_k)$, $l_c=24$; actor channel $k$ deg (1,3) in $(e_k,\hat v_k)$, $l_a=6$. Attitude bases still open (Phase 1.6) | Decided (outer loop) |
| A17 | Behaviour policy, exploration noise, $\mathcal M_j$, data length, initial-weight distribution, admissible initial policy | Alg. 2–4, Sec. VI | Stabilising PD-like policy + sum of sinusoids; $\mathcal M_j\ge2(l_c+3l_a)$; check rank/condition of $\bar{\mathcal N}$. Shown necessary in 1.4b: with on-policy data the actor columns of $\bar{\mathcal N}$ vanish, so exploration must be added to $\mu_j$ | Open (quadrotor values in 1.5) |
| A18 | Plant integration step and quadrature of $\eta$-integrals unspecified | — | RK4, $h=1$ ms; trapezoid over 50 sub-steps per $\Delta T$; $\int\mu\sigma$ exact for ZOH input (D23) | Decided |
| A19 | Initial $v_j$, $\omega_j$, $\hat e$, $\hat v$, $\hat\zeta$ not given; frame of $h_i$; leader attitude not modelled | Sec. VI | Zero velocities and rates; $\hat e(0)=e(0)$, $\hat v(0)=0$, $\hat\zeta(0)=0$; kinematic leader with inertial $h_1$ | Proposed |
| A20 | Typos: $o_j$ first $\tanh$ uses $\varphi$ where $\vartheta$ is meant; $\Omega_\mu\in$ should be $=$; $\chi_{rj}$ vs $\chi_j$ | Sec. V | Use corrected forms (consistent with §5.6 derivation) | Proposed |
| A21 | Units: $\lambda$ in mm, $J$ in kg·m²/rad², implicit unit gains in (6) ($\tilde e/x^*$ added to an acceleration; $\dot{\hat\zeta}$ units) | (6), Sec. VI | SI throughout; use printed numbers as-is | Note |
| A22 | Observer needs $m_j$, $g$, $R_j$, $f_{Tj}$, so the overall scheme is not fully model-free | (6) | Note in thesis | Note |
| A23 | Leader has constant velocity ⇒ adaptive term $\bar\zeta$ never needed in Sec. VI (Limitation 3) | Sec. VI | Reproduce as printed; add accelerating-leader test (our extension, labelled) | Note |
| A24 | Value of $g$ not stated | Sec. VI | $g=9.81$ m/s² | Proposed |
| A25 | Fig. 5: $\|\tilde e_{12}(0)\|\approx8.8$, $\|\tilde e_{13}(0)\|\approx1.7$, but $\|e_{12}(0)\|\approx0.64$, so the paper's $\hat e(0)$ is unknown | Fig. 5 | Keep A19 ($\hat e(0)=e(0)$); compare behaviour, not $t=0$ values | Open |
| A26 | Fig. 5 $\|\tilde v_{12}\|$ oscillates; with exact $\dot e$ and true thrust, (7) is linear with real eigenvalues ($-5.02,-1.23$ for pair 12) and sat only adds damping, so (6) alone cannot oscillate | Fig. 5, (7) | Hypothesis: paper fed commanded $\mu$ (attitude lag) or a differentiated/noisy $\dot e$; test in closed loop | Open |
| A27 | With the stated initial conditions the outer-loop input stays far from the bound: the LQR-limit policy peaks at 0.53 N / 0.83 N vs $\beta=5$, yet Fig. 4 shows saturation at ±5 in the first 0.5 s | Sec. VI, Fig. 4, §5.8 Consequence 4 | Validate on the stated ICs AND on an [Eng] large-error scenario that saturates $\mu$; the data region for learning must cover large errors | Explained: ±5 needs $|e|\gtrsim12$ = Fig. 7's plotted $e_z(0)$ (A11); validated on stated, large-error, near-saturation cases (Phase 1.5) |
| A28 | Sec. VI: NN weights "randomly initialized"; Alg. 1 Step 1 needs an admissible $\mu^{(0)}$ | Phase 1.5: random $N(0,0.1^2)$ actor starts converge per axis to the non-stabilising Riccati solutions (E8) | Random starts drawn from admissible policies (positive per-axis linear gains, cubic 0) (D28) | Decided |
| A29 | Inner/outer time-scale separation is weak: with $\mathcal S=10I$, $\mathcal D=8I$ the attitude loop's dominant pole is −1 rad/s, against −0.26…−0.82 for the outer loop (§5.8, §6.1) | Sec. VI weights; Fig. 8 | Expect attitude lag to change Fig. 4a,b and possibly to explain A26 (Fig. 5 oscillation). Measure in the cascade (Phase 1.6 Step 6); no change to paper weights | Open |
