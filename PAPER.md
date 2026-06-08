# Precision and Controllability Limits of Single-Input N-Link Pendulum Swing-Up and Balance

*A research-session write-up. Single horizontal velocity input at the base pivot;
uniform rods (mass 1, length 1, I_com = 1/12); angles measured from upright.*

---

## Abstract

We study an inverted N-link pendulum actuated **only** by the horizontal velocity
of its base pivot, and ask: given simulation parameters (gravity g, timestep dt),
what **precision** on link-angle measurement (δθ) and velocity command (δv) is
required to (1) balance upright and (2) swing up from hanging? We derive and
verify the closed-form dynamics, build LQR/TVLQR controllers, and measure
precision thresholds for N = 1–5, with a theory connecting them to the
unstable-pole structure (data-rate theorem) and the basin/noise-amplification
geometry. We then push the swing-up frontier: N ≤ 5 succeeds with realistic
angle-only sensing; **N = 6** succeeds only under full-state feedback, and we
remove its dependence on a hand-curated trajectory seed using a
**controllability-aware trajectory optimization** (robust to ±10% initial-state
perturbation); **N = 7** is shown to lie
beyond the architecture, with a diagnosis (a scalar controllability proxy cannot
guarantee all bending modes stay excitable). Headline: required angle precision
tightens ≈ 6–20× per added link; sensing — not actuation — is the binding
constraint; and swing-up controllability, not compute, is the wall at high N.

---

## 1. Problem and model

A chain of N identical uniform rods hangs from a pivot at (x, 0) that moves only
horizontally; x(t) (equivalently the pivot velocity, our control) is prescribed.
θ_i is rod i's angle from the **upright** vertical (θ = 0 up, θ = π hanging).

The only actuation is the pivot's X-velocity, quantized to a grid of step δv; the
controller sees link angles quantized to a grid of step δθ. "Required precision"
is the largest quantization step at which a task still succeeds, found by
log-bisection over seeded trials.

### 1.1 Dynamics (derived, then verified)

Euler–Lagrange gives, with 1-based i,l:

    M(θ) θ̈ = −ẍ b⊙cosθ − (A ⊙ sin(θ_i−θ_l)) θ̇² + g b⊙sinθ
    A_il = N − max(i,l) + 1/2  (i≠l),   A_ii = N − i + 1/4
    M_il = A_il cos(θ_i−θ_l) + δ_il/12,   b_i = N − i + 1/2

The θ-dynamics depend on the pivot **only through its acceleration ẍ**; a velocity
command is realized as ZOH acceleration a = (v_cmd − v)/dt over the step.

**Verification.** Cross-checked against an independent SymPy Lagrangian
derivation (N = 1–5, random states; max error < 1e-8) and energy conservation
under free swing (drift < 5e-9 over 10 s). The N=1 reduction
(1/3)θ̈ = −(1/2)ẍ cosθ + (g/2) sinθ matches a rod pivoting about its end.

---

## 2. Methods

- **Simulator:** RK4 with ZOH pivot acceleration; quantization hooks on angle
  measurement and velocity command.
- **Balance:** discrete LQR about upright; finite-difference and steady-state
  Kalman state estimators.
- **Swing-up:** N=1 energy pumping + LQR catch; N≥2 direct collocation
  (implicit-dynamics trapezoidal, CasADi/IPOPT) for the nominal, TVLQR tracking,
  hand-off to a balance LQR for the catch. Homotopy in N (a new tip link shadows
  the last) warm-starts higher N.
- **Precision protocol:** task succeeds if it stays upright (balance) or settles
  within 0.3 rad for the final 5 s (swing-up) across seeds; threshold = largest
  quantization step that still succeeds.

---

## 3. Results — Balance (Task 1)

### 3.1 Precision thresholds (g = 9.81, dt = 0.01, best controller)

| N | λ_max (1/s) | max δθ (rad) | max δv (m/s) |
|---|------|---------|------|
| 1 | 3.84 | 0.64 | 0.91 |
| 2 | 7.19 | 0.087 | 0.99 |
| 3 | 10.25 | 0.014 | 0.80 |
| 4 | 13.15 | 0.0025 | 0.69 |
| 5 | 15.85 | 4.6·10⁻⁴ | 0.59 |

- **Angle precision tightens ≈ 6.3× per link** (≈ 6.3⁻ᴺ); **velocity precision is
  nearly flat in N.** So for N ≥ 3, **sensing is the binding constraint**, not
  actuation — by N=5 the angle requirement is ~19 bits while velocity is ~13.
- Controller: a lag-free finite-difference estimator with aggressive LQR
  (r = 0.01, q_θ = 100) beat the Kalman filter against deadband (quantization)
  error for N ≤ 4; Kalman edges ahead only at N=5.

### 3.2 Theory: why precision collapses with N

Two multiplicative mechanisms, both measured:

- **Basin of attraction** (largest recoverable alternating tilt) shrinks ≈ 4×/link.
- **Closed-loop noise amplification** κ (the H∞ gain from angle-measurement error
  to true-angle response) grows ≈ 5×/link, because stabilizing more unstable
  poles with one scalar input forces high feedback gain — and high gain on the
  measurement is high gain on its error (a Bode/sensitivity constraint).

Predicted δθ_max ≈ basin/κ matches the empirical trend (≈ 20×/link in the H∞
worst case; ≈ 5–6×/link empirically, since real quantization error is partly
self-dithered, not adversarial). Computed through N=7: κ_θ ≈ 1.7 → 26,000 and
basin 0.70 → 1.4·10⁻⁴ rad as N: 1 → 7.

A complementary information-theoretic floor (data-rate theorem,
N/dt · log₂(Θ/δθ) ≥ Σλ⁺/ln2) sits ~4 orders looser than the basin/κ bound at our
operating point: the binding constraint is **dynamics** (noise amplification
through the unstable plant), not **information**.

### 3.3 Parameter scaling

An exact time-rescaling symmetry (τ = √(g/L)) makes the system at (g, dt)
identical to (g=1, dt√g): everything depends on the single dimensionless group
**u = dt√(g/L)**.

- **δθ_max = Φ_N(u)** — gravity enters *only* through u. Lower g ≡ smaller dt.
- **δv_max = √(gL)·Ψ_N(u)** — a √g prefactor: stronger gravity *loosens* the
  absolute velocity requirement.
- Velocity-command quantization amplification κ_v ≈ 4–5, **flat in N** (input-
  channel disturbances are exactly what feedback rejects) — confirming actuation
  is not the binding constraint.
- Empirically δθ is ~dt-independent (FD) / relaxes with smaller dt (Kalman); δv ∝ dt;
  weak g-dependence. N=5 is uncontrollable at dt = 0.02 (sampling ceiling).

### 3.4 fp32 sufficiency

Because fp32 has *relative* precision (finer near θ=0), it balances even N=7 in
both representations; the binding uniform-quantization limits are ~5·10⁻⁵ rad
(N=6) and ~2·10⁻⁵ rad (N=7) — the basin/κ worst-case bound is ~2 orders
conservative vs broadband quantization.

---

## 4. Results — Swing-up (Task 2)

### 4.1 N = 1–5 (realistic, angle-only sensing) + projected precision floor N = 6–8

| N | swing T (s) | max δθ (rad) | max δv (m/s) |
|---|------|------|------|
| 1 | 0.76 | 0.024 | 0.054 |
| 2 | 3.5 | 0.078 | 0.11 |
| 3 | 5.0 | 0.0085 | 0.34 |
| 4 | 7.0 | 9.8·10⁻⁵ | 0.014 |
| 5 | 12.0 | 6.9·10⁻⁵ | 0.0025 |
| 6 | — | 1.2·10⁻⁷ † | 7.8·10⁻⁵ † |
| 7 | — | 5.5·10⁻⁹ † | 1.4·10⁻⁵ † |
| 8 | — | 2.8·10⁻¹⁰ ‡ | 2.5·10⁻⁶ ‡ |

**† projected, not achieved.** N ≥ 6 swing-up succeeds only under full-state
feedback (§4.2), so no realistic angle-only run exists to *measure* these. The
N = 6–8 entries are the **basin/κ precision predictors** — the resolution the
catch into the upright basin *would* demand:
- **δθ_max ≈ basin/κ_θ** (§3.2), **calculated** for N = 6, 7
  (`results/basin_N6-7.json`: basin_alt 7.0·10⁻⁴ / 1.4·10⁻⁴ rad, κ_θ 5.9·10³ / 2.6·10⁴);
- **δv_max ≈ basin·dt/κ_v**, with κ_v the input-channel (pivot-accel) disturbance→θ
  gain — **nearly flat in N** (0.089 → 0.117 over N = 6–8, confirming §3.3's flat
  κ_v) and carrying the empirical **δv ∝ dt** (§3.3); dt = 0.01 here.

**‡ N = 8 estimated** by log-linear extrapolation of basin_alt (≈ 4.9×/link); κ
is computed directly. Both predictors are the adversarial worst case, so they run
~2–30× *tighter* than the N = 1–5 measured tolerances (real quantization is partly
self-dithered, §3.2) — read them as conservative floors, not point predictions.
Swing time stays blank (no realized angle-only trajectory, §5). Two scalings
diverge sharply: **δθ collapses ≈ 20×/link** (κ_θ compounds), reaching ~3·10⁻¹⁰ rad
by N = 8, while **δv tightens only ≈ 5×/link** (κ_v flat) — so velocity stays the
**non-binding** channel, ~10³–10⁴× looser than angle at these N.

Swing-up needs finer precision than balance (the binding phase is the **catch** —
entering the small upright basin), and the gap widens sharply at N=4–5. Two
methods were decisive at high N: **implicit-dynamics collocation** (θ̈ as decision
variables with M θ̈ = rhs as a constraint — avoids a combinatorial symbolic
inverse, ~5 s vs >5 min per N=5 solve) and a **nonlinear predictor-corrector
observer** (a low-pass finite-difference rate estimate lagged enough to
destabilize the high-gain TVLQR at N≥4, even with perfect measurements).

### 4.2 N = 6 — the full-state frontier

N=6 swing-up **diverges at dt = 0.01** but **succeeds at dt = 0.004** under
full-state TVLQR. Root cause, established by elimination:

- divergence time is **invariant to the controller** (swept control penalty r,
  gain caps) — it's the *trajectory*, not the feedback;
- **perfect-state TVLQR also diverges**, gains exploding to ~7·10⁴ at a fixed
  mid-swing instant;
- there, all six links are within ~30° (the chain is **near-straight**), which is
  **near-uncontrollable from one pivot** (a straight chain ≈ a rigid body; its
  internal bending modes are unactuatable). The trackable trajectories are the
  ones whose **"bend order"** keeps the chain shimmying — never dead-straight.

**Removing the "magic" (curated seed).** Initially only a hand-selected N=5 seed
produced a trackable N=6. We tried three ways to eliminate that dependence:

| approach | removes dependence? | finding |
|---|---|---|
| smaller dt | **no** | dt = 0.004 is a *window* (~[0.008, 0.004]); below it diverges again; gains grow as dt shrinks. Wrong bend order is genuinely uncontrollable at every dt. |
| minimal homotopy ladder | **no** | reproduces the coarse class (−0.5 rev/link) but not the *fine* bend order; N=6 won't converge. The pool/ranking machinery is load-bearing. |
| **controllability-aware trajopt** | **yes** | a one-sided soft floor on bend-mode excitation c(θ)=‖M⁻¹(b⊙cosθ)‖²_bend, from a neutral cold ladder, yields a trackable N=6 **and** cuts peak pivot acceleration ~7× (≈50 → 7.3 m/s²). |

The controllability term penalizes the bending-mode excitation **collapsing**
(which happens exactly when the chain straightens), as a soft hinge below a floor
— a gentle nudge that bumps the natural swing-up out of its uncontrollable
near-straight dip into a trackable basin, without globally fighting the swing.
Result: seed-free N=6, final 0.018°, maxK 68,110.

**Initial-condition robustness (perturbed start).** We do not rely on starting at
the exact nominal hang: starting each run from θ_i(0) = π + U(−δθ, δθ) with random
initial rates, the same TVLQR feedback rejects the deviation and still swings up.
The seed-free N=6 controls pass **16/16 seeds at ±0.10π (~10%) + ±0.5 rad/s** and
remain robust to **±0.8 rad (25% of π)** initial angle error. This is the *easy*
direction: the hanging start is near-stable, so the funnel is widest there —
opposite the mid-swing near-uncontrollable patch that sets the difficulty.

### 4.3 N = 7 — beyond the architecture

Seeding N=7 from the clean N=6 trajectory and applying the same controllability
floor, **no N=7 candidate is trackable at any dt ∈ [0.004, 0.015]**. Crucially:

- it is **not a gain-magnitude wall** — a TVLQR weight sweep drove maxK below
  N=6's (~4,000) and it still diverges at every (R, dt); **no gain stabilizes the
  N=7 trajectory**.
- **Diagnosis:** the *scalar* controllability proxy that sufficed for N=6 is
  insufficient at N=7. A 7-link chain has **6 bending modes**; flooring the
  *aggregate* bend-excitation does not guarantee *each individual mode* stays
  excitable from the single pivot. One mode can be ~uncontrollable mid-swing while
  the aggregate looks healthy — unstabilizable by any TVLQR.

---

## 5. What worked, what was a dead end

**Worked**
- Hand-derived closed-form dynamics + independent verification (foundation for everything).
- Implicit-dynamics collocation + homotopy in N (made N ≥ 4 tractable).
- Predictor-corrector observer (made high-N realistic swing-up trackable).
- basin × κ decomposition (explains and predicts the precision scaling).
- Smaller dt (dt = 0.004) as the N=6 *full-state* enabler.
- **Controllability-aware trajopt** (removed the curated-seed dependence at N=6; gentler control).
- Mesh refinement (coarse→fine warm starts) and bounded-iteration parallel solves (kept the search fast/observable).

**Dead ends (informative)**
- Smaller dt as a *general robustness* lever — refuted; it's a narrow window, not "smaller is better."
- Naive minimal homotopy ladder for N=6 — can't reproduce the fine bend order.
- Anti-alignment penalty (a controllability *proxy* that just penalized straightness) — backfired: more violent, not more controllable.
- Hard/aggressive controllability floors — distort into a *different* untrackable bend order.
- Scalar controllability floor at N=7 — insufficient for the richer mode structure.

**Methodological lesson recorded:** with `print_level=0` IPOPT solves are opaque;
progress visibility (inf_pr/inf_du, per-candidate file counts, stage logs) and
bounded iterations matter as much as the math. Several restarts were spent
re-learning that silence ≠ progress.

---

## 6. End state

- **Balance:** solved and characterized N = 1–5 with precision tables + scaling
  theory; N = 6–7 basin/κ computed; fp32-θ shown sufficient through N=7.
- **Swing-up:** robust N = 1–5 with realistic angle-only sensing (animations
  rendered); **N = 6 full-state, seed-free** via controllability-aware trajopt
  (the curated-seed "magic" eliminated, accelerations 7× gentler); **N = 7
  characterized as a frontier wall** with a precise cause.
- **Reproduction** (`repro/`): `generate_n6.py` (seed-free, recommended),
  `stage2_n6.py` (fast from a seed), `generate_nN.py` (general N; reproduces N=6,
  documents N=7), `simulate_n6.py` (verify swing-up + balance catch, render mp4),
  `perturbed_n6.py` (perturbed-start robustness challenge: 16/16 at ±10%).

This matches the broader literature, where published cart-pole swing-up tops out
at the **triple** pendulum (Glück–Kugi 2013), quadruple is rare, and 5+ is
research-frontier; the **data-rate theorem** (Nair–Evans, Tatikonda–Mitter) and
human stick-balancing "microchaos" work (Milton–Insperger; Csernák–Stépán) frame
the precision side.

---

## 7. Paths for improvement

1. **Per-mode controllability for N=7+.** Replace the scalar bend-excitation
   floor with a constraint on the **minimum singular value of the input-to-
   bending-mode coupling across all modes** (ensure every bending mode is
   excitable along the path). Directly targets the diagnosed N=7 cause; a
   harder NLP (SVD/eigenvalue constraints in CasADi) with no guarantee.
2. **Robust / lower-gain feedback.** The full-state results need ~7·10⁴ gains and
   (at N=6) transient ~5 g pivot accelerations — untrackable by any real observer
   or actuator. Controllability-aware trajectories that lower the required gain,
   plus H∞/robust or controllability-gated ("coast through low-controllability
   patches") feedback, could move N=6 toward *realistic* sensing.
3. **Learned policies.** RL/MPC with the verified dynamics as a differentiable
   model — particularly for the catch, where the basin is the binding constraint.
4. **Precision sweeps for swing-up N≥6** once a realistic-sensing controller
   exists, to *measure* the δθ/δv tables — replacing the basin/κ **projected**
   floors now listed for N = 6–8 (§4.1) with achieved tolerances.
5. **dt-window / robustness margins.** Ship dt in the *middle* of the trackable
   window (≈0.005–0.006 for N=6) rather than its fine edge (0.004) for margin.

---

## Appendix — reproduction & artifacts

- Core: `pendulum/{dynamics,sim,balance,trajopt,swingup_traj}.py`; tests in
  `tests/test_dynamics.py`.
- Reproduction: `repro/` (see `repro/README.md`).
- Precision data: `results/*.json`, `results/*_notes.md`; progress log:
  `PROGRESS.md`.
- Media: `media/` (balance + swing-up animations, acceleration comparison).
- This session's transcripts (verbatim, redacted): `sessions/` (generated by
  `sessions/gen_sessions.py`).
