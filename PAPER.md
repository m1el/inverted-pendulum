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
perturbation); **N = 7** — initially diagnosed as beyond the architecture — is
**solved** (full-state, dt = 0.002) by **predecessor enumeration**: time-reversed
braked falls from upright yield exact, seed-free nominals, 44/46 of which track,
with measured robustness (±0.78 rad initial angles; δθ = 8·10⁻⁶ rad with a
quantized 5 s catch). The trackable N = 7 class *winds* (links make 5–26
revolutions): no-rotation N = 7 trajectories exist (constructive NLP solutions)
but are unstabilizable from the pivot — their worst bending modes lose ~2.5×
input coupling exactly in the unstable final third. Headline: required angle
precision tightens ≈ 6–20× per added link; sensing — not actuation — is the
binding constraint; and the wall at high N is the controllability of the
*path family*, not link count or compute — at N = 7 trackability demands
winding.

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
| 7 | 56.7 ⁂ | 5.5·10⁻⁹ † | 1.4·10⁻⁵ † |
| 8 | — | 2.8·10⁻¹⁰ ‡ | 2.5·10⁻⁶ ‡ |

**⁂ N = 7 realized full-state** (reverse-fall, §4.3; RK4-consistent TVLQR at
dt = 0.002); its *full-state* quantization thresholds (angles quantized, rates
exact — the flip-table convention of §4.4, not this table's angle-only
protocol) are **measured**: δθ = 2·10⁻⁵ rad, δv = 2.7·10⁻⁴ m/s including a 5 s
quantized balance hold (6.4·10⁻⁴ / 1.0·10⁻² for the swing alone — the sustained
catch binds, as §3.2 predicts; N = 6 refined: 3.1·10⁻⁵ / 1.9·10⁻² with hold).
The angle-only floors above remain projections.

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

**The dt window is a nominal-consistency artifact.** The evaluation pipeline
resamples the h = 0.01 trapezoidal-collocation nominal onto the simulation grid
by linear interpolation; the resulting one-step defect (RK4 from nominal state k
under the applied control vs nominal state k+1) is ~10⁻³ rad/step — a persistent
disturbance scaled by gains up to ~10⁵. Re-solving the *same* candidate at
**h = 0.005** (one-step defect ~5·10⁻⁵) turns an everywhere-untrackable N=6
nominal into one **trackable at every dt ∈ [0.002, 0.012]**, with a funnel
measured ≥ ±0.5 rad + ±0.8 rad/s at both dt = 0.004 and dt = 0.01 (wider than
the unrefined trackable candidates: refinement widens the funnel, it does not
trade robustness for accuracy). Two cautionary findings from the same study
(`repro/consistent_nominal.py`): (i) a *more accurate* resampler (cubic-Hermite
dense output, 9× closer to the true flow) tracks strictly *worse* at h = 0.01 —
the linear interpolant's defect is a ~100 Hz zero-mean sawtooth that
self-dithers away, while the smooth Hermite defect forces the barely-controllable
direction coherently; near the stability knife-edge it is the defect's
**spectrum**, not magnitude, that decides survival. (ii) A defect-free nominal
rolled out from its exact start keeps the error at machine epsilon — such a test
only proves local stability and says nothing about the funnel; all trackability
claims here are therefore backed by perturbed-start protocols.

### 4.3 N = 7 — solved by predecessor enumeration; winding is essential

**The fast NLP class hits a real wall.** Seeding N=7 from the clean N=6
trajectory and applying the same controllability floor, **no fast (T ≈ 17 s)
NLP candidate is trackable at any dt ∈ [0.002, 0.015]**, under any resampler,
and the wall survives artifact elimination: a candidate mesh-refined to
h = 0.005 (one-step defect 2.3·10⁻⁵) still diverges everywhere. It is **not a
gain-magnitude wall** — TVLQR weight sweeps (maxK driven below N=6's ~4,000)
diverge at every (R, dt). The original diagnosis stands for this class: flooring
the *aggregate* bend-excitation does not keep *each* of the 6 bending modes
individually excitable from one pivot.

**The solution: enumerate the predecessors of upright (reverse falls).** The
dynamics are time-reversible (the EOM is even in θ̇), so a *braked fall* from
upright + ε·(mixture of unstable-mode directions), landed at the hang by a
gentle energy brake (|a| ≤ 4; a saturated brake never settles) plus a hang-LQR,
time-reversed, is an **exact** swing-up nominal: zero collocation/interpolation
defect (generated by the same RK4/ZOH as the simulator, at the simulation dt),
no settle band, no NLP, no seed. The arrival-shape family at upright *is* the
fall-departure family — a ~7-dimensional, directly enumerable space
(`repro/reverse_fall.py`; seconds per candidate).

- **N = 6 validation:** 16/36 arrival shapes track at dt = 0.004; the trackable
  family arrives along the **slow** unstable modes (λ = 3.6, 1.5 — both signs;
  all four fast-mode arrivals fail).
- **N = 7: 44/46 slow-4-mode mixtures track at dt = 0.002** (14/46 at 0.003,
  0/46 at ≥ 0.004 — the dt ladder narrows by ~2× per link, and the original
  sweep stopped at 0.004). Winner: final 0.010°, T = 56.7 s, max|a_ff| = 18.9,
  max|v| = 4.3, maxK = 1.7·10⁵.
- **Half of that dt requirement is controller-model mismatch, not physics.**
  The TVLQR's (Ad, Bd) via matrix exponential of the linearized continuous
  model differs from the simulator's RK4 step map by ~1% mid-swing — at
  10⁵ gains that eats the margin. Discretizing the gains against the **RK4
  Jacobian itself** (controller model ≡ simulator), the same N=7 nominal
  passes the full 16/16 protocol at **dt = 0.004 and 0.003** (new wall:
  0.005), and 14/46 of the dt = 0.004-native pool tracks (was 0/46) — the same
  timestep the original N=6 result needed. Caveat: at dt = 0.004 the arrival
  error (2.8°) exceeds the ~1.4·10⁻⁴ rad upright basin, so the *catch* fails —
  coarse dt buys swing trackability, not the complete maneuver; the flagship
  control (swing + held catch) runs at dt = 0.002. The no-rotation
  nominal is *not* rescued (its failure is modal, §above); and a 96-point
  blind Q/R/QF search on N=6 is a null (all candidates and the baseline
  saturate at the ±0.8 rad hang-basin cap): fix the model, not the weights.
- **Measured robustness** (8-seed all-pass bisection): initial perturbation
  **±0.78 rad** angles-only (±0.65 with ±1.6× rates) — the same "25% of π" class
  as N=6, and **16/16 at the ±0.1π + ±0.5 rad/s protocol**; full-state
  quantization δθ = 7.7·10⁻⁶ rad, δv = 1.05·10⁻⁴ m/s including a 5 s quantized
  catch (§4.1 note).

**Winding is essential at N = 7.** Every trackable N=7 trajectory found winds
(links make 5.5–26 net revolutions and whip to 37–62 rad/s). Imposing the
classical no-rotation property (θ ∈ [−1, 5.9]: hang-overshoot allowed, no link
completes a turn) on a slow (T = 25 s), gentle (|θ̇| ≤ 12), controllability-
floored NLP **solves** — producing the best-conditioned N=7 trajectory of the
project (aggregate bend-excitation cmin ≈ 1.0) — but the result is
**unstabilizable**: untrackable at every (dt, resampler, mesh h = 0.01/0.0075),
0/24 across a TVLQR weight sweep (R ∈ [0.01, 10] × terminal scale × dt), and
its open-loop replay departs at t = 16.2 s = 0.65 T. The mechanism is measured
(windowed per-mode input coupling, worst mode):

| window | no-rotation (untrackable) | reverse-fall (trackable) |
|---|---|---|
| t < 0.65 T | 0.30 | 0.42 |
| 0.65–0.9 T | **0.13** | **0.36** |
| > 0.9 T | **0.12** | **0.27** |

The no-rotation path's worst bending modes go ~2.5× deafer to the pivot exactly
in the unstable final third (the open-loop departure point); the winding path
keeps all six modes ≥ 0.27 throughout — spinning links cross the mode-deaf
configurations transversally at speed instead of dwelling near them. The
aggregate floor is thus provably the wrong surrogate at N = 7 (cmin ≈ 1.0
coexists with a worst-mode coupling of 0.12).

**The corridor's coupling ceiling, measured** (`repro/permode_hard_n7.py`).
Imposing the coupling requirement directly — hard windowed-RMS per-mode floors
Σ(v_jᵀu)² ≥ c²Σ‖u‖² per 1.5 s window over t/T ∈ [0.55, 0.95], warm-started from
a feasible no-rotation trajectory (baseline c₀ = 0.079) — brackets how much
coupling the corridor can buy at any price: **c ≤ 0.09 feasible** (solver finds
it in minutes; still untrackable, peak worst-mode pinned ≈ 0.13), **c = 0.11–
0.13 undecided** at 6,000 iterations (the boundary), **c ≥ 0.15 certified
locally infeasible** (IPOPT restoration convergence, three independent floors;
soft-penalty versions agree at 0/4). The no-rotation corridor's coupling
ceiling is ≈ 0.1 — roughly a third of what the trackable winding class carries
through the same windows. Within this corridor the coupling requirement is not
expensive; it is **unpurchasable** — the quantitative content of "winding is
essential" (modulo the locality of NLP infeasibility certificates).

**Minimal winding is optimal: the one-flip.** Sweeping bend-bounded free-turn
classes (joint bends |β| ≤ 92–120° — "no convolution" — whip ≤ 25, winding free
or prescribed; 30 solves) shows the coupling budget *peaks at one extra
revolution*: 1-turn solutions measure 0.34–0.59 worst-mode coupling, 2-turn
0.19–0.30, 0-turn ≤ 0.13 (the certified ceiling). Mechanism: a single fast pass
crosses the mode-deaf straight orientations transversally (a millisecond
blackout ≪ 1/λ — harmless by the data-rate argument), while a second revolution
forces sustained phase-aligned rotation — rigid-mode motion that crowds bending
shimmy out of the fixed time/whip budget. An energy floor (E ≥ V_up + 15 J over
the crossing) is needed to stop the optimizer from *creeping* over the top
(min-effort loves slow crossings, which dwell deaf — caught by visual
inspection, invisible to a late-window-only metric).

**The one-flip is trackable — at a measured price.** The defect law governs:
this violent class needs **h = 0.0025** before its nominal reaches the
trackable defect grade (6.3·10⁻⁵; at h = 0.005 it still carries 1.4·10⁻³ —
defects per mesh scale ~30× with trajectory violence). There, the gate opens:
**16/16 at ±0.02 rad ± 0.032 rad/s** (final 0.029°), 11/16 at ±0.05, 0/16 at
the full ±0.1π protocol; maxK = 7.2·10⁵
(`repro/n7_oneflip_controls.npz`; the verified iterate was an IPOPT max_iter
"failure" feasible to 10⁻⁸ — gate-any-feasible-iterate is load-bearing
methodology). The funnel-along-the-trajectory comparison
(`media/funnel_compare_N4-7.png`; per-phase worst-case-kick bisection) places
the two N=7 solutions at opposite ends of a sharp trade:

| flagship | start funnel | tightest | median | character |
|---|---|---|---|---|
| N=6 refined (15 s) | 0.80 rad | 6.3·10⁻⁶ | 5.8·10⁻² | baseline |
| N=7 slow (57 s, 9 rev) | **0.95 rad** | 3.9·10⁻⁶ | 6.9·10⁻³ | robust; erratic, bang-bang control |
| N=7 one-flip (12 s, 1 rev) | 0.084 rad | **closes (< 10⁻⁶)** | 1.7·10⁻⁴ | elegant; precision-only |

The catch bottleneck scales gently with N (~10×/link — the basin/κ law measured
along whole trajectories); *choosing the fast class* costs ~340× in median
funnel width and an outright closure at one mid-flip phase (survivable
end-to-end only because upstream deviations arrive exponentially contracted).
At the frontier, the trajectory-class choice dwarfs the dimensionality penalty.

### 4.4 Link-flip — point-to-point between unstable equilibria

A different task class: not hang→upright, but **upright → one link inverted →
upright** — a there-and-back between two *unstable* equilibria (the chain has an
equilibrium at every θ_i ∈ {0, π}). Same controllability-aware collocation +
full-state TVLQR; the maneuver momentarily reaches one-down (link L at π, the rest
up) and returns to the upright catch.

**It is strictly harder than swing-up at equal N.** Swing-up *starts at the hang*
(stable, wide funnel, gains ~200); the flip *departs and returns to the
dead-straight upright* — the maximally near-uncontrollable configuration — paying
the catch's gain explosion at **both** ends.

| N | flip trackable? | trackable links | peak maxK |
|---|---|---|---|
| 2 | yes | all (0,1) | 1.5·10³ |
| 3 | yes | all (0,1,2) | 6.5·10³ |
| 4 | yes | all (0–3) | 1.0·10⁵ |
| 5 | partial | middle only (2,3) | 1.8·10⁵ |
| 6 | no | none (repeated attempts) | — |

N=5 already exceeds the N=6 *swing-up* stiffness (maxK 1.8·10⁵ vs 6.8·10⁴) and
only the middle links survive; **N=6 produced no trackable flip** — over the wall.

**The stiffness is the catch, not the swing.** Plotting the TVLQR gain schedule
over time (`scripts/plot_stiffness.py`): gains are gentle (median |K|~200)
throughout, then **plateau the instant the chain reaches upright** — median |K|
jumps ≈90× (N=5) to ≈300× (N=6), peaking 1.7·10⁴ / 6.8·10⁴ (the ≈4×/link growth
of the balance amplification κ). Swing-up pays this once; the flip pays it twice
(it starts *and* ends near upright) — the mechanical reason it hits the wall a
link sooner.

**Required precision (full-state quantization thresholds).** Largest angle step δθ
and velocity-command step δv for which the flip still completes (one-down →
upright, balanced), by log-bisection:

| N | δθ_max (rad) | δv_max (m/s) |
|---|---|---|
| 2 | 1.9·10⁻¹ | 1.2·10⁻¹ |
| 3 | 2.3·10⁻² | 8.5·10⁻² |
| 4 | 3.7·10⁻³ | 5.1·10⁻² |
| 5 | 4.0·10⁻⁴ | 7.2·10⁻³ |

- **δθ collapses ≈ 6–9×/link** — the same angle-precision law as balance/swing-up
  (≈6.3×/link), now measured for the flip.
- **δv stays the non-binding channel** (~0.12 → 0.007, declining slowly).
- These are **full-state** thresholds (angle quantized, *rates exact*, as the flip
  is full-state TVLQR) — hence upper bounds; a realistic angle-only controller
  would be tighter, the same gap as table 4.1 vs its full-state counterpart.
- Which link you flip matters (e.g. N=5: L2 tolerates 4·10⁻⁴ rad, L3 only
  2.6·10⁻⁵), partly because the trackable dt differs per link.

**Fold-in-half — the maneuver, not N, sets the difficulty.** A *harder-looking*
variant: fold the upper N/2 links down as one (N=6 target [0,0,0,π,π,π], a crease
at the middle joint). It is in fact **much easier than flipping a single link** —
and, strikingly, **the N=6 fold is trackable where the N=6 single-link flip is
not**:

| maneuver | N=4 maxK | N=5 maxK | N=6 |
|---|---|---|---|
| single-link flip | 1.0·10⁵ | 1.8·10⁵ | **fails** |
| fold-in-half | 4.4·10³ | 5.3·10⁴ | **works** — dt=0.01, final 0.079°, maxK 6.5·10⁴ |

The folded target is *less* unstable (its upper links hang from an elevated joint),
and folding both halves as a coordinated crease keeps the chain **bent and
controllable throughout** (bend-excitation cmin ≈ 0.85 vs the flip's ≈0.2–0.6) —
it never enters the near-straight danger zone. So the N=6 fold even succeeds at the
*coarse* dt = 0.01 that the swing-up needed dt=0.004 to survive. **The binding
quantity is the path's controllability, not the link count or how "large" the
reconfiguration looks** — a 3-link fold beats a 1-link flip.

### 4.5 Bend order — definition and reproducibility

"Bend order" has been used loosely above; here it is made precise. Write the
configuration as absolute link angles θ_i and define the **joint bend angles**
β_i(t) = θ_{i+1}(t) − θ_i(t) — the signed fold at joint i (β_i = 0 ⇔ links i, i+1
collinear). A swing-up is a path θ(t) from the hang (θ_i = π) to upright (θ_i = 0);
its **bend order** is the pair **(W, F)**:

- **W — coarse (winding) class:** W_i = ([unwrap θ_i]ᵀ − [unwrap θ_i]₀)/2π, the net
  revolutions each absolute link angle makes. W is a **homotopy invariant** of the
  path (you cannot change it without rotating a link through a full turn).
- **F — fine (fold) sequence:** for each joint, the **sign word** of β_i(t) read off
  its zero-crossings (with a small deadband) — the ordered list of bend-reversals
  (straighten-then-refold-the-other-way events). Summarized by the per-joint
  **crossing count** and the inter-crossing **sign pattern**.

Two swing-ups share a bend order iff their W agree and their F agree up to (i) the
global mirror θ → −θ (which flips every β sign) and (ii) time-reparametrization.
Controllability is a property of F: a trajectory is trackable only if its fold
sequence **never drives all β_i through 0 at once** (the dead-straight,
near-uncontrollable configuration). The bend-mode excitation
c(θ) = ‖M⁻¹(b⊙cos θ)‖²_bend used as the trajopt floor (§4.2) is exactly a smooth
surrogate for distance from that all-β-zero set.

**What reproduces (`scripts/bend_topology.py`, all stable N = 4, 5, 6 solutions —
independent solvers, floors, seeds):**

- **W is universal — −0.5 rev for *every* link.** The chain unwinds exactly a half
  turn per link, hang→upright. The other winding classes the solver also finds
  (e.g. +0.5/link, or −1.5 on the base link) are precisely the **untrackable** ones:
  trackability *selects* W = (−½)ⁿ. This coarse class is reproduced 100% of the time.
- **F is a conserved family, not a point.** Within W = (−½)ⁿ the fold sign-pattern is
  largely shared *within a method* — 4 of 5 controllability-aware N=6 solutions share
  bend-sign (+,−,−,+,+), an S-shaped shimmy that intensifies toward the tip — while
  exact crossing counts vary by a few (N=6 per joint: ≈ 9–10, 7–13, 9–17, 13–18,
  13–20). A different solver (the original N=6) lands in a different fine class at the
  same W.

So the bend order is reproduced **as a topological class** (winding forced; "keep
shimmying, never dead-straight" forced), with a **continuum of fine fold-sequences
inside it**. This is why the coarse class transfers across N via the homotopy ladder
but the fine order does not — the latter is the load-bearing structure a hand-curated
seed encodes (§4.2).

**Scope correction (N = 7).** The claim "trackability selects W = (−½)ⁿ" is a
property of the fast NLP class at N ≤ 6, **not** of trackability per se — and at
N = 7 it *inverts*. The trackable N=7 reverse-fall trajectories (§4.3) live in
far-away winding classes (5.5–26 net revolutions per chain) and pass the full
robustness protocol, while every W = (−½)ⁿ-corridor N=7 trajectory tested —
fast or slow, coarse or refined — is unstabilizable. At the frontier, winding
stops being forbidden and becomes required.

### 4.6 Control jerk near the uncontrollable frontier

The N=6 full-state swing-up looks jerky mid-swing. Splitting the applied pivot
acceleration a(t) into feedforward + TVLQR feedback:

- the **feedforward** a_ff is smooth (jerk RMS 7.8, |a| ≤ 7.3 — the trajopt already
  penalizes ∫ȧ²); the **closed-loop** a has jerk RMS 19.8, peak 218 — a **2.5×**
  feedback amplification, concentrated in the *swing* (RMS 23) not the *catch* (1.3).
  Near upright the tracking error is tiny, so even the 6.8·10⁴ gains add little jerk;
  mid-swing the feedback is fighting a barely-controllable plant.

**Can a policy remove it? Not at the feedback layer.** We built the principled
smoother — a **jerk-penalizing input-augmented TVLQR**: augment the state with the
pivot acceleration (z = [θ, θ̇, v, a]) and make the control the jerk u = ȧ, so a(t) is
C¹ and jerk is penalized directly (`repro/swingup_n6_lowjerk.py`). It **fails**: any
penalty heavy enough to smooth the swing **destabilizes the rollout** (diverges), and
the stable region yields *more* jerk than baseline, not less — the near-uncontrollable
swing genuinely **needs the control bandwidth**. A time-varying penalty (smooth swing,
permissive catch) diverges identically. **The jerk is the unavoidable signature of
high-bandwidth stabilization of a plant barely controllable from one pivot.**

**The only lever is the trajectory's controllability.** Feedback jerk amplification
tracks the bend-excitation margin: the highly-controllable fold-in-half (cmin 0.85)
has feedback adding just **1.1×** jerk versus the swing-up's **2.5×** (cmin 0.73). A
more-controllable nominal needs less corrective feedback, so it is intrinsically
smoother. "Optimize a policy for low jerk" therefore means **shape the path to be more
controllable** (and keep the feedforward smooth) — the same bend-floor objective that
makes high N trackable at all — not penalize jerk at the controller.

---

## 5. What worked, what was a dead end

**Worked**
- Hand-derived closed-form dynamics + independent verification (foundation for everything).
- Implicit-dynamics collocation + homotopy in N (made N ≥ 4 tractable).
- Predictor-corrector observer (made high-N realistic swing-up trackable).
- basin × κ decomposition (explains and predicts the precision scaling).
- Smaller dt (dt = 0.004) as the N=6 *full-state* enabler — later traced to
  nominal truncation error: at mesh h = 0.005 the dt dependence disappears (§4.2).
- **Controllability-aware trajopt** (removed the curated-seed dependence at N=6; gentler control).
- **Reverse-fall generation** (predecessor enumeration via time-reversal) — exact,
  seed-free, NLP-free nominals; solved N=7 full-state (§4.3).
- **Windowed per-mode coupling diagnostic** — turned the N=7 "winding essential"
  claim from inference into measurement (§4.3).
- Mesh refinement (coarse→fine warm starts) and bounded-iteration parallel solves
  (kept the search fast/observable); mesh *ladders* (0.01→0.0075→0.005) where
  one-jump refinement stalls; IPOPT `output_file` logs for live inf_pr/inf_du.

**Dead ends (informative)**
- Smaller dt as a *general robustness* lever — refuted; it's a narrow window, not "smaller is better."
- Naive minimal homotopy ladder for N=6 — can't reproduce the fine bend order.
- Anti-alignment penalty (a controllability *proxy* that just penalized straightness) — backfired: more violent, not more controllable.
- Hard/aggressive controllability floors — distort into a *different* untrackable bend order.
- Scalar controllability floor at N=7 — insufficient for the richer mode structure
  (now measured: aggregate cmin ≈ 1.0 coexists with worst-mode coupling 0.12, §4.3).
- Jerk-penalizing (input-augmented) TVLQR to smooth the swing — refuted; any
  effective penalty diverges, the swing *needs* the bandwidth (§4.6). Path
  controllability, not the controller, is the jerk lever.
- Hermite dense-output resampling as a trackability fix — backwards at h = 0.01:
  the *more accurate* nominal tracks worse (its smooth defect forces the weak
  direction coherently; the linear sawtooth self-dithers, §4.2). Fix the mesh,
  not the interpolant.
- No-rotation N=7 (classical corridor) — *exists* but unstabilizable: 0/24 weight
  sweep; winding is essential (§4.3).
- Saturated (|a| = 25) braking for reverse falls — never lands; gentle (|a| ≤ 4)
  braking settles in ~45 s.

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
  (the curated-seed "magic" eliminated, accelerations 7× gentler), dt-independent
  once the mesh is fine enough (§4.2); **N = 7 solved full-state, twice over**
  (§4.3): the slow reverse-fall (57 s, 9 rev, ±0.78 rad funnel, measured δθ/δv)
  and the **one-flip** (12 s, 1 rev, bends ≤ 120°, 16/16 at ±0.02 rad — the
  "nice" corner), with the no-rotation class certified unstabilizable, the
  winding requirement measured per-mode, minimal winding shown optimal, and the
  robustness trade quantified (funnel comparison N = 4–7). **N = 8 untested**
  (the generators are N-generic; the dt ladder predicts ~0.001).
- **Reproduction** (`repro/`): `generate_n6.py` (seed-free, recommended),
  `stage2_n6.py` (fast from a seed), `generate_nN.py` (general N; reproduces N=6),
  `reverse_fall.py` (slow N=7; arrival-shape enumeration), `fullturn_n7.py`
  (one-flip N=7: bend-bounded winding classes, energy floor, emergent winding),
  `n7_robustness.py`
  (perturbation/quantization bisection), `consistent_nominal.py` +
  `refine_mesh.py` (nominal-consistency artifact studies), `norot_n7.py` +
  `norot_refine_ladder.py` + `permode_hard_n7.py` (no-rotation N=7: exists,
  unstabilizable; coupling-ceiling certificates),
  `simulate_n6.py` (verify + render; N-generic — produced the N=7 video),
  `perturbed_n6.py` (perturbed-start robustness challenge: 16/16 at ±10%).

This matches the broader literature, where published cart-pole swing-up tops out
at the **triple** pendulum (Glück–Kugi 2013), quadruple is rare, and 5+ is
research-frontier; the **data-rate theorem** (Nair–Evans, Tatikonda–Mitter) and
human stick-balancing "microchaos" work (Milton–Insperger; Csernák–Stépán) frame
the precision side.

---

## 7. Paths for improvement

1. **Windowed per-mode floor → a "nice" trackable N=7.** The N=7 cause is now
   measured (§4.3): the worst mode's coupling collapses in the unstable final
   third. The fix needs no SVD: with the fixed upright bending-mode basis V and
   the implicit u (M u = b⊙cosθ) already in the NLP, add per-mode hinges
   |v_jᵀu| ≥ c_j on nodes in t/T ∈ [0.6, 0.95] (or log-sum-exp over short
   windows — coupling must be available within each instability timescale
   1/λ, not pointwise). Combined with the no-rotation corridor this directly
   searches for a *gentle, classical, trackable* N=7; if infeasible, relax the
   corridor by per-link winding budgets w_i (θ_i ∈ [−1, 2πw_i + 5.9]) and
   minimize Σw_i — the minimal-winding Pareto frontier between "nice" and
   "trackable". A complementary route: trackability-in-the-loop continuation
   from the (ugly, trackable) reverse-fall solution — tighten whip/winding
   bounds in small homotopy steps, re-verify TVLQR trackability at each rung
   (seconds per check), stop at the boundary.
2. **Robust / lower-gain feedback.** The full-state results need ~7·10⁴ gains and
   (at N=6) transient ~5 g pivot accelerations — untrackable by any real observer
   or actuator. Controllability-aware trajectories that lower the required gain,
   plus H∞/robust or controllability-gated ("coast through low-controllability
   patches") feedback, could move N=6 toward *realistic* sensing.
3. **Learned policies.** RL/MPC with the verified dynamics as a differentiable
   model — particularly for the catch, where the basin is the binding constraint.
4. **Precision sweeps for swing-up N≥6** with realistic sensing — the
   *full-state* δθ/δv are now measured for N = 6–7 (§4.1 note, §4.3); the
   angle-only floors remain projections until an observer-based controller
   exists at these N.
5. ~~dt-window margins~~ **Resolved**: the window was nominal-truncation error
   (§4.2). Ship fine meshes (h ≤ 0.005, via ladders), not tuned dt.
6. **N = 8 via reverse falls.** The generator is N-generic and each candidate
   costs seconds; the dt ladder (0.004 → 0.002 per link) predicts dt ≈ 0.001.
   A direct test of whether predecessor enumeration scales.

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
