# Inverted N-link pendulum control — progress log

## Problem
- N-link (N=1..5) pendulum, uniform rods (mass 1, length 1, I_com = 1/12).
- Only control: X velocity of the pivot (one end of the first rod).
- Goal: given simulation params (gravity g, time step dt), find the required
  **precision** on (a) link-angle measurements and (b) X-velocity command,
  needed to (1) balance upright, (2) swing up from hanging.
- Precision model: quantization. Measured angles are rounded to a grid of step
  `dtheta`; the commanded pivot velocity is rounded to a grid of step `dv`.

## Dynamics (derived by hand, to be verified vs sympy)
With pivot at (x, 0) and angles θ_i measured from **upright**:

    Σ_l M_il(θ) θ̈_l = −ẍ b_i cosθ_i − Σ_l A_il sin(θ_i−θ_l) θ̇_l² + g b_i sinθ_i

    A_il = N − max(i,l) + 1/2   (i ≠ l),   A_ii = N − i + 1/4
    M_il = A_il cos(θ_i−θ_l) + δ_il/12,    b_i = N − i + 1/2     (1-based i,l)

Key fact: θ-dynamics depend only on pivot **acceleration** ẍ. Velocity control
v_cmd is implemented as ZOH acceleration a = (v_cmd − v)/dt over the step.

Sanity check N=1: (1/3)θ̈ = −(1/2)ẍcosθ + (g/2)sinθ ✓ (rod pivoting about end).

## Plan
- [x] Project setup (uv, git)
- [ ] Core dynamics `pendulum/dynamics.py` (closed form) + sympy cross-check + energy conservation test
- [ ] Simulator `pendulum/sim.py` (RK4, ZOH acceleration, quantization hooks)
- [ ] Balance: LQR (+ finite-difference / observer state estimate) for N=1..5
- [ ] Balance precision sweeps: max dtheta (dv→0), max dv (dtheta→0), joint boundary
- [x] Swing-up: N=1 energy pumping + LQR catch; N=2..5 trajectory optimization + TVLQR + catch
- [x] Swing-up precision sweeps N=1..5
- [ ] Swing-up N=6,7 (bonus, in progress)
- [ ] Report: RESULTS.md (g=9.81, dt=0.01; scaling notes)

## Log
- 2026-06-07: project setup; derived closed-form dynamics by hand.
- 2026-06-07: dynamics verified vs independent sympy Lagrangian derivation
  (N=1..5, err <1e-8) and energy conservation (drift <5e-9 over 10 s).
- 2026-06-07: LQR balance works N=1..5 with perfect observation. Basin of
  attraction shrinks fast with N (alternating tilt 0.02 rad fails for N>=4).
  Unstable pole lam_max: 3.8, 7.2, 10.3, 13.2, 15.9 /s for N=1..5.
- 2026-06-07: launched 3 parallel workstreams: (A) balance precision sweeps,
  (B) N=1 swing-up (energy pumping), (C) N=2..5 swing-up (casadi direct
  collocation + TVLQR + catch). Orchestrator measuring basin radii + linear
  noise-amplification (kappa) for a theory-side prediction of thresholds.
- 2026-06-07: basin + amplification measured (results/basin.json), dt=0.01:

  | N | lam_max /s | basin (alt tilt, rad) | kappa (meas->theta) | pred dtheta_max ~ basin/kappa |
  |---|-----------|----------------------|---------------------|-------------------------------|
  | 1 | 3.8 | 0.70    | 1.7  | 4e-1 |
  | 2 | 7.2 | 0.24    | 12   | 2e-2 |
  | 3 | 10.3 | 0.066  | 63   | 1e-3 |
  | 4 | 13.2 | 0.014  | 297  | 5e-5 |
  | 5 | 15.9 | 0.0033 | 1341 | 2.5e-6 |

  Both basin radius and noise amplification worsen ~4-6x per added link =>
  required angle precision shrinks ~25-40x per link (two multiplicative
  mechanisms). To be checked against agent A's empirical thresholds.
- 2026-06-07: extended to N=6,7 (results/basin_N6-7.json), computed directly,
  matching the log-linear extrapolation from N=1..5 (delta-theta trend
  ~20.3x/link, fit delta-theta ~ 8*20^-N rad):
  N=6: lam=18.3/s basin=7.0e-4 kappa=5948  pred dtheta ~1.2e-7
  N=7: lam=20.6/s basin=1.4e-4 kappa=26136 pred dtheta ~5.5e-9
  lam_max ≈ (3.0N+1.06)*sqrt(g/9.81), linear in N. Scaling laws:
  dtheta_max = Phi_N(dt*sqrt(g/L)); dv_max = sqrt(gL)*Psi_N(dt*sqrt(g/L)).
- 2026-06-07: N=1 SWING-UP done (agent B): energy pumping (kick at bottom,
  a = sat(kE*(E-E*)*thd*cos th) - kv*v - kx*x) + hysteretic LQR catch.
  Swing-up in ~0.76 s. Thresholds (g=9.81, dt=0.01): dtheta ~0.024 rad,
  dv ~0.054 m/s; balance-only tolerates ~0.52 rad / ~0.22 m/s => swing-up is
  ~22x stricter on angle, ~4x on velocity; binding phase is the CATCH.
  dt-scaling non-monotone for dtheta (peak near dt=0.01, FD-noise vs discrete-
  catch trade-off); dv threshold grows with dt; both rise ~sqrt(g) with g.
- 2026-06-07: CAVEAT found by independent re-verification: success is
  NON-MONOTONE in quantization step near threshold (e.g. dv=0.05 fails seeds
  that 0.055 passes; coarser quantization sometimes stabilizes - cf. microchaos
  literature). Bisection brackets optimistic by ~±20%. TODO: harden headline
  numbers with conservative grid-scan (largest q with all finer q passing).
- 2026-06-07: fp32-theta experiment: fp32 measurements balance N=7 in both
  from-up and from-down(ulp(pi)=2.4e-7) representations. Empirical uniform-
  quantization scan: N=6 limit ~3e-5..1e-4 rad, N=7 ~1e-5..3e-5 rad =>
  basin/kappa prediction is ~2 orders CONSERVATIVE (worst-case H-inf vs
  broadband quantization error); empirical per-link factor ~5x, not 20x.
  Velocity side: kappa_v ~ 4-5 flat in N (input-channel disturbance rejection
  is O(1)) => dv requirement tracks basin alone (~4x/link), sensing >> actuation
  as the binding constraint for large N.
- 2026-06-07: SWING-UP N=2..5 done (agent C): casadi direct collocation
  (IMPLICIT dynamics: thetadd as decision var, M*thetadd=rhs constraint - far
  faster than symbolic solve), homotopy warm-start (append tip link), nonlinear
  predictor-corrector observer (FD lag destabilized high-N TVLQR even at zero
  quant), TVLQR tracking + catch. Swing durations 3.5/5.0/7.0/12.0 s for N=2..5.
  Thresholds (g=9.81, dt=0.01):
    N=2: dtheta 7.8e-2, dv 1.1e-1
    N=3: dtheta 8.5e-3, dv 3.4e-1
    N=4: dtheta 9.8e-5, dv 1.4e-2
    N=5: dtheta 6.9e-5, dv 2.5e-3
  Independently re-verified zero-quant success (4 seeds) for all N; closed-loop
  trajectories rendered to media/swingup_N{2..5}.gif. N=3 fails at dt=0.02.
- 2026-06-07: N=6,7 swing-up (bonus): round-1 collocation found untrackable
  whip-crack trajectories (|thetadd|~1840); round-2 adds |thetad|<=12 bound +
  warm starts (N=7 from N=6). In progress.
- 2026-06-07: round-2 result: 9/11 IPOPT local-infeasibility. Diagnosis:
  round-1 N=6 needs |thetad|~25.7 rad/s (~4 rev/s) to pump energy in ~12s;
  |thetad|<=12 made feasible set empty. SCOPE NARROWED to N=6 only (user
  request). Round 3 (N=6 only, 11 solves): allow thd<=18-22, amax=60, add
  accel-rate smoothness penalty (favor trackable member of feasible family),
  T in 14-18s. Goal: one trackable nominal -> TVLQR+catch -> animate.
- 2026-06-08: N=6 SWING-UP NEGATIVE RESULT (rigorous). Generated gentle,
  trackable-stiffness nominals via agent-C's implicit solver (max|thdd|~330,
  vs N=5's 421) at h=0.01, homotopy from N=5. ALL diverge closed-loop at a
  FIXED mid-swing instant (t~6.8s for T=13, t~7.4s for T=15). Root cause,
  confirmed by elimination:
    * divergence time INVARIANT to controller: swept r in {5,20,100,soft},
      gain caps {200..1500} -> identical div time. Not the feedback.
    * PERFECT-STATE TVLQR also diverges there; TVLQR gain explodes 383 ->
      1196 -> 36194 -> 44812 just before blowup.
    * at that instant all 6 links are within ~30deg (chain near-STRAIGHT);
      controllability ratio collapses (~1e-13). A straight chain ~ single
      rigid body => internal bending modes near-unactuatable from one pivot.
  Trajectory-level fix attempt (anti-alignment penalty to keep chain curved)
  backfired: more violent (thd 38-57, thdd 2047) without improving trackability.
  CONCLUSION: N=6 swing-up is at/beyond the edge for single-velocity-input
  collocation+TVLQR. Matches literature (published swing-up tops out at triple;
  we achieved N=5). N=6 BALANCE (Task 1) works fine: recovers from 6.9deg
  uniform lean (worst-case alternating basin ~3e-4 rad). Rendered
  media/balance_N6.gif. Per user: deliver balance animation + document finding.
