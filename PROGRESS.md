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
- 2026-06-08: dt experiment (user suggestion): N=6 swing-up DOES work under
  FULL-STATE feedback at dt=0.004 (diverged at dt=0.01). Smaller dt keeps the
  high-gain (maxK~7e4) TVLQR in its funnel through the near-uncontrollable
  mid-swing. Realistic observer still can't feed 7e4 gains -> perfect-state only.
  Rendered media/swingup_N6_perfectstate.mp4.
- 2026-06-08: 2-STAGE REPRO built in repro/ (user request):
  stage1_n5.py (from-scratch N=5 gen+select) -> seed; stage2_n6.py (homotopy
  lift + DIRECT fine solve N=6, full-state trackability select at dt=0.004) ->
  n6_controls.npz; simulate_n6.py verifies + renders mp4; optimize_n6.py
  orchestrates both. Backed up the proven N=5 at repro/seeds/swingup_N5_GOOD.npz
  (T=12, all links -0.5 rev). Stage 2 from the good seed RELIABLY reproduces the
  trackable N=6 (final 0.069 deg, maxK=69448, VERIFICATION PASS) in ~1-2 min.
  KEY FINDING (from user's bend-order question): N=5 closed-loop score is
  necessary but NOT sufficient for trackable N=6 -- the N=5 BEND ORDER (relative
  left/right link bending, straight-crossings) must match a controllable
  topology. From-scratch N=5 seeds scoring 4/4 still failed to lift to trackable
  N=6; reproducing the exact bend order from scratch is an open search.
  Bug fixed: N=6 coarse->fine was unreliable (coarse h=0.05 failed ~8/9);
  direct fine solve from a good seed converges in ~60s.
- 2026-06-08: tried 3 approaches (parallel agents) to REMOVE the curated-seed
  "magic" (bend-order dependence), comparing least-magic/robust/fast:
  * smaller dt: REFUTED. dt=0.004 is a window ~[0.008,0.004], not "smaller
    better"; wrong bend order is genuinely uncontrollable at every dt (0.0005-
    0.01); maxK grows as dt shrinks. (shipped dt=0.004 is the window's fine edge)
  * minimal homotopy ladder: FALSIFIED. reproduces coarse class (-0.5 rev/link)
    but not the fine bend order (whippier, thd 27 vs 23); N=6 won't converge
    (0/3 ladder seeds). The pool/ranking/seed machinery is load-bearing.
  * CONTROLLABILITY-AWARE TRAJOPT: SUCCESS. soft one-sided floor on bend-mode
    excitation c(theta)=||M^-1(b*cos th)||^2_bend; from a NEUTRAL cold ladder
    (no curated seed) -> trackable N=6, final 0.018 deg, AND peak pivot accel
    7.3 m/s^2 (vs ~50 baseline, ~7x gentler -- removes the 5g catch spikes).
    Remaining 'magic' = a small soft-floor sweep (a few levels), parallel,
    principled. This ELIMINATES the seed dependence.
  Adopted as repro/generate_n6.py (canonical, seed-free, RECOMMENDED).
  simulate_n6.py now hands off to the upright balance LQR after swing-up (catch)
  so verification is robust to gentle/late arrival. Final repro/ layout:
  generate_n6.py (seed-free), stage2_n6.py (fast from seed), stage1_n5.py,
  optimize_n6.py (helpers), simulate_n6.py (verify+render), seeds/ (backup).
- 2026-06-08: extended controllability-aware approach to N=7 (repro/generate_nN.py,
  general-N). Seeded N=7 from our clean N=6 ctrb-aware trajectory (gentle,
  thd~20) to avoid whippy cold-ladder lifts. Result: N=7 NOT trackable.
  FRONTIER WALL, with diagnosis:
  * Even gentle, well-controlled N=7 nominals (cmin~0.81, thd~19) diverge under
    full-state TVLQR at EVERY dt in [0.004, 0.015].
  * NOT a gain-magnitude wall: a TVLQR weight sweep (R 0.1..50, maxK down to
    ~4000 -- BELOW N=6's 68k) still diverges at all (r,dt). So no gain stabilizes
    the N=7 trajectory.
  * CAUSE: the SCALAR controllability proxy (aggregate rigid-vs-bending split)
    that sufficed for N=6 is INSUFFICIENT for N=7. A 7-link chain has 6 bending
    modes; flooring the aggregate bend-excitation does not ensure EACH mode is
    individually excitable from the single pivot. One mode can be ~uncontrollable
    mid-swing while cmin looks healthy -> unstabilizable by any TVLQR.
  * Honest frontier: N<=5 full swing-up+balance (real observer); N=6 full-state
    swing-up, seed-free via scalar ctrb shaping; N=7 beyond this architecture.
    A per-mode (min-singular-value over all bending modes) controllability
    constraint would be the principled next step -- significantly harder NLP.
  repro/generate_nN.py works as the general-N generator (reproduces N=6).
- 2026-06-09: reconfiguration challenges, bend-order topology, and control jerk.
  LINK-FLIP (upright -> one link inverted -> upright; repro/flip_one_n6.py,
  N-generic; precision via repro/flip_quant.py): a there-and-back between two
  UNSTABLE equilibria, strictly harder than swing-up because it departs+returns to
  the dead-straight upright (near-uncontrollable) twice. Trackable N<=4 (all links),
  N=5 (middle links only, maxK 1.8e5 > N=6 swingup's 6.8e4), N=6 NONE (over the
  wall). Full-state quantization thresholds: dtheta collapses ~6-9x/link
  (1.9e-1 -> 4e-4 rad, N2->5), dv stays non-binding -- same precision law as
  balance/swing-up.
  FOLD-IN-HALF (upper N/2 links -> pi; N=6 [0,0,0,pi,pi,pi]; repro/fold_half.py):
  MUCH easier than a single-link flip and TRACKABLE at N=6 (dt=0.01, final 0.079
  deg, maxK 6.5e4) where the 1-link flip fails entirely. The folded target is less
  unstable and the coordinated crease keeps the chain bent/controllable throughout
  (cmin 0.85, never near-straight). KEY: difficulty is set by the PATH's
  controllability, not the link count or how "large" the reconfiguration looks.
  BEND ORDER -- made precise (scripts/bend_topology.py, written up PAPER 4.5). For
  absolute angles theta_i, joint bend angles beta_i = theta_{i+1}-theta_i; bend
  order = (W, F): W = per-link winding (unwrap delta-theta /2pi; homotopy invariant),
  F = per-joint sign-word of beta_i's zero-crossings (the fold/reversal sequence).
  Trackable iff F never sends all beta_i through 0 at once (c(theta) floor = smooth
  surrogate). REPRODUCIBILITY across all stable N=4,5,6 solutions: W is UNIVERSAL
  (-0.5 rev/link; other winding classes are exactly the untrackable ones ->
  trackability SELECTS the class, reproduced 100%). F is a CONSERVED FAMILY not a
  point: sign pattern largely shared within a method (4/5 ctrb-aware N=6 share
  (+,-,-,+,+)), crossing counts vary by a few. So reproduced as a topological class
  with a continuum of fine orders inside -- why the coarse class transfers across N
  but the fine (seed-encoded) order does not.
  CONTROL JERK (repro/swingup_n6_lowjerk.py, written up PAPER 4.6). N=6 swing-up
  jerk is FEEDBACK-induced (closed-loop a jerk RMS 19.8 vs feedforward 7.8 = 2.5x;
  peak 218), concentrated in the SWING (RMS 23) not the catch (1.3 -- error tiny
  there despite 6.8e4 gains). Tried to remove it with a jerk-penalizing input-
  augmented TVLQR (state += a, input = jerk u=adot, so a(t) is C^1): FAILS. Any
  penalty heavy enough to smooth the swing DIVERGES (the near-uncontrollable swing
  needs the bandwidth); the stable region is jerkier than baseline; time-varying
  (smooth-swing/permissive-catch) also diverges. Jerk = signature of high-bandwidth
  stabilization of a barely-controllable plant. ONLY lever is the path: feedback
  jerk amplification tracks controllability (fold cmin 0.85 -> 1.1x, swing-up cmin
  0.73 -> 2.5x). Low-jerk policy = more-controllable trajectory, not a smoother
  controller.
