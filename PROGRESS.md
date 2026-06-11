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
- 2026-06-09: ARTIFACT ELIMINATION + REVERSE-FALL GENERATION => N=7 SOLVED
  (full-state). Session with two threads:
  (1) Nominal-consistency artifacts (repro/consistent_nominal.py): the eval
  pipeline linearly resamples the h=0.01 trapezoidal nominal to the sim grid;
  measured one-step defect ~1e-3 rad/step. Hermite dense output (exact node
  derivatives) cuts it ~6x and is ~9x closer to the true flow -- but tracks
  WORSE (N=6 floor0.6/0.7 track at dt=0.002/0.004 under linear, NEVER under
  Hermite; dies at t=9.9s in the low-ctrb patch). Linear's defect is a ~100Hz
  zero-mean sawtooth (self-dithers, averages out); Hermite's is smooth and
  forces the barely-controllable direction coherently. Defect SPECTRUM, not
  magnitude, decides survival; the dt window is not an interpolation artifact.
  No N=7 NLP candidate tracks under any resampling variant at any dt (incl.
  0.002). Mesh refinement h=0.005 (repro/refine_mesh.py): [see results/]
  (2) REVERSE-FALL generation (repro/reverse_fall.py, user idea: enumerate the
  PREDECESSORS of upright): dynamics are time-reversible, so a gently BRAKED
  FALL from upright+eps*(mixture of unstable modes), landed at the hang by an
  energy brake (|a|<=4 -- gentle settles in ~45s; saturated 25 never settles)
  + hang-LQR, time-reversed, is an EXACT swing-up nominal (zero defect, no NLP,
  no settle band). Arrival-shape family == departure-direction mixtures.
  N=6: 16/36 arrival shapes trackable at dt=0.004 (best 0.10deg). Trackable
  family = SLOW-mode arrivals (lam 3.6,1.5 both signs; all four fast modes
  fail). N=7: 0/38 at dt>=0.004 BUT 44/46 at dt=0.002 with slow-4-mode
  mixtures (14/46 at 0.003). Winner slow7: final 0.010deg, T=56.7s,
  max|a_ff|=18.9, max|v|=4.3, maxK=1.7e5, PASSES 16/16 at +-0.1pi +-0.5rad/s
  (same protocol as N=6). Saved repro/n7_controls.npz. So the N=7 wall was a
  property of the fast 17s NLP trajectory class at dt>=0.004, not of N=7:
  long gentle pumps (cmin~0.4-0.5, BELOW the 0.6+ floors!) at dt=0.002 are
  trackable. CAVEATS: dt=0.002 (finer than N=6's 0.004), maxK 1.7e5 (2.5x
  N=6), T~57s, whip ~38rad/s, links wind ~9 revs (so W=(-1/2)^n selection in
  PAPER 4.5 is a property of the fast NLP class, NOT of trackability per se --
  reverse-fall trajectories live in far-away winding classes and track fine).
  PITFALLS hit: (a) evaluating a defect-free nominal from its exact start is
  VACUOUS (error stays at machine eps; only proves local stability) -- always
  probe with perturbed starts; (b) reversed nominals start at theta=pi+2pi*k
  (fall winds up the links), so "start at the hang" tests must use the
  NOMINAL's winding chart, not pi (56 rad apparent error otherwise).
- 2026-06-09 (cont): MESH-REFINEMENT VERDICT + N=7 ROBUSTNESS/PRECISION.
  Refinement h=0.01 -> 0.005 (repro/refine_mesh.py): N=6 floor0.9 (untrackable
  at EVERY dt at h=0.01) becomes trackable at EVERY dt in [0.002, 0.012], both
  resamplers, best 0.002deg. So the N=6 "dt window" was a NOMINAL-TRUNCATION
  artifact: defect ~5e-5/step kills the knife-edge, no dt tuning needed.
  Refined N=7 NLP candidate still fails at all dt (fast-class wall is real).
  N=7 reverse-fall robustness (repro/n7_robustness.py, 8 seeds all-pass,
  results/n7_controls_robustness*.json): initial pert +-0.78 rad angles-only
  (+-0.65 with +-1.6x rates) -- same "25% of pi" class as N=6. Quantization
  (full-state convention, rates exact): swing-only dtheta 8.3e-4 rad,
  dv 8.7e-3 m/s; with 5s quantized balance hold dtheta 7.7e-6, dv 1.05e-4
  (N=6 revfall: 4.6e-6 / 1e-3). The sustained CATCH binds, as predicted; the
  measured N=7 dtheta sits ~3 orders above the basin/kappa worst-case floor
  (5.5e-9), matching the known ~2-orders self-dithering conservatism.
  WHIP: all 44 trackable N=7 falls whip (max|thetad| 37-62 rad/s, winding
  5.5-26 rev; gentlest slow3/slow7 ~37-38). Gentle N=7 needs whip-aware
  generation (brake shaping or NLP-swing + reversed-fall tail hybrid) -- open.
- 2026-06-10: NO-ROTATION N=7 -- EXISTS BUT CANNOT BE HELD (winding is essential).
  User asked for the classical no-rotation property (theta in [-1, 5.9], no link
  completes a turn; NB classical solutions overshoot the hang to ~5.6, so the
  corridor must be wider than [0,pi]). repro/norot_n7.py: slow (T=25) gentle
  (|thetad|<=12) ctrb-floored NLP SOLVES it -- cmin~1.0, the best-conditioned
  N=7 trajectory of the project (media/n7_norot_nominal.mp4, open-loop render;
  T=35 and one-jump h=0.005 refines fail; ladder 0.01->0.0075->0.005 works,
  repro/norot_refine_ladder.py, with IPOPT output_file logs in runs_tmp/ipopt/).
  BUT it is untrackable at every (dt, resampler, defect grade h=0.01/0.0075,
  R in [0.01,10] x QF x dt -- 0/24 weight sweep), open-loop replay departs at
  t=16.2s=0.65T. MECHANISM (windowed per-mode coupling, n7_ideas.permode):
    window      norot(untrackable)  revfall(trackable)
    t<0.65T     0.30                0.42
    0.65-0.9T   0.13                0.36
    >0.9T       0.12                0.27
  The no-rotation path's worst bending modes go ~2.5x deafer to the pivot
  exactly in the unstable final third (open-loop departure = same boundary);
  the winding reverse-fall keeps all 6 modes >=0.27 throughout. So at N=7,
  no-rotation trajectories EXIST (constructive NLP solution) but their
  neighborhoods are effectively unstabilizable from one pivot: WINDING IS
  ESSENTIAL to trackability, not an artifact of the fall generator. PAPER 4.5's
  "trackability selects W=(-1/2)^n" inverts at N=7: that class becomes the
  unstabilizable one. (Refinement robustness aside: refined N=6 floor0.9 funnel
  measured >= +-0.5 rad at dt 0.004 AND 0.01 vs 2/4@0.5 for unrefined --
  mesh refinement widens, not narrows, the funnel.)
- 2026-06-10 (cont): CONTROLLER-MODEL MISMATCH (user hypothesis) -- CONFIRMED.
  The TVLQR discretized (Ad,Bd) via expm of the FD-linearized continuous model
  differs from the actual RK4 step map by ~1% rel. mid-swing (repro logs;
  largest where rates are high, vanishing at the catch). Rebuilding the gains
  with RK4-JACOBIAN discretization (FD of rk4_step itself -- controller model
  == simulator): N=7 reverse-fall slow7 passes the FULL 16/16 +-0.1pi
  +-0.5rad/s protocol at dt=0.004 AND 0.003 (was: nothing above 0.002; new
  wall at 0.005). So the N=7 "dt=0.002 requirement" was ~half discretization
  artifact; N=7 now tracks at the N=6-era dt=0.004. No-rotation nominal still
  0/4 (its failure is modal physics, not model error). N=6 refined unchanged.
  BLIND TVLQR weight tuning (repro/tune_tvlqr_n6.py, 96 log-space samples,
  objective = bisected funnel on refined N=6 @ dt=0.01): NULL -- baseline and
  all top candidates saturate at eps=0.80 rad (the hang-basin cap, cf. the
  +-0.8 rad N=6 limit). Default weights were already funnel-optimal; the
  binding limit is basin geometry, not weights. Lesson: fix the model, not
  the knobs.
- 2026-06-10 (cont): RK4-CONSISTENT TVLQR ADOPTED; FLAGSHIP BUNDLES REBUILT.
  repro/rk4_tvlqr.py: (Ad,Bd) = FD of rk4_step itself (controller model ==
  simulator). N=7 flagship repro/n7_controls.npz = revfall slow7 @ dt=0.002,
  RK4-Jac gains: final 0.0078deg, funnel 0.65/0.78 rad, with-5s-hold dtheta
  2e-5 rad / dv 2.7e-4 m/s (swing-only 6.4e-4 / 1.0e-2). Alternates kept:
  _dt004_rk4 (16/16 swing at dt=0.004, 14/46 of native pool tracks, but
  arrival 2.8 deg > upright basin 1.4e-4 rad -> CATCH FAILS; swing-only
  artifact), _expm_dt0.002 (original). N=6 refined rebuilt with RK4 gains
  (dt=0.01): funnel 0.78/0.85 rad, hold dtheta 3.1e-5 / dv 1.9e-2; xva plot +
  video re-rendered (media/n6_refined_rk4_xva.png, swingup_N6_refined_rk4.mp4,
  PASS). Native dt=0.004 fall regen produced a DIFFERENT trajectory (chaotic
  fall, dt-sensitive) that does NOT track -- family membership is per-
  trajectory; resampling a known-good dt=0.002 fall is the reliable route.
  PAPER 4.1 note + 4.3 updated with flagship numbers and the coarse-dt catch
  caveat.
- 2026-06-10 (cont): N=5 GIVEN THE N=6 TREATMENT (user request). All pool_ctrb_n5
  floors refined h=0.01->0.005 + RK4-Jacobian gains: all three finished refines
  pass 16/16 at dt=0.01 (old bundle needed dt=0.004), final <=0.007deg, maxK
  DOWN 17.5k->14.2k, closed-loop a(t) sits on the feedforward (was visibly
  jittery -- the defect-fighting the user spotted in the xva plot). Flagship:
  repro/n5_refined_controls.npz (floor0.6_h005); media/n5_refined_xva.png +
  swingup_N5_refined.mp4 (PASS). The mesh+discretization cure now confirmed at
  N=5 and N=6; N=4 (maxK 3.7k) doesn't visibly need it.
- 2026-06-10 (cont): WINDING THEOREM, QUANTITATIVE FORM (in progress).
  Soft per-mode no-rotation NLPs (repro/permode_norot_n7.py): 0/4, all max-iter
  on feasible iterates paying huge hinge penalty (objective 1.5k-4.8k vs 62
  base) -- the corridor resists coupling. HARD windowed per-mode constraints
  (repro/permode_hard_n7.py; windowed RMS form: sum(v_j^T u)^2 >= c^2 sum u^Tu
  per 1.5s window, t/T in [0.55,0.95]), warm from the FEASIBLE plain norot
  solution: c=0.15, 0.20, 0.27 all -> IPOPT LOCAL-INFEASIBILITY certificates.
  AUDIT: the windowed-RMS metric is stricter than the peak |v^T u|/|u| quoted
  elsewhere; the plain norot baseline measures c0=0.0787 (worst window x mode).
  So certificates = "corridor cannot even ~2x its windowed coupling". Bracket
  probes c in {0.07 sanity, 0.09, 0.11, 0.13} running to locate the ceiling;
  c=0.07 < c0 must be feasible (formulation check). Trackable revfall class
  for comparison: peak worst-mode 0.27+ through the same windows.
- 2026-06-10 (cont): COUPLING CEILING BRACKETED -- winding theorem final form.
  Hard windowed-RMS per-mode floors, warm from feasible norot (c0=0.0787):
  c=0.07 solved (2.5 min, sanity PASS; untrackable, peak worst-mode 0.129);
  c=0.09 solved (56 min; untrackable, peak still 0.129); c=0.11, 0.13 max_iter
  6000 = boundary; c=0.15, 0.20, 0.27 certified locally INFEASIBLE. Corridor
  coupling ceiling ~0.1 (peak never exceeded 0.14 even mid-restoration) vs the
  trackable winding class's 0.27+ through the same windows: the requirement is
  UNPURCHASABLE within the no-rotation corridor, not merely expensive.
  PAPER 4.3 updated with the measured bracket. Solve-time signature: feasible
  cases return in minutes, boundary grinds to max_iter, infeasible certifies in
  hours -- the runtime itself diagnoses which side of the wall you are on.
- 2026-06-10 (cont): FULL-TURN N=7 (user idea) -- MINIMAL-WINDING CLASS IS
  COUPLING-RICH. Trajectory class: hang -> ONE coordinated full revolution ->
  catch on the second upright arrival (target -2pi/link, W=-1.5; corridor
  [-2pi-1, 5.9], whip<=16). repro/fullturn_n7.py, 12 solves (T in {12,16,20} x
  floor {0.35,0.5} x 2 seeds). Solved cases measure late-window worst-mode
  coupling 0.30-0.66 -- ABOVE the trackable threshold (0.27), up to 5x the
  no-rotation ceiling (~0.1), with floor 0.5 > floor 0.35 (aggregate floor and
  per-mode coupling correlate within-class). All gates 0/4 at h=0.01 nominals
  -- the known defect-grade signature (cf. N=6 floor0.9: fail-everywhere ->
  track-everywhere after h=0.005 refinement); refine-and-regate pass pending.
  MECHANISM (transversal crossing): the single fast pass through upright
  crosses the mode-deaf orientations in milliseconds (<< 1/lambda), vs the
  no-rotation approach parking near them for the whole final third.
  INTERPRETATION (user): one input = one shared channel carrying 7 messages;
  coupling = per-mode channel gain. Low coupling on one mode makes correcting
  it interference-limited (shouting to the deaf mode injects garbage into the
  rest -- why NO gain works, 0/24). Rotation = time-multiplexing/scanning the
  channel across orientations; data-rate view: deaf-mode blackouts must be
  short vs 1/lambda. Unifies the sensing-side (3.2 data-rate) and actuation-
  side information constraints; candidate discussion section for the paper.
- 2026-06-10 (cont): CONTINUATION STRATEGY RETIRED. Whip-ramp on the 56.7s
  reverse-fall (K=5667 NLPs): anchor converged only as a 9000-iter acceptable
  iterate; all TD rungs (30/16/12) failed to converge even at tol 1e-6/5000
  iters. Moot anyway: the full-turn class dominates the objective (12-16s,
  1 rev, whip 16, minutes per solve). Debug iterates kept in pool_cont_n7/.
- 2026-06-10 (cont): FULL-TURN v2/v3 -- THE BALANCE-CREEP AND THE ENERGY FLOOR.
  User spotted in the v1 nominal video that the chain NEARLY BALANCES at the
  first upright crossing instead of passing through -- min-effort cost rewards
  slow crossings, subverting the transversal mechanism; the v1 coupling window
  [0.65T,T] missed it (metric now widened to [0.3T,T]). Fix: ENERGY FLOOR
  E >= V_up + margin over the crossing (forces transversality by construction).
  v2 (cold, floor span [0.25,0.8], margins 20/40): 0/8, all max_iter -- the
  early floor start compressed the pump infeasibly. v3 (warm from v1 solutions,
  span [0.45,0.8], margins 15/30): 1/8 -- T12_f0.5_s0_v3e15 solves, worst-mode
  0.485 over the FULL post-pump window (creep gone, video verified), but
  actuator-hot: |a| 24.1/25, |v| 13.3/14, 27 m travel (the 12-s flip pays in
  actuation what the 57-s revfall pays in time). GATE 0/4 at h=0.01 as usual;
  mesh ladder h=0.005 -> 0.0025 running = the decisive trackability test for
  the fast class. If 0/4 persists at h=0.0025 (defect ~16x below the known-bad
  grade): fast trajectories are TVLQR-untrackable regardless of coupling and
  defect, and trackable N=7 = slow + winding (revfall near-optimal).
- 2026-06-10 (cont): v4/v5 BEND-BOUNDED FREE-TURN SWEEPS (user relaxations:
  whip<=25, a<=40, v<=20; "nice" = NO CONVOLUTION, bends |b_i|<=92/120deg;
  winding free). 30 solves total. FINDINGS:
  (1) The nice region is FULL of high-coupling N=7 trajectories: solved cases
  span coupling 0.19-0.59, beating the no-rotation ceiling (~0.1) everywhere.
  (2) ONE TURN IS OPTIMAL: 1-extra-rev cases cluster at 0.34-0.59; 2-rev at
  0.19-0.30. Mechanism: brief straight pass over the top is a harmless
  ms-blackout (transversal), while a 2nd revolution forces sustained coherent
  (phase-aligned, bend-bounded) rotation = rigid-mode motion that crowds out
  bend shimmy in the fixed T/whip budget. Winding necessary, MINIMAL winding
  optimal -- the user's one-flip intuition, with mechanism.
  (3) v5 modulo-2pi terminal (sin=0, cos>=0.5): winding count is formally
  emergent but basin-sticky -- solver never migrates classes; exploration
  happens at the sweep level. The optimizer's effort cost does NOT prefer
  winding (classical is cheaper); the per-mode COUPLING is what prefers it.
  (4) All gates 0/4 at h=0.01 (defect grade; fast-class defects ~30x worse
  per mesh than gentle classes -- h=0.005 leaves 1.4e-3). CHAMPION for the
  trackability ladder: v4_T12_r1_b2.1_s0 (1 rev, bend<=120deg, C=0.594).
- 2026-06-11: ONE-FLIP N=7 TRACKABLE -- the defect law holds one rung deeper.
  v3 champion (T=12 energy-floor full-turn, C=0.485) ladder: h=0.005 leaves
  defect 1.4e-3 (fails), h=0.0025 reaches 6.3e-5 (the known-good grade) ->
  GATE OPENS: 16/16 at +-0.02 rad +-0.032 rad/s (worst 0.029 deg), 11/16 at
  +-0.05, 6/16 at +-0.1, 0/16 at the full +-0.1pi protocol. maxK 7.2e5 (4x
  revfall). Bundle: repro/n7_oneflip_controls.npz (final 0.029 deg, RK4-Jac,
  dt=0.002; the h=0.0025 iterate was a max_iter 'failure' feasible to 1e-8 --
  gate-any-feasible-iterate policy is what made this result visible).
  THE NICE-ROBUST TRADE, quantified at N=7:
    slow 9-rev revfall: funnel +-0.78 rad, maxK 1.7e5, T=57s, ugly;
    one-flip:           funnel +-0.02 rad, maxK 7.2e5, T=12s, nice.
  Speed raises path instability rates -> shrinks funnel ~30x and quadruples
  gains. Winding necessary; one turn sufficient; robustness is what fast
  costs. Defect ladder: gentle classes clean at h=0.005; fast classes need
  h=0.0025 (defect scales with trajectory violence, ~30x per class).
- 2026-06-11 (cont): FUNNEL-OVER-TIME COMPARISON, N=4..7 flagships
  (media/funnel_compare_N4-7.png; repro/n{4,5,6}_funnel.npz +
  n7_{revfall,oneflip}_funnel.npz; funnel_n6.py per-phase worst-case kick
  bisection, TVLQR + balance catch). start/min/median (rad):
    N=4 0.43/8.4e-4/2.4e-2; N5ref 0.78/8.8e-5/8.6e-2; N6ref 0.80/6.3e-6/5.8e-2;
    N7 revfall 0.95/3.9e-6/6.9e-3; N7 oneflip 0.084/0 (PINCHES at one phase,
    < 1e-6 vs worst-case kick)/1.7e-4.
  Bottleneck narrows ~10x/link (basin/kappa law along whole trajectories);
  the two N=7 solutions differ by ~100x in funnel at every phase (slow+wound
  = wide tube, fast+nice = precision tube with one zero-tolerance point that
  end-to-end tracking survives only because upstream deviations arrive
  contracted). N5-refined has the widest median of the project.
  NB the script's 'arrive' marker is wrap-based and misreads winding
  trajectories (flags the first crossing).
- 2026-06-11 (cont): DOCS CONSOLIDATED; SMOOTH-SLOW EXPERIMENTS LAUNCHED.
  README: headline updated (N=7 solved, winding mandatory) + new "Results --
  N=7 one-flip" section (swingup_N7_oneflip.webp/mp4, n7_oneflip_xva.png,
  funnel_compare_N4-7.png). PAPER 4.3: added "minimal winding is optimal"
  (sweep table, transversality mechanism, energy-floor lesson) + "one-flip
  trackable at a measured price" (h=0.0025 defect law, funnel trade table);
  PAPER 6 end-state refreshed (N=7 solved twice over; repro list).
  Slow-N=7 closed-loop artifacts rendered: swingup_N7_revfall_rk4.mp4 (PASS
  0.31deg) + n7_revfall_xva.png -- its a(t) is a bang-bang smear (time-reversed
  tanh-saturated brake; never met an optimizer); noted possible link between
  its self-dithering control and its robustness (speculative).
  RUNNING: (a) continuation revival from the feasible anchor, whip 30->22->16,
  with FEASIBILITY-FIRST IPOPT exits (acceptable_constr_viol 1e-8, dual/compl
  huge, obj_change 1e-5, acceptable_iter 8 -- stops when feasible+stalled);
  (b) slow-flip hedge (one-flip class at T=20/24, whip<=14). Goal: the third
  corner -- smooth AND slow AND robust; full verification (gate, funnel, xva,
  video) queued on any winner.
- 2026-06-11 (cont): README/PAPER refreshed with the REVISED N=6 as flagship.
  README: new "Refined (h=0.005 + RK4-consistent gains)" subsection with
  swingup_N6_refined_rk4.webp/mp4 + n6_refined_rk4_xva.png (closed loop sits
  on the smooth feedforward; the lone ~1 m/s^2 correction at t~10.3s is the
  genuine low-ctrb patch); original ctrbaware section kept as historical with
  the dt=0.004 caveat. PAPER 4.2: refined-flagship paragraph (RK4 gains,
  16/16 @ dt=0.01, funnel 0.78/0.85, maxK 5.9e4; N=5 refresh noted).
  PAPER 4.6: retrospective caveat -- the 2.5x feedback-jerk figure was
  measured on the defective h=0.01/expm pipeline and conflates defect-
  fighting with real feedback; the refined nominal's on-nominal closed loop
  adds ~no jerk. Conclusions weakened accordingly (bandwidth still needed
  off-nominal; path controllability still the lever).
