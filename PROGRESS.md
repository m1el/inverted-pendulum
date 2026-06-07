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
- [ ] Swing-up: N=1 energy pumping + LQR catch; N≥2 trajectory optimization + TVLQR + catch
- [ ] Swing-up precision sweeps
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
