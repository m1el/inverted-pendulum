# Swing-up (trajectory optimization + TVLQR) — notes

Task: bring the N-link chain from hanging (all theta_i = pi) to upright
(all theta_i = 0 mod 2pi) and stabilize, using only the pivot X-velocity.
Params: g = 9.81, dt = 0.01. Rods uniform, mass 1, length 1.

## Result summary

Swing-up achieved and stabilized for **N = 2, 3, 4, 5** (all pass the official
`protocol.swingup_trial` with zero quantization across the 4 default seeds,
horizon = T_swing + 20 s).

| N | swing T (s) | TVLQR preset | max dtheta (rad) | max dv (m/s) | joint dv @ dtheta=thr/10 |
|---|-------------|--------------|------------------|--------------|--------------------------|
| 2 | 3.5  | default | 7.80e-2 | 1.10e-1 | 1.50e-1 |
| 3 | 5.0  | vtight  | 8.50e-3 | 3.40e-1 | 2.64e-1 |
| 4 | 7.0  | default | 9.84e-5 | 1.38e-2 | 7.42e-3 |
| 5 | 12.0 | default | 6.88e-5 | 2.46e-3 | 3.98e-3 |

Headline values are the geometric mean of the bisection bracket
[lo, hi] in `results/swingup_traj.json` (9 log-bisection iters, 4 seeds, all
must succeed). `lo` = largest tested step that still succeeds, `hi` = smallest
that fails.

## Pipeline

1. **Dynamics in CasADi** (`pendulum/trajopt.py`): reimplemented the closed-form
   `M(theta) thetadd = -a*b*cos - (A*sin(dth))@thetad^2 + g*b*sin` symbolically.
   Cross-checked vs `Chain.thetadd` at 20 random states per N — max error
   ~1e-13 (machine precision). Two formulations:
   - *explicit* (`make_dynamics_fn`): builds thetadd via `ca.solve` of M;
     fine for N<=3 but the symbolic matrix inverse expands combinatorially and
     becomes pathologically slow for N>=4 (a single 120-node N=5 coarse solve
     took >5 min and did not converge).
   - *implicit* (`make_M_rhs_fn` + `solve_swingup_implicit`): keeps thetadd as a
     decision variable and enforces `M(theta)@thetadd = rhs` as an algebraic
     constraint. All expressions stay small/polynomial, so `expand=True` works
     and IPOPT Hessians are cheap. **This was the key enabler for N=5** — the
     same coarse solve dropped from >5 min (non-converging) to ~5 s, from random
     initial guesses.

2. **Direct collocation** over horizon T with K nodes:
   - explicit path: Hermite-Simpson (`solve_swingup`).
   - implicit path: trapezoidal (`solve_swingup_implicit`).
   - decision vars: theta[k], thetad[k], v[k] (pivot velocity), a[k] (pivot
     accel); v'=a. Boundary: theta(0)=pi, thetad(0)=0, v(0)=0;
     theta(T)=target (multiple of 2pi per link), thetad(T)=0, v(T)=0.
     |a|<=a_max, |v|<=12. Objective: integral of a^2 + small smoothness term.
   - A `settle_frac` tail constraint keeps theta within a band of the target
     over the last ~10-12% of the horizon (improves terminal accuracy, which
     matters a lot for the tiny N>=4 upright basin).
   - Mesh refinement: solve coarse (h=0.05 or 0.1) then warm-start a fine
     (h=0.01) re-solve.

3. **Open-loop check**: integrate the chain with `rk4_step` using the optimized
   a(t) (ZOH) from the exact initial state. Drift stays small mid-swing (N=2:
   ~2e-2 at 50%) and grows near upright — expected, the inverted equilibrium is
   open-loop unstable; TVLQR closes the loop.

4. **TVLQR** (`compute_tvlqr` in `pendulum/swingup_traj.py`): backward Riccati
   recursion along the nominal trajectory. State z=[theta, thetad, v], input a;
   each segment linearized by finite difference about (theta_k, thetad_k, a_k)
   and ZOH-discretized via matrix exponential. Three weight presets
   (default / tight / vtight) with increasing terminal weight; the per-N choice
   was made by closed-loop search (`scripts/select_*.py`).

5. **State estimator** — the single most important controller fix. The first
   attempt (low-pass finite difference of quantized angles) lagged the true
   velocity by ~half a step; for the high-gain TVLQR needed by N>=4 this
   destabilized the loop and diverged **even with zero quantization**. Replaced
   with a **nonlinear predictor-corrector observer**: each step it predicts the
   state forward one dt through the true (RK4) dynamics using the *known applied
   acceleration*, then corrects angle/velocity from the angle innovation
   (gains lp=0.7, lv=0.4). This is lag-free and lets N=4,5 track their
   trajectories.

6. **Catch / hand-off**: once near upright (all |wrap(theta)|<0.2 and
   |thetad_est|<2 and past the trajectory end) switch to the steady-state
   `KalmanBalancer` LQR from `pendulum/balance.py`, seeded with the current
   estimate. The KF design noise is floored (dtheta in [1e-4, 0.05], dv>=1e-3)
   so the filter ARE stays well-conditioned across the whole threshold sweep;
   the *actual* sim quantization is unchanged.

## Balance basin (why high N is hard)

Measured largest alternating-perturbation tilt the balance LQR can recover from
(perfect observation, `scripts/tune_balance.py`):

| N | upright basin (rad) |
|---|---------------------|
| 2 | 0.30   |
| 3 | 0.063  |
| 4 | 0.014  |
| 5 | 0.0033 |

The basin collapses ~5x per added link, and the max-dtheta swing-up thresholds
track it closely (0.078 / 0.0085 / 0.000098 / 0.000069). The TVLQR must
therefore deliver the terminal state into a window that shrinks just as fast,
which is why the angle-measurement precision requirement is so severe for N=5
(~7e-5 rad).

## What worked / what failed per N

- **N=2**: easy. Many trajectories solve from random guesses; default TVLQR
  preset. dtheta tolerance ~0.08 rad.
- **N=3**: explicit Hermite-Simpson from random guesses converges; needs the
  `vtight` (heavy terminal weight) TVLQR preset to land in the smaller basin.
- **N=4**: random multistarts mostly failed to converge; **homotopy warm-start**
  from the N=3 solution (append a tip link, `homotopy_guess`) converged reliably.
  Closed-loop initially diverged until the predictor-corrector observer was in.
- **N=5**: the genuinely hard one.
  - Explicit collocation was too slow to be usable (matrix-inverse expression
    blow-up) — abandoned after a single coarse solve failed to converge in >5 min.
  - Switching to **implicit dynamics** made coarse N=5 solves trivial (~5 s) and
    convergent from random seeds.
  - Staged refinement h=0.05 -> 0.02 -> 0.01 with a gradually tightened settle
    band; swept a_max in {25, 35, 50} (gentler trajectories track better) and
    several horizons/targets. Many candidates pass 4/4 closed-loop; selected
    cand_6 (T=12 s, zeros target, default preset, worst tail ~2.5e-5 rad).

## dt scaling (N=2, 3)

The controller indexes the nominal trajectory by wall-clock time
(`k = round(t / dt_nom)`), so it is robust to sim dt != trajectory grid step.
Zero-quantization swing-up results:

| N | dt=0.005 | dt=0.01 (nominal) | dt=0.02 |
|---|----------|-------------------|---------|
| 2 | success  | success | success |
| 3 | success  | success | fail    |

N=3 fails at the coarse dt=0.02 (the larger step + smaller upright basin make
the catch miss); N=2 is robust across the range. (An earlier version indexed the
feedforward by call count and failed at every non-nominal dt; switching to
time-indexing fixed it and left all nominal-dt results unchanged.)

## Caveats

- Open-loop trackability metrics for N>=4 are dominated by the terminal
  upright instability and are not a useful candidate ranking signal; selection
  was done by closed-loop simulation (and a cheap perfect-state TVLQR
  pre-screen for N=5).
- The controller bails out (commands the held velocity) if the estimate becomes
  non-finite, so a diverged run is recorded as a clean failure rather than
  raising on `int(NaN)` inside the quantizer — important for the bisection to
  probe large dv/dtheta safely.
- The shared machine was heavily oversubscribed (load 80-150 on 64 cores from
  other jobs) during much of the run, so wall-clock solve times above are
  inflated; the implicit solver is fast in isolation.

## Files

- `pendulum/trajopt.py` — CasADi dynamics + explicit & implicit collocation,
  `homotopy_guess`.
- `pendulum/swingup_traj.py` — `SwingupController` (TVLQR + predictor-corrector
  observer + KalmanBalancer catch), `make_controller_factory`, presets.
- `scripts/` — `gen_parallel.py`, `solve_highN.py`, `solve_n5.py` (candidate
  generation), `select_traj.py`, `select_n5.py` (closed-loop selection),
  `thresholds.py` (precision sweeps), `tune_balance.py` (balance basins),
  `finalize.py` (merge).
- `results/trajectories/swingup_N{2,3,4,5}.npz` — nominal trajectories
  (t, theta, thetad, a, v, T, target).
- `results/swingup_traj.json` — per-N durations, zero-quant success, thresholds,
  joint points, selected preset.
