# N=1 swing-up — precision (quantization) thresholds

Controller: `pendulum/swingup1.py::SwingUp1`. Sweep script: `scripts/swingup_n1.py`.
Raw numbers: `results/swingup1.json`. Horizon 25 s, 4 seeds per protocol point,
log-space bisection (`protocol.threshold_bisect`, 9 iters). Success criterion
(`protocol.swingup_trial`): in the final 5 s of the horizon `|wrap(theta)| < 0.3`.

## Controller

Two phases with hysteretic switching:

1. **Energy pumping.** Rod energy about the resting pivot
   `E = (1/6) thetad^2 + (g/2) cos(theta)`, target `E* = g/2`. Because
   `dE/dt = -(1/2) a thetad cos(theta)` (a = pivot acceleration), the pumping law

       a_pump = k_E * (E - E*) * thetad * cos(theta)

   drives E toward E* (sign chosen so energy moves the right way). It is added to
   cart-regulating terms `-k_v*v - k_x*x` that keep the cart from drifting
   without disturbing the pump (they vanish at the stationary cart state). The
   sum is saturated at `|a| <= a_max`.

   **Symmetry-breaking kick.** At exactly theta=pi, thetad=0 the pumping term is
   identically zero, so the rod stalls hanging (observed: 13 s of no motion before
   the 1e-3 seed grew). A fixed `a = a_max` nudge is applied while
   `|thetad| < kick_rate` and `|wrap(theta)| > pi - kick_angle` to break the
   deadlock. With it, swing-up reaches upright in ~0.76 s.

2. **LQR catch.** When `|wrap(theta)| < catch_angle` (0.30) and
   `|thetad| < catch_rate` (3.0), hand off to `balance.KalmanBalancer` (LQR +
   steady-state Kalman filter on the linearized upright model), seeded with the
   current angle/rate estimate. Release back to pumping if `|wrap(theta)|`
   exceeds `release_angle` (0.55) — hysteresis prevents chatter.

The controller sees only the quantized angle. `thetad` is estimated by finite
difference of the unwrapped measurement (the new sample is unwrapped onto the
branch of the running estimate so the pi -> 0 transit is handled), with a
light low-pass (`filt=0.5`). Energy is computed from these estimates.

Default gains: `k_E=3, k_v=0.6, k_x=0.05, a_max=30`. At dtheta=dv=0 swing-up is
robust across all seeds and reaches upright in <1 s.

## Headline thresholds (g=9.81, dt=0.01)

| quantity | bracket | headline (geom. mean) |
|---|---|---|
| max dtheta (dv=0) | [0.0236, 0.0244] | **0.024 rad** (~1.4 deg) |
| max dv     (dtheta=0) | [0.0535, 0.0552] | **0.054 m/s** |

Joint points (dv threshold at reduced dtheta):
- dtheta = max_dtheta/10 = 0.0024 -> max dv = 0.044 m/s
- dtheta = max_dtheta/3  = 0.0080 -> max dv = 0.054 m/s

The dv threshold is nearly unchanged when dtheta is shrunk, and dtheta at its own
limit already implies dv well below its limit, so the two constraints are largely
**independent** (the joint feasible region is close to a rectangle, not a sharp
trade-off corner).

## Comparison with balance-only (KalmanBalancer, same g, dt)

| quantity | balance-only | swing-up | ratio (bal/swing) |
|---|---|---|---|
| max dtheta | 0.524 ([0.516, 0.533]) | 0.024 | ~22x |
| max dv     | 0.224 ([0.221, 0.228]) | 0.054 | ~4x |

Swing-up is far more demanding than balancing for **angle** precision, modestly
more demanding for velocity. The bottleneck is the **catch**: the rod arrives at
the top with large angular velocity, and the finite-difference rate estimate
(noise ~ dtheta/dt) plus the LQR's small linear region give little margin to
arrest an energetic arrival. Balancing starts at rest near upright, so it
tolerates much coarser angle quantization (the Kalman filter averages noise over
a long settle). The pumping phase itself is robust to coarse quantization; coarse
dtheta breaks the catch, not the swing.

## dt scaling (g=9.81)

| dt | max dtheta | max dv |
|---|---|---|
| 0.002 | 0.0070 | 0.0164 |
| 0.005 | 0.0164 | 0.0350 |
| 0.010 | 0.0240 | 0.0544 |
| 0.020 | 0.0075 | 0.0397 |

- **max dtheta is non-monotonic, peaking near dt=0.01.** For fine dt the FD rate
  noise scales as dtheta/dt, so tolerable dtheta scales roughly *with* dt
  (0.002->0.007, 0.005->0.016, 0.01->0.024 is close to linear). For coarse dt
  (0.02) the discrete LQR catch loses stability margin and the per-step jump in
  state during the fast arrival is large, pulling the dtheta limit back down.
- **max dv grows with dt** up to dt=0.01 then is roughly flat: a velocity command
  quantum dv injects an acceleration error ~ dv/dt over one step, so a larger dt
  dilutes the same dv into a smaller acceleration disturbance — coarser dv is
  tolerable as dt grows, until the catch-stability ceiling caps it at dt=0.02.

## g scaling (dt=0.01)

| g | max dtheta | max dv |
|---|---|---|
| 4.905 | 0.0175 | 0.0410 |
| 9.81  | 0.0240 | 0.0544 |
| 19.62 | 0.0329 | 0.0579 |

Both thresholds **increase with g**: higher gravity means stronger restoring
authority near upright and faster, more energetic dynamics that the energy-based
pump and LQR exploit, so a coarser sensor/actuator still suffices. The growth is
sub-linear (dtheta roughly ~ sqrt(g): 0.0175, 0.024, 0.033 over a 4x g range).

## Failure modes near threshold

- **Coarse dtheta:** rate estimate at the energetic top is too noisy; the catch
  either triggers on a bad estimate and the LQR diverges, or the rod overshoots
  through the catch window and the cycle repeats without ever settling within the
  final-5 s window.
- **Coarse dv:** limit-cycle jitter about upright during the balance phase; once
  the velocity command quantum exceeds what the LQR needs for fine corrections,
  the rod oscillates with amplitude > 0.3 rad and fails the tail criterion.
- **Pumping phase** is essentially never the binding constraint — it succeeds at
  far coarser quantization than the catch/balance phase.

## Caveats

- Brackets are from 4 seeds (per the protocol) and 9 bisection iters; the
  geometric-mean headline is good to ~2%, but the true threshold could shift a
  little with more seeds. Thresholds are for this specific controller/gains; a
  catch tuned for energetic arrivals (e.g. a wider-region nonlinear catch or
  velocity feed-forward) would likely raise the dtheta limit.
- a_max=30 makes the swing very aggressive (near bang-bang); with a smaller
  a_max the swing takes longer but the qualitative thresholds and scaling hold.
