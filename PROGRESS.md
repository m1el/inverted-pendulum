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
