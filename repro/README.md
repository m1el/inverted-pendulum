# Minimal reproduction: 6-link pendulum swing-up (2-stage)

Two-stage optimisation, then a verify+animate consumer.

```bash
# Stage 1: generate + select an N=5 seed from scratch  -> repro/seed_N5.npz
uv run python repro/stage1_n5.py

# Stage 2: lift + refine to a trackable N=6           -> repro/n6_controls.npz
uv run python repro/stage2_n6.py [seed.npz]

# Verify (asserts upright) + render mp4
uv run python repro/simulate_n6.py out.mp4

# Or the whole from-scratch pipeline (Stage 1 then Stage 2):
uv run python repro/optimize_n6.py
```

## Stages

**Stage 1 — `stage1_n5.py`**  (N=5 seed, from scratch)
- Fast coarse homotopy ladder N=2→3→4, then an N=5 candidate pool by sweeping
  horizon × terminal target (the ±2π homotopy classes), each warm-started from
  N=4, coarse (h=0.05) → fine (h=0.01).
- Ranks candidates by closed-loop robustness with the realistic observer
  controller (seeds passed of 4, then smallest tail error); saves the winner.
- Uses N=5 actuator authority a_max=60 (cold/weak authority gives poor seeds).

**Stage 2 — `stage2_n6.py`**  (N=6 from a seed)
- Homotopy-lifts the N=5 seed to N=6 and does a **direct fine solve** (h=0.01)
  per terminal target × horizon. (A coarse→fine stage was tried and removed:
  the N=6 coarse solve is unreliable; the direct fine solve from a good seed
  converges in ~60 s.)
- Keeps the N=6 trackable under FULL-STATE TVLQR at dt=0.004 (smallest upright
  error), saves nominal + gain schedule to `n6_controls.npz`.
- Few solves from one seed → generous IPOPT iterations, no straggler stalls.

**`simulate_n6.py`** — closed-loop sim with the verified RK4 integrator + saved
gains; asserts the chain balances the final 5 s (`VERIFICATION: PASS`, exit 0);
renders an H.264/yuv420p mp4.

## The bend-order caveat (why a seed is backed up)

Stage 2's success depends on the N=5 seed's **bend order** — the relative left/
right link bending (which joints bend which way, when each crosses the straight/
aligned point). A high N=5 closed-loop *score* is necessary but NOT sufficient:
from-scratch N=5 seeds that score 4/4 can still carry a bend order that does not
lift to a trackable N=6. The known-good seed (T=12, all links ≈ −0.5 rev, a
coordinated near-aligned shimmy) is backed up at:

    repro/seeds/swingup_N5_GOOD.npz

If Stage 1's from-scratch seed does not yield a trackable N=6, run Stage 2
against the backed-up seed (the reliable path):

    uv run python repro/stage2_n6.py repro/seeds/swingup_N5_GOOD.npz

## Full-state caveat

The controller is FULL-STATE feedback (idealised sensing). The near-straight
mid-swing chain is near-uncontrollable from one pivot, forcing TVLQR gains
~7e4; no realistic angle-only observer can feed gains that large. The decisive
enabler for N=6 (vs N=5's dt=0.01) is the finer timestep dt=0.004.

Depends only on the verified library in `../pendulum/` (dynamics, simulator,
collocation solver). Stage-2-from-good-seed runs in ~1–2 min and is reliable;
from-scratch Stage 1 is a longer, lower-yield search.
