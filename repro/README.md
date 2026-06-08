# Minimal reproduction: 6-link pendulum swing-up

Two ways to produce a trackable N=6 swing-up, then a verify+animate consumer.
The **recommended** path needs no curated seed and gives gentle accelerations.

```bash
# RECOMMENDED — controllability-aware, SEED-FREE (~5 min)
uv run python repro/generate_n6.py            # -> repro/n6_controls.npz
uv run python repro/simulate_n6.py out.mp4    # verify (PASS) + render mp4

# FAST alternative — from the backed-up good N=5 seed (~1-2 min)
uv run python repro/stage2_n6.py repro/seeds/swingup_N5_GOOD.npz
uv run python repro/simulate_n6.py out.mp4
```

## Why two paths — and the "magic" (bend order)

N=6 swing-up only works when the trajectory keeps the chain *controllable* — a
coordinated left/right bending ("shimmy") that never goes dead-straight. A chain
that goes near-straight mid-swing is near-uncontrollable from the single pivot,
forcing TVLQR gains ~7e4 that can't be tracked. Whether a trajectory has this
good **bend order** is the crux. We tried three ways to remove the dependence on
a hand-curated good seed:

| approach | removes seed dependence? | result |
|---|---|---|
| smaller timestep dt | **no** | dt=0.004 is a *window* (≈[0.008,0.004]), not "smaller is better"; wrong bend order is genuinely uncontrollable at every dt |
| minimal homotopy ladder | **no** | reproduces the *coarse* class (−0.5 rev/link) but not the *fine* bend order; N=6 won't even converge (0/3 seeds) |
| **controllability-aware trajopt** | **yes** | adds a soft floor on bend-mode excitation → produces a trackable bend order from a neutral start, **and** cuts peak pivot accel ~7× |

So the bend-order property is real and largely irreducible — but it can be
**produced by construction** with a controllability objective, instead of
hand-curated. That is `generate_n6.py`.

## `generate_n6.py` — controllability-aware, seed-free  (RECOMMENDED)

- Builds a NEUTRAL cold homotopy ladder N=2→5 (target zeros, no curated seed).
- Lifts to N=6 with an added **one-sided soft floor** on the bend-mode
  excitation `c(θ) = ‖M⁻¹(b⊙cosθ)‖²_bend` (collapses exactly when the chain
  goes near-straight). Sweeps a few floor levels in parallel; keeps the first
  trackable one (smallest upright error).
- Output (verified): final **0.018°**, peak pivot accel **7.3 m/s²** (vs ~50 for
  the seed path), maxK 68110, dt=0.004.

## `stage2_n6.py` — from a given N=5 seed  (FAST)

Homotopy-lifts an N=5 seed to N=6 (direct fine solve per target × horizon),
keeps the dt=0.004-trackable one. From the backed-up good seed it is
deterministic and fast (~1–2 min) but depends on that seed's bend order.
`stage1_n5.py` regenerates a from-scratch N=5 seed (note: a from-scratch seed's
*score* can be high yet its bend order may not lift to a trackable N=6 — use the
backed-up `repro/seeds/swingup_N5_GOOD.npz` if so).

## `simulate_n6.py` — verify + animate

Closed-loop swing-up (TVLQR on the nominal) **then catch** (hand off to the
upright balance LQR and hold). Asserts the chain balances the final 5 s
(`VERIFICATION: PASS`, exit 0); renders an H.264/yuv420p mp4. The catch handoff
makes the check robust to how late a gentle swing-up arrives upright.

## Full-state caveat (both paths)

The controller is FULL-STATE feedback (idealised sensing). Even controllability-
aware, the mid-swing chain is near-uncontrollable enough to need maxK~7e4; no
realistic angle-only observer can feed gains that large. The decisive enabler
for N=6 vs N=5 is the finer timestep dt=0.004.

Depends only on the verified library in `../pendulum/`.
