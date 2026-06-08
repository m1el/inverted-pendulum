#!/usr/bin/env python
"""STAGE 2 of the 2-stage swing-up optimisation: given an N=5 seed, optimise +
refine the N=6 swing-up and emit the controls bundle.

Because this lifts from ONE good N=5 seed, it solves only a handful of N=6
candidates -- so it can afford GENEROUS IPOPT iterations (no straggler/barrier
problem), which fixes the low-yield that broke the all-in-one from-scratch runs.

Steps:
  - homotopy-lift the N=5 seed to N=6 (new tip link shadows the last),
  - sweep terminal target x horizon, each coarse (h=0.05) -> fine (h=0.01),
  - keep the N=6 that is trackable under FULL-STATE TVLQR at dt=0.004
    (smallest upright error), save its nominal + gain schedule.

Usage:
  uv run python repro/stage2_n6.py [seed_npz] [n_workers]
  default seed: repro/seeds/swingup_N5_GOOD.npz   (the backed-up good N=5)

Output: repro/n6_controls.npz  (consumed by simulate_n6.py)
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp

from pendulum.trajopt import solve_swingup_implicit, homotopy_guess
from repro.optimize_n6 import n6_perfect_state  # proven full-state checker

G = 9.81
N6_T = [14.0, 15.0, 16.0]
A_MAX, V_MAX, SETTLE_FRAC, SETTLE_BAND = 25.0, 14.0, 0.08, 0.15
POOL = pathlib.Path("repro/pool_n6")
OUT = "repro/n6_controls.npz"

# parse args: optional seed path (non-numeric) and worker count (numeric)
_args = sys.argv[1:]
SEED = next((a for a in _args if not a.isdigit()), "repro/seeds/swingup_N5_GOOD.npz")
NW = int(next((a for a in _args if a.isdigit()), min(12, mp.cpu_count() - 2)))


def n6_targets():
    return [("zeros", np.zeros(6)),
            ("first+2pi", np.array([2 * np.pi] + [0.0] * 5)),
            ("first-2pi", np.array([-2 * np.pi] + [0.0] * 5))]


def gen6(task):
    """DIRECT fine N=6 solve from the seed (homotopy warm start).

    The homotopy guess from a good N=5 is already an excellent warm start, so a
    single fine (h=0.01) solve converges -- this matches the proven approach.
    A coarse->fine stage was tried and removed: the N=6 *coarse* (h=0.05) solve
    is unreliable and failed ~8/9 candidates, whereas the direct fine solve
    converges in ~60 s.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    T, tname, tgt = task; tgt = np.asarray(tgt)
    guess = homotopy_guess(dict(np.load(SEED)))
    Kf = int(T / 0.01)
    sf = solve_swingup_implicit(6, T, Kf, g=G, a_max=A_MAX, v_max=V_MAX,
        theta_target=tgt, settle_frac=SETTLE_FRAC, settle_band=SETTLE_BAND,
        init_guess=guess, max_iter=4000, print_level=0, tol=1e-7)
    if sf["status"] != "solved":
        return None
    fn = POOL / f"N6_{tname}_T{T}.npz"
    np.savez(fn, t=sf["t"], theta=sf["theta"], thetad=sf["thetad"],
             a=sf["a"], v=sf["v"], T=sf["T"], target=tgt)
    return str(fn)


def eval6(fn):
    os.environ["OMP_NUM_THREADS"] = "1"
    ok, final, _ = n6_perfect_state(dict(np.load(fn)))
    return (ok, final, fn)


def main():
    t0 = time.monotonic()
    def log(m): print(f"[{time.monotonic()-t0:6.1f}s] {m}", flush=True)
    if not pathlib.Path(SEED).exists():
        raise SystemExit(f"seed not found: {SEED}")
    if POOL.exists():
        for p in POOL.glob("*.npz"):
            p.unlink()
    POOL.mkdir(parents=True, exist_ok=True)

    log(f"STAGE 2: N=6 from seed {SEED}  (workers={NW})")
    tasks = [(T, tn, tg) for T in N6_T for (tn, tg) in n6_targets()]
    log(f"generating {len(tasks)} N=6 candidates (homotopy + coarse->fine)...")
    with mp.Pool(NW) as pool:
        cands = [r for r in pool.map(gen6, tasks) if r]
    log(f"  {len(cands)}/{len(tasks)} solved")
    if not cands:
        raise SystemExit("no N=6 candidates solved")

    log("evaluating full-state trackability at dt=0.004...")
    with mp.Pool(NW) as pool:
        evals = pool.map(eval6, cands)
    track = sorted([e for e in evals if e[0]], key=lambda e: e[1])
    log(f"  {len(track)}/{len(cands)} trackable")
    if not track:
        # report the closest miss to aid debugging
        closest = sorted(evals, key=lambda e: e[1])[0]
        log(f"  closest: {pathlib.Path(closest[2]).name} final={np.degrees(closest[1]):.2f} deg")
        raise SystemExit("no trackable N=6 from this seed")

    ok, final, fn = track[0]
    log(f"BEST {pathlib.Path(fn).name}: final={np.degrees(final):.3f} deg")
    _, _, bundle = n6_perfect_state(dict(np.load(fn)))
    np.savez(OUT, **bundle)
    log(f"SAVED {OUT}  (nominal {bundle['theta_nom'].shape}, "
        f"gains {bundle['K'].shape}, maxK={bundle['maxK']:.0f})")


if __name__ == "__main__":
    main()
