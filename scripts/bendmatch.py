#!/usr/bin/env python
"""Bend-order-matched N=6 swing-up (parallel-safe, _bendmatch suffix only).

1. Reproduce the committed N=5 bend order FROM SCRATCH via homotopy ladder
   N=2->3->4->5 (all target=zeros, coarse h=0.05 -> fine h=0.01). This yields the
   coherent single-direction swing (each link net -0.5 rev), matching
   results/trajectories/swingup_N5.npz.
2. Verify the reproduced N=5: net-rev per link, tip/base reach times, and an
   open-loop RK4 integration of the nominal a(t) ending near upright.
3. Lift to N=6 by homotopy_guess, sweep T x target at the proven N=6 settings
   (a_max=25, v_max=14, settle_frac=0.08), coarse->fine.
4. Test FULL-STATE trackability at dt=0.004 with optimize_n6.n6_perfect_state.
5. Save winning bundle -> repro/n6_controls_bendmatch.npz.

Run:  uv run python scripts/bendmatch.py [n_workers]
"""
import sys, os, pathlib, time
sys.path.insert(0, ".")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.trajopt import solve_swingup_implicit, homotopy_guess
# reuse proven helpers (do not modify that file)
from repro.optimize_n6 import (n6_perfect_state, target_options,
                               A_MAX, V_MAX, SETTLE_FRAC, SETTLE_BAND, G)

OUTDIR = pathlib.Path("repro")
WORKDIR = pathlib.Path("runs/bendmatch_work")
WORKDIR.mkdir(parents=True, exist_ok=True)

# N=6 search grid
N6_T = [13.0, 14.0, 15.0, 16.0]

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def save_traj(fn, sol, tgt):
    np.savez(fn, t=sol["t"], theta=sol["theta"], thetad=sol["thetad"],
             a=sol["a"], v=sol["v"], T=sol["T"], target=tgt)


def reach_times(t, th, thr=0.5):
    """First time each link gets within thr of upright (0)."""
    out = []
    for i in range(th.shape[1]):
        idx = np.where(np.abs(th[:, i]) < thr)[0]
        out.append(float(t[idx[0]]) if len(idx) else None)
    return out


def open_loop_check(n, sol):
    """Integrate nominal a(t) open-loop with RK4 from pi; return max final angle."""
    chain = Chain(n, G)
    t = sol["t"]; a = sol["a"]
    dt = float(t[1] - t[0])
    y = np.concatenate([np.full(n, np.pi), np.zeros(n)])
    for k in range(len(a) - 1):
        y = rk4_step(chain, y, a[k], dt)
    wrap = (y[:n] + np.pi) % (2 * np.pi) - np.pi
    return float(np.max(np.abs(wrap)))


def build_n5_from_scratch():
    """Homotopy ladder N=2..5, target=zeros, coarse(h=0.05)->fine(h=0.01)."""
    ladder_T = {2: 4.0, 3: 5.0, 4: 7.0, 5: 12.0}
    prev = None
    for n in range(2, 6):
        T = ladder_T[n]
        Kc, Kf = int(T / 0.05), int(T / 0.01)
        tgt = np.zeros(n)
        guess = homotopy_guess(dict(np.load(prev))) if prev else None
        # N=5 bend-order spec: a_max=50, v_max=12, settle_frac=0.12
        amax, vmax, sfrac = 50.0, 12.0, 0.12
        log(f"  ladder N={n} T={T} coarse K={Kc} ...")
        sc = None
        for s in range(4):
            sc = solve_swingup_implicit(n, T, Kc, g=G, a_max=amax, v_max=vmax,
                theta_target=tgt, settle_frac=sfrac, settle_band=SETTLE_BAND,
                seed=s, init_guess=guess, max_iter=1500, print_level=0, tol=1e-6)
            if sc["status"] == "solved":
                break
        if sc["status"] != "solved":
            raise SystemExit(f"ladder N={n} coarse failed")
        log(f"  ladder N={n} fine K={Kf} ...")
        sf = solve_swingup_implicit(n, T, Kf, g=G, a_max=amax, v_max=vmax,
            theta_target=tgt, settle_frac=sfrac, settle_band=SETTLE_BAND,
            init_guess=sc, max_iter=1500, print_level=0, tol=1e-7)
        if sf["status"] != "solved":
            raise SystemExit(f"ladder N={n} fine failed")
        fn = WORKDIR / f"n{n}_bendmatch.npz"
        save_traj(fn, sf, tgt)
        prev = str(fn)
    return prev, sf


def w_gen_n6(task):
    """Solve one N=6 candidate coarse->fine. task=(T,tname,tgt,seed5_path)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    T, tname, tgt, seed5 = task
    tgt = np.asarray(tgt)
    guess = homotopy_guess(dict(np.load(seed5)))
    Kc, Kf = int(T / 0.05), int(T / 0.01)
    amax, vmax, sfrac = A_MAX[6], V_MAX[6], SETTLE_FRAC[6]
    sc = solve_swingup_implicit(6, T, Kc, g=G, a_max=amax, v_max=vmax,
        theta_target=tgt, settle_frac=sfrac, settle_band=SETTLE_BAND,
        init_guess=guess, max_iter=1500, print_level=0, tol=1e-6)
    if sc["status"] != "solved":
        return None
    sf = solve_swingup_implicit(6, T, Kf, g=G, a_max=amax, v_max=vmax,
        theta_target=tgt, settle_frac=sfrac, settle_band=SETTLE_BAND,
        init_guess=sc, max_iter=1200, print_level=0, tol=1e-7)
    if sf["status"] != "solved":
        return None
    fn = WORKDIR / f"n6_{tname}_T{T}_bendmatch.npz"
    save_traj(fn, sf, tgt)
    return str(fn)


def w_eval_n6(fn):
    os.environ["OMP_NUM_THREADS"] = "1"
    ok, final, _ = n6_perfect_state(dict(np.load(fn)))
    return (ok, final, fn)


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else min(8, mp.cpu_count() - 2)
    log(f"workers={nworkers}")

    # ---- Step 1: reproduce N=5 from scratch ----
    log("Step 1: building N=2..5 homotopy ladder (target=zeros)...")
    n5_path, n5 = build_n5_from_scratch()
    th = n5["theta"]; t = n5["t"]
    netrev = (th[-1] - th[0]) / (2 * np.pi)
    maxv = float(np.max(np.abs(n5["v"])))
    cost = float(np.trapezoid(n5["a"] ** 2, t))
    log(f"  N=5 net rev/link = {np.round(netrev,3)}")
    log(f"  N=5 max|v|={maxv:.3f}  cost(integ a^2)={cost:.2f}")
    rt = reach_times(t, th, thr=0.5)
    log(f"  N=5 reach times (thr 0.5): {[round(x,2) if x else None for x in rt]}")
    ol = open_loop_check(5, n5)
    log(f"  N=5 open-loop RK4 final max angle = {np.degrees(ol):.3f} deg")
    all_same = np.all(np.abs(netrev - (-0.5)) < 0.05)
    log(f"  BEND-ORDER MATCH (all -0.5 rev): {all_same}; open-loop valid: {ol < 0.2}")

    # ---- Step 3: lift to N=6 ----
    tasks = [(T, tn, tgt, n5_path)
             for T in N6_T for (tn, tgt) in target_options(6)]
    log(f"Step 3: generating N=6 pool: {len(tasks)} tasks...")
    with mp.Pool(nworkers) as pool:
        n6 = [r for r in pool.map(w_gen_n6, tasks) if r]
    log(f"  {len(n6)} N=6 candidates solved")
    if not n6:
        raise SystemExit("no N=6 candidates solved")

    # ---- Step 4: full-state trackability at dt=0.004 ----
    log(f"Step 4: evaluating {len(n6)} N=6 (full-state, dt=0.004)...")
    with mp.Pool(nworkers) as pool:
        evals = pool.map(w_eval_n6, n6)
    for ok, final, fn in sorted(evals, key=lambda e: e[1]):
        log(f"  {'OK ' if ok else '   '}{pathlib.Path(fn).name}: "
            f"final={np.degrees(final):.3f} deg")
    trackable = sorted([e for e in evals if e[0]], key=lambda e: e[1])
    if not trackable:
        closest = sorted(evals, key=lambda e: e[1])[0]
        log(f"NO trackable N=6. closest final={np.degrees(closest[1]):.3f} deg "
            f"({pathlib.Path(closest[2]).name})")
        return

    ok, final, fn = trackable[0]
    log(f"  BEST trackable: {pathlib.Path(fn).name} final={np.degrees(final):.4f} deg")

    # ---- Step 5: save bundle ----
    _, _, bundle = n6_perfect_state(dict(np.load(fn)))
    out = OUTDIR / "n6_controls_bendmatch.npz"
    np.savez(out, **bundle)
    log(f"SAVED -> {out}  maxK={bundle['maxK']:.0f}  final={np.degrees(bundle['final']):.4f} deg")
    # reload confirm
    chk = dict(np.load(out))
    log(f"  reload keys: {sorted(chk.keys())}")


if __name__ == "__main__":
    main()
