#!/usr/bin/env python
"""STAGE 1 of the 2-stage swing-up optimisation: generate + select an N=5 seed
from scratch (no pre-existing trajectory), to feed Stage 2 (repro/stage2_n6.py).

Steps:
  - fast coarse homotopy ladder N=2->3->4 (warm-start source),
  - generate an N=5 candidate POOL: sweep horizon x terminal target, each
    warm-started from N=4 (homotopy), coarse (h=0.05) -> fine (h=0.01),
  - RANK candidates by closed-loop robustness with the realistic observer
    controller (seeds passed of 4, then smallest tail error),
  - save the winner to the seed path (default repro/seed_N5.npz).

CAVEAT (important): a high N=5 closed-loop score is necessary but NOT sufficient
for Stage 2 to find a trackable N=6 -- the N=5's *bend order* (relative left/
right link bending) must also match a controllable topology. The known-good
committed seed (T=12, all links ~-0.5 rev, coordinated near-aligned shimmy) is
backed up at repro/seeds/swingup_N5_GOOD.npz; if a from-scratch seed does not
yield a trackable N=6 in Stage 2, fall back to that one.

Usage:  uv run python repro/stage1_n5.py [out_seed.npz] [n_workers]
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step, simulate
from pendulum.trajopt import solve_swingup_implicit, homotopy_guess

G = 9.81
DT_SEL = 0.01
A_MAX_N5, V_MAX_N5, SETTLE5 = 60.0, 12.0, 0.12
N5_T = [11.0, 13.0, 15.0]
POOL = pathlib.Path("repro/pool_n5")

_args = sys.argv[1:]
OUT = next((a for a in _args if not a.isdigit()), "repro/seed_N5.npz")
NW = int(next((a for a in _args if a.isdigit()), min(16, mp.cpu_count() - 2)))

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def n5_targets():
    return [("zeros", np.zeros(5)),
            ("first+2pi", np.array([2 * np.pi] + [0.0] * 4)),
            ("first-2pi", np.array([-2 * np.pi] + [0.0] * 4)),
            ("all+2pi", np.full(5, 2 * np.pi)),
            ("alt2pi", np.array([2 * np.pi * (i % 2) for i in range(5)], float))]


def build_ladder_to_4():
    prev = None
    for n in range(2, 5):
        T = {2: 4.0, 3: 5.0, 4: 7.0}[n]; K = int(T * 25)
        guess = homotopy_guess(dict(np.load(prev))) if prev else None
        sol = None
        for s in range(4):
            sol = solve_swingup_implicit(n, T, K, g=G, a_max=A_MAX_N5, v_max=V_MAX_N5,
                theta_target=np.zeros(n), settle_frac=0.1, settle_band=0.2,
                seed=s, init_guess=guess, max_iter=1500, print_level=0, tol=1e-6)
            if sol["status"] == "solved":
                break
        if sol["status"] != "solved":
            raise SystemExit(f"ladder N={n} failed")
        fn = POOL / f"_ladder_N{n}.npz"
        np.savez(fn, t=sol["t"], theta=sol["theta"], thetad=sol["thetad"],
                 a=sol["a"], v=sol["v"], T=T, target=np.zeros(n))
        prev = str(fn)
    return prev


def gen5(task):
    """Coarse->fine N=5 warm-started from N=4. task=(T,tname,tgt,n4path)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    T, tname, tgt, n4 = task; tgt = np.asarray(tgt)
    guess = homotopy_guess(dict(np.load(n4)))
    Kc, Kf = int(T / 0.05), int(T / 0.01)
    sc = solve_swingup_implicit(5, T, Kc, g=G, a_max=A_MAX_N5, v_max=V_MAX_N5,
        theta_target=tgt, settle_frac=SETTLE5, settle_band=0.15,
        init_guess=guess, max_iter=1500, print_level=0, tol=1e-6)
    if sc["status"] != "solved":
        return None
    sf = solve_swingup_implicit(5, T, Kf, g=G, a_max=A_MAX_N5, v_max=V_MAX_N5,
        theta_target=tgt, settle_frac=SETTLE5, settle_band=0.15,
        init_guess=sc, max_iter=2000, print_level=0, tol=1e-7)
    if sf["status"] != "solved":
        return None
    fn = POOL / f"N5_{tname}_T{T}.npz"
    np.savez(fn, t=sf["t"], theta=sf["theta"], thetad=sf["thetad"],
             a=sf["a"], v=sf["v"], T=sf["T"], target=tgt)
    return str(fn)


def rank5(fn):
    """Closed-loop robustness of an N=5 candidate. Returns (oks,-worst,fn)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    from pendulum.swingup_traj import SwingupController, compute_tvlqr, TVLQR_PRESETS
    chain = Chain(5, G)
    wrap = lambda a: (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
    d = dict(np.load(fn)); th, td, av, vv = d["theta"], d["thetad"], d["a"], d["v"]
    Ks = compute_tvlqr(chain, d, DT_SEL, **TVLQR_PRESETS["default"])
    y = np.concatenate([th[0], td[0]]); v = 0.0
    for k in range(len(th) - 1):
        z = np.concatenate([y[:5], y[5:], [v]]); zn = np.concatenate([th[k], td[k], [vv[k]]])
        a = av[k] - float((Ks[k] @ (z - zn))[0]); y = rk4_step(chain, y, a, DT_SEL); v += a * DT_SEL
        if not np.isfinite(y).all() or np.max(np.abs(y[:5] - th[k + 1])) > 0.3:
            return (0, -np.inf, fn)
    T = float(d["t"][-1]); nsteps = int((T + 20.0) / DT_SEL); tail = int(5.0 / DT_SEL)
    oks = 0; worst = 0.0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        y0 = np.zeros(10); y0[:5] = np.pi + rng.uniform(-1e-3, 1e-3, 5)
        c = SwingupController(chain, DT_SEL, 0.0, 0.0, fn, tvlqr_kw="default")
        res = simulate(chain, c, y0, DT_SEL, nsteps, record=True)
        te = np.max(np.abs(wrap(res["traj"]["y"][:, :5][-tail:])))
        oks += int(te < 0.3); worst = max(worst, te)
    return (oks, -worst, fn)


def main():
    if POOL.exists():
        for p in POOL.glob("*.npz"):
            p.unlink()
    POOL.mkdir(parents=True, exist_ok=True)

    log("STAGE 1: building N=2..4 ladder...")
    n4 = build_ladder_to_4()
    tasks = [(T, tn, tg, n4) for T in N5_T for (tn, tg) in n5_targets()]
    log(f"generating {len(tasks)} N=5 candidates (workers={NW})...")
    with mp.Pool(NW) as pool:
        cands = [r for r in pool.map(gen5, tasks) if r]
    log(f"  {len(cands)} N=5 candidates solved")
    if not cands:
        raise SystemExit("no N=5 candidates solved")

    log("ranking by observer closed-loop robustness (4 seeds)...")
    with mp.Pool(NW) as pool:
        ranked = sorted(pool.map(rank5, cands), reverse=True)
    for oks, nworst, fn in ranked[:5]:
        log(f"  {pathlib.Path(fn).name}: ok={oks}/4 worst={-nworst:.4f}")
    oks, nworst, fn = ranked[0]
    d = np.load(fn)
    np.savez(OUT, **{k: d[k] for k in d.files})
    net = ((d["theta"][-1] - d["theta"][0]) / (2 * np.pi)).round(2)
    log(f"SAVED seed -> {OUT}  ({pathlib.Path(fn).name}, ok={oks}/4, net rev={net})")
    log("NOTE: verify Stage 2 yields a trackable N=6 from this seed; if not, "
        "use repro/seeds/swingup_N5_GOOD.npz (bend order matters, not just score).")


if __name__ == "__main__":
    main()
