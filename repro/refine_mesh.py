#!/usr/bin/env python
"""Mesh-refine the candidate nominals (artifact elimination, stage 2).

The one-step-defect diagnostic (repro/consistent_nominal.py) shows that after
fixing the resampling (Hermite dense output + midpoint ZOH) the remaining
nominal defect is the trapezoidal collocation's own O(h^2) truncation error at
h=0.01. This script re-solves each pool candidate at finer meshes (h=0.005,
then h=0.0025), warm-started from the coarser solution, keeping the SAME
controllability floor (so the bend order is preserved, not re-discovered).
Refined trajectories go to repro/pool_refined_n{N}/floor{F}_h{H}.npz.

Run:  uv run python repro/refine_mesh.py [n_workers] [--n6only|--n7only]
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp

from repro.generate_nN import solve_ctrb_aware  # general-n ctrb-aware collocation

HS = (0.005,)            # pass --fine to chain a second refinement to h=0.0025
W_CTRB = 100.0

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def w_refine(task):
    os.environ["OMP_NUM_THREADS"] = "1"
    task, hs = task
    fn = pathlib.Path(task)
    floor = float(fn.stem.replace("floor", ""))
    d = dict(np.load(fn))
    n = d["theta"].shape[1]; T = float(d["T"])
    outdir = pathlib.Path(f"repro/pool_refined_n{n}")
    guess = {k: d[k] for k in ("theta", "thetad", "a", "v")}
    results = []
    for h in hs:
        K = int(round(T / h))
        sol = solve_ctrb_aware(n, T, K, w_ctrb=W_CTRB, floor_ctrb=floor,
                               init_guess=guess, max_iter=3000, tol=1e-7)
        if sol["status"] != "solved":
            results.append((h, None))
            break  # don't chain a finer solve off a failed warm start
        out = outdir / f"floor{floor:g}_h{h:g}.npz"
        np.savez(out, t=sol["t"], theta=sol["theta"], thetad=sol["thetad"],
                 a=sol["a"], v=sol["v"], T=T)
        results.append((h, str(out)))
        guess = {k: sol[k] for k in ("theta", "thetad", "a", "v")}
    return (str(fn), results)


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else min(20, mp.cpu_count() - 4)
    files = []
    if "--n7only" not in sys.argv:
        files += sorted(pathlib.Path("repro/pool_ctrb_n6").glob("floor*.npz"))
    if "--n6only" not in sys.argv:
        files += sorted(pathlib.Path("repro/pool_ctrb_n7").glob("floor*.npz"))
    for n in (6, 7):
        pathlib.Path(f"repro/pool_refined_n{n}").mkdir(exist_ok=True)
    hs = HS + (0.0025,) if "--fine" in sys.argv else HS
    log(f"refining {len(files)} candidates at h={hs} on {nw} workers")
    with mp.Pool(nw) as pool:
        for fn, results in pool.imap_unordered(w_refine, [(str(f), hs) for f in files]):
            msg = ", ".join(f"h={h}:{'ok' if out else 'FAILED'}" for h, out in results)
            log(f"  {'/'.join(fn.split('/')[-2:]):34s} {msg}")
    log("done")


if __name__ == "__main__":
    main()
