#!/usr/bin/env python
"""Mesh-ladder refinement of the no-rotation N=7 candidates (h 0.01 -> 0.0075
-> 0.005), with live IPOPT logs under runs_tmp/ipopt/ (tail for inf_pr/inf_du).
Then the dt x resampler trackability sweep on whatever rungs converged.

Run:  uv run python repro/norot_refine_ladder.py [n_workers]
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp

from repro.norot_n7 import solve_norot, POOL
import repro.consistent_nominal as cn

LOGDIR = pathlib.Path("runs_tmp/ipopt"); LOGDIR.mkdir(parents=True, exist_ok=True)
RUNGS = (0.0075, 0.005)
DTS = (0.002, 0.003, 0.004, 0.005)

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def w_ladder(tag):
    os.environ["OMP_NUM_THREADS"] = "1"
    d = dict(np.load(POOL / f"{tag}.npz"))
    T = float(d["T"]); floor = float(tag.split("_f")[1])
    guess = {k: d[k] for k in ("theta", "thetad", "a", "v")}
    out = []
    for h in RUNGS:
        sol = solve_norot(T, int(T / h), floor, guess, max_iter=5000,
                          ipopt_log=LOGDIR / f"{tag}_h{h:g}.log")
        if sol["status"] != "solved":
            out.append((h, None)); break
        fn = POOL / f"{tag}_h{h:g}.npz"
        np.savez(fn, t=sol["t"], theta=sol["theta"], thetad=sol["thetad"],
                 a=sol["a"], v=sol["v"], T=sol["T"])
        out.append((h, str(fn)))
        guess = {k: sol[k] for k in ("theta", "thetad", "a", "v")}
    return (tag, out)


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tags = ["T25_f0.5", "T25_f0.7"]
    log(f"ladder-refining {tags} at h={RUNGS} (IPOPT logs: {LOGDIR}/)")
    with mp.Pool(nw) as pool:
        results = pool.map(w_ladder, tags)
    fns = []
    for tag, out in results:
        log(f"  {tag}: " + ", ".join(f"h={h}:{'ok' if fn else 'FAILED'}" for h, fn in out))
        fns += [fn for _, fn in out if fn]
    if not fns:
        raise SystemExit("no rung converged")
    cases = [(fn, dt, v) for fn in fns for dt in DTS for v in ("linear", "hermite")]
    log(f"trackability sweep: {len(cases)} cases")
    with mp.Pool(min(16, len(cases))) as pool:
        res = pool.map(cn.w_case, cases)
    for fn in fns:
        for v in ("linear", "hermite"):
            rows = [r for r in res if r["fn"] == fn and r["variant"] == v]
            oks = [f"{r['dt']:g}" for r in rows if r["ok"]]
            best = min(rows, key=lambda r: r["final"])
            log(f"  {pathlib.Path(fn).stem:22s} {v:8s} ok@dt=[{','.join(oks)}] "
                f"best={np.degrees(best['final']):.3f}deg defect={best['defect_th']:.1e}")


if __name__ == "__main__":
    main()
