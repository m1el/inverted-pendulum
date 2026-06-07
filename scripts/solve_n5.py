"""Dedicated N=5 swing-up solver using IMPLICIT-dynamics collocation.

Staged mesh refinement h=0.05 -> 0.02 -> 0.01, with the settle band added on
refinement. Multiple targets, seeds, horizons, and a_max values (gentler
trajectories track better under TVLQR). Saves valid candidates to cand_N5/.

Usage: uv run python scripts/solve_n5.py [n_workers]
"""
import sys, pathlib, json, time, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import multiprocessing as mp

ROOT = pathlib.Path(__file__).resolve().parents[1]
G = 9.81
DT = 0.01
N = 5


def targets():
    return [("zeros", np.zeros(N)),
            ("alt2pi", np.array([2 * np.pi * (i % 2) for i in range(N)], float)),
            ("first+2pi", np.array([2 * np.pi] + [0.0] * (N - 1))),
            ("all+2pi", np.full(N, 2 * np.pi))]


def task(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    T, tname, tgt, seed, a_max = args
    from pendulum.trajopt import solve_swingup_implicit as solve
    try:
        s1 = solve(N, T, int(round(T / 0.05)), g=G, a_max=a_max, v_max=12,
                   theta_target=tgt, seed=seed, settle_frac=0.0,
                   max_iter=2000, tol=1e-5)
        if s1["status"] != "solved":
            return None
        s2 = solve(N, T, int(round(T / 0.02)), g=G, a_max=a_max, v_max=12,
                   theta_target=tgt, init_guess=s1, settle_frac=0.12,
                   settle_band=0.25, max_iter=2500, tol=1e-6)
        if s2["status"] != "solved":
            return None
        s3 = solve(N, T, int(round(T / DT)), g=G, a_max=a_max, v_max=12,
                   theta_target=tgt, init_guess=s2, settle_frac=0.12,
                   settle_band=0.2, max_iter=3000, tol=1e-7)
        # accept "solved" or near-feasible "failed" with tiny terminal residual
        term = float(np.max(np.abs(s3["theta"][-1] - tgt))) if "theta" in s3 else 9.9
        if s3.get("status") != "solved" and term > 1e-3:
            return None
        return dict(T=T, tname=tname, tgt=tgt, seed=seed, a_max=a_max,
                    status=s3["status"], cost=float(s3["cost"]), term=term,
                    maxv=float(np.max(np.abs(s3["v"]))),
                    t=s3["t"], theta=s3["theta"], thetad=s3["thetad"],
                    a=s3["a"], v=s3["v"])
    except Exception:
        return None


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    Ts = [11.0, 12.0, 13.0, 14.0]
    amaxes = [25.0, 35.0, 50.0]
    cdir = ROOT / "results" / "trajectories" / "cand_N5"
    cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("*.npz"):
        old.unlink()

    tasks = []
    for T in Ts:
        for tname, tgt in targets():
            for a_max in amaxes:
                for seed in range(3):
                    tasks.append((T, tname, tgt, seed, a_max))
    print(f"N=5: {len(tasks)} tasks, {nworkers} workers", flush=True)

    idx = 0; metas = []; t0 = time.time(); done = 0
    with mp.Pool(processes=nworkers) as pool:
        for r in pool.imap_unordered(task, tasks, chunksize=1):
            done += 1
            if r is None:
                if done % 20 == 0:
                    print(f"  ...{done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
                continue
            fn = cdir / f"cand_{idx}.npz"
            np.savez(fn, t=r["t"], theta=r["theta"], thetad=r["thetad"],
                     a=r["a"], v=r["v"], T=r["T"], target=r["tgt"])
            m = dict(idx=idx, T=r["T"], target=r["tname"], seed=r["seed"],
                     a_max=r["a_max"], status=r["status"], cost=round(r["cost"], 3),
                     term=round(r["term"], 6), maxv=round(r["maxv"], 3))
            metas.append(m)
            json.dump(metas, open(cdir / "meta.json", "w"), indent=2)
            print(f"[{done}/{len(tasks)}] cand{idx} T={r['T']} {r['tname']} seed={r['seed']} "
                  f"amax={r['a_max']} {r['status']} term={m['term']} cost={m['cost']} "
                  f"maxv={m['maxv']} ({time.time()-t0:.0f}s)", flush=True)
            idx += 1
    print(f"N=5: {idx} candidates", flush=True)


if __name__ == "__main__":
    main()
