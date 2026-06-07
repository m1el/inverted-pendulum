"""Dedicated N=5 swing-up solver with staged mesh refinement.

Stage 1: very coarse (h=0.1) solve, terminal equality only (no settle band),
         multiple random seeds + targets.  This is the hard part.
Stage 2: refine surviving solutions h=0.1 -> h=0.05 -> h=0.01, progressively
         adding the settle band.

Saves valid refined candidates to cand_N5/.

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
A_MAX = 60.0


def openloop_track(n, traj, frac=0.6, dt=DT):
    from pendulum.dynamics import Chain
    from pendulum.sim import rk4_step
    chain = Chain(n, G)
    a = traj["a"]; th = traj["theta"]; td = traj["thetad"]
    Nsteps = th.shape[0] - 1
    kmax = int(frac * Nsteps)
    y = np.concatenate([th[0], td[0]])
    md = 0.0
    for k in range(kmax):
        y = rk4_step(chain, y, a[k], dt)
        md = max(md, np.max(np.abs(y[:n] - th[k + 1])))
    return float(md)


def targets():
    return [("zeros", np.zeros(N)),
            ("alt2pi", np.array([2 * np.pi * (i % 2) for i in range(N)], float)),
            ("first+2pi", np.array([2 * np.pi] + [0.0] * (N - 1))),
            ("all+2pi", np.full(N, 2 * np.pi))]


def task(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    T, tname, tgt, seed = args
    from pendulum.trajopt import solve_swingup
    try:
        # Stage 1: very coarse, no settle band, loose tol
        K1 = int(round(T / 0.1))
        s1 = solve_swingup(N, T, K1, g=G, a_max=A_MAX, v_max=12,
                           theta_target=tgt, seed=seed, settle_frac=0.0,
                           max_iter=2000, tol=1e-4)
        if s1["status"] != "solved":
            return None
        # Stage 2: h=0.05, light settle band
        K2 = int(round(T / 0.05))
        s2 = solve_swingup(N, T, K2, g=G, a_max=A_MAX, v_max=12,
                           theta_target=tgt, init_guess=s1, settle_frac=0.10,
                           settle_band=0.3, max_iter=2500, tol=1e-6)
        if s2["status"] != "solved":
            return None
        # Stage 3: h=0.01, tighter settle band
        K3 = int(round(T / DT))
        s3 = solve_swingup(N, T, K3, g=G, a_max=A_MAX, v_max=12,
                           theta_target=tgt, init_guess=s2, settle_frac=0.10,
                           settle_band=0.2, max_iter=2500, tol=1e-7)
        if s3["status"] != "solved":
            return None
        track = openloop_track(N, s3, frac=0.6)
        term = float(np.max(np.abs(s3["theta"][-1] - tgt)))
        return dict(T=T, tname=tname, tgt=tgt, seed=seed,
                    cost=float(s3["cost"]), track=track, term=term,
                    maxv=float(np.max(np.abs(s3["v"]))),
                    t=s3["t"], theta=s3["theta"], thetad=s3["thetad"],
                    a=s3["a"], v=s3["v"])
    except Exception:
        return None


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    Ts = [10.0, 12.0, 14.0]
    cdir = ROOT / "results" / "trajectories" / "cand_N5"
    cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("*.npz"):
        old.unlink()

    tasks = []
    for T in Ts:
        for tname, tgt in targets():
            for seed in range(8):
                tasks.append((T, tname, tgt, seed))
    print(f"N=5: {len(tasks)} tasks, {nworkers} workers", flush=True)

    idx = 0; metas = []; t0 = time.time(); done = 0
    with mp.Pool(processes=nworkers) as pool:
        for r in pool.imap_unordered(task, tasks, chunksize=1):
            done += 1
            if r is None:
                if done % 10 == 0:
                    print(f"  ...{done}/{len(tasks)} processed ({time.time()-t0:.0f}s)", flush=True)
                continue
            fn = cdir / f"cand_{idx}.npz"
            np.savez(fn, t=r["t"], theta=r["theta"], thetad=r["thetad"],
                     a=r["a"], v=r["v"], T=r["T"], target=r["tgt"])
            m = dict(idx=idx, T=r["T"], target=r["tname"], seed=r["seed"],
                     cost=round(r["cost"], 3), track=round(r["track"], 5),
                     term=round(r["term"], 6), maxv=round(r["maxv"], 3))
            metas.append(m)
            json.dump(metas, open(cdir / "meta.json", "w"), indent=2)
            print(f"[{done}/{len(tasks)}] cand{idx} T={r['T']} {r['tname']} seed={r['seed']} "
                  f"track={m['track']} term={m['term']} cost={m['cost']} ({time.time()-t0:.0f}s)", flush=True)
            idx += 1
    print(f"N=5: {idx} valid candidates", flush=True)


if __name__ == "__main__":
    main()
