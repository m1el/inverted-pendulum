"""Focused swing-up solver for hard N (4,5) using homotopy + multistart.

Strategy per N:
  - Load the N-1 trajectory (results/trajectories/swingup_N{n-1}.npz) and build
    a homotopy initial guess (append a tip link).
  - Coarse solve (h=0.05) with relaxed settle band, multiple horizons & seeds,
    some warm-started from the homotopy guess, some random.
  - Refine surviving coarse solutions to h=0.01.
  - Score by open-loop trackability (first 70%) + terminal residual.
  - Save all valid refined candidates to cand_N{n}/.

Runs a bounded process pool so each solve gets real CPU.

Usage: uv run python scripts/solve_highN.py N [n_workers]
"""
import sys, pathlib, json, time, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import multiprocessing as mp

ROOT = pathlib.Path(__file__).resolve().parents[1]
G = 9.81
DT = 0.01


def openloop_track(n, traj, frac=0.7, dt=DT):
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


def target_options(n):
    opts = [("zeros", np.zeros(n)),
            ("first+2pi", np.array([2 * np.pi] + [0.0] * (n - 1))),
            ("all+2pi", np.full(n, 2 * np.pi)),
            ("alt2pi", np.array([2 * np.pi * (i % 2) for i in range(n)], float))]
    return opts


def task(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    n, T, tname, tgt, seed, use_homotopy, a_max = args
    from pendulum.trajopt import solve_swingup, homotopy_guess
    try:
        ig = None
        if use_homotopy:
            low = np.load(ROOT / "results" / "trajectories" / f"swingup_N{n-1}.npz")
            # resample lower solution to horizon T grid first
            low_traj = {"theta": low["theta"], "thetad": low["thetad"],
                        "a": low["a"], "v": low["v"], "t": low["t"]}
            ig = homotopy_guess(low_traj)
        Kc = int(round(T / 0.05))
        Kf = int(round(T / DT))
        sc = solve_swingup(n, T, Kc, g=G, a_max=a_max, v_max=12,
                           theta_target=tgt, seed=seed, settle_frac=0.10,
                           settle_band=0.25, init_guess=ig, max_iter=3000,
                           tol=1e-6)
        if sc["status"] != "solved":
            return None
        sf = solve_swingup(n, T, Kf, g=G, a_max=a_max, v_max=12,
                           theta_target=tgt, init_guess=sc, settle_frac=0.10,
                           settle_band=0.20, max_iter=3000, tol=1e-7)
        if sf["status"] != "solved":
            return None
        track = openloop_track(n, sf, frac=0.7)
        term = float(np.max(np.abs(sf["theta"][-1] - tgt)))
        return dict(n=n, T=T, tname=tname, tgt=tgt, seed=seed, homo=use_homotopy,
                    cost=float(sf["cost"]), track=track, term=term,
                    maxv=float(np.max(np.abs(sf["v"]))),
                    t=sf["t"], theta=sf["theta"], thetad=sf["thetad"],
                    a=sf["a"], v=sf["v"])
    except Exception as e:
        return None


def main():
    n = int(sys.argv[1])
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    Tmap = {4: [7.0, 8.0, 9.0, 10.0], 5: [10.0, 12.0, 14.0]}
    amap = {4: 50, 5: 60}
    Ts = Tmap[n]; a_max = amap[n]

    cdir = ROOT / "results" / "trajectories" / f"cand_N{n}"
    cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("*.npz"):
        old.unlink()

    tasks = []
    for T in Ts:
        for tname, tgt in target_options(n):
            # homotopy warm starts (deterministic)
            tasks.append((n, T, tname, tgt, 0, True, a_max))
            # random multistarts
            for seed in range(6):
                tasks.append((n, T, tname, tgt, seed, False, a_max))
    print(f"N={n}: {len(tasks)} tasks, {nworkers} workers", flush=True)

    idx = 0; metas = []
    t0 = time.time(); done = 0
    with mp.Pool(processes=nworkers) as pool:
        for r in pool.imap_unordered(task, tasks, chunksize=1):
            done += 1
            if r is None:
                continue
            fn = cdir / f"cand_{idx}.npz"
            np.savez(fn, t=r["t"], theta=r["theta"], thetad=r["thetad"],
                     a=r["a"], v=r["v"], T=r["T"], target=r["tgt"])
            m = dict(idx=idx, T=r["T"], target=r["tname"], seed=r["seed"],
                     homo=r["homo"], cost=round(r["cost"], 3),
                     track=round(r["track"], 5), term=round(r["term"], 6),
                     maxv=round(r["maxv"], 3))
            metas.append(m)
            json.dump(metas, open(cdir / "meta.json", "w"), indent=2)
            print(f"[{done}/{len(tasks)}] cand{idx} T={r['T']} {r['tname']} "
                  f"seed={r['seed']} homo={r['homo']} track={m['track']} "
                  f"term={m['term']} cost={m['cost']} ({time.time()-t0:.0f}s)", flush=True)
            idx += 1
    print(f"N={n}: {idx} valid candidates", flush=True)


if __name__ == "__main__":
    main()
