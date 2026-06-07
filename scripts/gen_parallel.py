"""Parallel swing-up candidate generation across all cores.

Distributes individual (N, T, target, seed) collocation solves over a process
pool. Each task: coarse (h=0.05) -> refine (h=0.01). Valid refined solutions
are saved to results/trajectories/cand_N{n}/cand_{idx}.npz.

Usage: uv run python scripts/gen_parallel.py [N ...]   (default 2 3 4 5)
"""
import sys, pathlib, json, time, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import multiprocessing as mp

ROOT = pathlib.Path(__file__).resolve().parents[1]
G = 9.81
DT = 0.01


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def target_options(n):
    opts = [("zeros", np.zeros(n)),
            ("first+2pi", np.array([2 * np.pi] + [0.0] * (n - 1))),
            ("all+2pi", np.full(n, 2 * np.pi)),
            ("first-2pi", np.array([-2 * np.pi] + [0.0] * (n - 1)))]
    if n >= 3:
        opts.append(("alt2pi", np.array([2 * np.pi * (i % 2) for i in range(n)], float)))
        opts.append(("first2+2pi", np.array([2 * np.pi, 2 * np.pi] + [0.0] * (n - 2))))
    return opts


def openloop_track(n, traj, frac=0.7, dt=DT):
    from pendulum.dynamics import Chain
    from pendulum.sim import rk4_step
    chain = Chain(n, G)
    a = traj["a"]; th = traj["theta"]; td = traj["thetad"]
    Nsteps = th.shape[0] - 1
    kmax = int(frac * Nsteps)
    y = np.concatenate([th[0], td[0]])
    maxdrift = 0.0
    for k in range(kmax):
        y = rk4_step(chain, y, a[k], dt)
        maxdrift = max(maxdrift, np.max(np.abs(y[:n] - th[k + 1])))
    return float(maxdrift)


def solve_task(args):
    n, T, tname, tgt, seed, a_max, v_max, settle_frac = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from pendulum.trajopt import solve_swingup
    try:
        Kc = int(round(T / 0.05))
        Kf = int(round(T / DT))
        sc = solve_swingup(n, T, Kc, g=G, a_max=a_max, v_max=v_max,
                           theta_target=tgt, seed=seed, settle_frac=settle_frac,
                           max_iter=2500)
        if sc["status"] != "solved":
            return None
        sf = solve_swingup(n, T, Kf, g=G, a_max=a_max, v_max=v_max,
                           theta_target=tgt, init_guess=sc, settle_frac=settle_frac,
                           max_iter=2500)
        if sf["status"] != "solved":
            return None
        track = openloop_track(n, sf, frac=0.7)
        term = float(np.max(np.abs(sf["theta"][-1] - tgt)))
        return dict(n=n, T=T, tname=tname, tgt=tgt, seed=seed,
                    cost=float(sf["cost"]), track=track, term=term,
                    maxv=float(np.max(np.abs(sf["v"]))),
                    maxa=float(np.max(np.abs(sf["a"]))),
                    t=sf["t"], theta=sf["theta"], thetad=sf["thetad"],
                    a=sf["a"], v=sf["v"])
    except Exception as e:
        return None


def build_tasks(Ns):
    Tmap = {2: [3.5, 4.0, 4.5, 5.0], 3: [5.0, 6.0, 7.0],
            4: [7.0, 8.0, 9.0, 10.0], 5: [9.0, 11.0, 13.0, 15.0]}
    amap = {2: 40, 3: 45, 4: 50, 5: 60}
    tasks = []
    for n in Ns:
        for T in Tmap[n]:
            for tname, tgt in target_options(n):
                for seed in range(8):
                    tasks.append((n, T, tname, tgt, seed, amap[n], 12.0, 0.12))
    return tasks


def main():
    Ns = [int(x) for x in sys.argv[1:]] or [2, 3, 4, 5]
    for n in Ns:
        d = ROOT / "results" / "trajectories" / f"cand_N{n}"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.npz"):
            old.unlink()
    tasks = build_tasks(Ns)
    # interleave by N so all N make progress
    tasks.sort(key=lambda t: (t[4], t[1]))  # by seed then T -> spreads N
    print(f"total tasks: {len(tasks)}; cores: {mp.cpu_count()}", flush=True)
    counters = {n: 0 for n in Ns}
    metas = {n: [] for n in Ns}
    t0 = time.time()
    done = 0
    with mp.Pool(processes=min(60, mp.cpu_count())) as pool:
        for r in pool.imap_unordered(solve_task, tasks, chunksize=1):
            done += 1
            if r is None:
                continue
            n = r["n"]; idx = counters[n]; counters[n] += 1
            fn = ROOT / "results" / "trajectories" / f"cand_N{n}" / f"cand_{idx}.npz"
            np.savez(fn, t=r["t"], theta=r["theta"], thetad=r["thetad"],
                     a=r["a"], v=r["v"], T=r["T"], target=r["tgt"])
            m = dict(idx=idx, T=r["T"], target=r["tname"], seed=r["seed"],
                     cost=round(r["cost"], 3), track=round(r["track"], 5),
                     term=round(r["term"], 6), maxv=round(r["maxv"], 3),
                     maxa=round(r["maxa"], 3))
            metas[n].append(m)
            print(f"[{done}/{len(tasks)}] N={n} cand{idx} T={r['T']} {r['tname']} "
                  f"seed={r['seed']} cost={m['cost']} track={m['track']} "
                  f"term={m['term']} ({time.time()-t0:.0f}s)", flush=True)
    for n in Ns:
        with open(ROOT / "results" / "trajectories" / f"cand_N{n}" / "meta.json", "w") as fh:
            json.dump(metas[n], fh, indent=2)
        print(f"N={n}: {counters[n]} candidates", flush=True)


if __name__ == "__main__":
    main()
