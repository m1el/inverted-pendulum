"""Generate swing-up trajectory candidates for a given N and save all valid ones.

For each (T, target, seed): solve coarse (h=0.05) then refine (h=0.01).
Score = open-loop drift over first 70% of horizon (trackability) + terminal
node residual.  Save every valid refined candidate to
results/trajectories/cand_N{n}/cand_{idx}.npz with metadata.

Usage: uv run python scripts/gen_candidates.py N [T1 T2 ...]
"""
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.trajopt import solve_swingup

G = 9.81
DT = 0.01


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def openloop_track(n, traj, frac=0.7, dt=DT):
    """Max open-loop angle drift over first `frac` of trajectory using
    node-aligned ZOH acceleration (measures collocation trackability)."""
    chain = Chain(n, G)
    a = traj["a"]; th = traj["theta"]; td = traj["thetad"]
    Nsteps = th.shape[0] - 1
    kmax = int(frac * Nsteps)
    y = np.concatenate([th[0], td[0]])
    maxdrift = 0.0
    for k in range(kmax):
        y = rk4_step(chain, y, a[k], dt)
        maxdrift = max(maxdrift, np.max(np.abs(y[:n] - th[k + 1])))
    return maxdrift


def target_options(n):
    opts = [("zeros", np.zeros(n))]
    opts.append(("first+2pi", np.array([2 * np.pi] + [0.0] * (n - 1))))
    opts.append(("all+2pi", np.full(n, 2 * np.pi)))
    opts.append(("first-2pi", np.array([-2 * np.pi] + [0.0] * (n - 1))))
    if n >= 3:
        opts.append(("alt2pi", np.array([2 * np.pi * (i % 2) for i in range(n)], float)))
    return opts


def main():
    n = int(sys.argv[1])
    if len(sys.argv) > 2:
        Ts = [float(x) for x in sys.argv[2:]]
    else:
        Ts = {2: [3.5, 4.0, 5.0], 3: [5.0, 6.0, 7.0],
              4: [7.0, 8.0, 9.0, 10.0], 5: [9.0, 11.0, 13.0, 15.0]}[n]
    a_max = {2: 40, 3: 45, 4: 50, 5: 60}[n]
    v_max = 12.0
    seeds = range(8)
    settle_frac = 0.12

    outdir = pathlib.Path(__file__).resolve().parents[1] / "results" / "trajectories" / f"cand_N{n}"
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*.npz"):
        old.unlink()

    print(f"=== N={n} Ts={Ts} a_max={a_max} ===", flush=True)
    idx = 0
    meta = []
    for T in Ts:
        Kc = int(round(T / 0.05))
        Kf = int(round(T / DT))
        for tname, tgt in target_options(n):
            for seed in seeds:
                t0 = time.time()
                try:
                    sc = solve_swingup(n, T, Kc, g=G, a_max=a_max, v_max=v_max,
                                       theta_target=tgt, seed=seed,
                                       settle_frac=settle_frac, max_iter=2500)
                except Exception as e:
                    continue
                if sc["status"] != "solved":
                    continue
                try:
                    sf = solve_swingup(n, T, Kf, g=G, a_max=a_max, v_max=v_max,
                                       theta_target=tgt, init_guess=sc,
                                       settle_frac=settle_frac, max_iter=2500)
                except Exception:
                    continue
                if sf["status"] != "solved":
                    continue
                track = openloop_track(n, sf, frac=0.7)
                term = np.max(np.abs(sf["theta"][-1] - tgt))
                fn = outdir / f"cand_{idx}.npz"
                np.savez(fn, t=sf["t"], theta=sf["theta"], thetad=sf["thetad"],
                         a=sf["a"], v=sf["v"], T=sf["T"], target=tgt)
                m = dict(idx=idx, T=T, target=tname, seed=seed,
                         cost=round(sf["cost"], 3), track=round(float(track), 5),
                         term=round(float(term), 6), maxv=round(float(np.max(np.abs(sf["v"]))), 3),
                         maxa=round(float(np.max(np.abs(sf["a"]))), 3))
                meta.append(m)
                print(f"  [{idx}] T={T} {tname} seed={seed} cost={m['cost']} "
                      f"track={m['track']} term={m['term']} maxv={m['maxv']} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                idx += 1
    with open(outdir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"N={n}: {idx} candidates saved to {outdir}", flush=True)


if __name__ == "__main__":
    main()
