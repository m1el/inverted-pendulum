"""Solve swing-up for one N: try several (T, target, seed) configs, refine to
h=0.01 grid, verify open-loop with rk4_step, save best to npz.

Usage: uv run python scripts/solve_one.py N [T ...]
"""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.trajopt import solve_swingup

G = 9.81
DT = 0.01


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def openloop_error(n, traj, dt=DT):
    """Integrate exact dynamics with ZOH acceleration a(t) at dt; return
    final wrapped angle error and thetad norm vs upright (target=0)."""
    chain = Chain(n, G)
    # resample a to dt grid via ZOH from traj nodes
    T = traj["t"][-1]
    nsteps = int(round(T / dt))
    tnodes = traj["t"]
    a_nodes = traj["a"]
    y = np.concatenate([traj["theta"][0], traj["thetad"][0]])
    v = 0.0
    maxv = 0.0
    for k in range(nsteps):
        tt = k * dt
        # ZOH: use a at the node interval. Interpolate a linearly (control is
        # piecewise; we use linear interp which matches HS control profile).
        a = float(np.interp(tt, tnodes, a_nodes))
        y = rk4_step(chain, y, a, dt)
        v += a * dt
        maxv = max(maxv, abs(v))
    th_err = np.max(np.abs(wrap(y[:n] - traj["theta"][-1])))
    td_err = np.max(np.abs(y[n:]))
    return th_err, td_err, maxv, y


def target_options(n):
    """Different homotopy-class targets (multiples of 2pi per link)."""
    opts = [np.zeros(n)]
    # rotate first link +2pi (one direction), all 0
    opts.append(np.array([2 * np.pi] + [0.0] * (n - 1)))
    opts.append(np.full(n, 2 * np.pi))
    opts.append(np.array([2 * np.pi * (i % 2) for i in range(n)], float))
    opts.append(np.array([-2 * np.pi] + [0.0] * (n - 1)))
    return opts


def solve_for_n(n, Ts, a_max=40.0, v_max=12.0, seeds=range(6), settle_frac=0.12):
    best = None
    for T in Ts:
        Kc = int(round(T / 0.05))  # coarse grid h=0.05
        for tgt in target_options(n):
            for seed in seeds:
                t0 = time.time()
                try:
                    sol = solve_swingup(n, T, Kc, g=G, a_max=a_max, v_max=v_max,
                                        theta_target=tgt, seed=seed,
                                        settle_frac=settle_frac, w_a=1.0,
                                        w_smooth=1e-3, max_iter=2000)
                except Exception as e:
                    print(f"  T={T} tgt0={tgt[0]:.2f} seed={seed}: EXC {e}")
                    continue
                if sol["status"] != "solved":
                    continue
                # refine to h=0.01
                Kf = int(round(T / DT))
                try:
                    solf = solve_swingup(n, T, Kf, g=G, a_max=a_max, v_max=v_max,
                                         theta_target=tgt, init_guess=sol,
                                         settle_frac=settle_frac, w_a=1.0,
                                         w_smooth=1e-3, max_iter=2000)
                except Exception as e:
                    print(f"  refine EXC {e}")
                    continue
                if solf["status"] != "solved":
                    continue
                th_err, td_err, maxv, _ = openloop_error(n, solf)
                dt_s = time.time() - t0
                print(f"  T={T} tgt0={tgt[0]:.2f} seed={seed} cost={solf['cost']:.2f} "
                      f"ol_th_err={th_err:.3e} ol_td_err={td_err:.3e} maxv={maxv:.2f} ({dt_s:.0f}s)",
                      flush=True)
                score = (th_err + 0.1 * td_err, solf["cost"])
                if best is None or score < best[0]:
                    best = (score, solf, th_err, td_err)
    return best


if __name__ == "__main__":
    n = int(sys.argv[1])
    if len(sys.argv) > 2:
        Ts = [float(x) for x in sys.argv[2:]]
    else:
        Ts = {2: [4.0, 5.0], 3: [5.0, 6.0, 7.0], 4: [7.0, 8.0, 9.0],
              5: [9.0, 11.0, 13.0]}[n]
    print(f"=== N={n} Ts={Ts} ===", flush=True)
    best = solve_for_n(n, Ts)
    if best is None:
        print(f"N={n}: NO SOLUTION FOUND")
        sys.exit(1)
    score, sol, th_err, td_err = best
    outdir = pathlib.Path(__file__).resolve().parents[1] / "results" / "trajectories"
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"swingup_N{n}.npz"
    np.savez(fn, t=sol["t"], theta=sol["theta"], thetad=sol["thetad"],
             a=sol["a"], v=sol["v"], T=sol["T"], target=sol["theta"][-1])
    print(f"N={n}: SAVED {fn} T={sol['T']} cost={sol['cost']:.2f} "
          f"ol_th_err={th_err:.3e} ol_td_err={td_err:.3e}")
