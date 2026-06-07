"""Select the best N=5 swing-up trajectory by closed-loop performance.

Tests each candidate (perfect-state TVLQR feasibility first, cheap) then full
closed-loop with the SwingupController across 4 seeds. Picks the one that
succeeds with smallest tail error; saves to swingup_N5.npz.

Usage: uv run python scripts/select_n5.py [n_workers]
"""
import sys, pathlib, json, time, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp

ROOT = pathlib.Path(__file__).resolve().parents[1]
G = 9.81
DT = 0.01
N = 5


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def perfect_state_ok(fn, preset):
    """Cheap pre-screen: can TVLQR track this traj with perfect state?"""
    from pendulum.dynamics import Chain
    from pendulum.sim import rk4_step
    from pendulum.swingup_traj import compute_tvlqr, TVLQR_PRESETS
    chain = Chain(N, G)
    d = np.load(fn)
    th = d["theta"]; td = d["thetad"]; av = d["a"]; vv = d["v"]; Ns = th.shape[0] - 1
    Ks = compute_tvlqr(chain, d, DT, **TVLQR_PRESETS[preset])
    y = np.concatenate([th[0], td[0]]); v = 0.0; md = 0.0
    for k in range(Ns):
        z = np.concatenate([y[:N], y[N:], [v]]); zn = np.concatenate([th[k], td[k], [vv[k]]])
        a = av[k] - float((Ks[k] @ (z - zn))[0])
        y = rk4_step(chain, y, a, DT); v += a * DT
        md = max(md, np.max(np.abs(y[:N] - th[k + 1])))
        if not np.isfinite(md):
            return False, np.inf
    return md < 0.3, float(np.max(np.abs(wrap(y[:N]))))


def closed_loop(fn, preset, seeds=range(4)):
    from pendulum.dynamics import Chain
    from pendulum.sim import simulate
    from pendulum.swingup_traj import SwingupController
    chain = Chain(N, G)
    T = float(np.load(fn)["t"][-1]); horizon = T + 20.0
    nsteps = int(round(horizon / DT)); tail = int(round(5.0 / DT))
    oks = 0; worst = 0.0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        y0 = np.zeros(2 * N); y0[:N] = np.pi + rng.uniform(-1e-3, 1e-3, N)
        c = SwingupController(chain, DT, 0.0, 0.0, str(fn), tvlqr_kw=preset)
        res = simulate(chain, c, y0, DT, nsteps, record=True)
        th = wrap(res["traj"]["y"][:, :N]); te = np.max(np.abs(th[-tail:]))
        if te < 0.3:
            oks += 1
        worst = max(worst, te)
    return oks, float(worst)


def eval_cand(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    fn, presets = args
    out = {"fn": fn, "best": None}
    for preset in presets:
        ok_ps, ps_err = perfect_state_ok(fn, preset)
        if not ok_ps:
            continue
        oks, worst = closed_loop(fn, preset)
        cur = (oks, -worst, preset, ps_err)
        if out["best"] is None or cur[:2] > out["best"][:2]:
            out["best"] = cur
    return out


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    cdir = ROOT / "results" / "trajectories" / "cand_N5"
    cands = sorted(cdir.glob("cand_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not cands:
        print("no N=5 candidates"); return
    presets = ["vtight", "tight", "default"]
    print(f"N=5: screening {len(cands)} candidates", flush=True)
    args = [(str(p), presets) for p in cands]
    results = []
    with mp.Pool(processes=nworkers) as pool:
        for out in pool.imap_unordered(eval_cand, args, chunksize=1):
            if out["best"] is not None:
                oks, nworst, preset, ps_err = out["best"]
                print(f"  {pathlib.Path(out['fn']).name}: ok={oks}/4 worst={-nworst:.4f} "
                      f"preset={preset} ps_err={ps_err:.4f}", flush=True)
                results.append((oks, nworst, out["fn"], preset))
    if not results:
        print("N=5: NO candidate passed closed-loop"); return
    results.sort(key=lambda r: (r[0], r[1]), reverse=True)
    oks, nworst, fn, preset = results[0]
    print(f"N=5: BEST {pathlib.Path(fn).name} ok={oks}/4 worst={-nworst:.4f} preset={preset}")
    d = np.load(fn)
    out = ROOT / "results" / "trajectories" / "swingup_N5.npz"
    np.savez(out, **{k: d[k] for k in d.files})
    json.dump(dict(file=pathlib.Path(fn).name, preset=preset, ok=oks,
                   worst_tail=float(-nworst)),
              open(ROOT / "results" / "trajectories" / "selection_N5.json", "w"), indent=2)
    print(f"N=5: saved {out} (preset {preset})")


if __name__ == "__main__":
    main()
