"""Select the best swing-up trajectory per N by closed-loop performance.

For each candidate, build SwingupController and run closed-loop (zero
quantization, seed-perturbed start) over horizon T+20s. Score by:
 - success (final 5s within 0.3 rad of upright across all seeds)
 - settling: max |wrap(theta)| over final 5s
Pick the candidate that succeeds across the most seeds with smallest tail error,
copy it to results/trajectories/swingup_N{n}.npz.

Usage: uv run python scripts/select_traj.py N [tvlqr-preset]
"""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pendulum.dynamics import Chain
from pendulum.sim import simulate
from pendulum.swingup_traj import SwingupController, BALANCE_TUNINGS

G = 9.81
DT = 0.01


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


TVLQR_PRESETS = {
    "default": dict(q_theta=50, q_thetad=5, q_v=1.0, r=0.5,
                    qf_theta=2000, qf_thetad=200, qf_v=10),
    "tight": dict(q_theta=100, q_thetad=10, q_v=2.0, r=0.2,
                  qf_theta=8000, qf_thetad=400, qf_v=20),
    "vtight": dict(q_theta=200, q_thetad=20, q_v=2.0, r=0.1,
                   qf_theta=20000, qf_thetad=1000, qf_v=20),
}


def eval_candidate(n, fn, tvlqr_kw, seeds=range(4), tail_s=5.0,
                   catch_angle=0.2, catch_rate=2.0):
    chain = Chain(n, G)
    data = np.load(fn)
    T = float(data["t"][-1])
    horizon = T + 20.0
    nsteps = int(round(horizon / DT))
    tail = int(round(tail_s / DT))
    n_ok = 0
    worst_tail = 0.0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        y0 = np.zeros(2 * n)
        y0[:n] = np.pi + rng.uniform(-1e-3, 1e-3, n)
        ctrl = SwingupController(chain, DT, 0.0, 0.0, str(fn),
                                 tvlqr_kw=tvlqr_kw, catch_angle=catch_angle,
                                 catch_rate=catch_rate)
        res = simulate(chain, ctrl, y0, DT, nsteps, dtheta=0.0, dv=0.0,
                       record=True)
        th = wrap(res["traj"]["y"][:, :n])
        tailerr = np.max(np.abs(th[-tail:]))
        if tailerr < 0.3:
            n_ok += 1
        worst_tail = max(worst_tail, tailerr)
    return n_ok, worst_tail


def main():
    n = int(sys.argv[1])
    presets = [sys.argv[2]] if len(sys.argv) > 2 else ["default", "tight", "vtight"]
    cdir = pathlib.Path(__file__).resolve().parents[1] / "results" / "trajectories" / f"cand_N{n}"
    cands = sorted(cdir.glob("cand_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not cands:
        print(f"N={n}: no candidates"); sys.exit(1)
    meta = {m["idx"]: m for m in json.load(open(cdir / "meta.json"))}
    # rank candidates by track (best trackability first) to test promising first
    cands = sorted(cands, key=lambda p: meta[int(p.stem.split("_")[1])]["track"])

    print(f"=== N={n} selecting among {len(cands)} candidates, presets={presets} ===", flush=True)
    best = None  # (n_ok, -worst_tail, idx, preset, fn)
    for p in cands:
        idx = int(p.stem.split("_")[1])
        m = meta[idx]
        for preset in presets:
            t0 = time.time()
            n_ok, worst = eval_candidate(n, p, TVLQR_PRESETS[preset])
            print(f"  [{idx}] {m['target']} T={m['T']} track={m['track']} "
                  f"preset={preset}: ok={n_ok}/4 worst_tail={worst:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
            key = (n_ok, -worst)
            if best is None or key > best[0]:
                best = (key, idx, preset, str(p))
            if n_ok == 4 and worst < 0.05:
                break  # good enough for this candidate
        if best is not None and best[0][0] == 4 and -best[0][1] < 0.05:
            # found a solid one; keep scanning a few more? stop early
            pass
    (n_ok, nworst), idx, preset, fn = best
    print(f"N={n}: BEST idx={idx} preset={preset} ok={n_ok}/4 worst_tail={-nworst:.4f}")
    # save selection
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "trajectories" / f"swingup_N{n}.npz"
    data = np.load(fn)
    np.savez(out, **{k: data[k] for k in data.files})
    sel = pathlib.Path(__file__).resolve().parents[1] / "results" / "trajectories" / f"selection_N{n}.json"
    json.dump(dict(idx=idx, preset=preset, n_ok=n_ok, worst_tail=float(-nworst),
                   meta=meta[idx], tvlqr=preset), open(sel, "w"), indent=2)
    print(f"N={n}: saved {out}")


if __name__ == "__main__":
    main()
