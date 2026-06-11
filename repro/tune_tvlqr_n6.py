#!/usr/bin/env python
"""Blind TVLQR weight tuning against the actual RK4 simulation (N=6 refined).

Search (log-space) over Q/R/QF scales; objective = the measured perturbation
funnel (largest eps with angles +-eps, rates +-1.6 eps passing 4/4 seeds),
bisected to 1.3x. The discretized (Ad, Bd) along the nominal are precomputed
ONCE (they don't depend on the weights), so each candidate costs one Riccati
pass + a handful of rollouts.

Run:  uv run python repro/tune_tvlqr_n6.py [n_workers] [n_samples]
"""
import sys, os, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp
from pendulum.dynamics import Chain
from pendulum.sim import rk4_step

G = 9.81; N = 6
BUNDLE = "repro/n6_refined_controls.npz"
_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)

_d = dict(np.load(BUNDLE))
DT = float(_d["dt"])
TH, TD, AFF, VN = _d["theta_nom"], _d["thetad_nom"], _d["a_ff"], _d["v_nom"]
NZ = 2 * N + 1

_ABS = None
def discretize():
    """RK4-Jacobian (Ad, Bd) along the nominal -- matches the simulator exactly."""
    global _ABS
    if _ABS is not None:
        return _ABS
    chain = Chain(N, G); eps = 1e-6
    ABs = []
    for k in range(len(TH) - 1):
        y0 = np.concatenate([TH[k], TD[k]]); f0 = rk4_step(chain, y0, AFF[k], DT)
        Ad = np.zeros((NZ, NZ)); Bd = np.zeros((NZ, 1))
        for j in range(2 * N):
            yp = y0.copy(); yp[j] += eps
            Ad[:2 * N, j] = (rk4_step(chain, yp, AFF[k], DT) - f0) / eps
        Ad[2 * N, 2 * N] = 1.0
        Bd[:2 * N, 0] = (rk4_step(chain, y0, AFF[k] + eps, DT) - f0) / eps
        Bd[2 * N, 0] = DT
        ABs.append((Ad, Bd))
    _ABS = ABs
    return ABs


def gains(params):
    r, qth, qtd, qv, qf = params
    Q = np.diag([qth] * N + [qtd] * N + [qv]); R = np.array([[r]])
    QF = qf * np.diag([2000.0] * N + [200.0] * N + [10.0])
    P = QF.copy(); ABs = discretize()
    Ks = np.zeros((len(ABs), 1, NZ))
    for k in range(len(ABs) - 1, -1, -1):
        Ad, Bd = ABs[k]; S = R + Bd.T @ P @ Bd
        Kk = np.linalg.solve(S, Bd.T @ P @ Ad); Ks[k] = Kk
        P = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ Kk
    return Ks


def funnel(Ks, seeds=4):
    chain = Chain(N, G)
    def ok(eps):
        for s in range(seeds):
            rng = np.random.default_rng(s)
            y = np.concatenate([TH[0] + rng.uniform(-eps, eps, N),
                                TD[0] + rng.uniform(-1.6 * eps, 1.6 * eps, N)])
            v = VN[0]
            for k in range(len(Ks)):
                z = np.concatenate([y[:N], y[N:], [v]])
                zn = np.concatenate([TH[k], TD[k], [VN[k]]])
                a = AFF[k] - float((Ks[k] @ (z - zn))[0])
                y = rk4_step(chain, y, a, DT); v += a * DT
                if not np.isfinite(y).all():
                    return False
            if float(np.max(np.abs((y[:N] + np.pi) % (2 * np.pi) - np.pi))) >= 0.05:
                return False
        return True
    lo = 0.05
    if not ok(lo):
        return 0.0
    while ok(lo * 2) and lo < 3.0:
        lo *= 2
    hi = lo * 2
    while hi / lo > 1.3:
        mid = np.sqrt(lo * hi)
        if ok(mid): lo = mid
        else: hi = mid
    return lo


def w_eval(params):
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        Ks = gains(params)
        f = funnel(Ks)
        return (f, params, float(np.abs(Ks).max()))
    except Exception:
        return (0.0, params, float("nan"))


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    log("precomputing RK4-Jacobian discretization (shared by all candidates)...")
    discretize()
    base = (0.1, 50.0, 5.0, 1.0, 1.0)        # r, q_th, q_td, q_v, qf_scale
    rng = np.random.default_rng(0)
    cands = [base]
    for _ in range(ns - 1):
        cands.append(tuple(b * 10 ** rng.uniform(-1.5, 1.5) for b in base))
    log(f"evaluating {len(cands)} weight sets on {nw} workers (objective: funnel eps)...")
    with mp.Pool(nw, initializer=discretize) as pool:
        res = pool.map(w_eval, cands)
    res.sort(key=lambda x: -x[0])
    log("top 10 (funnel eps | r, q_th, q_td, q_v, qf | maxK):")
    for f, p, mk in res[:10]:
        log(f"  eps={f:.3f}  r={p[0]:.3g} qth={p[1]:.3g} qtd={p[2]:.3g} qv={p[3]:.3g} qf={p[4]:.3g}  maxK={mk:.0f}")
    fb = [x for x in res if x[1] == base][0]
    log(f"baseline: eps={fb[0]:.3f} maxK={fb[2]:.0f}")
    with open("results/tvlqr_tuning_n6.json", "w") as fjs:
        json.dump([{"funnel": f, "params": list(p), "maxK": mk} for f, p, mk in res], fjs, indent=1)
    log("wrote results/tvlqr_tuning_n6.json")


if __name__ == "__main__":
    main()
