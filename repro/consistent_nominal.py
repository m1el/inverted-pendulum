#!/usr/bin/env python
"""Artifact elimination for TVLQR trackability: dynamics-consistent nominals.

The evaluation pipeline (optimize_n6.n6_perfect_state / generate_nN.perfect_state)
resamples the h=0.01 trapezoidal-collocation solution to the simulation grid with
LINEAR interpolation and applies the LEFT-NODE feedforward as ZOH. Both introduce
defects the TVLQR must fight as a persistent disturbance, scaled by gains up to
~1e5. This script measures and removes them:

  resampling variants
    linear : np.interp on theta/thetad/a/v, ZOH a at left node   [baseline]
    hermite: cubic-Hermite dense output using EXACT node derivatives
             (theta' = thetad, thetad' = thetadd from the dynamics -- the
             implicit collocation enforces M thetadd = rhs at nodes, so these
             are consistent), v' = a; ZOH a at the STEP MIDPOINT (the
             trapezoidal nominal treats a as piecewise linear, so the midpoint
             value is its 2nd-order-accurate ZOH equivalent)
    lin+mid / herm+node: the two fixes separately (to attribute the effect)

  diagnostics
    one-step defect: ||rk4(z_k, a_k) - z_{k+1}||_inf separately for theta and
    thetad -- exactly the per-step disturbance the tracking loop sees.

  experiment
    for each candidate trajectory (N=6 pool + N=7 pool) x dt x variant:
    build TVLQR (same Q/R/QF as the paper) on the resampled nominal, roll out
    full-state from the exact nominal start, report ok/final/maxK + defects.

Run:  uv run python repro/consistent_nominal.py [n_workers] [--quick]
Writes results to results/consistent_nominal.json and prints a table.
"""
import sys, os, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp
from scipy.interpolate import CubicHermiteSpline

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from repro.optimize_n6 import build_tvlqr

G = 9.81
OUTJ = "results/consistent_nominal.json"
DTS = (0.002, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012)
VARIANTS = ("linear", "lin+mid", "herm+node", "hermite")

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def node_thetadd(chain, d):
    """Exact angular accelerations at collocation nodes (dynamics-consistent)."""
    th, td, a = d["theta"], d["thetad"], d["a"]
    return np.vstack([chain.thetadd(th[k], td[k], a[k]) for k in range(len(a))])


def resample(d, n, dt, variant):
    """Resample a collocation solution to the simulation grid.

    Returns dict(theta(K,n), thetad(K,n), a_zoh(K,), v_nom(K,)). a_zoh[k] is the
    constant acceleration applied over [t_k, t_k+dt) and used for the TVLQR
    linearization at node k."""
    chain = Chain(n, G)
    t = d["t"]; T = float(t[-1])
    tn = np.arange(0.0, T + 1e-9, dt)
    tmid = tn + 0.5 * dt
    use_hermite = variant in ("hermite", "herm+node")
    use_mid = variant in ("hermite", "lin+mid")
    if use_hermite:
        tdd = node_thetadd(chain, d)
        sp_th = CubicHermiteSpline(t, d["theta"], d["thetad"], axis=0)
        sp_td = CubicHermiteSpline(t, d["thetad"], tdd, axis=0)
        sp_v = CubicHermiteSpline(t, d["v"], d["a"])
        theta = sp_th(tn); thetad = sp_td(tn); v_nom = sp_v(tn)
    else:
        theta = np.vstack([np.interp(tn, t, d["theta"][:, i]) for i in range(n)]).T
        thetad = np.vstack([np.interp(tn, t, d["thetad"][:, i]) for i in range(n)]).T
        v_nom = np.interp(tn, t, d["v"])
    a_at = np.clip(tmid, 0, T) if use_mid else tn
    a_zoh = np.interp(a_at, t, d["a"])
    return dict(t=tn, theta=theta, thetad=thetad, a_zoh=a_zoh, v_nom=v_nom)


def one_step_defect(chain, nom, dt):
    """Max one-step defect of the resampled nominal under the applied ZOH
    control: rk4(z_k, a_zoh[k]) - z_{k+1}. The per-step disturbance the TVLQR
    sees even with zero tracking error."""
    th, td, a = nom["theta"], nom["thetad"], nom["a_zoh"]
    n = th.shape[1]; K = len(a) - 1
    dth = 0.0; dtd = 0.0
    for k in range(K):
        y = rk4_step(chain, np.concatenate([th[k], td[k]]), a[k], dt)
        dth = max(dth, float(np.max(np.abs(y[:n] - th[k + 1]))))
        dtd = max(dtd, float(np.max(np.abs(y[n:] - td[k + 1]))))
    return dth, dtd


def rollout(chain, nom, Ks, dt):
    n = chain.n
    th, td, a_ff, v_nom = nom["theta"], nom["thetad"], nom["a_zoh"], nom["v_nom"]
    y = np.concatenate([th[0], td[0]]); v = 0.0; maxK = 0.0
    for k in range(len(Ks)):
        z = np.concatenate([y[:n], y[n:], [v]])
        zn = np.concatenate([th[k], td[k], [v_nom[k]]])
        a = a_ff[k] - float((Ks[k] @ (z - zn))[0])
        maxK = max(maxK, float(np.abs(Ks[k]).max()))
        y = rk4_step(chain, y, a, dt); v += a * dt
        if not np.isfinite(y).all():
            return False, np.inf, maxK
    final = float(np.max(np.abs((y[:n] + np.pi) % (2 * np.pi) - np.pi)))
    return final < 0.05, final, maxK


def w_case(task):
    os.environ["OMP_NUM_THREADS"] = "1"
    fn, dt, variant = task
    d = dict(np.load(fn))
    n = d["theta"].shape[1]
    chain = Chain(n, G)
    nom = resample(d, n, dt, variant)
    dth, dtd = one_step_defect(chain, nom, dt)
    Q = np.diag([50.0] * n + [5.0] * n + [1.0]); R = np.array([[0.1]])
    QF = np.diag([2000.0] * n + [200.0] * n + [10.0])
    Ks = build_tvlqr(chain, nom["theta"], nom["thetad"], nom["a_zoh"], dt, Q, R, QF)
    ok, final, maxK = rollout(chain, nom, Ks, dt)
    return dict(fn=str(fn), n=int(n), dt=dt, variant=variant, ok=bool(ok),
                final=float(final), maxK=float(maxK), defect_th=dth, defect_td=dtd)


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else min(48, mp.cpu_count() - 4)
    quick = "--quick" in sys.argv
    n6 = sorted(pathlib.Path("repro/pool_ctrb_n6").glob("floor*.npz"))
    n7 = sorted(pathlib.Path("repro/pool_ctrb_n7").glob("floor*.npz"))
    if quick:
        n6, n7 = n6[:2], n7[:4]
    files = n6 + n7
    tasks = [(str(f), dt, v) for f in files for dt in DTS for v in VARIANTS]
    log(f"{len(files)} trajectories ({len(n6)} N=6, {len(n7)} N=7) x {len(DTS)} dt x "
        f"{len(VARIANTS)} variants = {len(tasks)} cases on {nw} workers")
    res = []
    with mp.Pool(nw) as pool:
        for r in pool.imap_unordered(w_case, tasks):
            res.append(r)
            if len(res) % 20 == 0 or r["ok"]:
                log(f"  {len(res)}/{len(tasks)} done"
                    + (f"  [OK n={r['n']} {pathlib.Path(r['fn']).stem} dt={r['dt']:g} {r['variant']}"
                       f" final={np.degrees(r['final']):.3f}deg]" if r["ok"] else ""))

    pathlib.Path("results").mkdir(exist_ok=True)
    with open(OUTJ, "w") as f:
        json.dump(res, f, indent=1)
    log(f"wrote {OUTJ}")

    # summary: per trajectory x variant, which dts track
    log("=== summary (per trajectory x variant: trackable dts) ===")
    for fn in [str(f) for f in files]:
        name = "/".join(fn.split("/")[-2:])
        for v in VARIANTS:
            rows = [r for r in res if r["fn"] == fn and r["variant"] == v]
            oks = [f"{r['dt']:g}" for r in rows if r["ok"]]
            best = min(rows, key=lambda r: r["final"])
            log(f"  {name:34s} {v:9s} ok@dt=[{','.join(oks):28s}] "
                f"best_final={np.degrees(best['final']):8.3f}deg (dt={best['dt']:g}) "
                f"defect_th={best['defect_th']:.2e} defect_td={best['defect_td']:.2e}")


if __name__ == "__main__":
    main()
