#!/usr/bin/env python
"""Robustness of the reverse-fall swing-up controls (N=7, and N=6 for scaling).

Measures, by log-bisection (all seeds must pass, success = final < 0.05 rad):
  pert    : largest initial-state perturbation amplitude d (angles +-d uniform,
            rates +-d*1.6 uniform -- the repo protocol's 0.5/(0.1*pi) ratio),
            around the NOMINAL start (its winding chart, NOT pi).
  pert_th : largest ANGLE-only initial perturbation (rates exact).
  dtheta  : largest angle-measurement quantization step (persistent, full-state
            TVLQR: angles quantized, rates/velocity exact -- upper bound, same
            convention as the flip table in PAPER 4.4).
  dv      : largest velocity-COMMAND quantization step (v_cmd = quantize(v+a*dt),
            a = (v_cmd-v)/dt, as in pendulum/sim.py).

Run:  uv run python repro/n7_robustness.py [bundle.npz ...] [n_workers]
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step, quantize

G = 9.81
NSEEDS = 8
_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)

_B = {}
def load(fn):
    if fn not in _B:
        d = dict(np.load(fn))
        _B[fn] = d
    return _B[fn]


def rollout(args):
    """One closed-loop run: TVLQR swing-up, then (if hold>0) hand off to the
    upright balance LQR for `hold` seconds under the SAME quantization (the
    paper's protocol binds on the sustained catch, not the arrival).
    args=(fn, pert_th, pert_td, dth, dv, hold, seed)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    fn, pth, ptd, dth, dv, hold, seed = args
    d = load(fn)
    N = int(d["n"]); chain = Chain(N, G); dt = float(d["dt"])
    th, td, a_ff, vn, Ks = d["theta_nom"], d["thetad_nom"], d["a_ff"], d["v_nom"], d["K"]
    rng = np.random.default_rng(seed)
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    y = np.concatenate([th[0] + rng.uniform(-pth, pth, N),
                        td[0] + rng.uniform(-ptd, ptd, N)])
    v = vn[0]
    for k in range(len(Ks)):
        th_meas = quantize(y[:N], dth)
        z = np.concatenate([th_meas, y[N:], [v]])
        zn = np.concatenate([th[k], td[k], [vn[k]]])
        a = a_ff[k] - float((Ks[k] @ (z - zn))[0])
        if dv > 0:
            v_cmd = quantize(v + a * dt, dv)
            a = (v_cmd - v) / dt
        y = rk4_step(chain, y, a, dt); v += a * dt
        if not np.isfinite(y).all():
            return False
    if float(np.max(np.abs(wrap(y[:N])))) >= 0.05:
        return False
    if hold <= 0:
        return True
    # ---- sustained catch: upright balance LQR under the same quantization ----
    from pendulum.balance import upright_lqr
    Kb, _, _, _ = upright_lqr(chain, dt, r=0.01, q_theta=100)
    x = 0.0
    for k in range(int(round(hold / dt))):
        th_meas = quantize(wrap(y[:N]), dth)
        zb = np.concatenate([th_meas, y[N:], [x, v]])
        a = -float((Kb @ zb)[0])
        if dv > 0:
            v_cmd = quantize(v + a * dt, dv)
            a = (v_cmd - v) / dt
        ynew = rk4_step(chain, y, a, dt)
        vnew = v + a * dt; x += 0.5 * (v + vnew) * dt
        y, v = ynew, vnew
        if not np.isfinite(y).all() or np.max(np.abs(wrap(y[:N]))) > 0.3:
            return False
    return True


HOLD = 0.0   # set via --hold SECONDS: sustained balance-LQR catch under quantization

def passes(pool, fn, pth=0.0, ptd=0.0, dth=0.0, dv=0.0):
    res = pool.map(rollout, [(fn, pth, ptd, dth, dv, HOLD, s) for s in range(NSEEDS)])
    return all(res)


def bisect(pool, fn, setter, x0, name):
    """Largest x passing all seeds: bracket by factor 4, then log-bisect to 1.15x."""
    x = x0
    if passes(pool, fn, **setter(x)):
        lo = x
        while passes(pool, fn, **setter(lo * 4)):
            lo *= 4
        hi = lo * 4
    else:
        hi = x
        while not passes(pool, fn, **setter(hi / 4)):
            hi /= 4
            if hi < 1e-12:
                log(f"  {name}: < 1e-12 (fails at every level)"); return 0.0
        lo = hi / 4
    while hi / lo > 1.15:
        mid = np.sqrt(lo * hi)
        if passes(pool, fn, **setter(mid)):
            lo = mid
        else:
            hi = mid
    log(f"  {name}: {lo:.3g}")
    return lo


def main():
    global HOLD
    args = [a for a in sys.argv[1:]]
    if "--hold" in args:
        HOLD = float(args[args.index("--hold") + 1])
    nw = NSEEDS
    fns = [a for a in args if a.endswith(".npz")] or ["repro/n7_controls.npz"]
    with mp.Pool(nw) as pool:
        for fn in fns:
            d = load(fn)
            log(f"{fn}: N={int(d['n'])} dt={float(d['dt']):g} T={float(d['T']):.1f}s "
                f"maxK={float(d['maxK']):.0f} ({NSEEDS} seeds each, hold={HOLD:g}s)")
            r = {"hold": HOLD}
            if HOLD <= 0:
                r["pert"] = bisect(pool, fn, lambda x: dict(pth=x, ptd=1.6 * x), 0.3, "pert (angles+-d, rates+-1.6d) [rad]")
                r["pert_th"] = bisect(pool, fn, lambda x: dict(pth=x), 0.3, "pert_th (angles only) [rad]")
            r["dtheta"] = bisect(pool, fn, lambda x: dict(dth=x), 1e-5, "dtheta (angle quantization) [rad]")
            r["dv"] = bisect(pool, fn, lambda x: dict(dv=x), 1e-3, "dv (velocity-cmd quantization) [m/s]")
            out = pathlib.Path("results") / (pathlib.Path(fn).stem + f"_robustness{'_hold' if HOLD > 0 else ''}.json")
            import json
            json.dump({k: float(v) for k, v in r.items()}, open(out, "w"), indent=1)
            log(f"  -> {out}")


if __name__ == "__main__":
    main()
