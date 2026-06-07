"""Balance precision (quantization) threshold sweeps.

For BALANCE task, N=1..5, controllers FD and Kalman, find:
  - max dtheta (dv=0)
  - max dv (dtheta=0)
  - joint boundary: dv threshold at a few fixed dtheta values
LQR weight tuning per N (headline thresholds) for N>=3.
dt scaling and g scaling on the best controller.

Run: uv run scripts/sweep_balance.py [stage]
stages: headline | tune | dt | g | all   (default all)
Writes results/balance.json incrementally and prints progress.

Parallelized with multiprocessing (each threshold-bisect is one task).
"""

import sys, pathlib, json, time, itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from pendulum.dynamics import Chain
from pendulum.balance import FDBalancer, KalmanBalancer
from pendulum.protocol import balance_success, threshold_bisect

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results" / "balance.json"

# ---------------------------------------------------------------------------
# Controller factory helpers (picklable: module-level functions)
# ---------------------------------------------------------------------------

def make_fd(lqr_kw):
    def f(chain, dt, dtheta, dv):
        return FDBalancer(chain, dt, **lqr_kw)
    return f


# Floor the *noise model* steps fed to the KF's DARE. The actual quantization
# applied by the simulator still uses the true dtheta/dv (passed by protocol);
# only the KF's internal noise-variance estimate is floored. Without this,
# extremely fine dtheta (e.g. the bisection's lo=1e-7) drives meas_var toward
# zero and solve_discrete_are becomes ill-conditioned and raises.
KF_DTHETA_FLOOR = 1e-5   # rad: KF gains nothing from modeling finer angle noise
KF_DV_FLOOR = 1e-5       # m/s

class RobustKalmanBalancer(KalmanBalancer):
    """KalmanBalancer whose *noise model* uses floored dtheta/dv (to keep the
    filter DARE well-conditioned at very fine grids) while its *output
    quantization* uses the true dv (so it matches what the simulator applies)."""
    def __init__(self, chain, dt, dtheta, dv, **lqr_kw):
        dth_model = max(dtheta, KF_DTHETA_FLOOR)
        dv_model = max(dv, KF_DV_FLOOR)
        super().__init__(chain, dt, dth_model, dv_model, **lqr_kw)
        self.dv = dv  # restore true dv for self-quantization of the command

def make_kf(lqr_kw):
    def f(chain, dt, dtheta, dv):
        return RobustKalmanBalancer(chain, dt, dtheta, dv, **lqr_kw)
    return f


def success_fn_factory(n, g, dt, controller, lqr_kw, dtheta_fixed, dv_fixed, sweep):
    """Return a callable q -> bool for the requested sweep.

    sweep == 'dtheta': vary dtheta = q, dv = dv_fixed
    sweep == 'dv'    : vary dv = q,     dtheta = dtheta_fixed
    """
    chain = Chain(n, g)
    mk = make_fd(lqr_kw) if controller == "FD" else make_kf(lqr_kw)

    if sweep == "dtheta":
        def fn(q):
            return balance_success(chain, mk, dt, q, dv_fixed)
    else:
        def fn(q):
            return balance_success(chain, mk, dt, dtheta_fixed, q)
    return fn


# ---------------------------------------------------------------------------
# Worker entry points (top-level for pickling). Each returns (key, value).
# ---------------------------------------------------------------------------

def w_threshold(args):
    """Generic single threshold-bisect task.
    args: dict with key, n, g, dt, controller, lqr_kw, sweep, dtheta_fixed, dv_fixed
    """
    key = args["key"]
    fn = success_fn_factory(
        args["n"], args["g"], args["dt"], args["controller"], args["lqr_kw"],
        args.get("dtheta_fixed", 0.0), args.get("dv_fixed", 0.0), args["sweep"],
    )
    # lo floored at 1e-6: finer grids are physically irrelevant and stress the
    # filter DARE. hi=1.0 (failure expected there).
    lo, hi = args.get("lo", 1e-6), args.get("hi", 1.0)
    t0 = time.time()
    try:
        res = threshold_bisect(fn, lo=lo, hi=hi)
        return key, {"bracket": list(res), "secs": round(time.time() - t0, 1)}
    except Exception as e:  # never let one combo kill the pool
        return key, {"bracket": [0.0, 0.0], "secs": round(time.time() - t0, 1),
                     "error": repr(e)}


# ---------------------------------------------------------------------------

def gmean(bracket):
    lo, hi = bracket
    if hi == np.inf:
        return lo
    if lo == 0.0:
        return 0.0
    return float(np.sqrt(lo * hi))


def load():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {}

def save(d):
    RESULTS.write_text(json.dumps(d, indent=2))


# ---------------------------------------------------------------------------
# Stage 1: LQR tuning (headline thresholds) for N=3..5, both controllers.
# Modest grid over r and q_theta; pick kwargs maximizing min(max_dtheta,max_dv)
# (in a normalized sense). We evaluate max_dtheta and max_dv for each combo.
# ---------------------------------------------------------------------------

DEFAULT_KW = dict(q_theta=10.0, q_thetad=1.0, q_x=0.1, q_v=0.5, r=0.1)

def tune_grid():
    grid = []
    for r in [0.01, 0.1, 1.0]:
        for q_theta in [1.0, 10.0, 100.0]:
            kw = dict(DEFAULT_KW)
            kw["r"] = r
            kw["q_theta"] = q_theta
            grid.append(kw)
    return grid


def run_tasks(tasks, label):
    print(f"[{label}] launching {len(tasks)} tasks ...", flush=True)
    out = {}
    with ProcessPoolExecutor(max_workers=min(64, len(tasks))) as ex:
        futs = {ex.submit(w_threshold, t): t["key"] for t in tasks}
        for fut in as_completed(futs):
            key, val = fut.result()
            out[key] = val
            print(f"  [{label}] {key} -> {val['bracket']}  ({val['secs']}s)", flush=True)
    return out


def stage_tune():
    """Per N (1..5) and controller, sweep LQR grid for max_dtheta and max_dv.
    Pick the kwargs that maximize the geometric mean of the two headline gmeans.
    Store the winning kwargs and brackets."""
    grid = tune_grid()
    tasks = []
    for n in range(1, 6):
        for controller in ["FD", "Kalman"]:
            for gi, kw in enumerate(grid):
                for sweep in ["dtheta", "dv"]:
                    tasks.append(dict(
                        key=f"tune|N{n}|{controller}|g{gi}|{sweep}",
                        n=n, g=9.81, dt=0.01, controller=controller,
                        lqr_kw=kw, sweep=sweep,
                    ))
    res = run_tasks(tasks, "tune")

    # aggregate: pick best grid index per (N, controller)
    summary = {}
    for n in range(1, 6):
        for controller in ["FD", "Kalman"]:
            best = None
            for gi, kw in enumerate(grid):
                dth = res[f"tune|N{n}|{controller}|g{gi}|dtheta"]["bracket"]
                dv = res[f"tune|N{n}|{controller}|g{gi}|dv"]["bracket"]
                gdth, gdv = gmean(dth), gmean(dv)
                # normalized score: product of the two gmeans (both want big)
                score = gdth * gdv
                cand = (score, gi, kw, dth, dv, gdth, gdv)
                if best is None or score > best[0]:
                    best = cand
            _, gi, kw, dth, dv, gdth, gdv = best
            summary[f"N{n}|{controller}"] = dict(
                lqr_kwargs=kw, grid_index=gi,
                max_dtheta=dth, max_dv=dv,
                gmean_dtheta=gdth, gmean_dv=gdv,
            )
            print(f"** BEST N={n} {controller}: kw={kw} "
                  f"max_dtheta={gdth:.4g} max_dv={gdv:.4g}", flush=True)

    d = load()
    d["_tune"] = {"grid": grid, "summary": summary, "raw": res}
    save(d)
    return summary


# ---------------------------------------------------------------------------
# Stage 2: full headline + joint at g=9.81 dt=0.01 using best kwargs per (N,ctrl)
# ---------------------------------------------------------------------------

def stage_headline():
    d = load()
    summary = d["_tune"]["summary"]

    # determine joint fixed-dtheta values from each (N,ctrl) max_dtheta gmean
    tasks = []
    for n in range(1, 6):
        for controller in ["FD", "Kalman"]:
            kw = summary[f"N{n}|{controller}"]["lqr_kwargs"]
            base = f"N{n}|{controller}|dt0.01|g9.81"
            # headline (already have from tune, but recompute cleanly to store)
            tasks.append(dict(key=base + "|max_dtheta", n=n, g=9.81, dt=0.01,
                              controller=controller, lqr_kw=kw, sweep="dtheta"))
            tasks.append(dict(key=base + "|max_dv", n=n, g=9.81, dt=0.01,
                              controller=controller, lqr_kw=kw, sweep="dv"))
    res = run_tasks(tasks, "headline")

    # joint: dv threshold at fixed dtheta = maxdth/10 and maxdth/3
    jtasks = []
    for n in range(1, 6):
        for controller in ["FD", "Kalman"]:
            kw = summary[f"N{n}|{controller}"]["lqr_kwargs"]
            base = f"N{n}|{controller}|dt0.01|g9.81"
            maxdth = gmean(res[base + "|max_dtheta"]["bracket"])
            for frac in [10.0, 3.0]:
                dth_fixed = maxdth / frac
                jtasks.append(dict(
                    key=base + f"|joint|dth_over{int(frac)}",
                    n=n, g=9.81, dt=0.01, controller=controller, lqr_kw=kw,
                    sweep="dv", dtheta_fixed=dth_fixed,
                    _dth_fixed=dth_fixed,
                ))
    jres = run_tasks(jtasks, "joint")

    # assemble into per (N,ctrl) records
    d = load()
    for n in range(1, 6):
        for controller in ["FD", "Kalman"]:
            kw = summary[f"N{n}|{controller}"]["lqr_kwargs"]
            base = f"N{n}|{controller}|dt0.01|g9.81"
            joint = []
            for frac in [10.0, 3.0]:
                jk = base + f"|joint|dth_over{int(frac)}"
                maxdth = gmean(res[base + "|max_dtheta"]["bracket"])
                joint.append(dict(
                    dtheta_fixed=maxdth / frac,
                    dv_bracket=jres[jk]["bracket"],
                ))
            d[base] = dict(
                N=n, controller=controller, dt=0.01, g=9.81,
                lqr_kwargs=kw,
                max_dtheta=res[base + "|max_dtheta"]["bracket"],
                max_dv=res[base + "|max_dv"]["bracket"],
                joint=joint,
            )
    save(d)
    return d


def best_controller_per_N(d):
    """Choose best controller per N by gmean(max_dtheta)*gmean(max_dv)."""
    best = {}
    for n in range(1, 6):
        scores = {}
        for controller in ["FD", "Kalman"]:
            base = f"N{n}|{controller}|dt0.01|g9.81"
            rec = d[base]
            scores[controller] = gmean(rec["max_dtheta"]) * gmean(rec["max_dv"])
        best[n] = max(scores, key=scores.get)
    return best


# ---------------------------------------------------------------------------
# Stage 3: dt scaling on best controller per N
# ---------------------------------------------------------------------------

def stage_dt():
    d = load()
    best = best_controller_per_N(d)
    summary = d["_tune"]["summary"]
    tasks = []
    for n in range(1, 6):
        controller = best[n]
        kw = summary[f"N{n}|{controller}"]["lqr_kwargs"]
        for dt in [0.002, 0.005, 0.02]:
            base = f"N{n}|{controller}|dt{dt}|g9.81"
            tasks.append(dict(key=base + "|max_dtheta", n=n, g=9.81, dt=dt,
                              controller=controller, lqr_kw=kw, sweep="dtheta"))
            tasks.append(dict(key=base + "|max_dv", n=n, g=9.81, dt=dt,
                              controller=controller, lqr_kw=kw, sweep="dv"))
    res = run_tasks(tasks, "dt")
    d = load()
    for n in range(1, 6):
        controller = best[n]
        kw = summary[f"N{n}|{controller}"]["lqr_kwargs"]
        for dt in [0.002, 0.005, 0.02]:
            base = f"N{n}|{controller}|dt{dt}|g9.81"
            d[base] = dict(
                N=n, controller=controller, dt=dt, g=9.81, lqr_kwargs=kw,
                max_dtheta=res[base + "|max_dtheta"]["bracket"],
                max_dv=res[base + "|max_dv"]["bracket"],
            )
    save(d)
    return d


# ---------------------------------------------------------------------------
# Stage 4: g scaling on best controller for N in {1,3,5}
# ---------------------------------------------------------------------------

def stage_g():
    d = load()
    best = best_controller_per_N(d)
    summary = d["_tune"]["summary"]
    tasks = []
    for n in [1, 3, 5]:
        controller = best[n]
        # NOTE: retune is overkill; LQR kwargs from g=9.81 are reused. The LQR
        # is recomputed inside upright_lqr with the chain's actual g, so the
        # gains adapt to g automatically; only the weights are fixed.
        kw = summary[f"N{n}|{controller}"]["lqr_kwargs"]
        for g in [4.905, 19.62]:
            base = f"N{n}|{controller}|dt0.01|g{g}"
            tasks.append(dict(key=base + "|max_dtheta", n=n, g=g, dt=0.01,
                              controller=controller, lqr_kw=kw, sweep="dtheta"))
            tasks.append(dict(key=base + "|max_dv", n=n, g=g, dt=0.01,
                              controller=controller, lqr_kw=kw, sweep="dv"))
    res = run_tasks(tasks, "g")
    d = load()
    for n in [1, 3, 5]:
        controller = best[n]
        kw = summary[f"N{n}|{controller}"]["lqr_kwargs"]
        for g in [4.905, 19.62]:
            base = f"N{n}|{controller}|dt0.01|g{g}"
            d[base] = dict(
                N=n, controller=controller, dt=0.01, g=g, lqr_kwargs=kw,
                max_dtheta=res[base + "|max_dtheta"]["bracket"],
                max_dv=res[base + "|max_dv"]["bracket"],
            )
    save(d)
    return d


# ---------------------------------------------------------------------------

def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if stage in ("tune", "all"):
        stage_tune()
    if stage in ("headline", "all"):
        stage_headline()
    if stage in ("dt", "all"):
        stage_dt()
    if stage in ("g", "all"):
        stage_g()
    print(f"DONE stage={stage} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
