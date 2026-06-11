#!/usr/bin/env python
"""No-rotation slow N=7 swing-up: is winding ESSENTIAL to N=7 trackability?

The reverse-fall N=7 solutions (repro/reverse_fall.py) are trackable but whip
(37+ rad/s) and wind 5.5-26 revolutions. All "classical" NLP solutions have the
no-rotation property (W = -1/2 rev/link, theta stays in the [0,pi] corridor).
This script asks the NLP for trajectories that keep the classical property but
adopt the reverse-fall class's two winning traits -- SLOW horizon and (after
h=0.005 refinement) a defect-free-grade nominal:

  theta_i in [-1.0, pi+1.0]  (hard no-rotation corridor, W = -1/2 forced)
  |thetad| <= 12             (no whip)
  T in {25, 35} s            (slow; classical N=7 attempts used ~17 s)
  ctrb floor in {0.5, 0.7}   (soft one-sided, as generate_nN)
  no settle band             (minimal near-straight dwell; terminal eq only)
  h=0.01 solve -> h=0.005 refine -> TVLQR sweep dt in {0.002..0.01} x
  {linear, hermite} via repro.consistent_nominal.w_case

Warm start: the cached neutral ctrb-aware N=6 seed, homotopy-lifted to N=7 and
TIME-STRETCHED (rates scaled by Told/Tnew, accel by (Told/Tnew)^2).

Run:  uv run python repro/norot_n7.py [n_workers]
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import casadi as ca
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.trajopt import chain_constants, make_M_rhs_fn, homotopy_guess
from repro.generate_n6 import (bend_excite_ca, bend_rhs_ca, mass_matrix_ca,
                               bend_excite_np)
import repro.consistent_nominal as cn

G = 9.81
N = 7
A_MAX, V_MAX, TD_MAX = 25.0, 14.0, 12.0
TH_LO, TH_HI = -1.0, 5.9                  # no-rotation corridor: theta never
# completes a turn (< 2pi) nor passes the hang backwards (> -1). Classical
# solutions range [0, 5.6] (overshoot the hang by up to ~125 deg mid-swing,
# but never rotate) -- the corridor must include that.
SEED6 = "repro/pool_ctrb_n7/_neutral_seed.npz"
POOL = pathlib.Path("repro/pool_norot_n7")
DTS = (0.002, 0.003, 0.004, 0.005, 0.008, 0.01)

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def solve_norot(T, K, floor, init_guess, max_iter=3000, tol=1e-7, ipopt_log=None):
    A, b = chain_constants(N); h = T / K
    Mfn, rfn = make_M_rhs_fn(N, G)
    opti = ca.Opti()
    TH = opti.variable(N, K + 1); TD = opti.variable(N, K + 1)
    VV = opti.variable(1, K + 1); TDD = opti.variable(N, K + 1)
    AC = opti.variable(1, K + 1)
    cost = 0
    for k in range(K + 1):
        opti.subject_to(ca.mtimes(Mfn(TH[:, k]), TDD[:, k]) == rfn(TH[:, k], TD[:, k], AC[0, k]))
    for k in range(K):
        opti.subject_to(TH[:, k + 1] - TH[:, k] == 0.5 * h * (TD[:, k] + TD[:, k + 1]))
        opti.subject_to(TD[:, k + 1] - TD[:, k] == 0.5 * h * (TDD[:, k] + TDD[:, k + 1]))
        opti.subject_to(VV[0, k + 1] - VV[0, k] == 0.5 * h * (AC[0, k] + AC[0, k + 1]))
        cost += 0.5 * h * (AC[0, k] ** 2 + AC[0, k + 1] ** 2) + 1e-3 * (AC[0, k + 1] - AC[0, k]) ** 2
    opti.subject_to(TH[:, 0] == np.pi); opti.subject_to(TD[:, 0] == 0); opti.subject_to(VV[0, 0] == 0)
    opti.subject_to(TH[:, K] == 0);     opti.subject_to(TD[:, K] == 0); opti.subject_to(VV[0, K] == 0)
    opti.subject_to(opti.bounded(-A_MAX, ca.vec(AC), A_MAX))
    opti.subject_to(opti.bounded(-V_MAX, VV, V_MAX))
    opti.subject_to(opti.bounded(-TD_MAX, ca.vec(TD), TD_MAX))
    opti.subject_to(opti.bounded(TH_LO, ca.vec(TH), TH_HI))     # NO ROTATION
    cnodes = list(range(max(1, int(0.05 * K)), int(0.95 * K), 4))
    if floor > 0:
        U = opti.variable(N, len(cnodes))
        for j, k in enumerate(cnodes):
            opti.subject_to(ca.mtimes(mass_matrix_ca(TH[:, k], A), U[:, j]) == bend_rhs_ca(TH[:, k], A, b))
            cm = ca.sqrt(bend_excite_ca(TH[:, k], U[:, j], A) + 1e-9)
            short = ca.fmax(0.0, floor - cm)
            cost += 100.0 * short * short
    opti.minimize(cost)
    s = np.linspace(0, 1, K + 1)
    Kg = init_guess["theta"].shape[0] - 1; sg = np.linspace(0, 1, Kg + 1)
    thg = np.array([np.interp(s, sg, init_guess["theta"][:, j]) for j in range(N)])
    opti.set_initial(TH, np.clip(thg, TH_LO + 0.05, TH_HI - 0.05))
    opti.set_initial(TD, np.array([np.interp(s, sg, init_guess["thetad"][:, j]) for j in range(N)]))
    opti.set_initial(VV, np.interp(s, sg, init_guess["v"]).reshape(1, -1))
    opti.set_initial(AC, np.interp(s, sg, init_guess["a"]).reshape(1, -1))
    if floor > 0:
        chain = Chain(N, G)
        opti.set_initial(U, np.column_stack(
            [np.linalg.solve(chain.mass_matrix(np.clip(thg[:, k], TH_LO + .05, TH_HI - .05)),
                             b * np.cos(thg[:, k])) for k in cnodes]))
    s_opts = {"max_iter": max_iter, "print_level": 0, "sb": "yes",
              "tol": tol, "acceptable_tol": 1e-6, "mu_strategy": "adaptive"}
    if ipopt_log:
        s_opts["output_file"] = str(ipopt_log)
        s_opts["file_print_level"] = 5
    opti.solver("ipopt", {"expand": True}, s_opts)
    try:
        sol = opti.solve(); status = "solved"
    except RuntimeError:
        sol = opti.debug; status = "failed"
    try:
        return {"t": np.linspace(0, T, K + 1), "theta": np.array(sol.value(TH)).T,
                "thetad": np.array(sol.value(TD)).T, "v": np.array(sol.value(VV)).ravel(),
                "a": np.array(sol.value(AC)).ravel(), "status": status, "T": T, "n": N}
    except RuntimeError:
        return {"status": "failed"}


def stretched_guess(seed6, Tnew):
    """Lift N=6 seed to N=7 and time-stretch to Tnew (scale rates/accel)."""
    g = homotopy_guess(seed6)
    r = float(seed6["T"]) / Tnew
    return {"theta": g["theta"], "thetad": g["thetad"] * r,
            "a": g["a"] * r * r, "v": g["v"] * r}


def w_solve(task):
    os.environ["OMP_NUM_THREADS"] = "1"
    tag, T, floor, h = task
    seed6 = dict(np.load(SEED6))
    sol = solve_norot(T, int(T / h), floor, stretched_guess(seed6, T))
    if sol["status"] != "solved":
        return (tag, None)
    fn = POOL / f"{tag}.npz"
    np.savez(fn, t=sol["t"], theta=sol["theta"], thetad=sol["thetad"],
             a=sol["a"], v=sol["v"], T=sol["T"])
    return (tag, str(fn))


def w_refine(item):
    os.environ["OMP_NUM_THREADS"] = "1"
    tag, fn = item
    d = dict(np.load(fn))
    T = float(d["T"]); floor = float(tag.split("_f")[1].split("_")[0])
    guess = {k: d[k] for k in ("theta", "thetad", "a", "v")}
    sol = solve_norot(T, int(T / 0.005), floor, guess)
    if sol["status"] != "solved":
        return (tag, None)
    fnr = POOL / f"{tag}_h005.npz"
    np.savez(fnr, t=sol["t"], theta=sol["theta"], thetad=sol["thetad"],
             a=sol["a"], v=sol["v"], T=sol["T"])
    return (tag, str(fnr))


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    POOL.mkdir(parents=True, exist_ok=True)
    tasks = [(f"T{int(T)}_f{fl}", T, fl, 0.01) for T in (25.0, 35.0) for fl in (0.5, 0.7)]
    log(f"solving {len(tasks)} no-rotation slow N=7 at h=0.01 ...")
    with mp.Pool(nw) as pool:
        solved = [r for r in pool.map(w_solve, tasks) if r[1]]
    log(f"  {len(solved)}/{len(tasks)} solved at h=0.01")
    if not solved:
        raise SystemExit("no no-rotation N=7 solved at h=0.01")

    log("refining winners to h=0.005 ...")
    with mp.Pool(nw) as pool:
        refined = [r for r in pool.map(w_refine, solved) if r[1]]
    log(f"  {len(refined)}/{len(solved)} refined")
    use = refined if refined else solved

    log("trackability sweep (dt x {linear,hermite}) ...")
    cases = [(fn, dt, v) for _, fn in use for dt in DTS for v in ("linear", "hermite")]
    with mp.Pool(min(nw * 2, 32)) as pool:
        res = pool.map(cn.w_case, cases)
    chain = Chain(N, G)
    for _, fn in use:
        d = dict(np.load(fn))
        whip = float(np.abs(d["thetad"]).max())
        cmin = float(min(bend_excite_np(chain, d["theta"][i]) for i in range(0, len(d["t"]), 5)))
        for v in ("linear", "hermite"):
            rows = [r for r in res if r["fn"] == fn and r["variant"] == v]
            oks = [f"{r['dt']:g}" for r in rows if r["ok"]]
            best = min(rows, key=lambda r: r["final"])
            log(f"  {pathlib.Path(fn).stem:16s} {v:8s} ok@dt=[{','.join(oks)}] "
                f"best={np.degrees(best['final']):.3f}deg whip={whip:.1f} cmin={cmin:.2f}")


if __name__ == "__main__":
    main()
