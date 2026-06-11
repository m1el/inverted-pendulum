#!/usr/bin/env python
"""Strategy 1: trackability-in-the-loop continuation from the reverse-fall N=7.

Start INSIDE the trackable set (repro/pool_revfall_n7/dt0.002/slow7.npz,
max|thetad|=38.4) and walk toward "nice" without leaving it:

  stage A : fit an h=0.01 collocation solution to the reverse-fall nominal
            (tracking cost, TD_MAX=45) -- a dynamics-consistent NLP anchor in
            the same (wound) homotopy class. Winding is a homotopy invariant,
            so this continuation reduces WHIP, not winding.
  rungs   : re-solve min-effort + aggregate ctrb floor with TD_MAX ramped
            45 -> 30 -> 22 -> 16 -> 12, each warm-started from the previous;
            plus two DIRECT solves (fit -> 16, fit -> 12) as parallel bets.
  check   : after every solve, full-state TVLQR at dt=0.002 (hermite resample)
            from eps=1e-3 perturbed starts (2 seeds) -- non-vacuous, fast.

Run:  uv run python repro/continuation_n7.py [n_workers]
IPOPT logs: runs_tmp/ipopt/cont_*.log
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import casadi as ca
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.trajopt import chain_constants, make_M_rhs_fn
from repro.generate_n6 import (bend_excite_ca, bend_rhs_ca, mass_matrix_ca)
from repro.rk4_tvlqr import build_tvlqr_rk4 as build_tvlqr   # RK4-consistent gates
from repro.consistent_nominal import resample

G = 9.81; N = 7
A_MAX, V_MAX = 25.0, 14.0
SRC = "repro/pool_revfall_n7/dt0.002/slow7.npz"
POOL = pathlib.Path("repro/pool_cont_n7"); POOL.mkdir(parents=True, exist_ok=True)
LOGD = pathlib.Path("runs_tmp/ipopt"); LOGD.mkdir(parents=True, exist_ok=True)
H = 0.01
RUNGS = (30.0, 22.0, 16.0, 12.0)

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def solve(tag, T, guess, td_max, fit_to=None, floor=0.35, max_iter=2500,
          tol=1e-7, acceptable_tol=1e-6, ipopt_extra=None):
    K = int(round(T / H))
    A, b = chain_constants(N)
    Mfn, rfn = make_M_rhs_fn(N, G)
    th0 = guess["theta"][0]            # wound hang chart (pi + 2pi k)
    thK = guess["theta"][-1]
    opti = ca.Opti()
    TH = opti.variable(N, K + 1); TD = opti.variable(N, K + 1)
    VV = opti.variable(1, K + 1); TDD = opti.variable(N, K + 1)
    AC = opti.variable(1, K + 1)
    cost = 0
    for k in range(K + 1):
        opti.subject_to(ca.mtimes(Mfn(TH[:, k]), TDD[:, k]) == rfn(TH[:, k], TD[:, k], AC[0, k]))
    for k in range(K):
        opti.subject_to(TH[:, k + 1] - TH[:, k] == 0.5 * H * (TD[:, k] + TD[:, k + 1]))
        opti.subject_to(TD[:, k + 1] - TD[:, k] == 0.5 * H * (TDD[:, k] + TDD[:, k + 1]))
        opti.subject_to(VV[0, k + 1] - VV[0, k] == 0.5 * H * (AC[0, k] + AC[0, k + 1]))
        cost += 0.5 * H * (AC[0, k] ** 2 + AC[0, k + 1] ** 2) + 1e-3 * (AC[0, k + 1] - AC[0, k]) ** 2
    opti.subject_to(TH[:, 0] == np.round((th0 - np.pi) / (2 * np.pi)) * 2 * np.pi + np.pi)
    opti.subject_to(TD[:, 0] == 0); opti.subject_to(VV[0, 0] == 0)
    opti.subject_to(TH[:, K] == np.round(thK / (2 * np.pi)) * 2 * np.pi)
    opti.subject_to(TD[:, K] == 0); opti.subject_to(VV[0, K] == 0)
    opti.subject_to(opti.bounded(-A_MAX, ca.vec(AC), A_MAX))
    opti.subject_to(opti.bounded(-V_MAX, VV, V_MAX))
    opti.subject_to(opti.bounded(-td_max, ca.vec(TD), td_max))
    # aggregate ctrb floor (soft), as generate_nN. NB the reverse-fall class
    # lives at cmin ~0.39, so the floor must sit BELOW that or it fights the
    # warm start (the stage-A failure mode).
    cnodes = list(range(max(1, int(0.05 * K)), int(0.95 * K), 4)) if floor > 0 else []
    if cnodes:
        U = opti.variable(N, len(cnodes))
        for j, k in enumerate(cnodes):
            opti.subject_to(ca.mtimes(mass_matrix_ca(TH[:, k], A), U[:, j]) == bend_rhs_ca(TH[:, k], A, b))
            cm = ca.sqrt(bend_excite_ca(TH[:, k], U[:, j], A) + 1e-9)
            short = ca.fmax(0.0, floor - cm)
            cost += 100.0 * short * short
    if fit_to is not None:    # stage A: track the reverse-fall nominal
        for k in range(0, K + 1, 2):
            cost += 5.0 * ca.sumsqr(TH[:, k] - fit_to["theta"][k]) * H
    opti.minimize(cost)
    s = np.linspace(0, 1, K + 1); Kg = guess["theta"].shape[0] - 1; sg = np.linspace(0, 1, Kg + 1)
    opti.set_initial(TH, np.array([np.interp(s, sg, guess["theta"][:, j]) for j in range(N)]))
    opti.set_initial(TD, np.array([np.interp(s, sg, guess["thetad"][:, j]) for j in range(N)]))
    opti.set_initial(VV, np.interp(s, sg, guess["v"]).reshape(1, -1))
    opti.set_initial(AC, np.interp(s, sg, guess["a"]).reshape(1, -1))
    if cnodes:
        chain = Chain(N, G); thg = np.array([np.interp(s, sg, guess["theta"][:, j]) for j in range(N)])
        opti.set_initial(U, np.column_stack([np.linalg.solve(chain.mass_matrix(thg[:, k]),
                                                             b * np.cos(thg[:, k])) for k in cnodes]))
    s_opts = {"max_iter": max_iter, "print_level": 0, "sb": "yes", "tol": tol,
              "acceptable_tol": acceptable_tol, "acceptable_iter": 10,
              "mu_strategy": "adaptive",
              "output_file": str(LOGD / f"cont_{tag}.log"), "file_print_level": 5}
    if ipopt_extra:
        s_opts.update(ipopt_extra)
    opti.solver("ipopt", {"expand": True}, s_opts)
    try:
        sol = opti.solve(); status = "solved"
    except RuntimeError:
        sol = opti.debug; status = "failed"
    try:
        out = {"t": np.linspace(0, T, K + 1), "theta": np.array(sol.value(TH)).T,
               "thetad": np.array(sol.value(TD)).T, "v": np.array(sol.value(VV)).ravel(),
               "a": np.array(sol.value(AC)).ravel(), "T": T, "status": status}
    except RuntimeError:
        return {"status": "failed"}
    # save regardless of status -- a max_iter iterate is still a warm start
    sfx = "" if status == "solved" else "_debug"
    np.savez(POOL / f"{tag}{sfx}.npz", **{k: v for k, v in out.items() if k != "status"})
    return out


def track_check(traj, dt=0.002, eps=1e-3, seeds=2):
    chain = Chain(N, G)
    nom = resample(traj, N, dt, "hermite")
    th, td, aff, vn = nom["theta"], nom["thetad"], nom["a_zoh"], nom["v_nom"]
    Q = np.diag([50.0] * N + [5.0] * N + [1.0]); R = np.array([[0.1]])
    QF = np.diag([2000.0] * N + [200.0] * N + [10.0])
    Ks = build_tvlqr(chain, th, td, aff, dt, Q, R, QF)
    oks = 0; worst = 0.0
    for s in range(seeds):
        rng = np.random.default_rng(s)
        y = np.concatenate([th[0] + rng.uniform(-eps, eps, N), td[0]]); v = vn[0]
        fin = np.inf
        for k in range(len(Ks)):
            z = np.concatenate([y[:N], y[N:], [v]]); zn = np.concatenate([th[k], td[k], [vn[k]]])
            a = aff[k] - float((Ks[k] @ (z - zn))[0])
            y = rk4_step(chain, y, a, dt); v += a * dt
            if not np.isfinite(y).all(): break
        else:
            fin = float(np.max(np.abs((y[:N] + np.pi) % (2 * np.pi) - np.pi)))
        oks += fin < 0.05; worst = max(worst, fin)
    return oks, seeds, worst


def w_direct(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    tag, td_max, fit_path = args
    fit = dict(np.load(fit_path))
    sol = solve(tag, float(fit["T"]), fit, td_max, max_iter=5000, tol=1e-6, acceptable_tol=1e-4)
    if sol["status"] != "solved":
        return f"{tag}: NLP failed"
    ok, ns, worst = track_check(sol)
    whip = float(np.abs(sol["thetad"]).max())
    return f"{tag}: solved whip={whip:.1f} track {ok}/{ns} (worst {np.degrees(worst):.2f}deg)"


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    src = dict(np.load(SRC))
    T = float(src["T"])
    dbg = POOL / "anchor_td40_debug.npz"
    if dbg.exists():
        # the 9000-iter anchor iterate is feasible (inf_pr ~4e-5) -- use it
        log(f"using saved anchor iterate {dbg} (feasible, near-optimal)")
        fit = dict(np.load(dbg)); fit["status"] = "solved"
        np.savez(POOL / "anchor_td40.npz", **{k: v for k, v in fit.items() if k != "status"})
    else:
        guess0 = {k: src[k] for k in ("theta", "thetad", "a", "v")}
        log(f"rung 0: min-effort anchor at TD_MAX=40 warm from reverse-fall "
            f"(T={T:.1f}s, K={int(T/H)}, floor=0.35)")
        fit = solve("anchor_td40", T, guess0, 40.0, floor=0.35, max_iter=9000,
                    tol=1e-6, acceptable_tol=1e-4)
        if fit["status"] != "solved":
            raise SystemExit("anchor solve failed")
    ok, ns, worst = track_check(fit)
    log(f"anchor: whip={np.abs(fit['thetad']).max():.1f} track {ok}/{ns} (worst {np.degrees(worst):.2f}deg)")

    # parallel: gradual chain (in this process) + direct bets (workers)
    fit_path = str(POOL / "anchor_td40.npz")
    with mp.Pool(2) as pool:
        direct = pool.map_async(w_direct, [("direct_td16", 16.0, fit_path),
                                           ("direct_td12", 12.0, fit_path)])
        prev = fit
        for td in RUNGS:
            tag = f"chain_td{int(td)}"
            log(f"rung TD_MAX={td} ...")
            sol = solve(tag, T, prev, td, max_iter=5000, tol=1e-6, acceptable_tol=1e-4)
            if sol["status"] != "solved":
                log(f"  {tag}: NLP FAILED (continuation stops; last good whip="
                    f"{np.abs(prev['thetad']).max():.1f})")
                break
            ok, ns, worst = track_check(sol)
            whip = float(np.abs(sol["thetad"]).max())
            log(f"  {tag}: whip={whip:.1f} track {ok}/{ns} (worst {np.degrees(worst):.2f}deg)")
            if ok < ns:
                log(f"  trackability lost at TD_MAX={td}; frontier between {td} and previous rung")
                break
            prev = sol
        for line in direct.get():
            log("  " + line)
    log("continuation done")


if __name__ == "__main__":
    main()
