#!/usr/bin/env python
"""N=6 challenge: transit between two UNSTABLE equilibria and back.

    all-up [0]*6  ->  one-link-down (link L at pi, rest up)  ->  all-up

The nominal comes momentarily to rest at the one-down configuration (optionally
dwelling there), then returns and balances upright. It is a controllability-
aware implicit-collocation trajectory tracked by FULL-STATE TVLQR; verified by a
closed-loop RK4 rollout that ends in the upright catch (simulate_n6.py).

Both endpoints AND the start are the dead-straight vertical chain -- the
maximally near-uncontrollable configuration (the N=6 wall). The bend-mode soft
floor keeps the transit excitable; the dwell at one-down is genuinely bent and
easy, the hard part is leaving / re-entering straight-up.

Run:  uv run python repro/flip_one_n6.py [N] [L0,L1,...] [n_workers]
      (default N=6; default links = all; saves each feasible+trackable link)
Output: repro/flip_n{N}_L{L}_controls.npz   (consumed by simulate_n6.py)
"""
import sys, os, pathlib, time, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import casadi as ca
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.trajopt import chain_constants, make_M_rhs_fn
from repro.optimize_n6 import build_tvlqr
from repro.generate_n6 import (bend_excite_ca, bend_rhs_ca, mass_matrix_ca,
                               bend_excite_np)


def perfect_state(traj, n, dt):
    """Full-state TVLQR rollout at timestep dt. Returns (ok, final, bundle).

    Identical to generate_nN.perfect_state, inlined to avoid importing that
    module (it parses sys.argv at import time)."""
    chain = Chain(n, G); Th = float(traj["t"][-1])
    tn = np.arange(0.0, Th + 1e-9, dt)
    theta = np.vstack([np.interp(tn, traj["t"], traj["theta"][:, i]) for i in range(n)]).T
    thetad = np.vstack([np.interp(tn, traj["t"], traj["thetad"][:, i]) for i in range(n)]).T
    a_ff = np.interp(tn, traj["t"], traj["a"]); v_nom = np.interp(tn, traj["t"], traj["v"])
    Q = np.diag([50.0] * n + [5.0] * n + [1.0]); R = np.array([[0.1]])
    QF = np.diag([2000.0] * n + [200.0] * n + [10.0])
    Ks = build_tvlqr(chain, theta, thetad, a_ff, dt, Q, R, QF)
    y = np.concatenate([theta[0], thetad[0]]); v = 0.0; maxK = 0.0
    for k in range(len(Ks)):
        z = np.concatenate([y[:n], y[n:], [v]]); zn = np.concatenate([theta[k], thetad[k], [v_nom[k]]])
        a = a_ff[k] - float((Ks[k] @ (z - zn))[0]); maxK = max(maxK, float(np.abs(Ks[k]).max()))
        y = rk4_step(chain, y, a, dt); v += a * dt
        if not np.isfinite(y).all():
            return False, np.inf, None
    final = float(np.max(np.abs((y[:n] + np.pi) % (2 * np.pi) - np.pi)))
    bundle = dict(dt=dt, g=G, n=n, T=Th, t=tn, theta_nom=theta, thetad_nom=thetad,
                  a_ff=a_ff, v_nom=v_nom, K=Ks, maxK=maxK, final=final)
    return final < 0.05, final, bundle

G = 9.81
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
A_MAX, V_MAX = 30.0, 14.0
T_TOTAL = {2: 5.0, 3: 6.0, 4: 8.0, 5: 11.0, 6: 16.0}.get(N, 2.5 * N + 1.0)
DWELL = 0.0                       # seconds held at the one-down via (0 = touch & go)
SETTLE_FRAC, SETTLE_BAND = 0.10, 0.15
FLOORS = (0.65,)
DTS = (0.004, 0.005, 0.006, 0.008, 0.01)   # low N is less unstable -> coarser dt ok
H_DT = 0.01                       # collocation step

LINKS = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(N)))
NW = int(sys.argv[3]) if len(sys.argv) > 3 else min(16, mp.cpu_count() - 2)

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def solve(L, floor, w_ctrb=100.0, max_iter=5000, tol=1e-7):
    """Implicit collocation: all-up -> (link L down) -> all-up, ctrb-aware."""
    A, b = chain_constants(N)
    K = int(round(T_TOTAL / H_DT)); h = T_TOTAL / K
    kmid = K // 2; dh = int(round((DWELL / 2) / H_DT))
    Mfn, rfn = make_M_rhs_fn(N, G)
    theta_A = np.zeros(N)
    theta_B = np.zeros(N); theta_B[L] = np.pi

    opti = ca.Opti()
    TH = opti.variable(N, K + 1); TD = opti.variable(N, K + 1)
    VV = opti.variable(1, K + 1); TDD = opti.variable(N, K + 1); AC = opti.variable(1, K + 1)
    cost = 0
    for k in range(K + 1):
        opti.subject_to(ca.mtimes(Mfn(TH[:, k]), TDD[:, k]) == rfn(TH[:, k], TD[:, k], AC[0, k]))
    for k in range(K):
        opti.subject_to(TH[:, k + 1] - TH[:, k] == 0.5 * h * (TD[:, k] + TD[:, k + 1]))
        opti.subject_to(TD[:, k + 1] - TD[:, k] == 0.5 * h * (TDD[:, k] + TDD[:, k + 1]))
        opti.subject_to(VV[0, k + 1] - VV[0, k] == 0.5 * h * (AC[0, k] + AC[0, k + 1]))
        cost += 0.5 * h * (AC[0, k] ** 2 + AC[0, k + 1] ** 2) + 1e-3 * (AC[0, k + 1] - AC[0, k]) ** 2
    # endpoints: all-up at rest
    opti.subject_to(TH[:, 0] == theta_A); opti.subject_to(TD[:, 0] == 0); opti.subject_to(VV[0, 0] == 0)
    opti.subject_to(TH[:, K] == theta_A); opti.subject_to(TD[:, K] == 0); opti.subject_to(VV[0, K] == 0)
    # via: one-down at rest, held over the dwell window
    via = range(kmid - dh, kmid + dh + 1)
    for k in via:
        opti.subject_to(TH[:, k] == theta_B); opti.subject_to(TD[:, k] == 0)
    opti.subject_to(opti.bounded(-A_MAX, ca.vec(AC), A_MAX))
    opti.subject_to(opti.bounded(-V_MAX, VV, V_MAX))
    # settle band around the final all-up
    kset = int(round((1 - SETTLE_FRAC) * K))
    for k in range(kset, K + 1):
        opti.subject_to(opti.bounded(theta_A - SETTLE_BAND, TH[:, k], theta_A + SETTLE_BAND))
    # bend-mode controllability soft floor (skip the fixed dwell window)
    cnodes = [k for k in range(int(0.05 * K), K, 3) if k not in via]
    if w_ctrb > 0 and floor > 0 and cnodes:
        U = opti.variable(N, len(cnodes))
        for j, k in enumerate(cnodes):
            opti.subject_to(ca.mtimes(mass_matrix_ca(TH[:, k], A), U[:, j]) == bend_rhs_ca(TH[:, k], A, b))
            cm = ca.sqrt(bend_excite_ca(TH[:, k], U[:, j], A) + 1e-9)
            short = ca.fmax(0.0, floor - cm)
            cost += w_ctrb * short * short
    opti.minimize(cost)
    # initial guess: smooth cosine flip 0->pi->0 on link L (zero velocity at the
    # ends and at the via), rest ~0 -- a far better warm start than a kinked ramp
    s = np.linspace(0, 1, K + 1)
    thg = np.zeros((N, K + 1)); thg[L, :] = (np.pi / 2) * (1 - np.cos(2 * np.pi * s))
    opti.set_initial(TH, thg)
    opti.set_initial(VV, np.zeros((1, K + 1))); opti.set_initial(AC, np.zeros((1, K + 1)))
    if w_ctrb > 0 and floor > 0 and cnodes:
        chain = Chain(N, G)
        opti.set_initial(U, np.column_stack(
            [np.linalg.solve(chain.mass_matrix(thg[:, k]), b * np.cos(thg[:, k])) for k in cnodes]))
    opti.solver("ipopt", {"expand": True}, {"max_iter": max_iter, "print_level": 0,
                "sb": "yes", "tol": tol, "acceptable_tol": 1e-6, "mu_strategy": "adaptive"})
    try:
        sol = opti.solve(); status = "solved"
    except RuntimeError:
        sol = opti.debug; status = "failed"
    try:
        return {"t": np.linspace(0, T_TOTAL, K + 1), "theta": np.array(sol.value(TH)).T,
                "thetad": np.array(sol.value(TD)).T, "v": np.array(sol.value(VV)).ravel(),
                "a": np.array(sol.value(AC)).ravel(), "status": status, "L": L, "floor": floor}
    except RuntimeError:
        return {"status": "failed", "L": L, "floor": floor}


def w_solve(task):
    os.environ["OMP_NUM_THREADS"] = "1"
    L, floor = task
    return solve(L, floor)


def w_eval(traj):
    """dt-sweep full-state trackability; return best trackable bundle record."""
    os.environ["OMP_NUM_THREADS"] = "1"
    L = traj["L"]; chain = Chain(N, G)
    cmin = float(min(bend_excite_np(chain, traj["theta"][i]) for i in range(0, len(traj["t"]), 3)))
    best = None
    for dt in DTS:
        ok, final, bundle = perfect_state(traj, N, dt)
        if ok and (best is None or final < best[1]):
            best = (bundle, final, dt)
    if best:
        return dict(L=L, floor=traj["floor"], ok=True, dt=best[2], final=best[1],
                    cmin=cmin, maxK=float(best[0]["maxK"]), bundle=best[0])
    return dict(L=L, floor=traj["floor"], ok=False, dt=None, final=float("inf"), cmin=cmin, maxK=float("nan"))


def main():
    log(f"N={N} flip-one challenge: links={LINKS} floors={FLOORS} "
        f"T={T_TOTAL}s dwell={DWELL}s ({NW} workers)")
    tasks = list(itertools.product(LINKS, FLOORS))
    with mp.Pool(NW) as pool:
        sols = pool.map(w_solve, tasks)
    solved = [s for s in sols if s.get("status") == "solved"]
    log(f"  solved {len(solved)}/{len(tasks)} (link,floor) NLPs")
    if not solved:
        raise SystemExit("no flip trajectory solved")

    log(f"evaluating full-state trackability over dt in {DTS} ...")
    with mp.Pool(NW) as pool:
        evals = pool.map(w_eval, solved)
    # best per link
    best = {}
    for e in sorted(evals, key=lambda e: (not e["ok"], e["final"])):
        if e["ok"] and e["L"] not in best:
            best[e["L"]] = e
    for e in sorted(evals, key=lambda e: (e["L"], not e["ok"], e["final"])):
        log(f"  L={e['L']} floor={e['floor']} dt={e['dt']} ok={e['ok']!s:5s} "
            f"final={np.degrees(e['final']):8.3f}deg cmin={e['cmin']:.3f} maxK={e['maxK']:.0f}")
    if not best:
        raise SystemExit("no flip trajectory was full-state trackable")
    for L, e in sorted(best.items()):
        out = f"repro/flip_n{N}_L{L}_controls.npz"
        np.savez(out, **e["bundle"])
        log(f"SAVED {out}: L={L} dt={e['dt']} final={np.degrees(e['final']):.4f}deg maxK={e['maxK']:.0f}")
    log(f"trackable links: {sorted(best)}")


if __name__ == "__main__":
    main()
