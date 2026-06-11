#!/usr/bin/env python
"""Controllability-aware, SEED-FREE swing-up generator for GENERAL N.

Generalises repro/generate_n6.py to arbitrary link count. Same idea: a NEUTRAL
cold homotopy ladder N=2..(N-1) -> lift to N, with a one-sided SOFT FLOOR on the
bend-mode excitation c(theta)=||M^-1(b o cos theta)||^2_bend so the optimiser
keeps the chain controllable (never dead-straight at the wrong moment). Then
test full-state TVLQR trackability, SWEEPING dt (higher N is faster-unstable, so
its trackable dt-window is narrower and at smaller dt than N=6's [0.008,0.004]).

Run:  uv run python repro/generate_nN.py N [n_workers]
Output: repro/n{N}_controls.npz  (consumed by simulate_n6.py, which is N-generic)
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import casadi as ca
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.trajopt import (chain_constants, make_M_rhs_fn,
                              solve_swingup_implicit, homotopy_guess)
from repro.optimize_n6 import build_tvlqr, linearize_z  # general helpers
from repro.generate_n6 import (bend_excite_ca, bend_rhs_ca, mass_matrix_ca,
                               bend_excite_np)

G = 9.81
N = int(sys.argv[1]) if len(sys.argv) > 1 else 7
NW = int(sys.argv[2]) if len(sys.argv) > 2 else min(12, mp.cpu_count() - 2)
A_MAX, V_MAX = 25.0, 14.0
TD_MAX = 12.0                       # link-rate (whip) bound for trackability
SETTLE_FRAC, SETTLE_BAND = 0.08, 0.15
# horizon for the N rung, soft-floor levels, and dt window to sweep
T_N = {6: 15.0, 7: 17.0}.get(N, 2.0 * N + 3.0)
FLOORS = tuple(round(float(f), 2) for f in np.arange(0.40, 0.98, 0.04))  # 15 levels -> ~15 cores + wider bend-order search
DTS = (0.004, 0.005, 0.006, 0.008, 0.010)  # N>=7: gains grow as dt shrinks, so sweep LARGER dt
POOL = pathlib.Path(f"repro/pool_ctrb_n{N}")
OUT = f"repro/n{N}_controls.npz"

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def solve_ctrb_aware(n, T, K, w_ctrb, floor_ctrb, init_guess, max_iter=2500, tol=1e-7):
    """Controllability-aware implicit collocation (general n)."""
    A, b = chain_constants(n)
    tgt = np.zeros(n); h = T / K
    Mfn, rfn = make_M_rhs_fn(n, G)
    opti = ca.Opti()
    TH = opti.variable(n, K + 1); TD = opti.variable(n, K + 1)
    VV = opti.variable(1, K + 1); TDD = opti.variable(n, K + 1)
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
    opti.subject_to(TH[:, K] == tgt); opti.subject_to(TD[:, K] == 0); opti.subject_to(VV[0, K] == 0)
    opti.subject_to(opti.bounded(-A_MAX, ca.vec(AC), A_MAX))
    opti.subject_to(opti.bounded(-V_MAX, VV, V_MAX))
    opti.subject_to(opti.bounded(-TD_MAX, ca.vec(TD), TD_MAX))   # bound link whip (trackability)
    kset = int(round((1 - SETTLE_FRAC) * K))
    for k in range(kset, K + 1):
        opti.subject_to(opti.bounded(tgt - SETTLE_BAND, TH[:, k], tgt + SETTLE_BAND))
    cnodes = list(range(max(1, int(0.05 * K)), kset, 3))
    if w_ctrb > 0 and floor_ctrb > 0 and cnodes:
        U = opti.variable(n, len(cnodes))
        for j, k in enumerate(cnodes):
            opti.subject_to(ca.mtimes(mass_matrix_ca(TH[:, k], A), U[:, j]) == bend_rhs_ca(TH[:, k], A, b))
            cm = ca.sqrt(bend_excite_ca(TH[:, k], U[:, j], A) + 1e-9)
            short = ca.fmax(0.0, floor_ctrb - cm)
            cost += w_ctrb * short * short
    opti.minimize(cost)
    s = np.linspace(0, 1, K + 1); Kg = init_guess["theta"].shape[0] - 1; sg = np.linspace(0, 1, Kg + 1)
    thg = np.array([np.interp(s, sg, init_guess["theta"][:, j]) for j in range(n)])
    opti.set_initial(TH, thg)
    opti.set_initial(TD, np.array([np.interp(s, sg, init_guess["thetad"][:, j]) for j in range(n)]))
    opti.set_initial(VV, np.interp(s, sg, init_guess["v"]).reshape(1, -1))
    opti.set_initial(AC, np.interp(s, sg, init_guess["a"]).reshape(1, -1))
    if w_ctrb > 0 and floor_ctrb > 0 and cnodes:
        chain = Chain(n, G)
        opti.set_initial(U, np.column_stack([np.linalg.solve(chain.mass_matrix(thg[:, k]), b * np.cos(thg[:, k])) for k in cnodes]))
    opti.solver("ipopt", {"expand": True}, {"max_iter": max_iter, "print_level": 0,
                "sb": "yes", "tol": tol, "acceptable_tol": 1e-6, "mu_strategy": "adaptive"})
    try:
        sol = opti.solve(); status = "solved"
    except RuntimeError:
        sol = opti.debug; status = "failed"
    try:
        return {"t": np.linspace(0, T, K + 1), "theta": np.array(sol.value(TH)).T,
                "thetad": np.array(sol.value(TD)).T, "v": np.array(sol.value(VV)).ravel(),
                "a": np.array(sol.value(AC)).ravel(), "status": status, "T": T, "n": n}
    except RuntimeError:
        return {"status": "failed", "n": n}


def neutral_ladder(n_top):
    """Cold homotopy ladder N=2..(n_top), target zeros, fine-refined at the top."""
    prev = None
    for nn in range(2, n_top + 1):
        T = {2: 4.0, 3: 5.0, 4: 7.0, 5: 11.0, 6: 14.0}.get(nn, 2.0 * nn + 2.0)
        guess = homotopy_guess(prev) if prev else None
        sol = None
        for sd in range(5):
            sol = solve_swingup_implicit(nn, T, int(T / 0.05), g=G, a_max=60.0, v_max=14.0,
                theta_target=np.zeros(nn), settle_frac=0.1, settle_band=0.2,
                seed=sd, init_guess=guess, max_iter=1500, print_level=0, tol=1e-6)
            if sol["status"] == "solved":
                break
        if sol["status"] != "solved":
            raise SystemExit(f"neutral ladder N={nn} (coarse) failed")
        prev = sol
        log(f"  ladder N={nn} coarse ok (T={T})")
    # MESH-REFINE the top rung: coarse -> h=0.02 -> h=0.01 (the direct
    # coarse->fine jump is unreliable for N>=6). Fall back to the finest that
    # converged -- the N-rung controllability solve does its own fine collocation
    # so a medium/coarse seed is an acceptable (if weaker) warm start.
    Ttop = {5: 11.0, 6: 14.0}.get(n_top, 2.0 * n_top + 2.0)
    best = prev
    for hh in (0.02, 0.01):
        ref = solve_swingup_implicit(n_top, Ttop, int(Ttop / hh), g=G, a_max=60.0, v_max=14.0,
            theta_target=np.zeros(n_top), settle_frac=0.1, settle_band=0.2,
            init_guess=best, max_iter=3000, print_level=0, tol=1e-7)
        if ref["status"] == "solved":
            best = ref; log(f"  ladder N={n_top} refined to h={hh}")
        else:
            log(f"  ladder N={n_top} refine to h={hh} failed; using coarser seed")
            break
    return best


def perfect_state(traj, n, dt):
    """Full-state TVLQR rollout at timestep dt (general n). Returns (ok, final, bundle)."""
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


def w_solve(task):
    os.environ["OMP_NUM_THREADS"] = "1"
    floor, seed_path = task
    seedN = dict(np.load(seed_path)); guess = homotopy_guess(seedN)
    sf = solve_ctrb_aware(N, T_N, int(T_N / 0.01), w_ctrb=100.0, floor_ctrb=floor,
                          init_guess=guess, max_iter=2500)
    if sf["status"] != "solved":
        return (floor, None)
    fn = POOL / f"floor{floor}.npz"
    np.savez(fn, t=sf["t"], theta=sf["theta"], thetad=sf["thetad"], a=sf["a"], v=sf["v"], T=sf["T"])
    return (floor, str(fn))


def w_eval(item):
    """Sweep dt for one trajectory; return the best (smallest-final) trackable result."""
    os.environ["OMP_NUM_THREADS"] = "1"
    floor, fn = item
    d = dict(np.load(fn))
    chain = Chain(N, G)
    cmin = float(min(bend_excite_np(chain, d["theta"][i]) for i in range(0, len(d["t"]), 3)))
    best = None
    for dt in DTS:
        ok, final, bundle = perfect_state(d, N, dt)
        rec = dict(floor=floor, fn=fn, dt=dt, ok=bool(ok), final=float(final),
                   cmin=cmin, maxK=(float(bundle["maxK"]) if bundle else float("nan")))
        if ok and (best is None or final < best["final"]):
            best = rec
    return best if best else dict(floor=floor, fn=fn, dt=None, ok=False, final=float("inf"), cmin=cmin, maxK=float("nan"))


def main():
    if POOL.exists():
        for p in POOL.glob("*.npz"):
            if "_seed" not in p.name:
                p.unlink()
    POOL.mkdir(parents=True, exist_ok=True)
    seed_path = POOL / "_neutral_seed.npz"
    # --seed PATH lets N use an (N-1) trajectory produced by THIS generator at a
    # lower N (e.g. the controllability-aware N=6 result seeds N=7). Still
    # seed-free in spirit: every rung comes from the same principled method, and
    # a gentle fine (N-1) is a far better warm start than a cold coarse ladder.
    seed_arg = None
    if "--seed" in sys.argv:
        seed_arg = sys.argv[sys.argv.index("--seed") + 1]
    if seed_arg:
        import shutil; shutil.copy(seed_arg, seed_path)
        log(f"using provided N={N-1} seed: {seed_arg}")
    elif seed_path.exists() and "--rebuild" not in sys.argv:
        log(f"using cached neutral N={N-1} seed")
    else:
        log(f"building NEUTRAL ladder N=2..{N-1} (no curated seed)...")
        seed = neutral_ladder(N - 1)
        np.savez(seed_path, t=seed["t"], theta=seed["theta"], thetad=seed["thetad"],
                 a=seed["a"], v=seed["v"], T=seed["T"])
    log(f"N={N-1} seed ready")

    log(f"solving {len(FLOORS)} controllability-aware N={N} (T={T_N}, {NW} workers)...")
    with mp.Pool(NW) as pool:
        solved = [r for r in pool.map(w_solve, [(fl, str(seed_path)) for fl in FLOORS]) if r[1]]
    log(f"  {len(solved)}/{len(FLOORS)} solved")
    if not solved:
        raise SystemExit(f"no controllability-aware N={N} solved")

    log(f"evaluating trackability over dt in {DTS} ...")
    with mp.Pool(NW) as pool:
        evals = pool.map(w_eval, solved)
    evals.sort(key=lambda e: (not e["ok"], e["final"]))
    for e in evals:
        log(f"  floor={e['floor']} dt={e['dt']} ok={e['ok']!s:5s} "
            f"final={np.degrees(e['final']):8.3f}deg cmin={e['cmin']:.3f} maxK={e['maxK']:.0f}")
    track = [e for e in evals if e["ok"]]
    if not track:
        raise SystemExit(f"no trackable N={N} (closest {np.degrees(evals[0]['final']):.2f}deg)")
    best = track[0]
    _, _, bundle = perfect_state(dict(np.load(best["fn"])), N, best["dt"])
    np.savez(OUT, **bundle)
    log(f"SAVED {OUT}: N={N} dt={best['dt']} final={np.degrees(best['final']):.4f}deg maxK={bundle['maxK']:.0f}")


if __name__ == "__main__":
    main()
