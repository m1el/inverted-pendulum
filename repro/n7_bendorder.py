#!/usr/bin/env python
"""Idea #7: solve N=7 swing-up constrained to a PRESCRIBED bending order.

The free solver hits a per-mode controllability wall. Here we *prescribe* the
sign schedule of the joint bend angles beta_i(t)=theta_{i+1}-theta_i over the
swing -- which joints bend which way, and in which order -- forcing a coordinated
bend (like the trackable fold) instead of an uncontrollable near-straight shimmy.

A schedule maps (normalized time s, joint i) -> prescribed sign in {-1,0,+1};
where nonzero we constrain sign*beta_i >= margin over the active swing window.
We sweep several orders, solve each (controllability floor kept), and test
full-state TVLQR trackability.

Run:  uv run python repro/n7_bendorder.py [n_workers]
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import expm
import warnings; warnings.filterwarnings("ignore")
import casadi as ca
import multiprocessing as mp
from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.trajopt import chain_constants, make_M_rhs_fn
from repro.optimize_n6 import linearize_z
from repro.generate_n6 import bend_excite_ca, bend_rhs_ca, mass_matrix_ca, bend_excite_np

G = 9.81; N = 7; T_TOTAL = 17.0; H_DT = 0.01
A_MAX, V_MAX, TD_MAX = 25.0, 14.0, 12.0
SETTLE_FRAC, SETTLE_BAND = 0.08, 0.15
DTS = (0.004, 0.005, 0.006, 0.008, 0.01)
MARG = 0.05                      # required |beta| where a sign is prescribed
ACTIVE = (0.12, 0.90)            # normalized-time window where bend order is enforced
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 8
_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


# ---- prescribed bend-order schedules: s in [0,1], return (N-1) signs ----
def sched_all_right_late(s):
    return np.ones(N - 1) if s > 0.55 else np.zeros(N - 1)            # C-shape approach only
def sched_wave_bt(s):                                                # right-bend wave base->tip
    return np.array([1.0 if s > 0.25 + 0.5 * i / (N - 1) else -1.0 for i in range(N - 1)])
def sched_wave_tb(s):                                                # wave tip->base
    return np.array([1.0 if s > 0.25 + 0.5 * (N - 2 - i) / (N - 1) else -1.0 for i in range(N - 1)])
def sched_seq_flip(s):                                               # joints flip - -> + in base->tip order, end all +
    return np.array([1.0 if s > 0.35 + 0.45 * i / (N - 1) else -1.0 for i in range(N - 1)])
def sched_alt_then_unify(s):                                         # alternate, then all + late
    if s > 0.6: return np.ones(N - 1)
    return np.array([(-1.0) ** i for i in range(N - 1)])
def sched_Cshape_constant(s):                                        # one-sided C the whole active swing
    return np.ones(N - 1)
def sched_two_wave(s):                                               # down-wave then up-wave
    if s < 0.5: return np.array([-1.0 if s > 0.15 + 0.3 * i / (N - 1) else 1.0 for i in range(N - 1)])
    return np.array([1.0 if s > 0.55 + 0.3 * i / (N - 1) else -1.0 for i in range(N - 1)])

SCHEDULES = {"all_right_late": sched_all_right_late, "wave_base_tip": sched_wave_bt,
             "wave_tip_base": sched_wave_tb, "seq_flip_bt": sched_seq_flip,
             "alt_then_unify": sched_alt_then_unify, "Cshape_const": sched_Cshape_constant,
             "two_wave": sched_two_wave}


def solve(name, floor=0.6, w_ctrb=100.0, max_iter=3000):
    A, b = chain_constants(N); K = int(round(T_TOTAL / H_DT)); h = T_TOTAL / K
    Mfn, rfn = make_M_rhs_fn(N, G); sched = SCHEDULES[name]
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
    opti.subject_to(TH[:, 0] == np.pi); opti.subject_to(TD[:, 0] == 0); opti.subject_to(VV[0, 0] == 0)
    opti.subject_to(TH[:, K] == 0); opti.subject_to(TD[:, K] == 0); opti.subject_to(VV[0, K] == 0)
    opti.subject_to(opti.bounded(-A_MAX, ca.vec(AC), A_MAX))
    opti.subject_to(opti.bounded(-V_MAX, VV, V_MAX))
    opti.subject_to(opti.bounded(-TD_MAX, ca.vec(TD), TD_MAX))
    kset = int(round((1 - SETTLE_FRAC) * K))
    for k in range(kset, K + 1):
        opti.subject_to(opti.bounded(-SETTLE_BAND, TH[:, k], SETTLE_BAND))
    # --- prescribed bend-order sign constraints ---
    nconstr = 0
    for k in range(int(ACTIVE[0] * K), int(ACTIVE[1] * K), 2):
        s = k / K; sg = sched(s)
        for i in range(N - 1):
            if sg[i] != 0:
                opti.subject_to(sg[i] * (TH[i + 1, k] - TH[i, k]) >= MARG); nconstr += 1
    # controllability floor (aggregate) on a sparse node set
    cnodes = list(range(int(0.05 * K), kset, 4))
    if w_ctrb > 0 and floor > 0:
        U = opti.variable(N, len(cnodes))
        for j, k in enumerate(cnodes):
            opti.subject_to(ca.mtimes(mass_matrix_ca(TH[:, k], A), U[:, j]) == bend_rhs_ca(TH[:, k], A, b))
            cm = ca.sqrt(bend_excite_ca(TH[:, k], U[:, j], A) + 1e-9)
            short = ca.fmax(0.0, floor - cm); cost += w_ctrb * short * short
    opti.minimize(cost)
    # init: interpolate hang->upright, with a small bend nudge toward the schedule at mid
    s = np.linspace(0, 1, K + 1); thg = np.outer(np.full(N, np.pi), 1 - s)  # pi->0
    opti.set_initial(TH, thg); opti.set_initial(VV, np.zeros((1, K + 1))); opti.set_initial(AC, np.zeros((1, K + 1)))
    if w_ctrb > 0 and floor > 0:
        opti.set_initial(U, np.column_stack([np.linalg.solve(Chain(N, G).mass_matrix(thg[:, k]), b * np.cos(thg[:, k])) for k in cnodes]))
    opti.solver("ipopt", {"expand": True}, {"max_iter": max_iter, "print_level": 0, "sb": "yes",
                "tol": 1e-6, "acceptable_tol": 1e-5, "mu_strategy": "adaptive"})
    try:
        sol = opti.solve(); st = "solved"
    except RuntimeError:
        sol = opti.debug; st = "failed"
    try:
        return dict(name=name, status=st, nconstr=nconstr, t=np.linspace(0, T_TOTAL, K + 1),
                    theta=np.array(sol.value(TH)).T, thetad=np.array(sol.value(TD)).T,
                    v=np.array(sol.value(VV)).ravel(), a=np.array(sol.value(AC)).ravel())
    except RuntimeError:
        return dict(name=name, status="failed")


def perfect_state(traj, dt):
    chain = Chain(N, G); Th = traj["t"][-1]; tn = np.arange(0, Th + 1e-9, dt)
    th = np.vstack([np.interp(tn, traj["t"], traj["theta"][:, i]) for i in range(N)]).T
    td = np.vstack([np.interp(tn, traj["t"], traj["thetad"][:, i]) for i in range(N)]).T
    a_ff = np.interp(tn, traj["t"], traj["a"]); v_nom = np.interp(tn, traj["t"], traj["v"])
    nz = 2 * N + 1; Q = np.diag([50.0] * N + [5.0] * N + [1.0]); R = np.array([[0.1]]); QF = np.diag([2000.0] * N + [200.0] * N + [10.0])
    ABs = []
    for k in range(len(th) - 1):
        Ac, Bc = linearize_z(chain, th[k], td[k], a_ff[k])
        blk = np.zeros((nz + 1, nz + 1)); blk[:nz, :nz] = Ac * dt; blk[:nz, nz:] = Bc * dt; E = expm(blk)
        ABs.append((E[:nz, :nz], E[:nz, nz:]))
    P = QF.copy(); Ks = []
    for Ad, Bd in reversed(ABs):
        S = R + Bd.T @ P @ Bd; Kk = np.linalg.solve(S, Bd.T @ P @ Ad); Ks.append(Kk); P = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ Kk
    Ks = Ks[::-1]
    y = np.concatenate([th[0], td[0]]); v = 0.0; mk = 0.0
    for k in range(len(Ks)):
        z = np.concatenate([y[:N], y[N:], [v]]); zn = np.concatenate([th[k], td[k], [v_nom[k]]])
        a = a_ff[k] - float((Ks[k] @ (z - zn))[0]); mk = max(mk, np.abs(Ks[k]).max())
        y = rk4_step(chain, y, a, dt); v += a * dt
        if not np.isfinite(y).all(): return False, np.inf, mk
    fin = float(np.max(np.abs((y[:N] + np.pi) % (2 * np.pi) - np.pi)))
    return fin < 0.05, np.degrees(fin), mk


def w_run(name):
    os.environ["OMP_NUM_THREADS"] = "1"
    tr = solve(name)
    if tr["status"] != "solved":
        return dict(name=name, ok=False, status="NLP failed")
    cmin = float(min(bend_excite_np(Chain(N, G), tr["theta"][i]) for i in range(0, len(tr["t"]), 4)))
    best = None
    for dt in DTS:
        ok, fin, mk = perfect_state(tr, dt)
        if ok and (best is None or fin < best[1]): best = (dt, fin, mk)
    return dict(name=name, ok=best is not None, status="solved", cmin=cmin, nconstr=tr["nconstr"],
                dt=(best[0] if best else None), final=(best[1] if best else np.inf), maxK=(best[2] if best else np.nan))


def main():
    log(f"N=7 prescribed bend-order sweep: {len(SCHEDULES)} schedules ({NW} workers)")
    with mp.Pool(NW) as pool:
        res = pool.map(w_run, list(SCHEDULES))
    log("results:")
    for r in sorted(res, key=lambda r: (not r["ok"], r.get("final", 1e9))):
        if r["status"] != "solved":
            log(f"  {r['name']:16s}: NLP FAILED"); continue
        log(f"  {r['name']:16s}: {'TRACKABLE' if r['ok'] else 'not trackable':14s} "
            f"dt={r['dt']} final={r['final']:8.3f}deg cmin={r['cmin']:.3f} maxK={r['maxK']:.0f} ({r['nconstr']} sign-constr)")
    win = [r for r in res if r["ok"]]
    log(f"TRACKABLE bend orders: {[r['name'] for r in win] or 'NONE'}")


if __name__ == "__main__":
    main()
