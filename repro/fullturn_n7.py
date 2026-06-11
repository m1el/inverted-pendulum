#!/usr/bin/env python
"""Minimal-winding N=7: pump energy via ONE coordinated full turn (user idea).

Trajectory class: hang (theta=pi) -> ONE full revolution beyond the classical
unwind -> upright at theta = -2pi (the chain swings through upright once with
a bent configuration, completes the turn, and stabilizes on the next arrival).
Winding W = -1.5 rev per link: the smallest step beyond the certified-deaf
classical W = -0.5 class. Tests whether a single turn buys the late-window
per-mode coupling (>=0.2) that trackability demands.

Corridor: theta in [-2pi-1, 5.9] (one turn allowed, no more).
Sweep T x ctrb-floor x seed; gate with RK4-Jacobian TVLQR (dt 0.002/0.004) +
measure late-window worst-mode coupling.

Run:  uv run python repro/fullturn_n7.py [n_workers]
IPOPT logs: runs_tmp/ipopt/ft_*.log
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import casadi as ca
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.trajopt import chain_constants, make_M_rhs_fn
from repro.generate_n6 import bend_excite_ca, bend_rhs_ca, mass_matrix_ca, bend_excite_np
from repro.rk4_tvlqr import build_tvlqr_rk4
from pendulum.sim import rk4_step

G = 9.81; N = 7
A_MAX, V_MAX, TD_MAX = 25.0, 14.0, 16.0
TH_LO, TH_HI = -2 * np.pi - 1.0, 5.9
TARGET = -2 * np.pi * np.ones(N)
POOL = pathlib.Path("repro/pool_fullturn_n7"); POOL.mkdir(parents=True, exist_ok=True)
LOGD = pathlib.Path("runs_tmp/ipopt"); LOGD.mkdir(parents=True, exist_ok=True)
DTS = (0.002, 0.004)

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def solve_ft(tag, T, floor, seed, max_iter=3000, h=0.01, init_guess=None,
             e_margin=0.0, e_span=(0.25, 0.80), target=None, beta_max=0.0,
             td_max=None, a_max=None, v_max=None, th_bounds=None,
             free_winding=False, ipopt_extra=None):
    """e_margin > 0 adds an ENERGY FLOOR: total energy >= V_upright + e_margin
    on t/T in e_span -- forbids the near-balance creep over the top (the chain
    must cross upright with at least e_margin of kinetic energy: transversal
    by construction). beta_max > 0 bounds the JOINT BENDS |theta_{i+1}-theta_i|
    (no convolution -- the 'nice' constraint); td/a/v_max and target/th_bounds
    override the module defaults (free-turn class: wide th_bounds, winding via
    target)."""
    tgt = TARGET if target is None else np.asarray(target, float)
    tdm = TD_MAX if td_max is None else td_max
    am = A_MAX if a_max is None else a_max
    vm = V_MAX if v_max is None else v_max
    tlo, thi = (TH_LO, TH_HI) if th_bounds is None else th_bounds
    K = int(round(T / h))
    A, b = chain_constants(N)
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
    if free_winding:
        # upright MODULO full turns: winding count is EMERGENT, not prescribed.
        # sin=0 with cos>=0.5 pins theta_K to 2*pi*k exactly (healthy gradients).
        opti.subject_to(ca.sin(TH[:, K]) == 0)
        opti.subject_to(ca.cos(TH[:, K]) >= 0.5)
    else:
        opti.subject_to(TH[:, K] == tgt)
    opti.subject_to(TD[:, K] == 0); opti.subject_to(VV[0, K] == 0)
    opti.subject_to(opti.bounded(-am, ca.vec(AC), am))
    opti.subject_to(opti.bounded(-vm, VV, vm))
    opti.subject_to(opti.bounded(-tdm, ca.vec(TD), tdm))
    opti.subject_to(opti.bounded(tlo, ca.vec(TH), thi))
    if beta_max > 0:                       # NO CONVOLUTION: bounded joint bends
        for i in range(N - 1):
            opti.subject_to(opti.bounded(-beta_max, TH[i + 1, :] - TH[i, :], beta_max))
    if e_margin > 0:
        V_up = G * float(np.sum(b))           # potential at upright
        for k in range(int(e_span[0] * K), int(e_span[1] * K), 3):
            Mk = mass_matrix_ca(TH[:, k], A)
            E = 0.5 * ca.mtimes(TD[:, k].T, ca.mtimes(Mk, TD[:, k])) \
                + G * ca.mtimes(ca.DM(b).T, ca.cos(TH[:, k]))
            opti.subject_to(E >= V_up + e_margin)
    cn = list(range(max(1, int(0.05 * K)), int(0.95 * K), 4))
    U = opti.variable(N, len(cn))
    for j, k in enumerate(cn):
        opti.subject_to(ca.mtimes(mass_matrix_ca(TH[:, k], A), U[:, j]) == bend_rhs_ca(TH[:, k], A, b))
        cm = ca.sqrt(bend_excite_ca(TH[:, k], U[:, j], A) + 1e-9)
        short = ca.fmax(0.0, floor - cm)
        cost += 100.0 * short * short
    opti.minimize(cost)
    rng = np.random.default_rng(seed)
    s = np.linspace(0, 1, K + 1)
    if init_guess is not None:
        Kg = init_guess["theta"].shape[0] - 1; sg = np.linspace(0, 1, Kg + 1)
        thg = np.array([np.interp(s, sg, init_guess["theta"][:, j]) for j in range(N)])
        opti.set_initial(TH, thg)
        opti.set_initial(TD, np.array([np.interp(s, sg, init_guess["thetad"][:, j]) for j in range(N)]))
        opti.set_initial(VV, np.interp(s, sg, init_guess["v"]).reshape(1, -1))
        opti.set_initial(AC, np.interp(s, sg, init_guess["a"]).reshape(1, -1))
    else:
        thg = np.outer(np.full(N, np.pi), 1 - s) + np.outer(tgt, s)
        thg += rng.uniform(-0.5, 0.5, thg.shape) * np.sin(np.pi * s)
        opti.set_initial(TH, np.clip(thg, tlo + 0.05, thi - 0.05))
        opti.set_initial(TD, rng.uniform(-1, 1, (N, K + 1)))
        opti.set_initial(VV, rng.uniform(-2, 2, (1, K + 1)))
        opti.set_initial(AC, rng.uniform(-5, 5, (1, K + 1)))
    chain = Chain(N, G)
    opti.set_initial(U, np.column_stack([np.linalg.solve(chain.mass_matrix(thg[:, k]),
                                                         b * np.cos(thg[:, k])) for k in cn]))
    s_opts = {"max_iter": max_iter, "print_level": 0, "sb": "yes", "tol": 1e-6,
              "acceptable_tol": 1e-5, "acceptable_iter": 10, "mu_strategy": "adaptive",
              "output_file": str(LOGD / f"ft_{tag}.log"), "file_print_level": 5}
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
               "a": np.array(sol.value(AC)).ravel(), "T": T}
    except RuntimeError:
        return None, "failed"
    # save regardless: a feasible max_iter iterate is still a usable nominal
    np.savez(POOL / (f"{tag}.npz" if status == "solved" else f"{tag}_debug.npz"), **out)
    return out, status


def gate(sol):
    """RK4-Jac TVLQR at DTS; 4-seed +-0.05 rad perturbed; returns best line."""
    chain = Chain(N, G)
    t = sol["t"]; T = t[-1]
    best = None
    for dt in DTS:
        tn = np.arange(0, T + 1e-9, dt)
        th = np.vstack([np.interp(tn, t, sol["theta"][:, i]) for i in range(N)]).T
        td = np.vstack([np.interp(tn, t, sol["thetad"][:, i]) for i in range(N)]).T
        aff = np.interp(tn, t, sol["a"]); vn = np.interp(tn, t, sol["v"])
        Q = np.diag([50.0] * N + [5.0] * N + [1.0]); R = np.array([[0.1]])
        QF = np.diag([2000.0] * N + [200.0] * N + [10.0])
        Ks = build_tvlqr_rk4(chain, th, td, aff, dt, Q, R, QF)
        oks = 0
        for s in range(4):
            rng = np.random.default_rng(s)
            y = np.concatenate([th[0] + rng.uniform(-0.05, 0.05, N), td[0]]); v = vn[0]
            fin = np.inf
            for k in range(len(Ks)):
                z = np.concatenate([y[:N], y[N:], [v]]); zn = np.concatenate([th[k], td[k], [vn[k]]])
                a = aff[k] - float((Ks[k] @ (z - zn))[0])
                y = rk4_step(chain, y, a, dt); v += a * dt
                if not np.isfinite(y).all(): break
            else:
                fin = float(np.max(np.abs((y[:N] + np.pi) % (2 * np.pi) - np.pi)))
            oks += fin < 0.05
        if best is None or oks > best[1]:
            best = (dt, oks)
    return best


def w_case(task):
    os.environ["OMP_NUM_THREADS"] = "1"
    T, floor, seed, em = task
    tag = f"T{int(T)}_f{floor:g}_s{seed}_e{em:g}"
    sol, status = solve_ft(tag, T, floor, seed, e_margin=em)
    if status != "solved":
        return f"{tag}: {status}"
    sys.path.insert(0, "repro")
    from n7_ideas import permode
    chain = Chain(N, G)
    t = sol["t"]; T_ = t[-1]
    # coupling over the WHOLE post-pump segment (the v1 window missed the
    # near-balance creep at the first crossing)
    m = (t >= 0.3 * T_)
    cl = float(permode(chain, sol["theta"][m]).min())
    whip = float(np.abs(sol["thetad"]).max())
    dt, oks = gate(sol)
    return (f"{tag}: solved whip={whip:.1f} worst-mode[0.3T:]={cl:.3f} "
            f"GATE {oks}/4 @ dt={dt:g}")


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    tasks = [(T, 0.5, sd, em) for T in (12.0, 16.0) for sd in (0, 1) for em in (20.0, 40.0)]
    log(f"full-turn N=7 v2 (energy floor; target -2pi/link): {len(tasks)} solves on {nw} workers")
    with mp.Pool(nw) as pool:
        for line in pool.imap_unordered(w_case, tasks):
            log("  " + line)
    log("full-turn search done")


if __name__ == "__main__":
    main()
