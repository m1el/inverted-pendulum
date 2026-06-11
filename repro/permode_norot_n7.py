#!/usr/bin/env python
"""Strategy 2: windowed PER-MODE coupling floor + no-rotation corridor (N=7).

The no-rotation N=7 failed with aggregate cmin~1.0 but worst-mode coupling 0.12
in the unstable final third (PAPER 4.3) -- the aggregate floor is the wrong
surrogate. Here the NLP gets the *measured* requirement directly: for every
upright bending mode j and every short window W in the late swing, the
window-mean normalized coupling must clear a floor:

    mean_{k in W} (v_j^T u_k)^2  >=  c^2 * mean_{k in W} u_k^T u_k     (hinged)

with u_k the implicit input direction (M u = b o cos theta), V the fixed
upright mode basis (same as n7_ideas.permode), windows ~1.5 s over
t/T in [0.55, 0.95]. Plus the no-rotation corridor theta in [-1, 5.9],
|thetad| <= 12, T in {25, 30}, floors c in {0.20, 0.27}.

If it solves AND tracks -> a classical trackable N=7 exists.
If the NLP is infeasible at c=0.27 -> certified: no classical N=7 keeps all
modes audible at the trackable class's measured level (winding theorem).

Run:  uv run python repro/permode_norot_n7.py [n_workers]
IPOPT logs: runs_tmp/ipopt/pm_*.log
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import casadi as ca
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.trajopt import chain_constants, make_M_rhs_fn, homotopy_guess
from repro.generate_n6 import bend_excite_ca, bend_rhs_ca, mass_matrix_ca
from repro.continuation_n7 import track_check          # same checker
import repro.norot_n7 as nr

G = 9.81; N = 7
A_MAX, V_MAX, TD_MAX = 25.0, 14.0, 12.0
TH_LO, TH_HI = -1.0, 5.9
SEED6 = "repro/pool_ctrb_n7/_neutral_seed.npz"
POOL = pathlib.Path("repro/pool_permode_n7"); POOL.mkdir(parents=True, exist_ok=True)
LOGD = pathlib.Path("runs_tmp/ipopt"); LOGD.mkdir(parents=True, exist_ok=True)
WIN_SPAN = (0.55, 0.95)
WIN_LEN_S = 1.5

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def mode_basis():
    """Fixed upright bending-mode basis, exactly as n7_ideas.permode."""
    chain = Chain(N, G)
    M0 = chain.mass_matrix(np.zeros(N))
    _, V = np.linalg.eig(np.linalg.solve(M0, np.diag(chain.b)))
    V = np.real(V)
    return V / np.linalg.norm(V, axis=0, keepdims=True)


def solve_pm(tag, T, c_floor, max_iter=4000):
    h = 0.01; K = int(round(T / h))
    A, b = chain_constants(N)
    Mfn, rfn = make_M_rhs_fn(N, G)
    Vb = ca.DM(mode_basis())
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
    opti.subject_to(opti.bounded(TH_LO, ca.vec(TH), TH_HI))     # no rotation
    # implicit input directions u on a node set covering BOTH the aggregate
    # floor region and the per-mode windows
    cn = list(range(max(1, int(0.05 * K)), int(0.95 * K), 3))
    U = opti.variable(N, len(cn))
    for j, k in enumerate(cn):
        opti.subject_to(ca.mtimes(mass_matrix_ca(TH[:, k], A), U[:, j]) == bend_rhs_ca(TH[:, k], A, b))
        cm = ca.sqrt(bend_excite_ca(TH[:, k], U[:, j], A) + 1e-9)
        short = ca.fmax(0.0, 0.5 - cm)                 # keep the aggregate floor
        cost += 100.0 * short * short
    # windowed per-mode floors over t/T in WIN_SPAN
    wlen = max(1, int(WIN_LEN_S / (h * 3)))            # nodes-per-window on the cn grid
    lo = int(WIN_SPAN[0] * K); hi = int(WIN_SPAN[1] * K)
    widx = [j for j, k in enumerate(cn) if lo <= k < hi]
    nw_pm = 0
    for wstart in range(0, len(widx) - wlen + 1, wlen):
        cols = [widx[i] for i in range(wstart, wstart + wlen)]
        for m in range(N):
            num = 0; den = 0
            for j in cols:
                pj = ca.mtimes(Vb[:, m].T, U[:, j])
                num = num + pj * pj
                den = den + ca.mtimes(U[:, j].T, U[:, j])
            # hinge: window-mean (v^T u)^2 >= c^2 * window-mean u^T u
            short = ca.fmax(0.0, c_floor * c_floor * den - num)
            cost += 50.0 * short * short
            nw_pm += 1
    # warm start: neutral N=6 lift, time-stretched
    seed6 = dict(np.load(SEED6))
    g = homotopy_guess(seed6); r = float(seed6["T"]) / T
    s = np.linspace(0, 1, K + 1); Kg = g["theta"].shape[0] - 1; sg = np.linspace(0, 1, Kg + 1)
    thg = np.clip(np.array([np.interp(s, sg, g["theta"][:, j]) for j in range(N)]),
                  TH_LO + 0.05, TH_HI - 0.05)
    opti.set_initial(TH, thg)
    opti.set_initial(TD, np.array([np.interp(s, sg, g["thetad"][:, j]) for j in range(N)]) * r)
    opti.set_initial(VV, (np.interp(s, sg, g["v"]) * r).reshape(1, -1))
    opti.set_initial(AC, (np.interp(s, sg, g["a"]) * r * r).reshape(1, -1))
    chain = Chain(N, G)
    opti.set_initial(U, np.column_stack([np.linalg.solve(chain.mass_matrix(thg[:, k]),
                                                         b * np.cos(thg[:, k])) for k in cn]))
    opti.minimize(cost)
    opti.solver("ipopt", {"expand": True},
                {"max_iter": max_iter, "print_level": 0, "sb": "yes", "tol": 1e-7,
                 "acceptable_tol": 1e-6, "mu_strategy": "adaptive",
                 "output_file": str(LOGD / f"pm_{tag}.log"), "file_print_level": 5})
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
    if status == "solved":
        np.savez(POOL / f"{tag}.npz", **{k: v for k, v in out.items() if k != "status"})
    return out


def w_case(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    T, c = args
    tag = f"T{int(T)}_c{c}"
    sol = solve_pm(tag, T, c)
    if sol["status"] != "solved":
        return f"{tag}: NLP failed/infeasible"
    import sys as _s; _s.path.insert(0, "repro")
    from n7_ideas import permode
    chain = Chain(N, G)
    t = sol["t"]; T_ = t[-1]
    m = (t >= 0.65 * T_)
    cworst = float(permode(chain, sol["theta"][m]).min())
    ok, ns, worst = track_check(sol)
    whip = float(np.abs(sol["thetad"]).max())
    return (f"{tag}: solved whip={whip:.1f} late-worst-mode={cworst:.3f} "
            f"track {ok}/{ns} (worst {np.degrees(worst):.2f}deg)")


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cases = [(T, c) for T in (25.0, 30.0) for c in (0.20, 0.27)]
    log(f"solving {len(cases)} per-mode no-rotation N=7 cases: {cases}")
    with mp.Pool(nw) as pool:
        for line in pool.imap_unordered(w_case, cases):
            log("  " + line)
    log("per-mode search done")


if __name__ == "__main__":
    main()
