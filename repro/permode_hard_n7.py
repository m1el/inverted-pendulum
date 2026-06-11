#!/usr/bin/env python
"""Per-mode floor v2: HARD windowed constraints, warm from the feasible
no-rotation solution. Outcomes are decisive either way:

  EXIT solved      -> candidate classical N=7 with certified mode coupling;
                      track-check with RK4-Jacobian gains.
  EXIT infeasible  -> IPOPT restoration certificate: within the no-rotation
                      corridor (+ whip bound), NO trajectory holds every
                      bending mode's windowed coupling >= c. The winding
                      theorem in its publishable form.
  max_iter         -> inconclusive; iterate saved for coupling measurement.

Constraint per mode j, per ~1.5 s window W in t/T in [0.55, 0.95]:
    sum_{k in W} (v_j^T u_k)^2  >=  c^2 * sum_{k in W} u_k^T u_k
Sweep c in {0.15, 0.20, 0.27}; T = 25 (warm start repro/pool_norot_n7/T25_f0.5.npz).

Run:  uv run python repro/permode_hard_n7.py [n_workers]
IPOPT logs: runs_tmp/ipopt/pmh_*.log
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import casadi as ca
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.trajopt import chain_constants, make_M_rhs_fn
from repro.generate_n6 import bend_rhs_ca, mass_matrix_ca
from repro.permode_norot_n7 import mode_basis
from repro.rk4_tvlqr import build_tvlqr_rk4
from pendulum.sim import rk4_step

G = 9.81; N = 7
A_MAX, V_MAX, TD_MAX = 25.0, 14.0, 12.0
TH_LO, TH_HI = -1.0, 5.9
WARM = "repro/pool_norot_n7/T25_f0.5.npz"
POOL = pathlib.Path("repro/pool_permode_n7"); POOL.mkdir(parents=True, exist_ok=True)
LOGD = pathlib.Path("runs_tmp/ipopt"); LOGD.mkdir(parents=True, exist_ok=True)
WIN_SPAN = (0.55, 0.95); WIN_LEN_S = 1.5

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def solve_hard(c_floor, max_iter=6000):
    w = dict(np.load(WARM)); T = float(w["T"])
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
    opti.subject_to(opti.bounded(TH_LO, ca.vec(TH), TH_HI))
    cn = list(range(int(WIN_SPAN[0] * K), int(WIN_SPAN[1] * K), 3))
    U = opti.variable(N, len(cn))
    for j, k in enumerate(cn):
        opti.subject_to(ca.mtimes(mass_matrix_ca(TH[:, k], A), U[:, j]) == bend_rhs_ca(TH[:, k], A, b))
    wlen = max(1, int(WIN_LEN_S / (h * 3)))
    nconstr = 0
    for wstart in range(0, len(cn) - wlen + 1, wlen):
        cols = list(range(wstart, wstart + wlen))
        den = sum(ca.mtimes(U[:, j].T, U[:, j]) for j in cols)
        for m in range(N):
            num = sum(ca.mtimes(Vb[:, m].T, U[:, j]) ** 2 for j in cols)
            opti.subject_to(num >= c_floor * c_floor * den)      # HARD
            nconstr += 1
    opti.minimize(cost)
    # warm start from the feasible plain no-rotation solution
    s = np.linspace(0, 1, K + 1); sg = np.linspace(0, 1, w["theta"].shape[0] - 1 + 1)
    thg = np.array([np.interp(s, sg, w["theta"][:, j]) for j in range(N)])
    opti.set_initial(TH, thg)
    opti.set_initial(TD, np.array([np.interp(s, sg, w["thetad"][:, j]) for j in range(N)]))
    opti.set_initial(VV, np.interp(s, sg, w["v"]).reshape(1, -1))
    opti.set_initial(AC, np.interp(s, sg, w["a"]).reshape(1, -1))
    chain = Chain(N, G)
    opti.set_initial(U, np.column_stack([np.linalg.solve(chain.mass_matrix(thg[:, k]),
                                                         b * np.cos(thg[:, k])) for k in cn]))
    opti.solver("ipopt", {"expand": True},
                {"max_iter": max_iter, "print_level": 0, "sb": "yes", "tol": 1e-7,
                 "acceptable_tol": 1e-6, "mu_strategy": "adaptive",
                 "output_file": str(LOGD / f"pmh_c{c_floor}.log"), "file_print_level": 5})
    status = "solved"
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug
        # classify from the log: restoration failure / infeasible vs max_iter
        txt = (LOGD / f"pmh_c{c_floor}.log").read_text()
        if "Converged to a point of local infeasibility" in txt or "Restoration failed" in txt:
            status = "INFEASIBLE"
        elif "Maximum Number of Iterations" in txt:
            status = "max_iter"
        else:
            status = "failed"
    out = {"t": np.linspace(0, T, K + 1), "theta": np.array(sol.value(TH)).T,
           "thetad": np.array(sol.value(TD)).T, "v": np.array(sol.value(VV)).ravel(),
           "a": np.array(sol.value(AC)).ravel(), "T": T}
    np.savez(POOL / f"hard_c{c_floor}.npz", **out)     # save REGARDLESS of status
    return status, out, nconstr


def w_case(c):
    os.environ["OMP_NUM_THREADS"] = "1"
    status, sol, nc = solve_hard(c)
    # measure achieved late-window worst-mode coupling regardless of status
    sys.path.insert(0, "repro")
    from n7_ideas import permode
    chain = Chain(N, G)
    t = sol["t"]; T = t[-1]; m = (t >= 0.65 * T)
    cworst = float(permode(chain, sol["theta"][m]).min())
    line = f"c={c}: {status} ({nc} hard constraints) achieved late worst-mode={cworst:.3f}"
    if status == "solved":
        dt = 0.002
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
        line += f"  TRACK {oks}/4 (RK4-Jac, dt=0.002)"
    return line


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace(".", "").isdigit() and "." not in sys.argv[1] else 3
    cs = tuple(float(a) for a in sys.argv[2:]) or (0.15, 0.20, 0.27)
    log(f"hard per-mode floors c={cs}, warm from {WARM}")
    with mp.Pool(nw) as pool:
        for line in pool.imap_unordered(w_case, cs):
            log("  " + line)
    log("hard per-mode search done")


if __name__ == "__main__":
    main()
