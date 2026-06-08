"""N=6 swing-up trajectory that AVOIDS the near-uncontrollable straight-chain
manifold by penalizing inter-link alignment during the mid-swing.

Why: the plain minimum-effort N=6 nominal passes through a near-straight
(all links ~aligned) configuration mid-swing where the chain is near-
uncontrollable from the single pivot input -> closed-loop diverges there
regardless of controller. Keeping the chain curved (adjacent links spread)
keeps the bending modes controllable.

Cost adds, over the middle fraction of the horizon, +w_align * sum_i
cos(theta_i - theta_{i+1}); cos is large (+1) when links align, so minimizing
it pushes the chain toward curved configurations. The penalty is ramped to
zero near the endpoints so boundary conditions (hang / upright) are unaffected.

argv: T K seed [w_align] [a_max] [v_max]
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import casadi as ca

from pendulum.dynamics import Chain
from pendulum.trajopt import make_M_rhs_fn, chain_constants, homotopy_guess

G = 9.81


def solve(n, T, K, seed=0, w_align=2.0, a_max=25.0, v_max=14.0,
          settle_frac=0.08, settle_band=0.15, max_iter=6000):
    h = T / K
    Mfn, rfn = make_M_rhs_fn(n, G)
    opti = ca.Opti()
    TH = opti.variable(n, K + 1)
    TD = opti.variable(n, K + 1)
    VV = opti.variable(1, K + 1)
    TDD = opti.variable(n, K + 1)
    AC = opti.variable(1, K + 1)

    cost = 0
    s = np.linspace(0, 1, K + 1)
    # alignment-penalty window: 0 at ends, 1 in the middle (sin^2 bump)
    win = np.sin(np.pi * s) ** 2

    for k in range(K + 1):
        opti.subject_to(ca.mtimes(Mfn(TH[:, k]), TDD[:, k])
                        == rfn(TH[:, k], TD[:, k], AC[0, k]))
    for k in range(K):
        opti.subject_to(TH[:, k + 1] - TH[:, k] == 0.5 * h * (TD[:, k] + TD[:, k + 1]))
        opti.subject_to(TD[:, k + 1] - TD[:, k] == 0.5 * h * (TDD[:, k] + TDD[:, k + 1]))
        opti.subject_to(VV[0, k + 1] - VV[0, k] == 0.5 * h * (AC[0, k] + AC[0, k + 1]))
        cost += 0.5 * h * (AC[0, k] ** 2 + AC[0, k + 1] ** 2)
        cost += 1e-3 * (AC[0, k + 1] - AC[0, k]) ** 2

    # anti-alignment penalty over the middle window
    for k in range(K + 1):
        align = 0
        for i in range(n - 1):
            align = align + ca.cos(TH[i, k] - TH[i + 1, k])
        cost += w_align * win[k] * align

    opti.subject_to(TH[:, 0] == np.pi)
    opti.subject_to(TD[:, 0] == 0)
    opti.subject_to(VV[0, 0] == 0)
    opti.subject_to(TH[:, K] == 0)
    opti.subject_to(TD[:, K] == 0)
    opti.subject_to(VV[0, K] == 0)
    opti.subject_to(opti.bounded(-a_max, ca.vec(AC), a_max))
    opti.subject_to(opti.bounded(-v_max, VV, v_max))
    kstart = int(round((1 - settle_frac) * K))
    for k in range(kstart, K + 1):
        opti.subject_to(opti.bounded(-settle_band, TH[:, k], settle_band))

    opti.minimize(cost)

    # warm start from N=5 homotopy if available, else seeded ramp
    rng = np.random.default_rng(seed)
    try:
        g5 = homotopy_guess(dict(np.load("results/trajectories/swingup_N5.npz")))
        Kg = g5["theta"].shape[0] - 1
        sg = np.linspace(0, 1, Kg + 1)
        opti.set_initial(TH, np.array([np.interp(s, sg, g5["theta"][:, j]) for j in range(n)]))
        opti.set_initial(TD, np.array([np.interp(s, sg, g5["thetad"][:, j]) for j in range(n)]))
        opti.set_initial(VV, np.interp(s, sg, g5["v"]).reshape(1, -1))
        opti.set_initial(AC, np.interp(s, sg, g5["a"]).reshape(1, -1))
    except Exception:
        opti.set_initial(TH, np.outer(np.full(n, np.pi), (1 - s)))

    opti.solver("ipopt", {"expand": True},
                {"max_iter": max_iter, "print_level": 3, "sb": "yes",
                 "tol": 1e-7, "acceptable_tol": 1e-5, "mu_strategy": "adaptive"})
    try:
        sol = opti.solve(); status = "solved"
    except RuntimeError:
        print("FAILED"); return None
    th = np.array(sol.value(TH)).T; td = np.array(sol.value(TD)).T
    a = np.array(sol.value(AC)).ravel(); v = np.array(sol.value(VV)).ravel()
    ch = Chain(n, G)
    mtdd = max(np.abs(ch.thetadd(th[k], td[k], a[k])).max() for k in range(len(th)))
    # min controllability ratio along trajectory
    def ctrb(k):
        Ac, Bc = ch.linearize_at(th[k]); nz = Ac.shape[0]
        C = Bc.copy(); M = Bc.copy()
        for _ in range(nz - 1):
            M = Ac @ M; C = np.hstack([C, M])
        sv = np.linalg.svd(C, compute_uv=False); return sv.min() / sv.max()
    minctrb = min(ctrb(k) for k in range(0, len(th), 5))
    print(f"T={T} w={w_align} max|thd|={np.abs(td).max():.1f} max|a|={np.abs(a).max():.1f} "
          f"max|thdd|={mtdd:.0f} min_ctrb={minctrb:.1e}")
    out = f"results/trajectories_high/swingup_N6_curved_T{T}_w{w_align}_s{seed}.npz"
    np.savez(out, t=np.linspace(0, T, K + 1), theta=th, thetad=td, a=a, v=v,
             T=T, target=np.zeros(n))
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    T = float(sys.argv[1]); K = int(sys.argv[2]); seed = int(sys.argv[3])
    w = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    solve(6, T, K, seed, w_align=w)
