"""CasADi reimplementation of the chain dynamics + direct-collocation swing-up.

The continuous dynamics (matching pendulum.dynamics.Chain.thetadd):

    M(theta) thetadd = -xdd*b*cos(theta) - (A*sin(theta_i-theta_l)) @ thetad**2
                       + g*b*sin(theta)

    A_il = N - max(i,l) + 1/2 (i!=l), A_ii = N - i + 1/4, b_i = N - i + 1/2
    M_il = A_il*cos(theta_i-theta_l) + delta_il/12

State for trajopt: z = [theta(n), thetad(n), v]  (v = pivot velocity).
Control: a = pivot acceleration (xdd).  v' = a.
"""

from __future__ import annotations

import numpy as np
import casadi as ca


def chain_constants(n):
    i = np.arange(1, n + 1)
    mx = np.maximum.outer(i, i)
    A = (n - mx + 0.5).astype(float)
    np.fill_diagonal(A, n - i + 0.25)
    b = (n - i + 0.5).astype(float)
    return A, b


def thetadd_ca(theta, thetad, xdd, A, b, g):
    """Symbolic/numeric angular accelerations (CasADi MX/DM compatible).

    theta, thetad: (n,1) column vectors. xdd: scalar. Returns (n,1).
    """
    n = A.shape[0]
    Aca = ca.DM(A)
    bca = ca.DM(b.reshape(-1, 1))
    dth = ca.repmat(theta, 1, n) - ca.repmat(theta.T, n, 1)
    M = Aca * ca.cos(dth) + ca.DM(np.eye(n) / 12.0)
    rhs = (
        -xdd * bca * ca.cos(theta)
        - ca.mtimes(Aca * ca.sin(dth), thetad ** 2)
        + g * bca * ca.sin(theta)
    )
    return ca.solve(M, rhs)


def f_state(z, a, n, A, b, g):
    """Continuous state derivative for z=[theta,thetad,v], control a."""
    theta = z[0:n]
    thetad = z[n:2 * n]
    tdd = thetadd_ca(theta, thetad, a, A, b, g)
    return ca.vertcat(thetad, tdd, a)


def make_dynamics_fn(n, g=9.81):
    """Explicit ODE as an SX Function (expandable -> fast IPOPT Hessians)."""
    A, b = chain_constants(n)
    z = ca.SX.sym("z", 2 * n + 1)
    a = ca.SX.sym("a")
    zdot = f_state(z, a, n, A, b, g)
    return ca.Function("f", [z, a], [zdot])


def make_M_rhs_fn(n, g=9.81):
    """SX functions for the mass matrix M(theta) and rhs(theta,thetad,a).

    Used by the *implicit* collocation, which keeps thetadd as a decision
    variable and enforces M@thetadd == rhs as an algebraic constraint. This
    avoids the symbolic matrix inverse (ca.solve), whose expression blows up
    combinatorially for n>=4 and makes IPOPT Hessians extremely slow.
    """
    A, b = chain_constants(n)
    Aca = ca.DM(A); bca = ca.DM(b.reshape(-1, 1))
    theta = ca.SX.sym("theta", n)
    thetad = ca.SX.sym("thetad", n)
    a = ca.SX.sym("a")
    dth = ca.repmat(theta, 1, n) - ca.repmat(theta.T, n, 1)
    M = Aca * ca.cos(dth) + ca.DM(np.eye(n) / 12.0)
    rhs = (-a * bca * ca.cos(theta)
           - ca.mtimes(Aca * ca.sin(dth), thetad ** 2)
           + g * bca * ca.sin(theta))
    Mfn = ca.Function("M", [theta], [M])
    rfn = ca.Function("rhs", [theta, thetad, a], [rhs])
    return Mfn, rfn


# ---------------------------------------------------------------------------
def cross_check(n, g=9.81, ntests=20, seed=0):
    from .dynamics import Chain

    chain = Chain(n, g)
    A, b = chain_constants(n)
    rng = np.random.default_rng(seed)
    fn = make_dynamics_fn(n, g)
    max_err = 0.0
    for _ in range(ntests):
        theta = rng.uniform(-np.pi, np.pi, n)
        thetad = rng.uniform(-3, 3, n)
        a = rng.uniform(-10, 10)
        ref = chain.thetadd(theta, thetad, a)
        z = np.concatenate([theta, thetad, [0.0]])
        out = np.array(fn(z, a)).ravel()
        tdd = out[n:2 * n]
        max_err = max(max_err, np.max(np.abs(tdd - ref)))
    return max_err


# ---------------------------------------------------------------------------
def solve_swingup(
    n,
    T,
    K,
    g=9.81,
    a_max=40.0,
    v_max=12.0,
    theta_target=None,
    z_init=None,
    settle_frac=0.0,
    w_a=1.0,
    w_smooth=1e-3,
    w_term=0.0,
    seed=0,
    init_guess=None,
    max_iter=3000,
    print_level=0,
    settle_band=0.15,
    tol=1e-8,
):
    """Hermite-Simpson direct collocation swing-up.

    n: links. T: horizon (s). K: number of intervals (K+1 nodes).
    theta_target: length-n array, each a multiple of 2pi (target angle per link).
                  default zeros (upright).
    settle_frac: fraction of horizon at the end forced to stay near upright
                 (tighter terminal accuracy).
    init_guess: dict with 'theta','thetad','a','v' arrays of length K+1 (a:K+1)
                used as initial guess (resampled if needed).
    Returns dict: t, theta(K+1,n), thetad(K+1,n), a(K+1), v(K+1), cost, status.
    """
    A, b = chain_constants(n)
    if theta_target is None:
        theta_target = np.zeros(n)
    theta_target = np.asarray(theta_target, float)
    nz = 2 * n + 1
    h = T / K

    f = make_dynamics_fn(n, g)

    opti = ca.Opti()
    Z = opti.variable(nz, K + 1)       # states at nodes
    Ac = opti.variable(1, K + 1)        # control at nodes
    th = Z[0:n, :]
    tdv = Z[n:2 * n, :]
    vv = Z[2 * n, :]

    # Hermite-Simpson collocation
    cost = 0
    for k in range(K):
        zk = Z[:, k]
        zk1 = Z[:, k + 1]
        ak = Ac[0, k]
        ak1 = Ac[0, k + 1]
        amid = 0.5 * (ak + ak1)
        fk = f(zk, ak)
        fk1 = f(zk1, ak1)
        zmid = 0.5 * (zk + zk1) + (h / 8.0) * (fk - fk1)
        fmid = f(zmid, amid)
        # Simpson defect
        opti.subject_to(zk1 - zk == (h / 6.0) * (fk + 4 * fmid + fk1))
        cost += (h / 6.0) * (ak ** 2 + 4 * amid ** 2 + ak1 ** 2) * w_a
    # smoothness on control
    for k in range(K):
        cost += w_smooth * (Ac[0, k + 1] - Ac[0, k]) ** 2

    # boundary conditions
    z0 = np.zeros(nz)
    if z_init is None:
        z0[0:n] = np.pi
    else:
        z0[0:n] = z_init
    opti.subject_to(th[:, 0] == z0[0:n])
    opti.subject_to(tdv[:, 0] == 0)
    opti.subject_to(vv[0] == 0)
    # terminal
    opti.subject_to(th[:, K] == theta_target)
    opti.subject_to(tdv[:, K] == 0)
    opti.subject_to(vv[K] == 0)

    # path constraints
    opti.subject_to(opti.bounded(-a_max, ca.vec(Ac), a_max))
    opti.subject_to(opti.bounded(-v_max, vv, v_max))

    # settle segment: keep near target at the tail
    if settle_frac > 0:
        kstart = int(round((1 - settle_frac) * K))
        for k in range(kstart, K + 1):
            opti.subject_to(opti.bounded(theta_target - settle_band, th[:, k], theta_target + settle_band))
        cost += w_term * ca.sumsqr(th[:, K] - theta_target.reshape(-1, 1))

    opti.minimize(cost)

    # initial guess
    rng = np.random.default_rng(seed)
    if init_guess is not None:
        Kg = init_guess["theta"].shape[0] - 1
        sg = np.linspace(0, 1, Kg + 1)
        sn = np.linspace(0, 1, K + 1)
        thg = np.array([np.interp(sn, sg, init_guess["theta"][:, j]) for j in range(n)])
        tdg = np.array([np.interp(sn, sg, init_guess["thetad"][:, j]) for j in range(n)])
        vg = np.interp(sn, sg, init_guess["v"])
        ag = np.interp(sn, sg, init_guess["a"])
        opti.set_initial(th, thg)
        opti.set_initial(tdv, tdg)
        opti.set_initial(vv, vg)
        opti.set_initial(Ac, ag.reshape(1, -1))
    else:
        # heuristic: interpolate theta from pi to target, add noise
        s = np.linspace(0, 1, K + 1)
        thg = np.outer(z0[0:n], (1 - s)) + np.outer(theta_target, s)
        thg += rng.uniform(-0.5, 0.5, thg.shape) * np.sin(np.pi * s)
        opti.set_initial(th, thg)
        opti.set_initial(tdv, rng.uniform(-1, 1, (n, K + 1)))
        opti.set_initial(vv, rng.uniform(-2, 2, K + 1))
        opti.set_initial(Ac, rng.uniform(-5, 5, (1, K + 1)))

    p_opts = {}
    s_opts = {
        "max_iter": max_iter,
        "print_level": print_level,
        "sb": "yes",
        "tol": tol,
        "acceptable_tol": max(tol * 100, 1e-6),
        "mu_strategy": "adaptive",
    }
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()
        status = "solved"
    except RuntimeError:
        sol = opti.debug
        status = "failed"

    Zs = np.array(sol.value(Z))
    Acs = np.array(sol.value(Ac)).ravel()
    cost_v = float(sol.value(cost))
    t = np.linspace(0, T, K + 1)
    return {
        "t": t,
        "theta": Zs[0:n, :].T,
        "thetad": Zs[n:2 * n, :].T,
        "v": Zs[2 * n, :],
        "a": Acs,
        "cost": cost_v,
        "status": status,
        "T": T,
        "K": K,
        "n": n,
    }


def solve_swingup_implicit(
    n, T, K, g=9.81, a_max=40.0, v_max=12.0, theta_target=None,
    settle_frac=0.0, settle_band=0.2, w_a=1.0, w_smooth=1e-3,
    seed=0, init_guess=None, max_iter=3000, print_level=0, tol=1e-7,
):
    """Trapezoidal direct collocation with IMPLICIT dynamics.

    Decision vars per node: theta(n), thetad(n), v, tdd(n), a.
    Constraints: M(theta_k) tdd_k = rhs(theta_k, thetad_k, a_k)   (dynamics)
                 trapezoidal defects on (theta, thetad, v).
    Much faster than the explicit form for n>=4.
    """
    A, b = chain_constants(n)
    if theta_target is None:
        theta_target = np.zeros(n)
    theta_target = np.asarray(theta_target, float)
    h = T / K
    Mfn, rfn = make_M_rhs_fn(n, g)

    opti = ca.Opti()
    TH = opti.variable(n, K + 1)
    TD = opti.variable(n, K + 1)
    VV = opti.variable(1, K + 1)
    TDD = opti.variable(n, K + 1)
    AC = opti.variable(1, K + 1)

    cost = 0
    # node dynamics (implicit) + trapezoidal defects
    for k in range(K + 1):
        opti.subject_to(ca.mtimes(Mfn(TH[:, k]), TDD[:, k])
                        == rfn(TH[:, k], TD[:, k], AC[0, k]))
    for k in range(K):
        opti.subject_to(TH[:, k + 1] - TH[:, k]
                        == 0.5 * h * (TD[:, k] + TD[:, k + 1]))
        opti.subject_to(TD[:, k + 1] - TD[:, k]
                        == 0.5 * h * (TDD[:, k] + TDD[:, k + 1]))
        opti.subject_to(VV[0, k + 1] - VV[0, k]
                        == 0.5 * h * (AC[0, k] + AC[0, k + 1]))
        cost += 0.5 * h * (AC[0, k] ** 2 + AC[0, k + 1] ** 2) * w_a
        cost += w_smooth * (AC[0, k + 1] - AC[0, k]) ** 2

    # boundary
    opti.subject_to(TH[:, 0] == np.pi)
    opti.subject_to(TD[:, 0] == 0)
    opti.subject_to(VV[0, 0] == 0)
    opti.subject_to(TH[:, K] == theta_target)
    opti.subject_to(TD[:, K] == 0)
    opti.subject_to(VV[0, K] == 0)

    opti.subject_to(opti.bounded(-a_max, ca.vec(AC), a_max))
    opti.subject_to(opti.bounded(-v_max, VV, v_max))
    if settle_frac > 0:
        kstart = int(round((1 - settle_frac) * K))
        for k in range(kstart, K + 1):
            opti.subject_to(opti.bounded(theta_target - settle_band,
                                         TH[:, k], theta_target + settle_band))

    opti.minimize(cost)

    rng = np.random.default_rng(seed)
    s = np.linspace(0, 1, K + 1)
    if init_guess is not None:
        Kg = init_guess["theta"].shape[0] - 1
        sg = np.linspace(0, 1, Kg + 1)
        thg = np.array([np.interp(s, sg, init_guess["theta"][:, j]) for j in range(n)])
        tdg = np.array([np.interp(s, sg, init_guess["thetad"][:, j]) for j in range(n)])
        vg = np.interp(s, sg, init_guess["v"])
        ag = np.interp(s, sg, init_guess["a"])
        opti.set_initial(TH, thg)
        opti.set_initial(TD, tdg)
        opti.set_initial(VV, vg.reshape(1, -1))
        opti.set_initial(AC, ag.reshape(1, -1))
    else:
        thg = np.outer(np.full(n, np.pi), (1 - s)) + np.outer(theta_target, s)
        thg += rng.uniform(-0.5, 0.5, thg.shape) * np.sin(np.pi * s)
        opti.set_initial(TH, thg)
        opti.set_initial(TD, rng.uniform(-1, 1, (n, K + 1)))
        opti.set_initial(VV, rng.uniform(-2, 2, (1, K + 1)))
        opti.set_initial(AC, rng.uniform(-5, 5, (1, K + 1)))

    s_opts = {"max_iter": max_iter, "print_level": print_level, "sb": "yes",
              "tol": tol, "acceptable_tol": max(tol * 100, 1e-6),
              "mu_strategy": "adaptive"}
    opti.solver("ipopt", {"expand": True}, s_opts)
    try:
        sol = opti.solve(); status = "solved"
    except RuntimeError:
        sol = opti.debug; status = "failed"

    try:
        THv = np.array(sol.value(TH)); TDv = np.array(sol.value(TD))
        VVv = np.array(sol.value(VV)).ravel(); ACv = np.array(sol.value(AC)).ravel()
        cost_v = float(sol.value(cost))
    except RuntimeError:
        return {"status": "failed", "T": T, "K": K, "n": n}
    return {"t": np.linspace(0, T, K + 1), "theta": THv.T, "thetad": TDv.T,
            "v": VVv, "a": ACv, "cost": cost_v, "status": status,
            "T": T, "K": K, "n": n}


def homotopy_guess(sol_lower):
    """Build an initial guess for N links from an (N-1)-link solution by
    appending a copy of the last link's angle/rate (the chain tip), keeping the
    same time grid, control and pivot velocity. The new link starts at pi."""
    th = sol_lower["theta"]; td = sol_lower["thetad"]
    Kp1 = th.shape[0]
    new_th = np.hstack([th, th[:, -1:].copy()])
    new_td = np.hstack([td, td[:, -1:].copy()])
    # ensure new link's start is exactly pi (boundary will enforce anyway)
    new_th[0, -1] = np.pi
    return {"theta": new_th, "thetad": new_td, "a": sol_lower["a"],
            "v": sol_lower["v"]}


if __name__ == "__main__":
    for n in range(1, 6):
        err = cross_check(n)
        print(f"N={n}: max thetadd err vs numpy = {err:.2e}")
