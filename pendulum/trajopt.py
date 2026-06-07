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
            opti.subject_to(opti.bounded(theta_target - 0.15, th[:, k], theta_target + 0.15))
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
        "tol": 1e-8,
        "acceptable_tol": 1e-6,
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


if __name__ == "__main__":
    for n in range(1, 6):
        err = cross_check(n)
        print(f"N={n}: max thetadd err vs numpy = {err:.2e}")
