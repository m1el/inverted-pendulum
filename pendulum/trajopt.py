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
    # dth[i,l] = theta_i - theta_l
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
    A, b = chain_constants(n)
    z = ca.MX.sym("z", 2 * n + 1)
    a = ca.MX.sym("a")
    zdot = f_state(z, a, n, A, b, g)
    return ca.Function("f", [z, a], [zdot])


# ---------------------------------------------------------------------------
def cross_check(n, g=9.81, ntests=20, seed=0):
    """Compare casadi thetadd vs numpy Chain.thetadd at random states."""
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


if __name__ == "__main__":
    for n in range(1, 6):
        err = cross_check(n)
        print(f"N={n}: max thetadd err vs numpy = {err:.2e}")
