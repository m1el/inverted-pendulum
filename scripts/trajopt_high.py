"""Swing-up trajectory optimization for high N (6,7) via casadi direct
collocation (trapezoidal). Self-contained (independent of agent-C files).

Usage: uv run python scripts/trajopt_high.py N T seed [amax] [vmax]
Writes results/trajectories_high/swingup_N{N}_T{T}_s{seed}.npz on success.
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import casadi as ca

from pendulum.dynamics import Chain

G = 9.81


def casadi_thetadd(n, theta, thetad, xdd):
    """Closed-form dynamics in casadi symbols (mirrors Chain.thetadd)."""
    i = np.arange(1, n + 1)
    A = (n - np.maximum.outer(i, i) + 0.5)
    np.fill_diagonal(A, n - i + 0.25)
    b = ca.DM(n - i + 0.5)
    Adm = ca.DM(A)
    dth = ca.repmat(theta, 1, n) - ca.repmat(theta.T, n, 1)
    M = Adm * ca.cos(dth) + ca.DM(np.eye(n)) / 12.0
    rhs = (-xdd) * b * ca.cos(theta) \
        - ca.mtimes(Adm * ca.sin(dth), thetad**2) \
        + G * b * ca.sin(theta)
    return ca.solve(M, rhs)


def crosscheck(n, tol=1e-9):
    ch = Chain(n, G)
    th = ca.MX.sym("th", n, 1)
    thd = ca.MX.sym("thd", n, 1)
    a = ca.MX.sym("a")
    f = ca.Function("f", [th, thd, a], [casadi_thetadd(n, th, thd, a)])
    rng = np.random.default_rng(42)
    for _ in range(20):
        t0 = rng.uniform(-np.pi, np.pi, n)
        t1 = rng.uniform(-3, 3, n)
        xdd = rng.uniform(-20, 20)
        err = np.max(np.abs(np.array(f(t0, t1, xdd)).ravel() - ch.thetadd(t0, t1, xdd)))
        assert err < tol, f"casadi/numpy mismatch N={n}: {err}"
    print(f"N={n}: casadi dynamics cross-check OK")


def solve(n, T, seed, amax=40.0, vmax=15.0, h=0.02, max_iter=4000):
    K = int(round(T / h))
    opti = ca.Opti()
    TH = opti.variable(n, K + 1)
    THD = opti.variable(n, K + 1)
    V = opti.variable(1, K + 1)
    Acc = opti.variable(1, K)

    th = ca.MX.sym("th", n, 1)
    thd = ca.MX.sym("thd", n, 1)
    aa = ca.MX.sym("aa")
    fdyn = ca.Function("fdyn", [th, thd, aa], [casadi_thetadd(n, th, thd, aa)])

    for k in range(K):
        tdd0 = fdyn(TH[:, k], THD[:, k], Acc[0, k])
        tdd1 = fdyn(TH[:, k + 1], THD[:, k + 1], Acc[0, k])
        # trapezoidal collocation
        opti.subject_to(TH[:, k + 1] == TH[:, k] + 0.5 * h * (THD[:, k] + THD[:, k + 1]))
        opti.subject_to(THD[:, k + 1] == THD[:, k] + 0.5 * h * (tdd0 + tdd1))
        opti.subject_to(V[0, k + 1] == V[0, k] + h * Acc[0, k])

    opti.subject_to(TH[:, 0] == np.pi)
    opti.subject_to(THD[:, 0] == 0)
    opti.subject_to(V[0, 0] == 0)
    opti.subject_to(TH[:, K] == 0)
    opti.subject_to(THD[:, K] == 0)
    opti.subject_to(V[0, K] == 0)
    opti.subject_to(opti.bounded(-amax, Acc, amax))
    opti.subject_to(opti.bounded(-vmax, V, vmax))
    opti.subject_to(opti.bounded(-4 * np.pi, TH, 4 * np.pi))

    J = ca.sumsqr(Acc) * h + 0.1 * ca.sumsqr(THD) * h
    opti.minimize(J)

    # initial guess: interpolate pi -> 0 with seeded sinusoidal wiggles
    rng = np.random.default_rng(seed)
    s = np.linspace(0, 1, K + 1)
    th0 = np.pi * (1 - s)[None, :] * np.ones((n, 1))
    for i in range(n):
        nw = rng.integers(1, 4)
        for w in range(nw):
            th0[i] += rng.uniform(-1.5, 1.5) * np.sin(np.pi * (w + 1) * s)
    opti.set_initial(TH, th0)
    opti.set_initial(THD, np.gradient(th0, h, axis=1))

    opti.solver("ipopt", {"print_time": False},
                {"max_iter": max_iter, "print_level": 3, "tol": 1e-8,
                 "acceptable_tol": 1e-6, "mu_strategy": "adaptive"})
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"N={n} T={T} seed={seed}: FAILED ({e})")
        return None
    t = np.arange(K + 1) * h
    a = np.array(sol.value(Acc)).ravel()
    v = np.array(sol.value(V)).ravel()
    out = dict(
        t=t, theta=np.array(sol.value(TH)).T, thetad=np.array(sol.value(THD)).T,
        a=a, v=v, x=np.concatenate([[0], np.cumsum(0.5 * (v[1:] + v[:-1]) * h)]),
        J=float(sol.value(J)), amax=amax, vmax=vmax, h=h, T=T, n=n, seed=seed,
    )
    pathlib.Path("results/trajectories_high").mkdir(parents=True, exist_ok=True)
    fn = f"results/trajectories_high/swingup_N{n}_T{T}_s{seed}.npz"
    np.savez(fn, **out)
    print(f"N={n} T={T} seed={seed}: J={out['J']:.2f} saved {fn}")
    return fn


if __name__ == "__main__":
    n, T, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    amax = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
    vmax = float(sys.argv[5]) if len(sys.argv) > 5 else 15.0
    crosscheck(n)
    solve(n, T, seed, amax, vmax)
