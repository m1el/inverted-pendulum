"""Verify closed-form dynamics against an independent sympy Lagrangian
derivation, and check energy conservation under free swing."""

import numpy as np
import sympy as sp

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pendulum.dynamics import Chain
from pendulum.sim import rk4_step


def sympy_thetadd(n, g, theta_v, thetad_v, xdd_v):
    """Independent derivation via sympy Lagrangian mechanics."""
    t = sp.symbols("t")
    x = sp.Function("x")(t)
    th = [sp.Function(f"th{i}")(t) for i in range(n)]
    L_rod, m = 1, 1
    # joint positions
    px, py = x, sp.Integer(0)
    T, V = sp.Integer(0), sp.Integer(0)
    for i in range(n):
        cx = px + sp.Rational(1, 2) * L_rod * sp.sin(th[i])
        cy = py + sp.Rational(1, 2) * L_rod * sp.cos(th[i])
        vx, vy = sp.diff(cx, t), sp.diff(cy, t)
        T += sp.Rational(1, 2) * m * (vx**2 + vy**2)
        T += sp.Rational(1, 2) * (m * L_rod**2 / 12) * sp.diff(th[i], t) ** 2
        V += m * g * cy
        px = px + L_rod * sp.sin(th[i])
        py = py + L_rod * sp.cos(th[i])
    Lag = T - V
    eqs = [
        sp.expand(sp.diff(sp.diff(Lag, sp.diff(th[i], t)), t) - sp.diff(Lag, th[i]))
        for i in range(n)
    ]
    thdd = [sp.diff(th[i], t, 2) for i in range(n)]
    # extract linear system M thdd = rhs symbolically, evaluate numerically
    Msym, rhssym = sp.linear_eq_to_matrix(eqs, thdd)
    subs = {sp.diff(x, t, 2): xdd_v}
    for i in range(n):
        subs[sp.diff(th[i], t)] = thetad_v[i]
    for i in range(n):
        subs[th[i]] = theta_v[i]
    Mn = np.array(Msym.subs(subs).evalf().tolist(), dtype=float)
    rn = np.array(rhssym.subs(subs).evalf().tolist(), dtype=float).ravel()
    return np.linalg.solve(Mn, rn)


def test_vs_sympy():
    rng = np.random.default_rng(0)
    for n in range(1, 6):
        chain = Chain(n, g=9.81)
        for _ in range(3):
            theta = rng.uniform(-np.pi, np.pi, n)
            thetad = rng.uniform(-2, 2, n)
            xdd = rng.uniform(-5, 5)
            ours = chain.thetadd(theta, thetad, xdd)
            ref = sympy_thetadd(n, 9.81, theta, thetad, xdd)
            err = np.max(np.abs(ours - ref))
            assert err < 1e-8, f"N={n}: max err {err}"
        print(f"N={n}: matches sympy")


def test_energy_conservation():
    rng = np.random.default_rng(1)
    for n in range(1, 6):
        chain = Chain(n, g=9.81)
        theta = rng.uniform(-np.pi, np.pi, n)
        thetad = rng.uniform(-1, 1, n)
        y = np.concatenate([theta, thetad])
        e0 = chain.energy(y[:n], y[n:])
        h = 1e-4
        for _ in range(int(10.0 / h)):
            y = rk4_step(chain, y, 0.0, h)
        e1 = chain.energy(y[:n], y[n:])
        drift = abs(e1 - e0)
        assert drift < 1e-6, f"N={n}: energy drift {drift}"
        print(f"N={n}: energy drift over 10s = {drift:.2e}")


if __name__ == "__main__":
    test_vs_sympy()
    test_energy_conservation()
    print("all dynamics tests passed")
