#!/usr/bin/env python
"""TVLQR discretized against the ACTUAL simulator step (RK4-Jacobian).

The original pipeline (optimize_n6.build_tvlqr) discretizes via the matrix
exponential of the FD-linearized *continuous* model. The simulator applies
RK4 to the *nonlinear* dynamics; the two step maps differ by ~1% relative
mid-swing, which at 1e5 gains costs real stability margin (PROGRESS
2026-06-10: with RK4-Jacobian gains the N=7 reverse-fall tracks 16/16 at
dt=0.004, vs nothing above 0.002 with expm gains).

Here (Ad, Bd) at each node are finite differences of the rk4_step map itself,
so the Riccati recursion optimizes the gains for the exact discrete plant the
rollout uses. State z = [theta, thetad, v], input a (ZOH); v' = v + a*dt.
"""
from __future__ import annotations

import numpy as np

from pendulum.sim import rk4_step


def build_tvlqr_rk4(chain, theta, thetad, a_ff, dt, Q, R, QF, eps=1e-6):
    """Gain schedule K[k] (1, 2n+1) for the RK4/ZOH discrete plant."""
    n = chain.n; nz = 2 * n + 1
    M = len(theta) - 1
    ABs = []
    for k in range(M):
        y0 = np.concatenate([theta[k], thetad[k]])
        f0 = rk4_step(chain, y0, a_ff[k], dt)
        Ad = np.zeros((nz, nz)); Bd = np.zeros((nz, 1))
        for j in range(2 * n):
            yp = y0.copy(); yp[j] += eps
            Ad[:2 * n, j] = (rk4_step(chain, yp, a_ff[k], dt) - f0) / eps
        Ad[2 * n, 2 * n] = 1.0
        Bd[:2 * n, 0] = (rk4_step(chain, y0, a_ff[k] + eps, dt) - f0) / eps
        Bd[2 * n, 0] = dt
        ABs.append((Ad, Bd))
    P = QF.copy()
    Ks = np.zeros((M, 1, nz))
    for k in range(M - 1, -1, -1):
        Ad, Bd = ABs[k]
        S = R + Bd.T @ P @ Bd
        Kk = np.linalg.solve(S, Bd.T @ P @ Ad)
        Ks[k] = Kk
        P = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ Kk
    return Ks
