"""Closed-form dynamics of an N-link pendulum whose pivot moves horizontally.

Conventions
-----------
- N identical uniform rods: mass m = 1, length L = 1, I_com = 1/12.
- Pivot (free end of rod 1) at (x, 0); only horizontal motion, x(t) prescribed.
- theta_i is the angle of rod i measured from the *upright* vertical, so
  theta = 0 is the inverted (up) position and theta = pi hangs down.
  Rod i points along e_i = (sin theta_i, cos theta_i).

Equations of motion (Euler-Lagrange, derived by hand, verified vs sympy):

    M(theta) thetadd = -xdd * b * cos(theta)
                       - (A * sin(theta_i - theta_l)) @ thetad**2
                       + g * b * sin(theta)

    A_il = N - max(i,l) + 1/2  (i != l),   A_ii = N - i + 1/4    (1-based)
    M_il = A_il cos(theta_i - theta_l) + delta_il / 12
    b_i  = N - i + 1/2

The theta-dynamics depend on the pivot only through its acceleration xdd.
"""

from __future__ import annotations

import numpy as np


class Chain:
    """N-link chain pendulum with horizontally accelerated pivot."""

    def __init__(self, n: int, g: float = 9.81):
        self.n = n
        self.g = g
        i = np.arange(1, n + 1)
        # A_il = N - max(i,l) + 1/2 off-diagonal, N - i + 1/4 on diagonal
        mx = np.maximum.outer(i, i)
        self.A = n - mx + 0.5
        np.fill_diagonal(self.A, n - i + 0.25)
        self.b = n - i + 0.5
        # Linearization cache
        self._M0inv = np.linalg.inv(self.A + np.eye(n) / 12.0)

    # ----- continuous dynamics -------------------------------------------
    def mass_matrix(self, theta: np.ndarray) -> np.ndarray:
        dth = np.subtract.outer(theta, theta)
        return self.A * np.cos(dth) + np.eye(self.n) / 12.0

    def thetadd(self, theta: np.ndarray, thetad: np.ndarray, xdd: float) -> np.ndarray:
        """Angular accelerations given pivot acceleration xdd."""
        dth = np.subtract.outer(theta, theta)
        M = self.A * np.cos(dth) + np.eye(self.n) / 12.0
        rhs = (
            -xdd * self.b * np.cos(theta)
            - (self.A * np.sin(dth)) @ (thetad**2)
            + self.g * self.b * np.sin(theta)
        )
        return np.linalg.solve(M, rhs)

    def deriv(self, y: np.ndarray, xdd: float) -> np.ndarray:
        """y = [theta(n), thetad(n)] -> ydot."""
        n = self.n
        out = np.empty_like(y)
        out[:n] = y[n:]
        out[n:] = self.thetadd(y[:n], y[n:], xdd)
        return out

    # ----- energy (for tests; pivot velocity xd enters KE) ----------------
    def energy(self, theta: np.ndarray, thetad: np.ndarray, xd: float = 0.0) -> float:
        M = self.mass_matrix(theta)
        T = 0.5 * thetad @ M @ thetad
        T += xd * np.sum(self.b * thetad * np.cos(theta)) + 0.5 * self.n * xd**2
        V = self.g * np.sum(self.b * np.cos(theta))
        return T + V

    def tip_height(self, theta: np.ndarray) -> float:
        return float(np.sum(np.cos(theta)))

    def joint_positions(self, theta: np.ndarray, x: float = 0.0) -> np.ndarray:
        """(n+1, 2) array of joint coordinates, pivot first."""
        pts = np.zeros((self.n + 1, 2))
        pts[0] = (x, 0.0)
        pts[1:, 0] = x + np.cumsum(np.sin(theta))
        pts[1:, 1] = np.cumsum(np.cos(theta))
        return pts

    # ----- linearization about the upright equilibrium ---------------------
    def linearize_upright(self):
        """Continuous-time (Ac, Bc) for state z = [theta, thetad, x, v], input a = xdd.

        thetadd = M0^{-1} (g diag(b) theta - b a),  xd = v,  vd = a.
        """
        n = self.n
        nz = 2 * n + 2
        Ac = np.zeros((nz, nz))
        Bc = np.zeros((nz, 1))
        Ac[:n, n : 2 * n] = np.eye(n)
        Ac[n : 2 * n, :n] = self._M0inv @ (self.g * np.diag(self.b))
        Bc[n : 2 * n, 0] = -self._M0inv @ self.b
        Ac[2 * n, 2 * n + 1] = 1.0  # xd = v
        Bc[2 * n + 1, 0] = 1.0  # vd = a
        return Ac, Bc

    def linearize_at(self, theta0: np.ndarray):
        """Continuous (Ac, Bc) for state [theta, thetad] about (theta0, 0), input xdd.

        Used for time-varying LQR along trajectories; numerically differentiated.
        """
        n = self.n
        eps = 1e-6
        y0 = np.concatenate([theta0, np.zeros(n)])
        f0 = self.deriv(y0, 0.0)
        Ac = np.zeros((2 * n, 2 * n))
        for j in range(2 * n):
            yp = y0.copy()
            yp[j] += eps
            Ac[:, j] = (self.deriv(yp, 0.0) - f0) / eps
        Bc = ((self.deriv(y0, eps) - f0) / eps).reshape(-1, 1)
        return Ac, Bc
