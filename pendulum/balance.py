"""Balance (stabilize upright) controllers: discrete LQR with either
finite-difference velocity estimation or a steady-state Kalman filter.

State for LQR: z = [theta (n), thetad (n), x, v], input a = xdd.
The controller outputs a pivot-velocity command v_cmd = v + a*dt.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from .dynamics import Chain


def dlqr(Ad, Bd, Q, R):
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


def upright_lqr(chain: Chain, dt: float, q_theta=10.0, q_thetad=1.0, q_x=0.1, q_v=0.5, r=0.1):
    """Discrete LQR gain about the upright equilibrium."""
    Ac, Bc = chain.linearize_upright()
    Ad, Bd, *_ = cont2discrete((Ac, Bc, np.eye(Ac.shape[0]), 0), dt)
    n = chain.n
    Q = np.diag([q_theta] * n + [q_thetad] * n + [q_x, q_v])
    R = np.array([[r]])
    K, P = dlqr(Ad, Bd, Q, R)
    return K, P, Ad, Bd


class FDBalancer:
    """LQR with finite-difference velocity estimate from quantized angles."""

    def __init__(self, chain: Chain, dt: float, **lqr_kw):
        self.K, self.P, *_ = upright_lqr(chain, dt, **lqr_kw)
        self.n = chain.n
        self.dt = dt
        self.prev = None

    def reset(self):
        self.prev = None

    def __call__(self, theta_meas, t, v, x):
        if self.prev is None:
            thetad_est = np.zeros(self.n)
        else:
            thetad_est = (theta_meas - self.prev) / self.dt
        self.prev = theta_meas.copy()
        z = np.concatenate([theta_meas, thetad_est, [x, v]])
        a = float((-self.K @ z)[0])
        return v + a * self.dt


class KalmanBalancer:
    """LQR + steady-state Kalman filter on the linearized upright model.

    Treats angle quantization as measurement noise with variance dtheta^2/12
    and the (known) commanded acceleration as input. Velocity-command
    quantization is handled by feeding back the *actual* (quantized) command.
    """

    def __init__(self, chain: Chain, dt: float, dtheta: float, dv: float, **lqr_kw):
        self.n = n = chain.n
        self.dt = dt
        self.dv = dv
        self.K, self.P, Ad, Bd = upright_lqr(chain, dt, **lqr_kw)
        # KF on [theta, thetad] only (x, v are known exactly by bookkeeping)
        self.Ad = Ad[: 2 * n, : 2 * n]
        self.Bd = Bd[: 2 * n, :]
        C = np.zeros((n, 2 * n))
        C[:, :n] = np.eye(n)
        meas_var = max(dtheta, 1e-9) ** 2 / 12.0
        Rk = np.eye(n) * meas_var
        # process noise: actuation quantization enters through Bd, plus a floor
        act_var = max(dv, 1e-9) ** 2 / 12.0 / dt**2
        Qk = self.Bd @ self.Bd.T * act_var + np.eye(2 * n) * 1e-12
        Pk = solve_discrete_are(self.Ad.T, C.T, Qk, Rk)
        self.L = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + Rk)  # innovation gain
        self.C = C
        self.zhat = np.zeros(2 * n)
        self.last_a = 0.0

    def reset(self):
        self.zhat = np.zeros(2 * self.n)
        self.last_a = 0.0

    def __call__(self, theta_meas, t, v, x):
        # predict with last *actual* acceleration, then correct
        zpred = self.Ad @ self.zhat + self.Bd[:, 0] * self.last_a
        self.zhat = zpred + self.L @ (theta_meas - self.C @ zpred)
        z = np.concatenate([self.zhat, [x, v]])
        a = float((-self.K @ z)[0])
        # quantize our own output so the filter knows the *actual* command
        v_cmd = v + a * self.dt
        if self.dv > 0:
            v_cmd = round(v_cmd / self.dv) * self.dv
        self.last_a = (v_cmd - v) / self.dt
        return v_cmd
