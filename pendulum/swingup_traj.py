"""Swing-up controller: feedforward trajectory + time-varying LQR tracking,
with a hand-off to a balance LQR (Kalman) once near upright.

Compatible with pendulum.sim.simulate: controller(theta_meas, t, v, x) -> v_cmd.

State for TVLQR: z = [theta(n), thetad(n), v]   (v = pivot velocity).
Control: a = pivot acceleration.  Interface: v_cmd = v + a*dt.

The nominal trajectory is loaded from results/trajectories/swingup_N{n}.npz
(arrays t, theta, thetad, a, v, target).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from .dynamics import Chain
from .balance import KalmanBalancer, upright_lqr


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# Best balance tunings found by scripts/tune_balance.py (largest upright basin).
BALANCE_TUNINGS = {
    1: dict(q_theta=100, q_thetad=10, q_x=0.01, q_v=0.1, r=0.01),
    2: dict(q_theta=100, q_thetad=10, q_x=0.01, q_v=0.1, r=0.01),
    3: dict(q_theta=500, q_thetad=10, q_x=0.001, q_v=0.05, r=0.005),
    4: dict(q_theta=10, q_thetad=1, q_x=0.1, q_v=0.5, r=0.1),
    5: dict(q_theta=10, q_thetad=1, q_x=0.1, q_v=0.5, r=0.1),
}


def _linearize_z(chain: Chain, theta0, thetad0, a0):
    """Continuous (Ac, Bc) for z=[theta,thetad,v], input da, about a nominal
    point (theta0, thetad0, a0). v has trivial dynamics vd=a; here input is the
    acceleration a directly, so z'=[thetad, thetadd(theta,thetad,a), a].
    Numerically differentiated."""
    n = chain.n
    eps = 1e-6

    def fz(z, a):
        th = z[:n]; td = z[n:2 * n]
        tdd = chain.thetadd(th, td, a)
        return np.concatenate([td, tdd, [a]])

    z0 = np.concatenate([theta0, thetad0, [0.0]])
    f0 = fz(z0, a0)
    nz = 2 * n + 1
    Ac = np.zeros((nz, nz))
    for j in range(nz):
        zp = z0.copy(); zp[j] += eps
        Ac[:, j] = (fz(zp, a0) - f0) / eps
    Bc = ((fz(z0, a0 + eps) - f0) / eps).reshape(-1, 1)
    return Ac, Bc


def compute_tvlqr(chain: Chain, traj, dt,
                  q_theta=50.0, q_thetad=5.0, q_v=1.0, r=0.5,
                  qf_theta=2000.0, qf_thetad=200.0, qf_v=10.0):
    """Backward Riccati recursion along the nominal trajectory.

    Returns Ks: list of gain matrices (1 x nz) for k=0..N-1 (control nodes).
    States/inputs are deviations from nominal.
    """
    n = chain.n
    nz = 2 * n + 1
    theta = traj["theta"]; thetad = traj["thetad"]; a = traj["a"]
    Nsteps = theta.shape[0] - 1

    Q = np.diag([q_theta] * n + [q_thetad] * n + [q_v])
    R = np.array([[r]])
    Qf = np.diag([qf_theta] * n + [qf_thetad] * n + [qf_v])

    # discretize each segment (Euler is fine at dt=0.01 along smooth traj; use
    # matrix exponential via series for accuracy)
    from scipy.linalg import expm
    Ads = []; Bds = []
    for k in range(Nsteps):
        Ac, Bc = _linearize_z(chain, theta[k], thetad[k], a[k])
        # zero-order-hold discretization
        Mblk = np.zeros((nz + 1, nz + 1))
        Mblk[:nz, :nz] = Ac * dt
        Mblk[:nz, nz:nz + 1] = Bc * dt
        E = expm(Mblk)
        Ad = E[:nz, :nz]
        Bd = E[:nz, nz:nz + 1]
        Ads.append(Ad); Bds.append(Bd)

    P = Qf
    Ks = [None] * Nsteps
    for k in range(Nsteps - 1, -1, -1):
        Ad = Ads[k]; Bd = Bds[k]
        S = R + Bd.T @ P @ Bd
        K = np.linalg.solve(S, Bd.T @ P @ Ad)
        Ks[k] = K
        P = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K
    return Ks


class SwingupController:
    """Feedforward + TVLQR swing-up, switching to balance LQR near upright.

    Estimates angular velocity from quantized angle measurements via a
    time-varying Kalman-ish filter (here a simple low-pass finite difference
    during swing; the balance phase uses the steady-state KalmanBalancer).
    """

    def __init__(self, chain: Chain, dt, dtheta, dv, traj_file,
                 tvlqr_kw=None, balance_kw=None,
                 catch_angle=0.2, catch_rate=2.0):
        self.chain = chain
        self.n = n = chain.n
        self.dt = dt
        self.dtheta = dtheta
        self.dv = dv
        data = np.load(traj_file)
        self.tnodes = data["t"]
        self.nom_theta = data["theta"]
        self.nom_thetad = data["thetad"]
        self.nom_a = data["a"]
        self.nom_v = data["v"]
        self.target = data["target"] if "target" in data else self.nom_theta[-1]
        self.T = float(self.tnodes[-1])
        self.Nsteps = self.nom_theta.shape[0] - 1

        if tvlqr_kw is None:
            tvlqr_kw = {}
        self.Ks = compute_tvlqr(chain, data, dt, **tvlqr_kw)

        bkw = balance_kw if balance_kw is not None else BALANCE_TUNINGS.get(n, {})
        self.balance = KalmanBalancer(chain, dt, dtheta, dv, **bkw)

        self.catch_angle = catch_angle
        self.catch_rate = catch_rate
        self.reset()

    def reset(self):
        self.k = 0
        self.prev_theta = None
        self.thetad_est = np.zeros(self.n)
        self.caught = False
        self.balance.reset()
        # unwrap tracking: keep continuous angle estimate
        self.theta_cont = None

    def _estimate(self, theta_meas):
        """Continuous-angle + velocity estimate from quantized measurement.

        We unwrap the measurement to follow the nominal continuous trajectory
        (which sweeps through multiples of pi), then low-pass the finite
        difference for velocity."""
        if self.theta_cont is None:
            self.theta_cont = theta_meas.copy()
            self.prev_theta = theta_meas.copy()
            self.thetad_est = np.zeros(self.n)
            return self.theta_cont.copy(), self.thetad_est.copy()
        # unwrap relative to previous continuous estimate
        d = theta_meas - (self.theta_cont % (2 * np.pi))
        d = (d + np.pi) % (2 * np.pi) - np.pi
        new_cont = self.theta_cont + d
        raw_vel = (new_cont - self.theta_cont) / self.dt
        # low-pass
        alpha = 0.5
        self.thetad_est = (1 - alpha) * self.thetad_est + alpha * raw_vel
        self.theta_cont = new_cont
        return self.theta_cont.copy(), self.thetad_est.copy()

    def __call__(self, theta_meas, t, v, x):
        n = self.n
        theta_meas = np.asarray(theta_meas, float)
        theta_est, thetad_est = self._estimate(theta_meas)

        # Decide phase. Switch to balance when near upright AND past most of traj.
        wrapped = wrap(theta_est - self.target % (2 * np.pi))
        near = (np.all(np.abs(wrap(theta_est)) < self.catch_angle)
                and np.all(np.abs(thetad_est) < self.catch_rate))
        if self.caught or (near and self.k >= self.Nsteps - 1):
            if not self.caught:
                self.caught = True
                # seed balance KF state with current estimate (wrapped to upright)
                self.balance.zhat = np.concatenate([wrap(theta_est), thetad_est])
                self.balance.last_a = 0.0
            return self.balance(wrap(theta_meas), t, v, x)

        # --- TVLQR feedforward+feedback phase ---
        k = min(self.k, self.Nsteps - 1)
        z_nom = np.concatenate([self.nom_theta[k], self.nom_thetad[k], [self.nom_v[k]]])
        z_est = np.concatenate([theta_est, thetad_est, [v]])
        dz = z_est - z_nom
        a_ff = self.nom_a[k]
        a = a_ff - float((self.Ks[k] @ dz)[0])
        self.k += 1
        v_cmd = v + a * self.dt
        return v_cmd


def make_controller_factory(traj_dir, tvlqr_kw=None, balance_kw=None,
                            catch_angle=0.2, catch_rate=2.0):
    """Return make_controller(chain, dt, dtheta, dv) for protocol functions."""
    import pathlib
    traj_dir = pathlib.Path(traj_dir)

    def make(chain, dt, dtheta, dv):
        fn = traj_dir / f"swingup_N{chain.n}.npz"
        return SwingupController(chain, dt, dtheta, dv, str(fn),
                                 tvlqr_kw=tvlqr_kw, balance_kw=balance_kw,
                                 catch_angle=catch_angle, catch_rate=catch_rate)
    return make
