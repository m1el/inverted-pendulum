"""Closed-loop swing-up for high N: TVLQR tracking of a collocation trajectory
+ handoff to the tuned balance LQR. Saves closed-loop trajectory npz (+ checks
success). Perfect observation by default (for animation); quantization optional.

Usage: uv run python scripts/closedloop_high.py results/trajectories_high/swingup_N6_T12_s0.npz [dtheta] [dv]
Writes <same name>_cl.npz with t, theta, x for the animator.
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import expm

from pendulum.dynamics import Chain
from pendulum.sim import simulate
from pendulum.balance import FDBalancer
from pendulum.protocol import wrap

G = 9.81
DT = 0.01


def build_tvlqr(chain, traj, dt):
    """Backward Riccati along the (resampled-to-dt) nominal trajectory.
    State z = [theta, thetad, v], input a."""
    n = chain.n
    tN = np.arange(0, traj["t"][-1] + 1e-9, dt)
    TH = np.vstack([np.interp(tN, traj["t"], traj["theta"][:, i]) for i in range(n)]).T
    THD = np.vstack([np.interp(tN, traj["t"], traj["thetad"][:, i]) for i in range(n)]).T
    Vn = np.interp(tN, traj["t"], traj["v"])
    ta = traj["t"][:-1] if len(traj["a"]) == len(traj["t"]) - 1 else traj["t"]
    An = np.interp(tN, ta, traj["a"])

    K_steps = len(tN) - 1
    nz = 2 * n + 1
    Q = np.diag([30.0] * n + [3.0] * n + [0.3])
    R = np.array([[0.05]])
    QT = np.diag([3e4] * n + [3e3] * n + [30.0])

    P = QT.copy()
    Ks = np.zeros((K_steps, 1, nz))
    for k in range(K_steps - 1, -1, -1):
        Ac2, Bc2 = chain.linearize_at(TH[k])  # [theta,thetad] block, input a
        Ac = np.zeros((nz, nz))
        Ac[: 2 * n, : 2 * n] = Ac2
        Bc = np.zeros((nz, 1))
        Bc[: 2 * n, 0] = Bc2[:, 0]
        Bc[2 * n, 0] = 1.0
        M = np.zeros((nz + 1, nz + 1))
        M[:nz, :nz] = Ac * dt
        M[:nz, nz:] = Bc * dt
        Md = expm(M)
        Ad, Bd = Md[:nz, :nz], Md[:nz, nz:]
        H = R + Bd.T @ P @ Bd
        Kk = np.linalg.solve(H, Bd.T @ P @ Ad)
        P = Q + Ad.T @ P @ (Ad - Bd @ Kk)
        Ks[k] = Kk
    return tN, TH, THD, Vn, An, Ks


class SwingUpHigh:
    """TVLQR tracking + hysteretic handoff to tuned balance LQR."""

    def __init__(self, chain, dt, traj, dtheta=0.0, dv=0.0):
        self.chain, self.dt, self.n = chain, dt, chain.n
        (self.tN, self.TH, self.THD, self.Vn, self.An, self.Ks) = build_tvlqr(chain, traj, dt)
        self.balancer = FDBalancer(chain, dt, r=0.01, q_theta=100)
        self.caught = False
        self.hist = []

    def __call__(self, theta_meas, t, v, x):
        n = self.n
        self.hist.append(np.asarray(theta_meas, float))
        if len(self.hist) >= 3:
            thd = (3 * self.hist[-1] - 4 * self.hist[-2] + self.hist[-3]) / (2 * self.dt)
        elif len(self.hist) == 2:
            thd = (self.hist[-1] - self.hist[-2]) / self.dt
        else:
            thd = np.zeros(n)

        thw = wrap(theta_meas)
        if not self.caught and np.all(np.abs(thw) < 0.15) and np.max(np.abs(thd)) < 1.5:
            self.caught = True
            self.balancer.prev = None
        if self.caught:
            return self.balancer(thw, t, v, x)

        k = min(int(round(t / self.dt)), len(self.Ks) - 1)
        z = np.concatenate([theta_meas, thd, [v]])
        zn = np.concatenate([self.TH[k], self.THD[k], [self.Vn[k]]])
        a = self.An[min(k, len(self.An) - 1)] + float((self.Ks[k] @ (zn - z))[0])
        a = np.clip(a, -80, 80)
        return v + a * self.dt


def run(npz_path, dtheta=0.0, dv=0.0, settle=15.0):
    traj = dict(np.load(npz_path))
    n = int(traj["n"])
    chain = Chain(n, G)
    ctrl = SwingUpHigh(chain, DT, traj, dtheta, dv)
    horizon = float(traj["t"][-1]) + settle
    y0 = np.zeros(2 * n)
    y0[:n] = np.pi
    res = simulate(chain, ctrl, y0, DT, int(round(horizon / DT)),
                   dtheta=dtheta, dv=dv, record=True)
    th = res["traj"]["y"][:, :n]
    tail = int(5.0 / DT)
    ok = bool(np.all(np.abs(wrap(th[-tail:])) < 0.3))
    finalerr = float(np.max(np.abs(wrap(th[-1]))))
    print(f"{npz_path}: caught={ctrl.caught} success={ok} final_err={finalerr:.2e}")
    if ok:
        out = npz_path.replace(".npz", "_cl.npz")
        np.savez(out, t=res["traj"]["t"], theta=th, x=res["traj"]["x"])
        print(f"saved {out}")
        return out
    return None


if __name__ == "__main__":
    path = sys.argv[1]
    dq = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    dv = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    run(path, dq, dv)
