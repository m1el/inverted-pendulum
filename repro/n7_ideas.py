#!/usr/bin/env python
"""Experiments toward a trackable N=7 swing-up. Tests several ideas against a
baseline (untrackable) N=7 trajectory from the floor sweep.

  diagnostic : per-mode controllability along the trajectory (which mode dies)
  baseline   : 1-input (pivot) full-state TVLQR  -> expect divergence
  2-input    : pivot + base-joint torque         -> does extra actuation fix it?
  full-act   : torque at every joint (fully actuated) -> sanity (must track)
  small-dt   : 1-input at dt=0.002                -> numerics?
  coast      : 1-input, gains zeroed where worst-mode coupling is low

Run:  uv run python repro/n7_ideas.py [traj.npz]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import expm
import warnings; warnings.filterwarnings("ignore")
from pendulum.dynamics import Chain

G = 9.81
SRC = sys.argv[1] if len(sys.argv) > 1 else "repro/pool_ctrb_n7/floor0.88.npz"
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def resample(d, dt):
    t = d["t"]; T = t[-1]; tn = np.arange(0, T + 1e-9, dt); n = d["theta"].shape[1]
    th = np.vstack([np.interp(tn, t, d["theta"][:, i]) for i in range(n)]).T
    td = np.vstack([np.interp(tn, t, d["thetad"][:, i]) for i in range(n)]).T
    a = np.interp(tn, t, d["a"]); v = np.interp(tn, t, d["v"])
    return tn, th, td, a, v, n


def lin(chain, theta, thetad, a, ncols, torque_joints):
    """State z=[th,thd,v]; inputs = [xdd] + torque at each joint in torque_joints.
    Returns A (nz,nz), B (nz,ncols)."""
    n = chain.n; nz = 2 * n + 1
    Minv = np.linalg.inv(chain.mass_matrix(theta))
    def f(z, u):
        th, thd, v = z[:n], z[n:2 * n], z[2 * n]
        acc = chain.thetadd(th, thd, u[0])
        for j, jt in enumerate(torque_joints):
            acc = acc + Minv @ (np.eye(n)[jt]) * u[1 + j]
        return np.concatenate([thd, acc, [0.0]])
    z0 = np.concatenate([theta, thetad, [0.0]]); u0 = np.zeros(ncols); eps = 1e-6
    f0 = f(z0, u0); A = np.zeros((nz, nz)); B = np.zeros((nz, ncols))
    for j in range(nz):
        zp = z0.copy(); zp[j] += eps; A[:, j] = (f(zp, u0) - f0) / eps
    for j in range(ncols):
        up = u0.copy(); up[j] += eps; B[:, j] = (f(z0, up) - f0) / eps
    B[2 * n, 0] = 1.0  # xdd drives v
    return A, B


def tvlqr(chain, th, td, a_ff, dt, torque_joints, r_extra=1.0):
    n = chain.n; nz = 2 * n + 1; m = 1 + len(torque_joints); M = len(th) - 1
    Q = np.diag([50.0] * n + [5.0] * n + [1.0]); QF = np.diag([2000.0] * n + [200.0] * n + [10.0])
    R = np.diag([0.1] + [r_extra] * len(torque_joints))
    Ks = np.zeros((M, m, nz)); P = QF.copy(); ABs = []
    for k in range(M):
        A, B = lin(chain, th[k], td[k], a_ff[k], m, torque_joints)
        blk = np.zeros((nz + m, nz + m)); blk[:nz, :nz] = A * dt; blk[:nz, nz:] = B * dt
        E = expm(blk); ABs.append((E[:nz, :nz], E[:nz, nz:]))
    for k in range(M - 1, -1, -1):
        Ad, Bd = ABs[k]; S = R + Bd.T @ P @ Bd
        Kk = np.linalg.solve(S, Bd.T @ P @ Ad); Ks[k] = Kk
        P = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ Kk
    return Ks


def rk4_multi(chain, y, xdd, torques, torque_joints, dt):
    n = chain.n; Minv = np.linalg.inv(chain.mass_matrix(y[:n]))
    def f(yy):
        acc = chain.thetadd(yy[:n], yy[n:], xdd)
        for j, jt in enumerate(torque_joints):
            acc = acc + Minv @ np.eye(n)[jt] * torques[j]
        return np.concatenate([yy[n:], acc])
    k1 = f(y); k2 = f(y + 0.5 * dt * k1); k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(chain, th, td, a_ff, v_nom, Ks, dt, torque_joints, gate=None):
    n = chain.n; y = np.concatenate([th[0], td[0]]); v = 0.0; maxK = 0.0
    for k in range(len(Ks)):
        z = np.concatenate([y[:n], y[n:], [v]]); zn = np.concatenate([th[k], td[k], [v_nom[k]]])
        K = Ks[k]
        if gate is not None and not gate[k]:
            K = K * 0.0
        u = np.concatenate([[a_ff[k]], np.zeros(len(torque_joints))]) - K @ (z - zn)
        maxK = max(maxK, np.abs(K).max())
        y = rk4_multi(chain, y, u[0], u[1:], torque_joints, dt); v += u[0] * dt
        if not np.isfinite(y).all():
            return False, np.inf, maxK
    final = float(np.max(np.abs(wrap(y[:n]))))
    return final < 0.1, np.degrees(final), maxK


def permode(chain, th):
    """min over upright bending modes of (max over trajectory of |input coupling|)."""
    n = chain.n; M0 = chain.mass_matrix(np.zeros(n))
    _, V = np.linalg.eig(np.linalg.solve(M0, np.diag(chain.b))); V = np.real(V)
    cmax = np.zeros(n)
    for k in range(len(th)):
        w = np.linalg.solve(chain.mass_matrix(th[k]), chain.b * np.cos(th[k])); w /= np.linalg.norm(w) + 1e-12
        cmax = np.maximum(cmax, np.abs(V.T @ w))
    return cmax  # per-mode best excitation over the whole trajectory


def main():
    d = dict(np.load(SRC)); chain = Chain(int(d["theta"].shape[1]), G); n = chain.n
    print(f"N={n} baseline trajectory: {SRC}")
    dt = 0.004; tn, th, td, a_ff, v_nom, _ = resample(d, dt)

    cmax = permode(chain, th)
    worst = int(np.argmin(cmax))
    print(f"\n[diagnostic] per-mode best excitation over trajectory (min=worst mode):")
    print(f"  modes sorted: {np.round(np.sort(cmax),4)}")
    print(f"  WORST mode excitation = {cmax.min():.4f} (mode {worst})  -> near 0 = uncontrollable\n")

    def run(label, torque_joints, dt_=dt, gate=None, r_extra=1.0):
        tn2, th2, td2, a2, v2, _ = resample(d, dt_)
        Ks = tvlqr(chain, th2, td2, a2, dt_, torque_joints, r_extra)
        g2 = None
        if gate is not None:
            g2 = np.interp(np.arange(len(Ks)) * dt_, tn * 1, gate.astype(float))[: len(Ks)] > 0.5
        ok, fin, mk = rollout(chain, th2, td2, a2, v2, Ks, dt_, torque_joints, g2)
        print(f"  {label:42s}: {'TRACKS' if ok else 'diverges':9s} final={fin:8.2f}deg maxK={mk:.0f}")
        return ok

    print("[experiments]")
    run("baseline 1-input (pivot), dt=0.004", [])
    run("idea6 small dt=0.002", [], dt_=0.002)
    run("idea2 +base-joint torque (2 inputs)", [0])
    run(f"idea2 +torque at worst-coupled joint {worst}", [worst])
    run("idea2 +2 torques (base + tip)", [0, n - 1])
    run("full actuation (torque every joint) [sanity]", list(range(n)))
    # idea3 coast: gate OFF feedback where worst-mode coupling along traj is low
    M0 = chain.mass_matrix(np.zeros(n)); _, V = np.linalg.eig(np.linalg.solve(M0, np.diag(chain.b))); V = np.real(V)
    cw = np.array([abs(V[:, worst] @ (np.linalg.solve(chain.mass_matrix(th[k]), chain.b * np.cos(th[k])) /
                  (np.linalg.norm(np.linalg.solve(chain.mass_matrix(th[k]), chain.b * np.cos(th[k]))) + 1e-12)))
                  for k in range(len(th))])
    gate = cw > np.percentile(cw, 20)   # feedback ON only where worst mode is excitable
    run("idea3 coast (gate feedback by worst-mode coupling)", [], gate=gate)


if __name__ == "__main__":
    main()
