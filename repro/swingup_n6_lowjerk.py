#!/usr/bin/env python
"""Low-jerk N=6 swing-up policy: jerk-penalizing (input-augmented) TVLQR.

The baseline TVLQR commands pivot ACCELERATION a directly, so feedback error ->
instantaneous a jumps -> high jerk (da/dt) during the swing. Here we augment the
state with a and make the control input the JERK u = da/dt:

    state  z = [theta, thetad, v, a]   (2n+2)
    dyn    thetadd = f(theta,thetad,a),  vdot = a,  adot = u
    cost   penalize u (jerk) in R, plus state-tracking + an a-deviation term in Q

so a(t) is C^1 (continuous) by construction and jerk is directly penalized. The
augmented linearization reuses linearize_z: A_aug = [[A_old, B_old],[0,0]],
B_aug = [0;1] (the old input column 'a' becomes a state column).

Run:  uv run python repro/swingup_n6_lowjerk.py [r_jerk]
Compares baseline vs low-jerk on jerk RMS/peak and verifies swing-up.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import expm
import warnings; warnings.filterwarnings("ignore")

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.balance import upright_lqr
from repro.optimize_n6 import linearize_z

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
R_SWING = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0     # jerk penalty during the swing
R_CATCH = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3    # permissive at the catch
TAPER = 0.6                                                    # s ramp from swing->catch penalty


def build_tvlqr_jerk(chain, theta, thetad, a_ff, dt, q, qa, r_arr):
    """Backward Riccati for the jerk-input augmented system, per-step jerk
    penalty r_arr (len M). Returns Ks (M,1,2n+2)."""
    n = chain.n; nz = 2 * n + 2; M = len(theta) - 1
    Q = np.diag([q[0]] * n + [q[1]] * n + [q[2]] + [qa])
    QF = np.diag([2000.0] * n + [200.0] * n + [10.0] + [qa])
    Ads, Bds = [], []
    for k in range(M):
        Ao, Bo = linearize_z(chain, theta[k], thetad[k], a_ff[k])   # [th,thd,v] / input a
        A = np.zeros((nz, nz)); A[:nz - 1, :nz - 1] = Ao; A[:nz - 1, nz - 1:] = Bo  # 'a' column
        B = np.zeros((nz, 1)); B[nz - 1, 0] = 1.0                    # u = adot
        blk = np.zeros((nz + 1, nz + 1)); blk[:nz, :nz] = A * dt; blk[:nz, nz:] = B * dt
        E = expm(blk); Ads.append(E[:nz, :nz]); Bds.append(E[:nz, nz:])
    P = QF.copy(); Ks = np.zeros((M, 1, nz))
    for k in range(M - 1, -1, -1):
        Ad, Bd = Ads[k], Bds[k]; S = np.array([[r_arr[k]]]) + Bd.T @ P @ Bd
        Kk = np.linalg.solve(S, Bd.T @ P @ Ad); Ks[k] = Kk
        P = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ Kk
    return Ks


def rollout_baseline(c):
    n, dt = int(c["n"]), float(c["dt"]); chain = Chain(n, float(c["g"]))
    th, thd, a_ff, v_nom, Ks = c["theta_nom"], c["thetad_nom"], c["a_ff"], c["v_nom"], c["K"]
    y = np.concatenate([th[0], thd[0]]); v = 0.0; A = []
    for k in range(len(Ks)):
        z = np.concatenate([y[:n], y[n:], [v]]); zn = np.concatenate([th[k], thd[k], [v_nom[k]]])
        a = a_ff[k] - float((Ks[k] @ (z - zn))[0]); A.append(a)
        y = rk4_step(chain, y, a, dt); v += a * dt
    return np.array(A), float(np.max(np.abs(wrap(y[:n]))))


def rollout_lowjerk(c, Ks):
    n, dt = int(c["n"]), float(c["dt"]); chain = Chain(n, float(c["g"]))
    th, thd, a_ff, v_nom = c["theta_nom"], c["thetad_nom"], c["a_ff"], c["v_nom"]
    u_ff = np.gradient(a_ff, dt)                                    # nominal jerk
    y = np.concatenate([th[0], thd[0]]); v = 0.0; a = float(a_ff[0]); A = []
    for k in range(len(Ks)):
        z = np.concatenate([y[:n], y[n:], [v, a]])
        zn = np.concatenate([th[k], thd[k], [v_nom[k], a_ff[k]]])
        u = u_ff[k] - float((Ks[k] @ (z - zn))[0])
        A.append(a)
        y = rk4_step(chain, y, a, dt); v += a * dt; a += u * dt     # a integrates jerk -> C^1
    return np.array(A), float(np.max(np.abs(wrap(y[:n]))))


def stats(A, dt, arrive_k):
    j = np.gradient(A, dt)
    return dict(amax=np.abs(A).max(), jrms_pre=np.sqrt(np.mean(j[:arrive_k] ** 2)),
                jrms_post=np.sqrt(np.mean(j[arrive_k:] ** 2)), jmax=np.abs(j).max())


def main():
    c = dict(np.load("repro/n6_controls.npz")); dt = float(c["dt"])
    dist = np.abs(wrap(c["theta_nom"])).max(axis=1); arrive_k = int(np.argmax(dist < np.deg2rad(15)))
    Ab, fb = rollout_baseline(c)
    # time-varying jerk penalty: R_SWING (smooth) tapering to R_CATCH near upright
    M = len(c["a_ff"]) - 1; tp = int(round(TAPER / dt))
    logr = np.full(M, np.log(R_SWING))
    for k in range(max(0, arrive_k - tp), M):
        f = min(1.0, (k - (arrive_k - tp)) / max(1, tp))
        logr[k] = (1 - f) * np.log(R_SWING) + f * np.log(R_CATCH)
    r_arr = np.exp(logr)
    Ks = build_tvlqr_jerk(Chain(int(c["n"]), float(c["g"])), c["theta_nom"], c["thetad_nom"],
                          c["a_ff"], dt, q=(50.0, 5.0, 1.0), qa=2.0, r_arr=r_arr)
    Aj, fj = rollout_lowjerk(c, Ks)
    sb, sj = stats(Ab, dt, arrive_k), stats(Aj, dt, arrive_k)
    print(f"R_swing={R_SWING:g} R_catch={R_CATCH:g}  (arrive upright step {arrive_k}, t={arrive_k*dt:.2f}s)")
    print(f"{'policy':12s} {'|a|max':>7} {'jerkRMS swing':>14} {'jerkRMS catch':>14} {'jerk|max|':>10} {'final deg':>10}")
    print(f"{'baseline':12s} {sb['amax']:7.2f} {sb['jrms_pre']:14.1f} {sb['jrms_post']:14.2f} {sb['jmax']:10.1f} {np.degrees(fb):10.4f}")
    print(f"{'low-jerk':12s} {sj['amax']:7.2f} {sj['jrms_pre']:14.1f} {sj['jrms_post']:14.2f} {sj['jmax']:10.1f} {np.degrees(fj):10.4f}")
    print(f"reduction:   jerkRMS swing {sb['jrms_pre']/sj['jrms_pre']:.1f}x   peak {sb['jmax']/sj['jmax']:.1f}x")
    np.savez("repro/n6_lowjerk_controls.npz", Ks=Ks, baseline_a=Ab, lowjerk_a=Aj, dt=dt,
             arrive_k=arrive_k)
    return sb, sj


if __name__ == "__main__":
    main()
