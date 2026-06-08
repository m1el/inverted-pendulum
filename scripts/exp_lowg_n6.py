"""Test whether lower gravity / smaller dt helps N=6 swing-up.

Generates a nominal at gravity g, then runs PERFECT-STATE TVLQR at timestep dt.
Decisive question: does it still diverge at the same dimensionless config
(chain near-straight)? If yes -> obstruction is fundamental (scale-invariant).

INSTRUMENTED: prints a timestamped progress line at every stage so progress is
visible in the log (run with PYTHONUNBUFFERED=1). IPOPT print_level=5.
"""
import sys, time; sys.path.insert(0, '.')
import numpy as np
from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.protocol import wrap
from pendulum.trajopt import solve_swingup_implicit, homotopy_guess
from scipy.linalg import expm

T0 = time.monotonic()
def log(msg):
    print(f"[{time.monotonic()-T0:7.1f}s] {msg}", flush=True)

def _linz(ch, th, td, a):
    n = ch.n; eps = 1e-6; nz = 2*n+1
    z0 = np.concatenate([th, td, [0.0]])
    def f(z):
        return np.concatenate([z[n:2*n], ch.thetadd(z[:n], z[n:2*n], a), [0.0]])
    f0 = f(z0); A = np.zeros((nz, nz))
    for j in range(nz):
        zp = z0.copy(); zp[j] += eps; A[:, j] = (f(zp)-f0)/eps
    B = np.zeros((nz, 1))
    B[n:2*n, 0] = (ch.thetadd(th, td, a+eps)-ch.thetadd(th, td, a))/eps
    B[2*n, 0] = 1.0
    return A, B

def tvlqr(ch, th, td, a, dt, r=0.1):
    n = ch.n; nz = 2*n+1; N = len(th)-1
    Q = np.diag([50.]*n+[5.]*n+[1.]); R = np.array([[r]])
    P = np.diag([2000.]*n+[200.]*n+[10.])
    Ads = []; Bds = []
    for k in range(N):
        A, B = _linz(ch, th[k], td[k], a[k])
        M = np.zeros((nz+1, nz+1)); M[:nz, :nz] = A*dt; M[:nz, nz:] = B*dt; E = expm(M)
        Ads.append(E[:nz, :nz]); Bds.append(E[:nz, nz:])
    Ks = [None]*N
    for k in range(N-1, -1, -1):
        Ad, Bd = Ads[k], Bds[k]; S = R+Bd.T@P@Bd
        K = np.linalg.solve(S, Bd.T@P@Ad); Ks[k] = K
        P = Q+Ad.T@P@Ad-Ad.T@P@Bd@K
    return Ks

def run(tag, g, dt_sim, T):
    log(f"--- CASE {tag}: g={g} dt_sim={dt_sim} T={T} ---")
    ch = Chain(6, g); K = int(T*100)
    g5 = homotopy_guess(dict(np.load('results/trajectories/swingup_N5.npz')))
    log(f"    solving collocation (K={K} nodes)...")
    t1 = time.monotonic()
    sol = solve_swingup_implicit(6, T, K, g=g, a_max=25, v_max=14,
        settle_frac=0.08, settle_band=0.15, w_a=1.0, w_smooth=1e-3,
        init_guess=g5, max_iter=4000, print_level=5, tol=1e-7)
    log(f"    collocation {sol['status']} in {time.monotonic()-t1:.0f}s")
    if sol['status'] != 'solved':
        log(f"    CASE {tag}: traj FAILED"); return
    tn = np.arange(0, T+1e-9, dt_sim)
    th = np.vstack([np.interp(tn, sol['t'], sol['theta'][:, i]) for i in range(6)]).T
    td = np.vstack([np.interp(tn, sol['t'], sol['thetad'][:, i]) for i in range(6)]).T
    a = np.interp(tn, sol['t'], sol['a'])
    log(f"    building TVLQR ({len(th)} nodes)...")
    Ks = tvlqr(ch, th, td, a, dt_sim)
    y = np.concatenate([th[0], td[0]]); v = 0.0; maxK = 0; tdiv = None
    for k in range(len(Ks)):
        zn = np.concatenate([th[k], td[k], [np.interp(k*dt_sim, sol['t'], sol['v'])]])
        z = np.concatenate([y[:6], y[6:], [v]])
        a_cmd = a[k]-float((Ks[k]@(z-zn))[0]); maxK = max(maxK, np.abs(Ks[k]).max())
        y = rk4_step(ch, y, a_cmd, dt_sim); v += a_cmd*dt_sim
        if not np.isfinite(y).all(): tdiv = k*dt_sim; break
    fin = np.isfinite(y).all()
    fe = np.max(np.abs(wrap(y[:6]))) if fin else float('nan')
    frac = round(tdiv/T, 3) if tdiv else None
    log(f"    RESULT {tag}: div_t={tdiv} (frac={frac}) maxK={maxK:.0f} "
        f"final|wrap|={fe:.2e} {'OK' if fin and fe < 0.3 else 'DIVERGED'}")

if __name__ == '__main__':
    run("baseline",   9.81,   0.01,  15)
    run("smalldt",    9.81,   0.004, 15)
    run("lowg",       2.4525, 0.01,  30)
    run("lowg+dt",    2.4525, 0.004, 30)
    log("ALL DONE")
