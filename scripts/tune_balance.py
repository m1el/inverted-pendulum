"""Find LQR tunings that give a large upright basin for N=4,5.

Tests basin size against alternating perturbations (the hardest mode).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pendulum.dynamics import Chain
from pendulum.sim import simulate, upright_fail_check
from pendulum.balance import upright_lqr

DT = 0.01
G = 9.81


def make_lqr_ctrl(chain, dt, **kw):
    K, P, Ad, Bd = upright_lqr(chain, dt, **kw)
    n = chain.n

    def ctrl(theta_meas, t, v, x):
        # perfect-obs: we need thetad too; use a closure-stored prev
        nonlocal_prev[0] = theta_meas
        return None
    return K


def run_basin(chain, K, amp, dt=DT, horizon=20.0):
    """Perfect-observation closed loop from alternating tilt of size amp."""
    n = chain.n
    y0 = np.zeros(2 * n)
    y0[:n] = amp * (np.arange(n) % 2 * 2 - 1)

    prev = {"th": None}

    def ctrl(theta_meas, t, v, x):
        # finite diff velocity from exact theta (perfect obs)
        return None

    # do a manual integration loop with perfect state feedback
    from pendulum.sim import rk4_step
    y = y0.copy()
    v = 0.0; x = 0.0
    for k in range(int(horizon / dt)):
        z = np.concatenate([y[:n], y[n:], [x, v]])
        a = float((-K @ z)[0])
        v_cmd = v + a * dt
        y = rk4_step(chain, y, a, dt)
        x += (v + v_cmd) * dt / 2
        v = v_cmd
        if np.any(np.abs(y[:n]) > 0.5):
            return False, k * dt
    return np.max(np.abs(y[:n])) < 1e-3, horizon


def find_max_basin(chain, **kw):
    K, *_ = upright_lqr(chain, DT, **kw)
    # bisect amplitude
    lo, hi = 1e-5, 0.3
    ok_lo, _ = run_basin(chain, K, lo)
    if not ok_lo:
        return 0.0, K
    ok_hi, _ = run_basin(chain, K, hi)
    if ok_hi:
        return hi, K
    for _ in range(22):
        mid = np.sqrt(lo * hi)
        ok, _ = run_basin(chain, K, mid)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo, K


if __name__ == "__main__":
    import itertools
    # candidate tunings
    tunings = [
        dict(q_theta=10, q_thetad=1, q_x=0.1, q_v=0.5, r=0.1),
        dict(q_theta=100, q_thetad=10, q_x=0.01, q_v=0.1, r=0.01),
        dict(q_theta=100, q_thetad=1, q_x=0.001, q_v=0.05, r=0.01),
        dict(q_theta=300, q_thetad=5, q_x=0.001, q_v=0.05, r=0.01),
        dict(q_theta=500, q_thetad=10, q_x=0.001, q_v=0.05, r=0.005),
        dict(q_theta=1000, q_thetad=20, q_x=0.001, q_v=0.05, r=0.005),
        dict(q_theta=200, q_thetad=2, q_x=0.0001, q_v=0.01, r=0.02),
        dict(q_theta=50, q_thetad=0.5, q_x=0.001, q_v=0.1, r=0.05),
        dict(q_theta=50, q_thetad=2, q_x=0.01, q_v=0.2, r=0.05),
    ]
    for n in [2, 3, 4, 5]:
        chain = Chain(n, G)
        print(f"=== N={n} ===")
        best = (0.0, None)
        for kw in tunings:
            amp, K = find_max_basin(chain, **kw)
            print(f"  amp={amp:.4e}  {kw}")
            if amp > best[0]:
                best = (amp, kw)
        print(f"  BEST N={n}: amp={best[0]:.4e} {best[1]}")
