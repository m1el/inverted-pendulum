"""Smoke test: LQR balance for N=1..5 with perfect observation, plus
unstable-pole analysis of the upright equilibrium."""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from pendulum.dynamics import Chain
from pendulum.sim import simulate, upright_fail_check
from pendulum.balance import FDBalancer, KalmanBalancer

DT = 0.01
G = 9.81

for n in range(1, 6):
    chain = Chain(n, G)
    Ac, _ = chain.linearize_upright()
    eig = np.linalg.eigvals(Ac)
    lam_max = np.max(eig.real)
    y0 = np.zeros(2 * n)
    y0[:n] = 0.02 * (np.arange(n) % 2 * 2 - 1)  # alternating 0.02 rad tilt
    for name, ctrl in [
        ("FD ", FDBalancer(chain, DT)),
        ("KF ", KalmanBalancer(chain, DT, dtheta=0.0, dv=0.0)),
    ]:
        res = simulate(
            chain, ctrl, y0, DT, n_steps=int(30 / DT),
            fail_check=upright_fail_check(chain),
        )
        final_tilt = np.max(np.abs(res["y"][:n]))
        print(
            f"N={n} {name} lam_max={lam_max:5.2f}/s  success={res['success']}"
            f"  t={res['t']:5.1f}s  final max|theta|={final_tilt:.2e}  v={res['v']:+.2e}"
        )
