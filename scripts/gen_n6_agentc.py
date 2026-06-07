"""Generate an N=6 swing-up trajectory using agent-C's proven implicit-
collocation solver, homotopy-warm-started from the working N=5 solution,
with a terminal settle constraint and LOW pivot-acceleration authority
(the trackable regime: N=5 worked with max|a|~8 m/s^2, max|thetadd|~421).

argv: T K seed [a_max] [v_max] [settle_frac]
Saves results/trajectories/swingup_N6.npz on solve (agent-C schema).
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from pendulum.dynamics import Chain
from pendulum.trajopt import solve_swingup_implicit, homotopy_guess

T = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0
K = int(sys.argv[2]) if len(sys.argv) > 2 else int(T * 60)
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
a_max = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
v_max = float(sys.argv[5]) if len(sys.argv) > 5 else 14.0
settle = float(sys.argv[6]) if len(sys.argv) > 6 else 0.06

n5 = dict(np.load("results/trajectories/swingup_N5.npz"))
guess = homotopy_guess(n5)

sol = solve_swingup_implicit(
    6, T, K, a_max=a_max, v_max=v_max,
    settle_frac=settle, settle_band=0.15,
    w_a=1.0, w_smooth=1e-3, seed=seed, init_guess=guess,
    max_iter=5000, print_level=3, tol=1e-7,
)
print(f"status={sol['status']}")
if sol["status"] == "solved":
    ch = Chain(6, 9.81)
    th, td, a = sol["theta"], sol["thetad"], sol["a"]
    mtdd = max(np.abs(ch.thetadd(th[k], td[k], a[min(k, len(a) - 1)])).max()
               for k in range(len(th)))
    print(f"T={T} max|thd|={np.abs(td).max():.1f} max|a|={np.abs(a).max():.1f} "
          f"max|thdd|={mtdd:.0f} cost={sol['cost']:.1f}")
    out = f"results/trajectories_high/swingup_N6_agentc_T{T}_s{seed}.npz"
    np.savez(out, t=sol["t"], theta=th, thetad=td, a=a, v=sol["v"],
             T=T, target=np.zeros(6))
    print(f"saved {out}")
