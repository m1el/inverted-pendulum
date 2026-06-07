"""Estimate the basin of attraction of the LQR-balanced upright equilibrium,
and the linear-theory quantization-noise amplification, per N.

Outputs results/basin.json:
- basin radius r_N: largest amplitude of worst-case (alternating-sign) initial
  tilt that the perfect-observation LQR recovers from,
- random-direction basin radius (median over random unit directions),
- unstable pole lam_max of the open-loop upright equilibrium,
- closed-loop peak gain from angle-measurement noise to angle response
  (linear amplification factor kappa).
"""

import sys, pathlib, json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from pendulum.dynamics import Chain
from pendulum.sim import simulate, upright_fail_check
from pendulum.balance import FDBalancer, upright_lqr

DT, G = 0.01, 9.81
results = {}

for n in range(1, 6):
    chain = Chain(n, G)
    Ac, _ = chain.linearize_upright()
    lam_max = float(np.max(np.linalg.eigvals(Ac).real))

    def recovers(direction, amp):
        y0 = np.zeros(2 * n)
        y0[:n] = amp * direction
        ctrl = FDBalancer(chain, DT)
        res = simulate(chain, ctrl, y0, DT, int(20 / DT),
                       fail_check=upright_fail_check(chain, 0.7))
        return res["success"] and np.max(np.abs(res["y"][:n])) < 1e-3

    def radius(direction):
        lo, hi = 1e-5, 1.0
        if not recovers(direction, lo):
            return 0.0
        if recovers(direction, hi):
            return hi
        for _ in range(20):
            mid = np.sqrt(lo * hi)
            if recovers(direction, mid):
                lo = mid
            else:
                hi = mid
        return float(np.sqrt(lo * hi))

    alt = np.array([(-1.0) ** i for i in range(n)])
    r_alt = radius(alt)
    rng = np.random.default_rng(0)
    r_rand = []
    for _ in range(10):
        d = rng.normal(size=n)
        d /= np.linalg.norm(d) / np.sqrt(n)  # per-link RMS amplitude = amp
        r_rand.append(radius(d))

    # linear closed-loop noise gain: theta response to measurement quantization
    # z+ = Ad z + Bd a, a = -K (z + noise on theta part)
    K, P, Ad, Bd = upright_lqr(chain, DT)
    Acl = Ad - Bd @ K
    # peak of ||theta part of (zI - Acl)^{-1} Bd K_theta|| over frequency
    nz = Acl.shape[0]
    kappa = 0.0
    for w in np.linspace(0, np.pi, 400):
        zinv = np.linalg.inv(np.exp(1j * w) * np.eye(nz) - Acl)
        Tn = (zinv @ Bd @ K[:, :n])[:n, :]  # noise on theta meas -> theta
        kappa = max(kappa, float(np.linalg.svd(Tn, compute_uv=False)[0]))

    results[n] = {
        "lam_max": lam_max,
        "basin_alt": r_alt,
        "basin_rand_median": float(np.median(r_rand)),
        "basin_rand_min": float(np.min(r_rand)),
        "kappa_meas_to_theta": kappa,
        "predicted_dtheta_max": r_alt / kappa if kappa > 0 else None,
    }
    print(f"N={n}: lam_max={lam_max:5.2f}  basin_alt={r_alt:.2e}  "
          f"basin_rand_med={np.median(r_rand):.2e}  kappa={kappa:.1f}  "
          f"pred dtheta_max~{r_alt/kappa:.1e}")

pathlib.Path("results").mkdir(exist_ok=True)
json.dump(results, open("results/basin.json", "w"), indent=2)
print("saved results/basin.json")
