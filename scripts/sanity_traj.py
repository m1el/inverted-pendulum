"""Sanity-check: record a balance trajectory near the dtheta threshold and
report whether it limit-cycles (sustained oscillation) vs. settles.

Usage: uv run scripts/sanity_traj.py N controller dtheta dv [g] [dt]
Prints amplitude stats over the last 20 s and saves a PNG to runs/.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pendulum.dynamics import Chain
from pendulum.balance import FDBalancer, KalmanBalancer
from pendulum.sim import simulate, upright_fail_check

n = int(sys.argv[1]); controller = sys.argv[2]
dtheta = float(sys.argv[3]); dv = float(sys.argv[4])
g = float(sys.argv[5]) if len(sys.argv) > 5 else 9.81
dt = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01
lqr_kw = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}

chain = Chain(n, g)
if controller == "FD":
    ctrl = FDBalancer(chain, dt, **lqr_kw)
else:
    ctrl = KalmanBalancer(chain, dt, dtheta, dv, **lqr_kw)

rng = np.random.default_rng(0)
y0 = np.zeros(2 * n)
y0[:n] = rng.uniform(-5e-4, 5e-4, n)
T = 60.0
res = simulate(chain, ctrl, y0, dt, int(round(T / dt)),
               dtheta=dtheta, dv=dv, fail_check=upright_fail_check(chain),
               record=True)
print("success:", res["success"], "t_end:", res["t"])
th = res["traj"]["y"][:, :n]
t = res["traj"]["t"]
tail = t >= (T - 20.0)
if tail.sum() > 0:
    amp = np.max(np.abs(th[tail]), axis=0)
    std = np.std(th[tail], axis=0)
    print("last-20s max|theta| per link:", np.round(amp, 5).tolist())
    print("last-20s std per link:      ", np.round(std, 6).tolist())
    rng_amp = amp.max()
    print(f"interpretation: {'LIMIT-CYCLE (sustained ~%.4f rad)' % rng_amp if rng_amp > 5*dtheta else 'settled to quantization floor'}")

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i in range(n):
    ax[0].plot(t, th[:, i], label=f"link {i+1}")
ax[0].axhline(dtheta, color="k", ls=":", lw=0.6)
ax[0].axhline(-dtheta, color="k", ls=":", lw=0.6)
ax[0].set_ylabel("theta (rad)"); ax[0].legend(fontsize=7)
ax[0].set_title(f"N={n} {controller} dtheta={dtheta:.4g} dv={dv:.4g} g={g} dt={dt}")
ax[1].plot(t, res["traj"]["v"]); ax[1].set_ylabel("pivot v"); ax[1].set_xlabel("t (s)")
out = pathlib.Path(__file__).resolve().parents[1] / "runs" / f"traj_N{n}_{controller}.png"
fig.savefig(out, dpi=90, bbox_inches="tight")
print("saved", out)
