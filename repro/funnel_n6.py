#!/usr/bin/env python
"""Funnel width of the N=6 swing-up TVLQR vs phase along the trajectory.

Split the swing-up into K phases. At each phase k (state = nominal[k]) measure two
robustness margins, by log-bisection, of the closed-loop (TVLQR track-to-end +
balance catch):

  DISTURBANCE  = one-time kick to the link ANGLES at phase k (worst-case
                 alternating-sign direction, per-link amplitude eps); the max eps
                 that still recovers upright = region-of-attraction half-width.
  QUANTIZATION = angle-sensor quantization dtheta applied ONLY during the short
                 k-th segment (the M/K steps from k), sensing clean before/after;
                 the max dtheta that still recovers = LOCAL sensing margin at k.
                 (Localized, so it is a per-phase funnel -- not catch-limited.)

Success = final 2 s within 0.2 rad of upright. Output a plot max_robustness(t):
both should be wide near the hang (stable) and collapse at the near-uncontrollable
catch.

Run:  uv run python repro/funnel_n6.py [K] [n_workers]
"""
import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.balance import upright_lqr
from pendulum.protocol import threshold_bisect, wrap

K_PHASES = int(sys.argv[1]) if len(sys.argv) > 1 else 256
NW = int(sys.argv[2]) if len(sys.argv) > 2 else min(32, mp.cpu_count() - 2)
CATCH_HOLD, TAIL, UP_BAND = 4.0, 2.0, 0.20

CONTROLS = sys.argv[3] if len(sys.argv) > 3 else "repro/n6_controls.npz"
_c = dict(np.load(CONTROLS))
N, DT, G = int(_c["n"]), float(_c["dt"]), float(_c["g"])
TH, THD, A_FF, V_NOM, KS = _c["theta_nom"], _c["thetad_nom"], _c["a_ff"], _c["v_nom"], _c["K"]
M = len(KS)
SEG = max(1, M // K_PHASES)                          # steps in one segment (~M/K)
ALT = np.array([(-1.0) ** i for i in range(N)])     # worst-case alternating tilt

# Precompute the BASELINE closed-loop trajectory (the actual stable rollout). We
# perturb THIS, not the nominal -- so kick=0 reproduces the stable rollout and the
# measure is the true region of attraction, free of fresh-start-on-nominal artifacts.
def _baseline():
    chain = Chain(N, G); y = np.concatenate([TH[0], THD[0]]); v = 0.0
    Y = np.zeros((M, 2 * N)); V = np.zeros(M)
    for k in range(M):
        Y[k] = y; V[k] = v
        z = np.concatenate([y[:N], y[N:], [v]]); zn = np.concatenate([TH[k], THD[k], [V_NOM[k]]])
        a = A_FF[k] - float((KS[k] @ (z - zn))[0]); y = rk4_step(chain, y, a, DT); v += a * DT
    return Y, V
YCL, VCL = _baseline()


def rollout(k0, pert, dtheta):
    """Perturb the BASELINE closed-loop state at phase k0 by angle kick `pert`,
    with sensor quantization `dtheta` only during the k-th segment [k0, k0+SEG),
    then continue closed-loop + balance catch. True iff recovers upright."""
    chain = Chain(N, G)
    y = YCL[k0].copy(); y[:N] += pert; v = float(VCL[k0]); x = 0.0
    q_end = k0 + SEG
    for k in range(k0, M):
        th_meas = np.round(y[:N] / dtheta) * dtheta if (dtheta > 0 and k < q_end) else y[:N]
        z = np.concatenate([th_meas, y[N:], [v]])
        zn = np.concatenate([TH[k], THD[k], [V_NOM[k]]])
        a = A_FF[k] - float((KS[k] @ (z - zn))[0])
        y = rk4_step(chain, y, a, DT); v += a * DT
        if not np.isfinite(y).all():
            return False
    Kb, _, _, _ = upright_lqr(chain, DT, r=0.01, q_theta=100)   # catch: clean sensing
    hist = [y[:N].copy()]
    for _ in range(int(round(CATCH_HOLD / DT))):
        zb = np.concatenate([wrap(y[:N]), y[N:], [x, v]])
        a = -float((Kb @ zb)[0]); y = rk4_step(chain, y, a, DT); v += a * DT
        if not np.isfinite(y).all():
            return False
        hist.append(y[:N].copy())
    tail = int(round(TAIL / DT))
    return bool(np.all(np.abs(wrap(np.array(hist)[-tail:])) < UP_BAND))


def measure(k0):
    os.environ["OMP_NUM_THREADS"] = "1"
    dist = threshold_bisect(lambda e: rollout(k0, e * ALT, 0.0), lo=1e-6, hi=np.pi, iters=13)
    quant = threshold_bisect(lambda d: rollout(k0, np.zeros(N), d), lo=1e-7, hi=1.0, iters=11)
    g = lambda b: float(np.sqrt(b[0] * b[1])) if np.isfinite(b[1]) and b[0] > 0 else b[0]
    return k0, g(dist), g(quant)


def main():
    phases = np.unique(np.linspace(0, M - 1, K_PHASES).astype(int))
    with mp.Pool(NW) as pool:
        res = pool.map(measure, phases.tolist())
    res.sort()
    ks = np.array([r[0] for r in res]); t = ks * DT
    dist = np.array([r[1] for r in res]); quant = np.array([r[2] for r in res])
    dseq = np.abs(wrap(TH[:M])).max(axis=1); arrive = np.argmax(dseq < np.deg2rad(15)) * DT
    np.savez(f"repro/n{N}_funnel.npz", t=t, dist=dist, quant=quant, arrive=arrive)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.semilogy(t, dist, color="#c0392b", lw=1.6, label="max one-time angle kick (rad) — region-of-attraction")
    ax.semilogy(t, quant, color="#2c3e50", lw=1.6, label="max sensor quantization δθ (rad) — local to segment")
    ax.axvline(arrive, ls=":", color="gray"); ax.text(arrive + 0.1, ax.get_ylim()[1] * 0.4,
              f"reaches upright\nt≈{arrive:.1f}s", fontsize=8, color="gray")
    ax.set_xlabel("time along swing-up (s)"); ax.set_ylabel("max tolerated (rad, log)")
    ax.set_title(f"N={N} swing-up TVLQR funnel width vs phase "
                 f"({len(phases)} phases, seg≈{SEG*DT*1e3:.0f} ms)")
    ax.grid(alpha=0.25, which="both"); ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(f"media/n{N}_funnel.png", dpi=130)
    print(f"wrote media/n{N}_funnel.png  (arrive t={arrive:.2f}s)")
    print(f"disturbance funnel: start {dist[0]:.2e} -> min {dist.min():.2e} rad")
    print(f"quantization funnel: start {quant[0]:.2e} -> min {quant.min():.2e} rad")


if __name__ == "__main__":
    main()
