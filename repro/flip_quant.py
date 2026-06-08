#!/usr/bin/env python
"""Quantization (precision) thresholds for the flip maneuver.

For each saved flip bundle (repro/flip_n{N}_L{L}_controls.npz), find the largest
angle-measurement step dtheta and pivot-velocity-command step dv for which the
closed-loop maneuver still

    all-up -> link L down -> all-up, then balances upright,

via log-space bisection (pendulum.protocol.threshold_bisect). Mirrors
scripts/thresholds.py for swing-up: max dtheta at dv=0, max dv at dtheta=0.

NOTE: this is the FULL-STATE quantization threshold -- the angle fed to the
controller is quantized but rates are exact (the flip is full-state TVLQR).
A realistic angle-only (FD-rate) controller would be tighter, as for swing-up.

Run:  uv run python repro/flip_quant.py [bundle.npz ...]   (default: all flip_n*)
"""
import sys, glob, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import warnings; warnings.filterwarnings("ignore")

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.balance import upright_lqr
from pendulum.protocol import threshold_bisect, wrap

CATCH_HOLD = 4.0      # s of upright catch after the maneuver
TAIL = 2.0            # s window that must stay upright
UP_BAND = 0.30        # rad (matches swing-up success convention)
VIA_BAND = 0.6        # rad: flipped link must get this close to pi at the via


def rollout(c, dtheta, dv):
    """Closed-loop flip under angle/velocity quantization, then balance catch.
    Returns (reached_via, ended_upright)."""
    n, dt, g = int(c["n"]), float(c["dt"]), float(c["g"])
    chain = Chain(n, g)
    theta, thetad = c["theta_nom"], c["thetad_nom"]
    a_ff, v_nom, Ks = c["a_ff"], c["v_nom"], c["K"]
    # flipped link + via index from the nominal (link that reaches ~pi)
    amp = np.abs(wrap(theta)).max(axis=0); L = int(np.argmax(amp))
    kvia = int(np.argmax(np.abs(wrap(theta[:, L]))))

    def q(val, step):
        return np.round(val / step) * step if step > 0 else val

    y = np.concatenate([theta[0], thetad[0]]); v = 0.0; x = 0.0
    reached = False
    for k in range(len(Ks)):
        th_meas = q(y[:n], dtheta)
        z = np.concatenate([th_meas, y[n:], [v]])         # rate exact (full-state)
        zn = np.concatenate([theta[k], thetad[k], [v_nom[k]]])
        a = a_ff[k] - float((Ks[k] @ (z - zn))[0])
        v_cmd = q(v + a * dt, dv)
        a_real = (v_cmd - v) / dt
        y = rk4_step(chain, y, a_real, dt); v = v_cmd; x += 0.5 * (v + v_cmd) * dt
        if not np.isfinite(y).all():
            return reached, False
        if k >= kvia and abs(wrap(y[L] - np.pi)) < VIA_BAND:
            reached = True
    # ---- balance catch (same quantization) ----
    Kb, _, _, _ = upright_lqr(chain, dt, r=0.01, q_theta=100)
    TH = [y[:n].copy()]
    for _ in range(int(round(CATCH_HOLD / dt))):
        th_meas = q(wrap(y[:n]), dtheta)
        zb = np.concatenate([th_meas, y[n:], [x, v]])
        a = -float((Kb @ zb)[0])
        v_cmd = q(v + a * dt, dv); a_real = (v_cmd - v) / dt
        y = rk4_step(chain, y, a_real, dt); v = v_cmd; x += 0.5 * (v + v_cmd) * dt
        if not np.isfinite(y).all():
            return reached, False
        TH.append(y[:n].copy())
    TH = np.array(TH); tail = int(round(TAIL / dt))
    upright = bool(np.all(np.abs(wrap(TH[-tail:])) < UP_BAND))
    return reached, upright


def flip_ok(c, dtheta, dv):
    reached, upright = rollout(c, dtheta, dv)
    return reached and upright


def measure(path):
    c = dict(np.load(path))
    n = int(c["n"]); dt = float(c["dt"])
    amp = np.abs(wrap(c["theta_nom"])).max(axis=0); L = int(np.argmax(amp))
    # sanity: zero-quantization must succeed
    if not flip_ok(c, 0.0, 0.0):
        return dict(path=path, n=n, L=L, dt=dt, ok0=False)
    dth = threshold_bisect(lambda qv: flip_ok(c, qv, 0.0), lo=1e-6, hi=0.5, iters=9)
    dvv = threshold_bisect(lambda qv: flip_ok(c, 0.0, qv), lo=1e-6, hi=2.0, iters=9)
    return dict(path=path, n=n, L=L, dt=dt, ok0=True, dtheta=dth, dv=dvv,
                T=float(c["t"][-1]), maxK=float(c["maxK"]))


def main():
    paths = sys.argv[1:] or sorted(glob.glob("repro/flip_n*_L*_controls.npz"))
    if not paths:
        raise SystemExit("no flip bundles found (run repro/flip_one_n6.py N first)")
    print(f"{'N':>2} {'L':>2} {'dt':>6} {'swingT':>7} {'maxK':>8} "
          f"{'max dtheta(rad)':>16} {'max dv(m/s)':>14}")
    rows = []
    for p in paths:
        r = measure(p)
        if not r.get("ok0"):
            print(f"{r['n']:>2} {r['L']:>2}  zero-quant FAILED ({p})"); continue
        dthg = float(np.sqrt(r['dtheta'][0] * r['dtheta'][1])) if np.isfinite(r['dtheta'][1]) else r['dtheta'][0]
        dvg = float(np.sqrt(r['dv'][0] * r['dv'][1])) if np.isfinite(r['dv'][1]) else r['dv'][0]
        print(f"{r['n']:>2} {r['L']:>2} {r['dt']:>6.3f} {r['T']:>7.2f} {r['maxK']:>8.0f} "
              f"{dthg:>16.2e} {dvg:>14.2e}")
        rows.append((r['n'], r['L'], dthg, dvg))
    return rows


if __name__ == "__main__":
    main()
