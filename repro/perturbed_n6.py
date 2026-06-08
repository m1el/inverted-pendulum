#!/usr/bin/env python
"""Challenge: swing-up from a PERTURBED initial state (don't rely on the exact
nominal hanging start).

Starts each run from theta_i(0) = pi + U(-dtheta, dtheta) and
thetad_i(0) = U(-dthetad, dthetad), then runs the SAME controls bundle
(TVLQR feedforward+feedback on the nominal) and the balance-LQR catch. The
feedback rejects the initial deviation -- demonstrating the swing-up does not
depend on starting exactly at the nominal hanging state.

Usage:
  uv run python repro/perturbed_n6.py [controls.npz] [--render out.mp4]
Defaults: repro/n6_controls.npz, dtheta = 0.10*pi (~10%), dthetad = 0.5 rad/s,
verifies 16 seeds and reports the measured perturbation tolerance.

Exit 0 and "CHALLENGE: PASS" iff all seeds swing up and balance at the target
perturbation.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.balance import upright_lqr

RENDER = (sys.argv[sys.argv.index("--render") + 1]
          if "--render" in sys.argv else None)
_pos = [a for i, a in enumerate(sys.argv[1:], 1)
        if not a.startswith("--") and (i == 1 or sys.argv[i - 1] != "--render")]
CONTROLS = _pos[0] if _pos else "repro/n6_controls.npz"
TARGET_DTHETA = 0.10 * np.pi      # ~10% of pi
TARGET_DTHETAD = 0.5              # rad/s
SEEDS = 16
wrap = lambda a: (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rollout(c, dtheta, dthetad, seed, hold=6.0, record=False):
    """Swing-up (TVLQR on nominal) from a perturbed start, then balance-LQR catch."""
    n, dt, g = int(c["n"]), float(c["dt"]), float(c["g"])
    chain = Chain(n, g)
    th, td, a_ff, v_nom, Ks = c["theta_nom"], c["thetad_nom"], c["a_ff"], c["v_nom"], c["K"]
    rng = np.random.default_rng(seed)
    y = np.concatenate([th[0] + rng.uniform(-dtheta, dtheta, n),
                        td[0] + rng.uniform(-dthetad, dthetad, n)])
    v = 0.0; x = 0.0
    T, X, TS = [y[:n].copy()], [x], [0.0]
    for k in range(len(Ks)):
        z = np.concatenate([y[:n], y[n:], [v]]); zn = np.concatenate([th[k], td[k], [v_nom[k]]])
        a = a_ff[k] - float((Ks[k] @ (z - zn))[0])
        yn = rk4_step(chain, y, a, dt); vn = v + a * dt; x += 0.5 * (v + vn) * dt; y, v = yn, vn
        if record: T.append(y[:n].copy()); X.append(x); TS.append((k + 1) * dt)
        if not np.isfinite(y).all(): return (False, None)
    Kb, _, _, _ = upright_lqr(chain, dt, r=0.01, q_theta=100)
    t0 = len(Ks) * dt
    for k in range(int(round(hold / dt))):
        zb = np.concatenate([wrap(y[:n]), y[n:], [x, v]])
        a = -float((Kb @ zb)[0])
        yn = rk4_step(chain, y, a, dt); vn = v + a * dt; x += 0.5 * (v + vn) * dt; y, v = yn, vn
        if record: T.append(y[:n].copy()); X.append(x); TS.append(t0 + (k + 1) * dt)
        if not np.isfinite(y).all(): return (False, None)
    tail = int(round(5.0 / dt))
    if record:
        TH = np.array(T)
        ok = bool(np.all(np.abs(wrap(TH[-tail:])) < 0.10))
        return ok, (np.array(TS), TH, np.array(X))
    return bool(np.max(np.abs(wrap(y[:n]))) < 0.10), None


def main():
    c = dict(np.load(CONTROLS))
    n = int(c["n"])
    print(f"Perturbed-start swing-up challenge: N={n}, controls={CONTROLS}")
    # verify at target
    ok = sum(rollout(c, TARGET_DTHETA, TARGET_DTHETAD, s)[0] for s in range(SEEDS))
    print(f"target: dtheta=±{TARGET_DTHETA:.3f} rad ({100*TARGET_DTHETA/np.pi:.0f}% of pi), "
          f"dthetad=±{TARGET_DTHETAD} rad/s  ->  {ok}/{SEEDS} succeed")
    passed = (ok == SEEDS)
    print(f"CHALLENGE: {'PASS' if passed else 'FAIL'}")
    # measured tolerance (angle; dthetad=0)
    print("measured angle tolerance (dthetad=0, 8 seeds):")
    tol = 0.0
    for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        s = sum(rollout(c, p, 0.0, sd)[0] for sd in range(8))
        print(f"  dtheta=±{p:.2f} rad ({100*p/np.pi:.0f}% pi): {s}/8")
        if s == 8: tol = p
    print(f"-> robust to >= ±{tol:.2f} rad ({100*tol/np.pi:.0f}% of pi) initial angle error")
    if RENDER:
        from scripts.animate import gif
        okr, traj = rollout(c, TARGET_DTHETA, TARGET_DTHETAD, 0, record=True)
        if traj is not None:
            t, th, x = traj
            gif(t, th, x, RENDER, speed=1.0)
            print(f"wrote {RENDER} (perturbed-start example, success={okr})")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
