"""Shared evaluation protocol for precision (quantization) thresholds.

Precision model
---------------
- Angle measurements quantized to grid step `dtheta` (rad).
- Velocity commands quantized to grid step `dv` (m/s).
- A *threshold* is the largest quantization step for which the task still
  succeeds across all seeds, found by log-space bisection.

Tasks
-----
BALANCE: start near upright, theta_i ~ U(-5e-4, 5e-4), thetad = 0.
  Success: all |theta_i| < 0.5 rad for the whole horizon (default 60 s).
SWINGUP: start hanging (theta_i = pi + U(-1e-3, 1e-3)), thetad = 0.
  Success: in the final 5 s of the horizon, all |wrap(theta_i)| < 0.3 rad.
"""

from __future__ import annotations

import numpy as np

from .dynamics import Chain
from .sim import simulate, upright_fail_check

G_DEFAULT = 9.81
DT_DEFAULT = 0.01
BALANCE_T = 60.0
BALANCE_SEEDS = range(8)
SWINGUP_SEEDS = range(4)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def balance_trial(chain: Chain, make_controller, dt, dtheta, dv, seed,
                  horizon=BALANCE_T, init_amp=5e-4):
    rng = np.random.default_rng(seed)
    n = chain.n
    y0 = np.zeros(2 * n)
    y0[:n] = rng.uniform(-init_amp, init_amp, n)
    ctrl = make_controller(chain, dt, dtheta, dv)
    res = simulate(chain, ctrl, y0, dt, int(round(horizon / dt)),
                   dtheta=dtheta, dv=dv, fail_check=upright_fail_check(chain))
    return res["success"]


def balance_success(chain, make_controller, dt, dtheta, dv, seeds=BALANCE_SEEDS):
    return all(balance_trial(chain, make_controller, dt, dtheta, dv, s)
               for s in seeds)


def swingup_trial(chain: Chain, make_controller, dt, dtheta, dv, seed,
                  horizon, init_amp=1e-3):
    """make_controller(chain, dt, dtheta, dv) -> controller for the full
    swing-up + catch task. Success: final 5 s within 0.3 rad of upright."""
    rng = np.random.default_rng(seed)
    n = chain.n
    y0 = np.zeros(2 * n)
    y0[:n] = np.pi + rng.uniform(-init_amp, init_amp, n)
    ctrl = make_controller(chain, dt, dtheta, dv)
    n_steps = int(round(horizon / dt))
    tail = int(round(5.0 / dt))
    state = {"ok_since": None}

    res = simulate(chain, ctrl, y0, dt, n_steps, dtheta=dtheta, dv=dv,
                   record=True)
    th = wrap(res["traj"]["y"][:, :n])
    upright = np.all(np.abs(th) < 0.3, axis=1)
    return bool(np.all(upright[-tail:]))


def swingup_success(chain, make_controller, dt, dtheta, dv, horizon,
                    seeds=SWINGUP_SEEDS):
    return all(swingup_trial(chain, make_controller, dt, dtheta, dv, s, horizon)
               for s in seeds)


def threshold_bisect(success_fn, lo=1e-7, hi=1.0, iters=9):
    """Largest step q in [lo, hi] with success_fn(q) True (log bisection).

    Assumes success at lo (checked) and failure at hi (checked); returns
    (threshold_lo, threshold_hi) bracket, geometric mean is the headline.
    """
    if not success_fn(lo):
        return (0.0, lo)  # even finest step fails
    if success_fn(hi):
        return (hi, np.inf)  # never fails in range
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        if success_fn(mid):
            lo = mid
        else:
            hi = mid
    return (lo, hi)
