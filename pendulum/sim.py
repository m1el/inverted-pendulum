"""Simulator: RK4 integration with ZOH pivot acceleration, quantized I/O.

Control model: at each step k the controller sees angle measurements quantized
to a grid of step `dtheta` and outputs a pivot-velocity command quantized to a
grid of step `dv`. The pivot reaches that velocity at the end of the step via
constant acceleration a = (v_cmd - v) / dt (ZOH on acceleration).
"""

from __future__ import annotations

import numpy as np

from .dynamics import Chain


def quantize(x, step):
    if step <= 0:
        return np.asarray(x, dtype=float) if np.ndim(x) else float(x)
    return np.round(np.asarray(x, dtype=float) / step) * step


def rk4_step(chain: Chain, y: np.ndarray, xdd: float, dt: float) -> np.ndarray:
    k1 = chain.deriv(y, xdd)
    k2 = chain.deriv(y + 0.5 * dt * k1, xdd)
    k3 = chain.deriv(y + 0.5 * dt * k2, xdd)
    k4 = chain.deriv(y + dt * k3, xdd)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    chain: Chain,
    controller,
    y0: np.ndarray,
    dt: float,
    n_steps: int,
    dtheta: float = 0.0,
    dv: float = 0.0,
    substeps: int = 1,
    fail_check=None,
    record: bool = False,
):
    """Run closed-loop simulation.

    controller(theta_meas, t, v, x) -> v_cmd  (pivot velocity command).
    `v`, `x` are the controller's *own* bookkeeping of pivot velocity/position
    (exactly known: the actuator does what we last commanded).

    fail_check(y, t) -> bool: True means failure; simulation stops early.

    Returns dict with success flag, final state, and (optionally) trajectories.
    """
    n = chain.n
    y = np.array(y0, dtype=float)
    v = 0.0
    x = 0.0
    h = dt / substeps
    traj = {"t": [], "y": [], "v": [], "x": []} if record else None

    for k in range(n_steps):
        t = k * dt
        theta_meas = quantize(y[:n], dtheta)
        v_cmd = quantize(controller(theta_meas, t, v, x), dv)
        a = (v_cmd - v) / dt
        if record:
            traj["t"].append(t)
            traj["y"].append(y.copy())
            traj["v"].append(v)
            traj["x"].append(x)
        for _ in range(substeps):
            y = rk4_step(chain, y, a, h)
        x += (v + v_cmd) * dt / 2.0
        v = v_cmd
        if fail_check is not None and fail_check(y, t + dt):
            return _result(False, y, v, x, t + dt, traj)
    return _result(True, y, v, x, n_steps * dt, traj)


def _result(success, y, v, x, t, traj):
    out = {"success": success, "y": y, "v": v, "x": x, "t": t}
    if traj is not None:
        out["traj"] = {k: np.array(vv) for k, vv in traj.items()}
    return out


def upright_fail_check(chain: Chain, angle_limit: float = 0.5):
    """Failure when any link tilts more than angle_limit rad from upright."""

    def check(y, t):
        return bool(np.any(np.abs(y[: chain.n]) > angle_limit))

    return check
