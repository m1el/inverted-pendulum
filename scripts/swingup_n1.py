"""Swing-up N=1: verification + precision sweeps.

Usage:
  uv run python scripts/swingup_n1.py verify
  uv run python scripts/swingup_n1.py sweep      # full threshold sweep -> results/swingup1.json
"""

import sys, pathlib, json, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from pendulum.dynamics import Chain
from pendulum.sim import simulate
from pendulum.swingup1 import SwingUp1
from pendulum import protocol as P

G = 9.81
DT = 0.01
HORIZON = 25.0


def make_ctrl(chain, dt, dtheta, dv):
    return SwingUp1(chain, dt, dtheta, dv)


def verify():
    chain = Chain(1, G)
    # single recorded run from hanging
    ctrl = SwingUp1(chain, DT, 0.0, 0.0)
    y0 = np.array([np.pi, 0.0])
    n_steps = int(round(HORIZON / DT))
    res = simulate(chain, ctrl, y0, DT, n_steps, record=True)
    th = P.wrap(res["traj"]["y"][:, 0])
    t = res["traj"]["t"]
    # time to first reach within 0.3
    within = np.abs(th) < 0.3
    idx = np.argmax(within) if within.any() else -1
    print(f"reach-upright time: {t[idx]:.2f}s" if idx >= 0 else "never reached")
    tail = int(round(5.0 / DT))
    print(f"final 5s max|wrap(theta)| = {np.max(np.abs(th[-tail:])):.4f} rad")
    print(f"final v = {res['v']:+.3f}, final x = {res['x']:+.3f}")
    # full protocol success across seeds
    ok = P.swingup_success(chain, make_ctrl, DT, 0.0, 0.0, HORIZON)
    print(f"swingup_success (4 seeds, dtheta=dv=0): {ok}")
    return ok


def threshold_dtheta(chain, dt, dv, horizon, seeds):
    fn = lambda q: P.swingup_success(chain, make_ctrl, dt, q, dv, horizon, seeds=seeds)
    return P.threshold_bisect(fn)


def threshold_dv(chain, dt, dtheta, horizon, seeds):
    fn = lambda q: P.swingup_success(chain, make_ctrl, dt, dtheta, q, horizon, seeds=seeds)
    return P.threshold_bisect(fn)


def gmean(br):
    lo, hi = br
    if hi == np.inf:
        return lo
    if lo == 0.0:
        return 0.0
    return float(np.sqrt(lo * hi))


def sweep():
    seeds = range(4)
    out = {"g": G, "dt": DT, "horizon": HORIZON}
    t0 = time.time()

    # --- baseline g=9.81, dt=0.01 ---
    chain = Chain(1, G)
    br_dtheta = threshold_dtheta(chain, DT, 0.0, HORIZON, seeds)
    br_dv = threshold_dv(chain, DT, 0.0, HORIZON, seeds)
    max_dtheta = gmean(br_dtheta)
    print(f"[base] dtheta bracket={br_dtheta} gmean={max_dtheta:.3e}", flush=True)
    print(f"[base] dv     bracket={br_dv} gmean={gmean(br_dv):.3e}", flush=True)

    # joint points: dv threshold at dtheta = max_dtheta/10 and /3
    joint = {}
    for frac in (10, 3):
        dth = max_dtheta / frac
        br = threshold_dv(chain, DT, dth, HORIZON, seeds)
        joint[f"dtheta_over_{frac}"] = {"dtheta": dth, "dv_bracket": list(br),
                                        "dv_gmean": gmean(br)}
        print(f"[joint /{frac}] dtheta={dth:.3e} dv bracket={br} gmean={gmean(br):.3e}", flush=True)

    out["baseline"] = {
        "dtheta_bracket": list(br_dtheta), "max_dtheta": max_dtheta,
        "dv_bracket": list(br_dv), "max_dv": gmean(br_dv),
        "joint": joint,
    }

    # --- dt scaling ---
    out["dt_scaling"] = {}
    for dt in (0.002, 0.005, 0.02):
        ch = Chain(1, G)
        bdt = threshold_dtheta(ch, dt, 0.0, HORIZON, seeds)
        bdv = threshold_dv(ch, dt, 0.0, HORIZON, seeds)
        out["dt_scaling"][str(dt)] = {
            "dtheta_bracket": list(bdt), "max_dtheta": gmean(bdt),
            "dv_bracket": list(bdv), "max_dv": gmean(bdv),
        }
        print(f"[dt={dt}] dtheta gmean={gmean(bdt):.3e} dv gmean={gmean(bdv):.3e}", flush=True)

    # --- g scaling (dt=0.01) ---
    out["g_scaling"] = {}
    for g in (4.905, 19.62):
        ch = Chain(1, g)
        bdt = threshold_dtheta(ch, DT, 0.0, HORIZON, seeds)
        bdv = threshold_dv(ch, DT, 0.0, HORIZON, seeds)
        out["g_scaling"][str(g)] = {
            "dtheta_bracket": list(bdt), "max_dtheta": gmean(bdt),
            "dv_bracket": list(bdv), "max_dv": gmean(bdv),
        }
        print(f"[g={g}] dtheta gmean={gmean(bdt):.3e} dv gmean={gmean(bdv):.3e}", flush=True)

    out["elapsed_s"] = time.time() - t0
    pathlib.Path("results").mkdir(exist_ok=True)
    with open("results/swingup1.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote results/swingup1.json ({out['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        verify()
    elif cmd == "sweep":
        sweep()
