"""N=6 closed-loop swing-up with a controllability-regularized TVLQR.

Diagnosis (see PROGRESS.md): the N=6 swing-up nominal passes through a
near-STRAIGHT chain configuration mid-swing (~t/T=0.5) where the system is
near-uncontrollable from the single pivot input (ctrb smin/smax ~1e-13). The
standard backward-Riccati TVLQR responds with exploding gains (|K|~4e4) that
amplify tiny errors and diverge -- even under perfect state feedback.

Fix: cap the feedback-gain magnitude. Through the brief uncontrollable patch
the controller cannot correct errors anyway, so it should *coast* (bounded
gain) and rely on the trajectory's own feasibility; it re-acquires authority
once the chain leaves the straight manifold. Reuses agent-C's predictor-
corrector observer and balance catch via subclassing.
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from pendulum.dynamics import Chain
from pendulum.sim import simulate
from pendulum.protocol import wrap
from pendulum.swingup_traj import SwingupController


class SwingUpN6(SwingupController):
    def __init__(self, *a, gain_cap=400.0, **kw):
        super().__init__(*a, **kw)
        # cap each TVLQR gain row by 2-norm; preserves direction, bounds size
        capped = []
        for K in self.Ks:
            nrm = np.linalg.norm(K)
            capped.append(K * (gain_cap / nrm) if nrm > gain_cap else K)
        self.Ks = capped


def run(traj_file, dtheta=0.0, dv=0.0, gain_cap=400.0, settle=20.0, save=None):
    import shutil
    shutil.copy(traj_file, "results/trajectories/swingup_N6.npz")
    ch = Chain(6, 9.81)
    d = np.load(traj_file)
    T = float(d["t"][-1])
    ctrl = SwingUpN6(ch, 0.01, dtheta, dv, "results/trajectories/swingup_N6.npz",
                     gain_cap=gain_cap)
    y0 = np.zeros(12); y0[:6] = np.pi
    res = simulate(ch, ctrl, y0, 0.01, int((T + settle) / 0.01),
                   dtheta=dtheta, dv=dv, record=True)
    th = res["traj"]["y"][:, :6]
    fin = np.isfinite(th).all()
    tail = int(5 / 0.01)
    ok = bool(fin and np.all(np.abs(wrap(th[-tail:])) < 0.3))
    fe = np.max(np.abs(wrap(th[-1]))) if fin else float("nan")
    print(f"{pathlib.Path(traj_file).name} cap={gain_cap}: success={ok} "
          f"caught={ctrl.caught} final_err={fe:.2e}")
    if ok and save:
        np.savez(save, t=res["traj"]["t"], theta=th, x=res["traj"]["x"])
        print(f"  SAVED {save}")
    return ok


if __name__ == "__main__":
    import glob
    files = sorted(glob.glob("results/trajectories_high/swingup_N6_agentc_T1[345].0_s*.npz"))
    files = [f for f in files if abs(np.load(f)["t"][1] - np.load(f)["t"][0] - 0.01) < 1e-4]
    for cap in [200, 400, 800, 1500]:
        for f in files:
            if run(f, gain_cap=cap,
                   save="results/trajectories/swingup_N6_cl.npz"):
                print(f"=== WORKING: {f} at cap={cap}")
                sys.exit(0)
    print("no combination succeeded")
