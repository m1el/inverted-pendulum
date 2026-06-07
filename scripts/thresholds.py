"""Precision (quantization) thresholds for swing-up via protocol.threshold_bisect.

For each working N:
  - zero-quantization success check
  - max dtheta with dv=0
  - max dv with dtheta=0
  - joint point: max dv at dtheta = (dtheta_threshold)/10

Results written to results/swingup_traj.json.

Usage: uv run python scripts/thresholds.py [N ...]
"""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pendulum.dynamics import Chain
from pendulum.protocol import swingup_success, threshold_bisect, SWINGUP_SEEDS
from pendulum.swingup_traj import make_controller_factory

ROOT = pathlib.Path(__file__).resolve().parents[1]
G = 9.81
DT = 0.01


def main():
    Ns = [int(x) for x in sys.argv[1:]]
    if not Ns:
        Ns = [n for n in [2, 3, 4, 5]
              if (ROOT / "results" / "trajectories" / f"swingup_N{n}.npz").exists()]
    mk = make_controller_factory(str(ROOT / "results" / "trajectories"))

    # Each invocation writes per-N files to avoid races; merge separately.
    perN_dir = ROOT / "results" / "thr_perN"
    perN_dir.mkdir(parents=True, exist_ok=True)

    for n in Ns:
        chain = Chain(n, G)
        fn = ROOT / "results" / "trajectories" / f"swingup_N{n}.npz"
        T = float(np.load(fn)["t"][-1])
        horizon = T + 20.0
        print(f"=== N={n} T={T} horizon={horizon} ===", flush=True)

        t0 = time.time()
        zero_ok = swingup_success(chain, mk, DT, 0.0, 0.0, horizon)
        print(f"  zero-quant success: {zero_ok} ({time.time()-t0:.0f}s)", flush=True)

        def succ_dtheta(q):
            return swingup_success(chain, mk, DT, q, 0.0, horizon)

        def succ_dv(q):
            return swingup_success(chain, mk, DT, 0.0, q, horizon)

        t0 = time.time()
        dth_lo, dth_hi = threshold_bisect(succ_dtheta, lo=1e-6, hi=0.2, iters=9)
        print(f"  dtheta threshold: [{dth_lo:.3e}, {dth_hi:.3e}] ({time.time()-t0:.0f}s)", flush=True)

        t0 = time.time()
        dv_lo, dv_hi = threshold_bisect(succ_dv, lo=1e-6, hi=2.0, iters=9)
        print(f"  dv threshold: [{dv_lo:.3e}, {dv_hi:.3e}] ({time.time()-t0:.0f}s)", flush=True)

        # joint point: dv threshold at dtheta = dth_lo/10
        dth_joint = dth_lo / 10.0

        def succ_dv_joint(q):
            return swingup_success(chain, mk, DT, dth_joint, q, horizon)

        t0 = time.time()
        jdv_lo, jdv_hi = threshold_bisect(succ_dv_joint, lo=1e-6, hi=2.0, iters=9)
        print(f"  joint dv@dtheta={dth_joint:.2e}: [{jdv_lo:.3e}, {jdv_hi:.3e}] ({time.time()-t0:.0f}s)", flush=True)

        rec = dict(
            swing_duration=T,
            horizon=horizon,
            zero_quant_success=bool(zero_ok),
            dtheta_threshold=[dth_lo, dth_hi],
            dv_threshold=[dv_lo, dv_hi],
            joint=dict(dtheta=dth_joint, dv_threshold=[jdv_lo, jdv_hi]),
            seeds=list(SWINGUP_SEEDS),
        )
        json.dump(rec, open(perN_dir / f"N{n}.json", "w"), indent=2)
        print(f"  saved N={n} to {perN_dir / f'N{n}.json'}", flush=True)


if __name__ == "__main__":
    main()
