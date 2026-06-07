import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pendulum.trajopt import solve_swingup
from scripts.gen_parallel import openloop_track

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
configs = {4: [(8.0, 0), (8.0, 3), (9.0, 1), (8.0, 5), (10.0, 2)],
          5: [(11.0, 0), (12.0, 3), (13.0, 1), (11.0, 5), (14.0, 2)]}[n]
a_max = {4: 50, 5: 60}[n]
for T, seed in configs:
    t0 = time.time()
    sc = solve_swingup(n, T, int(round(T / 0.05)), a_max=a_max, v_max=12,
                       theta_target=np.zeros(n), seed=seed, settle_frac=0.12,
                       max_iter=3000)
    msg = f'T={T} seed={seed} coarse {sc["status"]} cost={sc["cost"]:.1f} ({time.time()-t0:.0f}s)'
    if sc['status'] == 'solved':
        msg += f' track={openloop_track(n, sc, frac=0.7):.4f}'
    print(msg, flush=True)
