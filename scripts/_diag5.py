import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pendulum.trajopt import solve_swingup, homotopy_guess

low = np.load('results/trajectories/swingup_N4.npz')
ig = homotopy_guess({'theta': low['theta'], 'thetad': low['thetad'],
                     'a': low['a'], 'v': low['v']})
print('guess shape', ig['theta'].shape, flush=True)
T = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
t0 = time.time()
sc = solve_swingup(5, T, int(round(T / 0.05)), a_max=60, v_max=12,
                   theta_target=np.zeros(5), init_guess=ig, settle_frac=0.10,
                   settle_band=0.25, max_iter=3000, tol=1e-6)
print('coarse', sc['status'], 'cost', round(sc['cost'], 1), f'({time.time()-t0:.0f}s)', flush=True)
if sc['status'] == 'solved':
    np.savez('results/trajectories/_diag5_coarse.npz', t=sc['t'], theta=sc['theta'],
             thetad=sc['thetad'], a=sc['a'], v=sc['v'], T=sc['T'], target=np.zeros(5))
    print('saved coarse', flush=True)
