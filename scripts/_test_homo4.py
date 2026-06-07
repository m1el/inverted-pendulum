import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from pendulum.trajopt import solve_swingup, homotopy_guess
from scripts.solve_highN import openloop_track
low=np.load('results/trajectories/swingup_N3.npz')
ig=homotopy_guess({'theta':low['theta'],'thetad':low['thetad'],'a':low['a'],'v':low['v']})
print('homotopy guess shape', ig['theta'].shape, flush=True)
t0=time.time()
sc=solve_swingup(4,7.0,140,a_max=50,v_max=12,theta_target=np.zeros(4),init_guess=ig,settle_frac=0.10,settle_band=0.25,max_iter=3000,tol=1e-6)
print('coarse',sc['status'],'cost',round(sc['cost'],1),f'({time.time()-t0:.0f}s)',flush=True)
if sc['status']=='solved':
    print('coarse track',round(openloop_track(4,sc,0.7),4),flush=True)
    t1=time.time()
    sf=solve_swingup(4,7.0,700,a_max=50,v_max=12,theta_target=np.zeros(4),init_guess=sc,settle_frac=0.10,settle_band=0.20,max_iter=3000,tol=1e-7)
    print('refine',sf['status'],'cost',round(sf['cost'],1),f'({time.time()-t1:.0f}s)',flush=True)
    if sf['status']=='solved':
        print('refine track',round(openloop_track(4,sf,0.7),4),'term',round(float(np.max(np.abs(sf['theta'][-1]))),5))
