import sys, time; sys.path.insert(0,'.')
import numpy as np
from pendulum.dynamics import Chain
from pendulum.sim import simulate
from pendulum.protocol import wrap
from pendulum.trajopt import solve_swingup_implicit, homotopy_guess
T0=time.monotonic()
def log(m): print(f"[{time.monotonic()-T0:6.1f}s] {m}", flush=True)

g=9.81; T=15.0; K=int(T*100)
log(f"solving N=6 nominal (g={g}, T={T}, K={K})...")
g5=homotopy_guess(dict(np.load('results/trajectories/swingup_N5.npz')))
sol=solve_swingup_implicit(6,T,K,g=g,a_max=25,v_max=14,settle_frac=0.08,
    settle_band=0.15,w_a=1.0,w_smooth=1e-3,init_guess=g5,max_iter=3000,
    print_level=0,tol=1e-7)
log(f"nominal status={sol['status']}")
if sol['status']!='solved':
    log("nominal failed"); sys.exit(1)
np.savez('results/trajectories/swingup_N6.npz', t=sol['t'], theta=sol['theta'],
         thetad=sol['thetad'], a=sol['a'], v=sol['v'], T=T, target=np.zeros(6))
log("saved nominal -> results/trajectories/swingup_N6.npz")

from pendulum.swingup_traj import SwingupController
ch=Chain(6,g)
for dt in [0.004, 0.002, 0.005, 0.01]:
    for dq,dv in [(0.0,0.0)]:
        ctrl=SwingupController(ch,dt,dq,dv,'results/trajectories/swingup_N6.npz')
        y0=np.zeros(12); y0[:6]=np.pi
        res=simulate(ch,ctrl,y0,dt,int((T+20)/dt),dtheta=dq,dv=dv,record=True)
        th=res['traj']['y'][:,:6]; fin=np.isfinite(th).all(); tail=int(5/dt)
        ok=bool(fin and np.all(np.abs(wrap(th[-tail:]))<0.3))
        fe=np.max(np.abs(wrap(th[-1]))) if fin else float('nan')
        log(f"REALISTIC dt={dt} dq={dq} dv={dv}: ok={ok} caught={ctrl.caught} final|wrap|={fe:.2e}")
        if ok:
            np.savez('results/trajectories/swingup_N6_cl.npz', t=res['traj']['t'],
                     theta=th, x=res['traj']['x'])
            log(f"*** SAVED closed-loop N=6 swing-up at dt={dt}")
            sys.exit(0)
log("no dt worked with observer")
