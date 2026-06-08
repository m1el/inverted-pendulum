"""Reproduce the perfect-state-trackable N=6 swing-up at dt=0.004 (the smalldt
case from exp_lowg_n6.py) and SAVE the closed-loop rollout for animation.
This is FULL-STATE FEEDBACK (idealized sensing): the TVLQR sees the true state,
no observer. Demonstrates the swing-up trajectory + feedback are valid; a
realistic observer cannot feed the ~70k gains (see PROGRESS.md)."""
import sys, time; sys.path.insert(0,'.')
import numpy as np
from pendulum.dynamics import Chain
from pendulum.sim import rk4_step
from pendulum.protocol import wrap
from pendulum.trajopt import solve_swingup_implicit, homotopy_guess
from scripts.exp_lowg_n6 import tvlqr   # reuse the exact TVLQR that worked
T0=time.monotonic()
def log(m): print(f"[{time.monotonic()-T0:6.1f}s] {m}", flush=True)

g=9.81; T=15.0; K=int(T*100); dt=0.004
ch=Chain(6,g)
log("solving nominal (same settings as smalldt case)...")
g5=homotopy_guess(dict(np.load('results/trajectories/swingup_N5.npz')))
sol=solve_swingup_implicit(6,T,K,g=g,a_max=25,v_max=14,settle_frac=0.08,
    settle_band=0.15,w_a=1.0,w_smooth=1e-3,init_guess=g5,max_iter=4000,
    print_level=0,tol=1e-7)
log(f"nominal status={sol['status']}")
# resample to dt
tn=np.arange(0,T+1e-9,dt)
th=np.vstack([np.interp(tn,sol['t'],sol['theta'][:,i]) for i in range(6)]).T
td=np.vstack([np.interp(tn,sol['t'],sol['thetad'][:,i]) for i in range(6)]).T
a=np.interp(tn,sol['t'],sol['a']); vn=np.interp(tn,sol['t'],sol['v'])
log("building TVLQR...")
Ks=tvlqr(ch,th,td,a,dt)
# perfect-state rollout, recording theta and pivot x
y=np.concatenate([th[0],td[0]]); v=0.0; x=0.0
TH=[y[:6].copy()]; XS=[x]; TS=[0.0]; maxK=0
for k in range(len(Ks)):
    z=np.concatenate([y[:6],y[6:],[v]]); zn=np.concatenate([th[k],td[k],[vn[k]]])
    ac=a[k]-float((Ks[k]@(z-zn))[0]); maxK=max(maxK,np.abs(Ks[k]).max())
    y=rk4_step(ch,y,ac,dt); vnew=v+ac*dt; x+=0.5*(v+vnew)*dt; v=vnew
    TH.append(y[:6].copy()); XS.append(x); TS.append((k+1)*dt)
    if not np.isfinite(y).all(): log(f"DIVERGED at t={k*dt:.2f}"); sys.exit(1)
# hold upright a bit longer with the terminal gain for a clean ending
fe=np.max(np.abs(wrap(y[:6])))
log(f"reached upright: final|wrap|={fe:.2e} maxK={maxK:.0f}")
if fe<0.3:
    np.savez('results/trajectories/swingup_N6_perfectstate.npz',
             t=np.array(TS), theta=np.array(TH), x=np.array(XS))
    log("*** SAVED results/trajectories/swingup_N6_perfectstate.npz")
else:
    log("did not reach upright; not saved"); sys.exit(1)
