"""N=6 dt=0.004 with LOWER-gain TVLQR (higher r) + clean observer.
Hypothesis: small dt keeps it in the funnel even with gentler gains, and
gentler gains tolerate observer error. Also test perfect-state at each r to
see how low the gains can go and still track."""
import sys, time; sys.path.insert(0,'.')
import numpy as np
from pendulum.dynamics import Chain
from pendulum.sim import simulate, rk4_step
from pendulum.protocol import wrap
from pendulum.swingup_traj import SwingupController
T0=time.monotonic()
def log(m): print(f"[{time.monotonic()-T0:6.1f}s] {m}", flush=True)
g=9.81; T=15.0; ch=Chain(6,g)

class CleanObs(SwingupController):
    def _estimate(self, theta_meas):
        if getattr(self,'_first',True):
            self.theta_cont=np.array(theta_meas,float); self.thetad_est=np.zeros(self.n); self._first=False
            return self.theta_cont.copy(), self.thetad_est.copy()
        y=rk4_step(self.chain,np.concatenate([self.theta_cont,self.thetad_est]),self.last_a,self.dt)
        th_pred,thd_pred=y[:self.n],y[self.n:]
        innov=((theta_meas-th_pred+np.pi)%(2*np.pi))-np.pi
        self.theta_cont=th_pred+innov          # la=1 trust clean meas
        self.thetad_est=thd_pred+0.5*innov/self.dt
        return self.theta_cont.copy(), self.thetad_est.copy()

dt=0.004
for r in [0.1, 1.0, 5.0, 20.0, 100.0]:
    kw=dict(q_theta=50,q_thetad=5,q_v=1,r=r,qf_theta=2000,qf_thetad=200,qf_v=10)
    # perfect-state first
    from pendulum.swingup_traj import compute_tvlqr
    data=dict(np.load('results/trajectories/swingup_N6.npz'))
    Ks=compute_tvlqr(ch,data,dt,**kw)
    tn=np.arange(0,T+1e-9,dt)
    th=np.vstack([np.interp(tn,data['t'],data['theta'][:,i]) for i in range(6)]).T
    td=np.vstack([np.interp(tn,data['t'],data['thetad'][:,i]) for i in range(6)]).T
    a=np.interp(tn,data['t'],data['a']); vn=np.interp(tn,data['t'],data['v'])
    y=np.concatenate([th[0],td[0]]); v=0.0; maxK=0; div=None
    for k in range(len(Ks)):
        z=np.concatenate([y[:6],y[6:],[v]]); zn=np.concatenate([th[k],td[k],[vn[k]]])
        ac=a[k]-float((Ks[k]@(z-zn))[0]); maxK=max(maxK,np.abs(Ks[k]).max())
        y=rk4_step(ch,y,ac,dt); v+=ac*dt
        if not np.isfinite(y).all(): div=k*dt; break
    ps_ok = np.isfinite(y).all() and np.max(np.abs(wrap(y[:6])))<0.3
    # observer
    ctrl=CleanObs(ch,dt,0.0,0.0,'results/trajectories/swingup_N6.npz',tvlqr_kw=kw); ctrl._first=True
    y0=np.zeros(12); y0[:6]=np.pi
    res=simulate(ch,ctrl,y0,dt,int((T+20)/dt),record=True)
    tth=res['traj']['y'][:,:6]; fin=np.isfinite(tth).all(); tail=int(5/dt)
    obs_ok=bool(fin and np.all(np.abs(wrap(tth[-tail:]))<0.3))
    log(f"r={r:5.1f} maxK={maxK:7.0f} perfect_state={'OK' if ps_ok else 'FAIL'} observer={'OK' if obs_ok else 'FAIL'}")
    if obs_ok:
        np.savez('results/trajectories/swingup_N6_cl.npz',t=res['traj']['t'],theta=tth,x=res['traj']['x'])
        log(f"*** SAVED N=6 swing-up r={r}"); sys.exit(0)
log("done")
