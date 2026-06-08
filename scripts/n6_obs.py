"""N=6 closed-loop swing-up at dt=0.004 with observer variants, to find one
that matches perfect-state performance. Perfect-state works at dt=0.004;
agent C's observer (dt=0.01/N<=5 tuned) fails. Try cleaner estimators."""
import sys, time; sys.path.insert(0,'.')
import numpy as np
from pendulum.dynamics import Chain
from pendulum.sim import simulate, rk4_step
from pendulum.protocol import wrap
from pendulum.swingup_traj import SwingupController, compute_tvlqr
T0=time.monotonic()
def log(m): print(f"[{time.monotonic()-T0:6.1f}s] {m}", flush=True)

g=9.81; T=15.0
data=dict(np.load('results/trajectories/swingup_N6.npz'))
ch=Chain(6,g)

class CleanObs(SwingupController):
    """Trust clean angle measurement directly; velocity from model-predict +
    strong angle-innovation correction (Luenberger-style, retuned)."""
    def __init__(self,*a,la=1.0,lv=1.0,**k):
        super().__init__(*a,**k); self.la=la; self.lv=lv
    def _estimate(self, theta_meas):
        if self.theta_cont is None or self._first:
            self.theta_cont=np.array(theta_meas,float); self.thetad_est=np.zeros(self.n)
            self._first=False
            return self.theta_cont.copy(), self.thetad_est.copy()
        y=np.concatenate([self.theta_cont,self.thetad_est])
        y=rk4_step(self.chain,y,self.last_a,self.dt)
        th_pred,thd_pred=y[:self.n],y[self.n:]
        # unwrap measurement near prediction
        innov=((theta_meas-th_pred+np.pi)%(2*np.pi))-np.pi
        self.theta_cont=th_pred+self.la*innov
        self.thetad_est=thd_pred+self.lv*innov/self.dt
        return self.theta_cont.copy(), self.thetad_est.copy()

# need _first flag
for dt in [0.004, 0.002]:
    for la,lv in [(1.0,1.0),(1.0,0.5),(0.8,0.3),(1.0,0.8)]:
        ctrl=CleanObs(ch,dt,0.0,0.0,'results/trajectories/swingup_N6.npz')
        ctrl._first=True; ctrl.la=la; ctrl.lv=lv
        y0=np.zeros(12); y0[:6]=np.pi
        res=simulate(ch,ctrl,y0,dt,int((T+20)/dt),record=True)
        th=res['traj']['y'][:,:6]; fin=np.isfinite(th).all(); tail=int(5/dt)
        ok=bool(fin and np.all(np.abs(wrap(th[-tail:]))<0.3))
        fe=np.max(np.abs(wrap(th[-1]))) if fin else float('nan')
        log(f"dt={dt} la={la} lv={lv}: ok={ok} caught={ctrl.caught} final|wrap|={fe:.2e}")
        if ok:
            np.savez('results/trajectories/swingup_N6_cl.npz',t=res['traj']['t'],theta=th,x=res['traj']['x'])
            log(f"*** SAVED N=6 swing-up dt={dt} la={la} lv={lv}"); sys.exit(0)
log("none worked - observer needs more work")
