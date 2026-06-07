"""Swing-up controller for the single-link pendulum (N=1).

Strategy
--------
1. Energy pumping. The rod energy about the (resting) pivot is

       E = (1/6) thetad^2 + (g/2) cos(theta),   E* = g/2 (upright).

   The pivot acceleration a is chosen to inject energy:

       a = sat( k_E * (E - E*) * thetad * cos(theta) )  -  k_v * v  -  k_x * x

   The first term pumps/dumps energy toward E*; the -k_v*v, -k_x*x terms keep the
   cart from drifting (they vanish near the desired stationary cart state and do
   not interfere with pumping which is dominated by the saturated first term).

   Sign check (sat positive): increasing E requires a*cos(theta) opposing the
   gravity torque appropriately; dE/dt = -a*thetad*cos(theta) for this system,
   so to *increase* E we want a opposite in sign to thetad*cos(theta).  Hence the
   pumping acceleration is a_pump = -k_E*(E-E*)*thetad*cos(theta): when E<E* (need
   more energy) and thetad*cos(theta)>0, push a negative. We fold the sign into
   k_E below.

2. Catch. When the angle is within the capture region and the angular rate is
   modest, hand off to an LQR balancer (KalmanBalancer on the linearized upright
   model).  Hysteresis (a wider release region) prevents chatter.

The controller only ever sees the quantized angle measurement; thetad and E are
estimated by finite differences of the (unwrapped) angle.
"""

from __future__ import annotations

import numpy as np

from .dynamics import Chain
from .balance import KalmanBalancer


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SwingUp1:
    """Energy-pumping swing-up + LQR catch for N=1.

    Callable: (theta_meas, t, v, x) -> v_cmd, compatible with sim.simulate().
    """

    def __init__(
        self,
        chain: Chain,
        dt: float,
        dtheta: float = 0.0,
        dv: float = 0.0,
        k_E: float = 1.0,
        k_v: float = 0.5,
        k_x: float = 0.1,
        a_max: float = 30.0,
        catch_angle: float = 0.30,
        catch_rate: float = 3.0,
        release_angle: float = 0.55,
        filt: float = 0.5,
    ):
        assert chain.n == 1, "SwingUp1 is for the single-link pendulum only"
        self.chain = chain
        self.dt = dt
        self.dtheta = dtheta
        self.dv = dv
        self.g = chain.g
        self.Estar = self.g / 2.0

        self.k_E = k_E
        self.k_v = k_v
        self.k_x = k_x
        self.a_max = a_max
        self.catch_angle = catch_angle
        self.catch_rate = catch_rate
        self.release_angle = release_angle
        self.filt = filt  # low-pass factor for thetad estimate (1 = no filter)

        # LQR catcher on the linearized upright model.
        self.balancer = KalmanBalancer(chain, dt, dtheta=dtheta, dv=dv)

        self.reset()

    def reset(self):
        self.prev_unwrapped = None
        self.thetad_est = 0.0
        self.catching = False
        self.balancer.reset()
        self.last_v_cmd = 0.0

    def _estimate(self, theta_meas):
        """Unwrap the measured angle relative to the running estimate and update
        the finite-difference angular-rate estimate."""
        th = float(theta_meas[0])
        if self.prev_unwrapped is None:
            unwrapped = th
            self.prev_unwrapped = unwrapped
            self.thetad_est = 0.0
            return unwrapped, 0.0
        # bring the new measurement into the same branch as the previous estimate
        delta = _wrap(th - self.prev_unwrapped)
        unwrapped = self.prev_unwrapped + delta
        raw_rate = (unwrapped - self.prev_unwrapped) / self.dt
        self.thetad_est = (
            self.filt * raw_rate + (1.0 - self.filt) * self.thetad_est
        )
        self.prev_unwrapped = unwrapped
        return unwrapped, self.thetad_est

    def __call__(self, theta_meas, t, v, x):
        theta, thetad = self._estimate(theta_meas)
        th_w = _wrap(theta)  # wrapped angle in (-pi, pi], 0 = upright

        # --- hysteretic mode switch ---
        if self.catching:
            if abs(th_w) > self.release_angle:
                self.catching = False
                self.balancer.reset()
        else:
            if abs(th_w) < self.catch_angle and abs(thetad) < self.catch_rate:
                self.catching = True
                # seed the balancer's filter with current estimate
                self.balancer.zhat = np.array([th_w, thetad])
                self.balancer.last_a = (self.last_v_cmd - v) / self.dt

        if self.catching:
            # Hand the LQR the *wrapped* angle so its small-angle model is valid.
            v_cmd = self.balancer(np.array([th_w]), t, v, x)
            self.last_v_cmd = v_cmd
            return v_cmd

        # --- energy pumping ---
        E = (1.0 / 6.0) * thetad**2 + (self.g / 2.0) * np.cos(th_w)
        # a_pump increases E: dE/dt = -a*thetad*cos(theta).  To drive E -> E*:
        #   want sign(a) = sign((E - E*) * thetad * cos(theta))
        a_pump = self.k_E * (E - self.Estar) * thetad * np.cos(th_w)
        a_pump = float(np.clip(a_pump, -self.a_max, self.a_max))
        # cart-regulating terms (do not saturate these so they always act)
        a = a_pump - self.k_v * v - self.k_x * x
        a = float(np.clip(a, -self.a_max, self.a_max))

        v_cmd = v + a * self.dt
        self.last_v_cmd = v_cmd
        return v_cmd
