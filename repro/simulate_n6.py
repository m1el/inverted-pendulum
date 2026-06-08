#!/usr/bin/env python
"""Minimal reproduction (2/2): ingest the controls, simulate, verify, animate.

Loads repro/n6_controls.npz (from optimize_n6.py), runs the closed-loop
swing-up under FULL-STATE feedback using the verified RK4 simulator and the
saved TVLQR gain schedule, asserts the chain reaches upright, and renders an
mp4 video.

Run:  uv run python repro/simulate_n6.py [out.mp4]
Exit code 0 and "VERIFICATION: PASS" iff the 6-link chain swings up and balances.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step

CONTROLS = "repro/n6_controls.npz"
OUT = sys.argv[1] if len(sys.argv) > 1 else "repro/n6_swingup.mp4"


def simulate(c):
    """Full-state closed loop: a = a_ff + K (z_nom - z). Returns t, theta, x."""
    n, dt, g = int(c["n"]), float(c["dt"]), float(c["g"])
    chain = Chain(n, g)
    theta, thetad = c["theta_nom"], c["thetad_nom"]
    a_ff, v_nom, Ks = c["a_ff"], c["v_nom"], c["K"]

    y = np.concatenate([theta[0], thetad[0]]); v = 0.0; x = 0.0
    T, X, TS = [y[:n].copy()], [x], [0.0]
    for k in range(len(Ks)):
        z = np.concatenate([y[:n], y[n:], [v]])
        zn = np.concatenate([theta[k], thetad[k], [v_nom[k]]])
        # TVLQR law: u = a_ff - K (z - z_nom)  (same convention as optimize_n6.py)
        a = a_ff[k] - float((Ks[k] @ (z - zn))[0])
        ynew = rk4_step(chain, y, a, dt)
        vnew = v + a * dt; x += 0.5 * (v + vnew) * dt
        y, v = ynew, vnew
        T.append(y[:n].copy()); X.append(x); TS.append((k + 1) * dt)
        if not np.isfinite(y).all():
            raise SystemExit(f"simulation diverged at t={k*dt:.2f}s")
    return np.array(TS), np.array(T), np.array(X)


def verify(theta, dt):
    n = theta.shape[1]
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    tail = int(round(5.0 / dt))
    upright = np.all(np.abs(wrap(theta[-tail:])) < 0.10)  # within ~6 deg for last 5 s
    final = float(np.max(np.abs(wrap(theta[-1]))))
    return bool(upright), final


def animate(t, theta, x, out, fps=30):
    n = theta.shape[1]
    chain = Chain(n)
    # subsample frames to ~fps real-time
    step = max(1, int(round((1.0 / fps) / (t[1] - t[0]))))
    idx = np.arange(0, len(t), step)
    fig, ax = plt.subplots(figsize=(5, 5))
    (line,) = ax.plot([], [], "o-", lw=2.5, ms=4)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5); ax.set_aspect("equal")
    lim = n + 0.5; ax.set_ylim(-lim, lim)

    def upd(j):
        i = idx[j]
        pts = chain.joint_positions(theta[i], x[i])
        line.set_data(pts[:, 0], pts[:, 1])
        ax.set_xlim(x[i] - lim, x[i] + lim)
        txt.set_text(f"t = {t[i]:5.2f} s   (N={n}, full-state feedback)")
        return line, txt

    anim = FuncAnimation(fig, upd, frames=len(idx), blit=False)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400,
                          extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out, writer=writer)
    plt.close(fig)


def main():
    if not pathlib.Path(CONTROLS).exists():
        raise SystemExit(f"missing {CONTROLS}; run repro/optimize_n6.py first")
    c = dict(np.load(CONTROLS))
    t, theta, x = simulate(c)
    ok, final = verify(theta, float(c["dt"]))
    print(f"final |angle| = {np.degrees(final):.3f} deg over all 6 links")
    print(f"VERIFICATION: {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)
    animate(t, theta, x, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
