"""Render a pendulum trajectory: GIF animation + filmstrip PNG.

Usage:
  uv run python scripts/animate.py results/trajectories/swingup_N3.npz out.gif
  (npz with arrays: t, theta [T,n], optionally x [T])
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from pendulum.dynamics import Chain


def load(path):
    d = np.load(path)
    t = d["t"]
    theta = d["theta"]
    x = d["x"] if "x" in d else np.zeros(len(t))
    return t, theta, x


def filmstrip(t, theta, x, out_png, n_frames=12):
    n = theta.shape[1]
    chain = Chain(n)
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    fig, axes = plt.subplots(1, n_frames, figsize=(2 * n_frames, 2.6), sharey=True)
    for ax, i in zip(axes, idx):
        pts = chain.joint_positions(theta[i], x[i])
        ax.plot(pts[:, 0], pts[:, 1], "o-", lw=2, ms=3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlim(x[i] - n - 0.5, x[i] + n + 0.5)
        ax.set_ylim(-n - 0.3, n + 0.3)
        ax.set_title(f"t={t[i]:.1f}s", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def gif(t, theta, x, out_gif, fps=25, speed=1.0):
    n = theta.shape[1]
    chain = Chain(n)
    dt_frame = speed / fps
    idx = np.searchsorted(t, np.arange(t[0], t[-1], dt_frame))
    idx = np.clip(idx, 0, len(t) - 1)
    fig, ax = plt.subplots(figsize=(5, 5))
    (line,) = ax.plot([], [], "o-", lw=2.5, ms=4)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    lim = n + 0.5
    ax.set_ylim(-lim, lim)

    def update(k):
        i = idx[k]
        pts = chain.joint_positions(theta[i], x[i])
        line.set_data(pts[:, 0], pts[:, 1])
        ax.set_xlim(x[i] - lim, x[i] + lim)
        txt.set_text(f"t = {t[i]:5.2f} s")
        return line, txt

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    t, theta, x = load(src)
    if out.endswith(".gif"):
        gif(t, theta, x, out)
    else:
        filmstrip(t, theta, x, out)
    print(f"wrote {out}")
