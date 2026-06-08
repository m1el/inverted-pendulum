"""Plot TVLQR policy stiffness (gain magnitude) over time for a controls bundle.

Top panel: per-step max|K| and ||K||_F on a log axis (the policy "stiffness").
Bottom panel: max link angle from upright -- shows the gain plateau coincides
with the near-upright catch/hold, where stabilizing the inverted near-straight
chain (N unstable poles, one pivot) is intrinsically high-gain.

Usage: uv run python scripts/plot_stiffness.py repro/n6_controls.npz media/out.png
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "media/tvlqr_stiffness.png"
c = dict(np.load(src))
n = int(c["n"]); dt = float(c["dt"]); K = c["K"]; theta = c["theta_nom"]
M = K.shape[0]; t = np.arange(M) * dt
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

maxK = np.abs(K).max(axis=(1, 2))
frob = np.linalg.norm(K.reshape(M, -1), axis=1)
dist = np.abs(wrap(theta[:M])).max(axis=1)
near = dist < np.deg2rad(15)
t_arrive = t[np.argmax(near)] if near.any() else np.nan
hot = maxK > 1e4

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
ax1.semilogy(t, maxK, lw=1.4, color="#c0392b", label=r"$\max_i |K_i|$  (stiffness)")
ax1.semilogy(t, frob, lw=1.0, color="#e67e22", alpha=0.7, label=r"$\|K\|_F$")
ax1.axhline(1e4, ls="--", lw=0.8, color="gray"); ax1.text(t[0] + 0.1, 1.25e4, r"$|K|=10^4$", fontsize=8, color="gray")
if hot.any(): ax1.axvspan(t[hot].min(), t[hot].max(), color="#c0392b", alpha=0.07)
if np.isfinite(t_arrive): ax1.axvline(t_arrive, ls=":", color="#2c3e50", lw=1)
ax1.set_ylabel("TVLQR gain magnitude")
ax1.set_title(f"N={n} swing-up TVLQR stiffness over time  "
              f"(dt={dt}s, {M} steps, max|K|={maxK.max():.0f})")
ax1.legend(loc="upper left", fontsize=9); ax1.grid(alpha=0.25, which="both")

ax2.plot(t, np.degrees(dist), lw=1.4, color="#2c3e50")
ax2.fill_between(t, 0, np.degrees(dist), color="#2c3e50", alpha=0.08)
if hot.any(): ax2.axvspan(t[hot].min(), t[hot].max(), color="#c0392b", alpha=0.07)
if np.isfinite(t_arrive):
    ax2.axvline(t_arrive, ls=":", color="#2c3e50", lw=1)
    ax2.text(t_arrive + 0.1, 0.6 * np.degrees(dist).max(),
             f"reaches upright\n(t≈{t_arrive:.1f}s)", fontsize=8, color="#2c3e50")
ax2.set_ylabel("max link angle\nfrom upright (deg)"); ax2.set_xlabel("time (s)"); ax2.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(out, dpi=130)

sw = maxK[t < t_arrive] if np.isfinite(t_arrive) else maxK
hold = maxK[t >= t_arrive] if np.isfinite(t_arrive) else maxK[:0]
print(f"wrote {out}")
print(f"N={n} dt={dt} steps={M} nz={2*n+1}  max|K|={maxK.max():.0f}  median|K|={np.median(maxK):.0f}")
if hold.size:
    print(f"arrive upright t={t_arrive:.2f}s | median|K| swing={np.median(sw):.0f} "
          f"hold={np.median(hold):.0f} ({np.median(hold)/np.median(sw):.0f}x) | "
          f"frac>1e4={hot.mean():.2f}")
