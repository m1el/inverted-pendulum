"""Read results/balance.json and emit results/balance_notes.md (human summary).
Run AFTER scripts/sweep_balance.py has populated balance.json.
"""
import sys, pathlib, json, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results" / "balance.json").read_text())

def gm(b):
    lo, hi = b
    if hi is None or hi == float("inf") or (isinstance(hi, float) and math.isinf(hi)):
        return lo
    if lo == 0.0:
        return 0.0
    return (lo * hi) ** 0.5

def best_ctrl(n):
    sc = {}
    for c in ["FD", "Kalman"]:
        r = d[f"N{n}|{c}|dt0.01|g9.81"]
        sc[c] = gm(r["max_dtheta"]) * gm(r["max_dv"])
    return max(sc, key=sc.get)

lines = []
A = lines.append
A("# Balance precision (quantization) thresholds\n")
A("Task: keep N-link pendulum upright (theta=0). Success = all |theta_i|<0.5 rad")
A("for 60 s across 8 seeds (init theta ~ U(-5e-4,5e-4)). Precision model: angle")
A("measurements rounded to grid `dtheta`, pivot-velocity command rounded to grid")
A("`dv`. Threshold = largest grid step still succeeding (log-bisection bracket;")
A("headline number is the geometric mean of the bracket).\n")
A("Controllers: **FD** = discrete LQR + finite-difference velocity estimate;")
A("**Kalman** = discrete LQR + steady-state Kalman filter (knows dtheta/dv).")
A("LQR weights tuned per N over r in {0.01,0.1,1}, q_theta in {1,10,100}.\n")

# Headline table
A("## Headline (g=9.81, dt=0.01): best controller per N\n")
A("| N | best ctrl | max dtheta (rad) | max dv (m/s) | LQR (q_theta, r) |")
A("|---|-----------|------------------|--------------|------------------|")
for n in range(1, 6):
    c = best_ctrl(n)
    r = d[f"N{n}|{c}|dt0.01|g9.81"]
    kw = r["lqr_kwargs"]
    A(f"| {n} | {c} | {gm(r['max_dtheta']):.3g} | {gm(r['max_dv']):.3g} | "
      f"({kw['q_theta']:g}, {kw['r']:g}) |")
A("")

# Both controllers side by side
A("## Both controllers (g=9.81, dt=0.01)\n")
A("| N | FD max dtheta | KF max dtheta | FD max dv | KF max dv |")
A("|---|---------------|---------------|-----------|-----------|")
for n in range(1, 6):
    rf = d[f"N{n}|FD|dt0.01|g9.81"]
    rk = d[f"N{n}|Kalman|dt0.01|g9.81"]
    A(f"| {n} | {gm(rf['max_dtheta']):.3g} | {gm(rk['max_dtheta']):.3g} | "
      f"{gm(rf['max_dv']):.3g} | {gm(rk['max_dv']):.3g} |")
A("")

# Joint trade-off
A("## Joint trade-off (best controller, g=9.81, dt=0.01)\n")
A("dv threshold when dtheta is held at a fraction of its own max. 'inf' means")
A("dv was never the binding constraint in [1e-6, 1.0].\n")
A("| N | ctrl | dtheta=maxdth/10 -> max dv | dtheta=maxdth/3 -> max dv |")
A("|---|------|----------------------------|---------------------------|")
for n in range(1, 6):
    c = best_ctrl(n)
    r = d[f"N{n}|{c}|dt0.01|g9.81"]
    j = r["joint"]
    def dvstr(jb):
        b = jb["dv_bracket"]
        hi = b[1]
        if hi is None or hi == float("inf") or (isinstance(hi, float) and math.isinf(hi)):
            return f">={b[0]:.3g} (not binding)"
        return f"{gm(b):.3g}"
    A(f"| {n} | {c} | {dvstr(j[0])} | {dvstr(j[1])} |")
A("")

# dt scaling
A("## dt scaling (best controller per N, g=9.81)\n")
A("max dtheta and max dv at dt in {0.002, 0.005, 0.01, 0.02}.\n")
dts = [0.002, 0.005, 0.01, 0.02]
A("| N | ctrl | metric | " + " | ".join(f"dt={x}" for x in dts) + " | scaling |")
A("|---|------|--------|" + "|".join(["-------"] * len(dts)) + "|---------|")
for n in range(1, 6):
    c = best_ctrl(n)
    for metric in ["max_dtheta", "max_dv"]:
        vals = []
        for x in dts:
            key = f"N{n}|{c}|dt{x}|g9.81"
            if key in d and metric in d[key]:
                vals.append(gm(d[key][metric]))
            else:
                vals.append(None)
        # estimate scaling exponent from smallest & largest dt available
        pts = [(x, v) for x, v in zip(dts, vals) if v and v > 0]
        scal = ""
        if len(pts) >= 2:
            (x0, v0), (x1, v1) = pts[0], pts[-1]
            p = math.log(v1 / v0) / math.log(x1 / x0)
            scal = f"~dt^{p:.2f}"
        vstr = " | ".join(f"{v:.3g}" if v else "-" for v in vals)
        A(f"| {n} | {c} | {metric} | {vstr} | {scal} |")
A("")

# g scaling
A("## g scaling (best controller, dt=0.01, N in {1,3,5})\n")
gs = [4.905, 9.81, 19.62]
A("| N | ctrl | metric | " + " | ".join(f"g={x}" for x in gs) + " | scaling |")
A("|---|------|--------|" + "|".join(["-------"] * len(gs)) + "|---------|")
for n in [1, 3, 5]:
    c = best_ctrl(n)
    for metric in ["max_dtheta", "max_dv"]:
        vals = []
        for x in gs:
            key = f"N{n}|{c}|dt0.01|g{x}"
            if key in d and metric in d[key]:
                vals.append(gm(d[key][metric]))
            else:
                vals.append(None)
        pts = [(x, v) for x, v in zip(gs, vals) if v and v > 0]
        scal = ""
        if len(pts) >= 2:
            (x0, v0), (x1, v1) = pts[0], pts[-1]
            p = math.log(v1 / v0) / math.log(x1 / x0)
            scal = f"~g^{p:.2f}"
        vstr = " | ".join(f"{v:.3g}" if v else "-" for v in vals)
        A(f"| {n} | {c} | {metric} | {vstr} | {scal} |")
A("")

out = ROOT / "results" / "balance_notes.md"
out.write_text("\n".join(lines) + "\n")
print("wrote", out)
print("\n".join(lines))
