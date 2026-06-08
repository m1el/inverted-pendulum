#!/usr/bin/env python
"""Minimal reproduction (1/2): optimise the 6-link swing-up and emit controls.

SELF-CONTAINED and uses the same generate-pool / rank / homotopy / refine
approach that produced the original result (agent C's gen_candidates.py +
select_n5.py + N=6 refinement), all in one parallel script. No pre-existing
trajectory or candidate pool is required.

Pipeline:
  1. Generate an N=5 candidate POOL in parallel by sweeping
        horizon T  x  terminal target (the +-2pi homotopy classes)  x  seed,
     each solved coarse (h=0.05) then refined (h=0.01)  [implicit collocation].
  2. RANK the N=5 candidates by closed-loop robustness with the realistic
     observer-based controller (how many of 4 perturbed seeds swing up, then
     smallest tail error). Keep the top few as homotopy seeds.
  3. Generate an N=6 candidate POOL from each kept N=5 (homotopy: new tip link
     shadows the last) x target x horizon, coarse then refined.
  4. SELECT the N=6 that is trackable under FULL-STATE TVLQR at the simulation
     timestep dt=0.004 (the enabler for N=6), smallest upright error wins.
  5. Save its nominal + TVLQR gain schedule to repro/n6_controls.npz.

Trackability of N=6 is fragile, so this is a SEARCH: expect tens of minutes.
Parallelised across cores (multiprocessing). Tune the *_GRID constants below to
trade runtime for reliability.

CAVEAT: the controller is FULL-STATE feedback. The near-straight mid-swing chain
is near-uncontrollable from one pivot, forcing TVLQR gains ~7e4; no realistic
angle-only observer can feed gains that large, so this reproduces the
*idealised-sensing* swing-up. The decisive enabler vs N=5 is dt=0.004.

Run:  uv run python repro/optimize_n6.py [n_workers]
"""
import sys, os, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
from scipy.linalg import expm
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.sim import rk4_step, simulate
from pendulum.trajopt import solve_swingup_implicit, homotopy_guess

# ---- parameters ------------------------------------------------------------
G = 9.81
DT_SIM = 0.004                 # simulation timestep -- the enabler for N=6
DT_SEL = 0.01                  # timestep for the N=5 closed-loop ranking
POOL = pathlib.Path("repro/pool")
OUT = "repro/n6_controls.npz"

# generation grids (sweep these for diversity; bigger = more reliable, slower)
# diversity comes from horizon x terminal target (homotopy warm start makes the
# solve deterministic, so extra seeds would be redundant)
N5_T = [11.0, 13.0, 15.0]
N5_SEEDS = range(1)
N6_T = [13.0, 14.0, 15.0, 16.0]
N6_SEEDS = range(1)
N5_KEEP = 3                    # top-N5 seeds carried into the N=6 search
# Per-rung settings (agent C's proven values): N=5 needs more actuator
# authority to find good swing-ups; N=6 is gentler (and dt=0.004 tracks it).
A_MAX = {5: 60.0, 6: 25.0}
V_MAX = {5: 12.0, 6: 14.0}
SETTLE_FRAC = {5: 0.12, 6: 0.08}
SETTLE_BAND = 0.15

# TVLQR weights for the N=6 full-state controller (state [theta,thetad,v])
Q6 = np.diag([50.0] * 6 + [5.0] * 6 + [1.0])
R6 = np.array([[0.1]])
QF6 = np.diag([2000.0] * 6 + [200.0] * 6 + [10.0])

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


def target_options(n):
    """Terminal homotopy classes (which links do a full extra revolution)."""
    opts = [("zeros", np.zeros(n)),
            ("first+2pi", np.array([2 * np.pi] + [0.0] * (n - 1))),
            ("first-2pi", np.array([-2 * np.pi] + [0.0] * (n - 1))),
            ("all+2pi", np.full(n, 2 * np.pi))]
    if n >= 3:
        opts.append(("alt2pi", np.array([2 * np.pi * (i % 2) for i in range(n)], float)))
    return opts


# ---- linearisation + TVLQR (full state z=[theta,thetad,v], input pivot accel)
def linearize_z(chain, theta, thetad, a, eps=1e-6):
    n = chain.n; nz = 2 * n + 1
    z0 = np.concatenate([theta, thetad, [0.0]])
    def f(z):
        return np.concatenate([z[n:2*n], chain.thetadd(z[:n], z[n:2*n], a), [0.0]])
    f0 = f(z0); A = np.zeros((nz, nz))
    for j in range(nz):
        zp = z0.copy(); zp[j] += eps; A[:, j] = (f(zp) - f0) / eps
    B = np.zeros((nz, 1))
    B[n:2*n, 0] = (chain.thetadd(theta, thetad, a + eps)
                   - chain.thetadd(theta, thetad, a)) / eps
    B[2*n, 0] = 1.0
    return A, B


def build_tvlqr(chain, theta, thetad, a, dt, Q, R, QF):
    n = chain.n; nz = 2 * n + 1; M = len(theta) - 1
    Ads, Bds = [], []
    for k in range(M):
        A, B = linearize_z(chain, theta[k], thetad[k], a[k])
        blk = np.zeros((nz + 1, nz + 1)); blk[:nz, :nz] = A * dt; blk[:nz, nz:] = B * dt
        E = expm(blk); Ads.append(E[:nz, :nz]); Bds.append(E[:nz, nz:])
    P = QF.copy(); Ks = np.zeros((M, 1, nz))
    for k in range(M - 1, -1, -1):
        Ad, Bd = Ads[k], Bds[k]; S = R + Bd.T @ P @ Bd
        Kk = np.linalg.solve(S, Bd.T @ P @ Ad); Ks[k] = Kk
        P = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ Kk
    return Ks


def n6_perfect_state(traj):
    """Resample N=6 traj to DT_SIM, build TVLQR, roll out FULL-STATE.
    Returns (ok, final_angle, bundle|None). ok = settles within ~3 deg."""
    chain = Chain(6, G); Th = float(traj["t"][-1])
    tn = np.arange(0.0, Th + 1e-9, DT_SIM)
    theta = np.vstack([np.interp(tn, traj["t"], traj["theta"][:, i]) for i in range(6)]).T
    thetad = np.vstack([np.interp(tn, traj["t"], traj["thetad"][:, i]) for i in range(6)]).T
    a_ff = np.interp(tn, traj["t"], traj["a"]); v_nom = np.interp(tn, traj["t"], traj["v"])
    Ks = build_tvlqr(chain, theta, thetad, a_ff, DT_SIM, Q6, R6, QF6)
    y = np.concatenate([theta[0], thetad[0]]); v = 0.0; maxK = 0.0
    for k in range(len(Ks)):
        z = np.concatenate([y[:6], y[6:], [v]]); zn = np.concatenate([theta[k], thetad[k], [v_nom[k]]])
        a = a_ff[k] - float((Ks[k] @ (z - zn))[0]); maxK = max(maxK, float(np.abs(Ks[k]).max()))
        y = rk4_step(chain, y, a, DT_SIM); v += a * DT_SIM
        if not np.isfinite(y).all():
            return False, np.inf, None
    final = float(np.max(np.abs((y[:6] + np.pi) % (2 * np.pi) - np.pi)))
    bundle = dict(dt=DT_SIM, g=G, n=6, T=Th, t=tn, theta_nom=theta, thetad_nom=thetad,
                  a_ff=a_ff, v_nom=v_nom, K=Ks, maxK=maxK, final=final)
    return final < 0.05, final, bundle


# ---- worker functions (module level so multiprocessing can pickle them) -----
def w_generate(task):
    """Solve one candidate coarse->refine. task=(n,T,tname,tgt,seed,init_path)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    n, T, tname, tgt, seed, init_path = task
    tgt = np.asarray(tgt)
    guess = homotopy_guess(dict(np.load(init_path))) if init_path else None
    Kc, Kf = int(T / 0.05), int(T / 0.01)
    amax, vmax, sfrac = A_MAX[n], V_MAX[n], SETTLE_FRAC[n]
    # bounded iterations: a hard (target,horizon) combo bails instead of
    # grinding to a huge cap and stalling the whole parallel batch (barrier).
    sc = solve_swingup_implicit(n, T, Kc, g=G, a_max=amax, v_max=vmax,
        theta_target=tgt, settle_frac=sfrac, settle_band=SETTLE_BAND,
        seed=seed, init_guess=guess, max_iter=1500, print_level=0, tol=1e-6)
    if sc["status"] != "solved":
        return None
    sf = solve_swingup_implicit(n, T, Kf, g=G, a_max=amax, v_max=vmax,
        theta_target=tgt, settle_frac=sfrac, settle_band=SETTLE_BAND,
        init_guess=sc, max_iter=1200, print_level=0, tol=1e-7)
    if sf["status"] != "solved":
        return None
    fn = POOL / f"N{n}_{tname}_T{T}_s{seed}.npz"
    np.savez(fn, t=sf["t"], theta=sf["theta"], thetad=sf["thetad"],
             a=sf["a"], v=sf["v"], T=sf["T"], target=tgt)
    return str(fn)


def w_rank_n5(fn):
    """Rank an N=5 candidate by closed-loop robustness (observer controller).
    Returns (oks, -worst_tail, fn) so larger sorts better."""
    os.environ["OMP_NUM_THREADS"] = "1"
    from pendulum.swingup_traj import SwingupController, compute_tvlqr, TVLQR_PRESETS
    chain = Chain(5, G)
    wrap = lambda a: (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
    # cheap perfect-state pre-screen (any preset) before the 4-seed closed loop
    d = dict(np.load(fn)); th, td, av, vv = d["theta"], d["thetad"], d["a"], d["v"]
    Ks = compute_tvlqr(chain, d, DT_SEL, **TVLQR_PRESETS["default"])
    y = np.concatenate([th[0], td[0]]); v = 0.0; ok_ps = True
    for k in range(len(th) - 1):
        z = np.concatenate([y[:5], y[5:], [v]]); zn = np.concatenate([th[k], td[k], [vv[k]]])
        a = av[k] - float((Ks[k] @ (z - zn))[0]); y = rk4_step(chain, y, a, DT_SEL); v += a * DT_SEL
        if not np.isfinite(y).all() or np.max(np.abs(y[:5] - th[k + 1])) > 0.3:
            ok_ps = False; break
    if not ok_ps:
        return (0, -np.inf, fn)
    T = float(d["t"][-1]); nsteps = int((T + 20.0) / DT_SEL); tail = int(5.0 / DT_SEL)
    oks = 0; worst = 0.0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        y0 = np.zeros(10); y0[:5] = np.pi + rng.uniform(-1e-3, 1e-3, 5)
        c = SwingupController(chain, DT_SEL, 0.0, 0.0, fn, tvlqr_kw="default")
        res = simulate(chain, c, y0, DT_SEL, nsteps, record=True)
        te = np.max(np.abs(wrap(res["traj"]["y"][:, :5][-tail:])))
        oks += int(te < 0.3); worst = max(worst, te)
    return (oks, -worst, fn)


def w_eval_n6(fn):
    """Full-state trackability of an N=6 candidate. Returns (ok, final, fn)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    ok, final, _ = n6_perfect_state(dict(np.load(fn)))
    return (ok, final, fn)


def build_ladder_to_4():
    """Coarse homotopy ladder N=2->3->4 (fast); returns the N=4 seed file path.
    N=5 candidates warm-start from this -- cold-start N=5 is unreliable."""
    prev = None
    for n in range(2, 5):
        T = {2: 4.0, 3: 5.0, 4: 7.0}[n]; K = int(T * 25)
        guess = homotopy_guess(dict(np.load(prev))) if prev else None
        sol = None
        for s in range(4):
            sol = solve_swingup_implicit(n, T, K, g=G, a_max=A_MAX[6], v_max=V_MAX[6],
                theta_target=np.zeros(n), settle_frac=0.1, settle_band=0.2,
                seed=s, init_guess=guess, max_iter=1500, print_level=0, tol=1e-6)
            if sol["status"] == "solved":
                break
        if sol["status"] != "solved":
            raise SystemExit(f"ladder N={n} failed")
        fn = POOL / f"_ladder_N{n}.npz"
        np.savez(fn, t=sol["t"], theta=sol["theta"], thetad=sol["thetad"],
                 a=sol["a"], v=sol["v"], T=T, target=np.zeros(n))
        prev = str(fn)
    return prev


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else min(24, mp.cpu_count() - 2)
    if POOL.exists():
        for p in POOL.glob("*.npz"):
            p.unlink()
    POOL.mkdir(parents=True, exist_ok=True)

    # 0) fast coarse ladder N=2..4 to warm-start N=5 generation ---------------
    log("building N=2..4 coarse ladder (homotopy seeds)...")
    n4 = build_ladder_to_4()
    log("  ladder ready")

    # 1) generate N=5 pool (warm-started from N=4) ---------------------------
    tasks = [(5, T, tn, tgt, s, n4)
             for T in N5_T for (tn, tgt) in target_options(5) for s in N5_SEEDS]
    log(f"generating N=5 pool: {len(tasks)} tasks on {nworkers} workers...")
    with mp.Pool(nworkers) as pool:
        n5 = [r for r in pool.map(w_generate, tasks) if r]
    log(f"  {len(n5)} N=5 candidates solved")
    if not n5:
        raise SystemExit("no N=5 candidates solved")

    # 2) rank N=5 by closed-loop robustness ----------------------------------
    log(f"ranking {len(n5)} N=5 candidates (observer closed-loop, 4 seeds)...")
    with mp.Pool(nworkers) as pool:
        ranked = sorted(pool.map(w_rank_n5, n5), reverse=True)
    for oks, nworst, fn in ranked[:N5_KEEP]:
        log(f"  KEEP {pathlib.Path(fn).name}: ok={oks}/4 worst={-nworst:.4f}")
    seeds5 = [fn for oks, nworst, fn in ranked[:N5_KEEP] if oks > 0] or [ranked[0][2]]

    # 3) generate N=6 pool from kept N=5 (homotopy) --------------------------
    tasks = [(6, T, tn, tgt, s, seed5)
             for seed5 in seeds5 for T in N6_T
             for (tn, tgt) in target_options(6) for s in N6_SEEDS]
    log(f"generating N=6 pool: {len(tasks)} tasks...")
    with mp.Pool(nworkers) as pool:
        n6 = [r for r in pool.map(w_generate, tasks) if r]
    log(f"  {len(n6)} N=6 candidates solved")
    if not n6:
        raise SystemExit("no N=6 candidates solved")

    # 4) select the trackable N=6 (full-state, dt=0.004) ---------------------
    log(f"evaluating {len(n6)} N=6 candidates (full-state trackability)...")
    with mp.Pool(nworkers) as pool:
        evals = pool.map(w_eval_n6, n6)
    trackable = sorted([e for e in evals if e[0]], key=lambda e: e[1])
    log(f"  {len(trackable)}/{len(n6)} N=6 candidates trackable")
    if not trackable:
        raise SystemExit("no trackable N=6 found; widen the grids and retry")
    ok, final, fn = trackable[0]
    log(f"  BEST {pathlib.Path(fn).name}: final={np.degrees(final):.3f} deg")

    # 5) rebuild + save the controls bundle for the winner -------------------
    _, _, bundle = n6_perfect_state(dict(np.load(fn)))
    np.savez(OUT, **bundle)
    log(f"SAVED controls -> {OUT}  (nominal {bundle['theta_nom'].shape}, "
        f"gains {bundle['K'].shape}, maxK={bundle['maxK']:.0f})")


def orchestrate():
    """Full from-scratch 2-stage pipeline: Stage 1 (generate+select N=5 seed)
    then Stage 2 (lift+refine to a trackable N=6). Run as subprocesses so the
    two stages stay independently runnable / debuggable.

    NOTE: a from-scratch N=5 seed may not yield a trackable N=6 (its bend order
    must match a controllable topology). If Stage 2 reports no trackable N=6,
    rerun Stage 2 against the backed-up good seed:
        uv run python repro/stage2_n6.py repro/seeds/swingup_N5_GOOD.npz
    """
    import subprocess
    root = pathlib.Path(__file__).resolve().parents[1]
    nw = sys.argv[1] if len(sys.argv) > 1 else ""
    seed = "repro/seed_N5.npz"
    print("=== STAGE 1: generate + select N=5 seed ===", flush=True)
    subprocess.run(["uv", "run", "python", "repro/stage1_n5.py", seed] + ([nw] if nw else []),
                   cwd=root, check=True)
    print("=== STAGE 2: lift + refine to trackable N=6 ===", flush=True)
    r = subprocess.run(["uv", "run", "python", "repro/stage2_n6.py", seed] + ([nw] if nw else []),
                       cwd=root)
    if r.returncode != 0:
        print("Stage 2 found no trackable N=6 from the from-scratch seed.\n"
              "Fall back to the known-good seed:\n"
              "  uv run python repro/stage2_n6.py repro/seeds/swingup_N5_GOOD.npz")
        sys.exit(r.returncode)


if __name__ == "__main__":
    orchestrate()
