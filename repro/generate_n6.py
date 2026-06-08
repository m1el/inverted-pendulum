#!/usr/bin/env python
"""Approach A: CONTROLLABILITY-AWARE trajectory optimization for N=6 swing-up.

GOAL: produce a TRACKABLE N=6 swing-up nominal WITHOUT the curated good N=5 seed
("no magic"). We start from a NEUTRAL homotopy ladder (target = zeros, fresh
cold ladder N=2..5 -> N=6) and add an explicit CONTROLLABILITY term to the
collocation so the optimizer keeps the chain in a controllable (trackable)
configuration by construction.

WHY THE BASELINE NEEDS MAGIC (from PROGRESS.md):
  Mid-swing, all 6 links pass through a near-STRAIGHT configuration. A straight
  chain is ~a single rigid body, so the internal BENDING modes become
  near-unactuatable from the one pivot input -> TVLQR gains explode (~7e4) and
  the closed loop diverges. Only trajectories whose "bend order" keeps the chain
  curved/shimmying (never dead-straight at the wrong moment) are trackable. The
  baseline gets that property only from a curated N=5 seed.

CONTROLLABILITY PROXY (principled, CasADi-cheap, well-scaled):
  The pivot input a enters the angular dynamics as  M(theta) tdd = -b*cos(theta)*a + ...
  so the INPUT-TO-ANGULAR-ACCELERATION direction is  u(theta) = M(theta)^{-1} (b o cos theta).
  Decompose u into the RIGID rotation mode (all links rotate together, direction
  1 = ones) and the BENDING complement, in the kinetic-energy (M) metric:
      u_rigid = (1^T M u)/(1^T M 1) * 1,   u_bend = u - u_rigid
  The controllability proxy at a node is the M-norm of the bending part:
      c(theta) = sqrt(u_bend^T M u_bend)
  This is exactly the "B-direction excitation of the bending modes" suggested by
  the task: how strongly the single input can directly drive the chain's
  internal shape. It collapses (-> small) precisely when the chain is straight
  (then u is almost pure rigid mode), and is O(1) and smooth otherwise. On the
  KNOWN-GOOD baseline nominal it never drops below ~0.97; on naive cold-start
  N=6 nominals it dips near 0 at the straight crossing. We REWARD it (soft) and/
  or FLOOR it (hard), driving the optimizer to a trackable bend order from a
  neutral start.

We sweep the soft weight w and the hard floor, solve each on neutral warm
starts, then test every result with repro.optimize_n6.n6_perfect_state at
dt=0.004 (full-state TVLQR). SUCCESS = ok=True. We report maxK and max|pivot
accel| (controllability-awareness should reduce both vs baseline maxK~69448).

Run:  uv run python repro/approachA_optimize.py [n_workers]
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warnings; warnings.filterwarnings("ignore")
import casadi as ca
import multiprocessing as mp

from pendulum.dynamics import Chain
from pendulum.trajopt import (chain_constants, make_M_rhs_fn,
                              solve_swingup_implicit, homotopy_guess)
from repro.optimize_n6 import n6_perfect_state

G = 9.81
N = 6
A_MAX, V_MAX = 25.0, 14.0
SETTLE_FRAC, SETTLE_BAND = 0.08, 0.15
POOL = pathlib.Path("repro/pool_ctrb")
OUT = "repro/n6_controls.npz"

_t0 = time.monotonic()
def log(m): print(f"[{time.monotonic()-_t0:7.1f}s] {m}", flush=True)


# ---- controllability proxy (CasADi) ---------------------------------------
def bend_excite_ca(theta, u, A):
    """Squared M-norm of the bending (non-rigid) part of the input-to-angular-
    acceleration direction u, where u solves  M(theta) u = b o cos(theta)
    (enforced as a separate IMPLICIT constraint, so NO symbolic matrix inverse
    appears here -> the whole NLP stays SX-expandable, unlike ca.solve which
    uses LinsolQr and breaks expand=True).

    With Mu = M u (computable, no inverse) the bending M-norm is
        ||u_bend||_M^2 = u^T M u - (1^T M u)^2 / (1^T M 1)
    because the rigid mode is 1=ones and the projection is M-orthogonal.
    theta,u: (n,1) MX. Returns scalar MX (well-scaled, O(1), smooth)."""
    n = A.shape[0]
    Aca = ca.DM(A); ones = ca.DM(np.ones((n, 1)))
    dth = ca.repmat(theta, 1, n) - ca.repmat(theta.T, n, 1)
    M = Aca * ca.cos(dth) + ca.DM(np.eye(n) / 12.0)
    Mu = ca.mtimes(M, u)
    uMu = ca.mtimes(u.T, Mu)                       # u^T M u
    oMu = ca.mtimes(ones.T, Mu)                    # 1^T M u
    oMo = ca.mtimes(ones.T, ca.mtimes(M, ones))    # 1^T M 1
    return uMu - oMu * oMu / oMo                    # = ||u_bend||_M^2


def bend_rhs_ca(theta, A, b):
    """rhs r = b o cos(theta) for the implicit u-defining constraint M u = r."""
    n = A.shape[0]
    bca = ca.DM(b.reshape(-1, 1))
    return bca * ca.cos(theta)


def mass_matrix_ca(theta, A):
    n = A.shape[0]
    Aca = ca.DM(A)
    dth = ca.repmat(theta, 1, n) - ca.repmat(theta.T, n, 1)
    return Aca * ca.cos(dth) + ca.DM(np.eye(n) / 12.0)


def bend_excite_np(chain, theta):
    """Numpy reference of sqrt(c) for diagnostics."""
    theta = np.asarray(theta); M = chain.mass_matrix(theta)
    u = np.linalg.solve(M, chain.b * np.cos(theta)); ones = np.ones(chain.n)
    proj = (ones @ M @ u) / (ones @ M @ ones); ub = u - proj * ones
    return float(np.sqrt(ub @ M @ ub))


# ---- controllability-aware implicit collocation ----------------------------
def solve_ctrb_aware(T, K, w_ctrb=0.0, floor_ctrb=0.0, hard_floor=0.0,
                     init_guess=None, theta_target=None, max_iter=2500,
                     tol=1e-7, print_level=0):
    """Trapezoidal IMPLICIT collocation (same skeleton as trajopt's) PLUS a
    controllability term that targets ONLY the configurations where the chain
    goes near-straight (the documented failure):

      ONE-SIDED soft penalty  w_ctrb * sum_k max(0, floor_ctrb - sqrt(c_k))^2
      (active only where bend-excitation sqrt(c) drops below floor_ctrb; healthy
       nodes are untouched, so it does NOT fight the swing-up globally), and/or
      a HARD floor  sqrt(c_k) >= hard_floor.

    c = ||u_bend||_M^2 (squared bend excitation), with u defined implicitly by
    M u = b o cos(theta) (extra decision vars, no symbolic inverse).
    """
    n = N
    A, b = chain_constants(n)
    if theta_target is None:
        theta_target = np.zeros(n)
    theta_target = np.asarray(theta_target, float)
    h = T / K
    Mfn, rfn = make_M_rhs_fn(n, G)

    opti = ca.Opti()
    TH = opti.variable(n, K + 1); TD = opti.variable(n, K + 1)
    VV = opti.variable(1, K + 1); TDD = opti.variable(n, K + 1)
    AC = opti.variable(1, K + 1)

    cost = 0
    for k in range(K + 1):
        opti.subject_to(ca.mtimes(Mfn(TH[:, k]), TDD[:, k])
                        == rfn(TH[:, k], TD[:, k], AC[0, k]))
    for k in range(K):
        opti.subject_to(TH[:, k + 1] - TH[:, k] == 0.5 * h * (TD[:, k] + TD[:, k + 1]))
        opti.subject_to(TD[:, k + 1] - TD[:, k] == 0.5 * h * (TDD[:, k] + TDD[:, k + 1]))
        opti.subject_to(VV[0, k + 1] - VV[0, k] == 0.5 * h * (AC[0, k] + AC[0, k + 1]))
        cost += 0.5 * h * (AC[0, k] ** 2 + AC[0, k + 1] ** 2)        # min effort
        cost += 1e-3 * (AC[0, k + 1] - AC[0, k]) ** 2                 # smoothness

    # boundary
    opti.subject_to(TH[:, 0] == np.pi); opti.subject_to(TD[:, 0] == 0)
    opti.subject_to(VV[0, 0] == 0)
    opti.subject_to(TH[:, K] == theta_target); opti.subject_to(TD[:, K] == 0)
    opti.subject_to(VV[0, K] == 0)
    opti.subject_to(opti.bounded(-A_MAX, ca.vec(AC), A_MAX))
    opti.subject_to(opti.bounded(-V_MAX, VV, V_MAX))

    kset = int(round((1 - SETTLE_FRAC) * K))
    for k in range(kset, K + 1):
        opti.subject_to(opti.bounded(theta_target - SETTLE_BAND, TH[:, k],
                                     theta_target + SETTLE_BAND))

    # CONTROLLABILITY term over the swing (skip the locked-down start/settle ends)
    # Subsample nodes (CSTRIDE) to keep the extra implicit u-vars cheap; the
    # bend-excitation varies smoothly so a stride of a few nodes is plenty.
    CSTRIDE = 3
    k_lo = max(1, int(0.05 * K)); k_hi = kset
    cnodes = list(range(k_lo, k_hi, CSTRIDE))
    use_ctrb = (w_ctrb > 0.0 and floor_ctrb > 0.0) or hard_floor > 0.0
    if use_ctrb and cnodes:
        U = opti.variable(n, len(cnodes))   # input->angaccel direction at cnodes
        for j, k in enumerate(cnodes):
            # implicit definition: M(theta_k) u_k = b o cos(theta_k)  (no inverse)
            opti.subject_to(ca.mtimes(mass_matrix_ca(TH[:, k], A), U[:, j])
                            == bend_rhs_ca(TH[:, k], A, b))
            c = bend_excite_ca(TH[:, k], U[:, j], A)    # squared bend excitation
            cm = ca.sqrt(c + 1e-9)                       # bend-excitation magnitude
            if w_ctrb > 0.0 and floor_ctrb > 0.0:
                short = ca.fmax(0.0, floor_ctrb - cm)    # one-sided hinge
                cost += w_ctrb * short * short           # penalize only collapse
            if hard_floor > 0.0:
                opti.subject_to(cm >= hard_floor)        # hard floor

    opti.minimize(cost)

    # neutral warm start (homotopy guess or pi->target ramp)
    s = np.linspace(0, 1, K + 1)
    if init_guess is not None:
        Kg = init_guess["theta"].shape[0] - 1; sg = np.linspace(0, 1, Kg + 1)
        thg = np.array([np.interp(s, sg, init_guess["theta"][:, j]) for j in range(n)])
        tdg = np.array([np.interp(s, sg, init_guess["thetad"][:, j]) for j in range(n)])
        vg = np.interp(s, sg, init_guess["v"]); ag = np.interp(s, sg, init_guess["a"])
        opti.set_initial(TH, thg); opti.set_initial(TD, tdg)
        opti.set_initial(VV, vg.reshape(1, -1)); opti.set_initial(AC, ag.reshape(1, -1))
    else:
        thg = np.outer(np.full(n, np.pi), (1 - s)) + np.outer(theta_target, s)
        opti.set_initial(TH, thg)

    # warm-start the controllability u-vars from the guessed theta (M u = r)
    if use_ctrb and cnodes:
        chain = Chain(n, G)
        Uinit = np.zeros((n, len(cnodes)))
        for j, k in enumerate(cnodes):
            tk = thg[:, k]
            Uinit[:, j] = np.linalg.solve(chain.mass_matrix(tk), b * np.cos(tk))
        opti.set_initial(U, Uinit)

    s_opts = {"max_iter": max_iter, "print_level": print_level, "sb": "yes",
              "tol": tol, "acceptable_tol": max(tol * 100, 1e-6),
              "mu_strategy": "adaptive"}
    opti.solver("ipopt", {"expand": True}, s_opts)
    try:
        sol = opti.solve(); status = "solved"
    except RuntimeError:
        sol = opti.debug; status = "failed"
    try:
        THv = np.array(sol.value(TH)); TDv = np.array(sol.value(TD))
        VVv = np.array(sol.value(VV)).ravel(); ACv = np.array(sol.value(AC)).ravel()
    except RuntimeError:
        return {"status": "failed", "T": T, "K": K, "n": n}
    return {"t": np.linspace(0, T, K + 1), "theta": THv.T, "thetad": TDv.T,
            "v": VVv, "a": ACv, "status": status, "T": T, "K": K, "n": n}


# ---- neutral homotopy ladder (NO curated seed) -----------------------------
def neutral_ladder():
    """Cold homotopy ladder N=2..5, target zeros, REFINED to h=0.01 at N=5.
    Returns the N=5 seed dict (homotopy-lifted to N=6 by the workers). This is
    the NEUTRAL start: a fresh cold ladder to zeros, NO curated/pre-existing
    trajectory. A FINE N=5 makes the N=6 lift converge reliably (the coarse
    N=6 lift is the known-unreliable step in this codebase)."""
    prev = None
    for nn in range(2, 6):
        T = {2: 4.0, 3: 5.0, 4: 7.0, 5: 11.0}[nn]
        Kc = int(T / 0.05)
        guess = homotopy_guess(prev) if prev else None
        sol = None
        for sd in range(4):
            sol = solve_swingup_implicit(nn, T, Kc, g=G, a_max=60.0, v_max=14.0,
                theta_target=np.zeros(nn), settle_frac=0.1, settle_band=0.2,
                seed=sd, init_guess=guess, max_iter=1500, print_level=0, tol=1e-6)
            if sol["status"] == "solved":
                break
        if sol["status"] != "solved":
            raise SystemExit(f"neutral ladder N={nn} (coarse) failed")
        prev = sol
    # refine N=5 to h=0.01 for a clean warm start to the N=6 lift
    T5 = 11.0; Kf = int(T5 / 0.01)
    fine = solve_swingup_implicit(5, T5, Kf, g=G, a_max=60.0, v_max=14.0,
        theta_target=np.zeros(5), settle_frac=0.1, settle_band=0.2,
        init_guess=prev, max_iter=2000, print_level=0, tol=1e-7)
    if fine["status"] != "solved":
        raise SystemExit("neutral ladder N=5 (fine refine) failed")
    return fine


# ---- workers ---------------------------------------------------------------
def w_solve(task):
    """Solve one controllability-aware N=6. task=(tag,T,w,floor,hard,seed_path)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    tag, T, w, floor, hard, seed_path = task
    seed5 = dict(np.load(seed_path))             # NEUTRAL fine N=5
    guess6 = homotopy_guess(seed5)               # lift N=5 -> N=6 (tip shadows)
    K = int(T / 0.01)
    sf = solve_ctrb_aware(T, K, w_ctrb=w, floor_ctrb=floor, hard_floor=hard,
                          init_guess=guess6, theta_target=np.zeros(N),
                          max_iter=2500, tol=1e-7)
    if sf["status"] != "solved":
        return (tag, None, None)
    fn = POOL / f"{tag}.npz"
    np.savez(fn, t=sf["t"], theta=sf["theta"], thetad=sf["thetad"],
             a=sf["a"], v=sf["v"], T=sf["T"])
    return (tag, str(fn), None)


def w_eval(item):
    """Trackability + metrics. item=(tag,fn). Returns dict of metrics."""
    os.environ["OMP_NUM_THREADS"] = "1"
    tag, fn = item
    d = dict(np.load(fn))
    chain = Chain(N, G)
    cvals = np.array([bend_excite_np(chain, d["theta"][i]) for i in range(len(d["t"]))])
    ok, final, bundle = n6_perfect_state(d)
    res = dict(tag=tag, fn=fn, ok=bool(ok), final=float(final),
               cmin=float(cvals.min()), cmed=float(np.median(cvals)),
               amax=float(np.max(np.abs(d["a"]))))
    if bundle is not None:
        res["maxK"] = float(bundle["maxK"])
        # closed-loop max pivot accel: recompute rollout to capture |a|
        res["accel_max"] = closed_loop_accel(bundle)
    return res


def closed_loop_accel(bundle):
    """Roll out the full-state closed loop and return max |pivot accel|."""
    from pendulum.sim import rk4_step
    chain = Chain(N, G); dt = float(bundle["dt"])
    theta = bundle["theta_nom"]; thetad = bundle["thetad_nom"]
    a_ff = bundle["a_ff"]; v_nom = bundle["v_nom"]; Ks = bundle["K"]
    y = np.concatenate([theta[0], thetad[0]]); v = 0.0; amax = 0.0
    for k in range(len(Ks)):
        z = np.concatenate([y[:N], y[N:], [v]])
        zn = np.concatenate([theta[k], thetad[k], [v_nom[k]]])
        a = a_ff[k] - float((Ks[k] @ (z - zn))[0])
        amax = max(amax, abs(a)); y = rk4_step(chain, y, a, dt); v += a * dt
        if not np.isfinite(y).all():
            break
    return float(amax)


# ---- main ------------------------------------------------------------------
def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    POOL.mkdir(parents=True, exist_ok=True)
    seed_path = POOL / "_neutral_seed_N5.npz"
    # clear stale candidate files (keep the cached neutral seed if present)
    for p in POOL.glob("*.npz"):
        if p.name != seed_path.name:
            p.unlink()

    if seed_path.exists() and "--rebuild" not in sys.argv:
        log("using cached NEUTRAL N=5 seed (NO curated seed; pass --rebuild to redo)")
        seed5 = dict(np.load(seed_path))
    else:
        log("building NEUTRAL homotopy ladder N=2..5 (target zeros, NO curated seed)...")
        seed5 = neutral_ladder()
        np.savez(seed_path, t=seed5["t"], theta=seed5["theta"], thetad=seed5["thetad"],
                 a=seed5["a"], v=seed5["v"], T=seed5["T"])
    chain5 = Chain(5, G)
    cseed = np.array([bend_excite_np(chain5, seed5["theta"][i])
                      for i in range(len(seed5["t"]))])
    log(f"  neutral N=5 seed ready (T={float(seed5['T'])}, cmin={cseed.min():.3f}, "
        f"cmed={np.median(cseed):.3f})")

    # Sweep the one-sided soft floor (w_ctrb, floor_ctrb) and the hard floor.
    # task = (tag, T, w_ctrb, floor_ctrb, hard_floor, seed_path)
    # w0 is the controllability-BLIND baseline from the SAME neutral start
    # (control). The soft cases penalize only bend-excitation below `floor`; the
    # hard cases constrain it >= floor. floor levels span the trackable band
    # (GOOD baseline never drops below ~0.97; neutral seed dips to ~0.44-0.60).
    sp = str(seed_path)
    # SOFT one-sided floor only. Empirically: hard floors and over-aggressive
    # soft floors distort the swing into a different but still-untrackable bend
    # order, so we sweep a few SOFT floor levels around the working band in
    # parallel and keep the first trackable one. Robust without a curated seed,
    # no large hyperparameter search.
    tasks = [(f"soft_w100_fl{fl}", 15.0, 100.0, fl, 0.0, sp)
             for fl in (0.6, 0.7, 0.8, 0.9)]

    log(f"solving {len(tasks)} controllability-aware N=6 ({nworkers} workers)...")
    with mp.Pool(nworkers) as pool:
        solved = [r for r in pool.map(w_solve, tasks) if r[1]]
    log(f"  {len(solved)}/{len(tasks)} solved")
    if not solved:
        raise SystemExit("no controllability-aware N=6 solved")

    log("evaluating trackability (full-state dt=0.004) + metrics...")
    with mp.Pool(nworkers) as pool:
        evals = pool.map(w_eval, [(t, f) for t, f, _ in solved])
    evals.sort(key=lambda e: (not e["ok"], e["final"]))
    log("results (sorted; ok first, then smallest upright error):")
    for e in evals:
        mk = e.get("maxK", float("nan")); am = e.get("accel_max", float("nan"))
        log(f"  {e['tag']:14s} ok={e['ok']!s:5s} final={np.degrees(e['final']):7.3f}deg "
            f"cmin={e['cmin']:.3f} maxK={mk:10.0f} accel_max={am:6.1f}")

    track = [e for e in evals if e["ok"]]
    if not track:
        best = evals[0]
        log(f"NO trackable N=6. closest: {best['tag']} final={np.degrees(best['final']):.3f}deg "
            f"cmin={best['cmin']:.3f}")
        raise SystemExit("no trackable N=6 (controllability-aware)")

    best = track[0]
    log(f"BEST trackable: {best['tag']} final={np.degrees(best['final']):.4f}deg "
        f"maxK={best.get('maxK'):.0f} accel_max={best.get('accel_max'):.1f}")
    _, _, bundle = n6_perfect_state(dict(np.load(best["fn"])))
    np.savez(OUT, **bundle)
    log(f"SAVED controls bundle -> {OUT} (maxK={bundle['maxK']:.0f}, "
        f"final={np.degrees(bundle['final']):.4f}deg)")


if __name__ == "__main__":
    main()
