"""Bending-crossing topology of swing-up trajectories, and its reproducibility.

For a chain swinging hang(theta=pi)->upright(0), characterize the *bend order*:
- per-link winding  w_i = unwrap(theta_i)[T]-[0] over 2pi  (the coarse class;
  the repo's "-0.5 rev/link"),
- per-joint relative angle  beta_i = theta_{i+1}-theta_i  (how the chain folds);
  its sign-change count = how many times joint i reverses its bend ("crossings"),
  and the canonicalized sign sequence = the fine topology.

Signature = (rounded windings, joint crossing-counts). Two solutions share a
topology iff signatures match up to the global mirror symmetry theta->-theta
(which flips every bend sign). We then check whether independently-found stable
solutions reproduce the same signature.

Usage: uv run python scripts/bend_topology.py
"""
import sys, glob, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

DEAD = 0.10   # rad deadband so tiny wiggles don't count as bend reversals


def load_theta(f):
    d = np.load(f)
    th = d["theta"] if "theta" in d.files else d["theta_nom"]
    return th


def signature(th):
    n = th.shape[1]
    thu = np.unwrap(th, axis=0)                      # continuous angles
    wind = (thu[-1] - thu[0]) / (2 * np.pi)          # per-link winding
    beta = thu[:, 1:] - thu[:, :-1]                  # n-1 joint bend angles
    # crossing count: sign changes of beta_i with a deadband
    cross = []
    sgn0 = []
    for i in range(n - 1):
        b = beta[:, i]; s = np.sign(b) * (np.abs(b) > DEAD)
        s = s[s != 0]
        c = int(np.sum(s[1:] != s[:-1])) if len(s) else 0
        cross.append(c)
        sgn0.append(int(s[0]) if len(s) else 0)
    return np.round(wind, 1), np.array(cross), np.array(sgn0), beta


def canon(cross, sgn0):
    """Canonicalize for the theta->-theta mirror (flip all bend signs)."""
    if len(sgn0) and (list(sgn0[sgn0 != 0][:1]) or [1])[0] < 0:
        sgn0 = -sgn0
    return tuple(cross), tuple(sgn0)


def main():
    groups = {
        4: ["results/trajectories/swingup_N4.npz"],
        5: (["results/trajectories/swingup_N5.npz", "repro/seeds/swingup_N5_GOOD.npz"]
            + sorted(glob.glob("repro/pool_ctrb_n5/floor*.npz"))),
        6: (["results/trajectories/swingup_N6.npz", "repro/seeds/n6_ctrbaware.npz"]
            + sorted(glob.glob("repro/pool_ctrb/soft_w100_fl*.npz"))),
    }
    # alternative homotopy classes (different by construction) for contrast
    alt = {5: sorted(glob.glob("repro/pool/N5_*.npz")),
           6: sorted(glob.glob("repro/pool/N6_*.npz"))}

    for N in (4, 5, 6):
        print(f"\n===== N={N} =====")
        sigs = {}
        for f in groups[N]:
            if not pathlib.Path(f).exists():
                continue
            wind, cross, sgn0, _ = signature(load_theta(f))
            key = canon(cross, sgn0)
            sigs.setdefault(key, []).append(pathlib.Path(f).name)
            print(f"  {pathlib.Path(f).name:34s} wind={list(wind)}  cross/joint={list(cross)}  bend-sgn={list(sgn0)}")
        print(f"  --> {len(sigs)} distinct topology class(es) among {sum(len(v) for v in sigs.values())} stable solutions")
        for k, names in sigs.items():
            print(f"      class cross={k[0]} sgn={k[1]}: {len(names)} solution(s)")
        if alt.get(N):
            print("  -- alternative homotopy classes (different target winding, mostly untrackable):")
            for f in alt[N]:
                wind, cross, sgn0, _ = signature(load_theta(f))
                print(f"     {pathlib.Path(f).name:34s} wind={list(wind)}  cross/joint={list(cross)}")


if __name__ == "__main__":
    main()
