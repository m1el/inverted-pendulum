"""Merge per-N threshold results into results/swingup_traj.json with a summary."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def gm(bracket):
    lo, hi = bracket
    if lo <= 0:
        return 0.0
    if not np.isfinite(hi):
        return lo
    return float(np.sqrt(lo * hi))


def main():
    perN = ROOT / "results" / "thr_perN"
    out = {}
    for f in sorted(perN.glob("N*.json")):
        n = int(f.stem[1:])
        rec = json.load(open(f))
        rec["dtheta_headline"] = gm(rec["dtheta_threshold"])
        rec["dv_headline"] = gm(rec["dv_threshold"])
        rec["joint"]["dv_headline"] = gm(rec["joint"]["dv_threshold"])
        # selected preset
        sel = ROOT / "results" / "trajectories" / f"selection_N{n}.json"
        if sel.exists():
            rec["selection"] = json.load(open(sel))
        out[str(n)] = rec
    json.dump(out, open(ROOT / "results" / "swingup_traj.json", "w"), indent=2)

    print(f"{'N':>2} {'T(s)':>5} {'dtheta':>10} {'dv':>10} {'joint_dv':>10}")
    for n in sorted(out, key=int):
        r = out[n]
        print(f"{n:>2} {r['swing_duration']:>5.1f} {r['dtheta_headline']:>10.3e} "
              f"{r['dv_headline']:>10.3e} {r['joint']['dv_headline']:>10.3e}")
    print("\nwrote results/swingup_traj.json")


if __name__ == "__main__":
    main()
