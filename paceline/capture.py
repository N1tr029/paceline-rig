"""Record raw IMU to CSV. This is step one of the whole programme.

You never tune a sensor-fusion filter live on a run -- one iteration per
run is hopeless. Record once, then iterate offline against the recording a
hundred times in an afternoon.

  python -m paceline.capture --source xreal --out data/treadmill_7min.csv
  python -m paceline.capture --source sim   --out data/sim.csv --seconds 120
"""
from __future__ import annotations
import argparse, csv, sys, time
from .sources import make_source, SimSource


def main(argv=None):
    ap = argparse.ArgumentParser(description="record raw IMU to CSV")
    ap.add_argument("--source", default="sim", choices=["sim", "xreal"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args(argv)

    src = make_source(a.source)
    is_sim = isinstance(src, SimSource)
    n = 0
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["t", "gx", "gy", "gz", "ax", "ay", "az"]
        if is_sim:
            head.append("truth_pitch_deg")
        w.writerow(head)
        started = time.perf_counter()
        for t, g, ac in src:
            row = [f"{t:.5f}", *[f"{v:.4f}" for v in g], *[f"{v:.4f}" for v in ac]]
            if is_sim:
                row.append(f"{src.truth_pitch_deg(t):.4f}")
            w.writerow(row)
            n += 1
            if t >= a.seconds:
                break
            if not is_sim and time.perf_counter() - started > a.seconds + 5:
                break
    print(f"wrote {n} samples ({a.seconds:.0f}s) -> {a.out}")
    if not is_sim:
        print("no ground truth in this file; analyze will report stride-band "
              "content instead of true error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
