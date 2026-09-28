"""Score a filter against a recording.

The headline number is NOT total RMS error. It is STRIDE-BAND error --
the residual between roughly 1 and 5 Hz, where footstrike and stride cycle
live. That is the error you feel: a line that jitters twice a second is
unusable, while the same magnitude of slow drift is barely noticeable and
is handled elsewhere (barometer-derived grade, periodic recalibration).

A filter that is quietly re-injecting stride motion looks acceptable on RMS
and screams in the spectrum. Always read the spectrum.

  python -m paceline.analyze --file data/sim.csv
  python -m paceline.analyze --file data/run.csv --plot
"""
from __future__ import annotations
import argparse, sys
import numpy as np

from .sources import ReplaySource, SimSource
from .attitude import GatedFilter, NaiveFilter

STRIDE_LO, STRIDE_HI = 1.0, 5.0
BUDGET_DEG = 0.30


def _run(filt, rows, truth):
    est, tru, ts = [], [], []
    prev_t = None
    for i, (t, g, a) in enumerate(rows):
        dt = 0.0025 if prev_t is None else max(1e-5, t - prev_t)
        prev_t = t
        filt.update(g, a, dt)
        est.append(filt.pitch)
        tru.append(truth[i] if truth and truth[i] is not None else None)
        ts.append(t)
    return np.array(ts), np.array(est, dtype=float), tru


def _band_rms(sig, fs, lo, hi):
    sig = sig - sig.mean()
    win = np.hanning(len(sig))
    F = np.fft.rfft(sig * win)
    f = np.fft.rfftfreq(len(sig), 1.0 / fs)
    # Parseval on the windowed signal, corrected for window power
    scale = 2.0 / (len(sig) * np.sqrt((win ** 2).mean()))
    mag = np.abs(F) * scale
    sel = (f >= lo) & (f <= hi)
    return float(np.sqrt((mag[sel] ** 2).sum() / 2.0)), f, mag


def main(argv=None):
    ap = argparse.ArgumentParser(description="score filters on a recording")
    ap.add_argument("--file", required=True)
    ap.add_argument("--skip", type=float, default=10.0, help="seconds to discard while the filter settles")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)

    src = ReplaySource(a.file)
    rows = list(src)
    truth = src.truth
    if len(rows) < 100:
        print("not enough samples"); return 1
    fs = (len(rows) - 1) / (rows[-1][0] - rows[0][0])
    print(f"{len(rows)} samples, {rows[-1][0]:.1f}s, {fs:.0f} Hz\n")

    results = {}
    for cls in (GatedFilter, NaiveFilter):
        f = cls()
        ts, est, tru = _run(f, rows, truth)
        m = ts >= a.skip
        ts, est = ts[m], est[m]
        tru_m = [t for t, keep in zip(tru, m) if keep]
        have_truth = tru_m and tru_m[0] is not None

        sig = est - np.array(tru_m, dtype=float) if have_truth else est
        band, freqs, mag = _band_rms(sig, fs, STRIDE_LO, STRIDE_HI)
        slow, _, _ = _band_rms(sig, fs, 0.0, 0.5)
        results[cls.name] = (band, slow, freqs, mag)

        label = "error" if have_truth else "estimate content"
        ok = "PASS" if band <= BUDGET_DEG else "FAIL"
        print(f"  {cls.name:6s}  stride-band {label} {band:6.3f} deg   [{ok}, budget {BUDGET_DEG}]")
        print(f"          slow (<0.5 Hz) {slow:6.3f} deg   (grade + recalibration handle this)")
        if isinstance(f, GatedFilter):
            print(f"          accel gate passed {f.gate_pass_rate*100:.0f}% of samples")
        peak = freqs[np.argmax(np.where((freqs > STRIDE_LO) & (freqs < STRIDE_HI), mag, 0))]
        print(f"          largest stride-band component at {peak:.2f} Hz\n")

    g, n = results["gated"][0], results["naive"][0]
    if n > 0:
        print(f"gating improves stride-band residual by {n/max(g,1e-9):.1f}x")

    if a.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot"); return 0
        fig, ax = plt.subplots(figsize=(9, 4))
        for name, (band, slow, freqs, mag) in results.items():
            ax.semilogy(freqs, np.maximum(mag, 1e-6), label=f"{name} ({band:.3f} deg)")
        ax.axvspan(STRIDE_LO, STRIDE_HI, alpha=.12, color="tab:red")
        ax.set_xlim(0, 10); ax.set_xlabel("Hz"); ax.set_ylabel("deg")
        ax.set_title("pitch error spectrum -- shaded band is what you feel")
        ax.legend(); fig.tight_layout(); fig.savefig("spectrum.png", dpi=140)
        print("wrote spectrum.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
