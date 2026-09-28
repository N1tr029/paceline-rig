"""Sanity checks. Run: python -m pytest -q   (or just execute this file)"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from paceline.sources import SimSource
from paceline.attitude import GatedFilter, NaiveFilter
from paceline.pace import Pacer, HOLD, PUSH, EASE


def _score(filt, seconds=60, skip=10):
    src = SimSource()
    err = []
    for t, g, a in src:
        if t > seconds:
            break
        filt.update(g, a, src.dt)
        if t > skip:
            err.append(filt.pitch - src.truth_pitch_deg(t))
    e = np.array(err)
    e = e - e.mean()
    win = np.hanning(len(e))
    F = np.abs(np.fft.rfft(e * win)) * 2.0 / (len(e) * np.sqrt((win ** 2).mean()))
    f = np.fft.rfftfreq(len(e), src.dt)
    sel = (f >= 1.0) & (f <= 5.0)
    return math.sqrt((F[sel] ** 2).sum() / 2.0)


def test_gating_beats_naive_in_the_stride_band():
    g = _score(GatedFilter())
    n = _score(NaiveFilter())
    assert g < 0.3, f"gated filter over budget: {g:.3f} deg"
    assert n > 1.0, f"naive filter should fail badly, got {n:.3f} deg"
    assert n / g > 10


def test_hold_distance_scales_with_pace():
    fast, slow = Pacer(360.0), Pacer(600.0)
    assert fast.hold_distance() > slow.hold_distance()
    assert abs(fast.hold_distance() / fast.v_target - 1.8) < 1e-6


def test_states_need_dwell_and_hysteresis():
    p = Pacer(400.0, dwell_s=2.0)
    for _ in range(600):
        p.update(480.0, 1 / 60)
    assert p.state == PUSH
    for _ in range(600):
        p.update(320.0, 1 / 60)
    assert p.state == EASE
    for _ in range(900):
        p.update(400.0, 1 / 60)
    assert p.state == HOLD


def test_line_stays_inside_what_optics_can_show():
    p = Pacer(400.0)
    for _ in range(6000):
        p.update(700.0, 1 / 60)
    assert 3.6 <= p.line_distance() <= 44.0
    for _ in range(12000):
        p.update(260.0, 1 / 60)
    assert 3.6 <= p.line_distance() <= 44.0


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn(); print(f"ok  {fn.__name__}")
    print("\nall passed")
