# paceline-rig

Software for the pace line stabilisation rig: read a head-mounted IMU, hold a
virtual pacer line still on the road, and prove it with numbers.

**Runs today with no hardware.** The simulated source generates realistic
running head motion, including the ~1 g of stride acceleration that breaks
naive attitude filters.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pygame

python -m paceline.capture --source sim --out data/sim.csv --seconds 90
python -m paceline.analyze --file data/sim.csv
python -m paceline.app                       # live, windowed, press F
```

## What it's for

A line drawn on the road has to stay welded to the tarmac while your head
pitches ±3° every stride. Get that wrong and it isn't just ugly — it's
nauseating. Everything here exists to get it right and to prove it.

## The one idea

Every stock attitude filter — complementary, Madgwick, Mahony, and the fusion
block baked into most IMUs — trusts the accelerometer to say which way is
down. On a running head the accelerometer sees roughly 1 g of stride
acceleration *on top of* the 1 g that really is down, so its idea of "down"
swings by tens of degrees at 2.8 Hz, exactly in time with the motion you are
trying to cancel. The filter then faithfully re-injects the bounce it exists
to remove.

`GatedFilter` fixes it two ways:

1. **Gate.** Only accept accelerometer samples whose magnitude is near 1 g —
   the quiet moments in the stride. Throws away most of the contaminated
   energy before any filtering has to attenuate it.
2. **Two poles, very slow.** A single pole at 0.1 Hz gives only ~23 dB at
   stride frequency; a 17° swing still leaks 1.2° through, over the entire
   0.3° budget. Two poles at τ ≈ 8 s gets comfortably under.

Because that loop is deliberately too slow to track a real attitude change,
road grade must come from the barometer and GNSS instead — it is not the
accelerometer's job.

Measured on the simulator:

```
  gated   stride-band error  0.062 deg   [PASS, budget 0.3]
  naive   stride-band error  2.397 deg   [FAIL, budget 0.3]
  gating improves stride-band residual by 38.4x
```

## Read the spectrum, not the RMS

The headline metric is **stride-band residual** (1–5 Hz), not total RMS.
Slow drift of the same magnitude is barely noticeable and is handled by grade
input and recalibration; error at 2.8 Hz is what makes people sick. A filter
quietly re-injecting stride motion looks fine on RMS and screams in the
spectrum. `analyze.py` reports both, and `--plot` draws it.

## Layout

| file | what it does |
|---|---|
| `sources.py` | where motion comes from: `SimSource`, `ReplaySource`, `XrealSource` |
| `attitude.py` | `GatedFilter` and `NaiveFilter` — the actual IP |
| `pace.py` | virtual pacer, cumulative gap, hold/push/ease states |
| `render.py` | world-locked line, perspective projection, latency prediction |
| `capture.py` | record raw IMU to CSV |
| `analyze.py` | score a filter against a recording |
| `app.py` | live: source → filter → pacer → display |

## Using real glasses

`XrealSource` reads raw IMU from Xreal/Nreal Air over USB HID. It is written
from the public reverse-engineered drivers and is **untested against
hardware** — if rates look like nonsense, fix `_decode()` against
[xreal-imu-python](https://github.com/GabiLegrand/xreal-imu-python). That
should be the only part needing changes.

```bash
pip install hidapi
python -m paceline.capture --source xreal --out data/treadmill.csv --seconds 300
python -m paceline.analyze --file data/treadmill.csv
python -m paceline.app --source xreal --display 1 --fullscreen --latency-ms 8.3
```

The glasses appear as a second 1920×1080 monitor — `--display 1` usually. Do
not use Xreal's own Beam/3DoF software; you want a dumb monitor plus a raw
sensor and nothing else.

Two things to get right before trusting a number:

- **`--ppd`** — pixels per degree. Measure it, don't guess. Draw a line of
  known pixel width, note where it lands against something at a known
  distance, divide.
- **`--latency-ms`** — measured motion-to-photon, not the spec sheet. This
  sets the prediction horizon. 120 Hz ≈ 8.3, 60 Hz ≈ 16.7. You cannot
  late-latch through DisplayPort, so prediction is doing the work.

## Without ground truth

Recordings from real hardware have no true attitude to compare against, so
`analyze` reports stride-band *content* in the estimate instead of error.
The naive filter will still stand out. For a real error figure you need
commanded motion — that is what the gimbal rig is for, and it is the last
thing to buy, not the first.

## Tests

```bash
python tests/test_core.py        # or: pytest -q
```

## Status

Simulator, filters, pacer, renderer and scoring all work. The Xreal HID decode
is unverified. Nothing here has been on a real head yet.
