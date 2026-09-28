"""Where head motion comes from.

Every source yields the same tuple:
    (t_seconds, gyro_dps[3], accel_ms2[3])

Axis convention used everywhere downstream, with the glasses worn normally:
    x  right      -> gyro x = ROLL rate  (head tips toward a shoulder)
    y  up         -> gyro y = YAW rate   (head turns left/right)
    z  forward    -> gyro z = PITCH rate (nodding)  [+ve = nose up]
Real devices rarely match. Use AxisMap to fix it once, here, rather than
sprinkling sign flips through the filter.
"""
from __future__ import annotations
import csv, math, time
import numpy as np

G = 9.81


class AxisMap:
    """Remap and sign-flip a device's raw axes into our convention.

    order: which raw index feeds each of our (roll, yaw, pitch) slots
    signs: +1 / -1 per slot
    """

    def __init__(self, order=(0, 1, 2), signs=(1, 1, 1)):
        self.order = tuple(order)
        self.signs = np.array(signs, dtype=float)

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        return v[list(self.order)] * self.signs


class SimSource:
    """Synthetic running head motion. No hardware needed.

    Deliberately includes the thing that breaks naive filters: roughly 1 g of
    stride acceleration on top of gravity, phase-locked to footstrike. The
    accelerometer's idea of "down" therefore swings by tens of degrees, 2.8
    times a second, exactly in time with the motion you are trying to cancel.

    Also reports ground truth, so analyze.py can score a filter honestly.
    """

    def __init__(self, rate_hz=400.0, stride_hz=2.8, pitch_amp_deg=3.0,
                 bob_g=0.95, surge_g=0.35, drift_deg=2.0, seed=7):
        self.dt = 1.0 / rate_hz
        self.f = stride_hz
        self.amp = pitch_amp_deg
        self.bob = bob_g
        self.surge = surge_g
        self.drift = drift_deg
        self.t = 0.0
        self.rng = np.random.default_rng(seed)

    def truth_pitch_deg(self, t):
        """What the head is really doing: stride nod + a slow postural drift."""
        return (self.amp * math.sin(2 * math.pi * self.f * t)
                + self.drift * math.sin(2 * math.pi * 0.05 * t))

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t
        self.t += self.dt
        h = 1e-4
        p0 = self.truth_pitch_deg(t - h)
        p1 = self.truth_pitch_deg(t + h)
        pitch = self.truth_pitch_deg(t)
        pitch_rate = (p1 - p0) / (2 * h)

        gyro = np.array([
            0.9 * math.sin(2 * math.pi * self.f * 0.5 * t),   # roll rate
            1.6 * math.sin(2 * math.pi * self.f * 0.5 * t + 0.7),  # yaw rate
            pitch_rate,
        ]) + self.rng.normal(0, 0.05, 3)

        # gravity as seen in the body frame, plus stride acceleration
        pr = math.radians(pitch)
        a_vert = self.bob * G * math.sin(2 * math.pi * self.f * t + 0.9)
        a_fore = self.surge * G * math.sin(2 * math.pi * self.f * t + 2.1)
        accel = np.array([
            self.rng.normal(0, 0.15),
            -G * math.sin(pr) + a_fore,
            G * math.cos(pr) + a_vert,
        ])
        return t, gyro, accel


class ReplaySource:
    """Replay a CSV written by capture.py.

    Columns: t,gx,gy,gz,ax,ay,az[,truth_pitch_deg]
    """

    def __init__(self, path, axes: AxisMap | None = None):
        self.path = path
        self.axes = axes or AxisMap()
        self.truth = []
        self._rows = []
        with open(path, newline="") as fh:
            r = csv.DictReader(fh)
            for row in r:
                self._rows.append((
                    float(row["t"]),
                    self.axes([float(row["gx"]), float(row["gy"]), float(row["gz"])]),
                    self.axes([float(row["ax"]), float(row["ay"]), float(row["az"])]),
                ))
                self.truth.append(float(row["truth_pitch_deg"]) if row.get("truth_pitch_deg") else None)
        self._i = 0

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self._rows):
            raise StopIteration
        row = self._rows[self._i]
        self._i += 1
        return row


class XrealSource:
    """Raw IMU from Xreal / Nreal Air over USB HID.

    The packet layout is not published and varies between models, so this
    does not guess: on open it probes the stream against several candidate
    layouts and keeps the one whose data obeys physics (|accel| ~ 9.81 and
    |gyro| ~ 0 while held still). See xreal.py.

    Hold the glasses still for a second when this starts.

    Never read the device's own fused quaternion instead of this. That
    inherits its accelerometer trust, which is the exact failure the gated
    filter exists to defeat.
    """

    def __init__(self, axes: AxisMap | None = None, verbose=True, vid=None, pid=None):
        try:
            import hid
        except ImportError as e:
            raise SystemExit(
                "hidapi not installed:  pip install hidapi\n"
                "Or work without hardware:  --source sim"
            ) from e
        from .xreal import enumerate_candidates, probe

        devs = enumerate_candidates(vid, pid)
        if not devs:
            raise SystemExit(
                "No Xreal/Nreal HID device found. Run `python -m paceline.doctor` "
                "for what to check."
            )
        last = None
        for d in devs[:3]:
            try:
                dev = hid.Device(path=d["path"])
            except Exception as e:
                last = e
                continue
            try:
                if verbose:
                    print(f"probing interface {d.get('interface_number')} "
                          f"- hold the glasses still...")
                self.layout = probe(dev, verbose=verbose)
                self.dev = dev
                self.info = d
                break
            except Exception as e:
                last = e
                dev.close()
        else:
            raise SystemExit(
                f"Found the device but could not decode it: {last}\n"
                "Run `python -m paceline.doctor` for detail."
            )
        self.axes = axes or AxisMap()
        self.t0 = time.perf_counter()
        self.dropped = 0

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass

    def __iter__(self):
        return self

    def __next__(self):
        buf = self.dev.read(64, timeout=500)
        if not buf:
            self.dropped += 1
            if self.dropped > 20:
                raise StopIteration
            return next(self)
        self.dropped = 0
        gyro, accel = self.layout.decode(bytes(buf))
        return time.perf_counter() - self.t0, self.axes(gyro), self.axes(accel)


def make_source(name, **kw):
    if name == "sim":
        return SimSource(**{k: v for k, v in kw.items() if k in
                            ("rate_hz", "stride_hz", "pitch_amp_deg", "seed")})
    if name == "replay":
        return ReplaySource(kw["file"])
    if name == "xreal":
        return XrealSource()
    raise ValueError(f"unknown source: {name}")
