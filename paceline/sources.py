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

    UNTESTED against hardware -- written from the public reverse-engineered
    drivers. If the packet layout below is wrong you will see nonsense rates;
    compare against https://github.com/GabiLegrand/xreal-imu-python and fix
    _decode, which is the only part that should need changing.

    Never read a device's own fused quaternion instead of this. Doing so
    inherits its accelerometer trust, which is the exact failure the gated
    filter exists to defeat.
    """

    VENDOR = 0x3318          # Xreal
    IMU_INTERFACE = 3

    def __init__(self, product_id=None, axes: AxisMap | None = None):
        try:
            import hid  # noqa
        except ImportError as e:
            raise SystemExit(
                "hidapi not installed. `pip install hidapi`, or run with "
                "--source sim to work without hardware."
            ) from e
        import hid
        devs = [d for d in hid.enumerate(self.VENDOR, 0)
                if d.get("interface_number") == self.IMU_INTERFACE]
        if product_id:
            devs = [d for d in devs if d["product_id"] == product_id]
        if not devs:
            raise SystemExit(
                "No Xreal IMU interface found. Plugged in? On Linux you may "
                "need a udev rule; on macOS, grant input-monitoring permission."
            )
        self.dev = hid.Device(path=devs[0]["path"])
        self.axes = axes or AxisMap()
        self.t0 = time.perf_counter()

    def _decode(self, buf):
        """Pull gyro (deg/s) and accel (m/s^2) out of one HID report."""
        import struct
        gx, gy, gz, ax, ay, az = struct.unpack_from("<6f", buf, 8)
        return np.array([gx, gy, gz]), np.array([ax, ay, az])

    def __iter__(self):
        return self

    def __next__(self):
        buf = self.dev.read(64, timeout=200)
        if not buf:
            raise StopIteration
        gyro, accel = self._decode(bytes(buf))
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
