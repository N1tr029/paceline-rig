"""Finding the glasses and working out how to read them.

The packet layout of the Xreal/Nreal IMU interface is not published; the
open-source drivers were reverse engineered and the details differ between
models and firmware revisions. Rather than hard-code one guess and hand you
nonsense rates, this module TRIES several layouts and picks the one whose
data actually looks like a real accelerometer.

The test is physics, not documentation: at rest, the acceleration vector
must have magnitude ~9.81 m/s^2 and the gyro must read near zero. Only one
candidate decode will satisfy both. Hold the glasses still while probing.

If every candidate fails, the report tells you what it saw, and you can add
a layout to CANDIDATES from
https://github.com/GabiLegrand/xreal-imu-python
"""
from __future__ import annotations
import struct
import numpy as np

G = 9.81
KNOWN_VIDS = (0x3318, 0x0486)          # Xreal, older Nreal
IMU_INTERFACES = (3, 4, 0)             # most likely first


class Layout:
    """One guess at how a HID report is packed."""

    def __init__(self, name, fmt, offset, gyro_scale, accel_scale, order=(0, 1, 2, 3, 4, 5)):
        self.name = name
        self.fmt = fmt
        self.offset = offset
        self.gs = gyro_scale
        self.as_ = accel_scale
        self.order = order

    @property
    def size(self):
        return struct.calcsize(self.fmt) + self.offset

    def decode(self, buf):
        v = struct.unpack_from(self.fmt, buf, self.offset)
        v = [v[i] for i in self.order]
        gyro = np.array(v[0:3], dtype=float) * self.gs
        accel = np.array(v[3:6], dtype=float) * self.as_
        return gyro, accel


CANDIDATES = [
    Layout("f32 @8, SI units",        "<6f",  8,  1.0,      1.0),
    Layout("f32 @8, g and rad/s",     "<6f",  8,  57.29578, G),
    Layout("f32 @16, SI units",       "<6f", 16,  1.0,      1.0),
    Layout("i32 @10, 1e-4 scale",     "<6i", 10,  1e-4 * 57.29578, 1e-4 * G),
    Layout("i16 @4, 2000dps / 8g",    "<6h",  4,  2000/32768, 8*G/32768),
    Layout("i16 @6, 2000dps / 8g",    "<6h",  6,  2000/32768, 8*G/32768),
    Layout("i16 @4, 1000dps / 4g",    "<6h",  4,  1000/32768, 4*G/32768),
]


def enumerate_candidates(vid=None, pid=None):
    """Every HID interface that might be the IMU."""
    import hid
    out = []
    vids = (vid,) if vid else KNOWN_VIDS
    for v in vids:
        for d in hid.enumerate(v, pid or 0):
            out.append(d)
    out.sort(key=lambda d: IMU_INTERFACES.index(d.get("interface_number"))
             if d.get("interface_number") in IMU_INTERFACES else 99)
    return out


def score_layout(layout, packets):
    """How physically plausible is this decode? Lower is better."""
    accs, gyros = [], []
    for buf in packets:
        if len(buf) < layout.size:
            return 1e9, None
        try:
            g, a = layout.decode(buf)
        except struct.error:
            return 1e9, None
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(a))):
            return 1e9, None
        accs.append(np.linalg.norm(a))
        gyros.append(np.linalg.norm(g))
    accs, gyros = np.array(accs), np.array(gyros)
    if accs.mean() < 1e-6:
        return 1e9, None
    # at rest: |accel| ~ 9.81 and steady; |gyro| ~ 0 and steady
    err = abs(accs.mean() - G) / G
    err += accs.std() / G
    err += min(gyros.mean(), 50.0) / 50.0
    return float(err), (accs.mean(), accs.std(), gyros.mean())


def probe(device, n_packets=120, verbose=True):
    """Read a burst, try every layout, return the best one."""
    packets = []
    for _ in range(n_packets * 3):
        buf = device.read(64, timeout=200)
        if buf:
            packets.append(bytes(buf))
        if len(packets) >= n_packets:
            break
    if len(packets) < 10:
        raise RuntimeError(
            f"only {len(packets)} HID reports in 3x the expected time. Wrong "
            "interface, or the device isn't streaming. Try another interface."
        )

    scored = []
    for lay in CANDIDATES:
        err, stats = score_layout(lay, packets)
        scored.append((err, lay, stats))
    scored.sort(key=lambda x: x[0])

    if verbose:
        print(f"  probed {len(packets)} reports, {len(packets[0])} bytes each")
        for err, lay, stats in scored[:4]:
            if stats:
                print(f"    {err:7.3f}  {lay.name:26s}  |a|={stats[0]:6.2f}"
                      f" +-{stats[1]:.2f}  |w|={stats[2]:6.2f}")
            else:
                print(f"    {'--':>7}  {lay.name:26s}  unusable")

    best_err, best, stats = scored[0]
    if best_err > 0.35:
        raise RuntimeError(
            "no candidate layout produced plausible physics.\n"
            f"  best was '{best.name}' at error {best_err:.2f} "
            f"(want |a|~9.81 and |w|~0 while held still)\n"
            "  Were the glasses moving? If not, add a layout to CANDIDATES in\n"
            "  paceline/xreal.py using github.com/GabiLegrand/xreal-imu-python"
        )
    if verbose:
        print(f"  -> using '{best.name}' (error {best_err:.3f})")
    return best
