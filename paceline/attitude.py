"""Attitude estimation -- the part that decides whether this product works.

Two filters live here so you can put them side by side:

  NaiveFilter   what every stock complementary / Madgwick / Mahony setup does,
                and what the fusion block inside most IMUs does: trust the
                accelerometer to say which way is down. Fine on a desk.
                On a running head the accelerometer sees ~1 g of stride
                acceleration on top of the 1 g that really is down, so its
                idea of "down" swings by tens of degrees at 2.8 Hz -- exactly
                in time with the motion you are cancelling. The filter then
                faithfully re-injects the bounce it exists to remove.

  GatedFilter   the fix. Two changes:
                  1. GATE. Only accept accelerometer samples whose magnitude
                     is near 1 g -- the quiet moments in the stride. This
                     throws away most of the contaminated energy before any
                     filtering has to attenuate it.
                  2. TWO POLES, very slow. A single pole at 0.1 Hz only gives
                     ~23 dB at stride frequency; a 17 deg swing still leaks
                     1.2 deg through, which is over the whole 0.3 deg budget.
                     Two poles at tau ~= 8 s gets you comfortably under.

                A correction loop that slow cannot track a real attitude
                change either -- running onto a hill, say. That is deliberate:
                road grade is supposed to come from the barometer and GNSS,
                not from asking the accelerometer to do two jobs at once.

Yaw has no gravity reference, so it gets the same treatment with GNSS course
over ground as the slow anchor instead (set_course). Over ten seconds your
head points where you are going; short head turns then behave correctly on
their own.

Angles are degrees throughout. Small-angle Euler integration is used rather
than quaternions: adequate for a head that stays within a few tens of degrees
of level, and far easier to read. Quaternions are the upgrade if you ever need
big attitudes.
"""
from __future__ import annotations
import math
import numpy as np

G = 9.81
ROLL, YAW, PITCH = 0, 1, 2


def _pitch_roll_from_accel(a):
    """Gravity direction -> pitch and roll, in degrees. +pitch = nose up."""
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    pitch = math.degrees(math.atan2(-ay, math.hypot(ax, az)))
    roll = math.degrees(math.atan2(ax, math.hypot(ay, az)))
    return pitch, roll


class NaiveFilter:
    """Complementary filter with a fast accelerometer correction. The trap."""

    name = "naive"

    def __init__(self, tau=0.5):
        self.tau = tau
        self.pitch = self.roll = self.yaw = 0.0
        self.rate = np.zeros(3)

    def update(self, gyro, accel, dt):
        self.rate = np.asarray(gyro, dtype=float)
        self.roll += self.rate[ROLL] * dt
        self.yaw += self.rate[YAW] * dt
        self.pitch += self.rate[PITCH] * dt

        pa, ra = _pitch_roll_from_accel(accel)
        k = min(1.0, dt / self.tau)
        self.pitch += (pa - self.pitch) * k
        self.roll += (ra - self.roll) * k
        return self.pitch


class GatedFilter:
    """Gyro for everything fast; gated, two-pole, very slow gravity anchor."""

    name = "gated"

    def __init__(self, tau_a=8.0, tau_b=8.0, gate_ms2=1.8, tau_yaw=10.0,
                 warmup_s=3.0, warmup_tau=0.25):
        self.tau_a, self.tau_b = tau_a, tau_b
        self.gate = gate_ms2
        self.tau_yaw = tau_yaw
        self.warmup_s, self.warmup_tau = warmup_s, warmup_tau
        self.pitch = self.roll = self.yaw = 0.0
        self._lp_pitch = self._lp_roll = 0.0
        self._course = None
        self.rate = np.zeros(3)
        self.n_total = 0
        self.n_gated = 0
        self._elapsed = 0.0
        self._since_accept = 0.0   # wall-clock gap between accepted samples

    @property
    def gate_pass_rate(self):
        return self.n_gated / self.n_total if self.n_total else 0.0

    def set_course(self, course_deg):
        """GNSS course over ground, degrees. Slow anchor for yaw only."""
        self._course = course_deg

    def prime(self, accel):
        """Snap to the current gravity vector. Call once while standing still."""
        p, r = _pitch_roll_from_accel(accel)
        self.pitch = self._lp_pitch = p
        self.roll = self._lp_roll = r
        self._elapsed = 0.0

    def update(self, gyro, accel, dt):
        self.rate = np.asarray(gyro, dtype=float)

        # --- fast path: gyro carries all of it ---
        self.roll += self.rate[ROLL] * dt
        self.yaw += self.rate[YAW] * dt
        self.pitch += self.rate[PITCH] * dt

        # --- slow path: gate, then two poles ---
        self._elapsed += dt
        self._since_accept += dt
        self.n_total += 1

        # Warm-up: converge fast for the first few seconds, then crawl. Without
        # this the loop takes minutes to find level from a cold start, because
        # the gate only passes a fraction of samples.
        warm = self._elapsed < self.warmup_s
        tau_a = self.warmup_tau if warm else self.tau_a
        tau_b = self.warmup_tau if warm else self.tau_b

        mag = float(np.linalg.norm(accel))
        if abs(mag - G) < self.gate:
            self.n_gated += 1
            pa, ra = _pitch_roll_from_accel(accel)
            # Gain uses time SINCE THE LAST ACCEPTED SAMPLE, not dt. Otherwise
            # the effective time constant silently stretches by 1/pass-rate,
            # so the filter would behave differently at different efforts.
            ka = min(1.0, self._since_accept / tau_a)
            self._lp_pitch += (pa - self._lp_pitch) * ka      # pole 1
            self._lp_roll += (ra - self._lp_roll) * ka
            self._since_accept = 0.0

        kb = min(1.0, dt / tau_b)                              # pole 2
        self.pitch += (self._lp_pitch - self.pitch) * kb
        self.roll += (self._lp_roll - self.roll) * kb

        if self._course is not None:
            err = (self._course - self.yaw + 180.0) % 360.0 - 180.0
            self.yaw += err * min(1.0, dt / self.tau_yaw)
        return self.pitch

    def predict(self, horizon_s):
        """Extrapolate attitude forward by the known display latency.

        This is half of what buys back the frame period you cannot late-latch
        away when you are rendering through DisplayPort. Running head pitch is
        periodic enough that 10-15 ms of prediction is accurate.
        """
        return (self.pitch + self.rate[PITCH] * horizon_s,
                self.roll + self.rate[ROLL] * horizon_s,
                self.yaw + self.rate[YAW] * horizon_s)


def make_filter(name):
    return {"gated": GatedFilter, "naive": NaiveFilter}[name]()
