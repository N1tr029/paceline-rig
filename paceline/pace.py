"""The virtual pacer.

Position of the line comes from the CUMULATIVE gap to the pacer -- an
integral, so it averages noise away. Colour comes from INSTANTANEOUS pace
deviation, which is twitchy but actionable. Keeping those on separate
channels is deliberate: it puts the noisiest signal on the least noisy
display dimension.

Tuning here is time- and percentage-based rather than fixed metres and
seconds-per-mile, because a 6:00/mi runner and a 10:00/mi runner need
different absolute numbers to feel the same thing.
"""
from __future__ import annotations

MI = 1609.34
HOLD_SECONDS = 1.8      # keep the line this far ahead in TIME, not metres
BAND_FRACTION = 0.01    # +-1% of target pace reads as "on pace"

HOLD, PUSH, EASE = "hold", "push", "ease"


class Pacer:
    def __init__(self, target_sec_per_mi=406.0, hold_s=HOLD_SECONDS,
                 band=BAND_FRACTION, hysteresis=1.3, dwell_s=2.0):
        self.target = float(target_sec_per_mi)
        self.hold_s = hold_s
        self.band = band
        self.hysteresis = hysteresis     # widen the band to LEAVE a state
        self.dwell_s = dwell_s           # minimum time in a state
        self.gap_m = 0.0
        self.dev = 0.0                   # smoothed sec/mi, +ve = slower
        self.dist_m = 0.0
        self.elapsed = 0.0
        self.state = HOLD
        self._state_age = 0.0

    @property
    def v_target(self):
        return MI / self.target

    def hold_distance(self):
        """Constant time ahead -> 8.0 m at 6:00 pace, 4.8 m at 10:00."""
        return self.hold_s * self.v_target

    def update(self, pace_sec_per_mi, dt):
        v_you = MI / max(1e-6, pace_sec_per_mi)
        self.gap_m += (self.v_target - v_you) * dt
        if abs(pace_sec_per_mi - self.target) < 0.01 * self.target:
            self.gap_m *= (1.0 - 0.35 * dt)          # gentle settle when level
        self.gap_m = max(-26.0, min(40.0, self.gap_m))
        self.dist_m += v_you * dt
        self.elapsed += dt

        self.dev += ((pace_sec_per_mi - self.target) - self.dev) * min(1.0, dt * 3.2)
        self._advance_state(dt)
        return self.state

    def _advance_state(self, dt):
        self._state_age += dt
        edge = self.band * self.target
        if self.state != HOLD:
            edge *= self.hysteresis
        want = HOLD
        if self.dev > edge:
            want = PUSH
        elif self.dev < -edge:
            want = EASE
        if want != self.state and self._state_age >= self.dwell_s:
            self.state = want
            self._state_age = 0.0

    def line_distance(self):
        """Where to draw it, in metres ahead. Clamped to what optics can show."""
        return max(3.6, min(44.0, self.hold_distance() + self.gap_m))

    def gap_seconds(self):
        return self.gap_m / self.v_target

    def flash_hz(self):
        """0 = solid. Ramps with how far back you are. Capped at 3 Hz."""
        if self.state != PUSH:
            return 0.0
        k = min(1.0, (abs(self.dev) - self.band * self.target) / (0.11 * self.target))
        return 0.9 + k * 2.1

    def width_scale(self):
        """Line contracts toward centre when you are ahead of pace.

        Started as legibility, turns out to halve the horizontal FOV the
        nearest position demands -- which is what makes the near end fit
        inside a shipping-class waveguide at all.
        """
        if self.state != EASE:
            return 1.0
        k = min(1.0, (abs(self.dev) - self.band * self.target) / (0.11 * self.target))
        return 1.0 - 0.62 * k
