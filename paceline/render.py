"""Draw a world-locked line on a head-mounted display.

The display is bolted to your head, so the line's screen position has to
counter-rotate against head attitude. Two things fight you:

  * the panel. At 60 Hz you spend 16.7 ms waiting for the next frame, which
    at running head-pitch rates (~60 deg/s peak) is 1.0 deg of error -- three
    times the entire budget. 120 Hz halves it.
  * you cannot late-latch through DisplayPort. Rendering on a host means you
    own the frame but not the scanout. PREDICTION has to close the rest, so
    --latency-ms is not a cosmetic knob; set it to your measured value.

Colours are the three states, but state is also encoded in FORM (solid /
broken+flashing / short+pulsing) because every efficient HUD light engine
shipping today is monochrome green, and because red-green is the worst
possible pair for the ~8% of men with red-green colour vision deficiency.
"""
from __future__ import annotations
import math

CAM_H = 1.42        # eye height above the road, m
LANE_HALF = 1.55    # half width of the drawn line, m

GO = (43, 224, 124)
PUSH_C = (59, 144, 255)
EASE_C = (255, 81, 64)
COLOURS = {"hold": GO, "push": PUSH_C, "ease": EASE_C}


class LineRenderer:
    def __init__(self, surface, px_per_deg=None, horizon_frac=0.38,
                 mono=False, flip_h=False, flip_v=False, grid=False):
        import pygame
        self.pg = pygame
        self.s = surface
        self.w, self.h = surface.get_size()
        # Xreal Air is ~46 deg diagonal on 1920x1080 -> ~40 deg horizontal.
        self.ppd = px_per_deg if px_per_deg else self.w / 40.0
        self.horizon0 = self.h * horizon_frac
        self.mono = mono
        self.flip_h, self.flip_v = flip_h, flip_v
        self.grid = grid          # fake road, for judging stability on a desk
        self.t = 0.0
        self.travelled = 0.0

    def _xy(self, d, lateral, pitch_deg, roll_deg):
        dep = math.degrees(math.atan(CAM_H / max(0.4, d)))
        half = math.degrees(math.atan(lateral / max(0.4, d)))
        y = self.horizon0 + (dep + pitch_deg) * self.ppd
        x = self.w / 2 + half * self.ppd
        if roll_deg:
            cx, cy = self.w / 2, self.horizon0
            a = math.radians(-roll_deg)
            dx, dy = x - cx, y - cy
            x = cx + dx * math.cos(a) - dy * math.sin(a)
            y = cy + dx * math.sin(a) + dy * math.cos(a)
        return x, y

    def _draw_grid(self, pitch_deg, roll_deg):
        """A stand-in for the road.

        On real glasses you look through them at actual tarmac and this is
        pointless. On a desk monitor it is the whole demo: the grid and the
        line are drawn in the same world frame, so a working filter keeps the
        line WELDED to the grid while both sweep with head motion. Switch to
        the naive filter and the line visibly detaches and jitters against it.
        """
        pg = self.pg
        dim = (26, 34, 32)
        for lat in (-3.2, 3.2):                       # road edges
            a = self._xy(4.0, lat, pitch_deg, roll_deg)
            b = self._xy(70.0, lat, pitch_deg, roll_deg)
            pg.draw.line(self.s, dim, a, b, 1)
        step = 5.0                                     # cross ticks every 5 m
        phase = self.travelled % step
        d = 5.0 - phase
        while d < 60.0:
            if d > 4.0:
                a = self._xy(d, -3.2, pitch_deg, roll_deg)
                b = self._xy(d, 3.2, pitch_deg, roll_deg)
                fade = max(0.15, 1.0 - d / 60.0)
                pg.draw.line(self.s, tuple(int(c * fade) for c in dim), a, b, 1)
            d += step

    def draw(self, pacer, pitch_deg, roll_deg, dt, reduce_motion=False):
        pg = self.pg
        self.t += dt
        self.travelled += pacer.v_target * dt if hasattr(pacer, "v_target") else 0.0
        self.s.fill((0, 0, 0))
        if self.grid:
            self._draw_grid(pitch_deg, roll_deg)

        d = pacer.line_distance()
        ws = pacer.width_scale()
        col = (255, 255, 255) if self.mono else COLOURS[pacer.state]

        alpha = 1.0
        hz = pacer.flash_hz()
        if hz and not reduce_motion:
            alpha = 0.46 + 0.54 * (0.5 + 0.5 * math.sin(self.t * hz * 2 * math.pi))
        elif pacer.state == "ease" and not reduce_motion:
            alpha = 0.80 + 0.20 * (0.5 + 0.5 * math.sin(self.t * 0.8 * 2 * math.pi))

        lx, ly = self._xy(d, -LANE_HALF * ws, pitch_deg, roll_deg)
        rx, ry = self._xy(d, LANE_HALF * ws, pitch_deg, roll_deg)
        shade = tuple(int(c * alpha) for c in col)

        if pacer.state == "push":
            # FORM, not just hue: broken bar reads as "push" on a mono panel
            n = 7
            for i in range(0, n, 2):
                a = i / n
                b = min(1.0, (i + 1) / n)
                p1 = (lx + (rx - lx) * a, ly + (ry - ly) * a)
                p2 = (lx + (rx - lx) * b, ly + (ry - ly) * b)
                pg.draw.line(self.s, shade, p1, p2, 7)
        else:
            pg.draw.line(self.s, shade, (lx, ly), (rx, ry),
                         11 if pacer.state == "ease" else 7)

        for px, py in ((lx, ly), (rx, ry)):          # end posts
            pg.draw.line(self.s, tuple(int(c * 0.55 * alpha) for c in col),
                         (px, py), (px, py - 22), 3)

        self._blit_flipped()

    def _blit_flipped(self):
        if self.flip_h or self.flip_v:
            flipped = self.pg.transform.flip(self.s.copy(), self.flip_h, self.flip_v)
            self.s.blit(flipped, (0, 0))
