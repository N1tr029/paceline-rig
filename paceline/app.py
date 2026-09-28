"""Live pace line. Source -> filter -> pacer -> display.

  python -m paceline.app                                   # simulated, windowed
  python -m paceline.app --source xreal --display 1 --fullscreen
  python -m paceline.app --source replay --file data/run.csv

Keys
  F   switch filter (gated <-> naive) -- the demo. Press it while moving.
  M   monochrome, as a real green-only light engine would look
  C   recalibrate level
  H/V mirror, if your combiner flips the image
  Q   quit
"""
from __future__ import annotations
import argparse, math, sys, time

from .sources import make_source, SimSource
from .attitude import GatedFilter, NaiveFilter
from .pace import Pacer
from .render import LineRenderer


def main(argv=None):
    ap = argparse.ArgumentParser(description="live pace line")
    ap.add_argument("--source", default="sim", choices=["sim", "xreal", "replay"])
    ap.add_argument("--file", help="csv, for --source replay")
    ap.add_argument("--filter", default="gated", choices=["gated", "naive"])
    ap.add_argument("--display", type=int, default=0, help="monitor index; the glasses are usually 1")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--target", type=float, default=406.0, help="target pace, sec/mile")
    ap.add_argument("--pace", type=float, default=None, help="your pace, sec/mile (default: scripted demo)")
    ap.add_argument("--latency-ms", type=float, default=8.3,
                    help="MEASURED motion-to-photon. Prediction horizon. 120Hz=8.3, 60Hz=16.7")
    ap.add_argument("--ppd", type=float, default=None, help="pixels per degree; measure it, don't guess")
    ap.add_argument("--mono", action="store_true")
    a = ap.parse_args(argv)

    try:
        import pygame
    except ImportError:
        print("pygame not installed:  pip install pygame"); return 1

    src = make_source(a.source, file=a.file) if a.source != "replay" else make_source("replay", file=a.file)
    filt = {"gated": GatedFilter, "naive": NaiveFilter}[a.filter]()
    other = {"gated": NaiveFilter, "naive": GatedFilter}[a.filter]()
    pacer = Pacer(a.target)

    pygame.init()
    flags = pygame.FULLSCREEN if a.fullscreen else 0
    screen = pygame.display.set_mode((1280, 720), flags, display=a.display)
    pygame.display.set_caption("pace line")
    pygame.mouse.set_visible(False)
    font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15)

    rend = LineRenderer(screen, px_per_deg=a.ppd, mono=a.mono)
    horizon = a.latency_ms / 1000.0
    clock = pygame.time.Clock()
    it = iter(src)
    t_prev = None
    hud = True
    running = True

    # scripted effort so it does something interesting with no runner attached
    KF = [(0, 0), (5, 0), (9, 78), (19, 78), (24, 0), (28, 0),
          (31, -78), (40, -78), (44, 0), (50, 0)]

    def scripted(t):
        x = t % 50
        for i in range(1, len(KF)):
            if x <= KF[i][0]:
                (t0, v0), (t1, v1) = KF[i - 1], KF[i]
                u = (x - t0) / (t1 - t0)
                u = u * u * (3 - 2 * u)
                return v0 + (v1 - v0) * u
        return 0.0

    t0 = time.perf_counter()
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key == pygame.K_f:
                    filt, other = other, filt
                elif e.key == pygame.K_m:
                    rend.mono = not rend.mono
                elif e.key == pygame.K_c:
                    if hasattr(filt, "prime"):
                        filt.prime(last_accel)
                elif e.key == pygame.K_h:
                    rend.flip_h = not rend.flip_h
                elif e.key == pygame.K_v:
                    rend.flip_v = not rend.flip_v
                elif e.key == pygame.K_TAB:
                    hud = not hud

        # drain everything the sensor has produced since the last frame
        n = 0
        try:
            while n < 64:
                t, g, acc = next(it)
                dt = 0.0025 if t_prev is None else max(1e-5, min(0.05, t - t_prev))
                t_prev = t
                filt.update(g, acc, dt)
                other.update(g, acc, dt)
                last_accel = acc
                n += 1
                if isinstance(src, SimSource) and n >= 8:
                    break
        except StopIteration:
            running = False

        frame_dt = clock.get_time() / 1000.0 or 1 / 60
        wall = time.perf_counter() - t0
        pace = a.pace if a.pace is not None else a.target + scripted(wall)
        pacer.update(pace, frame_dt)

        # predict forward by the display latency you cannot late-latch away
        if hasattr(filt, "predict"):
            pitch, roll, _ = filt.predict(horizon)
        else:
            pitch, roll = filt.pitch, filt.roll

        rend.draw(pacer, pitch, roll, frame_dt)

        if hud:
            lines = [
                f"{filt.name.upper():6s}  pitch {filt.pitch:+6.2f}  (other {other.pitch:+6.2f})",
                f"state {pacer.state:5s}  gap {pacer.gap_m:+5.1f} m  line {pacer.line_distance():4.1f} m",
                f"pace {int(pace)//60}:{int(pace)%60:02d}/mi   target {int(a.target)//60}:{int(a.target)%60:02d}",
                f"predict {a.latency_ms:.1f} ms   {clock.get_fps():4.0f} fps   [F]ilter [M]ono [C]al [TAB]hud",
            ]
            for i, s in enumerate(lines):
                screen.blit(font.render(s, True, (70, 96, 90)), (14, 12 + i * 19))

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
