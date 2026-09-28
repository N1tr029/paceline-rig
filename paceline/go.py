"""One command. Plug the glasses in and run this.

  python -m paceline.go

Finds the glasses' display, finds and decodes their IMU, calibrates level,
and launches fullscreen. Everything is overridable; nothing has to be.

  python -m paceline.go --target 6:46 --latency-ms 8.3
  python -m paceline.go --sim            # rehearse with no hardware
"""
from __future__ import annotations
import argparse, sys


def parse_pace(v):
    """Accept 6:46 or 406."""
    if ":" in str(v):
        m, s = str(v).split(":")
        return int(m) * 60 + float(s)
    return float(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description="plug in and run")
    ap.add_argument("--target", default="6:46", help="target pace, m:ss per mile")
    ap.add_argument("--pace", default=None, help="your pace; omit for the scripted demo")
    ap.add_argument("--latency-ms", type=float, default=8.3,
                    help="MEASURED motion-to-photon. 120Hz=8.3, 90Hz=11.1, 60Hz=16.7")
    ap.add_argument("--ppd", type=float, default=None, help="pixels per degree (measure it)")
    ap.add_argument("--display", type=int, default=None, help="override auto-detect")
    ap.add_argument("--sim", action="store_true", help="no hardware; rehearse the launch")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--grid", action="store_true",
                    help="draw the stand-in road. Off by default here, because "
                         "through real glasses you are looking at a real one")
    a = ap.parse_args(argv)

    from . import doctor
    args = ["--filter", "gated",
            "--target", str(parse_pace(a.target)),
            "--latency-ms", str(a.latency_ms)]
    if a.pace:
        args += ["--pace", str(parse_pace(a.pace))]
    if a.ppd:
        args += ["--ppd", str(a.ppd)]
    if not a.grid:
        args += ["--no-grid"]

    if a.sim:
        print("rehearsal: simulated motion, windowed, stand-in road on\n")
        args += ["--source", "sim"]
        if "--no-grid" in args:
            args.remove("--no-grid")
    else:
        print("looking for the glasses...\n")
        idx = a.display if a.display is not None else doctor.check_displays()
        if idx is None:
            print("\nNo second display. The glasses need a USB-C port that carries "
                  "DisplayPort -\nnot every port does. Run `python -m paceline.doctor` "
                  "for the full report,\nor rehearse with `python -m paceline.go --sim`.")
            return 1
        print()
        args += ["--source", "xreal", "--display", str(idx)]
        if not a.windowed:
            args += ["--fullscreen"]

    print("F = swap filter    G = stand-in road    M = mono    C = recalibrate    Q = quit\n")
    from . import app
    return app.main(args)


if __name__ == "__main__":
    sys.exit(main())
