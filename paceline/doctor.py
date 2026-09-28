"""Preflight. Run this before you trust anything.

  python -m paceline.doctor

Checks, in the order that will actually bite you:
  1. python deps
  2. displays -- are the glasses showing up as a monitor, and at what rate
  3. HID -- can we open the IMU interface at all (permissions live here)
  4. decode -- does the data look like real physics
"""
from __future__ import annotations
import sys

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def _p(state, msg, fix=None):
    print(f"[{state}] {msg}")
    if fix:
        for line in fix.strip().splitlines():
            print(f"         {line.strip()}")


def check_deps():
    good = True
    try:
        import numpy  # noqa
        _p(OK, "numpy")
    except ImportError:
        good = False
        _p(BAD, "numpy missing", "pip install numpy")
    try:
        import pygame
        if pygame.version.ver.startswith("2.6") and sys.version_info >= (3, 14):
            try:
                pygame.init(); pygame.font.SysFont("monospace", 12)
                _p(OK, f"pygame {pygame.version.ver}")
            except Exception:
                _p(WARN, f"pygame {pygame.version.ver} font module broken on "
                         f"python {sys.version_info.major}.{sys.version_info.minor}",
                   "pip uninstall pygame && pip install pygame-ce\n"
                   "(the app still runs, just without the text readout)")
        else:
            _p(OK, f"pygame {pygame.version.ver}")
    except ImportError:
        good = False
        _p(BAD, "pygame missing", "pip install pygame-ce")
    try:
        import hid  # noqa
        _p(OK, "hidapi")
    except ImportError:
        _p(WARN, "hidapi missing - simulator works, real glasses won't",
           "pip install hidapi")
    return good


def find_glasses_display():
    """Return (index, size) of the display most likely to be the glasses."""
    import pygame
    pygame.init()
    try:
        sizes = pygame.display.get_desktop_sizes()
    except Exception:
        return None, []
    best = None
    for i, sz in enumerate(sizes):
        if i == 0:
            continue
        if sz == (1920, 1080):
            best = i
    if best is None and len(sizes) > 1:
        best = 1
    return best, sizes


def check_displays():
    idx, sizes = find_glasses_display()
    if not sizes:
        _p(WARN, "could not enumerate displays")
        return None
    for i, sz in enumerate(sizes):
        tag = "  <- glasses?" if i == idx else ("  (primary)" if i == 0 else "")
        _p(OK, f"display {i}: {sz[0]}x{sz[1]}{tag}")
    if idx is None:
        _p(WARN, "only one display found - glasses not plugged in, or your "
                 "USB-C port doesn't carry DisplayPort",
           "Not every USB-C port does video. Look for a 'D' or lightning\n"
           "symbol next to the port, or try another one.")
    else:
        _p(OK, f"will render to display {idx}")
    return idx


def check_imu(verbose=True):
    try:
        import hid
    except ImportError:
        _p(WARN, "skipping IMU check - hidapi not installed")
        return None
    from .xreal import enumerate_candidates, probe

    devs = enumerate_candidates()
    if not devs:
        _p(BAD, "no Xreal/Nreal HID device found",
           "Plugged in? On Linux add a udev rule for vendor 3318.\n"
           "On macOS, System Settings > Privacy > Input Monitoring\n"
           "may need to allow your terminal.")
        return None
    for d in devs[:4]:
        _p(OK, f"HID {d['vendor_id']:#06x}:{d['product_id']:#06x} "
               f"interface {d.get('interface_number')} "
               f"{(d.get('product_string') or '').strip()}")

    print("\n  hold the glasses STILL for a moment...")
    for d in devs[:3]:
        try:
            dev = hid.Device(path=d["path"])
        except Exception as e:
            _p(WARN, f"interface {d.get('interface_number')}: cannot open ({e})")
            continue
        try:
            layout = probe(dev, verbose=verbose)
            _p(OK, f"IMU readable on interface {d.get('interface_number')} "
                   f"as '{layout.name}'")
            dev.close()
            return d, layout
        except Exception as e:
            _p(WARN, f"interface {d.get('interface_number')}: {e}")
            dev.close()
    _p(BAD, "found the device but couldn't decode a plausible IMU stream")
    return None


def main(argv=None):
    print("paceline doctor\n")
    print("dependencies"); check_deps(); print()
    print("displays"); idx = check_displays(); print()
    print("imu"); imu = check_imu(); print()
    if imu and idx is not None:
        print("ready. launch with:")
        print(f"  python -m paceline.go")
    else:
        print("not ready for glasses yet - the simulator still works:")
        print("  python -m paceline.app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
