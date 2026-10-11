#!/usr/bin/env python
"""The ODESC on the bench: is it there, what is it set to, does it survive a cut?

    ./config/odrive-check.sh            report: serial, firmware, bus voltage,
                                        the bus-limit config, errors
    ./config/odrive-check.sh --apply    write the cart's bus limits (below),
                                        save, reboot, re-read and compare
    ./config/odrive-check.sh --watch    poll the board until Ctrl-C and report
                                        every disappearance and return - the
                                        XH-M609 relay-drop test

Nothing here touches the axes. No motor calibration, no closed loop, nothing
that needs the wheels or more than a few milliamps from the supply, so it is
safe on a bench supply, on the drill pack, and on the C300.

The limits are the ones recorded in the power doc for firmware 0.5.1, which
has no enable_brake_resistor: the resistor is in use whenever
brake_resistance > 0. They are written as a dict so --apply and the report's
"differs from target" column cannot drift apart.
"""

import argparse
import sys
import time

TARGET = {
    "brake_resistance": 2.0,                 # the fitted 2 ohm / 50 W
    "dc_max_positive_current": 10.0,         # XH-M609 relay rating at 16-21 V
    "dc_max_negative_current": -0.5,         # fault threshold; regen goes to the resistor
    "max_regen_current": 0.0,                # none back into the pack
    "dc_bus_undervoltage_trip_level": 16.5,  # above the XH-M609's 16.0 V cut
    "dc_bus_overvoltage_trip_level": 26.0,   # 5S pack maxes at 21 V
    "enable_dc_bus_overvoltage_ramp": False,
}


def _find(timeout: float):
    import odrive
    return odrive.find_any(timeout=timeout)


def _report(o) -> bool:
    """Print the state. Returns True when every target field matches."""
    from odrive.utils import dump_errors

    print(f"serial   {o.serial_number}")
    print(f"firmware {o.fw_version_major}.{o.fw_version_minor}.{o.fw_version_revision}")
    print(f"vbus     {o.vbus_voltage:.2f} V")
    armed = getattr(o, "brake_resistor_armed", None)
    if armed is not None:
        print(f"brake resistor armed: {armed}")
    print()
    ok = True
    print(f"  {'field':<34}{'now':>10}   {'target':>8}")
    for k, want in TARGET.items():
        have = getattr(o.config, k, None)
        match = have is not None and (have == want or (isinstance(want, float) and abs(have - want) < 1e-6))
        ok = ok and match
        flag = "" if match else "   <- differs"
        print(f"  {k:<34}{str(have):>10}   {str(want):>8}{flag}")
    print()
    for i, ax in enumerate((o.axis0, o.axis1)):
        print(f"  axis{i}: state {ax.current_state}  error 0x{ax.error:04x}  "
              f"motor 0x{ax.motor.error:04x}  encoder 0x{ax.encoder.error:04x}")
    print()
    dump_errors(o)
    return ok


def cmd_report(args) -> int:
    o = _find(args.timeout)
    ok = _report(o)
    print("config matches the cart's bus limits." if ok else
          "config differs from the cart's bus limits - run with --apply.")
    return 0 if ok else 1


def cmd_apply(args) -> int:
    o = _find(args.timeout)
    print(f"serial {o.serial_number}, vbus {o.vbus_voltage:.2f} V - writing bus limits")
    for k, v in TARGET.items():
        if not hasattr(o.config, k):
            print(f"  this firmware has no config.{k}; skipping", file=sys.stderr)
            continue
        setattr(o.config, k, v)
        print(f"  {k} = {v}")
    o.save_configuration()
    print("saved. rebooting - the USB device goes away for a few seconds, which is normal.")
    try:
        o.reboot()
    except Exception:
        pass  # the connection is torn down by the reboot; that is the exception
    time.sleep(2.0)
    print("\nre-reading:\n")
    o = _find(max(args.timeout, 20.0))
    ok = _report(o)
    print("applied and verified." if ok else "SOMETHING DID NOT STICK - see the 'differs' rows.")
    return 0 if ok else 1


def cmd_watch(args) -> int:
    """Poll the board; say when it vanishes and when it is back.

    This is the XH-M609 test: with the board behind the module, raise the
    cut-off threshold above the pack voltage, confirm the board drops here,
    lower it, confirm the board comes back - and that it comes back to
    IDLE with no error, which is what the robot's re-arm logic expects.
    """
    import odrive
    print("watching; Ctrl-C to stop. Now trip the XH-M609 and restore it.\n")
    o = None
    present = False
    t_lost = None
    n_lost = 0
    try:
        while True:
            try:
                if o is None:
                    o = odrive.find_any(timeout=1.0)
                v = o.vbus_voltage
                states = (o.axis0.current_state, o.axis1.current_state)
                errs = (o.axis0.error, o.axis1.error)
                if not present:
                    present = True
                    gone = "" if t_lost is None else f" after {time.perf_counter() - t_lost:.1f} s away"
                    print(f"{time.strftime('%H:%M:%S')}  board present{gone}: vbus {v:.2f} V, "
                          f"axis states {states}, errors {errs}", flush=True)
                else:
                    print(f"\r{time.strftime('%H:%M:%S')}  vbus {v:6.2f} V  states {states}  "
                          f"errors {errs}   ", end="", flush=True)
                time.sleep(0.5)
            except (TimeoutError, Exception) as e:  # fibre: ChannelBrokenException, ObjectLostError
                if present:
                    present = False
                    t_lost = time.perf_counter()
                    n_lost += 1
                    print(f"\n{time.strftime('%H:%M:%S')}  board GONE ({type(e).__name__})", flush=True)
                o = None
                time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n\n{n_lost} loss/return cycle(s) seen.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true", help="write the bus limits, save, reboot, verify")
    g.add_argument("--watch", action="store_true", help="poll until Ctrl-C; report loss and return")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for the board")
    args = ap.parse_args()
    try:
        if args.apply:
            return cmd_apply(args)
        if args.watch:
            return cmd_watch(args)
        return cmd_report(args)
    except TimeoutError:
        print("no ODrive found. Is the board powered (its USB only works with DC on the "
              "terminals), and is the udev rule installed? `lsusb | grep 1209`", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
