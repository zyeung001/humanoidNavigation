#!/usr/bin/env python3
# fsr_monitor.py  --  RUN ON THE PI
"""
Live FSR readout through the ADS1115 (bus 1, 0x48). NO servos are touched.

This is the bench tool for bringing up foot force sensors: it streams the raw divider
voltage, the implied FSR resistance, and -- once two channels are wired -- the fore/aft
centre of pressure. Use it to (a) pick R_fixed, (b) prove the connector is solid, and
(c) confirm heel/toe respond the right way before anything goes into a control loop.

WIRING assumed (one divider per sensor, R_fixed at the BREADBOARD end):

    3.3V ---[ FSR ]---+--- ADS1115 AINx
                      |
                      +---[ R_fixed ]--- GND

So V rises with force. If your voltage FALLS when you press, you built the other
arrangement (FSR to GND) -- pass --inverted so the resistance maths still works.

CHOOSING R_fixed: the divider is steepest where R_fixed ~= R_fsr at your working load.
This robot is 2046 g, so a sensor carrying half a foot's share sees ~5 N, where an
FSR-402 sits around 3-4 kOhm -- NOT the 10 kOhm every finger-press tutorial uses. Run
with --rfixed set to whatever you actually soldered; the printed R_fsr column is what
you use to judge it. Aim for the standing load to land near mid-scale (~1.5-2.3 V).

CAUTION on interpreting this: FSRs are +-15-25% part to part and creep under sustained
load, so the "load" column is a CONDUCTANCE PROXY (1/R), monotonic with force but NOT
calibrated newtons. That is fine for centre of pressure, which only needs the ratio
between two sensors. Do not treat it as a force measurement without a weight calibration.

WHY "RAIL" APPEARS, and why it is not a reading: two hard points under a rigid foot
share load linearly only while BOTH touch the ground. The moment the load line moves
outside the span between them, one point lifts and takes zero, and the CoP expression
pins at +-span/2 with no gradual approach. A pinned value means "past the edge of what
these two sensors can see", not "the CoP is exactly at the sensor". The tool prints RAIL
rather than a number so a saturated frame is never mistaken for a measurement. Softening
the contact (thin rubber or foam under each puck) makes load transfer gradually and is
the cheapest way to get a graded signal instead of a cliff.

CONNECTOR CHECK (do this first, every session): wiggle the connector WITHOUT touching
the sensor. If the reading moves, your contact resistance is moving, and a flaky contact
in a resistive divider is indistinguishable from force -- an open circuit reads as
"no load", i.e. a phantom "this foot is off the ground".

  python3 scripts/deploy/fsr_monitor.py --channels 0 --rfixed 2000
  python3 scripts/deploy/fsr_monitor.py --channels 0,1 --rfixed 2000 --span 80 --avg 10
  python3 scripts/deploy/fsr_monitor.py --channels 0,1 --rfixed 2000 \
      --pos-heel -33.7 --pos-toe 40.0        # measured, not nominal
"""
import argparse
import time
from collections import deque

ADS_ADDR = 0x48
REG_CONV = 0x00
REG_CONFIG = 0x01

# OS=1 (start), PGA=001 (+-4.096V, covers a 3.3V rail), MODE=1 (single-shot),
# DR=111 (860 SPS), COMP_QUE=11 (disabled). Channel is OR'd in at bits 14:12.
CONFIG_BASE = 0xC3E3
PGA_FULL_SCALE = 4.096


def _open_bus(busnum: int):
    try:
        import smbus
    except ImportError:
        import smbus2 as smbus
    return smbus.SMBus(busnum)


def read_channel(bus, addr: int, ch: int) -> float:
    """One single-shot single-ended conversion on AINch. Returns volts."""
    cfg = CONFIG_BASE | (ch << 12)
    bus.write_i2c_block_data(addr, REG_CONFIG, [(cfg >> 8) & 0xFF, cfg & 0xFF])
    time.sleep(0.002)  # 860 SPS -> ~1.2 ms; a mux switch costs a full conversion
    r = bus.read_i2c_block_data(addr, REG_CONV, 2)
    raw = (r[0] << 8) | r[1]
    if raw > 0x7FFF:
        raw -= 0x10000
    return raw * PGA_FULL_SCALE / 32768.0


def fsr_resistance(volts: float, rfixed: float, vcc: float, inverted: bool) -> float:
    """Divider inverse. Returns ohms, or inf when the sensor is unloaded/open."""
    v = vcc - volts if inverted else volts
    if v <= 1e-4:
        return float("inf")
    if v >= vcc - 1e-4:
        return 0.0
    return rfixed * (vcc - v) / v


def main():
    p = argparse.ArgumentParser(description="Live FSR readout via ADS1115")
    p.add_argument("--channels", default="0", help="comma-separated AIN indices, e.g. '0,1'")
    p.add_argument("--rfixed", type=float, default=3300.0, help="divider resistor, ohms")
    p.add_argument("--vcc", type=float, default=3.3, help="divider supply rail, volts")
    p.add_argument("--inverted", action="store_true", help="FSR wired to GND instead of Vcc")
    p.add_argument("--span", type=float, default=80.0,
                   help="heel-to-toe spacing in mm, assumed SYMMETRIC about the foot centre")
    p.add_argument("--pos-heel", type=float, default=None,
                   help="measured heel sensor position, mm from foot centre (negative = rearward). "
                        "Overrides --span; use when the sensors are not symmetric.")
    p.add_argument("--pos-toe", type=float, default=None,
                   help="measured toe sensor position, mm from foot centre (positive = forward)")
    p.add_argument("--avg", type=int, default=1,
                   help="moving average over N samples (averages LOAD, then derives CoP)")
    p.add_argument("--rail-frac", type=float, default=0.05,
                   help="a channel carrying less than this fraction of the total is 'unloaded'")
    p.add_argument("--bus", type=int, default=1)
    p.add_argument("--addr", type=lambda s: int(s, 0), default=ADS_ADDR)
    p.add_argument("--hz", type=float, default=20.0)
    args = p.parse_args()

    chans = [int(c) for c in args.channels.split(",") if c.strip() != ""]
    if not all(0 <= c <= 3 for c in chans):
        raise SystemExit("channels must be in 0..3")

    # Sensor positions along the foot, mm from the FOOT's centre (not from each other).
    # CoP is a load-weighted average of these, so an off-centre pair biases every reading
    # unless the real positions are given -- measure them, do not assume the nominal.
    x_heel = args.pos_heel if args.pos_heel is not None else -args.span / 2.0
    x_toe = args.pos_toe if args.pos_toe is not None else args.span / 2.0
    if x_heel >= x_toe:
        raise SystemExit("heel position must be rearward of (less than) the toe position")

    bus = _open_bus(args.bus)
    print(f"ADS1115 bus {args.bus} @ 0x{args.addr:02X}, channels {chans}, "
          f"R_fixed={args.rfixed:.0f} ohm, Vcc={args.vcc:.2f} V"
          + (f", averaging {args.avg} samples" if args.avg > 1 else ""))
    if len(chans) == 2:
        mid = 0.5 * (x_heel + x_toe)
        print(f"ch{chans[0]}=HEEL at {x_heel:+.1f} mm, ch{chans[1]}=TOE at {x_toe:+.1f} mm "
              f"from foot centre (readable range {x_heel:+.1f}..{x_toe:+.1f} mm)")
        if abs(mid) > 1.0:
            print(f"  NOTE: pair is off-centre by {mid:+.1f} mm; readings are reported "
                  f"about the FOOT centre, so this offset is already accounted for.")
    print("Press a sensor and watch V rise. Ctrl-C to stop.\n")

    period = 1.0 / max(args.hz, 1e-3)
    history = deque(maxlen=max(args.avg, 1))
    try:
        while True:
            volts = [read_channel(bus, args.addr, c) for c in chans]
            res = [fsr_resistance(v, args.rfixed, args.vcc, args.inverted) for v in volts]
            history.append([0.0 if r == float("inf") else 1.0 / max(r, 1e-6) for r in res])
            # Average LOAD, never CoP: a railing signal averages to a plausible-looking
            # mid value that never occurred, which is exactly the artefact to avoid.
            cond = [sum(s[i] for s in history) / len(history) for i in range(len(chans))]

            cells = []
            for c, v, g in zip(chans, volts, cond):
                rtxt = "  open " if g <= 1e-9 else f"{1.0 / g / 1000.0:7.2f}k"
                cells.append(f"A{c} {v:5.3f}V R={rtxt}")
            line = "  |  ".join(cells)

            total = sum(cond)
            line += f"  |  load {1000 * total:6.2f} mS"

            if len(chans) == 2:
                if total <= 1e-9:
                    line += "  |  CoP    --     NO LOAD"
                elif min(cond) / total < args.rail_frac:
                    off = "toe" if cond[1] < cond[0] else "heel"
                    line += f"  |  CoP   RAIL   ({off} unloaded)"
                else:
                    # Load-weighted mean of the two sensor positions: + = toward the toe,
                    # measured from the FOOT centre. Ratio-based, so the uncalibrated
                    # conductance scale cancels out.
                    cop = (cond[0] * x_heel + cond[1] * x_toe) / total
                    bias = "TOE " if cop > 5 else ("HEEL" if cop < -5 else "    ")
                    line += f"  |  CoP {cop:+6.1f} mm {bias}"

            print("\r" + line + "     ", end="", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
