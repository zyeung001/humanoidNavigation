#!/usr/bin/env python3
"""Read a frame log and say what happened, as a timeline you can paste.

WHY. The hardest part of debugging this robot has never been the data -- it has been
describing what you saw. "it falls forward", "it leans back a bit", "this one doesn't
work" are the inputs a diagnosis gets built on, and they lose the direction, the magnitude,
the moment and which joint did it. Meanwhile deploy_standing.py --log already records
every one of those to disk at 40 Hz and nobody reads it.

This reads it instead. It finds the moments that matter, times them, measures them and
names the joints involved, so the record of a run is a timeline rather than a memory. Pair
it with live_viewer.py --replay to watch any moment it points at:

    python scripts/debug/diagnose_run.py logs/2026-09-19_stand.csv
    python scripts/debug/live_viewer.py --replay logs/2026-09-19_stand.csv --from 11.8 --to 14.2

WHAT IT LOOKS FOR. Tip-overs and which way. Sustained lean, with direction. Oscillation,
by frequency, because a 1.96 Hz limit cycle and a slow drift need opposite fixes. Feet
lifting, by which end unloaded -- the pelvis IMU cannot see this, since the waist deflects
under load and a robot pitched onto its toe reports a BACKWARD lean. Joints not following
their command, and joints railed at a servo limit, which is silent on hardware: the servo
stops at a firmware boundary applying zero force and reporting no error. Loop stalls,
rejected servo reads and stale FSR samples, because a control problem and a bus problem
look identical from the outside.

It reports what it measured and stays quiet about what it did not. A run with nothing in
it should print almost nothing.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = ROOT / "config" / "joint_servo_map.yaml"

LEAN_DEG = 8.0          # sustained attitude past this is worth a line
TIP_COS = 0.5           # upright_cos below this is a tip-over (matches the env)
SETTLE_S = 0.4          # an excursion must last this long to count, not a single sample
TRACK_UNITS = 12.0      # ~3.5 deg of sustained command-vs-actual error on one joint
RAIL_FRAC = 0.02        # within 2% of a servo limit counts as railed
STALL_MULT = 1.8        # loop_ms past this multiple of the median is a stall


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path}: empty")
    return rows


def col(rows, name):
    """One column as float, NaN where blank. Empty columns are normal in these logs."""
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        v = r.get(name, "")
        if v not in ("", None):
            try:
                out[i] = float(v)
            except ValueError:
                pass
    return out


def runs_where(mask, t, min_s):
    """Contiguous stretches where mask holds for at least min_s. Returns (i0, i1) pairs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if t[i - 1] - t[start] >= min_s:
                out.append((start, i - 1))
            start = None
    if start is not None and t[-1] - t[start] >= min_s:
        out.append((start, len(mask) - 1))
    return out


def dominant_freq(sig, dt):
    """(freq_hz, amplitude, share_of_power) of the strongest non-DC component."""
    x = sig[np.isfinite(sig)]
    if x.size < 32:
        return None
    x = x - x.mean()
    win = np.hanning(x.size)
    sp = np.abs(np.fft.rfft(x * win))
    fr = np.fft.rfftfreq(x.size, dt)
    sp[0] = 0.0
    if sp.sum() <= 0:
        return None
    k = int(np.argmax(sp))
    # amplitude of a Hanning-windowed sinusoid: 4/N scales the bin back to peak amplitude
    return float(fr[k]), float(4.0 * sp[k] / x.size), float(sp[k] / sp.sum())


def fsr_conductance(volts, rfixed, vcc):
    """Divider volts -> conductance. 0 where the sensor is open (unloaded)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = rfixed * (vcc - volts) / np.maximum(volts, 1e-9)
    g = np.where(np.isfinite(r) & (r > 0), 1.0 / np.maximum(r, 1e-6), 0.0)
    return np.where(np.isfinite(volts), g, np.nan)


PUSH_RATE = 0.5         # rad/s of body rotation: well above quiet standing (~0.09 on 9/24)
CALM_RATE = 0.2         # ...and back under this, for CALM_S, counts as stopped moving
CALM_S = 0.5
HOME_DEG = 3.0          # back within this of where it stood before the push = recovered


def push_report(t, pitch, roll, pg, rows, end, v0, v1, args):
    """One line per disturbance: how hard, which way, and whether it came back by itself.

    "Did it correct itself" has two failure modes that look like success from across the
    room. It can stop moving somewhere else -- on 9/24 it settled 3.4 deg further back after
    a burst of touches, which is surviving, not correcting. And a hand can do the
    correcting: during every 9/24 disturbance the instrumented foot lost most of its load,
    which a supporting hand produces -- but so does the foot rolling onto an edge, since the
    sensors sit on two small pads. So the foot column reports what the sensors saw and does
    not claim to know which. The protocol has to make it unambiguous: tap, hands off.
    """
    av = np.linalg.norm(np.array([col(rows, f"av_{a}") for a in "xyz"]), axis=0)
    has_fsr = np.isfinite(v0).any() or np.isfinite(v1).any()
    if has_fsr:
        load = (np.nan_to_num(fsr_conductance(v0, args.rfixed, args.vcc))
                + np.nan_to_num(fsr_conductance(v1, args.rfixed, args.vcc)))
    busy = np.isfinite(av) & (av > PUSH_RATE)
    if not busy[:end].any():
        return None
    calm_n = max(1, int(round(CALM_S / max(float(np.median(np.diff(t))), 1e-3))))
    # Normal standing load = the median over every calm frame before any tip. Taking only
    # the frames before the first disturbance landed inside the startup ramp on 9/24 and
    # left the foot column blank.
    quiet_ref = (np.arange(len(t)) < end) & (av < CALM_RATE) & (t > t[0] + 3.0)
    if has_fsr:
        quiet_ref &= load > 0
    ref_load = float(np.median(load[quiet_ref])) if has_fsr and quiet_ref.any() else None

    lines, i, n_home, n_else, n_never = [], 0, 0, 0, 0
    while i < end:
        if not busy[i]:
            i += 1
            continue
        j = i                                    # end of the disturbance: rotation dies down
        while j < end and np.any(busy[j:j + calm_n]):
            j += 1
        pre = slice(max(0, i - 2 * calm_n), i)
        p0, r0 = float(np.nanmedian(pitch[pre])), float(np.nanmedian(roll[pre]))
        dp, dr = pitch[i:j + 1] - p0, roll[i:j + 1] - r0
        k = int(np.nanargmax(np.hypot(dp, dr)))
        size = float(np.hypot(dp[k], dr[k]))
        # pitch + = forward; roll + = the robot's LEFT (sim convention, verified on hardware
        # 9/26 against the gyro on two runs)
        way = (("forward" if dp[k] > 0 else "back") if abs(dp[k]) >= abs(dr[k])
               else ("to its left" if dr[k] > 0 else "to its right"))
        foot = ""
        if ref_load:
            low = float(np.min(load[i:j + 1])) < 0.3 * ref_load
            foot = "   foot unloaded (hand, or foot on its edge)" if low else "   foot kept its load"
        # where it ends up once it has actually stopped: search from the END of this push
        s = next((m for m in range(j, max(j, end - calm_n))
                  if np.all(av[m:m + calm_n] < CALM_RATE)), None)
        if s is None:
            verdict = "did NOT settle before the run ended"
            n_never += 1
        else:
            off = float(np.hypot(np.nanmedian(pitch[s:s + calm_n]) - p0,
                                 np.nanmedian(roll[s:s + calm_n]) - r0))
            if off <= HOME_DEG:
                verdict = f"came back in {t[s] - t[i]:.1f}s"
                n_home += 1
            else:
                verdict = f"stopped {off:.1f} deg away from where it was"
                n_else += 1
        lines.append(f"t={t[i]:5.1f}s  {size:5.1f} deg {way:12s} {verdict}{foot}")
        i = j + calm_n
    total = n_home + n_else + n_never
    lines.append(f"-> {n_home} of {total} came back to where they started, {n_else} stopped "
                 f"somewhere else, {n_never} never settled"
                 + ("; it tipped over" if end < len(t) else "; it never tipped over"))
    return lines


def main():
    p = argparse.ArgumentParser(description="Turn a frame log into a timeline of events")
    p.add_argument("log")
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--rfixed", type=float, default=2000.0)
    p.add_argument("--vcc", type=float, default=3.3)
    p.add_argument("--lean-deg", type=float, default=LEAN_DEG)
    args = p.parse_args()

    rows = load(args.log)
    t = col(rows, "t_rel")
    if not np.isfinite(t).any():
        raise SystemExit(f"{args.log}: no t_rel column -- is this a deploy frame log?")
    dt = float(np.median(np.diff(t[np.isfinite(t)]))) or 0.025
    dur = float(np.nanmax(t) - np.nanmin(t))
    dofs = [j["dof"] for j in yaml.safe_load(open(args.map))["joints"]]
    limits = {j["dof"]: j.get("servo_limit")
              for j in yaml.safe_load(open(args.map))["joints"]}

    print(f"\n{args.log}")
    print(f"  {len(rows)} frames over {dur:.1f}s "
          f"({len(rows) / max(dur, 1e-9):.1f} Hz)\n")

    pg = np.array([col(rows, f"pg_{a}") for a in "xyz"])
    pitch = np.degrees(np.arctan2(pg[0], -pg[2]))
    roll = np.degrees(np.arctan2(pg[1], -pg[2]))
    up = col(rows, "upright_cos")
    events = []   # (time, one-line description)

    # ---- tip-overs -----------------------------------------------------------------
    # Everything after a tip is the robot on the floor, and past 90 deg the atan2 wraps and
    # invents leans that never happened (a fall forward printed "leaned LEFT 178 deg").
    # So the tip ends the analysis window: attitude, oscillation and feet are read up to it.
    end = len(t)
    tipped = np.isfinite(up) & (up < TIP_COS)
    if tipped.any():
        i = int(np.argmax(tipped))
        way = ("FORWARD" if pitch[i] > abs(roll[i]) else
               "BACKWARD" if -pitch[i] > abs(roll[i]) else
               "to its LEFT" if roll[i] > 0 else "to its RIGHT")
        events.append((t[i], f"TIPPED OVER {way} (upright_cos {up[i]:.2f}, "
                             f"pitch {pitch[i]:+.1f} roll {roll[i]:+.1f}) "
                             f"-- nothing after this is a standing measurement"))
        end = i

    # ---- sustained lean ------------------------------------------------------------
    for name, sig in (("pitch", pitch[:end]), ("roll", roll[:end])):
        for lo, hi in runs_where(np.abs(sig) > args.lean_deg, t[:end], SETTLE_S):
            seg = sig[lo:hi + 1]
            k = lo + int(np.argmax(np.abs(seg)))
            # roll + = the robot's LEFT (sim convention; verified on hardware 9/26 against the gyro
            # on two runs). This used to say RIGHT, which is how a fall to its left got reported
            # as a fall to its right.
            way = {"pitch": ("FORWARD", "BACKWARD"), "roll": ("to its LEFT", "to its RIGHT")}[name]
            events.append((t[lo], f"leaned {way[0] if sig[k] > 0 else way[1]} past "
                                  f"{args.lean_deg:.0f} deg for {t[hi] - t[lo]:.1f}s, "
                                  f"peak {sig[k]:+.1f} deg at t={t[k]:.1f}"))

    # ---- oscillation ---------------------------------------------------------------
    for name, sig in (("pitch", pitch[:end]), ("roll", roll[:end])):
        f = dominant_freq(sig, dt)
        if f and f[2] > 0.12 and f[1] > 0.5 and f[0] > 0.2:
            events.append((t[0], f"{name} OSCILLATED at {f[0]:.2f} Hz, "
                                 f"amplitude {f[1]:.1f} deg "
                                 f"({100 * f[2]:.0f}% of the signal power)"))

    # ---- feet ----------------------------------------------------------------------
    v0, v1 = col(rows, "fsr_v0"), col(rows, "fsr_v1")
    if np.isfinite(v0).any() or np.isfinite(v1).any():
        g0, g1 = (fsr_conductance(v, args.rfixed, args.vcc) for v in (v0, v1))
        tot = g0 + g1
        for lab, dead in (("TOE", g1[:end] < 0.05 * np.maximum(tot[:end], 1e-9)),
                          ("HEEL", g0[:end] < 0.05 * np.maximum(tot[:end], 1e-9))):
            for lo, hi in runs_where(dead & (tot[:end] > 1e-9), t[:end], SETTLE_S):
                other = "HEEL" if lab == "TOE" else "TOE"
                events.append((t[lo], f"{lab} came off the ground for {t[hi] - t[lo]:.1f}s "
                                      f"-- standing on the {other}"))
        for lo, hi in runs_where(tot[:end] < 1e-9, t[:end], SETTLE_S):
            events.append((t[lo], f"instrumented foot carried NO load for "
                                  f"{t[hi] - t[lo]:.1f}s (lifted, or weight on the other)"))

    # ---- joints: following, and railed ----------------------------------------------
    for d in dofs:
        cmd, act = col(rows, f"u_cmd.{d}"), col(rows, f"u_meas.{d}")
        if not (np.isfinite(cmd).any() and np.isfinite(act).any()):
            continue
        err = np.abs(cmd - act)
        for lo, hi in runs_where(err > TRACK_UNITS, t, SETTLE_S):
            k = lo + int(np.nanargmax(err[lo:hi + 1]))
            events.append((t[lo], f"{d} did NOT follow its command for {t[hi] - t[lo]:.1f}s "
                                  f"(worst {err[k]:.0f} units = {err[k] / 195 * 57.3:.1f} deg "
                                  f"at t={t[k]:.1f})"))
        lim = limits.get(d)
        if lim:
            span = max(lim[1] - lim[0], 1)
            railed = (cmd <= lim[0] + RAIL_FRAC * span) | (cmd >= lim[1] - RAIL_FRAC * span)
            for lo, hi in runs_where(railed, t, SETTLE_S):
                which = "low" if cmd[lo] <= lim[0] + RAIL_FRAC * span else "high"
                events.append((t[lo], f"{d} was COMMANDED TO ITS {which.upper()} SERVO LIMIT "
                                      f"for {t[hi] - t[lo]:.1f}s -- the servo clips this "
                                      f"silently, with no force and no error"))

    pushes = push_report(t, pitch, roll, pg, rows, end, v0, v1, args)

    # ---- loop and bus health --------------------------------------------------------
    notes = []
    lm = col(rows, "loop_ms")
    if np.isfinite(lm).any():
        med = float(np.nanmedian(lm))
        stalls = int(np.nansum(lm > STALL_MULT * med))
        if stalls:
            notes.append(f"{stalls} loop stall(s), worst {np.nanmax(lm):.0f} ms "
                         f"against a {med:.0f} ms median")
    rej = col(rows, "rej_total")
    if np.isfinite(rej).any() and np.nanmax(rej) > 0:
        notes.append(f"{int(np.nanmax(rej))} rejected servo read(s) -- a bus problem, "
                     "not a control problem")
    age = col(rows, "fsr_age_ms")
    if np.isfinite(age).any() and np.nanmax(age) > 200:
        notes.append(f"FSR samples went stale (worst {np.nanmax(age):.0f} ms); "
                     "foot numbers near then are not trustworthy")

    # ---- report ---------------------------------------------------------------------
    if events:
        print("  WHAT HAPPENED")
        seen = set()
        for when, what in sorted(events, key=lambda e: e[0]):
            if what in seen:
                continue
            seen.add(what)
            print(f"    t={when:6.1f}s  {what}")
    else:
        print("  Nothing notable: no tip, no sustained lean, no joint failed to follow,")
        print("  no command hit a servo limit.")

    if pushes:
        print("\n  PUSHES -- did it correct itself?")
        for line in pushes:
            print(f"    {line}")

    if notes:
        print("\n  RIG HEALTH")
        for n in notes:
            print(f"    {n}")

    if events:
        first = min(e[0] for e in events)
        print(f"\n  Watch it: python scripts/debug/live_viewer.py --replay {args.log} "
              f"--from {max(0.0, first - 2):.1f} --to {min(dur, first + 6):.1f}")
    print(f"\n  While standing: pitch {np.nanmedian(pitch[:end]):+.2f} deg "
          f"(spread {np.nanstd(pitch[:end]):.2f}), roll {np.nanmedian(roll[:end]):+.2f} "
          f"(spread {np.nanstd(roll[:end]):.2f})"
          + (f"   [first {t[end - 1] - t[0]:.1f}s of {dur:.1f}s; the rest is post-fall]"
             if end < len(t) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
