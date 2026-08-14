#!/usr/bin/env python3
# excite_joint.py  --  RUN ON THE PI
"""Drive ONE joint with a known waveform and record what the servo actually did.

This is the system-identification counterpart to the deploy logs. Standing logs are
nearly information-free about the plant: quiet-standing joint speeds sit BELOW one
encoder quantum (0.205 rad/s at 40 Hz), so the velocity channel reads 0 or +-0.205
with nothing in between and the actuator is barely excited. Hours of standing data
therefore teach you almost nothing about delay, lag or backlash. Deliberate
excitation of one joint at a time teaches you all three in about two minutes.

MODES
  step      square steps at several amplitudes, both directions -> dead time + tau.
            The bench number to reproduce is ~50 ms dead time + ~120 ms tau.
  chirp     sine sweep f0 -> f1 -> the frequency response, and with it the PHASE
            MARGIN that every oscillation on this robot has been traced to. Lag of
            180 deg is predicted near 1/(2T) ~ 2.9 Hz; sweeping through it measures
            directly what has so far only been inferred.
  backlash  slow triangle with reversals. Gear lash shows up as a flat region at
            each reversal where the command moves and the output does not -- and
            backlash is a limit-cycle generator in its own right.
  all       runs the three above in sequence, one log file each.

SAMPLING: a single servo read is ~0.4 ms, so this samples at --sample-hz (default
200) rather than the deploy loop's 40 Hz. That matters: at 40 Hz a 50 ms dead time
is two samples, which cannot separate delay from lag. At 200 Hz it is ten.

MEASUREMENT CONVENTION: each tick reads the position FIRST, then issues that tick's
command. So u_meas[k] responds to u_cmd[k-1] and earlier. That is a one-sample
bookkeeping offset (1/sample_hz), not a physical dead time -- subtract it before
quoting a delay.

SAFETY
  - Support the robot, or run it on joints that cannot drop it. Nothing here balances.
  - Only the named joint moves. Every other joint is held at its current position
    with torque on (--relax-others to leave them limp instead).
  - Commands are clipped to the joint's mapped servo_limit, and CLIPPED SAMPLES ARE
    COUNTED AND REPORTED: a clipped waveform silently invalidates the identification,
    which is the same failure class as the ctrlrange and EEPROM-limit bugs already
    found on this robot.
  - Ctrl-C ramps back to the starting pose and disables torque.
  - --dry-run prints the waveform and touches no hardware.

  python3 scripts/deploy/excite_joint.py --joint R_knee --mode all --amp-deg 8
  python3 scripts/deploy/excite_joint.py --joint R_shoulder_pitch --mode chirp --f1 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_real_map import SimRealMap, DEFAULT_MAP  # noqa: E402

MAX_AMP_DEG = 45.0     # refuse obviously wrong amplitudes before they reach a servo


def fit_centre(requested, lo, hi, amp_units):
    """Shift the excitation centre inward so the full swing fits inside the joint limit.

    Necessary because several joints on this robot are ONE-SIDED hinges: the knees and
    elbows have their straight pose sitting exactly on a firmware stop (R_knee is
    206..512 with straight at 512). Exciting symmetrically about straight would spend
    half the waveform clipped against the stop, which does not identify a plant -- it
    identifies a saturation. Returns (centre, shifted_by).
    """
    if hi - lo < 2 * amp_units:            # amplitude cannot fit at all; clip guard reports it
        return 0.5 * (lo + hi), 0.5 * (lo + hi) - requested
    centre = float(np.clip(requested, lo + amp_units, hi - amp_units))
    return centre, centre - requested


def waveform(mode, t, amp_units, args):
    """Command offset from centre, in servo units, at times t (seconds)."""
    if mode == "step":
        # Symmetric staircase: 0, +A, 0, -A, ... at several amplitude fractions, so one
        # run yields both directions and tests whether the response is amplitude-linear.
        out = np.zeros_like(t)
        seq = [f * s for f in (1.0, 0.5, 1.0) for s in (+1, -1)]
        for i, sgn in enumerate(seq):
            lo = 2 * i * args.hold
            out[(t >= lo + args.hold) & (t < lo + 2 * args.hold)] = sgn * amp_units
        return out
    if mode == "chirp":
        # Linear frequency sweep. Starts at sin(0)=0 so there is no step at t=0.
        T = max(args.secs, 1e-6)
        phase = 2 * np.pi * (args.f0 * t + (args.f1 - args.f0) * t**2 / (2 * T))
        return amp_units * np.sin(phase)
    if mode == "backlash":
        # Slow triangle. Speed is deliberately low so the reversal is dominated by lash
        # rather than by actuator lag -- the two look alike if you drive it fast.
        ph = (t / args.period) % 1.0
        return amp_units * (4 * np.abs(ph - 0.5) - 1.0)
    raise ValueError(mode)


def mode_duration(mode, args):
    if mode == "step":
        return 12 * args.hold
    if mode == "chirp":
        return args.secs
    return args.cycles * args.period


def run_mode(mode, args, m, idx, bus, imu, centre_units, log_dir):
    from frame_log import FrameLogger, default_log_path  # noqa: PLC0415

    dof = m.joints[idx].dof
    sid = int(m.servo_ids[idx])
    amp_units = args.amp_deg * np.pi / 180.0 * m.units_per_rad
    lo, hi = float(m.lim_lo[idx]), float(m.lim_hi[idx])
    dt = 1.0 / args.sample_hz
    n = int(mode_duration(mode, args) / dt)
    t = np.arange(n) * dt
    cmd = centre_units + waveform(mode, t, amp_units, args)
    clipped = int(np.sum((cmd < lo) | (cmd > hi)))
    cmd = np.clip(cmd, lo, hi).round().astype(int)

    print(f"\n=== {mode}: {dof} (servo {sid}), {n} samples @ {args.sample_hz:.0f} Hz, "
          f"{n * dt:.1f} s, amp {args.amp_deg:.1f} deg = {amp_units:.0f} units ===")
    print(f"    centre {centre_units:.0f}, command range {cmd.min()}..{cmd.max()}, "
          f"servo limit {lo:.0f}..{hi:.0f}")
    if clipped:
        pct = 100.0 * clipped / n
        print(f"    !! {clipped} samples ({pct:.1f}%) CLIPPED at the joint limit. A clipped "
              f"waveform invalidates the fit -- lower --amp-deg or move --centre.")
        if pct > 2.0 and not args.allow_clip:
            print("    refusing to run (>2% clipped). Re-run with --allow-clip to override.")
            return None

    if args.dry_run:
        # Sample at a fixed 0.1 s, not n/12: spreading 12 samples over a 30 s sweep
        # aliases a 0.2 Hz sine into a flat line and makes a correct chirp look broken.
        step = max(1, int(0.1 * args.sample_hz))
        print("    [dry-run] first 1.2 s @ 0.1 s (units): "
              + " ".join(str(int(c)) for c in cmd[:12 * step:step]))
        return None

    logger = FrameLogger(default_log_path(f"excite_{dof}_{mode}", log_dir), [dof], meta={
        "script": "excite_joint.py", "mode": mode, "joint": dof, "servo_id": sid,
        "sample_hz": args.sample_hz, "amp_deg": args.amp_deg, "centre_units": centre_units,
        "servo_limit": [lo, hi], "units_per_rad": float(m.units_per_rad),
        "sign": float(m.signs[idx]), "center": float(m.centers[idx]),
        "clipped_samples": clipped,
        "chirp": {"f0": args.f0, "f1": args.f1, "secs": args.secs} if mode == "chirp" else None,
        "backlash": {"period": args.period, "cycles": args.cycles} if mode == "backlash" else None,
        "step": {"hold": args.hold} if mode == "step" else None,
        "convention": "u_meas[k] is read BEFORE u_cmd[k] is written; jvel_f is a RAW "
                      "finite difference (rad/s), not the deploy EMA",
        "imu_logged": bool(imu),
    })

    prev_rad = None
    fails = 0
    t_start = time.time()
    try:
        for k in range(n):
            tick = t_start + k * dt
            now = time.time()
            if now < tick:
                time.sleep(tick - now)
            t0 = time.time()

            u = bus.read_pos(sid)
            if u is None or not np.isfinite(u):
                fails += 1
                u = float("nan")
            rad = float(m.units_to_rad(np.full(m.n, np.nan if not np.isfinite(u) else u))[idx]) \
                if np.isfinite(u) else float("nan")
            vel = (rad - prev_rad) / dt if (prev_rad is not None and np.isfinite(rad)) else float("nan")
            prev_rad = rad if np.isfinite(rad) else prev_rad

            bus.write_pos(sid, int(cmd[k]), speed=args.speed)

            if imu is not None:
                pg, av = imu.projected_gravity(), imu.angular_velocity()
            else:
                pg = av = (float("nan"),) * 3
            q_cmd = float(m.units_to_rad(np.full(m.n, float(cmd[k])))[idx])
            logger.log(t_wall=t0, step=k, loop_ms=dt * 1000.0,
                       busy_ms=(time.time() - t0) * 1000.0,
                       pg=pg, av=av, upright_cos=float("nan"), rej_total=fails,
                       u_meas=[u], u_cmd=[int(cmd[k])], q_cmd=[q_cmd], jvel_f=[vel])
    finally:
        written, dropped = logger.close()
        real_hz = n / max(time.time() - t_start, 1e-9)
        print(f"    -> {logger.path}")
        print(f"       {written} rows, {dropped} dropped, {fails} failed reads, "
              f"realized {real_hz:.0f} Hz"
              + ("  (slower than requested: bus-bound)" if real_hz < 0.9 * args.sample_hz else ""))
    return logger.path


def main():
    p = argparse.ArgumentParser(description="Single-joint excitation for system ID")
    p.add_argument("--joint", required=True, help="dof name, e.g. R_knee (see joint_servo_map.yaml)")
    p.add_argument("--mode", default="all", choices=["step", "chirp", "backlash", "all"])
    p.add_argument("--amp-deg", type=float, default=8.0, help="excitation amplitude, degrees")
    p.add_argument("--sample-hz", type=float, default=200.0)
    p.add_argument("--speed", type=int, default=0, help="servo move speed (0 = max)")
    p.add_argument("--hold", type=float, default=0.6, help="step: seconds per level")
    p.add_argument("--f0", type=float, default=0.2, help="chirp: start frequency, Hz")
    p.add_argument("--f1", type=float, default=5.0, help="chirp: end frequency, Hz")
    p.add_argument("--secs", type=float, default=30.0, help="chirp: sweep duration")
    p.add_argument("--period", type=float, default=8.0, help="backlash: triangle period, s")
    p.add_argument("--cycles", type=float, default=3.0, help="backlash: number of cycles")
    p.add_argument("--centre", default="current", choices=["current", "straight"],
                   help="excite about the current pose (default, no jump at start) or sim-straight. "
                        "Either way the centre is shifted inward if needed so the full swing "
                        "fits inside the joint limit (the knees and elbows are one-sided).")
    p.add_argument("--centre-units", type=float, default=None,
                   help="explicit excitation centre in servo units, overriding --centre")
    p.add_argument("--relax-others", action="store_true",
                   help="leave the other joints limp instead of holding them")
    p.add_argument("--imu", action="store_true",
                   help="also log pelvis attitude/rates (useful loaded, costs bus time)")
    p.add_argument("--imu-calib", default=str(ROOT / "config" / "imu_calib.yaml"))
    p.add_argument("--allow-clip", action="store_true", help="run even if >2%% of samples clip")
    p.add_argument("--log-dir", default=None)
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--dry-run", action="store_true", help="print the waveform, touch no hardware")
    args = p.parse_args()

    if args.amp_deg <= 0 or args.amp_deg > MAX_AMP_DEG:
        raise SystemExit(f"--amp-deg must be in (0, {MAX_AMP_DEG}]; got {args.amp_deg}")

    m = SimRealMap(args.map)
    dofs = [j.dof for j in m.joints]
    if args.joint not in dofs:
        raise SystemExit(f"unknown joint '{args.joint}'. choices: {dofs}")
    idx = dofs.index(args.joint)
    modes = ["step", "chirp", "backlash"] if args.mode == "all" else [args.mode]

    amp_units = args.amp_deg * np.pi / 180.0 * m.units_per_rad

    if args.dry_run:
        # No hardware, so "current" is unknowable -- preview about the mapped centre.
        req = args.centre_units if args.centre_units is not None else float(m.centers[idx])
        centre, shift = fit_centre(req, float(m.lim_lo[idx]), float(m.lim_hi[idx]), amp_units)
        print(f"[dry-run] {args.joint}: servo {int(m.servo_ids[idx])}, "
              f"limit {m.lim_lo[idx]:.0f}..{m.lim_hi[idx]:.0f}, centre {centre:.0f}"
              + (f"  (shifted {shift:+.0f} to fit +-{amp_units:.0f} units)" if abs(shift) > 0.5 else ""))
        for mode in modes:
            run_mode(mode, args, m, idx, None, None, centre, args.log_dir)
        print("\n[dry-run] OK. No hardware touched.")
        return

    from hardware import ServoBus, IMU  # noqa: PLC0415
    bus = ServoBus().connect()
    imu = None
    if args.imu:
        import os
        import yaml  # noqa: PLC0415
        remap = None
        if os.path.exists(args.imu_calib):
            with open(args.imu_calib) as f:
                remap = np.asarray(yaml.safe_load(f)["axis_remap"], dtype=np.float32)
        imu = IMU(axis_remap=remap).connect()
        imu.calibrate_gyro_bias(seconds=1.5)

    start_units = None
    try:
        bus.set_torque(m.servo_ids, True)
        start_units = bus.read_all(m.servo_ids)
        start_units = np.where(np.isfinite(start_units), start_units, m.centers)
        if args.relax_others:
            others = [s for i, s in enumerate(m.servo_ids) if i != idx]
            bus.set_torque(others, False)
            print("Other joints RELAXED (limp).")
        else:
            hold = np.clip(start_units, m.lim_lo, m.lim_hi).round().astype(int)
            bus.write_all(m.servo_ids, hold, speed=args.speed)
            print("Other joints HELD at their current position.")

        if args.centre_units is not None:
            req = float(args.centre_units)
        elif args.centre == "straight":
            req = float(m.centers[idx])
        else:
            req = float(start_units[idx])
        centre, shift = fit_centre(req, float(m.lim_lo[idx]), float(m.lim_hi[idx]), amp_units)
        if abs(shift) > 0.5:
            print(f"Excitation centre shifted {req:.0f} -> {centre:.0f} ({shift:+.0f} units) "
                  f"so +-{amp_units:.0f} units fits inside the joint limit.")
        # Ease to the excitation centre so the first sample is not a step of unknown size.
        cur = float(start_units[idx])
        for k in range(1, 41):
            bus.write_pos(int(m.servo_ids[idx]), int(round(cur + (centre - cur) * k / 40)),
                          speed=args.speed)
            time.sleep(0.025)

        for mode in modes:
            run_mode(mode, args, m, idx, bus, imu, centre, args.log_dir)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Ramping back to the starting pose and disabling torque.")
        try:
            if start_units is not None:
                bus.set_torque(m.servo_ids, True)
                cur = bus.read_all(m.servo_ids)
                cur = np.where(np.isfinite(cur), cur, start_units)
                for k in range(1, 41):
                    u = np.clip(cur + (start_units - cur) * k / 40, m.lim_lo, m.lim_hi)
                    bus.write_all(m.servo_ids, u.round().astype(int), speed=args.speed)
                    time.sleep(0.025)
        except Exception:
            pass
        bus.set_torque(m.servo_ids, False)
        bus.close()
        if imu is not None:
            imu.close()


if __name__ == "__main__":
    main()
