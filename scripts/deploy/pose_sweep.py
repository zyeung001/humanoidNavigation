#!/usr/bin/env python3
# pose_sweep.py  --  RUN ON THE PI
"""Find the pose this robot actually balances in, by measuring instead of asking sim.

WHY THIS EXISTS. Every standing policy so far was trained around a baseline pose that was
GUESSED three times: v1 used sim's settled stance, v2 zeroed it to straight, v3 widened
the box around straight. All three failed on hardware. The policy has since re-derived
v1's stance independently across ~30M steps -- it wants waist_yaw -0.52 and R_hip_pitch
+0.32 and is clipped short of both -- while the one time that stance WAS allowed, the real
robot hung crooked at 13-16 degrees. So sim is confident about a pose hardware rejects,
and nobody has ever measured which pose hardware would accept.

The feet can answer it. Centre of pressure is where the weight actually lands, and it does
not care what any model believes. This walks a trim through a range, holds each pose, and
reads where the pressure sits -- turning "where should it stand" from an argument into a
line with a zero crossing.

WHAT IT DOES NOT DO. It does not balance the robot. But DO NOT HOLD IT either: a hand
that takes any weight takes it away from the sensors, and the sweep then measures your arm.
Set it down, let go, and hover a hand to catch it -- the tilt cut ends the sweep before it
goes over. A run taken under support is flagged by total load, not left to be discovered
in the slope. No policy runs here, so nothing can seize.

READING THE RESULT. The SLOPE, in mm of CoP per degree of trim, is trustworthy. The
absolute zero is only as good as the FSR calibration: if the foot plate shares load with
the sensors, every reading is biased toward centre by the same amount. Run the flat-bar
offset test and pass --cop-offset, or treat the crossing as an estimate whose sign and
slope are solid and whose exact value is not.

  python3 scripts/deploy/pose_sweep.py --joints waist_pitch --range -6 6 --step 2
  python3 scripts/deploy/pose_sweep.py --joints R_hip_pitch,L_hip_pitch --range -6 6 --step 3
  python3 scripts/deploy/pose_sweep.py --dry-run

RECORDING. The live phase writes the same frame log as deploy_standing (frame_log.py, schema
2): logs/<time>_sweep.csv (or _probe.csv) + .meta.json + .console.txt, a row per sensor read
-- ramps, settles, dwells, the probe, and the ramp back. The printed summary is an average;
the log is what shows HOW a pose failed (which joint gave, which foot unloaded first, when).
--no-log turns it off. q_cmd here is the commanded units converted back to radians (no policy
runs), and jvel_f / q_prior / posture_* are NaN.

EVENTS. As in deploy_standing, single keys on the SSH terminal (events.py: x fell, c caught,
m mark, n note, f/b/l/r push, k kick) are stamped into .events.jsonl against the logged steps,
and after the result it asks "what did you see?". In a probe, press x the moment it goes over
and c when you catch it: the summary averages only upright samples, and the markers say which
stretch of the frames that was. --no-events / --no-prompt turn them off.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsr_monitor import fsr_resistance  # noqa: E402
from sim_real_map import SimRealMap, DEFAULT_MAP  # noqa: E402

MAX_TRIM_DEG = 12.0        # beyond this it is a contortion, not a trim
SIM_COP_MM = 4.3           # scripts/debug/sim_cop.py, straight pose
COM_HEIGHT_MM = 414.7      # for converting a CoP shift into an equivalent body lean


def cop_mm(volts, rfixed, vcc, x_heel, x_toe):
    """Load-weighted mean of the two sensor positions. None when a sensor is unloaded."""
    c, tot, _ = cop_detail(volts, rfixed, vcc, x_heel, x_toe)
    return c, tot


def cop_detail(volts, rfixed, vcc, x_heel, x_toe):
    """As cop_mm, plus WHICH end is unloaded -- "HEEL", "TOE" or "" when both carry.

    Worth reporting on its own because an unusable CoP is not a missing measurement, it is
    a measurement of which end the robot is standing on, and that is the one fact that
    tells you which way it is actually leaning. The pelvis IMU cannot: the waist stack
    deflects under load with no servo leaving position, so a robot pitched forward onto its
    toe reports a BACKWARD lean. On 9/4 that ambiguity was the whole question -- sim and
    hardware disagreed on the SIGN of the waist_pitch trim, and the two readings differ
    only in which sensor was the dead one.
    """
    g = []
    for v in volts:
        r = fsr_resistance(v, rfixed, vcc, False)
        g.append(0.0 if r == float("inf") else 1.0 / max(r, 1e-6))
    tot = g[0] + g[1]
    if tot <= 1e-9:
        return None, tot, "BOTH"
    if min(g) / tot < 0.05:
        # The dead sensor is the end that is off the ground; the robot is on the other one.
        return None, tot, ("HEEL" if g[0] < g[1] else "TOE")
    return (g[0] * x_heel + g[1] * x_toe) / tot, tot, ""


def load_baseline(path, m):
    """A whole-body pose in sim radians, from a config's standing.residual_baseline.

    Needed because the pose this robot actually balances in is not a trim off straight --
    it is a multi-joint offset (L_hip_yaw +7.9 deg, waist_pitch +7.1, L_hip_roll +6.2),
    which no single --trim can express. Reading it from the same file the training run uses
    keeps the pose held on the bench and the pose trained against as one thing that cannot
    drift apart.
    """
    import yaml
    b = yaml.safe_load(open(path))["standing"]["residual_baseline"]
    b = np.asarray(b, dtype=float)
    if b.shape != (m.n,):
        raise SystemExit(f"{path}: residual_baseline has {b.shape[0]} values, expected {m.n}")
    return b


def build_plan(m, dofs, names, trims, baseline=None):
    # Straight (the measured per-joint centres) unless a baseline pose is supplied, in
    # which case trims are applied on top of THAT.
    # Convert by hand rather than through rad_to_units, which clips to the servo limits
    # internally and silently. Going through it would hand this function a pose that had
    # already been trimmed, and the clip report below would then have nothing to find --
    # the tool would be quietly hiding exactly what it is here to expose.
    base = (m.centers.astype(float) if baseline is None
            else m.centers.astype(float)
            + m.signs.astype(float) * np.asarray(baseline, dtype=float) * m.units_per_rad)
    idxs = [dofs.index(n) for n in names]
    out, clipped = [], {}
    for tr in trims:
        u = base.copy()
        for i in idxs:
            u[i] += m.signs[i] * np.deg2rad(tr) * m.units_per_rad
        c = np.clip(u, m.lim_lo, m.lim_hi)
        # Report what the clip changed. Silently commanding a different pose than the one
        # asked for is the exact fault this whole line of work exists to find -- it is how
        # three arm joints spent months unable to reach straight -- and a tool that does it
        # quietly is no better than the firmware that did.
        for i in np.flatnonzero(np.abs(c - u) > 0.5):
            clipped.setdefault(dofs[i], (float(u[i]), float(c[i])))
        out.append((float(tr), c.round().astype(int)))
    return out, idxs, clipped


def main():
    p = argparse.ArgumentParser(description="Measure the pose the robot balances in")
    p.add_argument("--joints", default="waist_pitch",
                   help="comma-separated joints trimmed together (trim both hips as a pair, "
                        "or the robot twists instead of leaning)")
    p.add_argument("--range", type=float, nargs=2, default=[-6.0, 6.0], metavar=("LO", "HI"))
    p.add_argument("--step", type=float, default=2.0, help="degrees per point")
    p.add_argument("--probe", type=float, default=None, metavar="TRIM",
                   help="hold ONE trim and stream lean + CoP live until Ctrl-C, instead of "
                        "sweeping. This is the mode to use on a robot that will not stand "
                        "unaided: steady it, release, and watch which way it goes. Falls "
                        "backward -> probe a more forward trim; falls forward -> go back. "
                        "Five or six of these bisect the balance point to about a degree, "
                        "and none of them needs the robot to stay up.")
    p.add_argument("--dwell", type=float, default=4.0, help="seconds to settle and sample")
    p.add_argument("--ramp", type=float, default=2.0, help="seconds to move between points")
    p.add_argument("--speed", type=int, default=300)
    p.add_argument("--rfixed", type=float, default=2000.0)
    p.add_argument("--vcc", type=float, default=3.3)
    p.add_argument("--pos-heel", type=float, default=-40.0, help="mm from the foot centre")
    p.add_argument("--pos-toe", type=float, default=36.0)
    p.add_argument("--free-load", type=float, default=0.0030, metavar="SIEMENS",
                   help="total conductance this robot reads standing FREE on the "
                        "instrumented foot. A sweep coming in under it was taken with "
                        "something carrying the weight, and its slope is not usable. "
                        "Default 3.0 mS, from the 9/4 released probes (3.7-4.4 mS free, "
                        "2.1-2.7 mS held).")
    p.add_argument("--cop-offset", type=float, default=0.0,
                   help="mm to subtract, from the flat-bar zero test")
    p.add_argument("--fsr-channels", default="0,1")
    p.add_argument("--tilt-cut", type=float, default=0.80,
                   help="abort and ramp back if the pelvis tilts past this (cosine of upright; "
                        "0.80 is about 37 deg). This tool holds a pose like a statue and does "
                        "NOT balance, so without a cut it would keep driving servos while the "
                        "robot topples.")
    p.add_argument("--upright-deg", type=float, default=5.0,
                   help="a probe sample counts as UPRIGHT while |lean| stays under this. Only "
                        "those samples are averaged, because a reading taken while the robot "
                        "is toppling measures the topple and not the pose.")
    p.add_argument("--imu-calib", default=str(ROOT / "config" / "imu_calib.yaml"))
    p.add_argument("--no-imu", action="store_true",
                   help="skip the IMU entirely: no tilt cut and no lean cross-check")
    p.add_argument("--baseline", default=None, metavar="CONFIG",
                   help="hold the whole-body pose in a config's standing.residual_baseline "
                        "instead of straight, with any --trim applied on top. The pose this "
                        "robot balances in is a multi-joint offset, not a trim off straight, "
                        "so no --trim can express it. Point this at the same config the "
                        "training run uses.")
    p.add_argument("--fsr-layout", default=None, metavar="YAML",
                   help="FSR channel list with each sensor's foot and position (frame_log."
                        "load_fsr_layout). The CoP printed here uses the FIRST TWO channels as "
                        "heel, toe; every channel is logged.")
    p.add_argument("--log", default="", metavar="PATH",
                   help="frame log path (default: auto-named into logs/)")
    p.add_argument("--no-log", action="store_true", help="do not record the live phase")
    p.add_argument("--no-events", action="store_true",
                   help="do not read operator keys (x fell, c caught, m mark, n note, ...; "
                        "see events.py) into <log>.events.jsonl")
    p.add_argument("--no-prompt", action="store_true",
                   help="do not ask for the after-run 'what did you see?' note")
    p.add_argument("--no-servo-telemetry", action="store_true",
                   help="read servo position only, not speed/load/voltage/temperature "
                        "(same as deploy_standing)")
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Console recording starts before anything prints, so the plan, the clip report and the
    # result all land in <stem>.console.txt beside the frames they summarise.
    log_path = None
    if not args.no_log and not args.dry_run:
        from frame_log import ConsoleTee, default_log_path, sidecar_path  # noqa: PLC0415
        log_path = (Path(args.log) if args.log
                    else default_log_path("probe" if args.probe is not None else "sweep"))
        ConsoleTee(sidecar_path(log_path, "console.txt"))   # held by atexit until exit
        print(f"Recording this run -> {log_path}  (+ .meta.json, .console.txt; --no-log to skip)")

    m = SimRealMap(args.map)
    dofs = [j.dof for j in m.joints]
    names = [s.strip() for s in args.joints.split(",") if s.strip()]
    for n in names:
        if n not in dofs:
            raise SystemExit(f"unknown joint '{n}'. choices: {dofs}")
    lo, hi = sorted(args.range)
    if max(abs(lo), abs(hi)) > MAX_TRIM_DEG:
        raise SystemExit(f"trim beyond +-{MAX_TRIM_DEG} deg is a contortion, not a trim")
    if args.probe is not None:
        trims = np.array([float(args.probe)])
    else:
        trims = np.arange(lo, hi + 1e-9, abs(args.step))
        # Visit from the CENTRE outward, alternating sides, rather than marching from one
        # end to the other. A robot that cannot stand unaided topples at the extremes, and
        # marching from an end spends the first points there -- losing the middle of the
        # curve, which is the part that carries the crossing.
        mid = 0.5 * (lo + hi)
        trims = trims[np.argsort(np.abs(trims - mid), kind="stable")]
    baseline = load_baseline(args.baseline, m) if args.baseline else None
    plans, idxs, clipped = build_plan(m, dofs, names, trims, baseline)

    print(f"sweeping {', '.join(names)} over {lo:+.0f}..{hi:+.0f} deg in {args.step:g} deg steps")
    print(f"  {len(plans)} poses x ({args.ramp:g}s ramp + {args.dwell:g}s dwell) = "
          f"{len(plans)*(args.ramp+args.dwell)/60:.1f} min")
    print(f"  CoP from sensors at {args.pos_heel:+.0f}/{args.pos_toe:+.0f} mm, "
          f"R_fixed {args.rfixed:.0f} ohm, offset {args.cop_offset:+.1f} mm")
    if baseline is not None:
        print(f"  base pose from {args.baseline}: max offset from straight "
              f"{np.degrees(np.abs(baseline).max()):.1f} deg")
    if clipped:
        print(f"  !! {len(clipped)} joint(s) CLIPPED by their map servo_limit -- the pose")
        print("     actually commanded is NOT the pose asked for:")
        for d, (want, got) in clipped.items():
            print(f"       {d:14s} wanted {want:6.1f}, limited to {got:6.1f} "
                  f"({got - want:+.1f} units = {(got - want) / 195 * 57.3:+.2f} deg)")
    if args.dry_run:
        for tr, u in plans:
            print(f"  trim {tr:+5.1f} deg -> " + "  ".join(f"{dofs[i]}={u[i]}" for i in idxs))
        print("\n[dry-run] no hardware touched.")
        return 0

    if args.probe is not None:
        print(f"\nPROBE at trim {args.probe:+.1f} deg. Steady it, then RELEASE and watch.")
        print("  falls BACKWARD -> the balance point is at a more forward trim")
        print("  falls FORWARD  -> go back the other way")
        print("Ctrl-C ramps back and drops torque.\n")
    else:
        print("\nSET IT DOWN AND LET GO. Hover a hand to catch it, but do not let it take any")
        print("weight: a supporting hand carries load that would otherwise reach the sensors,")
        print("and every CoP sample in the run is then measuring your arm. On 9/4 a held sweep")
        print("read 2.1-2.7 mS with 11-16 mm of spread where the same robot released reads")
        print("3.7-4.4 mS with 1.5-3.3 mm; the trim ends came out indistinguishable. The tilt")
        print("cut stops the sweep before it goes over, which is what the catch is for.")
        print("Ctrl-C ramps back and drops torque.\n")

    from frame_log import FsrSampler, resolve_fsr  # noqa: PLC0415
    from hardware import ServoBus, release_on_hangup  # noqa: PLC0415

    fsr_chans, fsr_layout = resolve_fsr(args.fsr_channels, args.fsr_layout)
    if len(fsr_chans) < 2:
        raise SystemExit("the CoP here needs two FSR channels (heel, toe) first in the list")
    release_on_hangup()
    # Telemetry only when something records it: --no-log keeps the exact old 2-byte read.
    bus = ServoBus(telemetry=log_path is not None and not args.no_servo_telemetry).connect()
    if bus.telemetry:
        bus.check_telemetry(m.servo_ids)      # torque is still off; falls back loudly if needed
    fsr = FsrSampler(channels=fsr_chans)
    # The IMU earns its place twice here: it aborts before a topple, and it measures the
    # LEAN at each trim. Lean and centre of pressure are independent instruments aimed at
    # the same question, so if they disagree about which way the pose should move, that
    # disagreement is worth more than either number on its own.
    imu = None
    if not args.no_imu:
        import os

        import yaml  # noqa: PLC0415

        from hardware import IMU  # noqa: PLC0415
        remap = None
        if os.path.exists(args.imu_calib):
            remap = np.asarray(yaml.safe_load(open(args.imu_calib))["axis_remap"],
                               dtype=np.float32)
        imu = IMU(axis_remap=remap).connect()

    logger = None
    if log_path is not None:
        from frame_log import FrameLogger, files_meta  # noqa: PLC0415
        from hardware import TELEMETRY_DECODING  # noqa: PLC0415
        logger = FrameLogger(log_path, dofs, n_fsr=len(fsr_chans), meta={
            "script": "pose_sweep.py",
            "mode": "probe" if args.probe is not None else "sweep",
            "args": vars(args),
            "joints": names, "trims_deg": [tr for tr, _ in plans],
            "plan_units": {f"{tr:+g}": [int(v) for v in u] for tr, u in plans},
            "clipped": clipped,
            "baseline": None if baseline is None else [float(v) for v in baseline],
            "servo_ids": [int(s) for s in m.servo_ids],
            "units_per_rad": float(m.units_per_rad),
            "axis_remap": None if imu is None else imu.axis_remap.tolist(),
            "gyro_bias": None if imu is None else [float(v) for v in imu.gyro_bias],
            "gyro_note": "pose_sweep does not calibrate the gyro: av_* carry the raw offset",
            "q_cmd_note": "q_cmd = commanded units converted to sim rad (no policy runs)",
            **files_meta({"map": args.map, "baseline": args.baseline,
                          "imu_calib": None if args.no_imu else args.imu_calib,
                          "fsr_layout": args.fsr_layout,
                          # not in frame_log.DEPLOY_SOURCES (it predates events.py)
                          "source:events.py": Path(__file__).resolve().parent / "events.py"}),
            "servo_telemetry": {"enabled": bus.telemetry,
                                "requested": not args.no_servo_telemetry,
                                "telemetry_fallback": bus.telemetry_fallback is not None,
                                "fallback": bus.telemetry_fallback,
                                "decoding": TELEMETRY_DECODING},
            "fsr": fsr_layout,
        })
        print(f"Logging frames -> {log_path}")

    # Operator event markers: a sidecar of the frame log, stepped on its writer thread (see
    # events.py). Started inside the try below, so the `finally` always restores the terminal.
    events = None
    if logger is not None and not args.no_events:
        from events import EventRecorder  # noqa: PLC0415
        from frame_log import sidecar_path  # noqa: PLC0415
        events = EventRecorder(sidecar_path(log_path, "events.jsonl"), t0=logger.t0)
        logger.add_sink(events.on_row)
    elif log_path is None and not args.no_events:
        print("[events] off: --no-log (the markers are a sidecar of the frame log)")

    nan_j = np.full(m.n, np.nan, dtype=np.float32)
    nan3 = (float("nan"),) * 3
    rec = {"step": 0, "rej": 0, "t_prev": None, "sensor_errors": 0}

    def record(u_cmd, pg=None):
        """Log one frame: servos, IMU, feet. A no-op with --no-log, so an unrecorded sweep
        runs exactly as it always did. Reuses `pg` when the caller already read the IMU.

        It CANNOT RAISE. Every read in here exists only for the log, and the IMU deliberately
        does not retry mid-run (hardware.IMU.connect) on an I2C bus that has failed before
        (10/18). One OSError escaping from here used to end the sweep AND cut the ramp back
        short at its first step, dropping torque at the trim pose. A failed read is NaN in the
        row and counted in the meta's sensor_errors; the sweep goes on as if logging were off.
        """
        if logger is None:
            return
        try:
            t0 = time.time()
            loop_ms = float("nan") if rec["t_prev"] is None else (t0 - rec["t_prev"]) * 1000.0
            rec["t_prev"] = t0
            u_meas, tel = nan_j, None
            try:
                u_meas = bus.read_all(m.servo_ids)
                tel = bus.last_telemetry
            except Exception:
                rec["sensor_errors"] += 1
            if not np.all(np.isfinite(u_meas)):
                rec["rej"] += 1
            pg_row = av = acc = nan3
            if imu is not None:
                try:
                    pg_row = imu.projected_gravity() if pg is None else pg
                    acc = imu.last_accel_g
                    av = imu.angular_velocity()
                except Exception:
                    rec["sensor_errors"] += 1
            v, age = fsr.read()
            u_cmd = np.asarray(u_cmd, dtype=float)
            logger.log(t_wall=t0, step=rec["step"], loop_ms=loop_ms,
                       busy_ms=(time.time() - t0) * 1000.0, pg=pg_row, av=av,
                       upright_cos=-float(pg_row[2]), rej_total=rec["rej"],
                       u_meas=u_meas, u_cmd=u_cmd, q_cmd=m.units_to_rad(u_cmd), jvel_f=nan_j,
                       fsr=v, fsr_age_ms=age, acc=acc, telemetry=tel)
            rec["step"] += 1
        except Exception:
            rec["sensor_errors"] += 1

    def hold_for(seconds, u_cmd):
        """Sleep through a settle, recording at ~40 Hz instead of leaving a gap in the log."""
        t_end = time.time() + seconds
        while time.time() < t_end:
            t0 = time.time()
            record(u_cmd)
            time.sleep(max(0.0, min(1 / 40 - (time.time() - t0), t_end - time.time())))

    start = None
    rows = []
    probe_hist = []
    try:
        if events is not None:
            events.start()
            logger.update_meta(events=events.summary())   # a run that dies still says so
        bus.set_torque(m.servo_ids, True)
        start = bus.read_all(m.servo_ids)
        start = np.where(np.isfinite(start), start, m.centers.astype(float))
        cur = start.astype(float).copy()
        for tr, target in plans:
            steps = max(1, int(args.ramp * 40))
            for k in range(1, steps + 1):
                t0 = time.time()
                u_now = (cur + (target - cur) * k / steps).round().astype(int)
                bus.write_all(m.servo_ids, u_now, speed=args.speed)
                record(u_now)
                time.sleep(max(0.0, 1 / 40 - (time.time() - t0)))
            cur = target.astype(float)
            if args.probe is not None:
                # Hold and stream until Ctrl-C. No tilt cut here on purpose: the whole point
                # is to watch it go over and see WHICH WAY, so cutting torque mid-topple
                # would throw away the measurement being taken.
                print("  holding. release it now.\n")
                upright_since = None
                while True:
                    v, _ = fsr.read()
                    c, tot, dead = cop_detail(v, args.rfixed, args.vcc,
                                              args.pos_heel, args.pos_toe)
                    ln = roll = float("nan")
                    if imu is not None:
                        pg = imu.projected_gravity()
                        ln = float(np.degrees(np.arctan2(pg[0], -pg[2])))
                        # Lateral lean as well. Reporting only fore/aft left the tool blind
                        # to an entire axis: the 9/3 calibration moved L_hip_roll by 6.2 deg
                        # and L_hip_yaw by 7.9 deg, which shifts weight sideways, and a
                        # robot toppling sideways read as one standing perfectly straight.
                        roll = float(np.degrees(np.arctan2(pg[1], -pg[2])))
                    record(target, pg=None if imu is None else pg)
                    now = time.time()
                    probe_hist.append(
                        (now, ln, float("nan") if c is None else c - args.cop_offset, roll))
                    up = bool(np.isfinite(ln) and abs(ln) <= args.upright_deg)
                    upright_since = (upright_since or now) if up else None
                    held = 0.0 if upright_since is None else now - upright_since
                    where = ("FORWARD" if ln > args.upright_deg else
                             "BACKWARD" if ln < -args.upright_deg else "UPRIGHT")
                    ctxt = "  --  " if c is None else f"{c - args.cop_offset:+6.1f}"
                    # roll + = the robot's LEFT (verified on hardware 9/26 against the gyro).
                    # This used to print RIGHT for +, i.e. the wrong side.
                    side = ("LEFT " if roll > args.upright_deg else
                            "RIGHT" if roll < -args.upright_deg else "     ")
                    off = f"{dead} UP " if dead else "       "
                    print(f"\r  pitch {ln:+6.1f} {where:8s} roll {roll:+6.1f} {side} "
                          f"CoP {ctxt} mm {off} load {1000*tot:5.2f} mS  upright {held:4.1f}s  ",
                          end="", flush=True)
                    time.sleep(0.1)
            hold_for(args.dwell * 0.4, target)    # settle before sampling, not while moving
            samples, loads, deads = [], [], []
            t0 = time.time()
            while time.time() - t0 < args.dwell * 0.6:
                v, _age = fsr.read()
                c, tot, dead = cop_detail(v, args.rfixed, args.vcc,
                                          args.pos_heel, args.pos_toe)
                if c is not None:
                    samples.append(c - args.cop_offset)
                    loads.append(tot)
                else:
                    deads.append(dead)
                record(target)
                time.sleep(0.05)
            lean = float("nan")
            if imu is not None:
                pg = imu.projected_gravity()
                lean = float(np.degrees(np.arctan2(pg[0], -pg[2])))    # + = leaning forward
                if -float(pg[2]) < args.tilt_cut:
                    print(f"  trim {tr:+5.1f} deg -> TILT CUT at upright {-float(pg[2]):.2f}: "
                          f"it is going over. Aborting the sweep and ramping back.")
                    break
            if samples:
                med, sd = float(np.median(samples)), float(np.std(samples))
                rows.append((tr, med, sd, lean, float(np.median(loads))))
                print(f"  trim {tr:+5.1f} deg -> CoP {med:+7.1f} mm   spread {sd:4.1f}   "
                      f"load {1000*np.median(loads):5.2f} mS   lean {lean:+5.1f} deg   "
                      f"n={len(samples)}")
            else:
                rows.append((tr, float("nan"), float("nan"), lean, float("nan")))
                which = max(set(deads), key=deads.count) if deads else "?"
                print(f"  trim {tr:+5.1f} deg -> the {which} is UNLOADED "
                      f"(lean {lean:+5.1f} deg): the robot is standing on its "
                      f"{'TOE' if which == 'HEEL' else 'HEEL' if which == 'TOE' else '?'}, "
                      f"or is not on the instrumented foot")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        print("ramping back to the starting pose and releasing torque.")
        try:
            if start is not None:
                cur = bus.read_all(m.servo_ids)
                cur = np.where(np.isfinite(cur), cur, start)
                for k in range(1, 81):
                    t0 = time.time()
                    u = np.clip(cur + (start - cur) * k / 80, m.lim_lo, m.lim_hi)
                    bus.write_all(m.servo_ids, u.round().astype(int), speed=args.speed)
                    record(u.round())                 # cannot raise: see record()
                    time.sleep(max(0.0, 1 / 40 - (time.time() - t0)))
        except Exception:
            pass
        # Torque comes off BEFORE the logging cleanup: closing the FSR sampler and the writer
        # can take seconds (thread joins, a slow SD card), and the robot must not stay stiff.
        torque_off = False
        try:
            bus.set_torque(m.servo_ids, False)
            torque_off = True
        except Exception as e:
            print(f"[SAFETY] torque-off FAILED ({e!r}); retrying after the log is closed")
        try:
            if events is not None:
                events.close()       # terminal back to normal; never raises
            fsr.close()
            if logger is not None:
                # The per-trim summary rides along in the meta, so the numbers printed below
                # and the frames they came from stay one record.
                written, dropped = logger.close(
                    fsr_errors=fsr.errors, sensor_errors=rec["sensor_errors"],
                    sweep_points=[{"trim": tr, "cop_mm": c, "spread_mm": sd, "lean_deg": ln,
                                   "load_S": ld} for tr, c, sd, ln, ld in rows],
                    probe_samples=len(probe_hist))
                print(f"Frame log: {written} rows -> {logger.path}"
                      + (f"   WARNING: {dropped} rows DROPPED" if dropped else "")
                      + (f"   WARNING: {rec['sensor_errors']} failed sensor reads (NaN in the log)"
                         if rec["sensor_errors"] else ""))
        finally:
            try:
                if not torque_off:
                    bus.set_torque(m.servo_ids, False)   # a second failure raises, loudly
            finally:
                bus.close()
                if imu is not None:
                    imu.close()
                if events is not None:
                    events.save()    # the markers are on disk even if the report fails

    if args.probe is not None:
        probe_report(probe_hist, args)
    else:
        report(rows, names, args)
    if events is not None:
        # After the result is on screen, so "what did you see?" can answer it.
        summary = events.finish(prompt=not args.no_prompt)
        try:
            logger.update_meta(events=summary)
        except OSError:
            pass
    return 0


def probe_report(hist, args):
    """Summarise only the stretch where the robot was actually upright.

    The first version printed a live line and left whatever happened to be on screen when
    you hit Ctrl-C, which was almost always mid-topple: two probes at the same trim came
    back 30 mm apart for exactly that reason, with lean already past +10 deg in both. A
    centre-of-pressure reading is a statement about a pose, and it is only a statement
    about THAT pose while the robot is still in it.
    """
    print()
    print(f"--- PROBE at trim {args.probe:+.1f} deg ---")
    if not hist:
        print("  no samples captured.")
        return
    t0 = hist[0][0]
    up = [(t - t0, ln, c, rl) for t, ln, c, rl in hist
          if np.isfinite(ln) and abs(ln) <= args.upright_deg]
    print(f"  {len(hist)} samples over {hist[-1][0] - t0:.1f}s; {len(up)} upright "
          f"(|lean| <= {args.upright_deg:.0f} deg)")
    if not up:
        closest = min(hist, key=lambda r: abs(r[1]) if np.isfinite(r[1]) else 1e9)
        print(f"  NEVER upright -- closest approach was {closest[1]:+.1f} deg.")
        print("  This trim did not hold, so it has no centre of pressure to report.")
        return
    # The longest UNBROKEN upright stretch, not every upright sample: two seconds of
    # holding says something about the pose, whereas scattered frames on the way past
    # vertical are just the trajectory of a fall.
    runs, cur = [], [up[0]]
    for prev, nxt in zip(up[:-1], up[1:]):
        if nxt[0] - prev[0] < 0.35:
            cur.append(nxt)
        else:
            runs.append(cur)
            cur = [nxt]
    runs.append(cur)
    best = max(runs, key=len)
    held = best[-1][0] - best[0][0]
    lean = np.array([r[1] for r in best])
    cop = np.array([r[2] for r in best if np.isfinite(r[2])])
    roll = np.array([r[3] for r in best if np.isfinite(r[3])])
    print(f"  longest unbroken upright stretch: {held:.1f}s")
    print(f"  pitch over it {np.median(lean):+6.2f} deg   (spread {np.std(lean):.2f})")
    if len(roll):
        r_med = float(np.median(roll))
        print(f"  ROLL over it  {r_med:+6.2f} deg   (spread {np.std(roll):.2f})"
              + ("   <- leaning sideways; a lateral topple looks nothing like a pitch one"
                 if abs(r_med) > 3 else ""))
    if len(cop) < max(3, len(best) // 4):
        print(f"  CoP UNUSABLE: only {len(cop)} of {len(best)} upright samples had both")
        print("  sensors loaded. The instrumented foot is not carrying its share -- check")
        print("  the ROLL above, because weight shifted sideways is the usual reason.")
        return
    print(f"  CoP over it   {np.median(cop):+6.1f} mm    (spread {np.std(cop):.1f}, n={len(cop)})")
    print(f"  sim predicts  {SIM_COP_MM:+6.1f} mm at the straight pose")
    print()
    print(f"  log this as: trim {args.probe:+.1f} -> CoP {np.median(cop):+.1f} mm, "
          f"lean {np.median(lean):+.2f} deg, held {held:.1f}s")


def supported_warning(rows, free_load):
    """Flag a sweep taken while someone was holding the robot up.

    A hand on the robot carries load that would otherwise reach the sensors, so every CoP
    sample in the run measures the arm as much as the pose. It does not look like an error:
    the numbers arrive, the fit succeeds, and the slope is simply wrong. On 9/4 a held sweep
    read 2.1-2.7 mS against 3.7-4.4 mS released, with 11-16 mm of CoP spread against 1.5-3.3,
    and produced trim ends that were indistinguishable from each other. Total load is the
    one channel that can see it, because the missing weight has to go somewhere.
    """
    loads = [ld for *_, ld in rows if np.isfinite(ld)]
    if not loads:
        return
    med = float(np.median(loads))
    if med < free_load:
        print(f"\n  !! LOAD {1000 * med:.2f} mS, below the {1000 * free_load:.1f} mS this "
              "robot reads standing free.")
        print("     Something was carrying its weight -- a hand, a tether under tension, or "
              "the")
        print("     instrumented foot taking less than its share. The slope below is NOT "
              "usable.")
        print("     Set it down, let go entirely, and re-run; use --free-load to retune the "
              "threshold.")


def report(rows, names, args):
    good = [(t, c) for t, c, _, _, _ in rows if np.isfinite(c)]
    print("\n--- RESULT ---")
    supported_warning(rows, args.free_load)
    if len(good) < 3:
        print("  not enough loaded points to fit a line. Was the robot standing on the")
        print("  instrumented foot for the whole sweep?")
        return
    x = np.array([g[0] for g in good])
    y = np.array([g[1] for g in good])
    slope, icept = np.polyfit(x, y, 1)
    label = "+".join(names)
    print(f"  CoP moves {slope:+.2f} mm per degree of {label} trim")
    print(f"  measured CoP at trim 0: {icept:+.1f} mm     sim predicts {SIM_COP_MM:+.1f} mm")
    gap = icept - SIM_COP_MM
    print(f"  sim-vs-real gap {gap:+.1f} mm = "
          f"{np.degrees(np.arctan(gap / COM_HEIGHT_MM)):+.2f} deg of body lean")
    if abs(slope) > 1e-6:
        zero = -icept / slope
        inside = min(x) <= zero <= max(x)
        print(f"\n  pressure centres at a trim of {zero:+.1f} deg"
              + ("" if inside else "  -- EXTRAPOLATED beyond the sweep; widen --range to confirm"))
        print(f"  that is the baseline to train around: {label} shifted {zero:+.1f} deg")
    leans = [(t, ln) for t, _, _, ln, _ in rows if np.isfinite(ln)]
    if len(leans) >= 3 and abs(slope) > 1e-6:
        lx = np.array([v[0] for v in leans])
        ly = np.array([v[1] for v in leans])
        lslope, licept = np.polyfit(lx, ly, 1)
        print()
        print(f"  IMU cross-check: lean moves {lslope:+.2f} deg per degree of trim, "
              f"{licept:+.1f} deg at trim 0")
        if abs(lslope) > 1e-6:
            lzero = -licept / lslope
            agree = abs(lzero - (-icept / slope)) < 2.0
            print(f"  lean reaches vertical at trim {lzero:+.1f} deg"
                  + ("   AGREES with the pressure crossing" if agree else
                     "   DISAGREES with the pressure crossing -- trust neither until they do"))

    print("\n  Trust the slope. The absolute zero carries your FSR calibration error"
          + ("." if args.cop_offset else
             " --\n  run the flat-bar test and pass --cop-offset to remove it."))


if __name__ == "__main__":
    sys.exit(main())
