#!/usr/bin/env python3
# servo_step_response.py
"""
Measure a single SCS servo's step response (the real actuator bandwidth) so we can MODEL
it in sim instead of guessing. The sim2real diagnosis is that the standing policy relies on
fast, large corrective moves an ideal sim position-actuator executes instantly, but the real
servo lags -> the robot diverges. This quantifies that lag.

Commands ONE joint a step (default ~0.25 rad) at full speed, polls read_pos as fast as the
bus allows, and reports the time-to-63% (first-order time constant tau), time-to-90%, and
overshoot. Only the tested servo is torqued; the rest stay limp.

Run ON THE PI (needs the serial bus). Position the robot so the tested joint can move freely.

  python scripts/deploy/servo_step_response.py                  # R_knee (servo 7), 50 units
  python scripts/deploy/servo_step_response.py --joint R_hip_pitch --delta-units 60
  python scripts/deploy/servo_step_response.py --servo 7 --delta-units -50 --secs 1.5
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware import ServoBus              # noqa: E402
from sim_real_map import SimRealMap        # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint", default="R_knee", help="dof name in the map (e.g. R_knee)")
    p.add_argument("--servo", type=int, default=None, help="servo id (overrides --joint)")
    p.add_argument("--delta-units", type=int, default=-50,
                   help="step size in servo units from the current position (signed)")
    p.add_argument("--secs", type=float, default=1.5, help="logging duration after the step")
    p.add_argument("--settle-secs", type=float, default=0.5, help="hold-still log BEFORE the step")
    args = p.parse_args()

    m = SimRealMap()
    if args.servo is not None:
        sid = args.servo
        j = next((jj for jj in m.joints if jj.servo_id == sid), None)
        name = j.dof if j else f"servo{sid}"
        lim = j.servo_limit if j else (0, 1023)
    else:
        j = next(jj for jj in m.joints if jj.dof == args.joint)
        sid, name, lim = j.servo_id, j.dof, j.servo_limit

    bus = ServoBus().connect()
    bus.set_torque([sid], True)
    time.sleep(0.1)

    start = bus.read_pos(sid)
    if start is None:
        print(f"ERROR: could not read servo {sid}")
        bus.set_torque([sid], False)
        return
    target = int(np.clip(start + args.delta_units, lim[0], lim[1]))
    print(f"joint={name} servo={sid} limit={lim}")
    print(f"start={start} units  target={target} units  step={target - start} units "
          f"(~{abs(target - start) / m.units_per_rad:.3f} rad)\n")

    # Hold at start briefly to establish a flat baseline.
    bus.write_pos(sid, start, speed=0)
    t_log, p_log = [], []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.settle_secs:
        pos = bus.read_pos(sid)
        if pos is not None:
            t_log.append(time.perf_counter() - t0)
            p_log.append(pos)

    # ---- STEP ----
    t_step = time.perf_counter() - t0
    bus.write_pos(sid, target, speed=0)
    while time.perf_counter() - t0 < args.settle_secs + args.secs:
        pos = bus.read_pos(sid)
        if pos is not None:
            t_log.append(time.perf_counter() - t0)
            p_log.append(pos)

    bus.set_torque([sid], False)
    bus.close()

    t = np.array(t_log) - t_step          # t=0 at the step command
    pos = np.array(p_log, dtype=float)
    post = t >= 0
    tp, pp = t[post], pos[post]
    n = len(tp)
    rate = n / args.secs
    span = target - start
    final = float(np.median(pp[tp > tp.max() - 0.2])) if n else float("nan")
    reached = final - start

    def t_to_frac(frac):
        thr = start + frac * span
        for ti, pi in zip(tp, pp):
            if (span > 0 and pi >= thr) or (span < 0 and pi <= thr):
                return ti
        return None

    t63, t90 = t_to_frac(0.63), t_to_frac(0.90)
    peak = (pp.max() if span > 0 else pp.min()) if n else start
    overshoot = (peak - target) / span * 100 if span else 0.0

    print(f"sample rate: {rate:.0f} Hz ({n} samples over {args.secs}s)")
    print(f"commanded step: {span:+d} units   actually reached: {reached:+.0f} units "
          f"({reached / span * 100:.0f}% of command)" if span else "")
    print(f"tau  (time to 63%): {t63*1000:.0f} ms" if t63 else "tau: not reached")
    print(f"t90  (time to 90%): {t90*1000:.0f} ms" if t90 else "t90: not reached")
    print(f"overshoot: {overshoot:.0f}%   final steady error: {target - final:+.0f} units")
    print(f"\n>> implied max joint speed ~ {abs(span)/m.units_per_rad/(t63 or 1):.1f} rad/s "
          f"(at 40 Hz the policy can command up to "
          f"{80/m.units_per_rad*40:.0f} rad/s via the 80-unit clamp)")

    # Compact trace for eyeballing the curve shape.
    print("\nt(ms)  pos   (every ~25ms post-step)")
    last = -1
    for ti, pi in zip(tp, pp):
        if ti * 1000 - last >= 25:
            bar = int(abs(pi - start) / max(abs(span), 1) * 40)
            print(f"{ti*1000:5.0f}  {pi:4.0f}  {'#' * bar}")
            last = ti * 1000


if __name__ == "__main__":
    main()
