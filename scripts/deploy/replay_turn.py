#!/usr/bin/env python3
# replay_turn.py  --  RUN ON THE PI
"""
Open-loop replay of a recorded turning trajectory + hardware WTR logging.

Streams the position targets from record_turn_trajectory.py on a wall clock -- NO policy,
NO feedback, so the closed-loop seizure mechanism does not exist here. While replaying it
logs the two signals the hardware figure needs:

  - pelvis IMU gyro z (the pelvis actually rotating -- or not), integrated to a yaw estimate
  - waist_yaw servo readback (servo 1) -- the waist-twist DOF the hack policy exploits

Expected result: the HACK trajectory sweeps waist_yaw while pelvis gyro z stays ~0
(torso-heading "turn" = waist twist); the FIX trajectory shows no waist sweep.

Sequence: torque on -> ramp to trajectory frame 0 over --ramp-secs -> settle -> stream all
frames at the recorded rate -> ramp home -> torque off. Ctrl-C at any point ramps home and
releases. Safety tilt-cut (debounced, same convention as deploy_standing.py) drops torque
if the robot actually falls.

  python3 replay_turn.py --traj turn_traj_hack.npz --log turn_log_hack.csv
  python3 replay_turn.py --traj turn_traj_fix.npz  --log turn_log_fix.csv
  python3 replay_turn.py --traj turn_traj_hack.npz --dry-run     # print, no hardware
  python3 replay_turn.py --traj ... --no-imu                     # servo readback only
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_real_map import SimRealMap, DEFAULT_MAP  # noqa: E402

WAIST_YAW_IDX = 10  # joint index of waist_yaw (servo 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True, help=".npz from record_turn_trajectory.py")
    p.add_argument("--log", default=None, help="output CSV (default: <traj-stem>_log.csv)")
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--ramp-secs", type=float, default=3.0, help="ramp to frame 0 before replay")
    p.add_argument("--settle-secs", type=float, default=2.0, help="hold frame 0 before streaming")
    p.add_argument("--speed", type=int, default=0, help="servo move speed (0=max)")
    p.add_argument("--max-step-units", type=int, default=80,
                   help="per-joint per-step move clamp during replay (safety rail; recorded "
                        "trajectories move ~a few units/step, so this should never engage)")
    p.add_argument("--tilt-cut", type=float, default=0.5, help="cut torque if upright_cos < this")
    p.add_argument("--tilt-debounce", type=int, default=3)
    p.add_argument("--no-imu", action="store_true", help="skip IMU (log servo readback only)")
    p.add_argument("--imu-calib", default=str(Path(__file__).resolve().parents[2] / "config" / "imu_calib.yaml"))
    p.add_argument("--read-all-every", type=int, default=0,
                   help="also read ALL 17 servo positions every N frames (0=off; slows the bus, "
                        "the per-frame log always reads waist_yaw only)")
    p.add_argument("--dry-run", action="store_true", help="no hardware: print trajectory stats")
    args = p.parse_args()

    m = SimRealMap(args.map)
    data = np.load(args.traj, allow_pickle=True)
    targets_rad = np.asarray(data["targets_rad"], dtype=np.float32)   # [T, 17] absolute sim rad
    dt = float(data["dt"])
    yaw_cmd = np.asarray(data["yaw_cmd"], dtype=np.float32)
    T = targets_rad.shape[0]
    assert targets_rad.shape[1] == m.n == 17

    units = np.stack([m.rad_to_units(targets_rad[t]) for t in range(T)]).astype(int)  # [T, 17]
    step_moves = np.abs(np.diff(units, axis=0))
    wy = WAIST_YAW_IDX
    print(f"Trajectory: {args.traj}  {T} frames @ {1.0 / dt:.0f} Hz = {T * dt:.1f}s  "
          f"(model {data['model']}, heading_source={data['heading_source']})")
    print(f"  per-step servo move: max {step_moves.max()} units, p99 "
          f"{int(np.percentile(step_moves, 99))} (clamp {args.max_step_units})")
    print(f"  waist_yaw (servo {m.servo_ids[wy]}) target range: "
          f"{units[:, wy].min()}..{units[:, wy].max()} units "
          f"({np.ptp(targets_rad[:, wy]):.3f} rad)")
    if step_moves.max() > args.max_step_units:
        print("  WARNING: trajectory exceeds the per-step clamp; replay will lag those frames.")

    log_path = args.log or (Path(args.traj).stem + "_log.csv")
    if args.dry_run:
        print("[dry-run] OK: trajectory loads and converts. No hardware touched.")
        return

    from hardware import ServoBus, IMU  # noqa: E402
    bus = ServoBus().connect()
    imu = None
    if not args.no_imu:
        axis_remap = None
        try:
            import yaml  # noqa: E402
            with open(args.imu_calib) as f:
                axis_remap = np.asarray(yaml.safe_load(f)["axis_remap"], dtype=np.float32)
            print(f"IMU axis_remap loaded from {args.imu_calib}")
        except FileNotFoundError:
            pass
        imu = IMU(axis_remap=axis_remap).connect()

    home_units = m.rad_to_units(np.zeros(m.n, dtype=np.float32))
    waist_sid = int(m.servo_ids[wy])

    def ramp_to(target, secs):
        cur = bus.read_all(m.servo_ids).astype(float)
        steps = max(1, int(secs / dt))
        for k in range(1, steps + 1):
            u = (cur + (target - cur) * k / steps).round().astype(int)
            bus.write_all(m.servo_ids, u, speed=args.speed)
            time.sleep(dt)

    rows = []
    tilt_bad = 0
    cut = False
    try:
        print("Torque on; ramping to trajectory start pose...")
        bus.set_torque(m.servo_ids, True)
        ramp_to(units[0], args.ramp_secs)
        time.sleep(args.settle_secs)
        if imu is not None:
            print("Calibrating gyro bias (keep the robot still)...")
            imu.calibrate_gyro_bias(seconds=1.5)

        print(f"REPLAYING {T} frames open-loop. Hands off.")
        yaw_int = 0.0            # integrated pelvis gyro z -> pelvis yaw estimate (rad)
        prev = units[0].astype(float)
        t0 = time.time()
        t_last = t0
        for k in range(T):
            # wall-clock schedule: frame k goes out at t0 + k*dt regardless of bus jitter
            now = time.time()
            wait = t0 + k * dt - now
            if wait > 0:
                time.sleep(wait)
            u = np.clip(units[k], prev - args.max_step_units, prev + args.max_step_units)
            u = np.clip(u, m.lim_lo, m.lim_hi).round().astype(int)
            bus.write_all(m.servo_ids, u, speed=args.speed)
            prev = u.astype(float)

            t_now = time.time()
            gyro_z = pg0 = pg1 = pg2 = float("nan")
            if imu is not None:
                av = imu.angular_velocity()
                pg = imu.projected_gravity()
                gyro_z = float(av[2])
                pg0, pg1, pg2 = (float(pg[0]), float(pg[1]), float(pg[2]))
                yaw_int += gyro_z * (t_now - t_last)
                upright_cos = -pg2
                if upright_cos < args.tilt_cut:
                    tilt_bad += 1
                    if tilt_bad >= args.tilt_debounce:
                        print(f"\n[SAFETY] upright_cos={upright_cos:.2f} at frame {k}: cutting torque.")
                        bus.set_torque(m.servo_ids, False)
                        cut = True
                        break
                else:
                    tilt_bad = 0
            t_last = t_now

            try:
                waist_units = int(bus.read_pos(waist_sid))
            except Exception:
                waist_units = -1
            waist_rad = (m.signs[wy] * (waist_units - m.centers[wy]) / m.units_per_rad
                         if waist_units >= 0 else float("nan"))

            row = {
                "frame": k, "t": t_now - t0, "yaw_cmd": float(yaw_cmd[k]),
                "waist_yaw_target_rad": float(targets_rad[k, wy]),
                "waist_yaw_units": waist_units, "waist_yaw_rad": float(waist_rad),
                "pelvis_gyro_z": gyro_z, "pelvis_yaw_int": yaw_int,
                "proj_grav_x": pg0, "proj_grav_y": pg1, "proj_grav_z": pg2,
            }
            if args.read_all_every and k % args.read_all_every == 0:
                all_units = bus.read_all(m.servo_ids)
                for i in range(m.n):
                    row[f"j{i}_units"] = int(all_units[i])
            rows.append(row)

            if k % 40 == 0:
                print(f"  [{k:4d}/{T}] cmd={yaw_cmd[k]:+.1f}  waist={waist_units:4d}u "
                      f"({waist_rad:+.3f} rad)  gyro_z={gyro_z:+.3f}  yaw_int={yaw_int:+.3f}")

        if not cut:
            print("Replay complete. Ramping home...")
            ramp_to(home_units, 2.0)
    except KeyboardInterrupt:
        print("\nInterrupted: ramping home...")
        try:
            ramp_to(home_units, 2.0)
        except Exception:
            pass
    finally:
        try:
            bus.set_torque(m.servo_ids, False)
        except Exception:
            pass
        bus.close()
        if imu is not None:
            imu.close()
        if rows:
            keys = list(rows[0].keys())
            for r in rows:  # read-all frames have extra cols; unify
                for kk in r:
                    if kk not in keys:
                        keys.append(kk)
            with open(log_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            print(f"Logged {len(rows)} frames -> {log_path}")
            wr = [r["waist_yaw_rad"] for r in rows if np.isfinite(r["waist_yaw_rad"])]
            if wr:
                print(f"  waist_yaw readback range: {np.ptp(wr):.3f} rad")
            yi = [r["pelvis_yaw_int"] for r in rows if np.isfinite(r["pelvis_yaw_int"])]
            if yi:
                print(f"  integrated pelvis yaw net: {yi[-1]:+.3f} rad, range {np.ptp(yi):.3f} rad")


if __name__ == "__main__":
    main()
