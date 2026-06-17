#!/usr/bin/env python3
# deploy_standing.py
"""
Real-robot inference loop for the proprioceptive standing policy (RUN ON THE PI).

Pipeline (must match src/environments/standing_env.py exactly):
  sensors -> per-frame feature (57) -> 4-frame history (228) -> VecNormalize -> PPO.predict
  -> tau=0.3 action smoothing -> clip to sim range -> rad->units (sign/limit) -> servo write

Per-frame feature order (standing_env._proprioceptive_features):
  proj_grav(3) | base_ang_vel(3) | (jpos - default_joint_pos)(17) | jvel(17) | last_action(17)
History: last 4 frames, LEFT-padded with zeros until filled, concatenated oldest->newest.

Defaults match the deployed model:
  model  models/final_real_standing_model.zip   (19M, tau=0.3)
  vecnorm models/vecnorm_real_standing.pkl
  tau 0.3, control 40 Hz.

  python scripts/deploy/deploy_standing.py --dry-run     # no hardware: load+predict once, print
  python scripts/deploy/deploy_standing.py               # closed loop on the Pi
"""

from __future__ import annotations
import argparse
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_real_map import SimRealMap, DEFAULT_MAP  # noqa: E402

NJ = 17
FRAME = 6 + 3 * NJ          # 57
HISTORY = 4
OBS_DIM = FRAME * HISTORY   # 228


def load_vecnorm_stats(path):
    with open(path, "rb") as f:
        vn = pickle.load(f)
    mean = np.asarray(vn.obs_rms.mean, dtype=np.float32)
    var = np.asarray(vn.obs_rms.var, dtype=np.float32)
    eps = float(getattr(vn, "epsilon", 1e-8))
    clip = float(getattr(vn, "clip_obs", 50.0))
    assert mean.shape[0] == OBS_DIM, f"vecnorm obs dim {mean.shape[0]} != {OBS_DIM}"
    return mean, var, eps, clip


def normalize(obs, mean, var, eps, clip):
    return np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)


class ObsBuilder:
    """Maintains history + finite-diff joint velocity, emits the 228-dim obs."""

    def __init__(self, m: SimRealMap, dt: float):
        self.m = m
        self.dt = dt
        self.hist = deque(maxlen=HISTORY)
        self.prev_jpos = None
        self.last_action = np.zeros(NJ, dtype=np.float32)  # smoothed target (sim rad), env init=0

    def frame(self, proj_grav, ang_vel, jpos):
        if self.prev_jpos is None:
            jvel = np.zeros(NJ, dtype=np.float32)
        else:
            jvel = ((jpos - self.prev_jpos) / self.dt).astype(np.float32)
        self.prev_jpos = jpos.copy()
        return np.concatenate([proj_grav, ang_vel, jpos, jvel, self.last_action]).astype(np.float32)

    def obs(self, frame):
        self.hist.append(frame)
        frames = list(self.hist)
        if len(frames) < HISTORY:
            frames = [np.zeros(FRAME, dtype=np.float32)] * (HISTORY - len(frames)) + frames
        return np.concatenate(frames).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "final_real_standing_model.zip"))
    p.add_argument("--vecnorm", default=str(ROOT / "models" / "vecnorm_real_standing.pkl"))
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--tau", type=float, default=0.3, help="action smoothing (MUST match training)")
    p.add_argument("--hz", type=float, default=40.0)
    p.add_argument("--ramp-secs", type=float, default=2.0, help="open-loop ramp to home before closed loop")
    p.add_argument("--tilt-cut", type=float, default=0.5, help="hold+cut torque if upright_cos < this (~60deg)")
    p.add_argument("--max-step-units", type=int, default=80, help="per-joint per-step servo move clamp")
    p.add_argument("--speed", type=int, default=0, help="servo move speed (0=max)")
    p.add_argument("--require-verified", action="store_true",
                   help="refuse to drive joints whose sign is not bench-verified")
    p.add_argument("--dry-run", action="store_true", help="no hardware: load, predict once with zero sensors, print")
    args = p.parse_args()

    dt = 1.0 / args.hz
    m = SimRealMap(args.map)
    home_units = m.rad_to_units(m.default_joint_pos)

    unverified = m.unverified_idxs()
    if unverified:
        msg = f"[warn] {len(unverified)} joints have UNVERIFIED sign: {unverified} (run verify_signs.py)"
        if args.require_verified:
            print(msg + " -- aborting (--require-verified).")
            return
        print(msg)

    # ---- load policy + normalization ----
    from stable_baselines3 import PPO  # noqa: E402
    print(f"Loading model {args.model}")
    model = PPO.load(args.model, device="cpu")
    mean, var, eps, clip = load_vecnorm_stats(args.vecnorm)
    print(f"VecNormalize stats OK (obs_dim={OBS_DIM}, clip={clip})")

    builder = ObsBuilder(m, dt)

    def predict_units(proj_grav, ang_vel, jpos):
        frame = builder.frame(proj_grav, ang_vel, jpos)
        obs = builder.obs(frame)
        raw_action = model.predict(normalize(obs, mean, var, eps, clip), deterministic=True)[0]
        raw_action = np.asarray(raw_action, dtype=np.float32).ravel()
        # Match StandingEnv._process_action: SB3 clips the action to the space BEFORE the
        # env smooths it, so clip raw -> EMA-smooth -> clip again.
        raw_action = np.clip(raw_action, m.range_lo, m.range_hi)
        applied = (1.0 - args.tau) * builder.last_action + args.tau * raw_action
        applied = np.clip(applied, m.range_lo, m.range_hi)
        builder.last_action = applied
        return m.rad_to_units(applied), applied

    if args.dry_run:
        pg = np.array([0, 0, -1], dtype=np.float32)   # perfectly upright
        av = np.zeros(3, dtype=np.float32)
        jpos = np.zeros(NJ, dtype=np.float32)         # at home (encoder==default -> jpos 0)
        for step in range(HISTORY):                   # fill history
            units, applied = predict_units(pg, av, jpos)
        print("\n[dry-run] first deterministic action (sim rad):")
        print(np.array2string(applied, precision=3, suppress_small=True))
        print("[dry-run] -> servo units:")
        for i in range(NJ):
            print(f"  {m.joints[i].dof:<18} servo {m.servo_ids[i]:2d}: {int(units[i])}")
        print("\n[dry-run] OK: model + vecnorm + map wired correctly. No hardware touched.")
        return

    # ---- hardware ----
    from hardware import ServoBus, IMU  # noqa: E402
    bus = ServoBus().connect()
    imu = IMU().connect()

    def read_sensors():
        pg = imu.projected_gravity()
        av = imu.angular_velocity()
        units = bus.read_all(m.servo_ids)
        jpos = (m.units_to_rad(units) - m.default_joint_pos).astype(np.float32)
        return pg, av, jpos

    try:
        print("Enabling torque, calibrating gyro, ramping to home...")
        bus.set_torque(m.servo_ids, True)
        cur = bus.read_all(m.servo_ids)
        steps = max(1, int(args.ramp_secs / dt))
        for k in range(1, steps + 1):
            u = (cur + (home_units - cur) * k / steps).round().astype(int)
            bus.write_all(m.servo_ids, u, speed=args.speed)
            time.sleep(dt)
        imu.calibrate_gyro_bias(seconds=1.5)

        # warm up history with real frames at rest
        for _ in range(HISTORY):
            pg, av, jpos = read_sensors()
            builder.obs(builder.frame(pg, av, jpos))
            time.sleep(dt)

        print("Closed loop running (Ctrl-C to stop).")
        prev_units = home_units.copy().astype(float)
        while True:
            t0 = time.time()
            pg, av, jpos = read_sensors()

            upright_cos = -float(pg[2])
            if upright_cos < args.tilt_cut:
                print(f"\n[SAFETY] upright_cos={upright_cos:.2f} < {args.tilt_cut}: cutting torque.")
                bus.set_torque(m.servo_ids, False)
                break

            units, _ = predict_units(pg, av, jpos)
            # per-step move clamp (rate limit), then write
            units = np.clip(units, prev_units - args.max_step_units, prev_units + args.max_step_units)
            units = np.clip(units, m.lim_lo, m.lim_hi).round().astype(int)
            bus.write_all(m.servo_ids, units, speed=args.speed)
            prev_units = units.astype(float)

            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nStopping: ramping to home and disabling torque.")
        try:
            cur = bus.read_all(m.servo_ids)
            steps = max(1, int(1.0 / dt))
            for k in range(1, steps + 1):
                u = (cur + (home_units - cur) * k / steps).round().astype(int)
                bus.write_all(m.servo_ids, u, speed=args.speed)
                time.sleep(dt)
        except Exception:
            pass
    finally:
        bus.set_torque(m.servo_ids, False)
        bus.close()
        imu.close()


if __name__ == "__main__":
    main()
