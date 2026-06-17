#!/usr/bin/env python3
# measure_jitter.py
"""
Quantify standing-policy jitter for sim-to-real readiness.

Runs ONE deterministic episode and measures, per joint:
  - RMS joint speed (rad/s, deg/s)      -> how much each joint moves
  - velocity reversal rate              -> fraction of control steps where the
                                           joint velocity flips sign; ~0 = smooth,
                                           high = buzzing/chatter (the jitter)
  - effective jitter frequency (Hz)     -> reversal_rate * control_hz / 2
  - applied target step delta (RMS)     -> change in the post-smoothing servo
                                           target each control step (rad)

Mirrors scripts/evaluate.py env construction exactly (same VecNormalize, same
deterministic action) so the numbers correspond to what the recorded video shows.

Usage:
  python scripts/debug/measure_jitter.py --task standing \
    --model models/final_real_standing_model.zip \
    --vecnorm models/vecnorm_real_standing.pkl \
    --config config/real_humanoid_config.yaml --max-steps 5000
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_eval_env(task, config):
    if task == "standing":
        from src.environments import make_standing_env
        return make_standing_env(render_mode=None, config=config)
    elif task == "walking":
        from src.environments import make_walking_env
        return make_walking_env(render_mode=None, config=config)
    raise ValueError(task)


def find_wrapper_with(attr, env):
    """Walk the gym wrapper chain to find the object exposing `attr`."""
    cur = env
    for _ in range(20):
        if hasattr(cur, attr):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="standing", choices=["standing", "walking"])
    p.add_argument("--model", required=True)
    p.add_argument("--vecnorm", default=None)
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--tau", type=float, default=None,
                   help="Override action_smoothing_tau (inference smoothing probe)")
    args = p.parse_args()

    with open(args.config) as f:
        full = yaml.safe_load(f)
    config = full.get(args.task, full)
    if args.tau is not None:
        config = dict(config)
        config["action_smoothing"] = True
        config["action_smoothing_tau"] = args.tau
        print(f">>> OVERRIDE action_smoothing_tau = {args.tau}")

    env = make_eval_env(args.task, config)
    vec = DummyVecEnv([lambda: env])
    if args.vecnorm and os.path.exists(args.vecnorm):
        vec = VecNormalize.load(args.vecnorm, vec)
        vec.training = False
        vec.norm_reward = False

    model = PPO.load(args.model, env=vec)

    inner = vec.venv.envs[0]
    base = inner.unwrapped                       # MujocoEnv (qpos/qvel/data)
    wrap = find_wrapper_with("_last_action_rate", inner)  # StandingEnv wrapper
    nj = int(getattr(wrap, "n_joints", 17))
    control_hz = 1.0 / float(base.dt)

    # actuator names for human-readable mapping
    try:
        act_names = [base.model.actuator(i).name for i in range(base.model.nu)]
    except Exception:
        act_names = [f"act{i}" for i in range(nj)]

    obs = vec.reset()
    jvel_hist = []      # joint velocities (rad/s) each step
    adelta_hist = []    # applied-target step delta (rad) each step

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec.step(action)
        jvel_hist.append(np.asarray(base.data.qvel[6:6 + nj], dtype=np.float64).copy())
        if wrap is not None:
            adelta_hist.append(np.asarray(wrap._last_action_rate, dtype=np.float64).ravel().copy())
        if done[0]:
            break

    jvel = np.array(jvel_hist)            # (T, nj)
    T = jvel.shape[0]
    adelta = np.array(adelta_hist) if adelta_hist else None

    rms_speed = np.sqrt(np.mean(jvel ** 2, axis=0))        # rad/s per joint
    max_speed = np.max(np.abs(jvel), axis=0)
    # reversal rate: sign change of velocity between consecutive steps
    sign = np.sign(jvel)
    reversals = np.mean(sign[1:] * sign[:-1] < 0, axis=0)  # fraction of steps
    jit_hz = reversals * control_hz / 2.0
    rms_adelta = (np.sqrt(np.mean(adelta ** 2, axis=0)) if adelta is not None
                  else np.zeros(nj))

    print(f"\n{'='*78}")
    print(f"JITTER DIAGNOSTIC  ({T} steps, control {control_hz:.1f} Hz, dt={base.dt:.4f}s)")
    print(f"{'='*78}")
    print("Per joint, sorted by reversal rate (jitter signature):\n")
    print(f"{'joint(actuator)':<20} {'RMSspeed':>9} {'RMSspeed':>9} {'maxspeed':>9} "
          f"{'reversal':>9} {'jit~Hz':>7} {'RMSdΔtgt':>9}")
    print(f"{'':<20} {'rad/s':>9} {'deg/s':>9} {'deg/s':>9} "
          f"{'rate':>9} {'':>7} {'rad':>9}")
    print("-" * 78)
    order = np.argsort(-reversals)
    for j in order:
        name = act_names[j] if j < len(act_names) else f"j{j}"
        print(f"{j:>2} {name:<16} {rms_speed[j]:>9.3f} {np.degrees(rms_speed[j]):>9.1f} "
              f"{np.degrees(max_speed[j]):>9.1f} {reversals[j]:>9.2f} {jit_hz[j]:>7.1f} "
              f"{rms_adelta[j]:>9.4f}")
    print("-" * 78)
    print(f"{'AGGREGATE':<20} {np.sqrt(np.mean(rms_speed**2)):>9.3f} "
          f"{np.degrees(np.sqrt(np.mean(rms_speed**2))):>9.1f} "
          f"{np.degrees(np.max(max_speed)):>9.1f} {np.mean(reversals):>9.2f} "
          f"{np.mean(jit_hz):>7.1f} {np.sqrt(np.mean(rms_adelta**2)):>9.4f}")
    print(f"{'='*78}")
    print("Read: reversal rate ~0.0 = smooth; >0.3 = oscillating; ~0.5+ = step-rate buzz.")
    print("RMSdΔtgt = RMS change in applied servo target per control step (radians).")

    vec.close()


if __name__ == "__main__":
    main()
