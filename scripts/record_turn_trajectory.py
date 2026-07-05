"""Record a turning policy's joint-target trajectory for open-loop replay on the robot.

Runs a trained RealTurningEnv policy through the standardized turn-in-place schedule
(settle -> +0.5 rad/s -> -0.5 rad/s, same as measure_waist_twist_real.py) and records the
APPLIED (tau-smoothed) position target each control step -- the exact 17-dim absolute
sim-radian signal deploy_standing.py converts with SimRealMap.rad_to_units. Replaying
these targets on a clock reproduces the policy's motion with no feedback loop.

Also logs sim ground truth (pelvis yaw, torso yaw, waist_yaw joint angle, height) so the
hardware logs from replay_turn.py can be overlaid against what the sim did.

Usage:
    python scripts/record_turn_trajectory.py --arm hack
    python scripts/record_turn_trajectory.py --arm fix
    # or explicit paths:
    python scripts/record_turn_trajectory.py --model models/final_real_turn_hack.zip \
        --vecnorm models/vecnorm_real_turn_hack.pkl --config config/turning/heading_torso.yaml \
        --output data/turn_traj_hack.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "src"))

from src.environments import make_real_turning_env  # noqa: E402

# Same schedule as measure_waist_twist_real.py: (start, end, vx, vy, yaw_rate, phase)
SCHEDULE = [
    (0, 100, 0.0, 0.0, 0.0, "settle"),
    (100, 300, 0.0, 0.0, +0.5, "turn_ccw"),
    (300, 500, 0.0, 0.0, -0.5, "turn_cw"),
]
TOTAL_STEPS = 500

ARMS = {
    "hack": ("models/final_real_turn_hack.zip", "models/vecnorm_real_turn_hack.pkl",
             "config/turning/heading_torso.yaml"),
    "fix": ("models/final_real_turn_fix.zip", "models/vecnorm_real_turn_fix.pkl",
            "config/turning/heading_pelvis.yaml"),
}

WAIST_YAW_IDX = 10  # action/joint index of waist_yaw (Revolute 19, servo 1)


def phase_for(step):
    for s, e, vx, vy, yr, lab in SCHEDULE:
        if s <= step < e:
            return vx, vy, yr, lab
    s, e, vx, vy, yr, lab = SCHEDULE[-1]
    return vx, vy, yr, lab


def yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def main():
    import mujoco
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("hack", "fix"), default=None,
                    help="shorthand for the model/vecnorm/config triple of one ablation arm")
    ap.add_argument("--model", default=None)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", default=None, help="output .npz (default data/turn_traj_<arm>.npz)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.arm:
        model_path, vecnorm_path, config_path = ARMS[args.arm]
        model_path = args.model or model_path
        vecnorm_path = args.vecnorm or vecnorm_path
        config_path = args.config or config_path
        out_path = args.output or f"data/turn_traj_{args.arm}.npz"
    else:
        if not (args.model and args.vecnorm and args.config and args.output):
            ap.error("either --arm or all of --model/--vecnorm/--config/--output")
        model_path, vecnorm_path, config_path = args.model, args.vecnorm, args.config
        out_path = args.output

    cfg = (yaml.safe_load(open(PROJ / config_path)).get("turning") or {}).copy()
    # Deterministic nominal-body rollout: the robot is ONE fixed body, so the recorded
    # trajectory must come from the unrandomized model with clean obs. actuator_lag stays
    # ON -- it models the real servo response, so the policy runs in-distribution.
    cfg.update(obs_noise=False, domain_rand=False, actuator_rand=False,
               push_enabled=False, max_episode_steps=TOTAL_STEPS + 20)
    env = make_real_turning_env(config=cfg)
    dt = float(env.unwrapped.dt)

    m = env.unwrapped.model
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("0003_8", "base_link")}

    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(str(PROJ / vecnorm_path), venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(str(PROJ / model_path), device="cpu")
    print(f"Recording {model_path} (heading_source={cfg.get('heading_source')}) "
          f"dt={dt:.4f}s -> {1.0 / dt:.0f} Hz, {TOTAL_STEPS} steps = {TOTAL_STEPS * dt:.1f}s")

    np.random.seed(args.seed)
    obs = venv.reset()

    targets = np.zeros((TOTAL_STEPS, 17), dtype=np.float32)   # applied smoothed sim-rad targets
    yaw_cmd = np.zeros(TOTAL_STEPS, dtype=np.float32)
    pelvis_yaw = np.zeros(TOTAL_STEPS, dtype=np.float32)
    torso_yaw = np.zeros(TOTAL_STEPS, dtype=np.float32)
    waist_yaw = np.zeros(TOTAL_STEPS, dtype=np.float32)
    height = np.zeros(TOTAL_STEPS, dtype=np.float32)
    phases = []

    for step in range(TOTAL_STEPS):
        vx, vy, yr, lab = phase_for(step)
        env.fixed_command = (vx, vy, yr)
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = venv.step(a)
        # prev_action = the tau-smoothed, range-clipped target the sim actuators were driven
        # with this step (StandingEnv._process_action) -- the replay signal.
        targets[step] = np.asarray(env.prev_action, dtype=np.float32)
        yaw_cmd[step] = yr
        d = env.unwrapped.data
        pelvis_yaw[step] = yaw_from_quat(d.xquat[bid["base_link"]])
        torso_yaw[step] = yaw_from_quat(d.xquat[bid["0003_8"]])
        waist_yaw[step] = float(d.qpos[7 + WAIST_YAW_IDX])
        height[step] = float(infos[0]["height"])
        phases.append(lab)
        if bool(done[0]):
            raise RuntimeError(f"episode terminated at step {step} (fell?) -- trajectory invalid")

    twist = np.unwrap(torso_yaw) - np.unwrap(pelvis_yaw)
    turn = slice(100, TOTAL_STEPS)
    print(f"  sim ground truth over turn phases: pelvis_yaw range {np.ptp(np.unwrap(pelvis_yaw)[turn]):.3f} rad, "
          f"torso_yaw range {np.ptp(np.unwrap(torso_yaw)[turn]):.3f} rad, "
          f"waist_yaw range {np.ptp(waist_yaw[turn]):.3f} rad, WTR(std twist) {np.std(twist[turn]):.4f}")
    print(f"  target stats: max|target| {np.abs(targets).max():.3f} rad, "
          f"max per-step change {np.abs(np.diff(targets, axis=0)).max():.4f} rad "
          f"({np.abs(np.diff(targets, axis=0)).max() * 195:.1f} servo units)")

    out = PROJ / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        targets_rad=targets, dt=dt, yaw_cmd=yaw_cmd,
        sim_pelvis_yaw=pelvis_yaw, sim_torso_yaw=torso_yaw, sim_waist_yaw=waist_yaw,
        sim_height=height, phase=np.array(phases),
        model=str(model_path), heading_source=str(cfg.get("heading_source")),
        schedule=np.array(SCHEDULE, dtype=object),
    )
    print(f"Saved {out} ({TOTAL_STEPS} frames x 17 joints @ {1.0 / dt:.0f} Hz)")


if __name__ == "__main__":
    main()
