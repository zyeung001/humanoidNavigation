"""Measure waist-twist ratio (WTR) on a walking model under a standardized turning eval.

Primary metric:
    WTR = std(torso_yaw - pelvis_yaw)    over phases B+C of the schedule.

Schedule (default, at 50 Hz physics step):
    Phase A (steps   0-100): vx=0.3,  yaw_rate= 0.0    (warmup walk)
    Phase B (steps 100-300): vx=0.3,  yaw_rate=+0.5    (left turn)
    Phase C (steps 300-500): vx=0.3,  yaw_rate=-0.5    (right turn)

Outputs:
    data/wtr_per_step.csv   (per-step yaw signals across all episodes)
    data/wtr_summary.csv    (one row per episode with WTR and ranges)

Usage:
    # Pipeline validation with random policy:
    py -3.13 scripts/measure_waist_twist.py --random-policy --episodes 3

    # Real measurement on a trained model:
    py -3.13 scripts/measure_waist_twist.py \
        --model models/walking/final/model_final.zip \
        --vecnorm models/walking/final/vecnorm_final.pkl \
        --episodes 20
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils import configure_mujoco_gl
configure_mujoco_gl()

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environments import make_walking_curriculum_env


# ---------- Schedule ----------
SCHEDULE = [
    # (start_step_inclusive, end_step_exclusive, vx, vy, yaw_rate, phase_label)
    (0,   100, 0.3, 0.0,  0.0, 'A'),
    (100, 300, 0.3, 0.0, +0.5, 'B'),
    (300, 500, 0.3, 0.0, -0.5, 'C'),
]
TURN_PHASE_START = 100  # WTR computed from this step onward (B + C)


def phase_for_step(step: int):
    for s, e, vx, vy, yr, label in SCHEDULE:
        if s <= step < e:
            return label, vx, vy, yr
    return SCHEDULE[-1][5], SCHEDULE[-1][2], SCHEDULE[-1][3], SCHEDULE[-1][4]


# ---------- Helpers ----------
def yaw_from_quat(qw, qx, qy, qz):
    """Extract yaw (rotation around world z) from a unit quaternion."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def get_inner_env(vec_env):
    """Drill through VecNormalize / DummyVecEnv wrappers down to the actual gym env."""
    e = vec_env
    while hasattr(e, 'venv'):
        e = e.venv
    if hasattr(e, 'envs'):
        return e.envs[0]
    return e


def body_id_by_name(model, name: str):
    for i in range(model.nbody):
        if model.body(i).name == name:
            return i
    return None


def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle differences to [-pi, pi]."""
    return np.arctan2(np.sin(x), np.cos(x))


# ---------- Random-policy stub ----------
class _RandomPolicy:
    """Minimal predict() interface for pipeline validation."""

    def __init__(self, action_space, n_envs=1):
        self.action_space = action_space
        self.n_envs = n_envs

    def predict(self, obs, deterministic=True):
        a = np.stack([self.action_space.sample() for _ in range(self.n_envs)])
        return a, None


# ---------- Core eval loop ----------
def run_episode(model, vec_env, episode_idx, body_ids, per_step_writer, max_steps=500):
    inner = get_inner_env(vec_env)
    torso_id  = body_ids['torso']
    pelvis_id = body_ids['pelvis']
    rfoot_id  = body_ids['right_foot']
    lfoot_id  = body_ids['left_foot']

    # Set initial command BEFORE reset so reset reads phase A.
    inner.fixed_command = (SCHEDULE[0][2], SCHEDULE[0][3], SCHEDULE[0][4])

    obs = vec_env.reset()
    records = []

    for step in range(max_steps):
        # Update fixed_command based on schedule. The env's step() reads
        # fixed_command on every call, so mutation is live.
        phase, vx, vy, yr = phase_for_step(step)
        inner.fixed_command = (vx, vy, yr)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        data = inner.unwrapped.data
        torso_q  = data.xquat[torso_id]    # (w, x, y, z)
        pelvis_q = data.xquat[pelvis_id]
        torso_yaw  = yaw_from_quat(*torso_q)
        pelvis_yaw = yaw_from_quat(*pelvis_q)

        rfoot_xy = data.xpos[rfoot_id][:2]
        lfoot_xy = data.xpos[lfoot_id][:2]

        rec = {
            'episode': episode_idx,
            'step': step,
            'phase': phase,
            'torso_yaw': torso_yaw,
            'pelvis_yaw': pelvis_yaw,
            'twist': float(wrap_pi(np.array([torso_yaw - pelvis_yaw]))[0]),
            'cmd_yaw_rate': float(inner.commanded_yaw_rate),
            'rfoot_x': float(rfoot_xy[0]),
            'rfoot_y': float(rfoot_xy[1]),
            'lfoot_x': float(lfoot_xy[0]),
            'lfoot_y': float(lfoot_xy[1]),
        }
        records.append(rec)
        per_step_writer.writerow(rec)

        if bool(done[0]):
            break

    # Compute episode summary over the turning portion only.
    turn = [r for r in records if r['step'] >= TURN_PHASE_START]
    if not turn:
        return {
            'episode': episode_idx,
            'steps': len(records),
            'terminated_early': True,
            'wtr': float('nan'),
            'twist_mean': float('nan'),
            'twist_range': float('nan'),
            'pelvis_yaw_range': float('nan'),
            'torso_yaw_range': float('nan'),
            'foot_path_arc': float('nan'),
        }

    twist = np.array([r['twist'] for r in turn])
    pelvis_yaw = np.array([r['pelvis_yaw'] for r in turn])
    torso_yaw  = np.array([r['torso_yaw'] for r in turn])

    rfoot = np.array([(r['rfoot_x'], r['rfoot_y']) for r in turn])
    lfoot = np.array([(r['lfoot_x'], r['lfoot_y']) for r in turn])
    foot_path_arc = float(
        np.sum(np.linalg.norm(np.diff(rfoot, axis=0), axis=1))
        + np.sum(np.linalg.norm(np.diff(lfoot, axis=0), axis=1))
    )

    return {
        'episode': episode_idx,
        'steps': len(records),
        'terminated_early': len(records) < max_steps,
        'wtr': float(np.std(twist)),
        'twist_mean': float(np.mean(twist)),
        'twist_range': float(np.max(twist) - np.min(twist)),
        'pelvis_yaw_range': float(np.max(pelvis_yaw) - np.min(pelvis_yaw)),
        'torso_yaw_range':  float(np.max(torso_yaw)  - np.min(torso_yaw)),
        'foot_path_arc': foot_path_arc,
    }


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--vecnorm', type=str, default=None)
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--csv-out', type=str, default='data/wtr_per_step.csv')
    parser.add_argument('--summary-out', type=str, default='data/wtr_summary.csv')
    parser.add_argument('--random-policy', action='store_true',
                        help='Use random actions instead of a model (for pipeline validation).')
    args = parser.parse_args()

    # Walking config
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        cfg = (yaml_cfg.get('walking') or {}).copy()
    cfg['max_episode_steps'] = args.max_steps
    cfg['use_command_generator'] = False
    cfg['fixed_command'] = (SCHEDULE[0][2], SCHEDULE[0][3], SCHEDULE[0][4])
    # Force standard Humanoid-v5 (the training config points at an xml file
    # that lives on the other training machine).
    cfg.pop('xml_file', None)

    np.random.seed(args.seed)

    env = make_walking_curriculum_env(render_mode=None, config=cfg)
    vec_env = DummyVecEnv([lambda: env])
    if args.vecnorm and os.path.exists(args.vecnorm):
        print(f"Loading VecNormalize from: {args.vecnorm}")
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Resolve body IDs once.
    inner = get_inner_env(vec_env)
    model_mj = inner.unwrapped.model
    body_ids = {
        name: body_id_by_name(model_mj, name)
        for name in ('torso', 'pelvis', 'right_foot', 'left_foot')
    }
    missing = [k for k, v in body_ids.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing required bodies in MuJoCo model: {missing}")
    print(f"Body IDs: {body_ids}")

    # Policy
    if args.random_policy or args.model is None:
        policy = _RandomPolicy(env.action_space)
        print("Policy: random (pipeline validation only)")
    else:
        custom_objects = {
            'learning_rate': 0.0,
            'lr_schedule': lambda _: 0.0,
            'clip_range': lambda _: 0.0,
        }
        policy = PPO.load(args.model, device='cpu', custom_objects=custom_objects)
        print(f"Policy: loaded model from {args.model}")
        print(f"  expected obs dim: {policy.observation_space.shape}")
        print(f"  env obs dim:      {vec_env.observation_space.shape}")
        if policy.observation_space.shape != vec_env.observation_space.shape:
            raise RuntimeError("Obs dim mismatch between policy and env. Retrain or adapt.")

    # Output files
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    per_step_f = open(args.csv_out, 'w', newline='')
    per_step_w = csv.DictWriter(per_step_f, fieldnames=[
        'episode', 'step', 'phase',
        'torso_yaw', 'pelvis_yaw', 'twist', 'cmd_yaw_rate',
        'rfoot_x', 'rfoot_y', 'lfoot_x', 'lfoot_y',
    ])
    per_step_w.writeheader()

    summary_f = open(args.summary_out, 'w', newline='')
    summary_w = csv.DictWriter(summary_f, fieldnames=[
        'episode', 'steps', 'terminated_early',
        'wtr', 'twist_mean', 'twist_range',
        'pelvis_yaw_range', 'torso_yaw_range', 'foot_path_arc',
    ])
    summary_w.writeheader()

    print(f"Running {args.episodes} episodes (max {args.max_steps} steps each)")
    summaries = []
    for ep in range(args.episodes):
        s = run_episode(policy, vec_env, ep, body_ids, per_step_w, max_steps=args.max_steps)
        summaries.append(s)
        summary_w.writerow(s)
        print(f"  Ep{ep:2d}: steps={s['steps']:3d}  WTR={s['wtr']:.4f}  "
              f"pelvis_range={s['pelvis_yaw_range']:.3f}  torso_range={s['torso_yaw_range']:.3f}  "
              f"foot_arc={s['foot_path_arc']:.2f}")

    per_step_f.close()
    summary_f.close()

    valid = [s for s in summaries if not np.isnan(s['wtr'])]
    if valid:
        wtrs   = np.array([s['wtr'] for s in valid])
        pelvis = np.array([s['pelvis_yaw_range'] for s in valid])
        torso  = np.array([s['torso_yaw_range']  for s in valid])
        print("\n" + "=" * 60)
        print(f"Summary across {len(valid)}/{len(summaries)} valid episodes:")
        print(f"  WTR (waist-twist ratio):  mean={wtrs.mean():.4f}  std={wtrs.std():.4f}  "
              f"min={wtrs.min():.4f}  max={wtrs.max():.4f}")
        print(f"  Pelvis yaw range (rad):   mean={pelvis.mean():.3f}  std={pelvis.std():.3f}")
        print(f"  Torso  yaw range (rad):   mean={torso.mean():.3f}  std={torso.std():.3f}")
        if pelvis.mean() > 1e-6:
            print(f"  Torso/Pelvis range ratio: {torso.mean() / pelvis.mean():.2f}  "
                  f"(>1 hints torso turns more than pelvis -> twist hack)")
        print("=" * 60)

    print(f"\nWrote: {args.csv_out}")
    print(f"Wrote: {args.summary_out}")


if __name__ == '__main__':
    main()
