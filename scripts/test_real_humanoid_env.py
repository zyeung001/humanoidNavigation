"""End-to-end smoke test for the real-humanoid env pipeline.

Constructs StandingCurriculumEnv with models/humanoid_real.xml, runs a short
random-action rollout, and reports introspected constants + observed obs dim,
reward range, and termination behavior. Use this to verify the pipeline boots
before kicking off a real training run.

Run: py -3.13 scripts/test_real_humanoid_env.py
"""
import os
import sys

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import yaml

from src.environments.model_spec import introspect_model, summarize
from src.environments.standing_curriculum import make_standing_curriculum_env


def main():
    cfg_path = os.path.join(PROJ, 'config', 'real_humanoid_config.yaml')
    cfg = yaml.safe_load(open(cfg_path))['standing']

    print("=" * 60)
    print("Model introspection")
    print("=" * 60)
    spec = introspect_model(cfg['xml_file'])
    print(summarize(spec))
    print()

    print("=" * 60)
    print("Constructing StandingCurriculumEnv")
    print("=" * 60)
    env = make_standing_curriculum_env(render_mode=None, config=cfg)
    print()

    print("=" * 60)
    print("Rolling out 200 random-action steps")
    print("=" * 60)
    obs, info = env.reset(seed=0)
    print(f"obs shape: {obs.shape}  dtype: {obs.dtype}")

    rewards = []
    heights = []
    n_terminations = 0
    n_resets = 1
    step = 0
    max_steps = 200
    while step < max_steps:
        a = np.random.uniform(-0.3, 0.3, size=env.action_space.shape).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        rewards.append(float(r))
        heights.append(float(info.get('height', 0.0)))
        step += 1
        if term or trunc:
            n_terminations += int(term)
            env.reset(seed=step)
            n_resets += 1

    rewards = np.array(rewards)
    heights = np.array(heights)
    print(f"steps={step}  resets={n_resets}  terminations={n_terminations}")
    print(f"reward: mean={rewards.mean():+.2f}  min={rewards.min():+.2f}  max={rewards.max():+.2f}")
    print(f"scaled height: mean={heights.mean():.3f}  min={heights.min():.3f}  max={heights.max():.3f}")
    print(f"  (real COM-z range: {heights.min() * spec.standing_com_z / 1.40:.3f}m"
          f" .. {heights.max() * spec.standing_com_z / 1.40:.3f}m)")
    env.close()

    print()
    print("=" * 60)
    print("OK — pipeline boots end-to-end. Ready to train with:")
    print("  py -3.13 scripts/train_standing.py --config config/real_humanoid_config.yaml")
    print("=" * 60)


if __name__ == '__main__':
    main()
