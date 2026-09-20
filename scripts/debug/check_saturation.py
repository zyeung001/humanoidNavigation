#!/usr/bin/env python3
"""How much of the policy is being thrown away by the residual clamp.

THE METRIC THAT PREDICTED FAILURE. Four standing policies passed gate_standing_kick 6/6
and fell on hardware within seconds. The gate measures whether the CLIPPED policy stands;
it cannot see that some joints stopped being controllers. This can.

np.clip is flat, so beyond the clamp box the gradient with respect to the action is exactly
zero. A mean that drifts out gets nothing pulling it back, and with log_std clamped at -2.0
(sigma ~0.135 rad) every sample lands outside too -- the joint freezes wherever it drifted
and contributes no feedback for the rest of training. Measured on final_real_standing_stand
(9/5): 7 of 17 joints outside on over half of all frames, waist_yaw on 100% at -34 deg
against an 11.5 deg clamp, and the raw stance it asks for COLLAPSES when posed statically
(tilt 0.085, 25% of weight on the feet). The clip was the only thing standing.

READ `mean excursion^2` FIRST, not the pinned count. `% outside` is BINARY, so a joint 6 deg
past an 11.5 deg edge still reads 100% and the count can RISE while the problem shrinks.
That is exactly what happened on 9/5: 7 joints pinned became 8, while excursion^2 went
0.360 -> 0.159 and waist_yaw came in from -34 to -17 deg. Judging that run by the count
would have discarded a fix that was working.

Whether a pinned joint is still learning depends on residual_saturation_penalty. At 0 the
clip is the only thing acting outside the box and the gradient there is exactly zero. With
the penalty on it supplies that gradient, and 100% outside means "still converging", not
"dead". `wants` is also an independent read on the balance point, when it is small enough
to be physical.

  python scripts/debug/check_saturation.py --model models/final_real_standing_stand.zip \
      --vecnorm models/vecnorm_real_standing_stand.pkl --config config/real_humanoid_stand.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
from src.environments import make_standing_env  # noqa: E402

MAP = PROJ / "config" / "joint_servo_map.yaml"


def rollout(model, venv, episodes, steps):
    acts = []
    for _ in range(episodes):
        obs = venv.reset()
        for _ in range(steps):
            a, _ = model.predict(obs, deterministic=True)
            acts.append(np.asarray(a, dtype=float).ravel())
            obs, _, done, _ = venv.step(a)
            if done[0]:
                break
    return np.array(acts)


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    p = argparse.ArgumentParser(description="Residual-clamp saturation of a standing policy")
    p.add_argument("--model", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()

    cfg = (yaml.safe_load(open(PROJ / args.config)).get("standing") or {}).copy()
    # Nominal plant and clean obs: the question is where the POLICY puts its mean, not how it
    # copes with noise. Randomisation here would smear the very distribution being measured.
    cfg.update(obs_noise=False, domain_rand=False, actuator_rand=False, push_enabled=False,
               max_episode_steps=args.steps + 50)
    clamp = np.asarray(cfg.get("residual_clamp", 0.0), dtype=float)
    if not np.any(clamp > 0.0):
        raise SystemExit(f"{args.config} has no residual_clamp; nothing to saturate.")
    base = np.asarray(cfg["residual_baseline"], dtype=float)

    env = make_standing_env(config=cfg)
    venv = VecNormalize.load(str(PROJ / args.vecnorm), DummyVecEnv([lambda: env]))
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(str(PROJ / args.model), device="cpu")

    A = rollout(model, venv, args.episodes, args.steps)
    dofs = [j["dof"] for j in yaml.safe_load(open(MAP))["joints"]]
    lo, hi = base - clamp, base + clamp
    frac = ((A < lo - 1e-9) | (A > hi + 1e-9)).mean(axis=0)
    want = np.median(A, axis=0) - base
    excursion = np.maximum(np.abs(A - base) - clamp, 0.0)

    print(f"\n{A.shape[0]} frames over {args.episodes} episodes, "
          f"clamp +-{np.degrees(np.mean(clamp)):.1f} deg\n")
    print(f"{'joint':18s} {'% outside':>10s} {'wants':>9s}")
    print("-" * 42)
    for i in np.argsort(-frac):
        flag = ("  <-- PINNED, not learning" if frac[i] > 0.5 else
                "  <-- pressing" if frac[i] > 0.1 else "")
        print(f"{dofs[i]:18s} {100 * frac[i]:9.1f}% {np.degrees(want[i]):+8.1f}d{flag}")

    pinned = int((frac > 0.5).sum())
    exc2 = float(np.square(excursion).sum(axis=1).mean())
    sat_w = float(cfg.get("residual_saturation_penalty", 0.0))
    sat_cap = float(cfg.get("residual_saturation_cap", 50.0))
    print(f"\n  PRIMARY: mean excursion^2 = {exc2:.3f} rad^2      "
          "(0.000 = fully inside; 9/5 pre-penalty baseline 0.360)")
    print(f"  {100 * frac.mean():.1f}% of joint-frames clipped; "
          f"{pinned} of {len(frac)} joints pinned over half the time   "
          "<- binary, tiebreak only")
    if pinned and sat_w <= 0.0:
        print("\n  residual_saturation_penalty is OFF, so those joints have ZERO gradient and\n"
              "  are no longer controllers. A gate pass says nothing about them: it measures\n"
              "  the clipped policy, which is a different controller from the one being trained.")
    elif pinned:
        print(f"\n  residual_saturation_penalty is {sat_w:g}, charging "
              f"{min(sat_w * exc2, sat_cap):.2f} per step. That term supplies the gradient the\n"
              "  clip destroys, so these joints are outside but being PULLED IN, not dead.\n"
              "  Compare excursion^2 against the previous run: if it has barely moved, raise\n"
              "  the weight; if it is falling, the fix is working and needs more steps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
