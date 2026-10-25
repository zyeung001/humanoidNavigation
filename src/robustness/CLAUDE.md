# src/robustness/ — What Can Happen to the Robot, and How Well a Policy Copes

One definition of "scenario" (start state, disturbances, sensor faults, plant changes) used by
BOTH training (a weighted mix of wide scenarios) and evaluation (a fixed suite at fixed seeds).
Standing only.

| File | What |
|------|------|
| `scenarios.py` | `Scenario` = `Start` + `Disturb` + `Sensors` + `Plant`. `SUITE` (19 fixed-severity eval scenarios, `SUITE_VERSION`), `TRAINING` (wide ranges, for mixing) |
| `wrapper.py` | `ScenarioWrapper(gym.Wrapper)`: plays one scenario per episode on a `StandingEnv`. Training passes `mix`; eval passes `reset(options={"scenario": s, "stratum": (i, n)})` |
| `policies.py` | `SB3Policy` (.zip + .pkl, or live model), `NpzPolicy` (runs `scripts/deploy/numpy_policy.py`, the robot's code), `HoldPosePolicy` (a fixed pose, no feedback) |
| `suite.py` | `evaluate(policy, cfg, ...) -> SuiteResult`; `format_result()` prints the table |

## The score

`score` = mean over scenarios of the average % of the 15 s test spent upright. 100 = never fell.
Fixed seeds: same model -> same score. Paired: episode *i* of every scenario and every model
shares its seed, pushes and directions (`episode_seed(seed, i)`, per-aspect RNG streams in the
wrapper, stratified directions), so `push_5N_gyro - push_5N` is the gyro error's cost alone and
model A vs model B is compared episode by episode (`scripts/compare_models.py`).

## Reference results (10/16, standing-v2, 12 episodes)

| | score |
|---|---|
| hold-straight (all-zero pose, no feedback; also what an UNTRAINED net outputs) | 68.6 |
| hold-baseline (the 9/3 hardware standing pose, no feedback) | 60.8 |
| stand-pushes-215M | 16.0 (falls in ~2 s, as it did on the robot) |
| stand-195M | 14.1 |

**standing-v1 (before 10/16) is void**: the MJCF lacked the R hip-yaw-link/pelvis contact
exclude, so sim R_hip_roll was held at -0.03 rad by a stop the robot does not have, and every
policy braced against it (they fell sideways on the robot within 1.5 s).
**A policy that does not beat hold-straight is not balancing the robot.**

## Rules

- Changing `SUITE` in any way -> bump `SUITE_VERSION`. Comparisons refuse to mix versions.
- Sensor/plant faults are tested UNDER `push_5N`: alone, every controller scored 100% (the robot
  stands passively in sim), so they measured nothing.
- The wrapper restores nominal mass/friction/gains and env settings before EVERY reset (the env
  itself never puts randomised arrays back when randomisation is switched off).
- With no mix and no scenario the wrapper is a strict no-op (tested bit-for-bit).
- Eval workers (`workers=N`) give identical results to `workers=1`.
- `StandingEnv` hooks used here: `place_robot()`, `set_sensor_offsets()`, `episode_baseline`,
  `actuator_lag_scale`, `debug_print_every`.
