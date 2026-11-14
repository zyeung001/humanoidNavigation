# Where the standing limit is (11/11/2026, flex plant, standing-v2)

Every controller tried on the flex plant scored within about 1 point of the knee law
(knees += 0.5 pg_x + 0.8 gyro_y, gyro deadband 0.05). That covers PPO v1–v3, LCP smoothness and
an 18-point gain grid of the law. This page records why, and the one change that moves it.
`scripts/ceiling.py` reproduces every number here; the tools are in `src/robustness/oracle.py`.

All results are paired against the law alone on the same seeds, as % of the test upright.

## 1. The best linear controller cannot beat the law

A full-state discrete LQR on sim truth: base velocity, the servo filter, the dead-time buffer
and the flex joints. The whole env step was linearised by central differences, and the model
matched the true pitch response to a kick to 3 significant figures. A 9-point Q/R grid was run
on 12345 × 12, and the best one was then scored held-out:

| | vs law |
|---|---|
| training seeds 12345 × 12 | +0.2 ± 0.5 |
| held-out 777 × 24 | **−2.4 ± 0.7** |
| held-out 778 × 48 | **−1.9 ± 0.5** |

It loses on start (90 vs 97) and kick (40 vs 42). Aggressive gains are far worse (−11 to −29):
large starting errors drive them into the ±0.2 rad residual clamp.

## 2. Suite pushes of 4–6 N cannot be survived standing in place

The robot is 2.05 kg with its COM 0.417 m up, so ω₀ = √(g/z) = 4.9 rad/s. A 0.2 s push of F
moves the capture point by F·0.2 / (m·ω₀):

| push | 2 N | 3 N | 4 N | 5 N | 6 N |
|---|---|---|---|---|---|
| capture shift | 4.0 cm | 6.0 cm | **8.1 cm** | 10.1 cm | 12.1 cm |

The sole reaches 8 cm fore and aft of the ankle. A sampling MPC planning on the episode's own
sim falls at the same moment as the law on push_5N and kick_3x. It does so even with a 1 s
horizon and the clamp widened to 1.0 rad (it used up to 1.25 rad).

So 11 of the 19 suite scenarios read about 22% for every controller, and the suite score can
move only a few points.

## 3. In the 2–3 N band the falls are all fore/aft, and the robot has no ankles

On one torso push (777 × 12 per force), every fall is a push within about 20° of the ±x axis.
Sideways pushes survive even at 3 N. In falls, the passive `ankle_flex` joint (the fitted servo
play, 7 N·m/rad per foot) bends 0.17–0.25 rad; in survivors it bends ≤ 0.13 rad.

The legs are hip roll/yaw/pitch + knee. **There is no ankle servo**, so the ankle strategy,
the usual answer to small fore/aft pushes, is not available.

Stiffening that play spring is not the fix. The law gets worse with it: 2× −7.8, 4× −18,
10× −37, and even `quiet` falls.

## 4. An oracle finds a little room inside the clamp, at great cost

Falls in the single-push band (777 × 12 per force; the law falls 0 / 3 / 4 / 6 at
1.5 / 2 / 2.5 / 3 N):

| controller | falls |
|---|---|
| law | 13 |
| MPC, N 32, H 0.4 s (0.26 s per step) | 12 |
| MPC, N 64, H 1 s | 10 |
| MPC, N 256, H 1 s (about 3 s per step), on the law's falls only | about 8: saves 3/3 at 2 N, 0/4 at 2.5 N, 2/6 at 3 N |

The strong oracle saves the same episodes with clamp 0.2 and 1.0, so the room lies inside the
deployable box. It is nonlinear and anticipatory, which is why LQR could not reach it. It is
also small (about 5 of 48 band episodes) and costly to distil into a policy.

## 5. Adding ankle-pitch servos is the lever

`AnkleServoPolicy` models an added SCS15 ankle-pitch servo. The servo drives the rest angle of
the existing play spring through 50 ms of dead time and a 40 ms lag. A 2-gain ankle law runs on
top of the unchanged knee law:

    ankle_pitch = 0.3 * pg_x + 0.18 * gyro_y      (the same noisy IMU signals the law reads)

The gains were picked on the single-push band (777 × 12); falls 13 → 8, 2 N falls 3 → 0. They
were then confirmed on the full suite:

| series spring | 777 × 24 | **778 × 48 (fresh)** |
|---|---|---|
| k 7: the play only (optimistic) | +1.8 ± 0.7 | **+2.4 ± 0.4** (won/lost 433/93) |
| k 4.4: play in series with a kp-12 servo (conservative) | +1.7 ± 0.6 | **+1.7 ± 0.3** (400/100) |
| k 4.4, servo held at 0 (control) | −6.1 ± 1.1 | |

- **Where the gain is:** start 96 → 100 and kick 42.5 → 45.5–48. The 4–6 N rows stay about 22%,
  as physics says. This is the first change in the project that beats the law with the whole
  interval above 0.
- **It needs the law:** an unused servo only adds compliance (the −6.1 control).
- **It is latency-sensitive:** 75 ms of dead time gives +1.1 ± 0.3 (778 × 48, k 4.4).
- **Not modelled:** the added foot mass (about 2 × 56 g) and a second servo's backlash beyond the
  conservative spring.

**Conclusion:** on the current robot, standing in place is at the plant's limit for any
controller that could run on it. The next gain is ankle-pitch actuation; the HumanoidV4 design
already has it. The policy stack carries over as it is: `feedback_prior` takes any joint in
`config/joint_servo_map.yaml`, so the ankle law is one more config line once the servo is in the
map.

## 6. Without new hardware: stepping and sideways pushes (11/11–13)

Ankle servos are out (the robot stays as built), so the question was whether RL could find room by
moving the feet, or by reacting harder to the pushes the law falls on. These were one-off sim checks
(scratch scripts, not in this PR), every result replayed in the real env:

| push (one, 0.2 s, upper torso) | knee law | trajectory search, push KNOWN | closed-loop MPC, push only FELT |
|---|---|---|---|
| 3 N forward | falls | survives with a 2-step recovery (+16 cm, +32 cm), but needs 1.7 rad hip pitch: impossible inside the ±0.2 clamp | falls (clamp 1.5) |
| 3 N back, 4–5 N forward/back | falls | none found (clamp 1.5; also with the EMA smoothing removed) | falls |
| **5 N sideways** | **falls** | survives in place inside ±0.2 (hip roll +0.8, waist pitch +0.6 in 0.15 s) | **2 of 12 saved** (±0–30° off sideways, randomised plants) |
| 6 N sideways, 4 N diagonal | falls | none found | falls |

The search knew the push in advance; the MPC, like a policy, only feels it. Feedback alone saves
2/12 sideways 5 N pushes and delays the other falls by 0.1–0.35 s. Training PPO on 4–6 N pushes
was prepared (`flex_plant_v4_hardpush`) and NOT run: there is no room worth that run on this plant.
