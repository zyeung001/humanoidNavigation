# Turning Fix — Review Doc

**Purpose:** Have a second model sanity-check the changes below *before* the 2×20M-step retrain is launched. Focus areas flagged at the bottom.

---

## 1. Context

**Goal (paper Contribution 2):** validate a heading-source fix on the real robot in sim. Two arms, trained identically except for one config flag:

- **HACK** (`heading_source: torso`) — reward yaw measured at the chest (`0003_8`). The policy is *allowed* to satisfy a turn command by twisting the waist while the feet stay planted.
- **FIX** (`heading_source: pelvis`) — reward yaw measured at the freejoint root / pelvis (`base_link`). Twisting the waist earns nothing; the policy must rotate the whole lower body.

The deliverable is an open-loop turn replay on hardware showing hack→waist-twist vs fix→real turn. WTR (waist-twist ratio) = `std(torso_yaw − pelvis_yaw)`.

**Substrate:** `humanoid_real_v2.xml`, no-ankle morphology (hip roll/yaw/pitch + knee), position actuators, 228-dim proprioceptive obs + 11-dim command block = **239-dim obs**. Warm-started from a working standing model expanded 228→239 (`models/turn_warmstart.zip` / `.pkl`, parity verified to 2e-7).

**Morphology inversion (critical):** unlike Humanoid-v5 (root = torso), the real robot's freejoint root is the **pelvis**. `HeadingYaw(..., root_is_torso=False)` handles this: `qvel[5]` = pelvis yaw, torso yaw via `mj_objectVelocity('0003_8')`.

---

## 2. What went wrong in the first run

Both arms trained to 20M and were evaluated with `measure_waist_twist_real.py`. Result:

- **WTR was inverted:** FIX twisted *more* than HACK (0.127 vs 0.081) — the opposite of the hypothesis.
- A direct probe of `yaw_actual` vs `yaw_cmd` showed the real reason: **neither arm turned at all.** Both achieved <5% of commanded yaw. They just stood still and balanced. The "WTR" was measuring balance micro-jitter, not a turn.

**Root cause — the standing exploit.** The old turning reward paid a large, *unconditional* floor for staying upright at height (upright + height ≈ 6/step) plus a yaw term. Standing perfectly still banked ~6.8/step for free, and any attempt to turn risked a fall (negative). So PPO correctly converged to "never turn." Three things the working Humanoid-v5 turn recipe (`walking_env` + `config/variants/03_yaw`, `04_omni_sustained`) has that the real turning env was **missing**:

1. **No feet-air-time reward.** A planted-foot robot physically *cannot* rotate in place — foot friction locks yaw. Without a term rewarding foot-lift, the policy never discovers the pivot.
2. **Survival/floor was not turn-gated.** The upright+height floor was paid in full regardless of whether the agent turned → stand-still is a global optimum.
3. **No wrong-direction penalty.** The pure-Gaussian yaw reward `exp(-b·err²)` is essentially flat once `|err| > 0.5` rad/s, so it can't distinguish "turning the wrong way" from "not turning." No gradient out of the stand-still basin.

*(Aside: the FIX config header still carries an old note about a "chicken-and-egg" need for a step-turner warm-start. That concern is what the three terms below are meant to resolve without a separate walker. Header comment is stale, not load-bearing.)*

---

## 3. What I changed

### 3a. `src/environments/real_turning_env.py` — `_compute_task_reward`

Ported the three v5 turn-enablers. Full current reward assembly (lines ~132–204):

```python
yaw_actual = self._get_actual_yaw_rate()
yaw_err = yaw_actual - self.commanded_yaw_rate
yaw_cmd = self.commanded_yaw_rate

# (1) turn-gated survival factor — kills the stand-still exploit
if abs(yaw_cmd) > 1e-3:
    direction_ratio = float(np.clip((yaw_actual * np.sign(yaw_cmd)) / abs(yaw_cmd), 0.0, 1.0))
    survival_factor = self.turn_survival_floor + (1.0 - self.turn_survival_floor) * direction_ratio
else:
    survival_factor = 1.0   # no turn commanded -> standing is correct, full floor

upright_reward = self.reward_upright_weight * np.exp(-8.0 * (1.0 - upright_cos) ** 2)
height_reward  = self.reward_height_weight * np.exp(-10.0 * height_err ** 2)
floor_reward   = (upright_reward + height_reward) * survival_factor   # now gated

yaw_reward = 0.0
if standing_up:                                   # height>=1.2 and upright_cos>0.6
    yaw_reward = self.reward_yaw_weight * np.exp(-self.reward_yaw_bandwidth * yaw_err**2) * survival_factor

# (3) wrong-direction penalty — gradient past err=0.5
wrong_dir_pen = 0.0
if abs(yaw_cmd) > 1e-3 and (yaw_actual * np.sign(yaw_cmd)) < 0.0:
    wrong_dir_pen = -self.yaw_wrong_dir_penalty * min(abs(yaw_actual), abs(yaw_cmd))

# (2) feet air-time — reward foot-lift so the policy can pivot
feet_air_reward = 0.0
if self.feet_air_time_weight > 0:
    dt = float(self.env.unwrapped.dt)
    for i, body_id in enumerate(self.foot_body_ids):
        contact_force = float(np.linalg.norm(data.cfrc_ext[body_id]))
        if contact_force > self.contact_force_threshold:
            self.feet_air_time[i] = 0.0
        else:
            self.feet_air_time[i] += dt
            if self.feet_air_time[i] > self.min_air_time:
                air_bonus = min(self.feet_air_time[i] - self.min_air_time, 0.3) / 0.3
                feet_air_reward += self.feet_air_time_weight * air_bonus

tilt_rate_pen = -self.tilt_rate_penalty * float(ang[0]**2 + ang[1]**2)  # roll/pitch only, NOT z-yaw
control_cost  = -0.005 * float(np.sum(np.square(action)))
rate_pen      = -action_rate_penalty * sum(last_action_rate**2)  # if configured

reward = floor_reward + yaw_reward + feet_air_reward + wrong_dir_pen + tilt_rate_pen + control_cost + rate_pen
```

**`__init__` additions** (all config-gated with defaults):
```python
self.feet_air_time_weight     = cfg.get('feet_air_time_weight', 1.0)
self.min_air_time             = cfg.get('min_air_time', 0.15)
self.contact_force_threshold  = cfg.get('contact_force_threshold', 5.0)
self.foot_body_ids            = self.model_spec.foot_body_ids if self._is_custom else [6, 9]
self.feet_air_time            = np.zeros(len(self.foot_body_ids))
self.turn_survival_floor      = cfg.get('turn_survival_floor', 0.3)
self.yaw_wrong_dir_penalty    = cfg.get('yaw_wrong_dir_penalty', 1.5)
```
**`reset()`** now zeros `self.feet_air_time[:] = 0.0`.
New info keys for triage: `r_floor`, `r_feet_air`, `r_wrong_dir`, `survival_factor`.

### 3b. `config/turning/heading_torso.yaml` + `heading_pelvis.yaml`

Both got the identical four knobs (values above are also the code defaults, so the configs just make them explicit/tunable):
```yaml
feet_air_time_weight: 1.0
min_air_time: 0.15
turn_survival_floor: 0.3
yaw_wrong_dir_penalty: 1.5
```
Nothing else in either config changed. The two configs still differ only by `heading_source` (torso vs pelvis) and their output paths — that isolation is the whole experiment and must stay intact.

---

## 4. Reward-magnitude check (why this should un-stick the policy)

Per-step, under a commanded turn:

| Behavior | floor | yaw | air | wrong-dir | ≈ total |
|---|---|---|---|---|---|
| Stand still (old env) | 6.0 | ~0.8 | — | — | **~6.8** |
| Stand still (new env) | 1.9 (×0.31) | 0.31 (×0.31) | 0 | 0 | **~2.2** |
| Actually turning | up to 6.0 | up to 10.0 | up to ~2.0 | 0 | **~18** |

Standing still is cut to ~⅓; a real turn is now worth ~8× more. Smoke test (zero action, +0.5 cmd) confirmed `survival_factor≈0.31`, `floor≈1.88`, `yaw≈0.31`, `feet_air=0` — matching the "stand still" row.

---

## 5. Things to double-check (please scrutinize)

1. **`foot_body_ids` on the real model.** Code uses `self.model_spec.foot_body_ids` for the custom body. Confirm those IDs point at the actual foot bodies in `humanoid_real_v2.xml`, and that `data.cfrc_ext[body_id]` is nonzero when that foot is on the ground. If the IDs are wrong or feet never register contact, `feet_air_time` accumulates forever → the air bonus becomes a *constant* the policy earns by doing nothing (a NEW exploit). **This is the highest-risk item.** Worth a 1-step print of `cfrc_ext` norm per foot at reset (feet planted → should exceed the 5.0 threshold).

2. **`contact_force_threshold = 5.0`.** Is 5 N the right cutoff for this body's mass (~2.05 kg)? On the standing pose each foot carries ~10 N, so 5 N should read as "planted" — but verify against the actual `cfrc_ext` magnitude, which is a 6-vector (force+torque) norm, not pure vertical force. If the norm sits below 5 when planted, feet read as always-airborne (see risk #1).

3. **Double smoothing / `rate_pen` term.** `action_rate_penalty: 25.0` is large. Confirm `self._last_action_rate` is populated by the standing substrate at the point `_compute_task_reward` runs, and that this isn't double-counting the action-smoothing already applied upstream. A too-strong rate penalty could itself suppress the foot-lift the air term is trying to reward — the two terms pull against each other.

4. **`standing_up` gate on yaw.** Yaw reward requires `height>=1.2 and upright_cos>0.6`. During an aggressive pivot the robot may dip briefly below 1.2 m, zeroing the yaw reward exactly when it's turning. Is the threshold too tight for a turn-in-place on a no-ankle body? Consider whether it should be relaxed or whether `survival_factor` alone is enough gating.

5. **`survival_factor` sign robustness.** `direction_ratio` uses `yaw_actual * sign(yaw_cmd) / |yaw_cmd|` clipped to [0,1]. Confirm `_get_actual_yaw_rate()` sign convention matches the command sign convention (both CCW-positive about world z). If they're flipped, a *correct* turn would read as wrong-direction — penalized and floor-gated to 0.3 — and the policy would be trained to turn the wrong way. **Verify the sign once by hand** (e.g. set `fixed_command=(0,0,+0.5)`, apply a known CCW rotation, check `yaw_actual > 0`).

6. **Warm-start validity.** `turn_warmstart.zip` is unchanged (obs still 239; only the reward changed, which is env-side). Both arms should retrain from it, **not** from the failed `final_real_turn_*` checkpoints (those weights are in the stand-still basin). Confirm the train script loads policy + vecnorm and that `num_timesteps` resets to 0 → `--timesteps 20000000` = a full fresh 20M.

7. **Both arms use the shared reward.** The three new terms are heading-source-agnostic (they act on `yaw_actual` from whichever source is configured). That's intended — the experiment isolates *heading source*, everything else identical. Confirm nothing in the port accidentally hard-codes torso or pelsvis.

---

## 6. Retrain commands (USER runs these — not yet launched)

```bash
# HACK (torso heading -> expect waist twist)
python scripts/train_standing.py --config config/turning/heading_torso.yaml \
  --model models/turn_warmstart.zip --vecnorm models/turn_warmstart.pkl --timesteps 20000000

# FIX (pelvis heading -> must turn lower body)
python scripts/train_standing.py --config config/turning/heading_pelvis.yaml \
  --model models/turn_warmstart.zip --vecnorm models/turn_warmstart.pkl --timesteps 20000000
```

**Early success signal (~1–2M steps):** `survival_factor` and `r_feet_air` climb above the standing baseline, and `yaw_actual` starts tracking `yaw_cmd`. If `yaw_actual` stays ~0 while `r_feet_air` climbs, suspect risk #1 (a new air-time exploit). After training, re-run `measure_waist_twist_real.py` — only now should the hack-vs-fix WTR contrast be meaningful.
