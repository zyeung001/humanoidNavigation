"""What can happen to a standing robot, written down once.

A Scenario is one kind of episode: where the robot starts, what pushes it, how its sensors
lie and how its plant differs from the model. The SAME definitions drive both sides:

  * training samples from a weighted MIX of wide scenarios (config `training_scenarios`),
    so the policy meets the starts, pushes and sensor faults the robot meets;
  * evaluation runs a fixed SUITE of narrow scenarios at fixed seeds, so two models are
    scored on identical situations and the number means the same thing every time.

Keeping one definition is the point. When training randomisation and test conditions are
written separately they drift, and a model ends up graded on a world it never trained for
(or, worse, only on the one it did).

Every numeric field is a Range (lo, hi) sampled uniformly once per episode, or per event for
disturbances. An eval scenario pins lo == hi; a training scenario spreads them. In YAML a
scalar means (v, v).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Iterable, List, Optional, Tuple

Range = Tuple[float, float]

# Bump whenever SUITE changes in any way. Results carry it, and comparisons refuse to mix
# versions -- a score is only comparable with scores from the same suite.
SUITE_VERSION = "standing-v1"


def as_range(v) -> Range:
    if isinstance(v, (int, float)):
        return (float(v), float(v))
    lo, hi = v
    if float(lo) > float(hi):
        raise ValueError(f"range {v}: lo > hi")
    return (float(lo), float(hi))


@dataclass(frozen=True)
class Start:
    """Where the episode starts, applied on top of the env's own reset."""
    at_baseline: bool = False      # joints start ON the residual baseline, as after deploy's ramp
    ema_from_zero: bool = False    # ...but the command smoothing starts at 0, as deploy does today
    tilt_deg: Range = (0.0, 0.0)   # released leaning, about the feet, in a random direction
    ang_vel: Range = (0.0, 0.0)    # rad/s, random horizontal axis: released with a wobble
    lin_vel: Range = (0.0, 0.0)    # m/s, random horizontal direction
    joint_offset: Range = (0.0, 0.0)  # rad, each joint independently +-: not quite on the pose

    def is_noop(self) -> bool:
        return not (self.at_baseline or any(r[1] > 0 for r in
                    (self.tilt_deg, self.ang_vel, self.lin_vel, self.joint_offset)))


@dataclass(frozen=True)
class Disturb:
    """Repeated disturbances during the episode."""
    kind: str = "none"             # "none" | "push" (force on a body) | "kick" (base velocity step)
    magnitude: Range = (0.0, 0.0)  # push: newtons, horizontal; kick: m/s, horizontal
    duration_s: Range = (0.2, 0.2)  # push only: how long the hand stays on
    yaw_rate: Range = (0.0, 0.0)   # kick only: rad/s added about vertical, alternating sign
    first_s: Range = (2.0, 3.0)    # when the first one lands
    interval_s: Range = (3.0, 4.0)  # time between the following ones
    body: str = "torso"            # push only: "torso" (upper body) or "pelvis", or a body name

    def __post_init__(self):
        if self.kind not in ("none", "push", "kick"):
            raise ValueError(f"Disturb.kind must be none/push/kick, not {self.kind!r}")


@dataclass(frozen=True)
class Sensors:
    """Constant sensor errors for the episode."""
    gyro_bias: Range = (0.0, 0.0)     # rad/s, random direction in the roll/pitch plane
    imu_tilt_deg: Range = (0.0, 0.0)  # IMU mounted crooked, random axis
    encoder_bias: Range = (0.0, 0.0)  # rad, each joint independently +-
    noise: Optional[bool] = None      # force obs_noise on/off; None = as configured


@dataclass(frozen=True)
class Plant:
    """How the simulated robot differs from the MJCF. None = as configured."""
    mass_scale: Optional[Range] = None      # per body, uniform; turns domain_rand on
    friction_scale: Optional[Range] = None  # per geom, uniform; turns domain_rand on
    servo_gain: Optional[Range] = None      # per servo kp/kv scale; turns actuator_rand on
    servo_lag: Range = (1.0, 1.0)           # multiplies every servo's dead time and time constant


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str          # groups scenarios for subscores and for adaptive training weights
    about: str           # one line, printed in tables -- say what happens, not why
    start: Start = field(default_factory=Start)
    disturb: Disturb = field(default_factory=Disturb)
    sensors: Sensors = field(default_factory=Sensors)
    plant: Plant = field(default_factory=Plant)
    # The env's own random velocity kicks (push_enabled) and per-episode randomisation of the
    # baseline and servo profiles. None = leave as configured (training). The eval suite turns
    # them off so that ONLY what the scenario names is happening.
    env_pushes: Optional[bool] = None
    env_randomisation: Optional[bool] = None

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Scenario":
        """Build from YAML. Sub-blocks use the dataclass field names; ranges may be scalars."""
        def sub(kind, raw):
            raw = dict(raw or {})
            known = {f.name: f for f in fields(kind)}
            unknown = set(raw) - set(known)
            if unknown:
                raise ValueError(f"scenario {name!r}: unknown {kind.__name__} keys {sorted(unknown)}")
            for k, v in list(raw.items()):
                ann = str(known[k].type)
                if "Range" in ann and v is not None:
                    raw[k] = as_range(v)
            return kind(**raw)
        top = {"family", "about", "start", "disturb", "sensors", "plant", "env_pushes",
               "env_randomisation"}
        unknown = set(d) - top
        if unknown:
            raise ValueError(f"scenario {name!r}: unknown keys {sorted(unknown)}")
        return cls(
            name=name,
            family=d.get("family", "custom"),
            about=d.get("about", ""),
            start=sub(Start, d.get("start")),
            disturb=sub(Disturb, d.get("disturb")),
            sensors=sub(Sensors, d.get("sensors")),
            plant=sub(Plant, d.get("plant")),
            env_pushes=d.get("env_pushes"),
            env_randomisation=d.get("env_randomisation"),
        )


def _eval(name, family, about, **kw) -> Scenario:
    """An evaluation scenario: only what it names happens -- no env kicks, no randomisation."""
    kw.setdefault("env_pushes", False)
    kw.setdefault("env_randomisation", False)
    kw.setdefault("sensors", Sensors(noise=False))
    return Scenario(name=name, family=family, about=about, **kw)


def _push(newtons, every=(3.0, 4.0)):
    return Disturb(kind="push", magnitude=(newtons, newtons), duration_s=(0.2, 0.2),
                   first_s=(2.0, 2.5), interval_s=every, body="torso")


def _kick(mult, lin0=0.15, yaw0=0.5):
    # Multiples of the kick the policy trains on (push_lin_vel/push_ang_vel), the same
    # disturbance eval_standing.py's sweep used, now repeated through the episode.
    return Disturb(kind="kick", magnitude=(mult * lin0,) * 2, yaw_rate=(mult * yaw0,) * 2,
                   first_s=(2.0, 2.5), interval_s=(3.0, 4.0))


# ---------------------------------------------------------------------------------------
# THE EVALUATION SUITE. Fixed severities, graded so the score has resolution: a suite every
# model passes measures nothing (the old 6/6 kick gate), and one every model fails measures
# nothing either. Calibrated 10/7 against holding the pose with no policy, stand-175M and
# stand-195M:
#
#   * Sensor and plant faults ALONE did not separate them -- all three scored 100% even with
#     a 0.25 rad/s gyro bias or an 8 deg IMU tilt, because in sim this robot stands
#     passively in that pose. So every fault is tested UNDER the 5 N push, where balance is
#     actually being asked for, and the push_5N row is its control: episode i of every
#     scenario meets the same pushes at the same times (see suite.episode_seed), so
#     push_5N_gyro minus push_5N is the gyro error's cost and nothing else.
#   * Levels that every controller fails identically were dropped. Rotating at 1.0 rad/s
#     about the feet throws the COM ~9 cm, past what the feet can catch without a step.
#
# Robot weight is ~20 N, so 5 N on the upper torso for 0.2 s gives the body ~0.5 m/s.
# ---------------------------------------------------------------------------------------
_P5 = _push(5.0)
SUITE: List[Scenario] = [
    _eval("quiet", "quiet", "nominal robot, clean sensors, nothing happens for the whole test"),
    _eval("deploy_start", "start",
          "starts on the baseline pose with command smoothing at 0 -- what deploy_standing does",
          start=Start(at_baseline=True, ema_from_zero=True)),
    _eval("lean_6deg", "start", "let go leaning 6 deg in a random direction",
          start=Start(at_baseline=True, tilt_deg=(6, 6))),
    _eval("lean_8deg", "start", "let go leaning 8 deg in a random direction",
          start=Start(at_baseline=True, tilt_deg=(8, 8))),
    _eval("wobble", "start", "let go rotating 0.6 rad/s about the feet, drifting 0.1 m/s",
          start=Start(at_baseline=True, ang_vel=(0.6, 0.6), lin_vel=(0.1, 0.1))),
    _eval("pose_off", "start", "joints start up to 0.1 rad off the standing pose",
          start=Start(at_baseline=True, joint_offset=(0.1, 0.1))),
    _eval("push_4N", "push", "4 N on the upper torso for 0.2 s, every 3-4 s", disturb=_push(4.0)),
    _eval("push_5N", "push", "5 N on the upper torso for 0.2 s, every 3-4 s", disturb=_P5),
    _eval("push_6N", "push", "6 N on the upper torso for 0.2 s, every 3-4 s", disturb=_push(6.0)),
    _eval("kick_1.5x", "kick", "1.5x the trained velocity kick, every 3-4 s", disturb=_kick(1.5)),
    _eval("kick_2x", "kick", "2x the trained velocity kick, every 3-4 s", disturb=_kick(2.0)),
    _eval("kick_3x", "kick", "3x the trained velocity kick, every 3-4 s", disturb=_kick(3.0)),
    _eval("push_5N_gyro", "sensor", "push_5N, and the gyro reads a false 0.12 rad/s (the 9/26 fault)",
          disturb=_P5, sensors=Sensors(gyro_bias=(0.12, 0.12), noise=False)),
    _eval("push_5N_imu", "sensor", "push_5N, and the IMU reports gravity 4 deg off",
          disturb=_P5, sensors=Sensors(imu_tilt_deg=(4, 4), noise=False)),
    _eval("push_5N_encoders", "sensor", "push_5N, and every joint reads up to 3 deg off",
          disturb=_P5, sensors=Sensors(encoder_bias=(0.052, 0.052), noise=False)),
    _eval("push_5N_noise", "sensor", "push_5N, with the training sensor noise and biases",
          disturb=_P5, sensors=Sensors(noise=True)),
    _eval("push_5N_slow", "plant", "push_5N, every servo 1.5x slower and 25% weaker",
          disturb=_P5, plant=Plant(servo_lag=(1.5, 1.5), servo_gain=(0.75, 0.75))),
    _eval("push_5N_oddbody", "plant", "push_5N, masses +-35% and friction 0.4-1.6 (past training)",
          disturb=_P5, plant=Plant(mass_scale=(0.65, 1.35), friction_scale=(0.4, 1.6))),
    _eval("hardware_like", "combined",
          "deploy start leaning 3 deg + gyro 0.05 + IMU 2 deg + encoders 2 deg + noise + "
          "slower servos + 4 N pushes",
          start=Start(at_baseline=True, ema_from_zero=True, tilt_deg=(3, 3)),
          disturb=_push(4.0, every=(3.0, 5.0)),
          sensors=Sensors(gyro_bias=(0.05, 0.05), imu_tilt_deg=(2, 2), encoder_bias=(0.035, 0.035),
                          noise=True),
          plant=Plant(servo_lag=(1.25, 1.25), servo_gain=(0.85, 0.85))),
]

# ---------------------------------------------------------------------------------------
# TRAINING SCENARIOS: wide ranges, meant to be MIXED (see training_scenarios in config). Each
# one adds its spread ON TOP of the env's configured randomisation, so `as_configured` is
# exactly today's training episode and any mix containing it keeps today's distribution in.
# ---------------------------------------------------------------------------------------
TRAINING: List[Scenario] = [
    Scenario("as_configured", "quiet", "the env exactly as configured; nothing added"),
    Scenario("wide_start", "start", "start anywhere near the pose: leaning, moving, off-pose",
             start=Start(at_baseline=True, tilt_deg=(0, 10), ang_vel=(0, 0.8), lin_vel=(0, 0.2),
                         joint_offset=(0, 0.12))),
    Scenario("deploy_start", "start", "the robot's real start: on the pose, smoothing at 0",
             start=Start(at_baseline=True, ema_from_zero=True, tilt_deg=(0, 5))),
    Scenario("torso_pushes", "push", "0-7 N hand pushes on the upper torso, 0.1-0.4 s",
             disturb=Disturb(kind="push", magnitude=(0, 7), duration_s=(0.1, 0.4),
                             first_s=(1.0, 3.0), interval_s=(2.0, 4.0))),
    Scenario("big_kicks", "kick", "velocity kicks up to 3x the configured size",
             disturb=Disturb(kind="kick", magnitude=(0, 0.45), yaw_rate=(0, 1.5),
                             first_s=(1.0, 3.0), interval_s=(2.0, 4.0))),
    Scenario("sensor_faults", "sensor", "gyro bias to 0.15, IMU tilt to 5 deg, encoders to 4 deg",
             sensors=Sensors(gyro_bias=(0, 0.15), imu_tilt_deg=(0, 5), encoder_bias=(0, 0.07))),
    Scenario("slow_plant", "plant", "servos up to 1.6x slower and 40% weaker",
             plant=Plant(servo_lag=(1.0, 1.6), servo_gain=(0.6, 1.0))),
    Scenario("everything", "combined", "moderate amounts of all of the above at once",
             start=Start(at_baseline=True, ema_from_zero=True, tilt_deg=(0, 6), ang_vel=(0, 0.5),
                         joint_offset=(0, 0.08)),
             disturb=Disturb(kind="push", magnitude=(0, 5), duration_s=(0.1, 0.3),
                             first_s=(1.0, 3.0), interval_s=(2.5, 5.0)),
             sensors=Sensors(gyro_bias=(0, 0.1), imu_tilt_deg=(0, 3), encoder_bias=(0, 0.05)),
             plant=Plant(servo_lag=(1.0, 1.4), servo_gain=(0.75, 1.0))),
]


def _by_name(base: List[Scenario], extra: Optional[Dict[str, dict]]) -> Dict[str, Scenario]:
    out = {s.name: s for s in base}
    for name, d in (extra or {}).items():
        out[name] = Scenario.from_dict(name, d)
    return out


def training_scenarios(extra: Optional[Dict[str, dict]] = None) -> Dict[str, Scenario]:
    """TRAINING by name, plus any defined in the config's `training_scenarios.define`."""
    return _by_name(TRAINING, extra)


def suite_scenarios(names: Optional[Iterable[str]] = None,
                    families: Optional[Iterable[str]] = None) -> List[Scenario]:
    """Suite scenarios by name and/or family, in suite order. Neither given = the whole suite.

    There is deliberately no way to add scenarios here from a config: a score computed on a
    different set of scenarios is a different number, and it would still say SUITE_VERSION.
    """
    if names is None and families is None:
        return list(SUITE)
    reg = _by_name(SUITE, None)
    want, fams = set(names or ()), set(families or ())
    missing = want - set(reg)
    if missing:
        raise KeyError(f"unknown scenario(s) {sorted(missing)}; known: {sorted(reg)}")
    return [s for s in SUITE if s.name in want or s.family in fams]


def _check_unique(xs: List[Scenario], what: str):
    names = [s.name for s in xs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise AssertionError(f"duplicate {what} scenario names: {sorted(dup)}")


_check_unique(SUITE, "suite")
_check_unique(TRAINING, "training")
