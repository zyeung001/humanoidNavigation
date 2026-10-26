"""The showcase: one fixed, seeded perturbation timeline, played on a policy for the video.

The suite answers "how often does it fall"; this answers "what does it DO". It plays the same
30 s on every policy -- a quiet deploy start, torso pushes of 3, 4, 5 and 6 N from the four
sides, a 5 N push on the pelvis, a 2x velocity kick, and a 5 N push while the gyro lies -- so
two runs' videos can be put side by side and compared beat by beat. Severities are the
suite's: the robot weighs ~20 N, so 3-6 N is the range where its response decides the outcome.

Directions are RELATIVE TO THE ROBOT and turned into a world vector when the beat fires:
forward = base_link +X flattened (scripts/debug/sim_cop.py forward_axis; checked on hardware
9/26), left = z x forward. "From behind" means the force points toward the front.

Every beat goes through ScenarioWrapper.inject_push / inject_kick, the code path and physics
the suite's pushes and kicks use. Nothing here draws a random number, so the video is
bit-for-bit reproducible for a given policy and seed.

A FALL does not end the video: it plays 1 s more of pure physics (the fall itself), resets,
settles for 1 s and shifts every remaining beat 2 s later, so each beat still gets its own
window. After `max_resets` falls the next one ends it (worst case ~42 s).

No GL and no cv2 here: rendering is a callback (src/robustness/video.py), so all of this runs
in CI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from .scenarios import Start, _eval
from .suite import CALM_RATE, CALM_STEPS, _lean_deg, _recoveries, make_eval_env

# The deploy start (suite row deploy_start) with nothing scheduled: every beat is injected.
SHOWCASE_START = _eval("showcase", "showcase",
                       "deploy start, then the fixed showcase timeline injected by the driver",
                       start=Start(at_baseline=True, ema_from_zero=True))

_D = float(np.sqrt(0.5))


@dataclass(frozen=True)
class Beat:
    t: float                      # s on the planned timeline (falls shift it later)
    kind: str                     # start | push | kick | fault_on | fault_off
    name: str                     # short, for the end card and the timeline
    newtons: float = 0.0          # push
    body: str = "torso"           # push: torso | pelvis | a body name
    toward: Tuple[float, float] = (0.0, 0.0)   # unit direction it POINTS, in (forward, left)
    duration_s: float = 0.2       # push
    dv: float = 0.0               # kick: m/s
    yaw_rate: float = 0.0         # kick: rad/s
    gyro: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # fault_on: base-frame gyro bias, rad/s

    @property
    def graded(self) -> bool:
        """Beats with an outcome (held / fell). Faults alone do nothing visible in sim."""
        return self.kind in ("start", "push", "kick")


SHOWCASE: List[Beat] = [
    Beat(0.0, "start", "deploy start"),
    Beat(3.0, "push", "3 N front", 3.0, "torso", (-1.0, 0.0)),
    Beat(6.0, "push", "4 N left", 4.0, "torso", (0.0, -1.0)),        # suite push_4N
    Beat(9.0, "push", "5 N behind", 5.0, "torso", (1.0, 0.0)),       # push_5N
    Beat(12.0, "push", "6 N right", 6.0, "torso", (0.0, 1.0)),       # push_6N
    Beat(15.0, "push", "5 N pelvis left", 5.0, "pelvis", (0.0, -1.0)),   # low vs high
    Beat(18.0, "kick", "kick 2x", toward=(_D, _D), dv=0.30, yaw_rate=1.0),   # kick_2x
    Beat(21.0, "fault_on", "gyro fault", gyro=(0.0, 0.12, 0.0)),     # the 9/26 fault
    Beat(23.0, "push", "5 N behind+gyro", 5.0, "torso", (1.0, 0.0)),  # push_5N_gyro
    Beat(27.0, "fault_off", "fault off"),
]
SHOWCASE_SECONDS = 30.0

_SIDES = ["front", "front-left", "left", "back-left", "behind", "back-right", "right", "front-right"]


def robot_forward(model, data) -> np.ndarray:
    """base_link +X flattened into the horizontal plane: the robot's forward."""
    import mujoco  # noqa: PLC0415
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    fwd = data.xmat[bid if bid >= 0 else 1].reshape(3, 3)[:, 0].copy()
    fwd[2] = 0.0
    return fwd / np.linalg.norm(fwd)


def robot_left(fwd) -> np.ndarray:
    return np.cross([0.0, 0.0, 1.0], fwd)


def direction_label(vec, fwd, toward: bool = False) -> str:
    """Robot-relative side a horizontal force comes FROM (a force along +forward comes from
    'behind'), or with toward=True the side it points to. Eight sectors of 45 deg."""
    fwd = np.asarray(fwd, dtype=np.float64)
    x, y = float(np.dot(vec, fwd)), float(np.dot(vec, robot_left(fwd)))
    if not toward:
        x, y = -x, -y
    return _SIDES[int(round(np.degrees(np.arctan2(y, x)) / 45.0)) % 8]


def pelvis_attitude(model, data) -> Tuple[float, float]:
    """Pitch (forward +) and roll (left +) in degrees, from projected gravity, as the robot's
    IMU reports them (scripts/debug/sim_cop.py pelvis_attitude)."""
    import mujoco  # noqa: PLC0415
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pg = data.xmat[bid if bid >= 0 else 1].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    return (float(np.degrees(np.arctan2(pg[0], -pg[2]))),
            float(np.degrees(np.arctan2(pg[1], -pg[2]))))


def describe(beat: Beat, vec=None, fwd=None) -> str:
    """The event label: what is happening, where, how much."""
    if beat.kind == "push":
        side = direction_label(vec, fwd)
        return (f"PUSH {beat.newtons:.1f} N from {side} . {beat.body} . {beat.duration_s:.2f} s . "
                f"{beat.newtons * beat.duration_s:.1f} N*s")
    if beat.kind == "kick":
        return (f"KICK {beat.dv:.2f} m/s {direction_label(vec, fwd, toward=True)}, "
                f"yaw {beat.yaw_rate:+.1f} rad/s")
    if beat.kind == "fault_on":
        return f"FAULT ON: {fault_text(beat.gyro, prefix='')}"
    if beat.kind == "fault_off":
        return "FAULT OFF: the gyro reads true again"
    return "DEPLOY START: on the pose, command smoothing at 0"


def fault_text(gyro, prefix: str = "FAULT ") -> str:
    """'FAULT gyro reads +0.12 rad/s pitch' for a base-frame gyro bias (y = pitch, x = roll)."""
    g = np.asarray(gyro, dtype=float)
    parts = [f"{g[1]:+.2f} rad/s pitch" if g[1] else "", f"{g[0]:+.2f} rad/s roll" if g[0] else ""]
    return f"{prefix}gyro reads " + " ".join(p for p in parts if p)


def _short(beat: Beat, vec, fwd) -> str:
    """'5.0 N push from behind' / 'kick front-left': what a fall came after."""
    if beat.kind == "push":
        return f"{beat.newtons:.1f} N push from {direction_label(vec, fwd)}"
    if beat.kind == "kick":
        return f"{beat.dv:.2f} m/s kick {direction_label(vec, fwd, toward=True)}"
    return beat.name


# ------------------------------------------------------------------------------ result

@dataclass
class ShowcaseResult:
    label: str
    seed: int
    dt: float
    seconds: float               # how long the video runs (control steps x dt)
    planned_s: float             # the timeline's length before falls stretched it
    beats: List[dict]            # one per beat: when it fired and, if graded, its outcome
    falls: List[dict]
    events: List[dict]           # every push/kick as the wrapper applied it, on the global clock
    settings: dict
    trace: Optional[dict] = field(default=None, repr=False, compare=False)   # per step; not saved

    @property
    def graded(self) -> List[dict]:
        return [b for b in self.beats if b.get("outcome") is not None]

    @property
    def beats_held(self) -> int:
        return sum(b["outcome"] == "held" for b in self.graded)

    def summary(self) -> str:
        return f"beats held {self.beats_held}/{len(self.graded)}, falls {len(self.falls)}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("trace")
        d["beats_held"], d["beats_graded"] = self.beats_held, len(self.graded)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ShowcaseResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "trace"})

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, default=float))
        return path

    @classmethod
    def load(cls, path) -> "ShowcaseResult":
        return cls.from_dict(json.loads(Path(path).read_text()))


def outcome_line(b: dict) -> str:
    """One end-card row: '5 N behind: held, max lean 6.2 deg, recovered 0.9 s'."""
    o = b.get("outcome")
    if o is None:
        return f"{b['name']}: -"
    if o == "not reached":
        return f"{b['name']}: not reached"
    if o == "fell":
        return f"{b['name']}: FELL after {b['fall_after_s']:.2f} s"
    rec = b.get("recover_s")
    rec = f"recovered {rec:.2f} s" if rec is not None and np.isfinite(rec) else "never settled"
    return f"{b['name']}: held, max lean {b['max_lean_deg']:.1f} deg, {rec}"


# ------------------------------------------------------------------------------ driver

def run_showcase(policy, env_cfg: dict, seed: int = 12345, on_frame: Optional[Callable] = None,
                 render_every: int = 2, timeline: Optional[List[Beat]] = None,
                 seconds: float = SHOWCASE_SECONDS, fall_physics_s: float = 1.0,
                 settle_s: float = 1.0, max_resets: int = 6, wenv=None,
                 label: Optional[str] = None) -> ShowcaseResult:
    """Play the timeline on `policy`. `on_frame(wenv, frame)` is called every `render_every`
    control steps with a dict describing what to draw (see _frame). Restores numpy's global
    RNG afterwards, as suite.evaluate does."""
    import mujoco  # noqa: PLC0415
    timeline = sorted(timeline if timeline is not None else SHOWCASE, key=lambda b: b.t)
    rng_state = np.random.get_state()
    try:
        wenv = wenv if wenv is not None else make_eval_env(env_cfg, seconds=60.0)
        base = wenv.base
        m, d = base.env.unwrapped.model, base.env.unwrapped.data
        dt = float(base._control_dt)
        frame_skip = int(base.env.unwrapped.frame_skip)
        fall_steps, settle_steps = round(fall_physics_s / dt), round(settle_s / dt)
        shift = fall_steps + settle_steps

        beats = [{"name": b.name, "kind": b.kind, "planned_t": b.t, "t": b.t,
                  "step": round(b.t / dt), "fired": False, "_beat": b} for b in timeline]
        end_step = round(seconds / dt)
        events: List[dict] = []
        falls: List[dict] = []
        tr = {k: [] for k in ("t", "lean", "pitch", "roll", "com_z", "rate", "contacts", "force",
                              "force_body", "phase")}
        st = {"k": 0, "fault": None, "label": None, "label_until": -1, "push": None, "kick": None,
              "calm": CALM_STEPS, "since": 0, "last_beat": None, "recovered": None,
              "banner": None, "phase": "run", "settle_until": -1, "last_fired": None}

        def record(phase):
            f = d.xfrc_applied[:, :3]
            nz = np.nonzero(np.abs(f).sum(axis=1) > 0)[0]
            pitch, roll = pelvis_attitude(m, d)
            rate = float(np.linalg.norm(d.qvel[3:6]))
            tr["t"].append(st["k"] * dt)
            tr["lean"].append(_lean_deg(d.qpos))
            tr["pitch"].append(pitch)
            tr["roll"].append(roll)
            tr["com_z"].append(float(d.subtree_com[1][2]))
            tr["rate"].append(rate)
            tr["contacts"].append([float(c) for c in base._foot_contacts()])
            tr["force"].append(f.sum(axis=0).copy())
            tr["force_body"].append(int(nz[0]) if len(nz) else -1)
            tr["phase"].append(phase)
            # HUD settle state: settled / recovering for x s / recovered in y s
            was_calm = st["calm"] >= CALM_STEPS
            st["calm"] = st["calm"] + 1 if rate < CALM_RATE else 0
            if st["calm"] == 0 and was_calm:
                st["since"] = st["k"]
            lb = st["last_beat"]
            if lb is not None and st["recovered"] is None and st["calm"] >= CALM_STEPS:
                st["recovered"] = (st["k"] - lb - CALM_STEPS + 1) * dt

        def frame(phase, final=False):
            k = st["k"]
            t = k * dt
            planned_end = end_step * dt
            if st["calm"] >= CALM_STEPS:
                hud = ("recovered", st["recovered"]) if st["recovered"] is not None else ("settled", 0.0)
            else:
                hud = ("recovering", (k - st["since"]) * dt)
            end_card = None
            if (t >= planned_end - 2.0 and phase == "run") or final:   # not over a fall
                end_card = [outcome_line(b) for b in _outcomes(beats, tr, falls, k, dt) if b.get("outcome")]
            return {"t": t, "step": k, "seconds": planned_end, "phase": phase,
                    "lean": tr["lean"][-1], "pitch": tr["pitch"][-1], "roll": tr["roll"][-1],
                    "com_z": tr["com_z"][-1], "rate": tr["rate"][-1], "contacts": tr["contacts"][-1],
                    "hud": hud,
                    "label": st["label"] if k < st["label_until"] else None,
                    "fault": fault_text(st["fault"]) if st["fault"] is not None else None,
                    "push": st["push"], "kick": st["kick"],
                    "beats": [{k2: v for k2, v in b.items() if k2 != "_beat"} for b in beats],
                    "falls": list(falls), "banner": st["banner"], "end_card": end_card,
                    "summary": _summary(beats, tr, falls, k, dt) if end_card else None,
                    "fwd": robot_forward(m, d)}

        def emit(phase, final=False):
            if on_frame is not None and (st["k"] % render_every == 0 or final):
                on_frame(wenv, frame(phase, final))

        def fire(b):
            beat = b["_beat"]
            k = st["k"]
            fwd = robot_forward(m, d)
            vec = None
            if beat.kind in ("push", "kick"):
                u = beat.toward[0] * fwd + beat.toward[1] * robot_left(fwd)
                u = u / np.linalg.norm(u)
                if beat.kind == "push":
                    vec = beat.newtons * u
                    wenv.inject_push(vec, beat.body, beat.duration_s)
                else:
                    vec = beat.dv * u
                    wenv.inject_kick(vec, beat.yaw_rate)
                ev = dict(wenv.events[-1], step=k, t=k * dt, beat=beat.name,
                          side=direction_label(vec, fwd, toward=beat.kind == "kick"))
                events.append(ev)
                if beat.kind == "push":
                    st["push"] = {"t0": k * dt, "vec": ev["vec"], "body_id": ev["body_id"],
                                  "duration_s": ev["duration_steps"] * dt}
                else:
                    st["kick"] = {"t0": k * dt, "vec": ev["vec"], "text": describe(beat, vec, fwd)}
                # Settling is counted afresh from the beat, as suite._recoveries does: a calm
                # streak from before it says nothing about the recovery from it.
                st["last_beat"], st["recovered"], st["since"], st["calm"] = k, None, k, 0
                st["last_fired"] = (k, _short(beat, vec, fwd))
            elif beat.kind == "fault_on":
                st["fault"] = np.asarray(beat.gyro, dtype=np.float64)
                base.set_sensor_offsets(gyro=st["fault"])
            elif beat.kind == "fault_off":
                st["fault"] = None
                base.set_sensor_offsets()
            st["label"], st["label_until"] = describe(beat, vec, fwd), k + round(2.0 / dt)
            b.update(fired=True, step=k, t=k * dt)
            if vec is not None:
                b["vec"] = [float(x) for x in vec]

        def start(s):
            np.random.seed(s)        # StandingEnv draws its noise from the global RNG
            obs, _ = wenv.reset(seed=s, options={"scenario": SHOWCASE_START})
            if hasattr(policy, "reset"):
                policy.reset()
            if st["fault"] is not None:      # the reset cleared it; the fault is still on
                base.set_sensor_offsets(gyro=st["fault"])
            return obs

        obs = start(seed)
        record("run")
        emit("run")
        while st["k"] < end_step:
            for b in beats:
                if not b["fired"] and b["step"] <= st["k"]:
                    fire(b)
            phase = "settling" if st["k"] < st["settle_until"] else "run"
            if phase == "run":
                st["banner"] = None
            a = np.asarray(policy.act(obs), dtype=np.float32)
            obs, _, terminated, truncated, info = wenv.step(a)
            st["k"] += 1
            record(phase)
            if not terminated:
                emit(phase)
                if truncated:
                    break
                continue

            # ---- a fall: show it, then reset and carry on with the rest of the timeline
            k = st["k"]
            cause = info.get("termination_cause", "unknown")
            lf = st["last_fired"]
            after = (f"{(k - lf[0]) * dt:.2f} s after {lf[1]}" if lf is not None else "on release")
            falls.append({"t": k * dt, "step": k, "cause": cause,
                          "after": None if lf is None else lf[1],
                          "after_s": None if lf is None else (k - lf[0]) * dt})
            final = len(falls) > max_resets
            st["banner"] = f"FELL {k * dt:.2f} s ({cause}), {after}"
            emit("falling")
            wt = wenv._t
            for j in range(fall_steps):          # pure physics, last command held
                if wenv._push_until is not None and wt >= wenv._push_until:
                    d.xfrc_applied[wenv._push_bid, :] = 0.0
                    wenv._push_until = wenv._push_bid = None
                mujoco.mj_step(m, d, nstep=frame_skip)
                mujoco.mj_rnePostConstraint(m, d)
                wt += 1
                st["k"] += 1
                record("falling")
                emit("falling", final=final and j == fall_steps - 1)
            if final:
                break
            obs = start(seed + len(falls))
            st["settle_until"] = st["k"] + settle_steps
            st["banner"] += " -- reset"
            st["last_beat"], st["recovered"] = None, None
            for b in beats:
                if not b["fired"]:
                    b["step"] += shift
                    b["t"] = b["step"] * dt
            end_step += shift
    finally:
        np.random.set_state(rng_state)

    n = st["k"]
    out = _outcomes(beats, tr, falls, n, dt)
    trace = {k: np.asarray(v) for k, v in tr.items()}
    return ShowcaseResult(
        label=label or getattr(policy, "label", "policy"), seed=seed, dt=dt, seconds=n * dt,
        planned_s=float(seconds), beats=out, falls=falls, events=events,
        settings={"render_every": render_every, "fall_physics_s": fall_physics_s,
                  "settle_s": settle_s, "max_resets": max_resets,
                  "xml_file": env_cfg.get("xml_file"), "feedback_prior": base.prior_gain is not None,
                  "fingerprint": getattr(policy, "fingerprint", None)},
        trace=trace)


def _outcomes(beats, tr, falls, n, dt) -> List[dict]:
    """Each beat, and for graded beats: held/fell in its window (until the next beat fires),
    the largest lean in it, how long it took to settle (suite rule) and when it fell."""
    rates = np.asarray(tr["rate"])
    lean = np.asarray(tr["lean"])
    fired = [b for b in beats if b["fired"]]
    out = []
    for b in beats:
        row = {k: v for k, v in b.items() if k != "_beat"}
        row["outcome"] = None
        if b["_beat"].graded:
            if not b["fired"]:
                row["outcome"] = "not reached"
            else:
                s = b["step"]
                later = [x["step"] for x in fired if x["step"] > s]
                stop = min(later) if later else n
                fall = next((f for f in falls if s < f["step"] <= stop), None)
                end = fall["step"] if fall else stop
                if end <= s:
                    row["outcome"] = None          # still in progress
                else:
                    row["max_lean_deg"] = float(lean[s:end + 1].max())
                    rec = _recoveries(rates, [s], end, dt)[0]
                    row["recover_s"] = None if not np.isfinite(rec) else float(rec)
                    row["outcome"] = "fell" if fall else "held"
                    row["fall_after_s"] = (fall["step"] - s) * dt if fall else None
        out.append(row)
    return out


def _summary(beats, tr, falls, n, dt) -> str:
    g = [b for b in _outcomes(beats, tr, falls, n, dt) if b.get("outcome") is not None]
    held = sum(b["outcome"] == "held" for b in g)
    return f"beats held {held}/{len(g)}, falls {len(falls)}"
