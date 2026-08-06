#!/usr/bin/env python3
# deploy_standing.py
"""
Real-robot inference loop for the proprioceptive standing policy (RUN ON THE PI).

Pipeline (must match src/environments/standing_env.py exactly):
  sensors -> per-frame feature (57) -> 4-frame history (228) -> VecNormalize -> PPO.predict
  -> tau=0.3 action smoothing -> clip to sim range -> rad->units (sign/limit) -> servo write

Per-frame feature order (standing_env._proprioceptive_features):
  proj_grav(3) | base_ang_vel(3) | (jpos - default_joint_pos)(17) | jvel(17) | last_action(17)
History: last 4 frames, LEFT-padded with zeros until filled, concatenated oldest->newest.

Defaults match the deployed model:
  model  models/final_real_standing_model.zip   (19M, tau=0.3)
  vecnorm models/vecnorm_real_standing.pkl
  tau 0.3, control 40 Hz.

  python scripts/deploy/deploy_standing.py --dry-run     # no hardware: load+predict once, print
  python scripts/deploy/deploy_standing.py               # closed loop on the Pi
"""

from __future__ import annotations
import argparse
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_real_map import SimRealMap, DEFAULT_MAP  # noqa: E402

NJ = 17
FRAME = 6 + 3 * NJ          # 57
HISTORY = 4
OBS_DIM = FRAME * HISTORY   # 228


def load_vecnorm_stats(path):
    with open(path, "rb") as f:
        vn = pickle.load(f)
    mean = np.asarray(vn.obs_rms.mean, dtype=np.float32)
    var = np.asarray(vn.obs_rms.var, dtype=np.float32)
    eps = float(getattr(vn, "epsilon", 1e-8))
    clip = float(getattr(vn, "clip_obs", 50.0))
    assert mean.shape[0] == OBS_DIM, f"vecnorm obs dim {mean.shape[0]} != {OBS_DIM}"
    return mean, var, eps, clip


def normalize(obs, mean, var, eps, clip):
    return np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)


class _SB3Backend:
    """Wraps an SB3 PPO model + vecnorm stats to a uniform predict(raw_obs)->action."""

    def __init__(self, model, mean, var, eps, clip):
        self.model, self.mean, self.var, self.eps, self.clip = model, mean, var, eps, clip

    def predict(self, obs):
        z = normalize(obs, self.mean, self.var, self.eps, self.clip)
        return self.model.predict(z, deterministic=True)[0]


class ObsBuilder:
    """Maintains history + finite-diff joint velocity, emits the 228-dim obs."""

    def __init__(self, m: SimRealMap, dt: float, jvel_alpha: float = 0.35,
                 jvel_clamp: float = 2.5, zero_jvel: bool = False):
        self.m = m
        self.dt = dt
        self.jvel_alpha = float(jvel_alpha)
        self.jvel_clamp = float(jvel_clamp)
        self.zero_jvel = bool(zero_jvel)
        self.hist = deque(maxlen=HISTORY)
        self.prev_jpos = None
        self.reject_count = np.zeros(NJ, dtype=np.int32)  # consecutive implausible reads per joint
        self.rejects_total = 0                            # frames-with-a-reject, for --debug
        self.jvel_f = np.zeros(NJ, dtype=np.float32)      # low-passed joint velocity
        self.last_jvel_raw = np.zeros(NJ, dtype=np.float32)  # for --debug
        self.last_action = np.zeros(NJ, dtype=np.float32)  # smoothed target (sim rad), env init=0

    def frame(self, proj_grav, ang_vel, jpos):
        jpos = np.asarray(jpos, dtype=np.float32)
        if self.prev_jpos is None:
            jpos = np.where(np.isfinite(jpos), jpos, 0.0).astype(np.float32)
            raw = np.zeros(NJ, dtype=np.float32)
        else:
            # Plausibility gate on the POSITION itself, not just jvel: a reading that
            # implies |speed| > jvel_clamp (servo hw max ~2.1 rad/s) or a failed read
            # (NaN) is a corrupted reply -- clamping jvel alone still let the garbage
            # position into the jpos obs (the run-B residual oscillation). Hold the
            # last good value; if the same implausible value persists 3 frames it is
            # real motion (e.g. an external shove), so accept it then.
            bad = ~np.isfinite(jpos)
            if self.jvel_clamp > 0:
                bad |= np.abs(np.nan_to_num(jpos) - self.prev_jpos) / self.dt > self.jvel_clamp
            self.reject_count = np.where(bad, self.reject_count + 1, 0)
            hold = bad & (self.reject_count < 3)
            hold |= ~np.isfinite(jpos)           # NaN can never be accepted
            if hold.any():
                self.rejects_total += 1
            jpos = np.where(hold, self.prev_jpos, jpos).astype(np.float32)
            raw = ((jpos - self.prev_jpos) / self.dt).astype(np.float32)
        self.prev_jpos = jpos.copy()
        self.last_jvel_raw = raw   # keep the TRUE finite-diff for --debug (shows bus glitches)
        # Reject non-physical spikes BEFORE filtering: the servos cap at ~2.1 rad/s, so any
        # |jvel| past jvel_clamp is a garbled half-duplex read, not real motion. A single
        # glitched position sample otherwise injects an 8-12 rad/s spike that the policy (which
        # trained on clean near-zero jvel) amplifies into a 40 Hz seizure. Clamp kills it at
        # the source so the low-pass doesn't have to be cranked so hard it goes sluggish.
        jraw = np.clip(raw, -self.jvel_clamp, self.jvel_clamp) if self.jvel_clamp > 0 else raw
        # Low-pass the (clamped) finite-diff velocity. Quantized-encoder differencing at 40 Hz
        # is still noisier than sim; alpha=1.0 disables (raw), lower = smoother.
        a = self.jvel_alpha
        self.jvel_f = ((1.0 - a) * self.jvel_f + a * jraw).astype(np.float32)
        jvel_out = np.zeros(NJ, dtype=np.float32) if self.zero_jvel else self.jvel_f
        return np.concatenate([proj_grav, ang_vel, jpos, jvel_out, self.last_action]).astype(np.float32)

    def obs(self, frame):
        self.hist.append(frame)
        frames = list(self.hist)
        if len(frames) < HISTORY:
            frames = [np.zeros(FRAME, dtype=np.float32)] * (HISTORY - len(frames)) + frames
        return np.concatenate(frames).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "final_real_standing_model.zip"))
    p.add_argument("--vecnorm", default=str(ROOT / "models" / "vecnorm_real_standing.pkl"))
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--policy-npz", default=None,
                   help="torch-free NumPy policy bundle from export_policy.py. If set, runs with "
                        "numpy only (no torch/SB3/pkl); else loads the SB3 .zip + vecnorm .pkl.")
    p.add_argument("--tau", type=float, default=0.3, help="action smoothing (MUST match training)")
    p.add_argument("--hz", type=float, default=40.0)
    p.add_argument("--ramp-secs", type=float, default=2.0, help="open-loop ramp to home before closed loop")
    p.add_argument("--tilt-cut", type=float, default=0.5, help="hold+cut torque if upright_cos < this (~60deg)")
    p.add_argument("--tilt-debounce", type=int, default=3,
                   help="require upright_cos < tilt-cut for this many CONSECUTIVE frames before "
                        "cutting torque; rejects single-sample IMU/accel-transient glitches.")
    p.add_argument("--projgrav-alpha", type=float, default=0.35,
                   help="EMA low-pass on projected-gravity; 1.0=raw/off, lower=smoother. The IMU "
                        "reports specific force (gravity - body accel), so motion corrupts it; "
                        "gravity is quasi-static for standing, so smoothing rejects the transient.")
    p.add_argument("--jvel-clamp", type=float, default=2.5,
                   help="clamp finite-diff joint velocity to +/- this (rad/s) before the low-pass; "
                        "rejects non-physical spikes from garbled bus reads (servo max ~2.1). 0=off")
    p.add_argument("--max-step-units", type=int, default=80, help="per-joint per-step servo move clamp")
    p.add_argument("--arm-step-units", type=int, default=None,
                   help="tighter per-step move clamp for the ARM servos (12-17) only; defaults to "
                        "--max-step-units. Arms are light and not balance-critical, so throttling "
                        "them kills the whip that pops the horns off without slowing the legs' "
                        "fall-catching. Try 3-5 with legs at 10.")
    p.add_argument("--speed", type=int, default=0, help="servo move speed (0=max)")
    p.add_argument("--jvel-alpha", type=float, default=0.35,
                   help="low-pass on joint-velocity obs (1.0=raw/off, lower=smoother; anti-jitter)")
    p.add_argument("--angvel-alpha", type=float, default=1.0,
                   help="EMA low-pass on the base angular-velocity obs (1.0=raw/off, lower=smoother). "
                        "The angvel channel does double duty: its slow part catches the fall, its fast "
                        "part (amplified by servo lag) drives the limit-cycle jitter. Smoothing keeps "
                        "the fall-catching signal while killing the jitter. Try 0.2-0.3.")
    p.add_argument("--zero-angvel", action="store_true",
                   help="DIAGNOSTIC: feed zeros for the base angular-velocity obs channel. If the "
                        "limit-cycle sway STOPS, the gyro feedback is driving it (likely a flipped "
                        "sign -> anti-damping). If it keeps swaying, the gyro is not the engine.")
    p.add_argument("--zero-jvel", action="store_true",
                   help="DIAGNOSTIC: feed zeros for the joint-velocity obs channel. A sensitivity "
                        "probe shows the policy amplifies jvel noise into ~0.13 rad of command "
                        "jitter; for a standing robot true jvel ~ 0, so if the jitter DROPS with "
                        "jvel zeroed, the noisy finite-diff jvel is the jitter source.")
    p.add_argument("--freeze-after", type=int, default=0,
                   help="DIAGNOSTIC: run closed-loop for N steps to reach the policy's standing "
                        "pose, then FREEZE the command and hold it open-loop (sensors still read "
                        "for the safety cut). A fixed target can't limit-cycle, so this isolates "
                        "static balance from control instability: if the frozen pose HOLDS (stands "
                        "on release), the policy's pose is statically stable and the sway is a "
                        "feedback instability; if it TIPS, the policy's target pose itself is not "
                        "balanced on the real robot (mass/pose mismatch). 0 = off.")
    p.add_argument("--hold-pose", action="store_true",
                   help="DIAGNOSTIC: drive straight to the policy's NOMINAL standing pose (the "
                        "deterministic action at a perfect upright obs == the --dry-run pose) and HOLD "
                        "it open-loop forever (sensors read only for the safety cut). NO closed-loop "
                        "feedback runs, so it cannot limit-cycle/seize -- this is a pure test of "
                        "whether the policy's INTENDED stance is statically balanced on the real "
                        "robot, with the seizure removed entirely. Combine with --lean-back.")
    p.add_argument("--lean-back", type=float, default=0.0,
                   help="DIAGNOSTIC (with --hold-pose or --freeze-after): lean the torso back by this "
                        "many DEGREES at the waist_pitch joint before holding. The policy commands a "
                        "forward-leaning waist (~15 deg past neutral) that drops the real chest mass "
                        "past the toes and tips it. Dial this up until the held pose STANDS unaided to "
                        "measure how much the learned pose over-leans (confirms an upright-posture "
                        "retrain will fix it). Positive = lean back/upright.")
    p.add_argument("--residual-config", default=None,
                   help="YAML with standing.residual_clamp + standing.residual_baseline (e.g. "
                        "config/real_humanoid_residual.yaml). Mirrors training: clamps the raw "
                        "policy action to baseline +/- clamp BEFORE the tau-EMA, and ramps to "
                        "the BASELINE pose (the settled attractor) instead of straight. Use with "
                        "a model TRAINED in residual mode.")
    p.add_argument("--trim", default=None,
                   help="ON-ROBOT static calibration (residual mode only): comma-separated "
                        "joint=degrees offsets added to the residual baseline (shifts the ramp "
                        "target AND the clamp box together), e.g. "
                        "--trim \"R_hip_pitch=-2,L_hip_pitch=-2\" to lean the body 2 deg forward "
                        "or \"R_hip_roll=1,L_hip_roll=1\" for lateral trim. Keep trims small "
                        "(<=3 deg): the policy trained with baseline offsets up to ~3 deg "
                        "(residual_baseline_rand), so small trims are in-distribution. Iterate: "
                        "run, watch which way it leans, trim against it, run again.")
    p.add_argument("--debug", action="store_true", help="print obs/action diagnostics each --debug-every steps")
    p.add_argument("--debug-every", type=int, default=10)
    p.add_argument("--require-verified", action="store_true",
                   help="refuse to drive joints whose sign is not bench-verified")
    p.add_argument("--imu-calib", default=str(ROOT / "config" / "imu_calib.yaml"),
                   help="YAML with axis_remap (raw IMU axes -> pelvis/base frame). Identity if missing.")
    p.add_argument("--dry-run", action="store_true", help="no hardware: load, predict once with zero sensors, print")
    args = p.parse_args()

    dt = 1.0 / args.hz
    m = SimRealMap(args.map)
    home_units = m.rad_to_units(m.default_joint_pos)

    unverified = m.unverified_idxs()
    if unverified:
        msg = f"[warn] {len(unverified)} joints have UNVERIFIED sign: {unverified} (run verify_signs.py)"
        if args.require_verified:
            print(msg + " -- aborting (--require-verified).")
            return
        print(msg)

    # ---- load policy (numpy bundle OR torch/SB3) ----
    if args.policy_npz:
        from numpy_policy import NumpyPolicy  # noqa: E402
        npol = NumpyPolicy.load(args.policy_npz)
        assert npol.obs_dim == OBS_DIM, f"npz obs_dim {npol.obs_dim} != {OBS_DIM}"
        infer = npol.predict  # raw obs -> action; normalization baked in
        print(f"Policy: NumPy bundle {args.policy_npz} (no torch/SB3, obs_dim={npol.obs_dim})")
    else:
        from stable_baselines3 import PPO  # noqa: E402
        model = PPO.load(args.model, device="cpu")
        mean, var, eps, clip = load_vecnorm_stats(args.vecnorm)
        infer = _SB3Backend(model, mean, var, eps, clip).predict
        print(f"Policy: SB3 {args.model} + vecnorm (obs_dim={OBS_DIM}, clip={clip})")

    builder = ObsBuilder(m, dt, jvel_alpha=args.jvel_alpha, jvel_clamp=args.jvel_clamp,
                         zero_jvel=args.zero_jvel)

    res_lo = res_hi = None
    res_baseline = None
    if args.residual_config:
        import yaml  # noqa: E402
        with open(args.residual_config) as f:
            rc = yaml.safe_load(f)["standing"]
        # scalar OR per-joint vector -- must mirror standing_env.py exactly or deploy clamps
        # to a different box than training did.
        clamp = np.asarray(rc["residual_clamp"], dtype=np.float32)
        assert clamp.ndim == 0 or clamp.shape == (NJ,), \
            f"residual_clamp len {clamp.shape} != {NJ}"
        res_baseline = np.asarray(rc["residual_baseline"], dtype=np.float32)
        assert res_baseline.shape == (NJ,), f"residual_baseline len {res_baseline.shape} != {NJ}"
        if args.trim:
            dofs = [j.dof for j in m.joints]
            for part in args.trim.split(","):
                name, deg = part.split("=")
                name = name.strip()
                assert name in dofs, f"--trim: unknown joint '{name}' (choices: {dofs})"
                res_baseline[dofs.index(name)] += np.deg2rad(float(deg))
                print(f"TRIM: {name} {float(deg):+.1f} deg")
        res_lo = res_baseline - clamp
        res_hi = res_baseline + clamp
        _cd = (f"{float(clamp):.2f}" if clamp.ndim == 0
               else f"per-joint {float(clamp.min()):.2f}..{float(clamp.max()):.2f}")
        print(f"RESIDUAL mode: baseline +/- {_cd} rad from {args.residual_config}")

    # Per-joint residual diagnostics. The debug line's max|act| is a max over all 17 joints,
    # so a saturated run cannot be read: it does not say WHICH joints ride the clamp box or
    # in which direction. That distinction decides the fix -- a policy pinned on a few joints
    # in a consistent direction wants a wider box; one pinned on many joints in alternating
    # directions is chasing a pose the box cannot reach, and widening it only buys bigger
    # excursions. These counters separate the two.
    dofs = [j.dof for j in m.joints]
    sat_hi = np.zeros(NJ, dtype=int)
    sat_lo = np.zeros(NJ, dtype=int)
    dev_sum = np.zeros(NJ, dtype=float)
    n_dev = 0
    EDGE = 0.005   # rad of slack when calling a joint "at the edge"

    def predict_units(proj_grav, ang_vel, jpos):
        frame = builder.frame(proj_grav, ang_vel, jpos)
        obs = builder.obs(frame)
        raw_action = np.asarray(infer(obs), dtype=np.float32).ravel()
        # Match StandingEnv._process_action: SB3 clips the action to the space BEFORE the
        # env smooths it, so clip raw -> (residual clamp) -> EMA-smooth -> clip again.
        raw_action = np.clip(raw_action, m.range_lo, m.range_hi)
        if res_lo is not None:
            raw_action = np.clip(raw_action, res_lo, res_hi)
        applied = (1.0 - args.tau) * builder.last_action + args.tau * raw_action
        applied = np.clip(applied, m.range_lo, m.range_hi)
        builder.last_action = applied
        return m.rad_to_units(applied), applied

    if args.dry_run:
        pg = np.array([0, 0, -1], dtype=np.float32)   # perfectly upright
        av = np.zeros(3, dtype=np.float32)
        jpos = np.zeros(NJ, dtype=np.float32)         # at home (encoder==default -> jpos 0)
        for step in range(HISTORY):                   # fill history
            units, applied = predict_units(pg, av, jpos)
        print("\n[dry-run] first deterministic action (sim rad):")
        print(np.array2string(applied, precision=3, suppress_small=True))
        print("[dry-run] -> servo units:")
        for i in range(NJ):
            print(f"  {m.joints[i].dof:<18} servo {m.servo_ids[i]:2d}: {int(units[i])}")
        print("\n[dry-run] OK: model + vecnorm + map wired correctly. No hardware touched.")
        return

    # ---- hardware ----
    from hardware import ServoBus, IMU  # noqa: E402
    axis_remap = None
    if os.path.exists(args.imu_calib):
        import yaml  # noqa: E402
        with open(args.imu_calib) as f:
            axis_remap = np.asarray(yaml.safe_load(f)["axis_remap"], dtype=np.float32)
        print(f"IMU axis_remap loaded from {args.imu_calib}:\n{axis_remap}")
    bus = ServoBus().connect()
    imu = IMU(axis_remap=axis_remap).connect()

    def read_units_or(fallback):
        """read_all with NaN (validation-failed read) replaced by a safe fallback, so a
        dropped reply can't NaN-poison a ramp computation."""
        u = bus.read_all(m.servo_ids)
        return np.where(np.isfinite(u), u, np.asarray(fallback, dtype=float))

    pg_state = {"g": None}   # EMA-filtered projected gravity (closure state)
    av_state = {"w": None}   # EMA-filtered angular velocity (closure state)

    def read_sensors():
        # IMU is on the pelvis/base body, so proj_grav/ang_vel are already in the obs frame.
        pg = imu.projected_gravity()
        av = imu.angular_velocity()
        # Low-pass the angular velocity: keep the slow fall-catching component, drop the fast
        # jitter the servo lag amplifies into the limit cycle. 1.0 = raw (off).
        aw = args.angvel_alpha
        if av_state["w"] is None or aw >= 1.0:
            av_state["w"] = av
        else:
            av_state["w"] = ((1.0 - aw) * av_state["w"] + aw * av).astype(np.float32)
        av = av_state["w"]
        # EMA low-pass proj_grav and renormalize. The accelerometer reports specific force
        # (gravity - linear accel), so the robot's own motion swings the measured "gravity"
        # direction; gravity is quasi-static while standing, so smoothing rejects that
        # transient (and the false tilt-cuts it caused) without a gyro-fusion sign risk.
        a = args.projgrav_alpha
        if pg_state["g"] is None or a >= 1.0:
            pg_state["g"] = pg
        else:
            g = (1.0 - a) * pg_state["g"] + a * pg
            n = np.linalg.norm(g)
            pg_state["g"] = (g / n).astype(np.float32) if n > 1e-6 else g.astype(np.float32)
        pg = pg_state["g"]
        units = bus.read_all(m.servo_ids)
        abs_rad = m.units_to_rad(units)                    # absolute joint angles (sim, 0=straight)
        jpos = (abs_rad - m.default_joint_pos).astype(np.float32)
        return pg, av, jpos

    # Ramp target = SIM-STRAIGHT: all joints at sim 0 = the per-joint centers. Measured in sim,
    # the balanced policy's settled standing attractor is joints ~= 0 (symmetric to ~0.003 rad,
    # upright, low-gain), where the obs jpos = (0 - default_joint_pos) = -default -- the actual
    # IN-DISTRIBUTION standing observation. default_joint_pos (the keyframe) is only the sim
    # RESET pose; the policy drives away from it to straight. Do NOT ramp to a pose computed from
    # a synthetic jpos=0 obs (the old --dry-run/nominal-pose ramp): that obs corresponds to the
    # robot being AT the keyframe, which the policy does not stand in, so it returns a spurious
    # off-distribution asymmetric command. Starting AT straight puts the robot directly at the
    # settled obs, so the policy should just hold (calm, symmetric) like it does in sim.
    # RESIDUAL mode: ramp to the BASELINE pose instead -- the measured settled attractor of the
    # post-mass-fix model is NOT straight (e.g. one knee at -0.5 rad), so ramping to straight
    # started every run off-attractor and the policy lurched toward its real stance.
    if res_baseline is not None:
        home_units = np.clip(m.rad_to_units(res_baseline), m.lim_lo, m.lim_hi)
    else:
        home_units = m.rad_to_units(np.zeros(m.n, dtype=np.float32))

    # Per-joint per-step move clamp: legs/waist at --max-step-units (they need speed to catch a
    # fall), arms (shoulder/elbow) at the tighter --arm-step-units so they can't whip/pop a horn.
    arm_step = args.arm_step_units if args.arm_step_units is not None else args.max_step_units
    step_cap = np.full(NJ, float(args.max_step_units))
    arm_mask = np.array(["shoulder" in j.dof or "elbow" in j.dof for j in m.joints])
    step_cap[arm_mask] = float(arm_step)

    try:
        print("Enabling torque, calibrating gyro, ramping to home...")
        bus.set_torque(m.servo_ids, True)
        cur = read_units_or(home_units)
        steps = max(1, int(args.ramp_secs / dt))
        for k in range(1, steps + 1):
            u = (cur + (home_units - cur) * k / steps).round().astype(int)
            bus.write_all(m.servo_ids, u, speed=args.speed)
            time.sleep(dt)
        imu.calibrate_gyro_bias(seconds=1.5)

        # warm up history with real frames at rest
        for _ in range(HISTORY):
            pg, av, jpos = read_sensors()
            builder.obs(builder.frame(pg, av, jpos))
            time.sleep(dt)

        if args.hold_pose:
            # Compute the policy's nominal pose deterministically (synthetic upright + home obs,
            # like --dry-run). Feed it long enough for the tau-EMA to settle to the raw target.
            pg0 = np.array([0, 0, -1], dtype=np.float32)
            av0 = np.zeros(3, dtype=np.float32)
            jp0 = np.zeros(NJ, dtype=np.float32)
            for _ in range(40):
                units, applied = predict_units(pg0, av0, jp0)
            target = np.asarray(units, dtype=int).copy()
            if args.lean_back != 0.0:
                wp = next(i for i, j in enumerate(m.joints) if j.dof == "waist_pitch")
                d = int(round(m.signs[wp] * np.deg2rad(args.lean_back) * m.units_per_rad))
                target[wp] = int(np.clip(target[wp] + d, m.lim_lo[wp], m.lim_hi[wp]))
                print(f"[hold-pose] lean-back {args.lean_back:+.1f} deg: waist_pitch "
                      f"servo {m.servo_ids[wp]} {int(units[wp])} -> {target[wp]}")
            cur = read_units_or(target)
            steps = max(1, int(args.ramp_secs / dt))
            for k in range(1, steps + 1):
                u = (cur + (target - cur) * k / steps).round().astype(int)
                bus.write_all(m.servo_ids, u, speed=args.speed)
                time.sleep(dt)
            print("[hold-pose] HOLDING nominal pose open-loop (NO feedback -- cannot seize). "
                  "Ease your hands away: does it STAND or TIP?")
            tilt_bad = 0
            while True:
                pg, av, jpos = read_sensors()
                upright_cos = -float(pg[2])
                if upright_cos < args.tilt_cut:
                    tilt_bad += 1
                    if tilt_bad >= args.tilt_debounce:
                        print(f"\n[SAFETY] upright_cos={upright_cos:.2f} < {args.tilt_cut}: cutting torque.")
                        bus.set_torque(m.servo_ids, False)
                        break
                else:
                    tilt_bad = 0
                bus.write_all(m.servo_ids, target, speed=args.speed)
                time.sleep(dt)
            return

        print(f"Closed loop running (Ctrl-C to stop). jvel_alpha={args.jvel_alpha}, "
              f"max_step_units={args.max_step_units}, tau={args.tau}")
        prev_units = home_units.copy().astype(float)
        prev_applied = builder.last_action.copy()
        tilt_bad = 0
        step = 0
        frozen_units = None   # set once --freeze-after fires; held thereafter
        loop_ms_ema = dt * 1000.0   # EMA of realized loop period; flags if the bus can't keep 40 Hz
        t_prev = time.time()
        while True:
            t0 = time.time()
            loop_ms_ema = 0.9 * loop_ms_ema + 0.1 * (t0 - t_prev) * 1000.0
            t_prev = t0
            pg, av, jpos = read_sensors()
            if args.zero_angvel:
                av = np.zeros(3, dtype=np.float32)
            t_sense = time.time()

            upright_cos = -float(pg[2])
            # Debounce: only cut after the (filtered) tilt stays past the threshold for
            # several consecutive frames, so a lone glitch sample can't kill a good run.
            if upright_cos < args.tilt_cut:
                tilt_bad += 1
                if tilt_bad >= args.tilt_debounce:
                    print(f"\n[SAFETY] upright_cos={upright_cos:.2f} < {args.tilt_cut} for "
                          f"{tilt_bad} frames: cutting torque.")
                    bus.set_torque(m.servo_ids, False)
                    break
            else:
                tilt_bad = 0

            if args.freeze_after and step >= args.freeze_after:
                if frozen_units is None:
                    frozen_units = prev_units.astype(int).copy()
                    if args.lean_back != 0.0:
                        wp = next(i for i, j in enumerate(m.joints) if j.dof == "waist_pitch")
                        d = int(round(m.signs[wp] * np.deg2rad(args.lean_back) * m.units_per_rad))
                        frozen_units[wp] = int(np.clip(frozen_units[wp] + d, m.lim_lo[wp], m.lim_hi[wp]))
                        print(f"[FREEZE] lean-back {args.lean_back:+.1f} deg: waist_pitch "
                              f"servo {m.servo_ids[wp]} {int(prev_units[wp])} -> {frozen_units[wp]}")
                    print(f"\n[FREEZE] holding the policy's pose open-loop at step {step} "
                          f"(feedback off). Ease your hands away: does it STAND or TIP?")
                units = frozen_units
                applied = prev_applied   # keep debug readout sane
            else:
                units, applied = predict_units(pg, av, jpos)
                # per-step move clamp (rate limit), then write
                units = np.clip(units, prev_units - step_cap, prev_units + step_cap)
                units = np.clip(units, m.lim_lo, m.lim_hi).round().astype(int)
            bus.write_all(m.servo_ids, units, speed=args.speed)
            prev_units = units.astype(float)

            if res_baseline is not None:
                dev = np.asarray(applied, dtype=float) - res_baseline
                sat_hi += (dev >= clamp - EDGE)
                sat_lo += (dev <= -clamp + EDGE)
                dev_sum += dev
                n_dev += 1

            if args.debug and step % args.debug_every == 0:
                sense_ms = (t_sense - t0) * 1000.0
                busy_ms = (time.time() - t0) * 1000.0   # sense+predict+write, before sleep
                rate = 1000.0 / loop_ms_ema if loop_ms_ema > 0 else 0.0
                slow = "  <<SLOW: bus-bound, real Hz < target" if busy_ms > dt * 1000.0 else ""
                print(f"[{step:5d}] pg=[{pg[0]:+.2f},{pg[1]:+.2f},{pg[2]:+.2f}] "
                      f"|av|={np.linalg.norm(av):.2f} "
                      f"max|jvel_raw|={np.max(np.abs(builder.last_jvel_raw)):5.1f} "
                      f"max|jvel_f|={np.max(np.abs(builder.jvel_f)):5.1f} "
                      f"max|act|={np.max(np.abs(applied)):.2f} "
                      f"d_act={np.max(np.abs(applied - prev_applied)):.3f} "
                      f"rej={builder.rejects_total} "
                      f"| {rate:4.1f}Hz sense={sense_ms:4.1f}ms busy={busy_ms:4.1f}ms{slow}")
                if res_baseline is not None:
                    dev = np.asarray(applied, dtype=float) - res_baseline
                    top = np.argsort(-np.abs(dev))[:4]
                    print("          resid: "
                          + "  ".join(f"{dofs[i]}{dev[i]:+.2f}" for i in top)
                          + f"   at-edge {int(np.sum(np.abs(dev) >= clamp - EDGE))}/{NJ}")
            prev_applied = applied.copy()
            step += 1

            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nStopping: ramping to home and disabling torque.")
        try:
            cur = read_units_or(home_units)
            steps = max(1, int(1.0 / dt))
            for k in range(1, steps + 1):
                u = (cur + (home_units - cur) * k / steps).round().astype(int)
                bus.write_all(m.servo_ids, u, speed=args.speed)
                time.sleep(dt)
        except Exception:
            pass
    finally:
        if n_dev > 0:
            print(f"\n--- residual saturation summary ({n_dev} frames, clamp {clamp}) ---")
            print(f"{'joint':<18} {'mean dev':>9} {'at +edge':>9} {'at -edge':>9}")
            for i in np.argsort(-(sat_hi + sat_lo)):
                if sat_hi[i] + sat_lo[i] == 0 and abs(dev_sum[i] / n_dev) < 0.02:
                    continue
                print(f"{dofs[i]:<18} {dev_sum[i]/n_dev:+9.3f} "
                      f"{100.0*sat_hi[i]/n_dev:8.0f}% {100.0*sat_lo[i]/n_dev:8.0f}%")
        bus.set_torque(m.servo_ids, False)
        bus.close()
        imu.close()


if __name__ == "__main__":
    main()
