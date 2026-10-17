#!/usr/bin/env python3
"""Generate an MJCF whose waist has the compliance the real robot was measured to have.

WHY. Every standing policy passed its sim gates 6/6 and fell on hardware in about two
seconds, and the reason is now measured: the real waist is not rigid. A 0.4-3 Hz sweep of
waist_roll (`config/measured_actuator.yaml`) found the joint moving 1.37x FARTHER than
commanded at 1.05 Hz, collapsing to 0.49 at 2.15 Hz, then recovering. A resonance followed
by an anti-resonance is what a two-mass system does -- actuator inertia coupled through a
compliant element to a load inertia, here servo -> waist bracket -> torso. In a rigid MJCF
the chest pose is a pure function of joint angles, so none of it can exist, which is
exactly why sim could never reproduce the failure.

WHERE THE SPRING GOES. Between the servo body and everything it carries, NOT at the joint
the actuator drives. The real encoder is on the servo output shaft, so it reads the servo
side; the torso hangs beyond the bracket and back-drives that shaft through the bracket's
compliance. Modelling it as a softer actuator instead would reproduce a resonance but no
anti-resonance, and the anti-resonance is the part that says the load is a separate mass.

  base_link
    +-- 0003_6            <- servo body, driven by Revolute 22 (the ENCODER reads this)
          +-- flex body   <- new: near-massless, passive spring joint on the same axis
                +-- 0003_7 ... torso, arms, chest electronics (the LOAD)

ROLL ONLY, BY DEFAULT. waist_roll is the axis that was measured to resonate. waist_pitch
was re-measured from a clean start on the same day and came back ordinary -- gain 1.000,
monotonic phase, no peak -- so it gets no spring. Do not add compliance to an axis you
have not measured just because it shares the bracket; measure it first with
`excite_joint.py --mode chirp --f0 0.4 --f1 3 --secs 60`.

COMPATIBILITY. Inserting a body adds a DOF, which shifts qpos, so an env that reads
qpos[7:7+nu] silently reads the wrong joints. standing_env.py indexes actuated joints
through the actuator transmission for exactly this reason; on the rigid model that
resolves to the same slice it always used. Obs stays 228 and action stays 17, so existing
checkpoints resume unchanged.

WHAT IT ACHIEVES, honestly. Simulated waist_roll goes from rms 0.508 against the measured
frequency response to rms 0.140, and from a flat 0.844 rolloff that peaks nowhere to a
resonance of 1.27 at 1.25 Hz against hardware's 1.37 at 1.05. The shape is right and sim
can now express the failure mode at all, which it structurally could not before. It is not
yet a match: the peak sits about 20% high in frequency and the anti-resonance is shallower
than measured (0.56 vs 0.49) and late. Treat this as a model that contains the phenomenon,
not one that reproduces it to within measurement error.

  python scripts/debug/make_compliant_waist.py               # fitted defaults
  python scripts/debug/make_compliant_waist.py --measure     # sweep it and compare
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE_XML = ROOT / "models" / "humanoid_real_v2.xml"
OUT_XML = ROOT / "models" / "humanoid_real_v2_compliant.xml"

# Which actuated joint gets a spring beneath it, and what to call the new pieces.
AXES = {
    "roll": {"body": "0003_6", "joint": "Revolute 22", "flex": "waist_flex_roll"},
    "pitch": {"body": "0003_7", "joint": "Revolute 20", "flex": "waist_flex_pitch"},
    "yaw": {"body": "0003_8", "joint": "Revolute 19", "flex": "waist_flex_yaw"},
}


def find_parent(root, child):
    for p in root.iter():
        for c in list(p):
            if c is child:
                return p
    return None


def build(stiffness, damping, axes=("roll",), base=BASE_XML, out=OUT_XML,
          armature=None, joint_damping=None, kp=None, kv=None):
    """Insert the spring AND retune the driven joint. Both halves are required.

    Adding the spring alone changes nothing measurable, which is worth stating plainly
    because it was the first thing tried. The shipped model gives the waist joint
    armature 0.05 (three times the inertia it drives), joint damping 1.0, and an actuator
    kv of 2.0 -- together a damping ratio of 1.67, and the actuator's kv alone is worth
    2.0. An overdamped second-order system cannot have a resonant peak no matter what is
    hung off it, so the spring's effect was invisible until those three were fixed. None
    of the three was ever measured; they are defaults from the CAD conversion.
    """
    tree = ET.parse(base)
    root = tree.getroot()
    for axis in axes:
        spec = AXES[axis]
        body = next(b for b in root.iter("body") if b.get("name") == spec["body"])
        joint = next(j for j in body.iter("joint") if j.get("name") == spec["joint"])

        flex = ET.Element("body", {"name": spec["flex"], "pos": "0 0 0"})
        # Near-massless: the flex element is a bracket, not a link. Its inertia must not
        # change the load the servo sees, or the spring would be measuring something else.
        ET.SubElement(flex, "inertial", {
            "pos": "0 0 0", "mass": "1e-5", "diaginertia": "1e-9 1e-9 1e-9"})
        # Same axis and pivot as the driven joint, so the spring acts along the same DOF.
        # armature 0: this is a bracket flexing, not a geared rotor, so it carries none of
        # the reflected motor inertia the default 0.05 would add.
        ET.SubElement(flex, "joint", {
            "name": spec["flex"] + "_j", "pos": joint.get("pos"), "axis": joint.get("axis"),
            "range": "-0.5 0.5", "stiffness": f"{stiffness:g}", "damping": f"{damping:g}",
            "armature": "0", "actuatorfrcrange": "-100 100"})

        moved = [c for c in list(body) if c.tag == "body"]
        for c in moved:
            body.remove(c)
            flex.append(c)
        body.append(flex)

        if armature is not None:
            joint.set("armature", f"{armature:g}")
        if joint_damping is not None:
            joint.set("damping", f"{joint_damping:g}")
        act = next((a for a in root.iter("position") if a.get("joint") == spec["joint"]), None)
        if act is not None:
            if kp is not None:
                act.set("kp", f"{kp:g}")
            if kv is not None:
                act.set("kv", f"{kv:g}")
        print(f"  {axis}: spring under {spec['body']} ({spec['joint']}), "
              f"k={stiffness:g} N m/rad, c={damping:g} N m s/rad, "
              f"moved {len(moved)} child bodies onto it")
        if armature is not None:
            print(f"        driven joint retuned: armature={armature:g} damping={joint_damping:g} "
                  f"kp={kp:g} kv={kv:g}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="unicode")
    return out


def chirp_response(xml, joint_name, f0=0.4, f1=3.0, secs=60.0, amp_rad=0.1396, n_bins=26):
    """Drive one joint with the same sweep excite_joint.py uses and read the same joint back.

    Deliberately identical in form to the hardware measurement -- same frequencies, same
    amplitude (8 deg), same projection onto the chirp's phase function -- so the two Bode
    curves can be compared directly rather than through a model.
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qadr = m.jnt_qposadr[jid]
    aid = next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == jid)

    d.qpos[:] = 0
    d.qpos[3] = 1.0
    d.qpos[2] = 0.5          # held in the air: this measures the joint, not balance
    mujoco.mj_forward(m, d)

    dt = m.opt.timestep
    n = int(secs / dt)
    t = np.arange(n) * dt
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * secs))
    cmd = amp_rad * np.sin(phase)
    meas = np.empty(n)
    for k in range(n):
        d.ctrl[:] = 0
        d.ctrl[aid] = cmd[k]
        # Root pinned, exactly as the real robot was supported for its sweep.
        d.qpos[0:3] = [0, 0, 0.5]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qvel[0:6] = 0
        mujoco.mj_step(m, d)
        meas[k] = d.qpos[qadr]

    inst_f = f0 + (f1 - f0) * t / secs
    ref = np.exp(-1j * phase)
    u = cmd - cmd.mean()
    y = meas - meas.mean()
    edges = np.linspace(0, secs, n_bins + 1)
    freq, gain, ph = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (t >= a) & (t < b)
        U, Y = np.sum(u[s] * ref[s]), np.sum(y[s] * ref[s])
        if abs(U) < 1e-12:
            continue
        H = Y / U
        freq.append(float(np.mean(inst_f[s])))
        gain.append(float(abs(H)))
        ph.append(float(np.degrees(np.angle(H))))
    ph = np.degrees(np.unwrap(np.radians(np.array(ph))))
    return np.array(freq), np.array(gain), ph


# Hardware truth for waist_roll, from the 8/14 narrow sweep. See config/measured_actuator.yaml.
HW = {"resonance_hz": 1.05, "peak_gain": 1.366, "anti_hz": 2.15, "anti_gain": 0.487}


def summarize(freq, gain, label):
    pk, nt = int(np.argmax(gain)), int(np.argmin(gain))
    print(f"  {label:22s} peak {gain[pk]:.3f} @ {freq[pk]:.2f} Hz   "
          f"min {gain[nt]:.3f} @ {freq[nt]:.2f} Hz")
    return freq[pk], gain[pk], freq[nt], gain[nt]


def main():
    p = argparse.ArgumentParser(description="Build (and check) a compliant-waist MJCF")
    # Defaults are FITTED to the 8/14 hardware sweep, not guessed. They take the
    # simulated waist_roll response from rms 0.508 against hardware to rms 0.140.
    p.add_argument("--stiffness", type=float, default=1.551, help="spring, N m/rad")
    p.add_argument("--damping", type=float, default=0.0566, help="spring, N m s/rad")
    p.add_argument("--armature", type=float, default=0.0437,
                   help="driven-joint rotor inertia (shipped default 0.05 was never measured)")
    p.add_argument("--joint-damping", type=float, default=0.0,
                   help="driven-joint viscous damping (shipped default 1.0 alone gives zeta 1.7)")
    p.add_argument("--kp", type=float, default=8.81, help="actuator kp (shipped 12.0)")
    p.add_argument("--kv", type=float, default=0.642,
                   help="actuator kv (shipped 2.0, which alone forces zeta 2.0 -- overdamped)")
    p.add_argument("--axes", default="roll", help="comma-separated: roll,pitch,yaw")
    p.add_argument("--measure", action="store_true",
                   help="sweep rigid vs compliant and compare against the hardware numbers")
    p.add_argument("--out", default=str(OUT_XML))
    args = p.parse_args()

    axes = tuple(a.strip() for a in args.axes.split(",") if a.strip())
    print(f"building {args.out}")
    out = build(args.stiffness, args.damping, axes, out=Path(args.out),
                armature=args.armature, joint_damping=args.joint_damping,
                kp=args.kp, kv=args.kv)

    import mujoco
    m0 = mujoco.MjModel.from_xml_path(str(BASE_XML))
    m1 = mujoco.MjModel.from_xml_path(str(out))
    print(f"\nrigid    nq={m0.nq} nv={m0.nv} nu={m0.nu}")
    print(f"compliant nq={m1.nq} nv={m1.nv} nu={m1.nu}   (+{m1.nv - m0.nv} passive DOF, "
          f"actuator count unchanged)")

    if args.measure:
        print("\nfrequency response of waist_roll (Revolute 22), 8 deg sweep 0.4-3 Hz:")
        f0_, g0_, _ = chirp_response(BASE_XML, "Revolute 22")
        summarize(f0_, g0_, "sim RIGID")
        f1_, g1_, _ = chirp_response(out, "Revolute 22")
        summarize(f1_, g1_, "sim COMPLIANT")
        print(f"  {'HARDWARE':22s} peak {HW['peak_gain']:.3f} @ {HW['resonance_hz']:.2f} Hz   "
              f"min {HW['anti_gain']:.3f} @ {HW['anti_hz']:.2f} Hz")
        print(f"\n  {'f Hz':>6} {'rigid':>7} {'compliant':>10}")
        for a, b, c in zip(f1_, np.interp(f1_, f0_, g0_), g1_):
            print(f"  {a:6.2f} {b:7.3f} {c:10.3f}  {'#' * int(c * 24)}")


if __name__ == "__main__":
    main()
