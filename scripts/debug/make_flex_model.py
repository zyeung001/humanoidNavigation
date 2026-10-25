#!/usr/bin/env python3
"""Generate the plated-robot MJCF with the two things 10/25's hardware logs showed the sim lacks.

1. THE FLOATING BASE WAS MOVING THROUGH HONEY. The MJCF default class gives every joint
   damping 1.0 and armature 0.05 -- including the free joint, so the whole robot had 1 N*s/m
   of linear and 1 N*m*s/rad of angular drag against the world, and its rotational inertia
   roughly doubled. Rocking and tipping were damped away: no sim variant ever swung past
   upright (the robot swings 2-6 deg past), and the 0.2 s torso shove that topples it was
   5.2/5.6 N where it is 3.75 N without the drag. Here the free joint gets 0 and 0.

2. SERVO PLAY. With a foot flat (both left-foot FSRs loaded) the pelvis IMU pitches ~3x as far
   as the leg encoders explain (scratchpad flex_check.py on every log since 10/18, gyro-
   confirmed): ~70-80% of the pelvis pitch is in no encoder. The user: every servo has a small
   movement even when it is not rotating. The encoders read the servo output, so the play is
   invisible to them -- and to the policy. Modelled as one effective passive pitch hinge per
   foot (the sole box and foot mesh move into a child body of the shin, hinged about the
   foot's lateral axis at the top of the sole), with stiffness K per leg. Play at the hips
   instead was tried (series hinges at the hip-pitch and knee axes) and made the model fragile
   to every disturbance, so the effective hinge sits low.

WHAT IT ACHIEVES (knee2 stance, hold, against sysID run 20261025_020236 + the 10/24 tests):
  K 6-8: share of pelvis pitch in no encoder 0.60-0.68 (hw ~0.75); hand release from +8 swings
  to -2 (rigid: 0; hw from +15..+19 swings 2.5-6 past rest); swing period 1.7-2.9 s (hw 1.3-2.4);
  0.2 s shove to topple 1.2-1.6 N (rigid with the drag 5.2-5.6, the 10/18 pendulum test put hw
  ~3x weaker than that sim). NOT matched: hw came back from a +17..+19 deg forward lean, this
  model tips at ~11-12 (some of the robot's play is higher up the leg).

  python scripts/debug/make_flex_model.py               # K 7 -> models/humanoid_real_v2_soleplate_flex.xml
  python scripts/debug/make_flex_model.py --k 8 --out /tmp/x.xml
Training randomises K per episode (standing_env flex_rand_range scales these values).
"""
import argparse
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "models/humanoid_real_v2_compliant_soleplate.xml"
OUT = ROOT / "models/humanoid_real_v2_soleplate_flex.xml"
FOOT_MASS = 0.03                    # kg moved from each shin into its foot body
SHINS = {"0003_2": "R", "0003_5": "L"}
ZETA_PER_K = 0.177 / 8              # damping per unit stiffness: zeta ~0.1 of the body swing at K 8


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fmt(v):
    return " ".join(f"{x:.6f}" for x in v)


def build(k, c, src=SRC):
    text = Path(src).read_text()
    base = '<joint name="base_joint" type="free" />'
    assert base in text, "free joint line changed"
    text = text.replace(base, '<joint name="base_joint" type="free" damping="0" armature="0" />'
                              '  <!-- the default class must not drag the floating base -->')
    for shin, side in SHINS.items():
        m = re.search(rf'<body name="{shin}" pos="[^"]*">\s*<inertial pos="([^"]*)" mass="([^"]*)" fullinertia="([^"]*)" />', text)
        com, mass = np.array(m.group(1).split(), float), float(m.group(2))
        g = re.search(r'( *)<geom class="collision" type="box" pos="([^"]*)" quat="([^"]*)" size="0\.00250 0\.08000 0\.04000" />[^\n]*',
                      text[m.end():])
        indent, gpos, gquat = g.group(1), np.array(g.group(2).split(), float), np.array(g.group(3).split(), float)
        R = quat_mat(gquat / np.linalg.norm(gquat))
        normal = R[:, 0] if R[2, 0] > 0 else -R[:, 0]          # sole normal, pointing up the shin
        pivot = gpos + 0.0025 * normal                         # top-centre of the sole box
        axis = R[:, 2]                                         # the foot's lateral axis
        new_mass = mass - FOOT_MASS                            # total mass and COM unchanged
        new_com = (mass * com - FOOT_MASS * gpos) / new_mass
        I_foot = FOOT_MASS / 12 * np.array([0.16**2 + 0.08**2, 0.005**2 + 0.08**2, 0.005**2 + 0.16**2])
        seg = text[m.end():m.end() + g.end()]
        # the foot mesh moves too: model_spec finds the feet as the bodies owning the lowest geoms
        vm = re.search(r'( *)<geom class="visual" mesh="(rightfoot_\d)" pos="([^"]*)" quat="([^"]*)" />\n', seg)
        vpos = np.array(vm.group(3).split(), float)
        body = (f'{indent}<body name="foot_{side}" pos="{fmt(pivot)}">\n'
                f'{indent}  <inertial pos="{fmt(gpos - pivot)}" quat="{fmt(gquat)}" mass="{FOOT_MASS}" '
                f'diaginertia="{" ".join(f"{x:.6e}" for x in I_foot * 1.001)}" />\n'
                f'{indent}  <joint name="ankle_flex_{side}" pos="0 0 0" axis="{fmt(axis)}" range="-0.5 0.5" '
                f'stiffness="{k:g}" damping="{c:g}" armature="0.0001" actuatorfrcrange="-100 100" />\n'
                f'{indent}  <geom class="visual" mesh="{vm.group(2)}" pos="{fmt(vpos - pivot)}" quat="{vm.group(4)}" />\n'
                f'{indent}  <geom class="collision" type="box" pos="{fmt(gpos - pivot)}" quat="{g.group(3)}" '
                f'size="0.00250 0.08000 0.04000" />\n'
                f'{indent}</body>  <!-- servo play, lumped at the foot (scripts/debug/make_flex_model.py) -->')
        seg = seg.replace(vm.group(0), "")
        seg = seg[:len(seg) - len(g.group(0))] + body
        text = text[:m.end()] + seg + text[m.end() + g.end():]
        text = text.replace(m.group(0), m.group(0).replace(f'pos="{m.group(1)}" mass="{m.group(2)}"',
                                                             f'pos="{fmt(new_com)}" mass="{new_mass:.6f}"'), 1)
    # keyframe: each new hinge sits right after its leg's knee in qpos order
    km = re.search(r'<key name="home" qpos="([^"]*)" />', text)
    q = km.group(1).split()
    q = q[:7 + 4] + ["0"] + q[7 + 4:7 + 8] + ["0"] + q[7 + 8:]
    return text.replace(km.group(0), f'<key name="home" qpos="{" ".join(q)}" />')


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--k", type=float, default=7.0, help="foot-hinge stiffness per leg, N*m/rad")
    p.add_argument("--c", type=float, default=None, help="damping per leg (default scales with k)")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    c = a.k * ZETA_PER_K if a.c is None else a.c
    Path(a.out).write_text(build(a.k, c))
    print(f"wrote {a.out}  (k {a.k:g}, c {c:.4f} per leg; floating base undamped)")


if __name__ == "__main__":
    main()
