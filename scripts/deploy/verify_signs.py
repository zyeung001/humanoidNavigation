#!/usr/bin/env python3
# verify_signs.py
"""
Bench tool (RUN ON THE PI): confirm, for each MJCF action index, (1) which physical servo
moves and (2) whether +sim_rad produces the physically intended direction. Converts the 13
`sign_verified: false` joints in config/joint_servo_map.yaml into tested entries, and resolves
the ambiguous arm-DOF labels by letting you see which joint actually moves.

SAFETY: moves ONE joint at a time, a small angle, slowly, with the rest held at home. Start
with the robot supported/hanging so a wrong sign can't tip it. Hinges already verified (idx
3,7,13,16) are skipped unless you pass --include-verified.

Per joint it commands home -> home+delta -> home-delta -> home, then asks:
  [y] correct joint moved the intended + direction   -> keep sign, mark verified
  [r] correct joint moved, but reversed              -> FLIP sign, mark verified
  [w] WRONG joint moved                              -> leave unverified, flag servo_id/DOF
  [s] skip   [q] quit

Usage:
  python scripts/deploy/verify_signs.py                 # test all unverified joints (interactive)
  python scripts/deploy/verify_signs.py --only 2 6      # just these indices
  python scripts/deploy/verify_signs.py --delta-rad 0.1 --speed 200
  python scripts/deploy/verify_signs.py --apply         # write results back into the yaml
  python scripts/deploy/verify_signs.py --dry-run       # no hardware; print the plan
"""

from __future__ import annotations
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_real_map import SimRealMap, DEFAULT_MAP  # noqa: E402


def ramp_to(bus, servo_ids, start_units, goal_units, steps=40, dt=0.02, speed=150):
    start = np.asarray(start_units, dtype=np.float32)
    goal = np.asarray(goal_units, dtype=np.float32)
    for k in range(1, steps + 1):
        u = (start + (goal - start) * k / steps).round().astype(int)
        bus.write_all(servo_ids, u, speed=speed)
        time.sleep(dt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--only", type=int, nargs="*", default=None, help="action indices to test")
    p.add_argument("--include-verified", action="store_true")
    p.add_argument("--delta-rad", type=float, default=0.12, help="probe amplitude (sim rad)")
    p.add_argument("--speed", type=int, default=150)
    p.add_argument("--apply", action="store_true", help="write sign/sign_verified back into the yaml")
    p.add_argument("--dry-run", action="store_true", help="no hardware; just print the plan")
    args = p.parse_args()

    m = SimRealMap(args.map)

    if args.only is not None:
        targets = args.only
    else:
        targets = list(range(m.n)) if args.include_verified else m.unverified_idxs()
    print(f"Joints to test: {targets}\n")

    bus = None
    home_units = m.rad_to_units(m.default_joint_pos)  # nominal pose in units, index order
    if not args.dry_run:
        from hardware import ServoBus  # noqa: E402
        bus = ServoBus().connect()
        bus.set_torque(m.servo_ids, True)
        print("Holding all joints at home pose...")
        cur = bus.read_all(m.servo_ids)
        ramp_to(bus, m.servo_ids, cur, home_units, speed=args.speed)
        time.sleep(0.5)

    results: dict[int, dict] = {}
    try:
        for idx in targets:
            j = m.joints[idx]
            print("\n" + "=" * 70)
            print(f"TEST {m.describe(idx)}")
            print(f"  expect: physical servo {j.servo_id} ({j.dof}) moves on +{args.delta_rad} rad")
            if args.dry_run:
                u_plus = m.rad_to_units(_one_hot(m, idx, +args.delta_rad))[idx]
                u_minus = m.rad_to_units(_one_hot(m, idx, -args.delta_rad))[idx]
                print(f"  [dry-run] home={home_units[idx]} +d->{u_plus} -d->{u_minus}")
                continue

            base = home_units.copy()
            # +delta
            tgt = base.copy()
            tgt[idx] = m.rad_to_units(_one_hot(m, idx, +args.delta_rad))[idx]
            ramp_to(bus, [j.servo_id], [base[idx]], [tgt[idx]], speed=args.speed)
            time.sleep(0.4)
            # back to home, then -delta
            ramp_to(bus, [j.servo_id], [tgt[idx]], [base[idx]], speed=args.speed)
            tgt[idx] = m.rad_to_units(_one_hot(m, idx, -args.delta_rad))[idx]
            ramp_to(bus, [j.servo_id], [base[idx]], [tgt[idx]], speed=args.speed)
            time.sleep(0.4)
            ramp_to(bus, [j.servo_id], [tgt[idx]], [base[idx]], speed=args.speed)

            ans = input("  [y]correct  [r]reversed  [w]wrong joint  [s]skip  [q]quit > ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            if ans == "y":
                results[idx] = {"sign": j.sign, "sign_verified": True}
            elif ans == "r":
                results[idx] = {"sign": -j.sign, "sign_verified": True}
            elif ans == "w":
                results[idx] = {"sign": j.sign, "sign_verified": False, "note": "WRONG JOINT - check servo_id/DOF"}
    finally:
        if bus is not None:
            print("\nReturning to home and disabling torque...")
            cur = bus.read_all(m.servo_ids)
            ramp_to(bus, m.servo_ids, cur, home_units, speed=args.speed)
            bus.set_torque(m.servo_ids, False)
            bus.close()

    # ---- report ----
    print("\n" + "=" * 70 + "\nRESULTS")
    for idx in targets:
        if idx in results:
            r = results[idx]
            tag = "VERIFIED" if r["sign_verified"] else "FLAGGED"
            print(f"  idx {idx:2d} {m.joints[idx].dof:<18} sign {r['sign']:+d}  {tag} {r.get('note','')}")
        else:
            print(f"  idx {idx:2d} {m.joints[idx].dof:<18} (untested)")

    if args.apply and results:
        _apply_to_yaml(args.map, results)
        print(f"\nWrote verified signs into {args.map}")
    elif results:
        print("\n(use --apply to write these back into the yaml)")


def _one_hot(m: SimRealMap, idx: int, delta: float) -> np.ndarray:
    """default pose with a single joint nudged by delta sim-rad."""
    v = m.default_joint_pos.copy()
    v[idx] = v[idx] + delta
    return v


def _apply_to_yaml(path: str, results: dict[int, dict]):
    """In-place edit of the per-joint lines, preserving comments/formatting.
    Each joint is one line:  - {idx: N, ..., sign: S, sign_verified: B, ...}"""
    text = Path(path).read_text().splitlines()
    out = []
    for line in text:
        mobj = re.search(r"idx:\s*(\d+)\b", line)
        if mobj and line.lstrip().startswith("- {") and int(mobj.group(1)) in results:
            r = results[int(mobj.group(1))]
            line = re.sub(r"sign:\s*[+-]?\d+", f"sign: {r['sign']:+d}", line)
            line = re.sub(r"sign_verified:\s*\w+", f"sign_verified: {str(r['sign_verified']).lower()}", line)
        out.append(line)
    Path(path).write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
