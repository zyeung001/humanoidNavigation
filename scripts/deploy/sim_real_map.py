#!/usr/bin/env python3
# sim_real_map.py
"""
Shared sim <-> hardware joint map and unit conversions for the real humanoid.

Single source of truth = config/joint_servo_map.yaml (built from humanoid_real_v2.xml).
Both verify_signs.py and deploy_standing.py import this so the index/sign/limit logic
lives in exactly one place.

Index convention: action[i] (policy output) and qpos[7+i] (obs joint angle) share the
SAME index i, which is also this module's joint order (MJCF actuator == joint order).

Unit conversion (per joint i):
    units = center[i] + sign[i] * sim_rad * units_per_rad        # rad -> servo units
    sim_rad = sign[i] * (units - center[i]) / units_per_rad      # servo units -> rad
center defaults to the global servo_center (512), but a joint may override it with a
per-joint `center` field when its physical-straight neutral differs from 512 (e.g.
waist_roll measured at 485). units_per_rad ~= 195. SCS servos are 10-bit (0..1023).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml

DEFAULT_MAP = Path(__file__).resolve().parents[2] / "config" / "joint_servo_map.yaml"


@dataclass
class Joint:
    idx: int
    mjcf: str
    dof: str
    servo_id: int
    sign: int
    sign_verified: bool
    servo_limit: tuple[int, int]
    sim_range: tuple[float, float]
    center: int


class SimRealMap:
    def __init__(self, path: str | Path = DEFAULT_MAP):
        with open(path) as f:
            d = yaml.safe_load(f)
        self.units_per_rad = float(d["units_per_rad"])
        self.center = int(d["servo_center"])
        self.default_joint_pos = np.asarray(d["default_joint_pos"], dtype=np.float32)
        self.joints: list[Joint] = []
        for j in sorted(d["joints"], key=lambda x: x["idx"]):
            self.joints.append(Joint(
                idx=j["idx"], mjcf=j["mjcf"], dof=j["dof"], servo_id=j["servo_id"],
                sign=int(j["sign"]), sign_verified=bool(j["sign_verified"]),
                servo_limit=tuple(j["servo_limit"]), sim_range=tuple(j["sim_range"]),
                center=int(j.get("center", self.center)),
            ))
        self.n = len(self.joints)
        assert self.n == len(self.default_joint_pos) == 17, "expected 17 joints"
        # vectorized views, in index order
        self.servo_ids = np.array([j.servo_id for j in self.joints], dtype=int)
        self.signs = np.array([j.sign for j in self.joints], dtype=np.float32)
        self.centers = np.array([j.center for j in self.joints], dtype=np.float32)
        self.lim_lo = np.array([j.servo_limit[0] for j in self.joints], dtype=np.float32)
        self.lim_hi = np.array([j.servo_limit[1] for j in self.joints], dtype=np.float32)
        self.range_lo = np.array([j.sim_range[0] for j in self.joints], dtype=np.float32)
        self.range_hi = np.array([j.sim_range[1] for j in self.joints], dtype=np.float32)
        self.verified = np.array([j.sign_verified for j in self.joints], dtype=bool)

    # ---- conversions (vectorized over the 17 joints) ----
    def rad_to_units(self, sim_rad: np.ndarray) -> np.ndarray:
        sim_rad = np.clip(np.asarray(sim_rad, dtype=np.float32), self.range_lo, self.range_hi)
        units = self.centers + self.signs * sim_rad * self.units_per_rad
        return np.clip(units, self.lim_lo, self.lim_hi).round().astype(int)

    def units_to_rad(self, units: np.ndarray) -> np.ndarray:
        units = np.asarray(units, dtype=np.float32)
        return (self.signs * (units - self.centers) / self.units_per_rad).astype(np.float32)

    def unverified_idxs(self) -> list[int]:
        return [j.idx for j in self.joints if not j.sign_verified]

    def describe(self, idx: int) -> str:
        j = self.joints[idx]
        v = "verified" if j.sign_verified else "UNVERIFIED"
        return (f"idx {j.idx:2d}  {j.mjcf:<12} {j.dof:<18} servo {j.servo_id:2d}  "
                f"sign {j.sign:+d} ({v})  limit {j.servo_limit}")


if __name__ == "__main__":
    m = SimRealMap()
    print(f"Loaded {m.n} joints, center={m.center}, units/rad={m.units_per_rad}")
    for i in range(m.n):
        print(m.describe(i))
    print("\nUnverified-sign indices (need bench test):", m.unverified_idxs())
