#!/usr/bin/env python3
# hardware.py
"""
Hardware seam: servo bus + IMU drivers for the Raspberry Pi.

>>> THIS IS THE ONE FILE TO RECONCILE WITH YOUR EXISTING Pi DRIVER. <<<
The servo control code (set_limits.py / home.json / the SCS SDK calls) lives on the Pi,
not in this repo, so the calls below are written against the standard Waveshare/Feetech
`scservo_sdk` (SCSCL protocol for SCS-series servos) and `smbus2` for the ICM-20948.
If your Pi uses different call signatures, change them HERE only — the rest of the deploy
code talks to ServoBus / IMU, not the SDK.

All hardware imports are lazy so this module imports fine on the dev machine (Windows)
for logic/unit testing; they only fail if you actually open the bus without the libs.
"""

from __future__ import annotations
import time
import numpy as np

# Memory project-servo-mapping: /dev/ttyAMA0, 1000000 baud, Waveshare driver board.
DEFAULT_PORT = "/dev/ttyAMA0"
DEFAULT_BAUD = 1_000_000


class ServoBus:
    """Thin wrapper over scservo_sdk SCSCL. Positions are in raw servo units (0..1023)."""

    def __init__(self, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD):
        self.port, self.baud = port, baud
        self._ph = None   # PortHandler
        self._pk = None   # scscl packet handler

    def connect(self):
        # Lazy import so the dev machine can import this module without the SDK.
        from scservo_sdk import PortHandler, scscl  # type: ignore
        self._ph = PortHandler(self.port)
        if not self._ph.openPort():
            raise RuntimeError(f"failed to open {self.port}")
        if not self._ph.setBaudRate(self.baud):
            raise RuntimeError(f"failed to set baud {self.baud}")
        self._pk = scscl(self._ph)
        return self

    # ---- per-servo ops (servo_id is the physical bus ID 1..17) ----
    def read_pos(self, servo_id: int) -> int:
        pos, _, comm, err = self._pk.ReadPos(servo_id)
        if comm != 0 or err != 0:
            raise IOError(f"ReadPos id={servo_id} comm={comm} err={err}")
        return int(pos)

    def write_pos(self, servo_id: int, units: int, speed: int = 0, accel: int = 0):
        # SCSCL.WritePos(id, position, time, speed). time=0 -> use speed.
        comm, err = self._pk.WritePos(servo_id, int(units), 0, int(speed))
        if comm != 0 or err != 0:
            raise IOError(f"WritePos id={servo_id} comm={comm} err={err}")

    def read_all(self, servo_ids) -> np.ndarray:
        return np.array([self.read_pos(i) for i in servo_ids], dtype=np.float32)

    def write_all(self, servo_ids, units, speed: int = 0):
        for sid, u in zip(servo_ids, np.asarray(units).tolist()):
            self.write_pos(int(sid), int(u), speed=speed)

    def set_torque(self, servo_ids, enable: bool):
        for sid in servo_ids:
            try:
                self._pk.write1ByteTxRx(int(sid), 40, 1 if enable else 0)  # reg 40 = torque enable
            except Exception:
                pass

    def close(self):
        if self._ph is not None:
            try:
                self._ph.closePort()
            except Exception:
                pass


class IMU:
    """ICM-20948 accel+gyro over I2C (memory project-sensor-bringup: 0x68, NCS->3.3V for I2C).

    Returns vectors already remapped into the SIM BASE FRAME via `axis_remap` (a 3x3 signed
    permutation you calibrate once), so proj_grav / ang_vel match what standing_env computed.
    """

    def __init__(self, addr: int = 0x68, axis_remap: np.ndarray | None = None):
        self.addr = addr
        self._bus = None
        # Identity by default. MUST be calibrated to the physical IMU mounting (handoff item #2).
        self.axis_remap = np.eye(3, dtype=np.float32) if axis_remap is None else np.asarray(axis_remap, np.float32)
        self.gyro_bias = np.zeros(3, dtype=np.float32)

    def connect(self):
        from smbus2 import SMBus  # type: ignore
        self._bus = SMBus(1)
        # TODO: wake from sleep + set ranges per ICM-20948 datasheet (PWR_MGMT_1, etc.).
        # Left to your existing bring-up code; values below assume it's already configured.
        return self

    def _read_accel_raw(self) -> np.ndarray:
        # TODO: replace with your ICM-20948 register reads. Must return g-units, sensor frame.
        raise NotImplementedError("wire to your ICM-20948 accel registers")

    def _read_gyro_raw(self) -> np.ndarray:
        # TODO: replace with your ICM-20948 register reads. Must return rad/s, sensor frame.
        raise NotImplementedError("wire to your ICM-20948 gyro registers")

    def projected_gravity(self) -> np.ndarray:
        """proj_grav in sim base frame. At rest accel reads +1g UP = -proj_grav, so feed
        -normalize(accel) (handoff). Valid when quasi-static (true for standing)."""
        a = self._read_accel_raw().astype(np.float32)
        n = np.linalg.norm(a)
        a = a / n if n > 1e-6 else a
        return (self.axis_remap @ (-a)).astype(np.float32)

    def angular_velocity(self) -> np.ndarray:
        """base_ang_vel (rad/s) in sim base frame = remapped gyro minus bias."""
        g = self._read_gyro_raw().astype(np.float32)
        return (self.axis_remap @ g - self.gyro_bias).astype(np.float32)

    def calibrate_gyro_bias(self, seconds: float = 2.0, hz: float = 100.0):
        """Average the (remapped) gyro while the robot is held still."""
        n = max(1, int(seconds * hz))
        acc = np.zeros(3, dtype=np.float32)
        for _ in range(n):
            acc += self.axis_remap @ self._read_gyro_raw().astype(np.float32)
            time.sleep(1.0 / hz)
        self.gyro_bias = acc / n
        return self.gyro_bias

    def close(self):
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
