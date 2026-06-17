#!/usr/bin/env python3
# hardware.py
"""
Hardware seam: servo bus + IMU drivers for the Raspberry Pi.

Wired to the robot's actual drivers (reverse-engineered from the working bench scripts
~/move_servo.py, ~/read_positions.py, ~/stop_all.py, ~/read_imu.py):
  - Servos: raw Feetech-SCS half-duplex serial over pyserial (/dev/ttyAMA0 @ 1 Mbaud).
            NOT scservo_sdk. Packet = FF FF id len instr params... checksum.
  - IMU:    ICM-20948 over I2C (smbus) bus 1 @ 0x68, accel 0x2D / gyro 0x33.

All hardware imports are lazy so this module imports fine on the dev machine (Windows)
for logic/unit testing; they only fail if you actually open the bus without the libs.
"""

from __future__ import annotations
import time
import numpy as np

DEFAULT_PORT = "/dev/ttyAMA0"
DEFAULT_BAUD = 1_000_000

# Feetech SCS control-table registers
REG_TORQUE_ENABLE = 0x28
REG_GOAL_POSITION = 0x2A
REG_PRESENT_POSITION = 0x38


def _checksum(idv, length, instr, params):
    return (~(idv + length + instr + sum(params))) & 0xFF


class ServoBus:
    """Raw-serial Feetech SCS bus. Positions are servo units (0..1023)."""

    def __init__(self, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD):
        self.port, self.baud = port, baud
        self._ser = None

    def connect(self):
        import serial  # pyserial
        self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
        return self

    def _send(self, idv, instr, params, drain_ack=False):
        length = len(params) + 2
        cs = _checksum(idv, length, instr, params)
        self._ser.write(bytes([0xFF, 0xFF, idv, length, instr, *params, cs]))
        # Half-duplex bus: a write returns a 6-byte status packet. Draining it before the
        # next command prevents that reply from colliding with the next write (which dropped
        # a servo during batched homing). Mirrors the working bench scripts' move().
        if drain_ack:
            self._ser.read(6)

    def read_pos(self, servo_id: int, retries: int = 2):
        """Present position (units) or None after retries. (reg 0x38, 2 bytes)

        The half-duplex bus occasionally drops/garbles a reply; retry rather than feed a
        NaN into the obs. Each attempt is bounded by the serial read timeout."""
        for _ in range(retries + 1):
            self._ser.reset_input_buffer()
            self._send(servo_id, 0x02, [REG_PRESENT_POSITION, 0x02])
            resp = self._ser.read(8)
            if len(resp) >= 7 and resp[0] == 0xFF and resp[1] == 0xFF:
                return (resp[5] << 8) | resp[6]
        return None

    def write_pos(self, servo_id: int, units: int, speed: int = 0):
        u = int(units) & 0xFFFF
        s = int(speed) & 0xFFFF
        self._send(servo_id, 0x03,
                   [REG_GOAL_POSITION, (u >> 8) & 0xFF, u & 0xFF, (s >> 8) & 0xFF, s & 0xFF],
                   drain_ack=True)

    def read_all(self, servo_ids) -> np.ndarray:
        out = []
        for i in servo_ids:
            p = self.read_pos(int(i))
            out.append(float(p) if p is not None else np.nan)
        return np.array(out, dtype=np.float32)

    def write_all(self, servo_ids, units, speed: int = 0):
        for sid, u in zip(servo_ids, np.asarray(units).tolist()):
            self.write_pos(int(sid), int(u), speed=speed)

    def set_torque(self, servo_ids, enable: bool):
        for sid in servo_ids:
            self._send(int(sid), 0x03, [REG_TORQUE_ENABLE, 1 if enable else 0], drain_ack=True)

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass


class IMU:
    """ICM-20948 accel+gyro over I2C (bus 1, 0x68). Returns vectors already remapped into
    the SIM BASE FRAME via `axis_remap` (a 3x3 signed permutation you calibrate once)."""

    # ICM-20948 bank-0 registers (from ~/read_imu.py)
    REG_BANK_SEL = 0x7F
    PWR_MGMT_1 = 0x06
    ACCEL_XOUT_H = 0x2D
    GYRO_XOUT_H = 0x33
    ACC_LSB_PER_G = 16384.0     # +/-2g default
    GYRO_LSB_PER_DPS = 131.0    # +/-250 dps default

    def __init__(self, bus: int = 1, addr: int = 0x68, axis_remap: np.ndarray | None = None):
        self.busnum, self.addr = bus, addr
        self._bus = None
        # Identity by default. CALIBRATE to the physical IMU mounting (handoff item #2):
        # upright, projected_gravity() must read ~[0,0,-1].
        self.axis_remap = np.eye(3, dtype=np.float32) if axis_remap is None else np.asarray(axis_remap, np.float32)
        self.gyro_bias = np.zeros(3, dtype=np.float32)

    def connect(self):
        try:
            import smbus
        except ImportError:
            import smbus2 as smbus
        self._bus = smbus.SMBus(self.busnum)
        self._bus.write_byte_data(self.addr, self.REG_BANK_SEL, 0x00)  # bank 0
        time.sleep(0.01)
        self._bus.write_byte_data(self.addr, self.PWR_MGMT_1, 0x01)    # wake, auto clock
        time.sleep(0.05)
        return self

    @staticmethod
    def _s16(hi, lo):
        v = (hi << 8) | lo
        return v - 65536 if v & 0x8000 else v

    def _read_accel_g(self) -> np.ndarray:
        a = self._bus.read_i2c_block_data(self.addr, self.ACCEL_XOUT_H, 6)
        return np.array([self._s16(a[0], a[1]), self._s16(a[2], a[3]), self._s16(a[4], a[5])],
                        dtype=np.float32) / self.ACC_LSB_PER_G

    def _read_gyro_rads(self) -> np.ndarray:
        g = self._bus.read_i2c_block_data(self.addr, self.GYRO_XOUT_H, 6)
        dps = np.array([self._s16(g[0], g[1]), self._s16(g[2], g[3]), self._s16(g[4], g[5])],
                       dtype=np.float32) / self.GYRO_LSB_PER_DPS
        return np.deg2rad(dps)  # sim base_ang_vel (qvel) is rad/s

    def projected_gravity(self) -> np.ndarray:
        """proj_grav in sim base frame. At rest accel reads +1g UP = -proj_grav, so feed
        -normalize(accel) (handoff). Valid when quasi-static (true for standing)."""
        a = self._read_accel_g()
        n = np.linalg.norm(a)
        a = a / n if n > 1e-6 else a
        return (self.axis_remap @ (-a)).astype(np.float32)

    def angular_velocity(self) -> np.ndarray:
        return (self.axis_remap @ self._read_gyro_rads() - self.gyro_bias).astype(np.float32)

    def calibrate_gyro_bias(self, seconds: float = 2.0, hz: float = 100.0):
        n = max(1, int(seconds * hz))
        acc = np.zeros(3, dtype=np.float32)
        for _ in range(n):
            acc += self.axis_remap @ self._read_gyro_rads()
            time.sleep(1.0 / hz)
        self.gyro_bias = acc / n
        return self.gyro_bias

    def close(self):
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
