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

        The half-duplex bus occasionally drops/garbles a reply. VALIDATE the full reply
        (id echo, length, checksum) and retry on any mismatch: header-only acceptance let
        corrupted position bytes through on a large fraction of frames (7/27 held-seizure
        logs: 76% of frames implied physically impossible joint speeds), and those garbage
        positions fed the obs directly. The error byte (resp[4]) is allowed to be nonzero
        (voltage/temp flags) -- the position data is still valid then."""
        for _ in range(retries + 1):
            self._ser.reset_input_buffer()
            self._send(servo_id, 0x02, [REG_PRESENT_POSITION, 0x02])
            resp = self._ser.read(8)
            if (len(resp) == 8 and resp[0] == 0xFF and resp[1] == 0xFF
                    and resp[2] == servo_id and resp[3] == 0x04
                    and resp[7] == ((~(resp[2] + resp[3] + resp[4] + resp[5] + resp[6])) & 0xFF)):
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

    def read_reg(self, servo_id: int, reg: int, nbytes: int = 1, retries: int = 2):
        """Read nbytes from a control-table register -> list[int] (low addr first) or None.
        Response packet: FF FF id len err data... checksum -> data starts at byte 5."""
        for _ in range(retries + 1):
            self._ser.reset_input_buffer()
            self._send(servo_id, 0x02, [reg, nbytes])
            resp = self._ser.read(6 + nbytes)
            if (len(resp) == 6 + nbytes and resp[0] == 0xFF and resp[1] == 0xFF
                    and resp[2] == servo_id
                    and resp[-1] == ((~sum(resp[2:-1])) & 0xFF)):
                return list(resp[5:5 + nbytes])
        return None

    def write_reg(self, servo_id: int, reg: int, values):
        """Write one or more bytes to a control-table register (values: int or list[int])."""
        if isinstance(values, int):
            values = [values]
        self._send(servo_id, 0x03, [reg, *[int(v) & 0xFF for v in values]], drain_ack=True)

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
        # Bounded retry, at CONNECT ONLY. On 9/26 the very first bank-select write failed with
        # EIO and killed a deploy before the policy ran; a scan minutes later found the IMU
        # answering (WHO_AM_I 0xEA) and 3000 back-to-back frame reads with zero errors. A
        # startup transient -- most likely the IMU not yet up when the script first spoke --
        # should cost a fraction of a second, not the attempt. Mid-run reads deliberately do
        # NOT retry: a fault there must surface, and the loop's shutdown path handles it.
        last = None
        for attempt in range(5):
            try:
                self._bus.write_byte_data(self.addr, self.REG_BANK_SEL, 0x00)  # bank 0
                time.sleep(0.01)
                who = self._bus.read_byte_data(self.addr, 0x00)                 # WHO_AM_I
                if who != 0xEA:
                    raise OSError(f"WHO_AM_I 0x{who:02X} at 0x{self.addr:02X}, expected 0xEA")
                self._bus.write_byte_data(self.addr, self.PWR_MGMT_1, 0x01)    # wake, auto clock
                time.sleep(0.05)
                if attempt:
                    print(f"[imu] connected on attempt {attempt + 1} (startup transient)")
                return self
            except OSError as e:
                last = e
                time.sleep(0.2)
        raise OSError(f"IMU not answering on I2C bus {self.busnum} at 0x{self.addr:02X} after 5 "
                      f"tries ({last}). Check its power and SDA/SCL wiring.")

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

    def calibrate_gyro_bias(self, seconds: float = 2.0, hz: float = 100.0,
                            tries: int = 5, max_tilt_deg: float = 1.0):
        """Gyro zero offset, measured only over a window in which the robot did not ROTATE.

        WHY THE GATE. The offset is the mean gyro reading over the window, so any real rotation
        during it is baked in as "zero" and then subtracted from every reading for the rest of
        the run. deploy_standing calibrates right after the ramp to home, while the robot is
        being held and settling. On 9/24 and 9/26 that left +0.079 and +0.133 rad/s of false
        pitch rate -- 2.6x and 4.4x the +-0.03 rad/s bias the policy was trained against --
        while the true offset measured on a still robot is 0.008, and this same calibration
        leaves 0.0001 when the robot is still. 175M stood on 9/24 and fell forward on 9/26 with
        the only measured difference being how still it was held for those 1.5 s.

        The accelerometer sees rotation independently of the gyro, so the window is rejected
        if the tilt it reports moved by more than max_tilt_deg between the start and end. Hand
        tremor with no net rotation passes, and averages out of the mean anyway. A 1 deg gate
        over a 1.5 s window caps the error at ~0.012 rad/s, inside the trained range.
        """
        n = max(20, int(seconds * hz))
        edge = 10
        best = None
        for attempt in range(tries):
            acc = np.zeros(3, dtype=np.float64)
            pg_start, pg_end = [], []
            for i in range(n):
                acc += self.axis_remap @ self._read_gyro_rads()
                if i < edge:
                    pg_start.append(self.projected_gravity())
                elif i >= n - edge:
                    pg_end.append(self.projected_gravity())
                time.sleep(1.0 / hz)
            a, b = np.mean(pg_start, axis=0), np.mean(pg_end, axis=0)
            tilt = lambda g: np.degrees([np.arctan2(g[0], -g[2]), np.arctan2(g[1], -g[2])])  # noqa: E731
            moved = float(np.max(np.abs(tilt(b) - tilt(a))))
            bias = (acc / n).astype(np.float32)
            if best is None or moved < best[0]:
                best = (moved, bias)
            if moved <= max_tilt_deg:
                if attempt:
                    print(f"[imu] gyro calibrated on attempt {attempt + 1} "
                          f"(robot was still to {moved:.2f} deg)")
                self.gyro_bias = bias
                return self.gyro_bias
            print(f"[imu] robot rotated {moved:.1f} deg during gyro calibration -- hold it STILL "
                  f"(try {attempt + 1}/{tries})")
        moved, bias = best
        err = np.radians(moved) / (n / hz)
        print(f"[imu] WARNING: never still for {n / hz:.1f} s. Using the stillest window "
              f"({moved:.1f} deg of rotation, so up to ~{err:.3f} rad/s of false gyro rate; "
              f"the policy was trained for +-0.03). Consider restarting.")
        self.gyro_bias = bias
        return self.gyro_bias

    def close(self):
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
