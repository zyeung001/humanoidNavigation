#!/usr/bin/env python3
# hardware.py
"""
Hardware seam: servo bus + IMU drivers for the Raspberry Pi.

Wired to the robot's actual drivers (reverse-engineered from the working bench scripts
~/move_servo.py, ~/read_positions.py, ~/stop_all.py, ~/read_imu.py):
  - Servos: raw Feetech-SCS half-duplex serial over pyserial (/dev/ttyAMA0 @ 1 Mbaud).
            NOT scservo_sdk. Packet = FF FF id len instr params... checksum.
            read_all() reads position, speed, load, voltage and temperature in ONE 8-byte read
            per servo (0x38-0x3F) and keeps the raw telemetry in last_telemetry for the frame
            log; ServoBus(telemetry=False) is the old position-only read.
  - IMU:    ICM-20948 over I2C (smbus) bus 1 @ 0x68, accel 0x2D / gyro 0x33. The raw accel
            behind each projected_gravity() is kept in last_accel_g (no second I2C read).

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
REG_PRESENT_POSITION = 0x38      # 2 bytes, HIGH byte first (read_pos, calibrate_centers' limits)
REG_PRESENT_SPEED = 0x3A         # 2 bytes, high first; bit 15 = direction (SCSCL)
REG_PRESENT_LOAD = 0x3C          # 2 bytes, high first; bit 10 = direction, 0..1000 magnitude
REG_PRESENT_VOLTAGE = 0x3E       # 1 byte, 0.1 V per unit
REG_PRESENT_TEMPERATURE = 0x3F   # 1 byte, deg C
# Position through temperature are contiguous, so one 8-byte read returns all five. That is the
# same single transaction per servo as the 2-byte position read, with 6 more reply bytes
# (~60 us at 1 Mbaud): ~1 ms per 17-servo frame, against the ~10 ms the reads already cost.
STATE_NBYTES = REG_PRESENT_TEMPERATURE - REG_PRESENT_POSITION + 1     # 8
TELEMETRY_FIELDS = ("spd_raw", "load_raw", "volt_raw", "temp_raw")

# How the RAW telemetry registers decode, recorded in every log's meta sidecar. The logs store
# the raw register value on purpose (as fsr_v stores raw volts): if any of this turns out wrong
# on this firmware, every old log can still be decoded the right way later.
TELEMETRY_DECODING = {
    "spd_raw": "reg 0x3A-0x3B, big-endian (same byte order as position). Sign-magnitude: "
               "bit 15 = direction, bits 0-14 = speed in encoder steps per second. Decode: "
               "v = raw & 0x7FFF; v = -v if raw & 0x8000.",
    "load_raw": "reg 0x3C-0x3D, big-endian. Sign-magnitude: bit 10 = direction, bits 0-9 = "
                "0..1000 in 0.1 % of max output (PWM duty, a torque PROXY, not newton-metres). "
                "Decode: v = raw & 0x3FF; v = -v if raw & 0x400.",
    "volt_raw": "reg 0x3E, 1 byte, 0.1 V per unit (e.g. 74 = 7.4 V).",
    "temp_raw": "reg 0x3F, 1 byte, degrees C.",
    "confidence": "Addresses and byte order: high -- the SCSCL control table (FT SCServo SDK "
                  "scscl.py: PRESENT_POSITION 56, SPEED 58, LOAD 60, VOLTAGE 62, TEMPERATURE 63) "
                  "and read_pos() / calibrate_centers.py both read this bus high byte first. "
                  "Sign bits and units: medium -- from the SDK, not yet checked on this robot "
                  "(push a joint each way with torque on and watch load_raw's bit 10 flip).",
}


def _checksum(idv, length, instr, params):
    return (~(idv + length + instr + sum(params))) & 0xFF


def release_on_hangup():
    """Make a dropped SSH session (SIGHUP) or a kill (SIGTERM) stop the run the way Ctrl-C does.

    Python's default for both is to exit on the spot, skipping every `finally`, so the servos
    kept holding the last pose at full torque. On 10/18 the laptop driving a run shut off and
    the robot was then handled while still stiff. This raises KeyboardInterrupt instead, so the
    caller's own ramp-and-release path runs. The terminal is gone after a hangup, so output is
    sent to /dev/null first: a print that raises would skip the torque-off (a console tee keeps
    writing its file and drops only the terminal). Later signals are ignored so a second one
    cannot interrupt the cleanup.
    """
    import os
    import signal
    import sys

    sigs = [s for s in (getattr(signal, "SIGHUP", None), getattr(signal, "SIGTERM", None)) if s is not None]

    def _stop(signum, frame):
        for s in sigs:
            signal.signal(s, signal.SIG_IGN)
        if signum == getattr(signal, "SIGHUP", None):
            # A console tee (frame_log.ConsoleTee) keeps recording to its file and only loses
            # the dead terminal: the shutdown after a hangup is exactly what its log is for.
            devnull = open(os.devnull, "w")
            for name in ("stdout", "stderr"):
                stream = getattr(sys, name)
                if hasattr(stream, "detach_terminal"):
                    stream.detach_terminal(devnull)
                else:
                    setattr(sys, name, devnull)
        raise KeyboardInterrupt

    for s in sigs:
        signal.signal(s, _stop)


def _reply_data(resp, servo_id: int, nbytes: int):
    """Data bytes of a READ reply, or None unless the WHOLE packet checks out.

    Reply = FF FF id len err data... checksum, with len = nbytes + 2. Every field is checked
    (header, id echo, length, checksum): header-only acceptance let corrupted bytes through on
    a large fraction of frames (7/27). The error byte is allowed to be nonzero -- the servo
    sets voltage/temperature flags there and the data is still valid."""
    if (len(resp) == 6 + nbytes and resp[0] == 0xFF and resp[1] == 0xFF
            and resp[2] == servo_id and resp[3] == nbytes + 2
            and resp[-1] == ((~sum(resp[2:-1])) & 0xFF)):
        return resp[5:5 + nbytes]
    return None


class ServoBus:
    """Raw-serial Feetech SCS bus. Positions are servo units (0..1023).

    telemetry=True (the default) makes read_all() fetch speed, load, voltage and temperature
    in the SAME read as the position, and leave them in `last_telemetry` for the frame log.
    False falls back to the old 2-byte position read (deploy --no-servo-telemetry), for the
    case where the longer replies turn out to cost the 40 Hz loop more than expected.
    """

    def __init__(self, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD, telemetry: bool = True):
        self.port, self.baud = port, baud
        self.telemetry = bool(telemetry)
        self._ser = None
        # (4, n) raw spd/load/volt/temp from the last read_all(), rows in TELEMETRY_FIELDS
        # order; NaN where a read failed or telemetry is off. None before the first read.
        self.last_telemetry = None

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
            d = _reply_data(self._ser.read(8), servo_id, 2)
            if d is not None:
                return (d[0] << 8) | d[1]
        return None

    def read_state(self, servo_id: int, retries: int = 2):
        """(position, speed, load, voltage, temperature) as RAW register values, or None.

        One 8-byte read from 0x38: the same single bus transaction as read_pos(), validated
        and retried the same way, so a garbled reply can never put a bad position into the
        obs just because the read got longer. Speed and load are the raw 16-bit words with
        their direction bits still in them; TELEMETRY_DECODING says how to sign them, which
        is left to whoever reads the log."""
        for _ in range(retries + 1):
            self._ser.reset_input_buffer()
            self._send(servo_id, 0x02, [REG_PRESENT_POSITION, STATE_NBYTES])
            d = _reply_data(self._ser.read(6 + STATE_NBYTES), servo_id, STATE_NBYTES)
            if d is not None:
                return ((d[0] << 8) | d[1], (d[2] << 8) | d[3], (d[4] << 8) | d[5], d[6], d[7])
        return None

    def write_pos(self, servo_id: int, units: int, speed: int = 0):
        u = int(units) & 0xFFFF
        s = int(speed) & 0xFFFF
        self._send(servo_id, 0x03,
                   [REG_GOAL_POSITION, (u >> 8) & 0xFF, u & 0xFF, (s >> 8) & 0xFF, s & 0xFF],
                   drain_ack=True)

    def read_all(self, servo_ids) -> np.ndarray:
        """Positions in units, NaN for a rejected read. With telemetry on, the same reads also
        fill last_telemetry, so the log gets speed/load/voltage/temperature for no extra bus
        transactions."""
        out = []
        tel = np.full((len(TELEMETRY_FIELDS), len(servo_ids)), np.nan, dtype=np.float32)
        for k, i in enumerate(servo_ids):
            if self.telemetry:
                st = self.read_state(int(i))
                if st is None:
                    out.append(np.nan)
                    continue
                out.append(float(st[0]))
                tel[:, k] = st[1:]
            else:
                p = self.read_pos(int(i))
                out.append(float(p) if p is not None else np.nan)
        self.last_telemetry = tel
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
        # The raw accelerometer (g, remapped, NOT normalized) behind the last projected_gravity()
        # call, kept so the frame log gets it without a second I2C read. projected_gravity()
        # normalizes the magnitude away, and the magnitude is what tells a quasi-static tilt
        # from a jolt: |a| far from 1 g means the robot is accelerating and pg is not gravity.
        self.last_accel_g = np.full(3, np.nan, dtype=np.float32)

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
        self.last_accel_g = (self.axis_remap @ a).astype(np.float32)
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
