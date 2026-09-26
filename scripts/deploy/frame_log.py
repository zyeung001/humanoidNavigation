#!/usr/bin/env python3
# frame_log.py  --  RUN ON THE PI (imported by the deploy scripts)
"""Per-frame recorder for the real robot: what was commanded, what the servos did.

Every hardware finding on this robot so far came from reading a text log after the
fact -- and several were missed for weeks because the data needed to see them was
never written down. The arm ran for days against a frozen observation and measured
nothing; the servos sat at the wrong P gain for a month; a stale policy file was
deployed twice. This module exists so that every run leaves a complete record with
no extra effort at the robot.

WHAT IS RECORDED, and why each column earns its place:
  - u_meas  raw encoder units, straight off the bus. The rawest thing there is.
  - u_cmd   raw units actually written, AFTER rate-limit and firmware clipping. The
            gap between u_cmd and u_meas is the servo's tracking error -- the
            fundamental quantity behind every oscillation on this machine.
  - q_cmd   the policy's applied action in sim radians, BEFORE unit quantization.
            Not recoverable from u_cmd (rounding and clamping destroy it), so it is
            stored separately rather than derived.
  - jvel_f  the velocity the policy actually saw, which is not the true joint speed
            (encoder quantization makes it a 0.205 rad/s staircase) and cannot be
            recomputed later from positions alone.
  - pg/av   pelvis attitude and rates, already in the observation frame.
  - fsr_v   RAW divider volts, never a force. FSR calibration is uncertain and WILL
            change; storing newtons computed from today's guess would freeze every
            old log at today's error. Raw volts plus the parameters in the sidecar
            can be re-derived under any later calibration.

STEP 3 (live pose view) IS BUILT ON THIS, so two things are deliberate:
  1. A row plus the servo map is sufficient to pose the robot -- `pose_from_row()`
     returns absolute joint angles and pelvis attitude, which is exactly what a
     viewer or a sim replay needs. Nothing else has to be reconstructed.
  2. `add_sink()` attaches extra consumers (a UDP publisher, a live window) that run
     on the WRITER thread, never the control thread. A slow or crashed viewer can
     therefore never stall the 40 Hz loop.

COST IN THE CONTROL LOOP: `log()` builds a tuple and does one queue put -- a few
microseconds. All formatting and disk I/O happen on the writer thread. This matters
because the standing loop already runs ~21 ms of its 25 ms budget.

VERSION PROVENANCE: the sidecar records a git SHA, but note the Pi is synced by scp
rather than git, so its SHA can describe a tree that is not what is running. The md5
of every file that actually matters is recorded alongside it, and those are the ones
to trust.
"""
from __future__ import annotations

import csv
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

# Columns that are not per-joint, in row order.
SCALAR_COLUMNS = [
    "t_wall",       # unix epoch seconds, the clock every stream is aligned on
    "t_rel",        # seconds since the run started
    "step",         # control-loop iteration
    "loop_ms",      # realized period, so a stalled frame is visible in the data
    "busy_ms",      # sense+predict+write, before the sleep
    "pg_x", "pg_y", "pg_z",
    "av_x", "av_y", "av_z",
    "upright_cos",
    "rej_total",    # cumulative frames with a rejected servo read
    "fsr_v0", "fsr_v1",
    "fsr_age_ms",   # staleness of the FSR sample; it is sampled off-thread
]
JOINT_BLOCKS = ["u_meas", "u_cmd", "q_cmd", "jvel_f"]


def build_columns(dofs):
    """Full ordered column list for a robot with these joint names."""
    cols = list(SCALAR_COLUMNS)
    for block in JOINT_BLOCKS:
        cols += [f"{block}.{d}" for d in dofs]
    return cols


def file_md5(path):
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def git_sha():
    """Best-effort. On the Pi this describes the git tree, NOT necessarily the files
    running -- the repo is kept in sync by scp. Trust the md5s instead."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, cwd=str(Path(__file__).resolve().parents[2]))
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class FrameLogger:
    """Buffered CSV writer. The control loop calls log(); a daemon thread does the I/O.

    Rows are dropped rather than blocking if the writer falls behind -- a logger must
    never be able to stall the control loop. Drops are counted and reported, because a
    silently truncated log is worse than a short one.
    """

    def __init__(self, path, dofs, meta=None, queue_size=20000, flush_every=40):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = build_columns(dofs)
        self.n_joints = len(dofs)
        self._q = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._sinks = []
        self._latest = None          # newest row, for in-process live consumers
        self.dropped = 0
        self.written = 0
        self.t0 = time.time()
        self.flush_every = int(flush_every)

        meta = dict(meta or {})
        meta.update(schema_version=SCHEMA_VERSION, columns=self.columns, dofs=list(dofs),
                    started_unix=self.t0, started_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    git_sha=git_sha(), git_sha_caveat="Pi syncs by scp; trust file_md5s",
                    argv=list(sys.argv))
        self.meta_path = self.path.with_suffix(".meta.json")
        self.meta_path.write_text(json.dumps(meta, indent=2, default=str))

        self._fh = open(self.path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.columns)
        self._thread = threading.Thread(target=self._run, name="frame-log", daemon=True)
        self._thread.start()

    def log(self, *, t_wall, step, loop_ms, busy_ms, pg, av, upright_cos, rej_total,
            u_meas, u_cmd, q_cmd, jvel_f, fsr=(float("nan"),) * 2, fsr_age_ms=float("nan")):
        """Called from the control loop. Must stay cheap: tuple build + one queue put."""
        row = (t_wall, t_wall - self.t0, step, loop_ms, busy_ms,
               pg[0], pg[1], pg[2], av[0], av[1], av[2], upright_cos, rej_total,
               fsr[0], fsr[1], fsr_age_ms,
               *u_meas, *u_cmd, *q_cmd, *jvel_f)
        self._latest = row
        try:
            self._q.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def latest(self):
        """Newest row as a dict, for an in-process live view. None before the first log."""
        return None if self._latest is None else dict(zip(self.columns, self._latest))

    def add_sink(self, fn):
        """Attach an extra consumer, called as fn(row_tuple) on the WRITER thread.

        This is the seam Step 3 plugs into: a UDP publisher or live window can be
        added here without the control loop ever knowing it exists, and an exception
        inside a sink is swallowed so a broken viewer cannot take down a robot run.
        """
        self._sinks.append(fn)

    def _run(self):
        pending = 0
        while not (self._stop.is_set() and self._q.empty()):
            try:
                row = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            # t_wall is a unix epoch (~1.8e9) and must keep sub-millisecond resolution, so
            # it gets fixed decimals; 6 significant figures would round it to ~1000 s and
            # silently destroy the one column every other stream is aligned on.
            self._writer.writerow([f"{row[0]:.6f}"] + [_fmt(v) for v in row[1:]])
            self.written += 1
            pending += 1
            if pending >= self.flush_every:
                self._fh.flush()
                pending = 0
            for fn in self._sinks:
                try:
                    fn(row)
                except Exception:      # a viewer must never kill a hardware run
                    pass
        self._fh.flush()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5.0)
        try:
            self._fh.close()
        except OSError:
            pass
        meta = json.loads(self.meta_path.read_text())
        meta.update(rows_written=self.written, rows_dropped=self.dropped,
                    ended_unix=time.time(), duration_s=time.time() - self.t0)
        self.meta_path.write_text(json.dumps(meta, indent=2, default=str))
        return self.written, self.dropped


def _fmt(v):
    """Compact but lossless-enough: 6 significant figures beats 17 digits of float noise
    when the file has ~80 columns at 40 Hz. numpy scalars are matched explicitly --
    np.float32 is NOT a subclass of float, so a plain isinstance(v, float) test would let
    every joint angle through unformatted and write a different precision per column."""
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    if isinstance(v, np.integer):
        return int(v)
    return v


class FsrSampler:
    """Samples the ADS1115 on its own thread so the control loop never waits on I2C.

    A conversion is ~1.2 ms at 860 SPS and switching the mux costs a full conversion,
    so polling two channels inline would spend several ms of a 25 ms budget that is
    already ~21 ms full. The loop reads whatever the latest sample is and records how
    stale it was, which is honest and costs nothing.
    """

    def __init__(self, channels=(0, 1), addr=0x48, bus=1, hz=50.0):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fsr_monitor import _open_bus, read_channel      # noqa: PLC0415
        self._read = read_channel
        self._bus = _open_bus(bus)
        self.addr = addr
        self.channels = tuple(channels)
        self.period = 1.0 / max(hz, 1e-3)
        self.volts = [float("nan")] * len(self.channels)
        self.t_sample = 0.0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fsr", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.volts = [self._read(self._bus, self.addr, c) for c in self.channels]
                self.t_sample = time.time()
            except Exception:
                self.errors += 1
            time.sleep(self.period)

    def read(self):
        """Latest sample and its age in ms."""
        age = (time.time() - self.t_sample) * 1000.0 if self.t_sample else float("nan")
        return list(self.volts), age

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


class UdpSink:
    """Broadcast each frame as a datagram so a viewer elsewhere can draw the robot live.

    UDP on purpose. The Pi has no MuJoCo -- deliberately, it runs numpy-only -- so the 3D
    view has to live on another machine, and the control loop must not care whether anyone
    is watching. A datagram that nobody receives costs one syscall and is dropped by the
    network; a TCP connection to a viewer that stalls or dies would back-pressure into the
    40 Hz loop. Losing frames only makes the picture stutter.

    Attach through FrameLogger.add_sink(), so the packing happens on the writer thread and
    never on the control thread.
    """

    MAGIC = 0xA7
    FMT = "<BIf3ff17f17f"      # magic, step, t_rel, proj_grav, upright_cos, u_meas, u_cmd

    def __init__(self, host, port, n_joints=17):
        import socket
        import struct
        self._struct = struct
        self._n = n_joints
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sent = 0
        self.errors = 0

    def __call__(self, row):
        # Row layout is SCALAR_COLUMNS then the joint blocks, in JOINT_BLOCKS order.
        n, s = self._n, len(SCALAR_COLUMNS)
        try:
            pkt = self._struct.pack(
                self.FMT, self.MAGIC, int(row[2]), float(row[1]),
                float(row[5]), float(row[6]), float(row[7]), float(row[11]),
                *[float(v) for v in row[s:s + n]],              # u_meas
                *[float(v) for v in row[s + n:s + 2 * n]])      # u_cmd
            self._sock.sendto(pkt, self._addr)
            self.sent += 1
        except OSError:
            self.errors += 1

    def close(self):
        self._sock.close()


def pose_from_row(row, servo_map):
    """Row (dict or Series) -> everything needed to draw or replay the robot.

    This is the Step 3 / replay entry point. Absolute joint angles come from the RAW
    encoder units through the same map the deploy loop used, so a viewer reproduces
    what the robot did rather than what it was told to do.

    Returns dict with:
      q_meas  (n,) absolute joint angles, sim radians, 0 = straight
      q_cmd   (n,) commanded absolute joint angles, sim radians
      proj_grav (3,) pelvis attitude in the observation frame
      t       seconds since the run started
    """
    dofs = [j.dof for j in servo_map.joints]
    u_meas = np.array([float(row[f"u_meas.{d}"]) for d in dofs])
    return {
        "t": float(row["t_rel"]),
        "q_meas": np.asarray(servo_map.units_to_rad(u_meas), dtype=np.float32),
        "q_cmd": np.array([float(row[f"q_cmd.{d}"]) for d in dofs], dtype=np.float32),
        "proj_grav": np.array([float(row["pg_x"]), float(row["pg_y"]), float(row["pg_z"])],
                              dtype=np.float32),
    }


def default_log_path(tag, log_dir=None):
    d = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    return d / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.csv"
