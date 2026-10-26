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

SCHEMA 2 (10/25) adds what it took to explain the falls that schema 1 could only show:
  - acc_*   the raw accelerometer, g, in the sim base frame, NOT normalized. pg is a
            filtered unit vector and has lost the magnitude; |acc| far from 1 g says the
            robot was accelerating and pg was not gravity at that instant (a fall or a
            push looks like a tilt in pg alone).
  - spd_raw/load_raw/volt_raw/temp_raw  per servo, RAW register values from the same
            bus read as u_meas. Load is the only torque signal on this robot: a joint that
            is commanded, does not move, and shows load is on a stop or carrying the body;
            one that shows none is slipping or unpowered. Voltage sag under load and a
            heating servo explain the "it got worse after a minute" runs. Raw, for the
            same reason fsr_v is raw: the decoding (hardware.TELEMETRY_DECODING, copied
            into the meta) is from the SDK and not yet checked on this firmware.
  - q_prior the feedback law's own contribution to the command, rad, NaN with no law.
            q_cmd is the sum of policy + law + posture after clipping; this is the only
            way to see which of them moved a joint.
  - posture_lean / posture_zero / posture_corr  the posture controller's filtered lean,
            measured zero and knee correction, deg (NaN with it off). Its 10/25 run went
            unstable and the log could not show why: the internals were never written.
  - fsr_v0..fsr_v{N-1}  any number of channels over several ADS1115 boards; which board,
            pin, foot and position each one is lives in the meta (`fsr.channels`).

Schema-1 columns keep their names and meanings; readers look columns up BY NAME, so a
v1 log still reads (missing v2 columns are simply absent). The FSR columns now come
LAST because their count varies, which keeps the fixed scalars followed directly by the
joint blocks -- the layout UdpSink packs from.

SIDECARS next to <stem>.csv:
  <stem>.meta.json     configuration, provenance, decoding (written at start, completed
                       at close, so a run that dies still says what it was)
  <stem>.console.txt   everything the run printed, each line prefixed with unix time so
                       it lines up with t_wall (ConsoleTee)
  <stem>.events.jsonl  operator markers, one JSON object per line (written elsewhere)

STEP 3 (live pose view) IS BUILT ON THIS, so two things are deliberate:
  1. A row plus the servo map is sufficient to pose the robot -- `pose_from_row()`
     returns absolute joint angles and pelvis attitude, which is exactly what a
     viewer or a sim replay needs. Nothing else has to be reconstructed.
  2. `add_sink()` attaches extra consumers (a UDP publisher, a live window) that run
     on the WRITER thread, never the control thread. A slow or crashed viewer can
     therefore never stall the 40 Hz loop.

COST IN THE CONTROL LOOP: `log()` builds a tuple and does one queue put -- tens of
microseconds even at ~200 columns. All formatting and disk I/O happen on the writer
thread. This matters because the standing loop already runs ~21 ms of its 25 ms budget.

VERSION PROVENANCE: the sidecar records a git SHA, but note the Pi is synced by scp
rather than git, so its SHA can describe a tree that is not what is running. The md5
of every file that actually matters is recorded alongside it, and those are the ones
to trust.
"""
from __future__ import annotations

import atexit
import csv
import hashlib
import json
import platform
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
DEPLOY_DIR = Path(__file__).resolve().parent
NAN = float("nan")

# Columns that are not per-joint, in row order. Fixed length on purpose: UdpSink and the
# live viewer find the joint blocks at len(SCALAR_COLUMNS).
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
    "fsr_age_ms",   # staleness of the FSR sample; it is sampled off-thread
    "acc_x", "acc_y", "acc_z",                          # v2: raw accel, g, remapped
    "posture_lean", "posture_zero", "posture_corr",     # v2: posture controller, deg
]
# u_meas and u_cmd must stay first: UdpSink packs them by position.
JOINT_BLOCKS = ["u_meas", "u_cmd", "q_cmd", "jvel_f",
                "q_prior", "spd_raw", "load_raw", "volt_raw", "temp_raw"]
TELEMETRY_BLOCKS = ("spd_raw", "load_raw", "volt_raw", "temp_raw")   # = hardware.TELEMETRY_FIELDS
DEFAULT_N_FSR = 2

# Schema 1, for reading the logs written before 10/25 (its FSR columns sat mid-row).
V1_SCALAR_COLUMNS = ["t_wall", "t_rel", "step", "loop_ms", "busy_ms",
                     "pg_x", "pg_y", "pg_z", "av_x", "av_y", "av_z", "upright_cos",
                     "rej_total", "fsr_v0", "fsr_v1", "fsr_age_ms"]
V1_JOINT_BLOCKS = ["u_meas", "u_cmd", "q_cmd", "jvel_f"]

# What each column means, copied into every meta so a log explains itself.
COLUMN_NOTES = {
    "acc_*": "raw accelerometer, g, axis_remap applied (sim base frame), NOT normalized; "
             "pg ~ -acc/|acc| before pg's EMA",
    "spd_raw/load_raw/volt_raw/temp_raw.*": "RAW servo registers 0x3A/0x3C/0x3E/0x3F from the "
                                            "same read as u_meas; see servo_telemetry.decoding. "
                                            "NaN = read rejected or telemetry off",
    "q_prior.*": "feedback-law contribution to the command, sim rad, before clipping; NaN = no law",
    "posture_lean": "posture controller's filtered lean, deg (+ forward), self_tilt removed",
    "posture_zero": "its measured upright zero, deg; NaN until measured. error = lean - zero",
    "posture_corr": "knee correction in effect, deg",
    "fsr_v*": "RAW ADS1115 volts per channel; layout in meta fsr.channels",
}


def fsr_columns(n_fsr):
    return [f"fsr_v{i}" for i in range(int(n_fsr))]


def build_columns(dofs, n_fsr=DEFAULT_N_FSR):
    """Full ordered column list for a robot with these joint names and FSR channels."""
    cols = list(SCALAR_COLUMNS)
    for block in JOINT_BLOCKS:
        cols += [f"{block}.{d}" for d in dofs]
    return cols + fsr_columns(n_fsr)


def build_columns_v1(dofs):
    """The schema-1 layout (logs before 10/25), for readers and tests."""
    cols = list(V1_SCALAR_COLUMNS)
    for block in V1_JOINT_BLOCKS:
        cols += [f"{block}.{d}" for d in dofs]
    return cols


def sidecar_path(csv_path, kind):
    """<stem>.<kind> next to the CSV, e.g. sidecar_path(p, 'console.txt')."""
    p = Path(csv_path)
    return p.with_name(f"{p.stem}.{kind}")


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


# Every deploy-side source that can change what a run does. Hashed into each run's meta: the
# Pi is synced by scp, and "which version of hardware.py was on it" has been the question
# more than once.
DEPLOY_SOURCES = ("deploy_standing.py", "pose_sweep.py", "hardware.py", "frame_log.py",
                  "sim_real_map.py", "numpy_policy.py", "posture_control.py", "fsr_monitor.py")


def environment_meta():
    """Where and with what the run happened: host, interpreter, numpy."""
    return {"hostname": socket.gethostname(), "python": sys.version.split()[0],
            "python_full": sys.version, "numpy": np.__version__, "platform": platform.platform()}


def files_meta(files):
    """md5 of every file that shaped the run, plus the deploy sources themselves.

    files: {role: path or None}. Returns {"files": {role: {path, md5}}, "file_md5": {path: md5}}
    -- the second is the schema-1 shape, kept so older readers still find it. A missing file
    is recorded with md5 None rather than dropped: "it was asked for and was not there" is
    itself a finding."""
    out = {}
    for role, path in files.items():
        if path:
            out[role] = {"path": str(path), "md5": file_md5(path)}
    for name in DEPLOY_SOURCES:
        p = DEPLOY_DIR / name
        out[f"source:{name}"] = {"path": str(p), "md5": file_md5(p)}
    return {"files": out, "file_md5": {v["path"]: v["md5"] for v in out.values()}}


class ConsoleTee:
    """Copy everything the run prints (stdout AND stderr) to <stem>.console.txt.

    The console is where a run explains itself -- the posture zero it measured, the tilt
    cut, the gyro calibration retries, the traceback -- and until now it lived only in an
    SSH scrollback that was gone by the time anyone looked at the CSV. Each line in the
    file starts with unix time so it lines up with the CSV's t_wall.

    It stays installed until interpreter exit (atexit), not until the run's `finally`, so
    an uncaught exception's traceback -- printed after main() unwinds -- lands in the file
    too. A failing terminal (dead SSH session) never stops the file: terminal write errors
    are swallowed, and release_on_hangup() swaps only the terminal for /dev/null.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered: a print costs one small write, and a crash loses at most a partial line.
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8", errors="replace")
        self._orig = (sys.stdout, sys.stderr)
        self._lock = threading.Lock()
        self.stdout = _TeeStream(sys.stdout, self)
        self.stderr = _TeeStream(sys.stderr, self)
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self._closed = False
        self._at_line_start = True
        atexit.register(self.close)

    def _record(self, s):
        if self._closed or not s:
            return
        # A '\r' progress update (pose_sweep's probe readout) becomes its own line in the file,
        # each with its own time, instead of one line thousands of updates long.
        with self._lock:
            try:
                for piece in re.split(r"(\r\n|\n|\r)", s):
                    if piece == "\r":
                        if not self._at_line_start:
                            self._fh.write("\n")
                            self._at_line_start = True
                    elif piece in ("\n", "\r\n"):
                        if self._at_line_start:          # a blank line still gets its time
                            self._fh.write(f"{time.time():.3f} | ")
                        self._fh.write("\n")
                        self._at_line_start = True
                    elif piece:
                        if self._at_line_start:
                            self._fh.write(f"{time.time():.3f} | ")
                        self._fh.write(piece)
                        self._at_line_start = False
            except (OSError, ValueError):
                pass

    def close(self):
        if self._closed:
            return
        if sys.stdout is self.stdout:
            sys.stdout = self._orig[0]
        if sys.stderr is self.stderr:
            sys.stderr = self._orig[1]
        with self._lock:
            self._closed = True
            try:
                self._fh.close()
            except OSError:
                pass


class _TeeStream:
    def __init__(self, terminal, tee):
        self.terminal = terminal
        self._tee = tee

    def write(self, s):
        try:
            self.terminal.write(s)
        except Exception:      # a dead terminal must never stop the record, or the run
            pass
        self._tee._record(s)
        return len(s)

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass

    def detach_terminal(self, replacement):
        """Keep recording, stop writing to the (gone) terminal. See release_on_hangup()."""
        self.terminal = replacement

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.terminal, name)


class FrameLogger:
    """Buffered CSV writer. The control loop calls log(); a daemon thread does the I/O.

    Rows are dropped rather than blocking if the writer falls behind -- a logger must
    never be able to stall the control loop. Drops are counted and reported, because a
    silently truncated log is worse than a short one. A row of the wrong length (a caller
    passing 3 FSR values to a 2-channel log) is dropped and counted as malformed on the
    writer thread rather than shifting every later column.
    """

    def __init__(self, path, dofs, meta=None, queue_size=20000, flush_every=40,
                 n_fsr=DEFAULT_N_FSR):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_fsr = int(n_fsr)
        self.columns = build_columns(dofs, self.n_fsr)
        self.n_joints = len(dofs)
        self._q = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._sinks = []
        self._latest = None          # newest row, for in-process live consumers
        self.dropped = 0
        self.malformed = 0
        self.written = 0
        self.t0 = time.time()
        self.flush_every = int(flush_every)
        # Pre-built NaN fills, so a missing block costs the loop nothing to build.
        self._nan3 = (NAN, NAN, NAN)
        self._nan_j = (NAN,) * self.n_joints
        self._nan_tel = (NAN,) * (len(TELEMETRY_BLOCKS) * self.n_joints)
        self._nan_fsr = (NAN,) * self.n_fsr

        meta = dict(meta or {})
        meta.update(schema_version=SCHEMA_VERSION, columns=self.columns, dofs=list(dofs),
                    n_fsr=self.n_fsr, column_notes=COLUMN_NOTES,
                    started_unix=self.t0, started_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    git_sha=git_sha(), git_sha_caveat="Pi syncs by scp; trust file_md5s",
                    argv=list(sys.argv), sidecars={
                        "console": sidecar_path(self.path, "console.txt").name,
                        "events": sidecar_path(self.path, "events.jsonl").name},
                    **environment_meta())
        self.meta = meta
        self.meta_path = self.path.with_suffix(".meta.json")
        self._write_meta()

        self._fh = open(self.path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(self.columns)
        self._thread = threading.Thread(target=self._run, name="frame-log", daemon=True)
        self._thread.start()

    def _write_meta(self):
        self.meta_path.write_text(json.dumps(self.meta, indent=2, default=str))

    def update_meta(self, **kw):
        """Add facts learned after the start (the gyro bias, the run mode). Writes the file
        synchronously, so call it between phases, never inside the control loop."""
        self.meta.update(kw)
        self._write_meta()

    def log(self, *, t_wall, step, loop_ms, busy_ms, pg, av, upright_cos, rej_total,
            u_meas, u_cmd, q_cmd, jvel_f, fsr=None, fsr_age_ms=NAN, acc=None,
            q_prior=None, telemetry=None, posture=None):
        """Called from the control loop. Must stay cheap: tuple build + one queue put.

        acc (3,) g; q_prior (n,) rad; telemetry (4, n) raw spd/load/volt/temp in
        TELEMETRY_BLOCKS order (ServoBus.last_telemetry); posture (lean, zero, corr) deg;
        fsr (n_fsr,) volts. Anything omitted or None is logged as NaN.
        """
        row = (t_wall, t_wall - self.t0, step, loop_ms, busy_ms,
               pg[0], pg[1], pg[2], av[0], av[1], av[2], upright_cos, rej_total, fsr_age_ms,
               *(self._nan3 if acc is None else acc),
               *(self._nan3 if posture is None else posture),
               *u_meas, *u_cmd, *q_cmd, *jvel_f,
               *(self._nan_j if q_prior is None else q_prior),
               *(self._nan_tel if telemetry is None else np.ravel(telemetry).tolist()),
               *(self._nan_fsr if fsr is None else fsr))
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
        ncol = len(self.columns)
        while not (self._stop.is_set() and self._q.empty()):
            try:
                row = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if len(row) != ncol:
                self.malformed += 1
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

    def close(self, **meta):
        """Stop the writer and complete the meta (row counts, end time, plus any **meta)."""
        self._stop.set()
        self._thread.join(timeout=5.0)
        try:
            self._fh.close()
        except OSError:
            pass
        self.meta.update(meta)
        self.meta.update(rows_written=self.written, rows_dropped=self.dropped,
                         rows_malformed=self.malformed,
                         ended_unix=time.time(), duration_s=time.time() - self.t0)
        self._write_meta()
        return self.written, self.dropped


def _fmt(v):
    """Compact but lossless-enough: 6 significant figures beats 17 digits of float noise
    when the file has ~200 columns at 40 Hz. numpy scalars are matched explicitly --
    np.float32 is NOT a subclass of float, so a plain isinstance(v, float) test would let
    every joint angle through unformatted and write a different precision per column."""
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    if isinstance(v, np.integer):
        return int(v)
    return v


# ---- foot sensors ----------------------------------------------------------------------

ADS1115_ADDRS = (0x48, 0x49, 0x4A, 0x4B)    # ADDR pin to GND, VDD, SDA, SCL
DEFAULT_FSR_ADDR = 0x48
DEFAULT_FSR_CHANNELS = "0,1"                 # today's wiring: left heel, left toe on 0x48


def parse_fsr_channels(spec, default_addr=DEFAULT_FSR_ADDR):
    """'0,1' or '0x48:0,0x48:1,0x49:0' -> [(addr, ain), ...] in column order.

    A bare pin means the default board, so the old '0,1' keeps meaning what it always did."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, c = part.split(":", 1)
            addr, ain = int(a.strip(), 0), int(c.strip(), 0)
        else:
            addr, ain = int(default_addr), int(part, 0)
        if addr not in ADS1115_ADDRS:
            raise ValueError(f"FSR channel '{part}': 0x{addr:02X} is not an ADS1115 address "
                             f"({', '.join(f'0x{x:02X}' for x in ADS1115_ADDRS)})")
        if not 0 <= ain <= 3:
            raise ValueError(f"FSR channel '{part}': AIN{ain} -- an ADS1115 has AIN0..AIN3")
        if (addr, ain) in out:
            raise ValueError(f"FSR channel '{part}' listed twice")
        out.append((addr, ain))
    if not out:
        raise ValueError(f"no FSR channels in '{spec}'")
    return out


def default_fsr_layout(channels):
    """Layout for channels given without a layout file. Positions are NOT known then, and
    are recorded as None rather than guessed -- a guessed position turns into a confident
    wrong centre of pressure. Today's 2-channel wiring is at least named."""
    chans = []
    legacy = {(DEFAULT_FSR_ADDR, 0): ("left", "L_heel"), (DEFAULT_FSR_ADDR, 1): ("left", "L_toe")}
    for addr, ain in channels:
        foot, name = legacy.get((addr, ain), (None, f"0x{addr:02X}:{ain}"))
        chans.append({"adc": addr, "ain": ain, "foot": foot, "name": name,
                      "x_mm": None, "y_mm": None})
    return {"channels": chans, "rfixed_ohm": None, "vcc": None, "source": None,
            "note": "fsr_v* are RAW divider volts. No --fsr-layout given: R_fixed and sensor "
                    "positions are NOT known to this run -- record them with it"}


def load_fsr_layout(path):
    """An --fsr-layout YAML -> the layout dict recorded in the meta.

    fsr:
      rfixed_ohm: 2000
      vcc: 3.3
      channels:          # column order: the first entry is fsr_v0
        - {adc: 0x48, ain: 0, foot: left, name: L_heel_out, x_mm: -60, y_mm: 20}

    x_mm + forward, y_mm + to the robot's LEFT, both from the centre of THAT foot's sole, so
    a per-foot centre of pressure is sum(g_i * x_i) / sum(g_i) without knowing the stance."""
    import yaml  # noqa: PLC0415
    with open(path) as f:
        d = yaml.safe_load(f)
    d = d.get("fsr", d)
    chans = []
    for k, c in enumerate(d["channels"]):
        addr = int(str(c["adc"]), 0) if isinstance(c["adc"], str) else int(c["adc"])
        ain = int(c["ain"])
        parse_fsr_channels(f"0x{addr:02X}:{ain}")          # same validation as the CLI
        foot = c.get("foot")
        if foot not in ("left", "right", None):
            raise ValueError(f"{path}: channel {k} foot '{foot}' (left, right or omitted)")
        chans.append({"adc": addr, "ain": ain, "foot": foot,
                      "name": c.get("name", f"0x{addr:02X}:{ain}"),
                      "x_mm": None if c.get("x_mm") is None else float(c["x_mm"]),
                      "y_mm": None if c.get("y_mm") is None else float(c["y_mm"])})
    pairs = [(c["adc"], c["ain"]) for c in chans]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{path}: a channel is listed twice")
    return {"channels": chans, "rfixed_ohm": d.get("rfixed_ohm"), "vcc": d.get("vcc"),
            "source": str(path),
            "note": "fsr_v* are RAW divider volts in this channel order; positions from the file"}


def resolve_fsr(channels_spec=None, layout_path=None):
    """(channels [(addr, ain)], layout dict) from the CLI. The layout file, when given, IS the
    channel list; a --fsr-channels that disagrees with it is an error, not a silent pick."""
    if layout_path:
        layout = load_fsr_layout(layout_path)
        chans = [(c["adc"], c["ain"]) for c in layout["channels"]]
        if channels_spec and channels_spec != DEFAULT_FSR_CHANNELS \
                and parse_fsr_channels(channels_spec) != chans:
            raise ValueError(f"--fsr-channels {channels_spec} disagrees with {layout_path}; "
                             "give one or the other")
        return chans, layout
    chans = parse_fsr_channels(channels_spec or DEFAULT_FSR_CHANNELS)
    return chans, default_fsr_layout(chans)


class FsrSampler:
    """Samples the ADS1115 board(s) on their own thread so the control loop never waits on I2C.

    A conversion is ~1.2 ms at 860 SPS and switching the mux costs a full conversion,
    so polling channels inline would spend several ms of a 25 ms budget that is
    already ~21 ms full. The loop reads whatever the latest sample is and records how
    stale it was, which is honest and costs nothing.

    Several boards convert IN PARALLEL: each conversion wait covers the k-th channel of every
    board at once, so 4 channels on each of two boards costs four waits per sweep, not eight.
    A board that errors only blanks its own channels (NaN, counted in `errors`); the others
    keep updating.

    channels: [(addr, ain), ...], or plain pins on `addr` (the old signature).
    """

    def __init__(self, channels=(0, 1), addr=DEFAULT_FSR_ADDR, bus=1, hz=50.0, smbus=None):
        sys.path.insert(0, str(DEPLOY_DIR))
        import fsr_monitor as fm      # noqa: PLC0415
        self._fm = fm
        self._bus = smbus if smbus is not None else fm._open_bus(bus)
        self.channels = [c if isinstance(c, tuple) else (int(addr), int(c)) for c in channels]
        self.addr = int(addr)
        self.period = 1.0 / max(hz, 1e-3)
        self.volts = [NAN] * len(self.channels)
        self.t_sample = 0.0
        self.errors = 0
        # Group k = the k-th channel of each board: those convert during the same wait.
        by_board = {}
        for idx, (a, c) in enumerate(self.channels):
            by_board.setdefault(a, []).append((idx, a, c))
        depth = max(len(v) for v in by_board.values())
        self._groups = [[v[k] for v in by_board.values() if k < len(v)] for k in range(depth)]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fsr", daemon=True)
        self._thread.start()

    def sweep(self):
        """One reading of every channel, volts (NaN where a board failed)."""
        fm = self._fm
        out = [NAN] * len(self.channels)
        for group in self._groups:
            started = []
            for idx, a, c in group:
                cfg = fm.CONFIG_BASE | (c << 12)
                try:
                    self._bus.write_i2c_block_data(a, fm.REG_CONFIG, [(cfg >> 8) & 0xFF, cfg & 0xFF])
                    started.append((idx, a))
                except Exception:
                    self.errors += 1
            time.sleep(0.002)      # 860 SPS -> ~1.2 ms per conversion, all boards at once
            for idx, a in started:
                try:
                    r = self._bus.read_i2c_block_data(a, fm.REG_CONV, 2)
                    raw = (r[0] << 8) | r[1]
                    if raw > 0x7FFF:
                        raw -= 0x10000
                    out[idx] = raw * fm.PGA_FULL_SCALE / 32768.0
                except Exception:
                    self.errors += 1
        return out

    def _run(self):
        while not self._stop.is_set():
            try:
                self.volts = self.sweep()
                self.t_sample = time.time()
            except Exception:
                self.errors += 1
            self._stop.wait(self.period)      # not sleep: close() should not wait out a period

    def read(self):
        """Latest sample and its age in ms."""
        age = (time.time() - self.t_sample) * 1000.0 if self.t_sample else NAN
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


def _get(row, key):
    """Column as float, NaN if absent or blank (a v1 row has no v2 columns)."""
    try:
        v = row[key]
    except (KeyError, IndexError):
        return NAN
    if v is None or v == "":
        return NAN
    return float(v)


def pose_from_row(row, servo_map):
    """Row (dict or Series, schema 1 or 2) -> everything needed to draw or replay the robot.

    This is the Step 3 / replay entry point. Absolute joint angles come from the RAW
    encoder units through the same map the deploy loop used, so a viewer reproduces
    what the robot did rather than what it was told to do.

    Returns dict with:
      q_meas  (n,) absolute joint angles, sim radians, 0 = straight
      q_cmd   (n,) commanded absolute joint angles, sim radians
      proj_grav (3,) pelvis attitude in the observation frame
      acc     (3,) raw accelerometer, g (schema 2; NaN for a schema-1 row)
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
        "acc": np.array([_get(row, "acc_x"), _get(row, "acc_y"), _get(row, "acc_z")],
                        dtype=np.float32),
    }


def default_log_path(tag, log_dir=None):
    d = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    return d / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.csv"
