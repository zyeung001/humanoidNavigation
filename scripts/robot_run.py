#!/usr/bin/env python3
"""One command per robot run, from the laptop: sync the Pi, run in tmux, fetch the logs, diagnose.

WHY. A hardware run used to be five hand-typed steps -- scp the changed files, check md5s by eye,
ssh in, run, scp the logs back -- and every one of them has failed silently at least once: a stale
numpy_policy.py on the Pi, a log never copied back, a run killed by a dropped Wi-Fi link. After
every run Claude needs the WHOLE record (frame log + meta + console + events) on the laptop to say
what the robot physically did, so the copy back cannot be a step someone remembers.

    python scripts/robot_run.py sync  -- --policy-npz models/x.npz --residual-config config/stances/knee2.yaml
    python scripts/robot_run.py run   -- --policy-npz models/x.npz --residual-config config/stances/knee2.yaml
    python scripts/robot_run.py run --script pose_sweep.py -- --joints waist_pitch
    python scripts/robot_run.py attach        # back into a run after an SSH drop
    python scripts/robot_run.py stop          # Ctrl-C the run from any terminal (ramp + torque off)
    python scripts/robot_run.py fetch         # copy new logs into logs/pi/ and diagnose them
    ... add --dry-run to any of them to print the exact ssh/scp commands and touch nothing.

Everything after `--` is passed to the deploy script unchanged, except that file arguments
(--policy-npz, --residual-config, --posture, --map, --imu-calib, --baseline, --model, --vecnorm)
are synced first and rewritten to repo-relative paths, which is what they are on the Pi too.
The deploy scripts' argparse accepts abbreviations (`--policy` IS --policy-npz on the Pi), so an
abbreviation is resolved against the target script's own flags the way argparse does and spelled
out; one that could be a file flag but is ambiguous is refused. Otherwise `--policy x.npz` would
skip the sync and the bundle check and the Pi would run whatever stale x.npz it already had.

SYNC. One ssh call md5sums the deploy file set on the Pi (every scripts/deploy/*.py + the files
the run reads + the default servo map and IMU calibration); only files that differ are copied,
then md5sum'd again, and any mismatch is an error. Text files (.py .yaml .yml .json .txt .sh
.md) are compared and copied as their CRLF->LF content, i.e. byte-identical to the git blob:
this laptop checks out with core.autocrlf=true, so its config/imu_calib.yaml is CRLF while the
Pi's (from git, LF) is the very same calibration -- a raw-byte md5 called them different
(7e6bd0d0 vs bf19d884 on 10/25), and every copy left the Pi's checkout dirty for `git pull`.
Hashing the normalised text makes the Pi's md5s (and the frame log meta's file_md5) equal the
git blob's. A Pi file that differs only in line endings is re-copied as LF, with no warning.
Two files are written ON the Pi by its own calibration tools (imu_calib.yaml by calibrate_imu.py
--apply, joint_servo_map.yaml by verify_signs.py / fix_*_limits.py / calibrate_centers.py): when
their CONTENT differs the Pi copy is KEPT and the table says so, because the laptop's copy may be
the stale one. --push-calibration overwrites them deliberately. `run` refuses to start on a kept
copy unless --keep-pi-calibration says that is intended (the warning would otherwise vanish
under the tmux screen), and then copies the Pi's servo map into logs/pi/ so diagnose_run.py
reads the map the robot actually used.
Bundles (.npz) are checked with the numpy_policy.py being synced before anything is copied, and
their `requires` list is printed: a bundle that would be refused on the Pi fails here instead.

RUN. The deploy script runs INSIDE tmux on the Pi (`tmux new-session -d`, then `ssh -t ...
tmux attach`), so a dropped SSH session no longer ends the run. That trades away the 10/18
hangup fix (hardware.release_on_hangup: an SSH drop released the torque): after a drop the robot
KEEPS RUNNING until you `attach` and Ctrl-C, run `stop`, or the tilt cut fires. --no-tmux runs
over plain `ssh -t` instead, where a drop releases the torque as before. When the script exits,
the tmux pane waits for Enter so the last lines can be read; the session then ends and the logs
are fetched. Detaching (Ctrl-b d) or a drop leaves it running: nothing is fetched until it ends,
and if the Pi cannot be asked (the link is down) the run is assumed to be still going.

FETCH. Every file in the Pi's logs/ that is not in logs/pi/, or is LARGER there, is copied
(<stem>.csv, .meta.json, .console.txt, .events.jsonl and any other <stem>.* sidecar), so a log
fetched while it was still being written is fetched again once it has grown. A Pi file SMALLER
than the laptop's copy is a different run under a reused name -- the Pi has no RTC, and after an
unclean power-off fake-hwclock restores an earlier time, so time.strftime can hand out a name
again -- and that whole <stem>.* group is saved as <stem>-pi2.* (next free N) instead of
overwriting the laptop's only copy of the old run. (Same name AND same size is not detected.)
Each new .csv is then run through scripts/debug/diagnose_run.py (output also saved as
<stem>.diagnose.txt) and scripts/debug/run_report.py when that exists; a missing post-processor
is a warning. They run with PYTHONIOENCODING=utf-8 so a degree sign survives a Windows console.

The address lives in config/robot.yaml (gitignored; copy config/robot.yaml.example). Needs
OpenSSH ssh/scp on PATH (Windows 10+ ships them; Git Bash works too) and tmux on the Pi.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "robot.yaml"
NUMPY_POLICY = Path(__file__).resolve().parent / "deploy" / "numpy_policy.py"

DEFAULT_SCRIPT = "deploy_standing.py"
# Deploy-script arguments that name a file the run reads: synced, and rewritten repo-relative.
FILE_FLAGS = ("--policy-npz", "--residual-config", "--posture", "--map", "--imu-calib", "--baseline",
              "--model", "--vecnorm")
# Read by every hardware script even when not passed (their argparse defaults).
DEFAULT_FILES = {"--map": "config/joint_servo_map.yaml", "--imu-calib": "config/imu_calib.yaml"}
# Written on the Pi by its calibration tools: never overwritten silently (see SYNC above).
PI_CALIBRATED = frozenset(DEFAULT_FILES.values())
# Compared and copied as CRLF->LF text (see SYNC above).
TEXT_SUFFIXES = frozenset((".py", ".yaml", ".yml", ".json", ".txt", ".sh", ".md"))
POST_PROCESSORS = ("scripts/debug/diagnose_run.py", "scripts/debug/run_report.py")
LOCAL_LOGS = "logs/pi"
SCP_BATCH = 40          # files per scp call: keeps a Windows command line far under its 32 KB cap

# Remote paths go into a remote shell and into scp's host:path unquoted, so they are restricted to
# characters neither can misread. Local paths never need this: scp runs with cwd=ROOT on
# repo-relative paths (no drive-letter colon for scp to take for a host, no spaces).
_SAFE_PATH = re.compile(r"[A-Za-z0-9_.~/+-]+")
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.+-]+")


class Abort(Exception):
    """A failure the user must act on: printed as one message, exit status 1."""


def show(argv) -> str:
    """argv as one line you could paste into this machine's shell."""
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def is_text(rel) -> bool:
    return PurePosixPath(Path(rel).as_posix()).suffix.lower() in TEXT_SUFFIXES


def payload(path) -> bytes:
    """The bytes the Pi should hold for a laptop file: text as CRLF->LF (what git stores; the
    laptop's autocrlf checkout adds the CRs), anything else (.npz, .zip, .pkl) untouched."""
    data = Path(path).read_bytes()
    return data.replace(b"\r\n", b"\n") if is_text(path) else data


# ---------------------------------------------------------------- config

def load_config(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise Abort(f"{path} not found: copy config/robot.yaml.example to config/robot.yaml and set "
                    f"host/user/repo (it is gitignored, so the robot's address never lands in git)")
    with open(path) as f:
        return check_config(yaml.safe_load(f) or {}, path)


def check_config(cfg: dict, where="robot config") -> dict:
    out = {"python": "python3", "port": None, "ssh_options": [], "tmux_session": "robot", **cfg}
    for key in ("host", "user", "repo"):
        if not out.get(key):
            raise Abort(f"{where}: `{key}` is not set")
    for key, pat in (("host", _SAFE_NAME), ("user", _SAFE_NAME), ("tmux_session", _SAFE_NAME),
                     ("repo", _SAFE_PATH), ("python", _SAFE_PATH)):
        if not pat.fullmatch(str(out[key])):
            raise Abort(f"{where}: {key}={out[key]!r} has characters a remote shell would misread")
    out["repo"] = str(out["repo"]).rstrip("/")
    if not isinstance(out["ssh_options"], list):
        raise Abort(f"{where}: ssh_options must be a list, e.g. [-i, C:/Users/me/.ssh/id_ed25519]")
    out["ssh_options"] = [str(o) for o in out["ssh_options"]]
    return out


# ---------------------------------------------------------------- the network seam

class Shell:
    """The only code that starts processes; tests replace it."""

    def run(self, argv, capture=True, cwd=None, env=None):
        return subprocess.run(argv, capture_output=capture, text=True, encoding="utf-8",
                              errors="replace", cwd=cwd, env=env)

    def interactive(self, argv, cwd=None):
        """Hand the terminal to ssh -t until it exits. Ctrl-C belongs to the robot run on the
        other end: subprocess.run would kill ssh on KeyboardInterrupt (= detach, run continues),
        so the Ctrl-C would never reach the robot. Wait through it instead."""
        p = subprocess.Popen(argv, cwd=cwd)
        while True:
            try:
                return p.wait()
            except KeyboardInterrupt:
                continue


class Robot:
    def __init__(self, cfg, dry_run=False, shell=None, echo=print):
        self.cfg = cfg
        self.dry_run = dry_run
        self.shell = shell or Shell()
        self.echo = echo
        self.target = f"{cfg['user']}@{cfg['host']}"
        self.repo = cfg["repo"]
        self.session = cfg["tmux_session"]

    # argv builders -------------------------------------------------
    def ssh_argv(self, remote_cmd, tty=False):
        argv = ["ssh"] + (["-t"] if tty else [])
        if self.cfg["port"]:
            argv += ["-p", str(self.cfg["port"])]
        return argv + self.cfg["ssh_options"] + [self.target, remote_cmd]

    def scp_argv(self, sources, dest):
        argv = ["scp", "-q"]
        if self.cfg["port"]:
            argv += ["-P", str(self.cfg["port"])]
        return argv + self.cfg["ssh_options"] + [str(s) for s in sources] + [str(dest)]

    def remote(self, rel) -> str:
        return f"{self.target}:{self.repo}/{rel}"

    # execution -----------------------------------------------------
    def call(self, argv, check=True, cwd=None):
        """Run one ssh/scp (cwd=ROOT unless given). Dry run: print it and return None."""
        if self.dry_run:
            self.echo("$ " + show(argv))
            return None
        r = self.shell.run(argv, capture=True, cwd=str(cwd or ROOT))
        if check and r.returncode != 0:
            raise Abort(f"`{show(argv)}` failed (exit {r.returncode}):\n{(r.stderr or r.stdout or '').strip()}")
        return r

    def interactive(self, argv):
        if self.dry_run:
            self.echo("$ " + show(argv))
            return 0
        return self.shell.interactive(argv, cwd=str(ROOT))

    # remote operations ---------------------------------------------
    def md5s(self, files):
        """{repo-relative path: (md5, md5 with every CR removed)} for the files that exist on the
        Pi, in ONE ssh call. The second hash tells "same text, CRLF line endings" (a copy scp'd
        from an autocrlf checkout) from a real difference. None in a dry run (nothing is known)."""
        r = self.call(self.ssh_argv(
            f"cd {self.repo} || exit 3; for f in {' '.join(files)}; do [ -f \"$f\" ] && echo "
            "\"$(md5sum < \"$f\" | cut -c1-32) $(tr -d '\\r' < \"$f\" | md5sum | cut -c1-32) $f\"; "
            "done; exit 0"), check=False)
        if r is None:
            return None
        if r.returncode == 3:
            raise Abort(f"{self.repo} does not exist on {self.target}: fix `repo` in config/robot.yaml")
        if r.returncode != 0:
            raise Abort(f"md5sum on the Pi failed (exit {r.returncode}): {(r.stderr or '').strip()}")
        out = {}
        for line in r.stdout.splitlines():
            parts = line.split(None, 2)
            if len(parts) == 3 and len(parts[0]) == len(parts[1]) == 32:
                out[parts[2]] = (parts[0], parts[1])
        return out

    def copy_up(self, files):
        """scp repo-relative `files` to the same paths under the Pi's repo, creating directories.
        What is sent is payload() (text as LF), staged in a temp dir that mirrors the repo paths;
        scp runs with that dir as cwd, so argv still holds only short relative paths."""
        dirs = sorted({PurePosixPath(f).parent.as_posix() for f in files})
        new_dirs = [d for d in dirs if d != "."]
        if new_dirs:
            self.call(self.ssh_argv(f"cd {self.repo} && mkdir -p {' '.join(new_dirs)}"))
        with tempfile.TemporaryDirectory(prefix="robot_run_") as stage:
            if not self.dry_run:
                for f in files:
                    (Path(stage) / f).parent.mkdir(parents=True, exist_ok=True)
                    (Path(stage) / f).write_bytes(payload(ROOT / f))
            for d in dirs:
                group = [f for f in files if PurePosixPath(f).parent.as_posix() == d]
                dest = self.remote("" if d == "." else d + "/")
                for i in range(0, len(group), SCP_BATCH):
                    self.call(self.scp_argv(group[i:i + SCP_BATCH], dest), cwd=stage)

    def session_state(self):
        """'running' / 'ended' for the tmux session, 'unknown' when ssh itself failed (exit 255
        after a Wi-Fi drop: the run may well still be live); None in a dry run."""
        r = self.call(self.ssh_argv(f"tmux has-session -t ={self.session} 2>/dev/null && echo running "
                                    "|| echo ended"), check=False)
        if r is None:
            return None
        if r.returncode != 0:
            return "unknown"
        return "running" if "running" in r.stdout else "ended"


# ---------------------------------------------------------------- sync

def deploy_files():
    return sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts" / "deploy").glob("*.py"))


def repo_relative(value, remote_repo) -> str:
    """A deploy-script file argument as a repo-relative POSIX path that exists locally."""
    raw = value.replace("\\", "/")
    if raw.startswith(remote_repo + "/"):
        rel = PurePosixPath(raw[len(remote_repo) + 1:])
    elif Path(value).is_absolute():
        try:
            rel = PurePosixPath(Path(value).resolve().relative_to(ROOT.resolve()).as_posix())
        except ValueError:
            raise Abort(f"{value}: outside the repo ({ROOT}); the Pi only has the repo, so pass a "
                        f"path inside it") from None
    else:
        rel = PurePosixPath(raw)
    if ".." in rel.parts:
        raise Abort(f"{value}: paths with '..' are not synced; pass it relative to the repo root")
    rel = rel.as_posix()
    if not _SAFE_PATH.fullmatch(rel):
        raise Abort(f"{value}: rename it without spaces or shell characters before syncing it to the Pi")
    if not (ROOT / rel).is_file():
        raise Abort(f"{value}: not found at {ROOT / rel}")
    return rel


def script_flags(script) -> frozenset:
    """The --long flags scripts/deploy/<script> declares (read from its source: the laptop has no
    serial/I2C libraries to import it with). Empty if none are found."""
    src = (ROOT / "scripts" / "deploy" / script).read_text(encoding="utf-8", errors="replace")
    return frozenset(re.findall(r"""add_argument\(\s*["'](--[A-Za-z0-9][\w-]*)["']""", src))


def resolve_flag(flag, known):
    """`flag` as argparse on the Pi will read it: an exact flag stays, an abbreviation that is a
    prefix of exactly one known flag becomes that flag (allow_abbrev is on in the deploy
    scripts). An abbreviation that might be a FILE flag but is ambiguous is refused here: argparse
    would refuse it too, but only after the sync that could not know which file to check."""
    if not flag.startswith("--") or len(flag) <= 2 or flag in known or flag in FILE_FLAGS:
        return flag
    matches = sorted(k for k in known if k.startswith(flag))
    if not any(m in FILE_FLAGS for m in matches):
        return flag
    if len(matches) > 1:
        raise Abort(f"{flag} is ambiguous ({', '.join(matches)}): spell the flag out")
    return matches[0]


def run_files(run_args, remote_repo, known=None):
    """(files the run reads, run_args with those files rewritten repo-relative and abbreviated file
    flags spelled out). `known` = the target script's flags (script_flags); FILE_FLAGS if empty."""
    known = frozenset(known or FILE_FLAGS)
    files, out, given = [], [], set()
    i = 0
    while i < len(run_args):
        arg = run_args[i]
        flag, eq, val = arg.partition("=")
        flag = resolve_flag(flag, known)
        if flag in FILE_FLAGS and (eq or i + 1 < len(run_args)):
            if not eq:
                i += 1
                val = run_args[i]
            rel = repo_relative(val, remote_repo)
            files.append(rel)
            given.add(flag)
            out += [f"{flag}={rel}"] if eq else [flag, rel]
        else:
            out.append(f"{flag}{eq}{val}")
        i += 1
    for flag, rel in DEFAULT_FILES.items():
        if flag not in given and (ROOT / rel).is_file():
            files.append(rel)
    return list(dict.fromkeys(files)), out


def _numpy_policy():
    spec = importlib.util.spec_from_file_location("numpy_policy_synced", NUMPY_POLICY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_bundles(files, echo=print):
    """Print each bundle's `requires` and refuse one the synced numpy_policy.py cannot run."""
    npzs = [f for f in files if f.endswith(".npz")]
    if not npzs:
        return {}
    np_pol = _numpy_policy()
    infos = {}
    for f in npzs:
        try:
            info = np_pol.describe(str(ROOT / f))
        except np_pol.BundleError as e:
            raise Abort(f"the Pi would refuse this bundle: {e}") from None
        infos[f] = info
        what = ("legacy bundle (no requires list); its keys need " if info["legacy"]
                else f"bundle format {info['format']}, requires ")
        echo(f"{f}: {what}{info['requires'] or '[] (plain network)'}")
    return infos


def table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in [header] + rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def sync(robot, files, push_calibration=False):
    """Make the Pi's copy of `files` match the laptop's. Returns {"copied", "kept", "same"}."""
    echo = robot.echo
    check_bundles(files, echo)
    data = {f: payload(ROOT / f) for f in files}
    local = {f: md5_bytes(data[f]) for f in files}
    before = robot.md5s(files)
    rows, copy, kept, same = [], [], [], []
    for f in files:
        pi, pi_text = (None, None) if before is None or f not in before else before[f]
        if before is None:
            action = "copy if it differs (dry run)"
            if f in PI_CALIBRATED and not push_calibration:
                action = "copy only if missing on the Pi (Pi-calibrated)"
            else:
                copy.append(f)
        elif pi == local[f]:
            action = "same"
            same.append(f)
        elif pi is None:
            action = "copy (not on the Pi)"
            copy.append(f)
        elif is_text(f) and pi_text == md5_bytes(data[f].replace(b"\r", b"")):
            action = "copy (same text; the Pi's copy has CRLF line endings)"
            copy.append(f)
        elif f in PI_CALIBRATED and not push_calibration:
            action = "DIFFERS: kept the Pi's copy (Pi-calibrated; --push-calibration overwrites)"
            kept.append(f)
        else:
            action = "copy (differs)"
            copy.append(f)
        rows.append((f, local[f], "?" if before is None else (pi or "-"), action))
    echo(table(rows, ("file", "local md5", "Pi md5 before", "action")))

    if copy:
        robot.copy_up(copy)
        after = robot.md5s(copy)
        if after is not None:
            bad = [f"  {f}: local {local[f]}, Pi {after.get(f, ('missing',))[0]}" for f in copy
                   if after.get(f, (None,))[0] != local[f]]
            if bad:
                raise Abort("md5 mismatch on the Pi after copying -- do NOT run until this is fixed:\n"
                            + "\n".join(bad))
            echo(f"verified: {len(copy)} copied file(s) now match on the Pi")
    else:
        echo("the Pi already has every file")
    for f in kept:
        echo(f"WARNING: {f} differs on the Pi and was NOT copied; a run uses the Pi's version "
             f"(its md5 is in the log's meta.json). Decide which copy is right: commit the Pi's, or "
             f"--push-calibration the laptop's.")
    return {"copied": copy, "kept": kept, "same": same,
            "pi_md5": {f: before[f][0] for f in kept}}


# ---------------------------------------------------------------- run

def keep_or_refuse(robot, synced, files, keep):
    """`run` on a Pi-calibrated file the Pi kept: refuse unless --keep-pi-calibration, because the
    WARNING printed by sync is covered by the tmux screen a moment later. When kept on purpose and
    the file is the servo map the run reads, copy the Pi's map into logs/pi/pi_maps/ and return its
    path, so diagnose_run.py reads the map the robot actually used, not the laptop's."""
    if not synced["kept"]:
        return None
    if not keep:
        raise Abort(f"{', '.join(synced['kept'])} differ(s) on the Pi (see the table above). Decide "
                    "which copy is right, then re-run with --push-calibration (laptop's copy) or "
                    "--keep-pi-calibration (Pi's copy).")
    pi_map = DEFAULT_FILES["--map"]
    if pi_map not in synced["kept"] or pi_map not in files:      # kept: only the IMU calibration
        return None
    rel = f"{LOCAL_LOGS}/pi_maps/{PurePosixPath(pi_map).stem}.{synced['pi_md5'][pi_map][:8]}.yaml"
    if not robot.dry_run:
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    robot.call(robot.scp_argv([robot.remote(pi_map)], rel))
    robot.echo(f"the Pi's {pi_map} is in {rel}; diagnose_run.py will read it (--map)")
    return rel


def remote_command(robot, script, run_args) -> str:
    return " ".join([f"cd {robot.repo} &&", robot.cfg["python"], f"scripts/deploy/{script}",
                     shlex.join(run_args)]).rstrip()


def start_in_tmux(robot, script, run_args):
    s = robot.session
    inner = (remote_command(robot, script, run_args)
             + f'; st=$?; echo; echo "[robot_run] {script} exited with status $st.'
             ' Press Enter to close this session (the logs are fetched after it closes)."; read -r _')
    remote = (f"command -v tmux >/dev/null || exit 3; tmux has-session -t ={s} 2>/dev/null && exit 4; "
              f"tmux new-session -d -s {s} {shlex.quote(inner)}")
    r = robot.call(robot.ssh_argv(remote), check=False)
    if r is None:
        return
    if r.returncode == 3:
        raise Abort("tmux is not installed on the Pi: `sudo apt install tmux` there (or use --no-tmux)")
    if r.returncode == 4:
        raise Abort(f"a run is already in tmux session '{s}' on the Pi: `robot_run.py attach` to see "
                    f"it, `robot_run.py stop` to end it")
    if r.returncode != 0:
        raise Abort(f"could not start the run in tmux (exit {r.returncode}): {(r.stderr or '').strip()}")


def attach_and_collect(robot, report=True, map_path=None):
    """Attach to the tmux session; fetch + diagnose once the session has ended. Anything but a
    definite 'ended' (incl. an ssh that failed because the link is still down) counts as running:
    fetching then would diagnose a half-written log of a live run."""
    robot.interactive(robot.ssh_argv(f"tmux attach -t ={robot.session}", tty=True))
    state = robot.session_state()
    if state not in ("ended", None):
        why = ("detached or the connection dropped" if state == "running"
               else "could not reach the Pi to check, so assume it is")
        robot.echo(f"\nThe run is still going in tmux session '{robot.session}' on the Pi ({why}). "
                   "The robot is NOT released. `robot_run.py attach` to get back in, `robot_run.py "
                   "stop` to end it; logs are fetched when it ends.")
        return []
    return fetch(robot, report=report, map_path=map_path)


def with_log(script, run_args, echo=print):
    """deploy_standing.py without --log records nothing; a robot run must leave a log. An explicit
    --no-log (logging/schema-v2 makes logging the default and lets --no-log opt out) is respected."""
    if "--no-log" in run_args:
        return run_args
    if script == DEFAULT_SCRIPT and not any(a == "--log" or a.startswith("--log=") for a in run_args):
        echo("adding --log: every robot run is recorded (pass --log PATH to name it)")
        return run_args + ["--log"]
    return run_args


# ---------------------------------------------------------------- fetch

def remote_logs(robot):
    """{name: size} of the files directly in the Pi's logs/ (ONE ssh call); None in a dry run."""
    r = robot.call(robot.ssh_argv(f"cd {robot.repo}/logs 2>/dev/null || exit 0; "
                                  "find . -maxdepth 1 -type f -printf '%s %f\\n'"))
    if r is None:
        return None
    out = {}
    for line in r.stdout.splitlines():
        size, _, name = line.strip().partition(" ")
        if not size.isdigit() or not name:
            continue
        if not _SAFE_NAME.fullmatch(name):
            robot.echo(f"WARNING: skipping logs/{name!r} on the Pi: rename it without spaces or shell characters")
            continue
        out[name] = int(size)
    return out


def plan_fetch(listing, local_dir):
    """([(Pi name, local name)] to copy, {stem: new stem}) -- see FETCH in the module docstring.
    Files are grouped by stem (the name up to its first '.') so a run's sidecars move together."""
    def size(name):
        f = local_dir / name
        return f.stat().st_size if f.is_file() else None

    stems = {}
    for n in sorted(listing):
        stems.setdefault(n.split(".", 1)[0], []).append(n)
    plan, renamed = [], {}
    for stem, names in stems.items():
        targets = {n: n for n in names}
        if any((size(n) or 0) > listing[n] for n in names):        # a different run, same name
            k = 2
            while True:
                targets = {n: f"{stem}-pi{k}{n[len(stem):]}" for n in names}
                if all((size(t) or 0) <= listing[n] for n, t in targets.items()):
                    break
                k += 1
        todo = [(n, t) for n, t in targets.items() if size(t) is None or size(t) < listing[n]]
        if todo and targets[names[0]] != names[0]:
            renamed[stem] = f"{stem}-pi{k}"            # warned about only while there is a copy to make
        plan += todo
    return plan, renamed


def fetch(robot, report=True, map_path=None):
    """Copy every Pi log file that is new or has grown into logs/pi/. Returns the local names."""
    echo = robot.echo
    local_dir = ROOT / LOCAL_LOGS
    listing = remote_logs(robot)
    if listing is None:
        echo("(dry run: the Pi's logs are not listed; each new file would be copied with)")
        robot.call(robot.scp_argv([robot.remote("logs/<new file>")], LOCAL_LOGS))
        if report:
            post_process(robot, ["<stem>.csv"], map_path=map_path)
        return []
    plan, renamed = plan_fetch(listing, local_dir)
    if not plan:
        echo(f"nothing new: all {len(listing)} file(s) in the Pi's logs/ are already in {LOCAL_LOGS}/")
        return []
    local_dir.mkdir(parents=True, exist_ok=True)
    for stem, new_stem in renamed.items():
        echo(f"WARNING: the Pi's logs/{stem}.* are SMALLER than the copies already in {LOCAL_LOGS}/: a "
             f"different run reused the name (the Pi's clock went back?). Saving them as {new_stem}.*; "
             f"the old copies are untouched.")
    was = {t: (local_dir / t).stat().st_size for _, t in plan if (local_dir / t).is_file()}
    same = [n for n, t in plan if n == t]
    for i in range(0, len(same), SCP_BATCH):
        robot.call(robot.scp_argv([robot.remote(f"logs/{n}") for n in same[i:i + SCP_BATCH]], LOCAL_LOGS))
    for n, t in plan:
        if n != t:
            robot.call(robot.scp_argv([robot.remote(f"logs/{n}")], f"{LOCAL_LOGS}/{t}"))
    rows = []
    for n, t in plan:
        got = (local_dir / t).stat().st_size if (local_dir / t).is_file() else None
        note = "" if t not in was else f"updated (was {was[t]} bytes)"
        if n != t:
            note = f"from the Pi's {n}"
        if got != listing[n]:
            note = f"SIZE {got} != Pi {listing[n]}: still being written? fetch again after the run"
        rows.append((t, listing[n], note))
    echo(f"\narrived in {LOCAL_LOGS}/:")
    echo(table(rows, ("file", "bytes", "")))
    if report:
        post_process(robot, [t for _, t in plan if t.endswith(".csv")], map_path=map_path)
    return [t for _, t in plan]


def post_process(robot, csvs, map_path=None):
    """diagnose_run.py (+ run_report.py if it exists) on each new log. Never fatal.
    `map_path`: the servo map the robot actually ran with, when the Pi kept its own (see SYNC).
    The children print UTF-8 (deg signs, arrows): this laptop's default is cp1252, which would
    turn them into mojibake -- or raise in the child and lose the report."""
    echo = robot.echo
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    for tool in POST_PROCESSORS:
        if not (ROOT / tool).is_file():
            echo(f"WARNING: {tool} not found; skipped (not merged on this branch yet?)")
            continue
        for name in csvs:
            argv = [sys.executable, tool, f"{LOCAL_LOGS}/{name}"]
            if map_path and tool.endswith("diagnose_run.py"):
                argv += ["--map", map_path]
            if robot.dry_run:
                echo("$ " + show(argv))
                continue
            echo(f"\n===== {Path(tool).name} {name} =====")
            r = robot.shell.run(argv, capture=True, cwd=str(ROOT), env=env)
            text = (r.stdout or "") + (r.stderr or "")
            echo(text.rstrip())
            if tool.endswith("diagnose_run.py") and r.stdout:
                (ROOT / LOCAL_LOGS / (name[:-len(".csv")] + ".diagnose.txt")).write_text(
                    r.stdout, encoding="utf-8")
            if r.returncode != 0:
                echo(f"WARNING: {tool} exited {r.returncode} on {name}")


# ---------------------------------------------------------------- CLI

def main(argv=None, shell=None, echo=print, cfg=None):
    """CLI entry. Tests pass `shell` (a fake process layer) and `cfg` (instead of config/robot.yaml)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    run_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, run_args = argv[:i], argv[i + 1:]

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(CONFIG), help="robot address file (default config/robot.yaml)")
    common.add_argument("--dry-run", action="store_true",
                        help="print the exact ssh/scp commands; no network, no files touched")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, text in (("sync", "make the Pi's deploy files match the laptop's (md5-checked)"),
                       ("run", "sync, run the deploy script in tmux on the Pi, then fetch + diagnose"),
                       ("attach", "reattach to the tmux run; fetch + diagnose once it has ended"),
                       ("stop", "send Ctrl-C to the tmux run (it ramps home and releases torque)"),
                       ("fetch", "copy new logs from the Pi into logs/pi/ and diagnose them")):
        sp = sub.add_parser(name, parents=[common], help=text, description=text)
        if name in ("sync", "run"):
            sp.add_argument("--script", default=DEFAULT_SCRIPT,
                            help="scripts/deploy/<SCRIPT> to run (default deploy_standing.py)")
            sp.add_argument("--push-calibration", action="store_true",
                            help="also overwrite the Pi's imu_calib.yaml / joint_servo_map.yaml when they differ")
        if name == "run":
            sp.add_argument("--keep-pi-calibration", action="store_true",
                            help="run on the Pi's own imu_calib.yaml / joint_servo_map.yaml when they differ "
                                 "from the laptop's (without this, or --push-calibration, run refuses)")
            sp.add_argument("--no-tmux", action="store_true",
                            help="plain ssh -t: an SSH drop ends the run and releases torque, as before")
        if name in ("run", "attach", "fetch"):
            sp.add_argument("--no-report", action="store_true", help="fetch only; skip diagnose_run/run_report")
    args = p.parse_args(argv)

    try:
        if run_args and args.cmd not in ("sync", "run"):
            raise Abort(f"`{args.cmd}` takes no deploy-script arguments (got {run_args})")
        cfg = load_config(args.config) if cfg is None else check_config(cfg)
        robot = Robot(cfg, dry_run=args.dry_run, shell=shell, echo=echo)
        report = not getattr(args, "no_report", False)
        if args.cmd in ("sync", "run"):
            if not (ROOT / "scripts" / "deploy" / args.script).is_file():
                raise Abort(f"scripts/deploy/{args.script} does not exist")
            files, run_args = run_files(run_args, robot.repo, script_flags(args.script))
            base = deploy_files()
            synced = sync(robot, base + [f for f in files if f not in base],
                          push_calibration=args.push_calibration)
            if args.cmd == "run":
                map_path = keep_or_refuse(robot, synced, files, args.keep_pi_calibration)
                run_args = with_log(args.script, run_args, echo)
                if args.no_tmux:
                    robot.interactive(robot.ssh_argv(remote_command(robot, args.script, run_args), tty=True))
                    fetch(robot, report=report, map_path=map_path)
                else:
                    start_in_tmux(robot, args.script, run_args)
                    attach_and_collect(robot, report=report, map_path=map_path)
        elif args.cmd == "attach":
            attach_and_collect(robot, report=report)
        elif args.cmd == "stop":
            robot.call(robot.ssh_argv(f"tmux send-keys -t ={robot.session}: C-c"))
            echo(f"sent Ctrl-C to tmux session '{robot.session}': the run ramps home and releases "
                 "torque. `robot_run.py attach` to watch it finish and fetch the logs.")
        elif args.cmd == "fetch":
            fetch(robot, report=report)
    except Abort as e:
        print(f"robot_run: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
