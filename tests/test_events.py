"""Operator event markers (scripts/deploy/events.py) and their wiring into the deploy scripts.

No terminal is touched: a fake termios checks the mode TtyKeys sets and restores, a key queue
stands in for the SSH terminal, and the deploy scripts run against the fake servo bus, IMU and
ADC boards from test_frame_log.py.
"""
import copy
import io
import json
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

DEPLOY = Path("scripts/deploy").resolve()
sys.path.insert(0, str(DEPLOY))

import events  # noqa: E402
import frame_log  # noqa: E402
import hardware  # noqa: E402
from test_frame_log import DOFS, _npz, _read, _run, _stance, robot  # noqa: E402,F401

REAL_RELEASE_ON_HANGUP = hardware.release_on_hangup


# Several tests stub time.sleep to a no-op; waits that must really yield use this.
_REAL_SLEEP = time.sleep


# ---- fakes ---------------------------------------------------------------------------------

class FakeKeys:
    """The SSH terminal as a key queue. Every enter/restore is counted and, with `log`,
    appended to it, so a test can check the order against torque-off."""

    def __init__(self, log=None, restore_error=None):
        self.q = queue.Queue()
        self.entered = 0
        self.restored = 0
        self.recorded = 0      # events the recorder actually kept (see the keys fixture)
        self.log = log
        self.restore_error = restore_error
        self.on_restore = None

    def press(self, ch):
        self.q.put(ch)

    def enter(self):
        self.entered += 1

    def read_key(self, timeout, stop=None):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def restore(self):
        self.restored += 1
        if self.on_restore is not None:
            self.on_restore()
        if self.log is not None:
            self.log.append(("restore",))
        if self.restore_error is not None:
            raise self.restore_error


class FakeTermios:
    ISIG, ICANON, ECHO, IEXTEN = 0x1, 0x2, 0x8, 0x8000
    VTIME, VMIN, VSUSP, VQUIT = 5, 6, 10, 1
    TCSANOW = 0

    class error(Exception):
        pass

    def __init__(self):
        lflag = self.ISIG | self.ICANON | self.ECHO | self.IEXTEN
        self.attrs = [0x500, 0x5, 0xBF, lflag, 15, 15, [b"\x1a"] * 32]
        self.sets = []
        self.fail_set = False

    def tcgetattr(self, fd):
        return copy.deepcopy(self.attrs)

    def tcsetattr(self, fd, when, attrs):
        if self.fail_set:
            raise self.error(5, "Input/output error")
        self.sets.append(copy.deepcopy(attrs))
        self.attrs = copy.deepcopy(attrs)


class FakeSelect:
    """Ready while `buf` (the bytes the terminal has) is not empty; otherwise waits a little of
    the timeout, as select() would, so a reader thread over it does not spin."""

    def __init__(self, buf):
        self.buf = buf

    def select(self, r, w, x, timeout):
        if not self.buf and timeout:
            time.sleep(min(timeout, 0.005))
        return (r if self.buf else []), [], []


def _tty(buf, termios_mod=None):
    """A TtyKeys over a byte list: the operator types by appending to `buf`."""
    return events.TtyKeys(7, termios_mod or FakeTermios(), FakeSelect(buf),
                          read=lambda fd, n: buf.pop(0) if buf else b"")


def _wait_for(cond, timeout=2.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


# ---- the terminal --------------------------------------------------------------------------

def test_cbreak_keeps_ctrl_c_disables_ctrl_z_and_restores_the_exact_mode():
    t = FakeTermios()
    before = copy.deepcopy(t.attrs)
    k = events.TtyKeys(7, t, FakeSelect([]))
    k.enter()
    lflag = t.attrs[3]
    assert not lflag & t.ICANON and not lflag & t.ECHO    # key by key, not echoed
    assert lflag & t.ISIG                                 # Ctrl-C still raises SIGINT
    assert t.attrs[6][t.VMIN] == 1 and t.attrs[6][t.VTIME] == 0
    assert t.attrs[6][t.VSUSP] == b"\x00"                 # Ctrl-Z cannot freeze the loop
    assert t.attrs[6][t.VQUIT] == b"\x00"                 # Ctrl-backslash cannot skip torque-off
    k.restore()
    assert t.attrs == before
    k.restore()                                           # idempotent: one restore only
    assert len(t.sets) == 2


def test_read_key_returns_a_char_none_on_timeout_and_empty_when_the_terminal_is_gone():
    buf = [b"f"]
    k = events.TtyKeys(7, FakeTermios(), FakeSelect(buf), read=lambda fd, n: buf.pop(0) if buf else b"")
    assert k.read_key(0.0) == "f"
    assert k.read_key(0.0) is None
    buf.append(b"")                                       # EOF after a hangup
    assert k.read_key(0.0) == ""

    def eio(fd, n):
        raise OSError(5, "Input/output error")
    k2 = events.TtyKeys(7, FakeTermios(), FakeSelect([b"x"]), read=eio)
    assert k2.read_key(0.0) == ""


@pytest.mark.parametrize("seq", [b"\x1b[A", b"\x1b[B", b"\x1b[C", b"\x1b[D", b"\x1b[F",
                                 b"\x1bOF", b"\x1b[H", b"\x1b[1;5C", b"\x1b[3~", b"\x1b"])
def test_arrow_and_navigation_keys_are_dropped_whole(seq):
    # Down = ESC [ B, Right = ESC [ C, End = ESC [ F / ESC O F: byte by byte they would be
    # push_back, caught and push_forward.
    buf = [bytes([b]) for b in seq]
    k = _tty(buf)
    assert k.read_key(0.0) is None and buf == []
    buf.append(b"F")                                       # a real Shift+F afterwards still counts
    assert k.read_key(0.0) == "F"


def test_a_closing_reader_leaves_the_next_byte_to_the_operator():
    buf = [b"p"]                                           # their answer to the after-run question
    stop = threading.Event()
    stop.set()
    assert _tty(buf).read_key(0.0, stop) is None and buf == [b"p"]


def test_reader_thread_ignores_escape_sequences_and_keeps_real_keys(tmp_path):
    buf = []
    rec = events.EventRecorder(tmp_path / "e.jsonl", t0=0.0, keys=_tty(buf)).start()
    for chunk in (b"\x1b[B", b"\x1b[C", b"\x1b[F", b"\x1bOF", b"m", b"\x1b[A", b"x"):
        buf.extend(bytes([b]) for b in chunk)              # one SSH packet per key
        assert _wait_for(lambda: not buf)
        time.sleep(2 * events.ESC_DRAIN_S)                 # the operator's next key comes later
    rec.close()
    assert [e["label"] for e in rec.events] == ["mark", "fell"]


def test_no_terminal_means_no_keys_and_says_why(tmp_path, monkeypatch, capsys):
    monkeypatch.undo()          # the real TtyKeys.open (conftest's guard); every input is a non-tty
    keys, why = events.TtyKeys.open(io.StringIO())
    assert keys is None and why
    with open(tmp_path / "f.txt", "w") as f:              # a real fd that is not a tty
        keys, why = events.TtyKeys.open(f)
    assert keys is None and why

    rec = events.EventRecorder(tmp_path / "run.events.jsonl", stdin=io.StringIO()).start()
    assert rec.enabled is False and rec.reason
    assert "[events] off:" in capsys.readouterr().out

    def never(prompt):
        raise AssertionError("prompted without a terminal")
    summary = rec.finish(input_fn=never)
    assert summary["enabled"] is False and summary["count"] == 0 and summary["file"] is None
    assert not (tmp_path / "run.events.jsonl").exists()


def test_a_terminal_that_refuses_cbreak_runs_without_keys(tmp_path):
    class Refuses(FakeKeys):
        def enter(self):
            raise OSError(25, "Inappropriate ioctl for device")
    keys = Refuses()
    rec = events.EventRecorder(tmp_path / "e.jsonl", keys=keys).start()
    assert rec.enabled is False and "cbreak" in rec.reason
    rec.close()
    assert keys.restored == 0                              # nothing was changed, nothing to undo


# ---- the recorder --------------------------------------------------------------------------

def test_events_take_the_nearest_logged_step(tmp_path, capsys):
    rec = events.EventRecorder(tmp_path / "e.jsonl", t0=100.0, keys=FakeKeys())
    rec.key("m", t_wall=99.5)                              # during the ramp-in: no frame yet
    for step, t in [(0, 100.000), (1, 100.025)]:
        rec.on_row((t, t - 100.0, step))
    rec.key("f", t_wall=100.030)                           # closer to step 1 than to step 2
    rec.key("B", t_wall=100.045)                           # upper case works; closer to step 2
    rec.key("q", t_wall=100.046)                           # not a key: ignored, with a hint
    rec.key(" ", t_wall=100.046)
    rec.on_row((100.050, 0.05, 2))
    rec.key("x", t_wall=100.070)                           # after the last frame
    out = capsys.readouterr().out
    assert "'q' is not a key" in out and "[event] push_forward" in out
    rec.save()
    lines = events.read_events(tmp_path / "e.jsonl")
    assert [e["label"] for e in lines] == ["mark", "push_forward", "push_back", "fell"]
    assert [e["step"] for e in lines] == [0, 1, 2, 2]
    assert [e["key"] for e in lines] == ["m", "f", "b", "x"]
    assert all(list(e) == list(events.EVENT_FIELDS) for e in lines)
    assert lines[1]["t_wall"] == 100.030 and lines[1]["t_rel"] == pytest.approx(0.030)
    raw = (tmp_path / "e.jsonl").read_text().splitlines()
    assert all(json.loads(s) for s in raw) and len(raw) == 4


def test_each_key_is_on_disk_the_moment_it_is_pressed(tmp_path):
    rec = events.EventRecorder(tmp_path / "e.jsonl", t0=0.0, keys=FakeKeys())
    rec.on_row((10.0, 10.0, 400))
    rec.key("c", t_wall=10.01)
    lines = events.read_events(tmp_path / "e.jsonl")      # before any save(): a crash keeps it
    assert lines == [{"t_wall": 10.01, "t_rel": 10.01, "step": 400, "key": "c",
                      "label": "caught", "note": ""}]


def test_reader_thread_records_keys_and_close_restores_once(tmp_path, capsys):
    keys = FakeKeys()
    rec = events.EventRecorder(tmp_path / "e.jsonl", keys=keys).start()
    assert keys.entered == 1 and rec.enabled
    assert "[events] keys:" in capsys.readouterr().out     # the legend
    rec.on_row((time.time(), 0.0, 0))
    keys.press("k")
    keys.press("n")
    assert _wait_for(lambda: len(rec.events) == 2)
    alive_at_restore = []
    keys.on_restore = lambda: alive_at_restore.append(rec._thread.is_alive())
    rec.close()
    rec.close()
    rec._restore()                                          # the atexit backstop after close()
    assert keys.restored == 1
    assert alive_at_restore == [False]                      # stopped BEFORE line mode is back
    assert not rec._thread.is_alive()
    keys.press("x")                                         # after close: not read
    time.sleep(0.15)
    assert [e["label"] for e in rec.events] == ["kick", "note"]


def test_finish_asks_for_the_n_notes_and_the_run_note(tmp_path):
    keys = FakeKeys()
    rec = events.EventRecorder(tmp_path / "e.jsonl", t0=0.0, keys=keys).start()
    rec.on_row((5.0, 5.0, 200))
    rec.key("n", t_wall=5.0)
    rec.key("f", t_wall=5.0)
    rec.key("n", t_wall=5.0)
    answers = iter(["left foot slid", "", "pushed twice, caught it once"])
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return next(answers)
    summary = rec.finish(input_fn=ask)
    assert keys.restored == 1 and len(asked) == 3
    assert "What did you see?" in asked[-1]
    lines = events.read_events(tmp_path / "e.jsonl")
    assert [(e["label"], e["note"]) for e in lines] == [
        ("note", "left foot slid"), ("push_forward", ""), ("note", ""),
        ("run_note", "pushed twice, caught it once")]
    assert lines[-1]["key"] == "" and lines[-1]["step"] == 200
    assert summary["count"] == 4 and summary["counts"]["note"] == 2
    assert summary["operator_note"] == "pushed twice, caught it once"
    assert summary["file"] == "e.jsonl" and summary["enabled"] is True


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_a_prompt_that_is_abandoned_still_writes_the_events(tmp_path, exc):
    rec = events.EventRecorder(tmp_path / "e.jsonl", t0=0.0, keys=FakeKeys()).start()
    rec.key("x", t_wall=1.0)

    def ask(prompt):
        raise exc
    summary = rec.finish(input_fn=ask)
    assert summary["count"] == 1 and summary["operator_note"] is None
    assert events.read_events(tmp_path / "e.jsonl")[0]["label"] == "fell"


def test_an_unanswered_question_gives_up(tmp_path, monkeypatch, capsys):
    class Unattended(FakeKeys):
        def line_ready(self, timeout):
            assert timeout == events.PROMPT_TIMEOUT_S
            return False                                   # nobody typed anything in time

    def never(prompt=""):
        raise AssertionError("read a line nobody typed")
    monkeypatch.setattr("builtins.input", never)
    rec = events.EventRecorder(tmp_path / "e.jsonl", t0=0.0, keys=Unattended()).start()
    rec.key("n", t_wall=1.0)
    summary = rec.finish()
    assert "note prompt skipped (no answer in 120 s)" in capsys.readouterr().out
    assert summary["count"] == 1 and summary["operator_note"] is None
    assert events.read_events(tmp_path / "e.jsonl")[0]["label"] == "note"


def test_the_question_reads_a_line_once_one_is_waiting(tmp_path, monkeypatch):
    class Answered(FakeKeys):
        def line_ready(self, timeout):
            return True
    monkeypatch.setattr("builtins.input", lambda prompt="": "it leaned left")
    rec = events.EventRecorder(tmp_path / "e.jsonl", t0=0.0, keys=Answered()).start()
    assert rec.finish()["operator_note"] == "it leaned left"


def test_a_run_stopped_by_a_signal_never_asks(tmp_path):
    # hardware.release_on_hangup() leaves SIGTERM ignored once a SIGTERM or SIGHUP has stopped
    # the run: nobody may be at the terminal, and a second kill would do nothing.
    def never(prompt):
        raise AssertionError("asked after a kill")
    saved = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        assert events.stopped_by_signal()
        rec = events.EventRecorder(tmp_path / "e.jsonl", keys=FakeKeys()).start()
        rec.key("f")
        assert rec.finish(input_fn=never)["count"] == 1
    finally:
        signal.signal(signal.SIGTERM, saved)
    assert not events.stopped_by_signal()


def test_no_prompt_and_a_hung_up_terminal_never_ask(tmp_path):
    def never(prompt):
        raise AssertionError("asked")
    rec = events.EventRecorder(tmp_path / "a.jsonl", keys=FakeKeys()).start()
    rec.key("m")
    assert rec.finish(prompt=False, input_fn=never)["count"] == 1

    keys = FakeKeys(restore_error=OSError(5, "Input/output error"))
    rec = events.EventRecorder(tmp_path / "b.jsonl", keys=keys).start()
    keys.press("c")
    keys.press("")                                         # EOF: the SSH session dropped
    assert _wait_for(lambda: rec._gone)
    assert rec.finish(input_fn=never)["count"] == 1       # restore raised and was swallowed
    assert keys.restored == 1


# ---- wired into the deploy scripts ---------------------------------------------------------

@pytest.fixture
def keys(monkeypatch, robot):  # noqa: F811
    """A fake terminal the deploy scripts will open, its restore logged next to torque-off."""
    k = FakeKeys(log=robot["events"])
    monkeypatch.setattr(events.TtyKeys, "open", classmethod(lambda cls, stdin=None: (k, None)))
    real_key = events.EventRecorder.key

    def key(self, ch, t_wall=None):
        ev = real_key(self, ch, t_wall)
        if ev is not None:
            k.recorded += 1
        return ev
    monkeypatch.setattr(events.EventRecorder, "key", key)
    return k


@pytest.fixture
def answers(monkeypatch, robot):  # noqa: F811
    """builtins.input, answering the after-run prompts; each prompt logged in robot events."""
    state = {"replies": [], "asked": []}

    def ask(prompt=""):
        state["asked"].append(prompt)
        robot["events"].append(("prompt",))
        return state["replies"].pop(0) if state["replies"] else ""
    monkeypatch.setattr("builtins.input", ask)
    return state


def _press_at_accel_reads(robot, keys, schedule, raise_at=None):  # noqa: F811
    """Press keys (from the control thread, as the operator would mid-frame) at given IMU reads."""
    i2c = robot["i2c"]
    real = i2c.read_i2c_block_data

    def read(addr, reg, n):
        out = real(addr, reg, n)
        if addr == 0x68 and reg == hardware.IMU.ACCEL_XOUT_H:
            if i2c.accel_reads in schedule:
                want = keys.recorded + 1
                keys.press(schedule[i2c.accel_reads])
                # Wait until the reader thread has RECORDED it, not just taken it off the queue:
                # a key taken once the run is stopping is dropped by design. A plain sleep here
                # was a no-op in the tests that stub time.sleep, so CI raced the stop.
                t_end = time.time() + 2.0
                while keys.recorded < want and time.time() < t_end:
                    _REAL_SLEEP(0.001)
            if raise_at is not None and i2c.accel_reads == raise_at[0]:
                raise_at[1]()
        return out
    i2c.read_i2c_block_data = read


def _assert_nearest(rows, evs):
    for e in evs:
        nearest = min(rows, key=lambda r: abs(float(r["t_wall"]) - e["t_wall"]))
        assert e["step"] == int(nearest["step"]), e


def _hold_args(tmp_path, log, *extra):
    return ["--policy-npz", str(_npz(tmp_path / "hold.npz", _stance())),
            "--residual-config", "config/stances/knee2.yaml", "--hold-pose",
            "--ramp-secs", "0.05", "--imu-calib", str(tmp_path / "none.yaml"),
            "--log", str(log), *extra]


def test_deploy_marks_keys_against_the_logged_frames_and_asks_after_torque_off(
        tmp_path, monkeypatch, robot, keys, answers):  # noqa: F811
    robot["i2c"].tilt_after = 40
    _press_at_accel_reads(robot, keys, {15: "f", 30: "c", 42: "x"})
    answers["replies"] = ["pushed it forward, it went over"]
    log = tmp_path / "hold.csv"
    _run(monkeypatch, "deploy_standing", _hold_args(tmp_path, log))
    rows = _read(log)
    evs = events.read_events(frame_log.sidecar_path(log, "events.jsonl"))
    assert [e["label"] for e in evs] == ["push_forward", "caught", "fell", "run_note"]
    _assert_nearest(rows, evs[:3])
    t0 = float(rows[0]["t_wall"]) - float(rows[0]["t_rel"])
    for e in evs:
        assert e["t_rel"] == pytest.approx(e["t_wall"] - t0, abs=1e-6)   # the CSV's t_rel clock
    assert evs[-1]["note"] == "pushed it forward, it went over"
    assert evs[-1]["step"] == int(rows[-1]["step"])
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    assert meta["events"]["count"] == 4 and meta["events"]["file"] == "hold.events.jsonl"
    assert meta["events"]["operator_note"] == "pushed it forward, it went over"
    assert meta["sidecars"]["events"] == "hold.events.jsonl"
    assert meta["files"]["source:events.py"]["md5"]
    console = frame_log.sidecar_path(log, "console.txt").read_text()
    assert "[events] keys:" in console and "[event] push_forward" in console
    # torque off -> terminal restored -> log closed -> only then the question
    ev = robot["events"]
    assert ev.index(("torque", False)) < ev.index(("restore",)) < ev.index(("log_close",)) \
        < ev.index(("prompt",))
    assert keys.entered == 1 and keys.restored == 1
    assert set(robot["bus"]._ser.torque.values()) == {0}


def test_deploy_ctrl_c_restores_the_terminal_after_torque_off(
        tmp_path, monkeypatch, robot, keys, answers):  # noqa: F811
    monkeypatch.setattr(time, "sleep", lambda s: None)
    robot["i2c"].tilt_after = None
    robot["i2c"].interrupt_at = 20
    log = tmp_path / "stop.csv"
    _run(monkeypatch, "deploy_standing", _hold_args(tmp_path, log, "--no-prompt"))
    ev = robot["events"]
    assert ev.index(("torque", False)) < ev.index(("restore",)) < ev.index(("log_close",))
    assert keys.restored == 1 and not answers["asked"]
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    assert meta["events"]["enabled"] is True and meta["events"]["count"] == 0


def _raise_signal_at(monkeypatch, robot, keys, signame, at):  # noqa: F811
    """Install the REAL release_on_hangup and deliver `signame` at accel read `at`. A SIGHUP
    also hangs up the fake terminal (its reader sees EOF, its restore fails), as SSH does.
    Returns a callable that puts the signal handlers back."""
    sig = getattr(signal, signame, None)
    if sig is None:
        pytest.skip(f"no {signame} on this platform")
    monkeypatch.setattr(hardware, "release_on_hangup", REAL_RELEASE_ON_HANGUP)
    saved = {s: signal.getsignal(s) for s in (getattr(signal, "SIGHUP", None), signal.SIGTERM)
             if s is not None}

    def deliver():
        if signame == "SIGHUP":
            keys.restore_error = OSError(5, "Input/output error")   # the terminal is gone
            keys.press("")                                         # its reader gets EOF
        signal.raise_signal(sig)

    def put_back():
        for s, h in saved.items():
            signal.signal(s, h)
    robot["i2c"].tilt_after = None
    _press_at_accel_reads(robot, keys, {12: "f"}, raise_at=(at, deliver))
    return put_back


@pytest.mark.parametrize("no_prompt", [True, False])
@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_deploy_sigterm_and_sighup_restore_the_terminal_and_never_ask(
        tmp_path, monkeypatch, robot, keys, answers, signame, no_prompt):  # noqa: F811
    # Without --no-prompt too: a run killed from another session, by a timeout or by the
    # laptop tool has nobody at its terminal, and a second kill is ignored -- a question here
    # would hold the process forever.
    put_back = _raise_signal_at(monkeypatch, robot, keys, signame, 25)
    log = tmp_path / "sig.csv"
    try:
        _run(monkeypatch, "deploy_standing",
             _hold_args(tmp_path, log, *(["--no-prompt"] if no_prompt else [])))
    finally:
        put_back()
    ev = robot["events"]
    assert ev.index(("torque", False)) < ev.index(("restore",)) < ev.index(("log_close",))
    assert keys.restored == 1 and answers["asked"] == []
    assert set(robot["bus"]._ser.torque.values()) == {0}
    evs = events.read_events(frame_log.sidecar_path(log, "events.jsonl"))
    assert [e["label"] for e in evs] == ["push_forward"]
    _assert_nearest(_read(log), evs)
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    assert meta["events"]["count"] == 1 and meta["events"]["operator_note"] is None


def test_deploy_never_asks_while_torque_may_still_be_on(
        tmp_path, monkeypatch, robot, keys, answers):  # noqa: F811
    # torque-off fails, and so does the retry: the loud traceback must not wait behind
    # "what did you see?", and the operator must not be told the run is over.
    monkeypatch.setattr(time, "sleep", lambda s: None)
    real = hardware.ServoBus.set_torque

    def set_torque(self, servo_ids, enable):
        if not enable:
            robot["events"].append(("torque", False))
            raise OSError(5, "servo bus timeout")
        return real(self, servo_ids, enable)
    monkeypatch.setattr(hardware.ServoBus, "set_torque", set_torque)
    robot["i2c"].tilt_after = None
    _press_at_accel_reads(robot, keys, {12: "c"})
    robot["i2c"].interrupt_at = 20
    log = tmp_path / "stiff.csv"
    with pytest.raises(OSError, match="servo bus timeout"):
        _run(monkeypatch, "deploy_standing", _hold_args(tmp_path, log))
    assert robot["events"].count(("torque", False)) == 2 and answers["asked"] == []
    assert keys.restored == 1
    evs = events.read_events(frame_log.sidecar_path(log, "events.jsonl"))
    assert [e["label"] for e in evs] == ["caught"]
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    assert meta["events"]["count"] == 1


def test_deploy_crash_still_restores_the_terminal_and_keeps_the_events(
        tmp_path, monkeypatch, robot, keys, answers):  # noqa: F811
    robot["i2c"].tilt_after = None

    def boom():
        raise RuntimeError("I2C bus fell over")
    _press_at_accel_reads(robot, keys, {12: "k"}, raise_at=(25, boom))
    log = tmp_path / "crash.csv"
    with pytest.raises(RuntimeError, match="I2C bus fell over"):
        _run(monkeypatch, "deploy_standing", _hold_args(tmp_path, log, "--no-prompt"))
    ev = robot["events"]
    assert ev.index(("torque", False)) < ev.index(("restore",))
    evs = events.read_events(frame_log.sidecar_path(log, "events.jsonl"))
    assert [e["label"] for e in evs] == ["kick"]


def test_deploy_no_events_never_opens_the_terminal(tmp_path, monkeypatch, robot, keys, answers):  # noqa: F811
    log = tmp_path / "quiet.csv"
    _run(monkeypatch, "deploy_standing", _hold_args(tmp_path, log, "--no-events"))
    assert keys.entered == 0 and keys.restored == 0 and not answers["asked"]
    assert not frame_log.sidecar_path(log, "events.jsonl").exists()
    assert "events" not in json.loads(log.with_suffix(".meta.json").read_text())


def test_deploy_without_a_terminal_runs_as_before(tmp_path, monkeypatch, robot, answers):  # noqa: F811
    # conftest makes stdin a non-terminal: the real no-op path.
    log = tmp_path / "piped.csv"
    _run(monkeypatch, "deploy_standing", _hold_args(tmp_path, log))
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    assert meta["events"]["enabled"] is False and meta["events"]["reason"]
    assert meta["rows_written"] > 5 and not answers["asked"]
    assert "[events] off:" in frame_log.sidecar_path(log, "console.txt").read_text()
    assert not frame_log.sidecar_path(log, "events.jsonl").exists()


def test_pose_sweep_marks_keys_and_asks_after_the_result(
        tmp_path, monkeypatch, robot, keys, answers):  # noqa: F811
    robot["i2c"].tilt_after = None
    _press_at_accel_reads(robot, keys, {10: "m", 40: "n"})
    answers["replies"] = ["toe lifted at +2", "fine"]
    log = tmp_path / "sweep.csv"
    _run(monkeypatch, "pose_sweep", [
        "--joints", "waist_pitch", "--range", "-2", "2", "--step", "2", "--dwell", "0.2",
        "--ramp", "0.05", "--imu-calib", str(tmp_path / "none.yaml"), "--log", str(log)])
    rows = _read(log)
    evs = events.read_events(frame_log.sidecar_path(log, "events.jsonl"))
    assert [(e["label"], e["note"]) for e in evs] == [
        ("mark", ""), ("note", "toe lifted at +2"), ("run_note", "fine")]
    _assert_nearest(rows, evs[:2])
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    assert meta["events"]["count"] == 3 and meta["files"]["source:events.py"]["md5"]
    ev = robot["events"]
    assert ev.index(("torque", False)) < ev.index(("restore",)) < ev.index(("log_close",)) \
        < ev.index(("prompt",))
    console = frame_log.sidecar_path(log, "console.txt").read_text()
    assert "--- RESULT ---" in console                    # the question comes after the result


def _sweep_args(tmp_path, log):
    return ["--joints", "waist_pitch", "--range", "-2", "2", "--step", "2", "--dwell", "0.2",
            "--ramp", "0.05", "--imu-calib", str(tmp_path / "none.yaml"), "--log", str(log)]


def test_pose_sweep_ctrl_c_asks_after_the_result(tmp_path, monkeypatch, robot, keys, answers):  # noqa: F811
    robot["i2c"].tilt_after = None
    _press_at_accel_reads(robot, keys, {8: "x"})
    robot["i2c"].interrupt_at = 20
    answers["replies"] = ["went over backwards"]
    log = tmp_path / "sweep_stop.csv"
    _run(monkeypatch, "pose_sweep", _sweep_args(tmp_path, log))
    ev = robot["events"]
    assert ev.index(("torque", False)) < ev.index(("restore",)) < ev.index(("log_close",)) \
        < ev.index(("prompt",))
    evs = events.read_events(frame_log.sidecar_path(log, "events.jsonl"))
    assert [(e["label"], e["note"]) for e in evs] == [("fell", ""), ("run_note", "went over backwards")]
    console = frame_log.sidecar_path(log, "console.txt").read_text()
    assert "interrupted." in console and "--- RESULT ---" in console


def test_pose_sweep_crash_keeps_the_events_and_never_asks(
        tmp_path, monkeypatch, robot, keys, answers):  # noqa: F811
    robot["i2c"].tilt_after = None
    _press_at_accel_reads(robot, keys, {3: "k"})
    real = hardware.ServoBus.write_all
    calls = {"n": 0}

    def write_all(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 4:                               # once, mid-sweep (the ramp back works)
            raise RuntimeError("servo bus fell over")
        return real(self, *a, **k)
    monkeypatch.setattr(hardware.ServoBus, "write_all", write_all)
    log = tmp_path / "sweep_crash.csv"
    with pytest.raises(RuntimeError, match="servo bus fell over"):
        _run(monkeypatch, "pose_sweep", _sweep_args(tmp_path, log))
    ev = robot["events"]
    assert ev.index(("torque", False)) < ev.index(("restore",)) < ev.index(("log_close",))
    assert answers["asked"] == []
    evs = events.read_events(frame_log.sidecar_path(log, "events.jsonl"))
    assert [e["label"] for e in evs] == ["kick"]


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_pose_sweep_signal_restores_the_terminal_and_never_asks(
        tmp_path, monkeypatch, robot, keys, answers, signame):  # noqa: F811
    put_back = _raise_signal_at(monkeypatch, robot, keys, signame, 20)
    log = tmp_path / "sweep_sig.csv"
    try:
        _run(monkeypatch, "pose_sweep", _sweep_args(tmp_path, log))
    finally:
        put_back()
    ev = robot["events"]
    assert ev.index(("torque", False)) < ev.index(("restore",)) < ev.index(("log_close",))
    assert keys.restored == 1 and answers["asked"] == []
    evs = events.read_events(frame_log.sidecar_path(log, "events.jsonl"))
    assert [e["label"] for e in evs] == ["push_forward"]
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    assert meta["events"]["count"] == 1 and meta["events"]["operator_note"] is None
