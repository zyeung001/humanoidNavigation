"""Operator event markers (scripts/deploy/events.py).

No terminal is touched: a fake termios checks the mode TtyKeys sets and restores, and a key
queue stands in for the SSH terminal.
"""
import copy
import io
import json
import queue
import sys
import time
from pathlib import Path

import pytest

DEPLOY = Path("scripts/deploy").resolve()
sys.path.insert(0, str(DEPLOY))

import events  # noqa: E402


# ---- fakes ---------------------------------------------------------------------------------

class FakeKeys:
    """The SSH terminal as a key queue. Every enter/restore is counted and, with `log`,
    appended to it, so a test can check the order against torque-off."""

    def __init__(self, log=None, restore_error=None):
        self.q = queue.Queue()
        self.entered = 0
        self.restored = 0
        self.log = log
        self.restore_error = restore_error

    def press(self, ch):
        self.q.put(ch)

    def enter(self):
        self.entered += 1

    def read_key(self, timeout):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def restore(self):
        self.restored += 1
        if self.log is not None:
            self.log.append(("restore",))
        if self.restore_error is not None:
            raise self.restore_error


class FakeTermios:
    ISIG, ICANON, ECHO, IEXTEN = 0x1, 0x2, 0x8, 0x8000
    VTIME, VMIN, VSUSP = 5, 6, 10
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
    def __init__(self, buf):
        self.buf = buf

    def select(self, r, w, x, timeout):
        return (r if self.buf else []), [], []


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
    rec.close()
    rec.close()
    rec._restore()                                          # the atexit backstop after close()
    assert keys.restored == 1
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
