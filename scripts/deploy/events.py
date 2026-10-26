#!/usr/bin/env python3
# events.py  --  RUN ON THE PI (imported by the deploy scripts)
"""Operator event markers: one keypress on the SSH terminal stamps what just happened.

WHY. The frame log records what the robot did, never what was done TO it. A push, a hand
that caught it, a foot kicked by accident -- all of them look like the robot moving on its
own in the CSV, and a catch looks like a recovery. On 10/25 the robot took ~20 pushes
untethered and the log cannot say when any of them happened; the only way to line them up
afterwards was to guess from spikes in acc_*. The operator knows the instant it happens. One
key costs them nothing and turns "it fell at some point after a push" into "push_forward at
t_rel 41.20, fell at 42.05", directly against the frames.

KEYS (a one-line legend is printed when the run starts; each event is echoed as recorded):
  f push_forward   b push_back   l push_left   r push_right
  x fell           c caught      k kick (any impact)
  m mark           generic "look here"
  n note           stamps the time now; the text is asked for after the run, when the
                   robot is safe and both hands are free
Ctrl-C still stops the run: the terminal is put in cbreak mode, NOT raw, so the signal keys
keep working -- except Ctrl-Z, which is disabled while keys are read. Suspending a 40 Hz
balance loop freezes the robot at full torque on its last command.

OUTPUT: <stem>.events.jsonl next to the frame log, one JSON object per line with exactly
  {"t_wall", "t_rel", "step", "key", "label", "note"}
  t_wall  unix seconds at the keypress (the clock every stream is aligned on)
  t_rel   t_wall minus the frame log's t0, so it reads against the CSV's t_rel directly
  step    the step of the LOGGED FRAME nearest in time to the keypress (a key pressed during
          the ramp-in, before any frame, gets the first frame; one after the last, the last)
  key     the key pressed (lower case); "" for the after-run note
  label   what the key means (KEYS below); "run_note" for the after-run note
  note    free text: the 'n' notes and the after-run "what did you see?" answer, else ""
After the run (torque already off) the operator is asked for each 'n' note and for a one-line
"what did you see?", whose line has the run's last step. Each event is also appended to the
file the moment it is pressed, so a run that dies still leaves its markers; the file is
rewritten at the end with the exact steps and the notes.

NEVER IN THE WAY OF THE ROBOT:
  - The keys are read on their own thread. The control loop does nothing at all for them:
    the step is matched on the frame logger's WRITER thread (attach on_row with add_sink).
  - The terminal mode is restored on every exit: the deploy script's `finally` calls close()
    right AFTER torque-off (never before), and an atexit hook backs that up. SIGTERM and
    SIGHUP arrive as KeyboardInterrupt through hardware.release_on_hangup(), so they take the
    same path; after a hangup the terminal is gone and restoring it is skipped quietly.
  - No terminal, no keys: stdin piped, `ssh pi cmd` without -t, nohup, Windows (no termios)
    or a run in the background (reading the terminal would STOP the process, robot torqued)
    -- each prints one line saying why and the run goes on exactly as before.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from pathlib import Path

# Order is the legend's order.
KEYS = {
    "f": "push_forward", "b": "push_back", "l": "push_left", "r": "push_right",
    "x": "fell", "c": "caught", "k": "kick", "m": "mark", "n": "note",
}
EVENT_FIELDS = ("t_wall", "t_rel", "step", "key", "label", "note")   # the shared data contract
RUN_NOTE_LABEL = "run_note"
LEGEND = ("[events] keys: f/b/l/r push fwd/back/left/right | x fell | c caught | "
          "k kick/impact | m mark | n note (text after the run) | Ctrl-C stops as usual")


class TtyKeys:
    """Single keypresses from the controlling terminal, in cbreak mode.

    cbreak, not raw: keys arrive one at a time without Enter and are not echoed, but output
    processing and the signal keys are left alone -- Ctrl-C must still stop the run. Ctrl-Z
    is the exception (see the module docstring)."""

    def __init__(self, fd, termios_mod, select_mod, read=os.read):
        self.fd = fd
        self._termios = termios_mod
        self._select = select_mod
        self._read = read
        self._saved = None

    @classmethod
    def open(cls, stdin=None):
        """(TtyKeys, None) when keys can be read safely, else (None, why not)."""
        try:
            import select  # noqa: PLC0415
            import termios  # noqa: PLC0415
        except ImportError:
            return None, "termios is not available here (Windows?)"
        stdin = sys.stdin if stdin is None else stdin
        try:
            fd = stdin.fileno()
        except (AttributeError, OSError, ValueError):
            return None, "stdin has no file descriptor"
        if not os.isatty(fd):
            return None, "stdin is not a terminal (piped, nohup, or ssh without -t)"
        try:
            # A background job that touches its terminal is stopped by SIGTTIN/SIGTTOU -- the
            # whole process, control loop included, with the servos still torqued.
            if os.tcgetpgrp(fd) != os.getpgrp():
                return None, "the run is in the background (reading keys would stop it)"
        except OSError:
            return None, "cannot tell whether the run owns the terminal"
        return cls(fd, termios, select), None

    def enter(self):
        t = self._termios
        self._saved = t.tcgetattr(self.fd)
        mode = t.tcgetattr(self.fd)
        mode[3] &= ~(t.ICANON | t.ECHO)          # lflag: key by key, not echoed; ISIG kept
        mode[6][t.VMIN] = 1
        mode[6][t.VTIME] = 0
        if hasattr(t, "VSUSP"):
            mode[6][t.VSUSP] = b"\x00"           # _POSIX_VDISABLE: Ctrl-Z does nothing
        t.tcsetattr(self.fd, t.TCSANOW, mode)

    def read_key(self, timeout):
        """One character, None if nothing arrived within timeout, "" if the terminal is gone."""
        try:
            ready, _, _ = self._select.select([self.fd], [], [], timeout)
            if not ready:
                return None
            data = self._read(self.fd, 1)
        except (OSError, ValueError):
            return ""
        return data.decode("latin-1") if data else ""

    def restore(self):
        """Put the terminal back as it was. Raises if it cannot (the caller swallows that)."""
        if self._saved is not None:
            saved, self._saved = self._saved, None
            self._termios.tcsetattr(self.fd, self._termios.TCSANOW, saved)


def _nearest_step(prev, cur, t):
    """Step of whichever logged frame, prev or cur (each (step, t_wall) or None), is closer to t."""
    if prev is None:
        return None if cur is None else cur[0]
    if cur is None:
        return prev[0]
    return prev[0] if abs(t - prev[1]) <= abs(cur[1] - t) else cur[0]


class EventRecorder:
    """Read operator keys during a run and write them to <stem>.events.jsonl.

        events = EventRecorder(sidecar_path(log_path, "events.jsonl"), t0=logger.t0)
        logger.add_sink(events.on_row)      # step matching, on the writer thread
        events.start()                      # legend; keys from now on
        ...                                 # the run
        finally: torque off; events.close() # terminal restored
        ...      events.finish()            # after-run notes, file written; -> meta summary

    keys: a key source with enter() / read_key(timeout) / restore() -- the terminal when None
    (TtyKeys.open), a fake in the tests.
    """

    def __init__(self, path, t0=None, keys=None, stdin=None):
        self.path = Path(path)
        self.t0 = time.time() if t0 is None else float(t0)
        self.events = []
        self.enabled = False
        self.reason = None
        self.operator_note = None
        self._keys = keys
        self._stdin = stdin
        self._lock = threading.Lock()      # events/_open/_prev: reader vs writer thread
        self._prev = None                  # (step, t_wall) of the newest logged frame
        self._open = []                    # events still waiting for the frame after them
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._restored = False
        self._gone = False                 # the terminal hung up: no after-run prompt

    # ---- run time ----------------------------------------------------------------------------

    def start(self):
        """Begin reading keys, or say in one line why this run has none. Never raises."""
        try:
            self.path.unlink()             # a stale file from an earlier run on the same --log
        except OSError:
            pass
        reason = None
        if self._keys is None:
            self._keys, reason = TtyKeys.open(self._stdin)
        if self._keys is not None:
            try:
                self._keys.enter()
            except Exception as e:        # a terminal that refuses cbreak: run without keys
                self._keys, reason = None, f"could not set the terminal to cbreak ({e!r})"
        if self._keys is None:
            self.reason = reason
            print(f"[events] off: {reason}. No keypress markers this run.", flush=True)
            return self
        self.enabled = True
        atexit.register(self._restore)     # backstop: close() normally does it first
        self._thread = threading.Thread(target=self._run, name="events", daemon=True)
        self._thread.start()
        print(LEGEND, flush=True)
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                ch = self._keys.read_key(0.1)
            except Exception:
                ch = ""
            if ch is None:
                continue
            if ch == "":                   # EOF / EIO: the SSH session is gone
                self._gone = True
                return
            if self._stop.is_set():
                return
            try:
                self.key(ch)
            except Exception:              # a marker must never take anything else down
                pass

    def key(self, ch, t_wall=None):
        """Record one keypress (the reader thread calls this; tests may too). Returns the event,
        or None for a key that means nothing."""
        t = time.time() if t_wall is None else float(t_wall)
        c = ch.lower()
        label = KEYS.get(c)
        if label is None:
            if c.isprintable() and not c.isspace():
                print(f"[events] '{ch}' is not a key. {LEGEND[len('[events] '):]}", flush=True)
            return None
        ev = {"t_wall": t, "t_rel": t - self.t0, "step": None, "key": c, "label": label, "note": ""}
        with self._lock:
            # Provisional: the frame before the key. on_row() replaces it with the nearest
            # frame once the one after it is logged.
            ev["step"] = None if self._prev is None else self._prev[0]
            self.events.append(ev)
            self._open.append(ev)
        step_txt = "before the first frame" if ev["step"] is None else f"step ~{ev['step']}"
        print(f"[event] {label:<12} t_rel {ev['t_rel']:8.2f} s  {step_txt}"
              + ("  (note text after the run)" if c == "n" else ""), flush=True)
        self._append(ev)
        return ev

    def on_row(self, row):
        """FrameLogger sink (writer thread): match open events to their nearest logged frame.

        Rows are (t_wall, t_rel, step, ...). An event resolves when the first frame AFTER it is
        logged: it is then between that frame and the previous one, and takes the closer."""
        t, step = float(row[0]), int(row[2])
        with self._lock:
            if self._open:
                cur = (step, t)
                still = []
                for ev in self._open:
                    if ev["t_wall"] <= t:
                        ev["step"] = _nearest_step(self._prev, cur, ev["t_wall"])
                    else:
                        still.append(ev)
                self._open = still
            self._prev = (step, t)

    def _append(self, ev):
        """Crash safety: the line is on disk the moment the key is pressed (step provisional)."""
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(_line(ev))
        except OSError:
            pass

    # ---- end of run ------------------------------------------------------------------------

    def _restore(self):
        if self._restored or self._keys is None:
            return
        self._restored = True
        try:
            self._keys.restore()
        except Exception:                  # the terminal hung up: there is nothing to restore
            pass

    def close(self):
        """Stop reading keys and restore the terminal. Call it AFTER torque-off. Never raises,
        returns within ~0.1 s."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._restore()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def finish(self, prompt=True, input_fn=None):
        """close(), ask for the notes (unless prompt is False, keys were off, or the terminal is
        gone), write the file, and return the summary for the meta. Call it after torque-off
        AND after the frame log is closed: every frame is then matched, every row is on disk,
        and an operator who never answers costs nothing but the prompt."""
        self.close()
        if prompt and self.enabled and not self._gone:
            ask = input if input_fn is None else input_fn
            try:
                for ev in self.events:
                    if ev["key"] == "n" and not ev["note"]:
                        ev["note"] = ask(f"[events] note for the 'n' at t_rel {ev['t_rel']:.1f} s "
                                         f"(step {ev['step']}): ").strip()
                text = ask("[events] What did you see? One line, Enter to skip: ").strip()
                if text:
                    t = time.time()
                    self.operator_note = text
                    self.events.append({"t_wall": t, "t_rel": t - self.t0,
                                        "step": None if self._prev is None else self._prev[0],
                                        "key": "", "label": RUN_NOTE_LABEL, "note": text})
            except (EOFError, KeyboardInterrupt, OSError, RuntimeError):
                print("\n[events] note prompt skipped", flush=True)
        self.save()
        if self.events:
            print(f"[events] {len(self.events)} event(s) -> {self.path}", flush=True)
        return self.summary()

    def save(self):
        """Write every event with its final step and note (atomic rewrite). Events no frame came
        after take the last frame. Nothing is written for a run without events."""
        with self._lock:
            for ev in self._open:
                ev["step"] = _nearest_step(self._prev, None, ev["t_wall"])
            self._open = []
            events = list(self.events)
        if not events:
            return None
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(_line(ev) for ev in events)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[events] could not write {self.path}: {e!r}", flush=True)
            return None
        return self.path

    def summary(self):
        """What the meta records about the events of this run."""
        counts = {}
        for ev in self.events:
            counts[ev["label"]] = counts.get(ev["label"], 0) + 1
        return {"enabled": self.enabled, "reason": self.reason,
                "file": self.path.name if self.events else None,
                "count": len(self.events), "counts": counts,
                "operator_note": self.operator_note, "keys": dict(KEYS),
                "fields": list(EVENT_FIELDS),
                "step": "step of the logged frame nearest the keypress; run_note = the last frame"}


def _line(ev):
    return json.dumps({k: ev[k] for k in EVENT_FIELDS}) + "\n"


def read_events(path):
    """<stem>.events.jsonl -> list of dicts (empty if the file does not exist)."""
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
