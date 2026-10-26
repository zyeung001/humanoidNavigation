import importlib
import io
import os
import sys
from pathlib import Path

import pytest

# Configs and MJCFs name their files relative to the repo root (xml_file, actuator_profiles),
# exactly as the training scripts are run. Run the tests from there too.
os.chdir(Path(__file__).resolve().parents[1])

DEPLOY = Path(__file__).resolve().parents[1] / "scripts" / "deploy"


def _deploy_events_module():
    """scripts/deploy/events.py, imported once without leaving scripts/deploy on sys.path."""
    if "events" not in sys.modules:
        sys.path.insert(0, str(DEPLOY))
        try:
            importlib.import_module("events")
        finally:
            sys.path.remove(str(DEPLOY))
    return sys.modules["events"]


@pytest.fixture(autouse=True)
def _no_operator_terminal(monkeypatch):
    """The deploy scripts read operator keys from the real terminal (scripts/deploy/events.py).
    A test run from one (pytest -s) must never switch it to cbreak or sit at the after-run
    prompt, so every test sees stdin as not-a-terminal: the scripts take their real no-op path.
    Tests that want keys inject a fake key source over this."""
    ev = _deploy_events_module()
    real_open = ev.TtyKeys.open
    monkeypatch.setattr(ev.TtyKeys, "open", classmethod(lambda cls, stdin=None: real_open(io.StringIO())))
