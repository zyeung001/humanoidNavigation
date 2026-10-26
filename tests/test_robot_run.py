"""scripts/robot_run.py against a fake Pi: what gets copied, what gets fetched, how commands are
built and quoted, and that --dry-run never starts a process. No network, no hardware."""

import hashlib
import importlib.util
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest


def _module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _module("scripts/robot_run.py")
numpy_policy = _module("scripts/deploy/numpy_policy.py")

CFG = {"host": "pi-host", "user": "pi", "repo": "/home/pi/humanoidnavigation"}


def _md5(b):
    return hashlib.md5(b).hexdigest()


def _bundle(path, requires=None):
    arrays = numpy_policy.stamp({
        "n_hidden": np.array(0, dtype=np.int64), "out_w": np.zeros((17, 4), np.float32),
        "out_b": np.zeros(17, np.float32), "obs_mean": np.zeros(4, np.float32),
        "obs_var": np.ones(4, np.float32), "clip_obs": np.array(10.0, np.float32),
        "epsilon": np.array(1e-8, np.float32)})
    if requires is not None:
        arrays["requires"] = np.array(requires, dtype=str)
    np.savez(path, **arrays)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A laptop checkout (with a space in its path, like C:/Users/My Name/...)."""
    root = tmp_path / "my robot repo"
    files = {
        "scripts/deploy/deploy_standing.py": b"print('deploy')\n",
        "scripts/deploy/pose_sweep.py": b"print('sweep')\n",
        "scripts/deploy/numpy_policy.py": b"# new\n",
        "config/joint_servo_map.yaml": b"joints: laptop\n",
        "config/imu_calib.yaml": b"axis_remap: laptop\n",
        "config/stances/knee2.yaml": b"standing: {}\n",
        "scripts/debug/diagnose_run.py": b"# fake\n",
    }
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / "models").mkdir()
    _bundle(root / "models" / "law.npz")
    monkeypatch.setattr(rr, "ROOT", root)
    return root


class FakePi:
    """Plays ssh/scp/tmux/md5sum/find against an in-memory Pi; records every process started."""

    def __init__(self, root, files=None, logs=None, corrupt=False, tmux_rc=0, still_running=False):
        self.root = root
        self.files = dict(files or {})          # repo-relative -> bytes
        self.logs = dict(logs or {})            # name -> bytes, in <repo>/logs
        self.corrupt = corrupt
        self.tmux_rc = tmux_rc
        self.still_running = still_running
        self.calls = []

    class R:
        def __init__(self, rc=0, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def run(self, argv, capture=True, cwd=None):
        self.calls.append((list(argv), cwd))
        if argv[0] == "ssh":
            return self._ssh(argv[-1])
        if argv[0] == "scp":
            return self._scp(argv, Path(cwd))
        if argv[0] == sys.executable:                   # a post-processor
            return self.R(0, f"diagnosis of {argv[-1]}\n")
        raise AssertionError(argv)

    def interactive(self, argv, cwd=None):
        self.calls.append((list(argv), cwd))
        return 0

    def _ssh(self, cmd):
        if "md5sum --" in cmd:
            names = cmd.split("md5sum -- ")[1].split(" 2>/dev/null")[0].split()
            return self.R(0, "".join(f"{_md5(self.files[n])}  {n}\n" for n in names if n in self.files))
        if "find . -maxdepth 1" in cmd:
            return self.R(0, "".join(f"{len(b)} {n}\n" for n, b in self.logs.items()))
        if "new-session" in cmd:
            self.tmux_cmd = cmd
            return self.R(self.tmux_rc)
        if "echo running" in cmd:
            return self.R(0, "running\n" if self.still_running else "ended\n")
        return self.R(0)                                # mkdir -p, send-keys

    def _scp(self, argv, cwd):
        args = [a for a in argv[1:] if a != "-q"]
        srcs, dest = args[:-1], args[-1]
        prefix = f"{CFG['user']}@{CFG['host']}:{CFG['repo']}/"
        if dest.startswith(prefix):                     # laptop -> Pi
            d = dest[len(prefix):]
            for s in srcs:
                data = (cwd / s).read_bytes()
                self.files[d + Path(s).name] = data + (b"x" if self.corrupt else b"")
        else:                                           # Pi logs -> laptop
            for s in srcs:
                name = s.split("/logs/")[1]
                (cwd / dest / name).write_bytes(self.logs[name])
        return self.R(0)

    def processes(self, prog):
        return [a for a, _ in self.calls if a[0] == prog]


def _main(args, pi, out):
    return rr.main(args, shell=pi, echo=out.append, cfg=dict(CFG))


# ------------------------------------------------------------------ sync

def test_sync_copies_only_what_differs_and_keeps_pi_calibration(repo):
    pi = FakePi(repo, files={
        "scripts/deploy/deploy_standing.py": b"print('deploy')\n",     # same
        "scripts/deploy/numpy_policy.py": b"# old\n",                  # differs -> copy
        "config/joint_servo_map.yaml": b"joints: measured on the Pi\n",  # Pi-calibrated -> keep
        # pose_sweep.py, imu_calib.yaml, the stance and the npz are missing -> copy
    })
    out = []
    assert _main(["sync", "--", "--policy-npz", "models/law.npz", "--residual-config",
                  "config/stances/knee2.yaml"], pi, out) == 0
    for rel in ("scripts/deploy/numpy_policy.py", "scripts/deploy/pose_sweep.py", "config/imu_calib.yaml",
                "config/stances/knee2.yaml", "models/law.npz"):
        assert pi.files[rel] == (repo / rel).read_bytes(), rel
    assert pi.files["config/joint_servo_map.yaml"] == b"joints: measured on the Pi\n"
    copied = [s for a in pi.processes("scp") for s in a[2:-1]]
    assert sorted(copied) == sorted(["scripts/deploy/numpy_policy.py", "scripts/deploy/pose_sweep.py",
                                     "config/imu_calib.yaml", "config/stances/knee2.yaml", "models/law.npz"])
    text = "\n".join(out)
    assert "DIFFERS: kept the Pi's copy" in text and "WARNING: config/joint_servo_map.yaml" in text
    assert "verified: 5 copied file(s)" in text
    md5_calls = [a for a in pi.processes("ssh") if "md5sum" in a[-1]]
    assert len(md5_calls) == 2          # one before (every file), one after (the copied ones)


def test_push_calibration_overwrites_the_pi_copy(repo):
    pi = FakePi(repo, files={"config/joint_servo_map.yaml": b"joints: old\n"})
    assert _main(["sync", "--push-calibration"], pi, []) == 0
    assert pi.files["config/joint_servo_map.yaml"] == b"joints: laptop\n"


def test_an_md5_mismatch_after_copying_fails_loudly(repo, capsys):
    pi = FakePi(repo, corrupt=True)
    assert _main(["sync"], pi, []) == 1
    assert "md5 mismatch" in capsys.readouterr().err


def test_sync_prints_the_bundle_requires_list(repo):
    out = []
    assert _main(["sync", "--", "--policy-npz", "models/law.npz"], FakePi(repo), out) == 0
    assert "models/law.npz: bundle format 2, requires [] (plain network)" in out


def test_a_bundle_the_pi_would_refuse_stops_sync_before_anything_is_copied(repo, capsys):
    _bundle(repo / "models" / "future.npz", requires=["obs_history8"])
    pi = FakePi(repo)
    assert _main(["sync", "--", "--policy-npz", "models/future.npz"], pi, []) == 1
    assert "obs_history8" in capsys.readouterr().err
    assert pi.calls == []


def test_file_arguments_are_rewritten_repo_relative(repo):
    files, args = rr.run_files(["--policy-npz", str(repo / "models" / "law.npz"),
                                "--posture=config\\stances\\knee2.yaml", "--tau", "0.3",
                                "--map", CFG["repo"] + "/config/joint_servo_map.yaml"], CFG["repo"])
    assert args == ["--policy-npz", "models/law.npz", "--posture=config/stances/knee2.yaml", "--tau", "0.3",
                    "--map", "config/joint_servo_map.yaml"]
    assert files == ["models/law.npz", "config/stances/knee2.yaml", "config/joint_servo_map.yaml",
                     "config/imu_calib.yaml"]       # the default --imu-calib is synced too


@pytest.mark.parametrize("bad", ["../outside.npz", "models/missing.npz"])
def test_a_file_argument_outside_the_repo_or_missing_is_refused(repo, bad):
    with pytest.raises(rr.Abort):
        rr.run_files(["--policy-npz", bad], CFG["repo"])


# ------------------------------------------------------------------ commands and quoting

def test_commands_keep_paths_with_spaces_as_one_argument(repo):
    cfg = rr.check_config(dict(CFG, port=2222, ssh_options=["-i", "C:/Users/My Name/.ssh/id_ed25519"]))
    robot = rr.Robot(cfg, shell=FakePi(repo))
    argv = robot.scp_argv(["models/law.npz"], robot.remote("models/"))
    assert argv == ["scp", "-q", "-P", "2222", "-i", "C:/Users/My Name/.ssh/id_ed25519", "models/law.npz",
                    "pi@pi-host:/home/pi/humanoidnavigation/models/"]
    assert robot.ssh_argv("ls", tty=True)[:4] == ["ssh", "-t", "-p", "2222"]
    shown = rr.show(argv)
    assert '"C:/Users/My Name/.ssh/id_ed25519"' in shown or "'C:/Users/My Name/.ssh/id_ed25519'" in shown


def test_scp_runs_in_the_repo_on_relative_paths_so_a_spaced_root_never_reaches_argv(repo):
    pi = FakePi(repo)
    assert _main(["sync"], pi, []) == 0
    for argv, cwd in pi.calls:
        assert cwd == str(repo)
        assert not any(str(repo) in a for a in argv)


def test_run_arguments_survive_both_shells_into_tmux(repo):
    pi = FakePi(repo)
    assert _main(["run", "--", "--policy-npz", "models/law.npz", "--trim", "waist_pitch=1.5, R_knee=-1",
                  "--log", "logs/it's a test.csv"], pi, []) == 0
    outer = shlex.split(pi.tmux_cmd)                    # what the Pi's login shell sees
    inner = outer[outer.index("new-session") + 4]       # tmux new-session -d -s robot <cmd>
    lex = shlex.shlex(inner, posix=True, punctuation_chars=True)  # what tmux's sh -c sees
    lex.whitespace_split = True
    words = list(lex)
    assert words[:4] == ["cd", CFG["repo"], "&&", "python3"]
    i = words.index("--trim")
    assert words[i + 1] == "waist_pitch=1.5, R_knee=-1"
    assert "logs/it's a test.csv" in words


def test_run_adds_log_to_deploy_standing_only(repo):
    assert rr.with_log("deploy_standing.py", ["--tau", "0.3"], echo=lambda *_: None)[-1] == "--log"
    assert rr.with_log("deploy_standing.py", ["--log=x.csv"], echo=lambda *_: None) == ["--log=x.csv"]
    assert rr.with_log("pose_sweep.py", ["--joints", "waist_pitch"], echo=lambda *_: None) == [
        "--joints", "waist_pitch"]


def test_run_refuses_to_start_over_a_live_session(repo, capsys):
    pi = FakePi(repo, tmux_rc=4)
    assert _main(["run"], pi, []) == 1
    assert "already in tmux session" in capsys.readouterr().err
    assert not [a for a, _ in pi.calls if "attach" in a[-1]]


def test_a_detached_run_is_not_fetched(repo):
    pi = FakePi(repo, logs={"20261025_120000_standing.csv": b"t\n1\n"}, still_running=True)
    out = []
    assert _main(["run"], pi, out) == 0
    assert "still going" in "\n".join(out)
    assert not (repo / "logs" / "pi").exists()


def test_stop_sends_ctrl_c_into_the_session(repo):
    pi = FakePi(repo)
    assert _main(["stop"], pi, []) == 0
    assert pi.processes("ssh")[-1][-1] == "tmux send-keys -t =robot: C-c"


# ------------------------------------------------------------------ fetch

def test_fetch_copies_new_and_grown_files_skips_the_rest_and_diagnoses_new_logs(repo):
    local = repo / "logs" / "pi"
    local.mkdir(parents=True)
    (local / "A_standing.csv").write_bytes(b"0123456789")
    (local / "C_standing.csv").write_bytes(b"short")            # fetched mid-run last time
    logs = {"A_standing.csv": b"0123456789", "A_standing.meta.json": b"{}",
            "B_standing.csv": b"t,x\n", "B_standing.console.txt": b"hello",
            "B_standing.events.jsonl": b"{}\n", "C_standing.csv": b"short, now longer"}
    pi = FakePi(repo, logs=logs)
    out = []
    assert _main(["fetch"], pi, out) == 0
    fetched = sorted(s.split("/logs/")[1] for a in pi.processes("scp") for s in a[2:-1])
    assert fetched == ["A_standing.meta.json", "B_standing.console.txt", "B_standing.csv",
                       "B_standing.events.jsonl", "C_standing.csv"]
    for n, b in logs.items():
        assert (local / n).read_bytes() == b
    diagnosed = [a[-1] for a in pi.processes(sys.executable)]
    assert diagnosed == ["logs/pi/B_standing.csv", "logs/pi/C_standing.csv"]
    assert (local / "B_standing.diagnose.txt").read_text() == "diagnosis of logs/pi/B_standing.csv\n"
    text = "\n".join(out)
    assert "run_report.py not found" in text and "updated (was 5 bytes)" in text

    pi2 = FakePi(repo, logs=logs)                                # nothing new the second time
    out2 = []
    assert _main(["fetch"], pi2, out2) == 0
    assert pi2.processes("scp") == [] and "nothing new" in out2[-1]


def test_fetch_runs_run_report_when_it_exists(repo):
    (repo / "scripts" / "debug" / "run_report.py").write_text("# PR4\n")
    pi = FakePi(repo, logs={"B_standing.csv": b"t\n"})
    assert _main(["fetch"], pi, []) == 0
    assert [a[1] for a in pi.processes(sys.executable)] == ["scripts/debug/diagnose_run.py",
                                                            "scripts/debug/run_report.py"]


# ------------------------------------------------------------------ dry run

@pytest.mark.parametrize("cmd", [["sync", "--dry-run", "--", "--policy-npz", "models/law.npz"],
                                 ["run", "--dry-run", "--", "--policy-npz", "models/law.npz"],
                                 ["run", "--dry-run", "--no-tmux"], ["attach", "--dry-run"],
                                 ["stop", "--dry-run"], ["fetch", "--dry-run"]])
def test_dry_run_prints_commands_and_starts_nothing(repo, cmd):
    pi = FakePi(repo)
    out = []
    assert _main(cmd, pi, out) == 0
    assert pi.calls == []
    assert any(line.startswith("$ ssh ") for line in out)
    assert not (repo / "logs").exists()


def test_a_missing_config_says_how_to_make_one(tmp_path, capsys):
    assert rr.main(["fetch", "--config", str(tmp_path / "robot.yaml")], shell=FakePi(tmp_path)) == 1
    assert "robot.yaml.example" in capsys.readouterr().err


@pytest.mark.parametrize("key,value", [("repo", "/home/pi/my repo"), ("host", "pi;rm -rf ~"),
                                       ("tmux_session", "a b")])
def test_unsafe_remote_values_are_refused(key, value):
    with pytest.raises(rr.Abort):
        rr.check_config(dict(CFG, **{key: value}))
