"""Hardware-run recording (scripts/deploy: frame_log.py schema 2, hardware.py telemetry).

Everything here runs without a robot: a fake serial port that answers like an SCS servo bus
(including a corrupt reply), a fake I2C bus for the IMU and the ADS1115 boards, and the deploy
scripts' own main() driven against them.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

DEPLOY = Path("scripts/deploy").resolve()
sys.path.insert(0, str(DEPLOY))

import frame_log  # noqa: E402
import hardware  # noqa: E402
from sim_real_map import SimRealMap  # noqa: E402

MAP = SimRealMap()
DOFS = [j.dof for j in MAP.joints]
NJ = len(DOFS)


# ---- fakes ---------------------------------------------------------------------------------

class FakeSerial:
    """An SCS bus of ideal servos: a goal write moves the servo there, a read returns the
    control table. `corrupt` holds indices of READ requests whose reply checksum is broken."""

    def __init__(self, positions, speed=0x8005, load=0x0400 | 123, volt=74, temp=35, corrupt=()):
        self.pos = {int(k): int(v) for k, v in positions.items()}
        self.speed, self.load, self.volt, self.temp = speed, load, volt, temp
        self.corrupt = set(corrupt)
        self.read_requests = []      # (sid, reg, nbytes)
        self._out = bytearray()

    def _table(self, sid):
        t = bytearray(0x40)
        p = self.pos[sid]
        t[0x38:0x3A] = bytes([(p >> 8) & 0xFF, p & 0xFF])
        t[0x3A:0x3C] = bytes([(self.speed >> 8) & 0xFF, self.speed & 0xFF])
        t[0x3C:0x3E] = bytes([(self.load >> 8) & 0xFF, self.load & 0xFF])
        t[0x3E], t[0x3F] = self.volt, self.temp
        return t

    def reset_input_buffer(self):
        self._out.clear()

    def write(self, pkt):
        assert pkt[0] == 0xFF and pkt[1] == 0xFF
        sid, length, instr = pkt[2], pkt[3], pkt[4]
        params = list(pkt[5:5 + length - 2])
        assert pkt[-1] == (~(sid + length + instr + sum(params))) & 0xFF, "bad request checksum"
        if instr == 0x02:
            reg, n = params
            k = len(self.read_requests)
            self.read_requests.append((sid, reg, n))
            reply = [0xFF, 0xFF, sid, n + 2, 0x00, *self._table(sid)[reg:reg + n]]
            reply.append((~sum(reply[2:])) & 0xFF)
            if k in self.corrupt:
                reply[-1] ^= 0x5A
            self._out += bytes(reply)
        elif instr == 0x03:
            if params[0] == hardware.REG_GOAL_POSITION:
                self.pos[sid] = (params[1] << 8) | params[2]
            ack = [0xFF, 0xFF, sid, 2, 0]
            self._out += bytes(ack + [(~sum(ack[2:])) & 0xFF])

    def read(self, n):
        out = bytes(self._out[:n])
        del self._out[:n]
        return out

    def close(self):
        pass


def fake_bus(positions=None, telemetry=True, **kw):
    bus = hardware.ServoBus(telemetry=telemetry)
    bus._ser = FakeSerial(positions or {int(s): 512 for s in MAP.servo_ids}, **kw)
    return bus


class FakeI2C:
    """ICM-20948 accel/gyro registers + any number of ADS1115 boards on one bus."""

    def __init__(self, accel_g=(0.0, 0.0, 1.0), ads=None, dead=()):
        self.accel_g = accel_g
        self.accel_reads = 0
        self.tilt_after = None           # accel reads after which the robot "falls over"
        self.ads = ads or {}             # addr -> {ain: volts}
        self.dead = set(dead)            # ADS addresses that NACK
        self._mux = {}

    @staticmethod
    def _be16(v):
        v = int(round(v)) & 0xFFFF
        return [(v >> 8) & 0xFF, v & 0xFF]

    def read_i2c_block_data(self, addr, reg, n):
        if addr in self.dead:
            raise OSError(121, "Remote I/O error")
        if addr == 0x68 and reg == hardware.IMU.ACCEL_XOUT_H:
            self.accel_reads += 1
            a = self.accel_g
            if self.tilt_after is not None and self.accel_reads > self.tilt_after:
                a = (1.0, 0.0, 0.0)      # on its side: upright_cos 0
            return sum((self._be16(v * hardware.IMU.ACC_LSB_PER_G) for v in a), [])
        if addr == 0x68 and reg == hardware.IMU.GYRO_XOUT_H:
            return [0] * 6
        if addr in self.ads and reg == 0x00:
            volts = self.ads[addr][self._mux[addr]]
            return self._be16(volts / 4.096 * 32768.0)
        raise AssertionError(f"unexpected read 0x{addr:02X} reg 0x{reg:02X}")

    def write_i2c_block_data(self, addr, reg, data):
        if addr in self.dead:
            raise OSError(121, "Remote I/O error")
        cfg = (data[0] << 8) | data[1]
        self._mux[addr] = (cfg >> 12) & 0x3

    def write_byte_data(self, *a):
        pass

    def read_byte_data(self, addr, reg):
        return 0xEA

    def close(self):
        pass


# ---- hardware.py: telemetry in the same read -------------------------------------------------

def test_read_state_is_one_8_byte_read_of_raw_registers():
    bus = fake_bus({5: 0x01F4})
    assert bus.read_state(5) == (0x01F4, 0x8005, 0x0400 | 123, 74, 35)
    assert bus._ser.read_requests == [(5, 0x38, 8)]


def test_read_all_fills_telemetry_with_no_extra_transactions_and_retries_a_corrupt_reply():
    ids = [int(s) for s in MAP.servo_ids]
    bus = fake_bus({s: 400 + s for s in ids}, corrupt={3})   # the 4th servo's first reply is bad
    u = bus.read_all(ids)
    assert u.tolist() == [400.0 + s for s in ids]
    # one 8-byte read per servo, plus the single retry for the corrupt reply
    reqs = bus._ser.read_requests
    assert len(reqs) == NJ + 1 and all(r[1:] == (0x38, 8) for r in reqs)
    assert bus.last_telemetry.shape == (4, NJ)
    assert np.all(bus.last_telemetry[0] == 0x8005) and np.all(bus.last_telemetry[3] == 35)


def test_a_servo_that_never_answers_cleanly_is_nan_in_position_and_telemetry():
    ids = [int(s) for s in MAP.servo_ids]
    bus = fake_bus(corrupt={0, 1, 2})        # servo 0: first try + both retries corrupt
    u = bus.read_all(ids)
    assert np.isnan(u[0]) and np.all(np.isfinite(u[1:]))
    assert np.all(np.isnan(bus.last_telemetry[:, 0]))
    assert np.all(np.isfinite(bus.last_telemetry[:, 1:]))


def test_no_telemetry_falls_back_to_the_2_byte_position_read():
    ids = [int(s) for s in MAP.servo_ids]
    bus = fake_bus(telemetry=False)
    u = bus.read_all(ids)
    assert np.all(u == 512)
    assert all(r[1:] == (0x38, 2) for r in bus._ser.read_requests)
    assert np.all(np.isnan(bus.last_telemetry))


def test_read_pos_rejects_a_reply_with_a_wrong_length_byte():
    # A reply whose length field disagrees is garbage even if its checksum happens to match.
    reply = [0xFF, 0xFF, 5, 0x05, 0, 0x01, 0xF4]
    reply.append((~sum(reply[2:])) & 0xFF)
    assert hardware._reply_data(bytes(reply), 5, 2) is None
    good = [0xFF, 0xFF, 5, 0x04, 0x20, 0x01, 0xF4]     # nonzero error byte is still valid data
    good.append((~sum(good[2:])) & 0xFF)
    assert list(hardware._reply_data(bytes(good), 5, 2)) == [0x01, 0xF4]


def test_imu_keeps_the_raw_remapped_accel_it_normalized_away():
    remap = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    imu = hardware.IMU(axis_remap=remap)
    imu._bus = FakeI2C(accel_g=(0.25, -0.5, 1.1))
    pg = imu.projected_gravity()
    raw = np.array([0.25, -0.5, 1.1])
    np.testing.assert_allclose(imu.last_accel_g, remap @ raw, atol=1e-3)     # g, not unit
    np.testing.assert_allclose(pg, -(remap @ raw) / np.linalg.norm(raw), atol=1e-3)


# ---- frame_log.py: schema 2 ------------------------------------------------------------------

def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_v2_round_trip_writes_every_block_and_poses_from_the_row(tmp_path):
    path = tmp_path / "run.csv"
    log = frame_log.FrameLogger(path, DOFS, n_fsr=3, meta={"script": "test"})
    u = MAP.centers + np.arange(NJ)
    tel = np.vstack([np.full(NJ, 0x8005), np.full(NJ, 0x0400 | 7), np.full(NJ, 74), np.full(NJ, 36)])
    log.log(t_wall=1.79e9 + 0.123456, step=0, loop_ms=25.0, busy_ms=21.0,
            pg=(0.01, -0.02, -0.999), av=(0.1, 0.2, 0.3), upright_cos=0.999, rej_total=0,
            u_meas=u, u_cmd=u + 1, q_cmd=np.full(NJ, 0.05), jvel_f=np.zeros(NJ),
            fsr=(0.5, 1.5, 2.5), fsr_age_ms=4.0, acc=(0.01, -0.02, 1.01),
            q_prior=np.full(NJ, -0.01), telemetry=tel, posture=(1.5, 0.25, -2.0))
    # a frame that supplies none of the optional blocks: all NaN, columns still aligned
    log.log(t_wall=1.79e9 + 0.2, step=1, loop_ms=25.0, busy_ms=21.0, pg=(0, 0, -1), av=(0, 0, 0),
            upright_cos=1.0, rej_total=0, u_meas=u, u_cmd=u, q_cmd=np.zeros(NJ), jvel_f=np.zeros(NJ))
    log.close(extra="kept")

    rows = _read(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["schema_version"] == 2 and meta["rows_written"] == 2 and meta["extra"] == "kept"
    assert meta["columns"] == list(rows[0].keys()) == frame_log.build_columns(DOFS, 3)
    assert {"hostname", "python", "numpy"} <= meta.keys()
    r = rows[0]
    assert r["t_wall"] == "1790000000.123456"
    assert float(r["acc_z"]) == pytest.approx(1.01)
    assert float(r["posture_zero"]) == 0.25 and float(r["posture_corr"]) == -2.0
    assert float(r[f"q_prior.{DOFS[4]}"]) == -0.01
    assert int(float(r[f"spd_raw.{DOFS[0]}"])) == 0x8005
    assert int(float(r[f"load_raw.{DOFS[-1]}"])) == 0x0407
    assert float(r["fsr_v2"]) == 2.5
    for col in ("acc_x", "posture_lean", f"q_prior.{DOFS[0]}", f"temp_raw.{DOFS[0]}", "fsr_v0"):
        assert rows[1][col] == "nan"
    pose = frame_log.pose_from_row(r, MAP)
    np.testing.assert_allclose(pose["q_meas"], MAP.units_to_rad(u), atol=1e-6)
    np.testing.assert_allclose(pose["acc"], [0.01, -0.02, 1.01], atol=1e-6)


def test_a_schema_1_log_still_poses(tmp_path):
    path = tmp_path / "old.csv"
    cols = frame_log.build_columns_v1(DOFS)
    vals = {c: "0" for c in cols}
    vals.update({"t_rel": "1.5", "pg_z": "-1"})
    vals.update({f"u_meas.{d}": str(int(c) + 10) for d, c in zip(DOFS, MAP.centers)})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(vals)
    pose = frame_log.pose_from_row(_read(path)[0], MAP)
    assert pose["t"] == 1.5 and np.all(np.isnan(pose["acc"]))
    np.testing.assert_allclose(pose["q_meas"], MAP.signs * 10 / MAP.units_per_rad, atol=1e-6)


def test_udp_offsets_still_find_u_meas_and_u_cmd_after_the_scalars():
    cols = frame_log.build_columns(DOFS)
    s = len(frame_log.SCALAR_COLUMNS)
    assert cols[s] == f"u_meas.{DOFS[0]}" and cols[s + NJ] == f"u_cmd.{DOFS[0]}"


def test_a_row_of_the_wrong_length_is_counted_not_written_shifted(tmp_path):
    log = frame_log.FrameLogger(tmp_path / "r.csv", DOFS, n_fsr=2)
    log.log(t_wall=1.0, step=0, loop_ms=0, busy_ms=0, pg=(0, 0, -1), av=(0, 0, 0), upright_cos=1,
            rej_total=0, u_meas=np.zeros(NJ), u_cmd=np.zeros(NJ), q_cmd=np.zeros(NJ),
            jvel_f=np.zeros(NJ), fsr=(1.0, 2.0, 3.0))
    log.close()
    assert log.malformed == 1 and log.written == 0


def test_files_meta_hashes_the_run_files_and_every_deploy_source(tmp_path):
    f = tmp_path / "policy.npz"
    f.write_bytes(b"weights")
    m = frame_log.files_meta({"policy": f, "missing": tmp_path / "nope.yaml", "unused": None})
    import hashlib
    assert m["files"]["policy"]["md5"] == hashlib.md5(b"weights").hexdigest()
    assert m["files"]["missing"]["md5"] is None and "unused" not in m["files"]
    for name in frame_log.DEPLOY_SOURCES:
        assert m["files"][f"source:{name}"]["md5"] is not None, name
    assert m["file_md5"][str(f)] == m["files"]["policy"]["md5"]


# ---- console tee ------------------------------------------------------------------------------

class _Broken:
    def write(self, s):
        raise OSError("terminal gone")

    def flush(self):
        raise OSError("terminal gone")


def test_console_tee_records_stdout_stderr_and_progress_lines_with_times(tmp_path, capsys):
    path = frame_log.sidecar_path(tmp_path / "run.csv", "console.txt")
    assert path.name == "run.console.txt"
    tee = frame_log.ConsoleTee(path)
    try:
        print("hello", 42)
        print("\r  pitch +1.0", end="")
        print("\r  pitch +2.0", end="")
        print()
        print("oops", file=sys.stderr)
        sys.stdout.detach_terminal(_Broken())        # what release_on_hangup does on SIGHUP
        print("after hangup")
    finally:
        tee.close()
    assert sys.stdout is not tee.stdout
    lines = path.read_text().splitlines()
    text = [ln.split(" | ", 1)[1] for ln in lines]
    assert text == ["hello 42", "  pitch +1.0", "  pitch +2.0", "oops", "after hangup"]
    assert all(float(ln.split(" | ", 1)[0]) > 1e9 for ln in lines)


# ---- FSR channels over several boards ------------------------------------------------------

def test_fsr_channel_specs_parse_old_and_multi_board_forms():
    assert frame_log.parse_fsr_channels("0,1") == [(0x48, 0), (0x48, 1)]
    assert frame_log.parse_fsr_channels("0x48:0, 0x48:1,0x49:0") == [(0x48, 0), (0x48, 1), (0x49, 0)]
    for bad in ("0x50:0", "0x48:4", "0,0", ""):
        with pytest.raises(ValueError):
            frame_log.parse_fsr_channels(bad)


def test_the_shipped_4_per_foot_layout_loads_and_overrides_the_default_channels():
    chans, layout = frame_log.resolve_fsr("0,1", "config/fsr_layout_4per_foot.yaml")
    assert chans == [(0x48, i) for i in range(4)] + [(0x49, i) for i in range(4)]
    assert {c["foot"] for c in layout["channels"]} == {"left", "right"}
    assert layout["rfixed_ohm"] == 2000
    with pytest.raises(ValueError):
        frame_log.resolve_fsr("0x48:0", "config/fsr_layout_4per_foot.yaml")
    chans, layout = frame_log.resolve_fsr("0,1", None)       # today's wiring, unchanged
    assert chans == [(0x48, 0), (0x48, 1)]
    assert [c["name"] for c in layout["channels"]] == ["L_heel", "L_toe"]
    assert layout["channels"][0]["x_mm"] is None


def _first_sample(s):
    import time
    t_end = time.time() + 5
    while not s.t_sample and time.time() < t_end:
        time.sleep(0.01)
    return s.read()[0]


def test_fsr_sampler_reads_two_boards_in_parallel_and_a_dead_board_only_blanks_its_own():
    ads = {0x48: {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0}, 0x49: {0: 2.5, 1: 3.0, 2: 0.25, 3: 0.75}}
    chans = frame_log.parse_fsr_channels("0x48:0,0x48:1,0x48:2,0x48:3,0x49:0,0x49:1,0x49:2,0x49:3")
    s = frame_log.FsrSampler(channels=chans, smbus=FakeI2C(ads=ads), hz=1e-3)
    try:
        assert len(s._groups) == 4               # four conversion waits for eight channels
        # the background thread's first sweep; it then idles, so sweep() below has the bus
        np.testing.assert_allclose(_first_sample(s), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.25, 0.75], atol=1e-3)
        s._bus.dead = {0x49}
        v = s.sweep()
        np.testing.assert_allclose(v[:4], [0.5, 1.0, 1.5, 2.0], atol=1e-3)
        assert np.all(np.isnan(v[4:])) and s.errors >= 4
    finally:
        s.close()


def test_fsr_sampler_still_takes_plain_pins_on_one_board():
    s = frame_log.FsrSampler(channels=[0, 1], smbus=FakeI2C(ads={0x48: {0: 1.0, 1: 2.0}}), hz=1e-3)
    try:
        assert s.channels == [(0x48, 0), (0x48, 1)]
        np.testing.assert_allclose(_first_sample(s), [1.0, 2.0], atol=1e-3)
    finally:
        s.close()


# ---- the deploy scripts, end to end against the fakes --------------------------------------

def _npz(path, stance, prior_gain=None):
    obs_dim = 228
    stance = np.asarray(stance, np.float32)
    extra = {} if prior_gain is None else {"prior_gain": np.asarray(prior_gain, np.float32)}
    np.savez(path, n_hidden=np.array(0, dtype=np.int64),
             out_w=np.zeros((len(stance), obs_dim), np.float32), out_b=stance,
             obs_mean=np.zeros(obs_dim, np.float32), obs_var=np.ones(obs_dim, np.float32),
             clip_obs=np.array(50.0, np.float32), epsilon=np.array(1e-8, np.float32), **extra)
    return path


@pytest.fixture
def robot(monkeypatch):
    """Swap the hardware seam for fakes; returns the fakes so a test can steer them."""
    state = {"i2c": FakeI2C(), "tees": []}
    state["i2c"].tilt_after = 30

    class Bus(hardware.ServoBus):
        def connect(self):
            self._ser = FakeSerial({int(s): 512 for s in MAP.servo_ids})
            state["bus"] = self
            return self

    class Imu(hardware.IMU):
        def connect(self):
            self._bus = state["i2c"]
            return self

        def calibrate_gyro_bias(self, *a, **k):
            self.gyro_bias = np.array([0.001, -0.002, 0.003], dtype=np.float32)
            return self.gyro_bias

    class Tee(frame_log.ConsoleTee):
        def __init__(self, path):
            super().__init__(path)
            state["tees"].append(self)

    class Fsr:
        def __init__(self, channels, **kw):
            self.channels, self.errors = list(channels), 0

        def read(self):
            return [1.0 + k for k in range(len(self.channels))], 3.0

        def close(self):
            pass

    monkeypatch.setattr(hardware, "ServoBus", Bus)
    monkeypatch.setattr(hardware, "IMU", Imu)
    monkeypatch.setattr(hardware, "release_on_hangup", lambda: None)
    monkeypatch.setattr(frame_log, "ConsoleTee", Tee)
    monkeypatch.setattr(frame_log, "FsrSampler", Fsr)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    yield state
    for t in state["tees"]:
        t.close()


def _run(monkeypatch, module, argv):
    import importlib
    mod = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [module + ".py", *argv])
    return mod.main()


def _stance():
    with open("config/stances/knee2.yaml") as f:
        return yaml.safe_load(f)["standing"]["residual_baseline"]


def test_hold_pose_now_writes_rows_meta_and_console(tmp_path, monkeypatch, robot):
    npz = _npz(tmp_path / "hold.npz", _stance())
    log = tmp_path / "hold.csv"
    _run(monkeypatch, "deploy_standing", [
        "--policy-npz", str(npz), "--residual-config", "config/stances/knee2.yaml",
        "--hold-pose", "--lean-back", "2", "--ramp-secs", "0.05",
        "--imu-calib", str(tmp_path / "none.yaml"), "--log", str(log),
        "--log-fsr", "--fsr-channels", "0x48:0,0x48:1,0x49:0"])
    rows = _read(log)
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    console = frame_log.sidecar_path(log, "console.txt").read_text()
    assert len(rows) >= 5 and meta["rows_written"] == len(rows)
    assert meta["mode"] == "hold_pose" and meta["args"]["lean_back"] == 2.0
    assert meta["gyro_bias"] == pytest.approx([0.001, -0.002, 0.003])
    assert meta["files"]["source:hardware.py"]["md5"] and meta["files"]["policy"]["md5"]
    assert meta["servo_telemetry"]["enabled"] is True
    assert [c["adc"] for c in meta["fsr"]["channels"]] == [0x48, 0x48, 0x49]
    assert "[hold-pose] HOLDING" in console and "[SAFETY]" in console
    r = rows[-1]
    assert float(r["fsr_v2"]) == 3.0
    assert int(float(r[f"temp_raw.{DOFS[0]}"])) == 35
    assert float(r["acc_x"]) == pytest.approx(1.0, abs=1e-3)     # the fall the fake staged
    assert r[f"jvel_f.{DOFS[0]}"] == "nan" and r[f"q_prior.{DOFS[0]}"] == "nan"
    # q_cmd is the held pose, with the lean-back on the waist
    wp = DOFS.index("waist_pitch")
    q = np.array([float(r[f"q_cmd.{d}"]) for d in DOFS])
    assert q[wp] == pytest.approx(_stance()[wp] + np.deg2rad(2.0), abs=1e-4)
    u_cmd = [float(r[f"u_cmd.{d}"]) for d in DOFS]
    assert u_cmd == [float(robot["bus"]._ser.pos[int(s)]) for s in MAP.servo_ids]


def test_closed_loop_logs_the_law_and_the_posture_controller(tmp_path, monkeypatch, robot):
    gain = np.zeros((NJ, 6), np.float32)
    gain[2, 4] = 0.1                                   # R_hip_pitch += 0.1 * gyro_y
    npz = _npz(tmp_path / "law.npz", _stance(), prior_gain=gain)
    log = tmp_path / "loop.csv"
    _run(monkeypatch, "deploy_standing", [
        "--policy-npz", str(npz), "--residual-config", "config/stances/knee2.yaml",
        "--posture", "config/stances/posture.yaml", "--ramp-secs", "0.05",
        "--imu-calib", str(tmp_path / "none.yaml"), "--log", str(log), "--no-servo-telemetry"])
    rows = _read(log)
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    assert meta["mode"] == "closed_loop" and len(rows) >= 5
    assert meta["prior_gain"][2][4] == pytest.approx(0.1) and meta["posture"]["gain"]
    assert meta["servo_telemetry"]["enabled"] is False
    r = rows[0]
    assert np.isfinite(float(r[f"q_prior.{DOFS[2]}"]))
    assert np.isfinite(float(r["posture_lean"])) and np.isfinite(float(r["posture_corr"]))
    assert r[f"load_raw.{DOFS[0]}"] == "nan"            # telemetry off
    assert r["fsr_v0"] == "nan" and "fsr_v1" in r       # no FSRs: the two v1 columns, NaN


def test_no_log_writes_nothing(tmp_path, monkeypatch, robot):
    npz = _npz(tmp_path / "hold.npz", _stance())
    monkeypatch.setattr(frame_log, "DEFAULT_LOG_DIR", tmp_path / "logs")
    _run(monkeypatch, "deploy_standing", [
        "--policy-npz", str(npz), "--residual-config", "config/stances/knee2.yaml",
        "--hold-pose", "--ramp-secs", "0.05", "--imu-calib", str(tmp_path / "none.yaml"),
        "--no-log"])
    assert not (tmp_path / "logs").exists() and not robot["tees"]


def test_pose_sweep_records_its_live_phase(tmp_path, monkeypatch, robot):
    robot["i2c"].tilt_after = None
    log = tmp_path / "sweep.csv"
    _run(monkeypatch, "pose_sweep", [
        "--joints", "waist_pitch", "--range", "-2", "2", "--step", "2", "--dwell", "0.2",
        "--ramp", "0.05", "--imu-calib", str(tmp_path / "none.yaml"), "--log", str(log)])
    rows = _read(log)
    meta = json.loads(log.with_suffix(".meta.json").read_text())
    console = frame_log.sidecar_path(log, "console.txt").read_text()
    assert meta["mode"] == "sweep" and meta["trims_deg"] == [0.0, -2.0, 2.0]
    assert len(meta["sweep_points"]) == 3 and meta["rows_written"] == len(rows) > 20
    assert "--- RESULT ---" in console and "sweeping waist_pitch" in console
    wp = DOFS.index("waist_pitch")
    u = {float(r[f"u_cmd.{DOFS[wp]}"]) for r in rows}
    assert len(u) >= 3                                  # it logged the poses it moved through
    assert all(r[f"jvel_f.{DOFS[0]}"] == "nan" for r in rows)
