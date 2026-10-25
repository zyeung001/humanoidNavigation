#!/usr/bin/env python3
"""Watch the real robot in 3D, live, while it runs.

Opens MuJoCo's own interactive window on the actual model -- real meshes, orbitable,
zoomable -- and poses it from the robot's encoders as they arrive. The ghost skeleton
drawn through it is what the policy COMMANDED for the same frame, so the gap you see
between ghost and robot is the tracking error, live, in the shape of the robot.

WHY IT RUNS HERE AND NOT ON THE PI. The Pi is numpy-only on purpose -- no torch, no
MuJoCo -- so there is nothing on it that can render. The robot broadcasts a 161-byte
datagram per frame and this listens. UDP because the control loop must not care whether
anyone is watching: an unreceived datagram costs one syscall, while a TCP viewer that
stalls would back-pressure into a 40 Hz loop that is already using 21 ms of its 25.

  # on the Pi, streaming to this machine
  python3 scripts/deploy/deploy_standing.py --policy-npz ... --log --stream 192.168.86.20:9870

  # here
  python scripts/debug/live_viewer.py --listen 9870
  python scripts/debug/live_viewer.py --replay logs/20260814_154939_standing.csv
  python scripts/debug/live_viewer.py --selftest        # no window, no robot

FINDING THIS MACHINE'S ADDRESS: --listen prints the addresses to stream to on startup.

The root is drawn tilted by the measured projected gravity, so the robot leans on screen
the way it leaned in the room. Yaw is not observable from gravity and is left at zero --
the robot may be facing a different way than it was; everything about its POSE is real.
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "deploy"))
sys.path.insert(0, str(ROOT / "scripts" / "debug"))

from frame_log import UdpSink  # noqa: E402
from pose_viewer import CHAINS, POINTS  # noqa: E402
from sim_real_map import SimRealMap, DEFAULT_MAP  # noqa: E402

GHOST_RGBA = np.array([0.37, 0.66, 0.78, 0.85], dtype=np.float32)   # steel cyan = commanded


def tilt_quat(pg):
    """Quaternion rotating the model's -Z onto the measured gravity direction.

    Gravity fixes two of the three rotational degrees of freedom; yaw is simply not in the
    measurement, so it stays zero rather than being invented. Drawing the lean is worth it
    -- a robot that is falling over should look like it.
    """
    g = np.asarray(pg, dtype=float)
    n = np.linalg.norm(g)
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    g = g / n
    down = np.array([0.0, 0.0, -1.0])
    axis = np.cross(down, g)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if g[2] < 0 else np.array([0.0, 1.0, 0.0, 0.0])
    axis /= s
    ang = np.arctan2(s, float(np.dot(down, g)))
    # CONJUGATE, because pg is gravity in the BODY frame: the root orientation R we want
    # satisfies R^T @ down = pg, i.e. R rotates pg onto down. The rotation built above takes
    # down onto pg -- that is R^T, not R. Returning it unconjugated drew every tilt MIRRORED
    # on both axes (a 10 deg forward lean drawn as 10 deg back, left as right), from the day
    # this viewer was written on 8/14 until a round-trip test caught it on 9/26: pose a known
    # tilt, read what the IMU would report, reconstruct from that, compare.
    return np.concatenate([[np.cos(ang / 2)], -np.sin(ang / 2) * axis])


class Poser:
    """Holds the model and turns encoder units into a posed, tilted robot."""

    def __init__(self, xml, map_path=DEFAULT_MAP):
        import mujoco
        self.mj = mujoco
        self.map = SimRealMap(map_path)
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.ghost = mujoco.MjData(self.model)     # second state, for the commanded pose
        self.qadr = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        self.jid = {nm: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nm)
                    for k, nm in POINTS.values() if k == "j"}
        self.bid = {nm: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
                    for k, nm in POINTS.values() if k == "b"}
        self.nj = len(self.map.joints)

    def _points(self, d):
        raw = {}
        for name, (kind, ref) in POINTS.items():
            if kind == "j":
                raw[name] = np.array(d.xanchor[self.jid[ref]])
            elif kind == "g":
                raw[name] = np.array(d.geom_xpos[ref])
            elif kind == "b":
                raw[name] = np.array(d.xipos[self.bid[ref]])
        for name, (kind, ref) in POINTS.items():
            if kind == "mid":
                raw[name] = 0.5 * (raw[ref[0]] + raw[ref[1]])
        return raw

    def pose(self, u_meas, u_cmd, pg):
        q = np.asarray(self.map.units_to_rad(np.asarray(u_meas, dtype=np.float32)))
        self.data.qpos[:] = 0
        self.data.qpos[3:7] = tilt_quat(pg)
        self.data.qpos[self.qadr] = q
        self.mj.mj_forward(self.model, self.data)
        # Stand it on the floor: the log carries no world height, only joint angles and
        # attitude. Reference the FEET, not body origins -- this model came out of CAD and
        # its body frames sit wherever the exporter left them, so the lowest body origin is
        # nowhere near the lowest part of the robot.
        feet_z = min(float(self.data.geom_xpos[g][2]) for g in (27, 47))
        self.data.qpos[2] += -feet_z + 0.005
        self.mj.mj_forward(self.model, self.data)

        qc = np.asarray(self.map.units_to_rad(np.asarray(u_cmd, dtype=np.float32)))
        self.ghost.qpos[:] = self.data.qpos
        self.ghost.qpos[self.qadr] = qc
        self.mj.mj_forward(self.model, self.ghost)
        return np.degrees(qc - q)

    def draw_ghost(self, scn):
        """Commanded pose as a translucent skeleton laid over the real robot."""
        pts = self._points(self.ghost)
        for _, chain in CHAINS:
            for a, b in zip(chain[:-1], chain[1:]):
                if scn.ngeom >= scn.maxgeom:
                    return
                g = scn.geoms[scn.ngeom]
                self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_CAPSULE,
                                     np.zeros(3), np.zeros(3), np.zeros(9), GHOST_RGBA)
                self.mj.mjv_connector(g, self.mj.mjtGeom.mjGEOM_CAPSULE, 0.006,
                                      pts[a].astype(float), pts[b].astype(float))
                scn.ngeom += 1


def parse_packet(buf, nj=17):
    if len(buf) != struct.calcsize(UdpSink.FMT) or buf[0] != UdpSink.MAGIC:
        return None
    v = struct.unpack(UdpSink.FMT, buf)
    return {"step": v[1], "t": v[2], "pg": np.array(v[3:6]), "upright": v[6],
            "u_meas": np.array(v[7:7 + nj]), "u_cmd": np.array(v[7 + nj:7 + 2 * nj])}


class Target:
    """The most recent frame the robot sent, updated off the render thread.

    Rendering used to advance only when a datagram arrived, which tied the picture's
    frame rate to Wi-Fi arrival jitter and made a perfectly steady 40 Hz robot look
    stuttery. The producer now just keeps this up to date and the renderer runs at its
    own steady rate, easing toward it.
    """

    def __init__(self, nj):
        self.nj = nj
        self.frame = None
        self.count = 0
        self.done = False


def produce_live(port, nj, tgt):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", int(port)))
    sock.settimeout(0.5)
    host = socket.gethostbyname_ex(socket.gethostname())[2]
    print(f"listening on udp/{port}. Stream to one of: "
          + ", ".join(f"{h}:{port}" for h in host if not h.startswith("127.")))
    print("waiting for the robot...")
    seen = False
    while not tgt.done:
        try:
            buf, _ = sock.recvfrom(2048)
        except socket.timeout:
            continue
        f = parse_packet(buf, nj)
        if not f:
            continue
        if not seen:
            seen = True
            print("robot connected.")
        tgt.frame = f
        tgt.count += 1
    sock.close()


def produce_replay(path, nj, speed, tgt, t_from=None, t_to=None):
    import csv
    rows = list(csv.DictReader(open(path)))
    dofs = [j.dof for j in SimRealMap(DEFAULT_MAP).joints]
    # A window, so diagnose_run.py can point straight at the moment it found. Watching a
    # 200-second run to see a fall at t=137 is how incidents go unwatched.
    lo = -np.inf if t_from is None else float(t_from)
    hi = np.inf if t_to is None else float(t_to)
    rows = [r for r in rows if lo <= float(r["t_rel"]) <= hi]
    if not rows:
        print(f"no frames between {lo} and {hi} s -- check the window")
        tgt.done = True
        return
    span = "" if t_from is None and t_to is None else \
        f" [t={float(rows[0]['t_rel']):.1f}..{float(rows[-1]['t_rel']):.1f}s]"
    print(f"replaying {len(rows)} frames from {Path(path).name} at {speed:g}x{span}")
    # Time the window from its own first frame, or a --from of 137 would sit idle for
    # 137 seconds before drawing anything.
    base = float(rows[0]["t_rel"])
    t0 = time.time()
    for r in rows:
        if tgt.done:
            return
        while time.time() - t0 < (float(r["t_rel"]) - base) / max(speed, 1e-6):
            time.sleep(0.002)
        tgt.frame = {"step": int(float(r["step"])), "t": float(r["t_rel"]),
                     "pg": np.array([float(r["pg_x"]), float(r["pg_y"]), float(r["pg_z"])]),
                     "upright": float(r["upright_cos"] or 0),
                     "u_meas": np.array([float(r[f"u_meas.{d}"]) for d in dofs]),
                     "u_cmd": np.array([float(r[f"u_cmd.{d}"]) for d in dofs])}
        tgt.count += 1


def main():
    p = argparse.ArgumentParser(description="Live 3D view of the real robot")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--listen", type=int, metavar="PORT", help="watch the robot live over UDP")
    src.add_argument("--replay", metavar="CSV", help="play back a recorded frame log")
    src.add_argument("--selftest", action="store_true",
                     help="check posing and packet round-trip without opening a window")
    p.add_argument("--xml", default=str(ROOT / "models" / "humanoid_real_v2.xml"))
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    p.add_argument("--from", dest="t_from", type=float, default=None, metavar="SEC",
                   help="replay only from this t_rel. diagnose_run.py prints the window "
                        "for each event it finds, so you can watch the moment itself.")
    p.add_argument("--to", dest="t_to", type=float, default=None, metavar="SEC",
                   help="replay only up to this t_rel")
    p.add_argument("--no-ghost", action="store_true", help="hide the commanded skeleton")
    p.add_argument("--fps", type=float, default=60.0,
                   help="render rate, held steady regardless of when packets arrive")
    p.add_argument("--smooth", type=float, default=0.05,
                   help="seconds of easing toward the newest pose. UDP over Wi-Fi arrives "
                        "unevenly, so drawing only on arrival makes a steady 40 Hz robot "
                        "look stuttery; this costs ~50 ms of lag to remove that. 0 disables.")
    p.add_argument("--status-every", type=int, default=40)
    args = p.parse_args()

    poser = Poser(args.xml, args.map)
    print(f"model {Path(args.xml).name}: {poser.model.nq} qpos, {poser.model.nu} actuators")

    if args.selftest:
        centers = poser.map.centers.astype(float)
        dev = poser.pose(centers, centers + 8, [0.05, -0.02, -0.99])
        pts = poser._points(poser.data)
        low = min(v[2] for v in pts.values())
        print(f"  posed at map centres: worst commanded-vs-actual {np.abs(dev).max():.2f} deg")
        print(f"  lowest skeleton point {low:+.3f} m (should sit near the floor)")
        print(f"  pelvis {pts['pelvis'].round(3)}  head {pts['head'].round(3)}")
        row = [0.0] * len(__import__("frame_log").SCALAR_COLUMNS) + list(centers) \
            + list(centers + 8) + [0.0] * 34
        row[2], row[1] = 7, 0.175
        row[5], row[6], row[7], row[11] = 0.05, -0.02, -0.99, 0.99
        sink = UdpSink("127.0.0.1", 9999, poser.nj)
        pkt = struct.pack(UdpSink.FMT, UdpSink.MAGIC, 7, 0.175, 0.05, -0.02, -0.99, 0.99,
                          *centers, *(centers + 8))
        back = parse_packet(pkt, poser.nj)
        sink.close()
        ok = (back["step"] == 7 and abs(back["t"] - 0.175) < 1e-6
              and np.allclose(back["u_meas"], centers, atol=1e-3))
        print(f"  packet {len(pkt)} bytes, round-trip {'OK' if ok else 'FAILED'}")
        print("selftest passed" if ok else "selftest FAILED")
        return 0 if ok else 1

    import threading

    import mujoco.viewer

    tgt = Target(poser.nj)
    producer = threading.Thread(
        target=(produce_live if args.listen else produce_replay),
        args=((args.listen, poser.nj, tgt) if args.listen
              else (args.replay, poser.nj, args.speed, tgt, args.t_from, args.t_to)),
        daemon=True)
    producer.start()

    # Displayed state, eased toward the target. Smoothing is applied to the ENCODER
    # UNITS, not to the drawn geometry, so the robot stays a physically consistent pose
    # at every rendered instant rather than becoming a blend of two shapes.
    show_meas = show_cmd = show_pg = None
    period = 1.0 / max(args.fps, 1.0)

    with mujoco.viewer.launch_passive(poser.model, poser.data,
                                      show_left_ui=False, show_right_ui=False) as v:
        v.cam.distance = 1.6
        v.cam.elevation = -12
        v.cam.azimuth = 135
        v.cam.lookat[:] = [0, 0, 0.35]
        n = 0
        last = time.time()
        last_report = 0
        while v.is_running():
            t_frame = time.time()
            f = tgt.frame
            if f is not None:
                if show_meas is None:
                    show_meas, show_cmd, show_pg = f["u_meas"].copy(), f["u_cmd"].copy(), f["pg"].copy()
                # dt-aware exponential ease: alpha depends on the real frame interval, so
                # the motion looks the same whether the renderer hits 60 fps or 30.
                a = 1.0 - np.exp(-period / max(args.smooth, 1e-4)) if args.smooth > 0 else 1.0
                show_meas += a * (f["u_meas"] - show_meas)
                show_cmd += a * (f["u_cmd"] - show_cmd)
                show_pg += a * (f["pg"] - show_pg)
                dev = poser.pose(show_meas, show_cmd, show_pg)
                v.user_scn.ngeom = 0
                if not args.no_ghost:
                    poser.draw_ghost(v.user_scn)
                n += 1
                if args.status_every and tgt.count - last_report >= args.status_every:
                    last_report = tgt.count
                    worst = int(np.argmax(np.abs(dev)))
                    fps = n / max(time.time() - last, 1e-9)
                    n, last = 0, time.time()
                    print(f"\r  t={f['t']:7.2f}s  upright={f['upright']:.3f}  "
                          f"worst {poser.map.joints[worst].dof} {dev[worst]:+6.2f}deg  "
                          f"{tgt.count} frames in, {fps:4.1f} fps out ", end="", flush=True)
            v.sync()
            time.sleep(max(0.0, period - (time.time() - t_frame)))
    tgt.done = True
    print("\nviewer closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
