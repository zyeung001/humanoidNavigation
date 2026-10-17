#!/usr/bin/env python3
"""Turn a frame log into a window that shows what the robot was doing.

Step 3 of the instrumentation plan. The state itself was already recoverable --
frame_log.pose_from_row() turns any logged frame into absolute joint angles plus pelvis
attitude -- but a column of numbers is not a window. This renders the pose.

WHAT IT SHOWS, and why these things and not others:
  - Two orthographic stick figures, front and side, built by posing the actual MJCF with
    the joint angles that were RECORDED, not the ones that were commanded. The commanded
    pose is drawn ghosted behind it. The gap between the two lines IS the tracking error,
    which is the quantity every unresolved question on this robot comes back to.
  - A deviation rack: one row per joint, zero-centred, actual as a bar and commanded as a
    tick. Sorted by error so the worst joint is always the top row rather than wherever
    the kinematic tree happens to put it.
  - Loop health as chips, because a log taken at 31 Hz with dropped rows means something
    different from the same log taken at 40 Hz, and that difference should not need to be
    dug out of a column.

The page is self-contained: geometry is solved here with MuJoCo and baked in as arrays, so
the viewer does no kinematics and needs nothing at runtime.

  python scripts/debug/pose_viewer.py --log logs/20260814_154939_standing.csv
  python scripts/debug/pose_viewer.py --log <csv> --out page.html --fps 20
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Stick-figure topology, expressed as JOINT ANCHORS rather than body origins.
#
# This MJCF came out of CAD, and its body frames sit wherever the exporter put them --
# posing the skeleton from body xpos puts the torso 65 cm off to one side. mj_forward
# publishes each hinge's true world pivot in data.xanchor, which is what a stick figure
# actually wants. Extremities that are not hinges (feet, hands) come from the geom or
# body centre of mass instead.
#
# Each point is (kind, name): "j" = joint anchor, "g" = geom position, "b" = body COM,
# "mid" = midpoint of two other points already resolved.
POINTS = {
    "pelvis":   ("mid", ("r_hip", "l_hip")),
    "r_hip":    ("j", "Revolute 18"), "r_knee": ("j", "Revolute 21"), "r_foot": ("g", 27),
    "l_hip":    ("j", "Revolute 24"), "l_knee": ("j", "Revolute 27"), "l_foot": ("g", 47),
    "waist":    ("j", "Revolute 22"), "chest": ("j", "Revolute 19"), "head": ("b", "0003_8"),
    "r_sh":     ("j", "Revolute 2"),  "r_elb": ("j", "Revolute 1"),  "r_hand": ("b", "mount_2"),
    "l_sh":     ("j", "Revolute 4"),  "l_elb": ("j", "Revolute 3"),  "l_hand": ("b", "mount_3"),
}
CHAINS = [
    ("r_leg", ["pelvis", "r_hip", "r_knee", "r_foot"]),
    ("l_leg", ["pelvis", "l_hip", "l_knee", "l_foot"]),
    ("spine", ["pelvis", "waist", "chest", "head"]),
    ("r_arm", ["chest", "r_sh", "r_elb", "r_hand"]),
    ("l_arm", ["chest", "l_sh", "l_elb", "l_hand"]),
]


def load_frames(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    meta_path = Path(path).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return rows, meta


def solve(rows, meta, xml, fps):
    """Pose the model per frame and project the skeleton. Returns everything the page needs."""
    import mujoco
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "deploy"))
    from sim_real_map import SimRealMap

    smap = SimRealMap(ROOT / "config" / "joint_servo_map.yaml")
    dofs = meta.get("dofs") or [j.dof for j in smap.joints]
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    jid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
           for k, nm in POINTS.values() if k == "j"}
    bid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
           for k, nm in POINTS.values() if k == "b"}
    for nm, i in {**jid, **bid}.items():
        if i < 0:
            raise SystemExit(f"{xml} has no '{nm}' -- the skeleton map does not fit this model")

    t = np.array([float(r["t_rel"]) for r in rows])
    stride = max(1, int(round(len(rows) / max(t[-1] - t[0], 1e-9) / fps)))
    keep = list(range(0, len(rows), stride))

    u_meas = np.array([[float(r[f"u_meas.{j}"]) for j in dofs] for r in rows])
    q_cmd = np.array([[float(r[f"q_cmd.{j}"]) for j in dofs] for r in rows])
    q_meas = np.array([smap.units_to_rad(u) for u in u_meas])

    def skeleton(q):
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[qadr] = q
        mujoco.mj_forward(m, d)
        # World is +X forward, +Y left, +Z up. Front view looks down -X, so the robot's
        # left lands on the viewer's right; side view looks along +Y.
        raw = {}
        for name, (kind, ref) in POINTS.items():
            if kind == "j":
                raw[name] = d.xanchor[jid[ref]]
            elif kind == "g":
                raw[name] = d.geom_xpos[ref]
            elif kind == "b":
                raw[name] = d.xipos[bid[ref]]
        for name, (kind, ref) in POINTS.items():
            if kind == "mid":
                raw[name] = 0.5 * (np.asarray(raw[ref[0]]) + np.asarray(raw[ref[1]]))
        return {n: (float(-p[1]), float(p[0]), float(p[2])) for n, p in raw.items()}

    frames = []
    for k in keep:
        a, c = skeleton(q_meas[k]), skeleton(q_cmd[k])
        frames.append({
            "t": round(float(t[k]), 3),
            "a": {n: [round(v[0], 4), round(v[1], 4), round(v[2], 4)] for n, v in a.items()},
            "c": {n: [round(v[0], 4), round(v[1], 4), round(v[2], 4)] for n, v in c.items()},
            "dev": [round(float(np.degrees(q_cmd[k][i] - q_meas[k][i])), 2)
                    for i in range(len(dofs))],
            "pg": [round(float(rows[k].get("pg_x", 0) or 0), 3),
                   round(float(rows[k].get("pg_y", 0) or 0), 3),
                   round(float(rows[k].get("pg_z", 0) or 0), 3)],
        })

    # Put the figure where a viewer expects it: ground under the lowest foot, pelvis on the
    # centre line. Then fit the viewBox to the actual extents rather than guessing them --
    # a hardcoded box silently crops whichever pose happens to be widest.
    allp = np.array([[fr[s][n] for n in POINTS] for fr in frames for s in ("a", "c")])
    z0 = float(min(allp[:, :, 2].min(), 0.0))
    cx = float(np.mean([fr["a"]["pelvis"][0] for fr in frames]))
    cy = float(np.mean([fr["a"]["pelvis"][1] for fr in frames]))
    for fr in frames:
        for s in ("a", "c"):
            fr[s] = {n: [round(p[0] - cx, 4), round(p[1] - cy, 4), round(p[2] - z0, 4)]
                     for n, p in fr[s].items()}
    pts = np.array([[fr[s][n] for n in POINTS] for fr in frames for s in ("a", "c")])
    pad = 0.05
    box = {
        "front": [float(pts[:, :, 0].min()) - pad, -(float(pts[:, :, 2].max()) + pad),
                  float(np.ptp(pts[:, :, 0])) + 2 * pad, float(pts[:, :, 2].max()) + 2 * pad],
        "side": [float(pts[:, :, 1].min()) - pad, -(float(pts[:, :, 2].max()) + pad),
                 float(np.ptp(pts[:, :, 1])) + 2 * pad, float(pts[:, :, 2].max()) + 2 * pad],
    }
    span = max(box["front"][2], box["side"][2], box["front"][3])
    for k in box:      # square both views on the same scale so limbs are comparable
        box[k] = [box[k][0] - (span - box[k][2]) / 2, box[k][1] - (span - box[k][3]) / 2,
                  span, span]
        box[k] = [round(v, 4) for v in box[k]]

    dev_all = np.degrees(q_cmd - q_meas)
    loop = np.array([float(r["loop_ms"]) for r in rows if r.get("loop_ms")])
    health = {
        "frames": len(rows),
        "kept": len(keep),
        "duration": round(float(t[-1] - t[0]), 2),
        "hz": round(len(rows) / max(t[-1] - t[0], 1e-9), 1),
        "loop_ms_p95": round(float(np.percentile(loop, 95)), 1) if len(loop) else None,
        "rejects": int(float(rows[-1].get("rej_total", 0) or 0)),
        "dropped": int(meta.get("rows_dropped", 0) or 0),
        "worst_dev": round(float(np.abs(dev_all).max()), 2),
        "worst_joint": dofs[int(np.argmax(np.abs(dev_all).max(axis=0)))],
        "mean_dev": round(float(np.abs(dev_all).mean()), 3),
    }
    envelope = [[round(float(t[k]), 2), round(float(np.abs(dev_all[k]).max()), 2)] for k in keep]
    return {"dofs": dofs, "frames": frames, "health": health, "envelope": envelope,
            "box": box, "stroke": round(span / 90, 5), "dot": round(span / 130, 5),
            "meta": {k: meta.get(k) for k in
                     ("script", "source", "note", "git_sha", "started_iso", "hz", "tau")},
            "file": Path(rows and meta.get("_path", "")).name}


TEMPLATE = r"""<title>__TITLE__</title>
<style>
:root{
  --ground:#f6f7f8; --panel:#ffffff; --sunk:#eceef1; --line:#d8dce1;
  --ink:#171c22; --ink-2:#4b555f; --ink-3:#7b8791;
  --cmd:#3d87a6; --act:#c46b28;
  --ok:#4d7d4a; --warn:#94770f; --bad:#a83c2b;
  --shadow:0 1px 2px rgba(16,21,27,.06),0 8px 24px -16px rgba(16,21,27,.35);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#10151b; --panel:#171d25; --sunk:#0c1116; --line:#28313b;
  --ink:#e6eaee; --ink-2:#a2aeb9; --ink-3:#6c7883;
  --cmd:#5fa8c7; --act:#e0894a;
  --ok:#6e9e6b; --warn:#c9a227; --bad:#c4523f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -18px rgba(0,0,0,.8);
}}
:root[data-theme="dark"]{
  --ground:#10151b; --panel:#171d25; --sunk:#0c1116; --line:#28313b;
  --ink:#e6eaee; --ink-2:#a2aeb9; --ink-3:#6c7883;
  --cmd:#5fa8c7; --act:#e0894a;
  --ok:#6e9e6b; --warn:#c9a227; --bad:#c4523f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -18px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:14px; line-height:1.5; padding:20px;
}
.mono{font-family:ui-monospace,"SF Mono","Cascadia Mono","JetBrains Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
.eyebrow{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3);font-weight:600}
h1{margin:0;font-size:19px;font-weight:650;letter-spacing:-.01em;text-wrap:balance}
header{display:flex;flex-wrap:wrap;gap:12px 22px;align-items:flex-end;justify-content:space-between}
.badge{display:inline-flex;align-items:center;gap:6px;padding:2px 9px;border-radius:3px;
  font-size:10px;letter-spacing:.1em;text-transform:uppercase;font-weight:700;
  border:1px solid var(--line);color:var(--ink-2);background:var(--sunk)}
.badge.sim{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,var(--line))}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:7px 11px;
  display:flex;flex-direction:column;gap:2px;min-width:92px;box-shadow:var(--shadow)}
.chip b{font-size:15px;font-weight:650;letter-spacing:-.01em}
.chip.ok b{color:var(--ok)} .chip.warn b{color:var(--warn)} .chip.bad b{color:var(--bad)}
.grid{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:14px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:7px;
  padding:13px 14px;box-shadow:var(--shadow);min-width:0}
.card h2{margin:0 0 10px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--ink-3);font-weight:650}
.stage{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.view{background:var(--sunk);border-radius:5px;position:relative}
.view figcaption{position:absolute;left:8px;top:6px;font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);font-weight:600}
svg{display:block;width:100%;height:auto}
.key{display:flex;gap:16px;margin-top:9px;font-size:11px;color:var(--ink-2)}
.key i{display:inline-block;width:14px;height:0;border-top-width:2px;border-top-style:solid;
  vertical-align:middle;margin-right:5px}
.rack{display:flex;flex-direction:column;gap:1px;max-height:406px;overflow-y:auto}
.row{display:grid;grid-template-columns:106px 1fr 54px;gap:8px;align-items:center;
  padding:2px 0;font-size:11px}
.row .nm{color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.gauge{position:relative;height:13px;background:var(--sunk);border-radius:2px;overflow:hidden}
.gauge .zero{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--line)}
.gauge .bar{position:absolute;top:2px;bottom:2px;background:var(--act);border-radius:1px}
.row .val{text-align:right;font-size:11px;color:var(--ink-2)}
.row.hot .val{color:var(--bad);font-weight:650}
.tl{position:relative;cursor:pointer}
.transport{display:flex;align-items:center;gap:10px;margin-top:9px;flex-wrap:wrap}
button{font:inherit;font-size:12px;padding:5px 13px;border-radius:4px;cursor:pointer;
  border:1px solid var(--line);background:var(--sunk);color:var(--ink);font-weight:600}
button:hover{border-color:var(--ink-3)}
button:focus-visible{outline:2px solid var(--cmd);outline-offset:2px}
input[type=range]{flex:1;min-width:150px;accent-color:var(--cmd)}
.t{font-size:12px;color:var(--ink-2);min-width:118px}
footer{color:var(--ink-3);font-size:11px;border-top:1px solid var(--line);padding-top:11px}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
<header>
  <div>
    <div class="eyebrow">Frame log · pose replay</div>
    <h1>__HEADLINE__</h1>
    <div style="margin-top:6px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
      __BADGES__
    </div>
  </div>
  <div class="chips" id="chips"></div>
</header>

<div class="grid">
  <section class="card">
    <h2>Pose — actual, with commanded ghosted</h2>
    <div class="stage">
      <figure class="view" style="margin:0"><figcaption>Front</figcaption>
        <svg id="front" preserveAspectRatio="xMidYMid meet"></svg></figure>
      <figure class="view" style="margin:0"><figcaption>Side</figcaption>
        <svg id="side" preserveAspectRatio="xMidYMid meet"></svg></figure>
    </div>
    <div class="key mono">
      <span><i style="border-color:var(--act)"></i>actual (encoder)</span>
      <span><i style="border-color:var(--cmd);border-top-style:dashed"></i>commanded</span>
      <span id="pg"></span>
    </div>
  </section>

  <section class="card">
    <h2>Tracking deviation — commanded minus actual, worst first</h2>
    <div class="rack mono" id="rack"></div>
  </section>
</div>

<section class="card">
  <h2>Worst-joint deviation over the run</h2>
  <div class="tl"><svg id="tl" viewBox="0 0 1000 84" preserveAspectRatio="none"
       style="height:84px"></svg></div>
  <div class="transport">
    <button id="play">Play</button>
    <input type="range" id="scrub" min="0" value="0" step="1">
    <span class="t mono" id="time"></span>
  </div>
</section>

<footer>__FOOTER__</footer>
</div>

<script>
const D = __DATA__;
const F = D.frames, N = F.length;
const CH = __CHAINS__;
let i = 0, playing = false, timer = null;
const SW = D.stroke, DOT = D.dot;
document.getElementById("front").setAttribute("viewBox", D.box.front.join(" "));
document.getElementById("side").setAttribute("viewBox", D.box.side.join(" "));

const chips = [
  ["frames", D.health.frames, ""],
  ["rate", D.health.hz + " Hz", D.health.hz > 38 ? "ok" : "warn"],
  ["loop p95", (D.health.loop_ms_p95 ?? "–") + " ms", D.health.loop_ms_p95 > 30 ? "warn" : "ok"],
  ["rejects", D.health.rejects, D.health.rejects > 0 ? "warn" : "ok"],
  ["dropped", D.health.dropped, D.health.dropped > 0 ? "bad" : "ok"],
  ["worst dev", D.health.worst_dev + "°", D.health.worst_dev > 5 ? "bad"
     : D.health.worst_dev > 2 ? "warn" : "ok"],
];
document.getElementById("chips").innerHTML = chips.map(([k, v, s]) =>
  `<div class="chip ${s}"><span class="eyebrow">${k}</span><b class="mono">${v}</b></div>`).join("");

function draw(svgId, key, ax) {
  const f = F[i], out = [];
  for (const set of [["c", "var(--cmd)", 0.35, "3 3"], ["a", "var(--act)", 1, ""]]) {
    const [src, col, op, dash] = set;
    for (const [, chain] of CH) {
      const pts = chain.map(n => f[src][n]).map(p => `${p[ax]},${-p[2]}`).join(" ");
      out.push(`<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="${SW}"
        stroke-linejoin="round" stroke-linecap="round" opacity="${op}"
        ${dash ? `stroke-dasharray="${dash}"` : ""}/>`);
    }
    for (const [, chain] of CH)
      for (const n of chain) {
        const p = f[src][n];
        out.push(`<circle cx="${p[ax]}" cy="${-p[2]}" r="${DOT}" fill="${col}" opacity="${op}"/>`);
      }
  }
  const bx = ax ? D.box.side : D.box.front;
  out.push(`<line x1="${bx[0]}" y1="0" x2="${bx[0]+bx[2]}" y2="0" stroke="var(--line)"
    stroke-width="${SW/2.5}"/>`);
  document.getElementById(svgId).innerHTML = out.join("");
}

function rack() {
  const f = F[i];
  const order = D.dofs.map((n, k) => [n, f.dev[k]])
                      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const SCALE = 8;   // degrees at full deflection
  document.getElementById("rack").innerHTML = order.map(([n, v]) => {
    const w = Math.min(Math.abs(v) / SCALE, 1) * 50;
    const left = v >= 0 ? 50 : 50 - w;
    return `<div class="row ${Math.abs(v) > 5 ? "hot" : ""}">
      <span class="nm">${n}</span>
      <span class="gauge"><span class="zero"></span>
        <span class="bar" style="left:${left}%;width:${w}%"></span></span>
      <span class="val">${v > 0 ? "+" : ""}${v.toFixed(2)}°</span></div>`;
  }).join("");
}

(function timeline() {
  const E = D.envelope, t1 = E[E.length - 1][0] || 1;
  const my = Math.max(...E.map(p => p[1]), 1);
  const pts = E.map(p => `${(p[0] / t1) * 1000},${80 - (p[1] / my) * 72}`).join(" ");
  document.getElementById("tl").innerHTML =
    `<polyline points="${pts}" fill="none" stroke="var(--act)" stroke-width="1.5"/>
     <polyline points="0,80 ${pts} 1000,80" fill="var(--act)" opacity="0.10" stroke="none"/>
     <line id="cur" x1="0" y1="0" x2="0" y2="84" stroke="var(--cmd)" stroke-width="2"/>
     <text x="6" y="12" fill="var(--ink-3)" font-size="10">${my.toFixed(1)}°</text>`;
})();

function render() {
  draw("front", "a", 0); draw("side", "a", 1); rack();
  const f = F[i];
  document.getElementById("time").textContent =
    `t ${f.t.toFixed(2)}s   ${i + 1}/${N}`;
  document.getElementById("pg").textContent =
    `proj-grav [${f.pg.map(v => (v >= 0 ? "+" : "") + v.toFixed(2)).join(" ")}]`;
  const c = document.getElementById("cur");
  if (c) c.setAttribute("x1", (i / (N - 1)) * 1000), c.setAttribute("x2", (i / (N - 1)) * 1000);
  document.getElementById("scrub").value = i;
}

const scrub = document.getElementById("scrub");
scrub.max = N - 1;
scrub.addEventListener("input", e => { i = +e.target.value; stop(); render(); });
function stop() { playing = false; clearInterval(timer); document.getElementById("play").textContent = "Play"; }
document.getElementById("play").addEventListener("click", () => {
  if (playing) return stop();
  playing = true; document.getElementById("play").textContent = "Pause";
  timer = setInterval(() => { i = (i + 1) % N; render(); }, 50);
});
render();
</script>
"""


def main():
    p = argparse.ArgumentParser(description="Render a frame log as a pose replay page")
    p.add_argument("--log", required=True)
    p.add_argument("--xml", default=str(ROOT / "models" / "humanoid_real_v2.xml"))
    p.add_argument("--out", default=str(ROOT / "logs" / "pose_viewer.html"))
    p.add_argument("--fps", type=float, default=20.0, help="frames baked into the page")
    args = p.parse_args()

    rows, meta = load_frames(args.log)
    data = solve(rows, meta, args.xml, args.fps)
    h, mt = data["health"], data["meta"]
    src = (mt.get("source") or "").upper()
    badges = [f'<span class="badge{" sim" if src == "SIMULATION" else ""}">'
              f'{src or "hardware"}</span>',
              f'<span class="badge">{h["duration"]}s · {h["frames"]} frames</span>',
              f'<span class="badge">{Path(args.log).name}</span>']
    note = mt.get("note")
    html = (TEMPLATE
            .replace("__TITLE__", f"Pose replay — {Path(args.log).stem}")
            .replace("__HEADLINE__", "What the robot was doing")
            .replace("__BADGES__", "\n      ".join(badges))
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__CHAINS__", json.dumps(CHAINS))
            .replace("__FOOTER__", (
                f"Worst joint over the whole run: <strong>{h['worst_joint']}</strong> at "
                f"{h['worst_dev']}°, mean absolute deviation {h['mean_dev']}° across all 17. "
                + (f"{note} " if note else "")
                + "Geometry solved with MuJoCo from the recorded encoder values; "
                  "the commanded pose is the policy's action for the same frame.")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"{args.log}: {h['frames']} frames -> {h['kept']} baked at ~{args.fps:g} fps")
    print(f"  worst {h['worst_joint']} {h['worst_dev']}deg, mean |dev| {h['mean_dev']}deg, "
          f"{h['hz']} Hz, {h['rejects']} rejects, {h['dropped']} dropped")
    print(f"  -> {args.out}  ({Path(args.out).stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
