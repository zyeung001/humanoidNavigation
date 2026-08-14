#!/usr/bin/env python3
"""Fit a servo model from excite_joint.py logs: dead time, tau, phase margin, backlash.

This is the half of system identification that turns recorded motion into numbers
sim can use. Every oscillation on this robot -- standing at 1.96 Hz, the arm at
0.9 Hz, the sensor-free offline emulation at 1.6-1.8 Hz -- has been attributed to
loop delay and phase margin, but the phase margin itself was inferred from a bench
step response rather than measured. The chirp fit measures it directly.

WHAT EACH FIT PRODUCES, and what to distrust about it:

  step      Dead time and first-order tau, per edge, separated by direction and
            amplitude. Dead time is taken as the first sample moving more than
            MOTION_COUNTS encoder counts, so it can never resolve better than one
            sample period plus one quantum -- quote it with that floor in mind.
            Asymmetry between rising and falling edges is real information: it
            usually means gravity is helping one direction.

  chirp     Gain and phase against frequency, by projecting both command and
            response onto the chirp's own known phase function. That is matched to
            a sweeping signal in a way a fixed-frequency DFT bin is not. The
            reported f_180 is where feedback through this actuator turns positive;
            every measured limit cycle on this robot should sit just below it.
            Cross-checked against the step model: if the step-fitted delay and tau
            do not predict the measured phase curve, one of the two fits is wrong,
            and that disagreement is worth more than either number alone.

  backlash  Width of the hysteresis loop between the rising and falling branches.
            CAUTION: this measures backlash PLUS the servo's own deadband, which
            cannot be separated by this test. The deadband is readable from the
            registers (servo_deadband.py; arms measured at 16 units, legs at 1), so
            subtract it -- pass --deadband-units to have that done here.

  python scripts/debug/fit_actuator.py --logs logs/
  python scripts/debug/fit_actuator.py --logs logs/ --joint waist_pitch --plot
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MOTION_COUNTS = 2.0     # encoder counts that count as "it started moving" (1 count is noise)
SETTLE_FRAC = 0.632     # first-order time constant crossing


def load_log(path):
    """One excite_joint.py CSV + its sidecar -> arrays. Returns None if unusable."""
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("script") != "excite_joint.py":
        return None
    joint = meta["joint"]
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 20:
        return None
    t = np.array([float(r["t_rel"]) for r in rows])
    u_cmd = np.array([float(r[f"u_cmd.{joint}"]) for r in rows])
    u_meas = np.array([float(r[f"u_meas.{joint}"]) for r in rows])
    ok = np.isfinite(u_meas)
    if ok.sum() < 20:
        return None
    if not ok.all():        # bridge failed reads so edges are not split by gaps
        u_meas = np.interp(t, t[ok], u_meas[ok])
    return {"t": t, "u_cmd": u_cmd, "u_meas": u_meas, "meta": meta, "joint": joint,
            "mode": meta["mode"], "path": path, "n_bad": int((~ok).sum()),
            "upr": float(meta["units_per_rad"]), "fs": len(t) / max(t[-1] - t[0], 1e-9)}


def fit_step(log):
    """Dead time + tau from every command edge, by Smith's two-point method.

    Deliberately NOT "time until it moved more than N counts". That threshold estimator
    is biased high by everything standing between the command and visible motion --
    gear lash, the encoder quantum, the threshold itself -- and on a synthetic plant
    with a known 50 ms dead time and 4 counts of lash it reported 75 ms.

    Smith's method reads the times at 35.3% and 85.3% of the SETTLED response, which
    for a first-order-plus-dead-time step are D + 0.435*tau and D + 1.917*tau. Working
    in fractions of the final value makes it insensitive to how much amplitude the lash
    ate, and it never has to decide when motion "started". The first-motion time is
    still reported, because that is the number you see by eye on a plot.
    """
    t, uc, um = log["t"], log["u_cmd"], log["u_meas"]
    edges = np.flatnonzero(np.abs(np.diff(uc)) > MOTION_COUNTS) + 1
    out = []
    for i, k in enumerate(edges):
        stop = edges[i + 1] if i + 1 < len(edges) else len(t)
        if stop - k < 10:
            continue
        y0 = np.median(um[max(0, k - 8):k])           # pre-edge level
        target = uc[k] - uc[k - 1]
        seg_t, seg_y = t[k:stop] - t[k], um[k:stop] - y0
        yf = np.median(seg_y[-max(3, len(seg_y) // 5):])   # settled response
        if abs(yf) < 3 * MOTION_COUNTS:               # never really moved; nothing to fit
            continue
        frac = seg_y / yf                              # signed, so both directions work
        t_at = {}
        for f in (0.353, 0.853):
            hit = np.flatnonzero(frac >= f)
            if not len(hit):
                break
            t_at[f] = seg_t[hit[0]]
        if len(t_at) < 2 or t_at[0.853] <= t_at[0.353]:
            continue
        tau = 0.6749 * (t_at[0.853] - t_at[0.353])
        delay = t_at[0.353] - 0.4353 * tau
        if tau <= 0 or delay < -0.5 * tau:
            continue
        moved = np.flatnonzero(np.abs(seg_y) > MOTION_COUNTS)
        # A step held for less than delay + 5*tau never reaches its final value, so the
        # "settled" level the percentages are measured against is itself wrong and the
        # whole fit skews. On synthetic plants a 0.6 s hold turned a true 70/180 ms into
        # a reported 96/135. Flag it rather than quietly returning the wrong number.
        out.append({"dir": "up" if target > 0 else "down", "cmd_units": abs(target),
                    "resp_units": abs(yf), "delay_ms": max(delay, 0.0) * 1000,
                    "tau_ms": tau * 1000, "gain": abs(yf) / abs(target),
                    "settled": bool(seg_t[-1] >= delay + 5 * tau),
                    "hold_s": float(seg_t[-1]), "need_s": float(delay + 5 * tau),
                    "first_motion_ms": (seg_t[moved[0]] * 1000) if len(moved) else float("nan")})
    return out


def _chirp_phase(t, meta):
    c = meta.get("chirp") or {}
    f0, f1, T = c.get("f0", 0.2), c.get("f1", 5.0), c.get("secs", max(t[-1], 1e-9))
    return 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * T)), f0, f1, T


def fit_chirp(log, n_bins=14):
    """Gain/phase vs frequency by projecting onto the chirp's own phase function."""
    t, uc, um = log["t"], log["u_cmd"], log["u_meas"]
    phase, f0, f1, T = _chirp_phase(t, log["meta"])
    inst_f = f0 + (f1 - f0) * t / max(T, 1e-9)
    uc = uc - uc.mean()
    um = um - um.mean()
    ref = np.exp(-1j * phase)
    edges = np.linspace(t[0], t[-1], n_bins + 1)
    freq, gain, ph = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (t >= a) & (t < b)
        if s.sum() < 20:
            continue
        U, Y = np.sum(uc[s] * ref[s]), np.sum(um[s] * ref[s])
        if abs(U) < 1e-9:
            continue
        H = Y / U
        freq.append(float(np.mean(inst_f[s])))
        gain.append(float(abs(H)))
        ph.append(float(np.degrees(np.angle(H))))
    freq, gain = np.array(freq), np.array(gain)
    ph = np.unwrap(np.radians(np.array(ph)))
    ph = np.degrees(ph)
    ph -= 360.0 * np.round(ph[0] / 360.0)     # start in (-180, 180]
    f180 = None
    below = np.flatnonzero(ph <= -180.0)
    if len(below) and below[0] > 0:
        i = below[0]
        f180 = float(np.interp(-180.0, [ph[i], ph[i - 1]], [freq[i], freq[i - 1]]))
    return {"freq": freq, "gain": gain, "phase_deg": ph, "f180": f180}


def model_phase(freq, delay_ms, tau_ms):
    """First-order lag + pure dead time, the model standing_env.actuator_lag implements."""
    w = 2 * np.pi * np.asarray(freq)
    return np.degrees(-np.arctan(w * tau_ms / 1000.0) - w * delay_ms / 1000.0)


def fit_backlash(log, deadband_units=0.0, delay_ms=None, tau_ms=None):
    """Hysteresis loop width between rising and falling branches of the triangle.

    The raw loop width is NOT backlash. A triangle is always moving, so the actuator's
    own lag contributes a velocity-dependent tracking error that trails the command on
    the way up and leads it on the way down -- opposite signs, so it adds to the loop
    width exactly like lash does. On the synthetic plant that dynamic term was larger
    than the lash it was supposed to be measuring (4.6 units of lag against 4.0 of lash,
    reported together as 8.0).

    Given delay and tau from the step fit, that term is 2*v*(delay+tau) and is removed
    here. Without them, only the raw width is reported and it is an upper bound. Driving
    the triangle slower shrinks the correction and is the cheapest way to trust it.
    """
    t, uc, um = log["t"], log["u_cmd"], log["u_meas"]
    # The command is quantized to integer units and the ramp is slow, so consecutive
    # samples are usually IDENTICAL. Differentiating that staircase directly returns
    # zero on most samples, which both mislabels the branch and reports a speed of 0.
    # Smooth over ~50 ms first (several counts of travel) before taking the sign.
    w = max(3, int(0.05 * len(t) / max(t[-1] - t[0], 1e-9)))
    kern = np.ones(w) / w
    uc_s = np.convolve(uc, kern, mode="same")
    duc = np.gradient(uc_s, t)
    lo, hi = np.percentile(uc, [30, 70])
    band = (uc > lo) & (uc < hi)          # mid-travel only: the ends are turnarounds
    band[:w] = band[-w:] = False          # convolution edges are not trustworthy
    rising, falling = band & (duc > 0), band & (duc < 0)
    if rising.sum() < 10 or falling.sum() < 10:
        return None
    off_r = np.median(um[rising] - uc[rising])
    off_f = np.median(um[falling] - uc[falling])
    width = abs(off_f - off_r)
    # Total variation over the run: robust to quantization in a way a pointwise
    # derivative is not, since the individual unit steps still sum to the true path.
    speed = float(np.sum(np.abs(np.diff(uc))) / max(t[-1] - t[0], 1e-9))
    dyn = None
    if delay_ms is not None and tau_ms is not None:
        dyn = 2.0 * speed * (delay_ms + tau_ms) / 1000.0
    corrected = width if dyn is None else max(width - dyn, 0.0)
    lash = max(corrected - 2 * deadband_units, 0.0)
    return {"width_units": width, "width_deg": np.degrees(width / log["upr"]),
            "speed_units_s": speed, "dynamic_units": dyn, "corrected_units": corrected,
            "lash_units": lash, "lash_deg": np.degrees(lash / log["upr"]),
            "rising_offset": off_r, "falling_offset": off_f}


def main():
    p = argparse.ArgumentParser(description="Fit servo dynamics from excite_joint.py logs")
    p.add_argument("--logs", default="logs", help="directory of logs, or a single CSV")
    p.add_argument("--joint", default=None, help="restrict to one joint")
    p.add_argument("--deadband-units", type=float, default=0.0,
                   help="servo deadband (reg 26/27) to subtract from the hysteresis width; "
                        "measured 16 on the arms, 1 on legs+waist")
    p.add_argument("--plot", action="store_true", help="write a Bode plot per joint (needs matplotlib)")
    p.add_argument("--out", default=None, help="write the fitted numbers to JSON")
    args = p.parse_args()

    src = Path(args.logs)
    files = sorted(src.glob("*.csv")) if src.is_dir() else [src]
    logs = [x for x in (load_log(f) for f in files) if x]
    if args.joint:
        logs = [x for x in logs if x["joint"] == args.joint]
    if not logs:
        raise SystemExit(f"no excite_joint.py logs found in {src}")

    by_joint = defaultdict(dict)
    for x in logs:
        by_joint[x["joint"]][x["mode"]] = x
    print(f"{len(logs)} logs, {len(by_joint)} joint(s): {', '.join(sorted(by_joint))}\n")

    results = {}
    for joint in sorted(by_joint):
        modes = by_joint[joint]
        r = {}
        print(f"=== {joint} " + "=" * (58 - len(joint)))
        for mode, x in sorted(modes.items()):
            print(f"    {mode:9s} {len(x['t']):6d} samples @ {x['fs']:5.0f} Hz"
                  + (f"   ({x['n_bad']} failed reads bridged)" if x["n_bad"] else ""))

        if "step" in modes:
            edges = fit_step(modes["step"])
            if edges:
                d = np.array([e["delay_ms"] for e in edges])
                ta = np.array([e["tau_ms"] for e in edges])
                g = np.array([e["gain"] for e in edges])
                r["delay_ms"] = float(np.median(d))
                r["tau_ms"] = float(np.median(ta))
                r["step_gain"] = float(np.median(g))
                fm = np.array([e["first_motion_ms"] for e in edges])
                unsettled = [e for e in edges if not e["settled"]]
                print(f"\n  STEP  ({len(edges)} edges)")
                if unsettled:
                    need = max(e["need_s"] for e in unsettled)
                    r["unsettled"] = len(unsettled)
                    print(f"    !! {len(unsettled)}/{len(edges)} edges did NOT settle "
                          f"({unsettled[0]['hold_s']:.2f} s hold, needs {need:.2f} s). "
                          f"These numbers are biased -- re-run excite_joint with "
                          f"--hold {np.ceil(need * 10) / 10:.1f} or longer.")
                print(f"    dead time  {np.median(d):6.1f} ms   [{d.min():.0f}..{d.max():.0f}]"
                      f"   (floor: 1 sample = {1000 / modes['step']['fs']:.1f} ms)")
                print(f"    first motion {np.nanmedian(fm):4.0f} ms   -- what you see by eye; "
                      f"later than the dead time because lash and the encoder quantum "
                      f"must be crossed first")
                print(f"    tau        {np.median(ta):6.1f} ms   [{ta.min():.0f}..{ta.max():.0f}]")
                print(f"    dc gain    {np.median(g):6.3f}       (1.0 = reaches the command)")
                for direction in ("up", "down"):
                    sub = [e for e in edges if e["dir"] == direction]
                    if sub:
                        print(f"      {direction:5s} delay {np.median([e['delay_ms'] for e in sub]):5.1f} ms"
                              f"  tau {np.median([e['tau_ms'] for e in sub]):5.1f} ms"
                              f"  gain {np.median([e['gain'] for e in sub]):.3f}")

        if "chirp" in modes:
            c = fit_chirp(modes["chirp"])
            r["f180_hz"] = c["f180"]
            print("\n  CHIRP")
            print(f"    {'f (Hz)':>8} {'gain':>7} {'phase':>8}")
            for f, g, ph in zip(c["freq"], c["gain"], c["phase_deg"]):
                print(f"    {f:8.2f} {g:7.3f} {ph:7.1f} deg")
            if c["f180"]:
                print(f"    --> phase crosses -180 deg at {c['f180']:.2f} Hz. Measured limit "
                      f"cycles should sit just BELOW this.")
            else:
                print("    --> phase never reached -180 deg in this sweep; raise --f1 and re-run.")
            if "delay_ms" in r:
                pred = model_phase(c["freq"], r["delay_ms"], r["tau_ms"])
                err = float(np.max(np.abs(pred - c["phase_deg"])))
                r["model_phase_err_deg"] = err
                verdict = ("consistent" if err < 25 else
                           "the plant is NOT first-order+delay alone -- backlash or a "
                           "higher-order term is adding phase the model does not have")
                print(f"    step-model prediction vs measured phase: max error {err:.0f} deg")
                print(f"    --> {verdict}")

        if "backlash" in modes:
            b = fit_backlash(modes["backlash"], args.deadband_units,
                             r.get("delay_ms"), r.get("tau_ms"))
            if b:
                r["hysteresis_units"] = b["width_units"]
                r["lash_units"] = b["lash_units"]
                print("\n  BACKLASH")
                print(f"    raw loop width   {b['width_units']:5.1f} units "
                      f"= {b['width_deg']:.2f} deg   at {b['speed_units_s']:.1f} units/s")
                if b["dynamic_units"] is not None:
                    print(f"    minus dynamic lag {b['dynamic_units']:5.1f} units "
                          f"(2 x speed x (delay+tau)) -> {b['corrected_units']:.1f}")
                else:
                    print("    no step fit available -> this is an UPPER BOUND; the "
                          "actuator's own lag is still in it")
                if args.deadband_units:
                    print(f"    minus 2x{args.deadband_units:.0f}-unit deadband -> lash "
                          f"{b['lash_units']:.1f} units = {b['lash_deg']:.2f} deg")
                else:
                    print("    pass --deadband-units to subtract the servo deadband too")

        if args.plot and "chirp" in modes:
            _plot(joint, fit_chirp(modes["chirp"]), r)
        results[joint] = r
        print()

    _emit_config(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


def _emit_config(results):
    d = [v["delay_ms"] for v in results.values() if "delay_ms" in v]
    t = [v["tau_ms"] for v in results.values() if "tau_ms" in v]
    if not d or not t:
        return
    print("=" * 64)
    print("standing_env.py actuator_lag, from the measured spread across joints.")
    print("Randomize over the RANGE, not the mean: the point of the range is that the")
    print("policy cannot assume a plant it will not meet on the robot.\n")
    print("standing:")
    print(f"  actuator_delay_ms: [{min(d):.0f}, {max(d):.0f}]     # measured {min(d):.0f}-{max(d):.0f} over {len(d)} joints")
    print(f"  actuator_tau_ms:   [{min(t):.0f}, {max(t):.0f}]     # measured {min(t):.0f}-{max(t):.0f}")
    f = [v["f180_hz"] for v in results.values() if v.get("f180_hz")]
    if f:
        print(f"\n# phase crosses -180 deg at {min(f):.2f}-{max(f):.2f} Hz across joints.")
        print("# Compare against the observed limit cycles: standing 1.96 Hz, arm 0.9 Hz.")


def _plot(joint, c, r):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (--plot needs matplotlib; skipped)")
        return
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].semilogx(c["freq"], 20 * np.log10(np.maximum(c["gain"], 1e-6)), "o-")
    ax[0].set_ylabel("gain (dB)")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[1].semilogx(c["freq"], c["phase_deg"], "o-", label="measured")
    if "delay_ms" in r:
        ax[1].semilogx(c["freq"], model_phase(c["freq"], r["delay_ms"], r["tau_ms"]),
                       "--", label=f"step fit ({r['delay_ms']:.0f}ms + {r['tau_ms']:.0f}ms)")
    ax[1].axhline(-180, color="r", ls=":", label="-180 deg")
    ax[1].set_ylabel("phase (deg)")
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, which="both", alpha=0.3)
    ax[0].set_title(f"{joint} frequency response")
    out = Path("logs") / f"bode_{joint}.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"    plot -> {out}")


if __name__ == "__main__":
    main()
