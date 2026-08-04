#!/usr/bin/env python3
"""motion_sweep_24h.py — every alpha motion, every model, one 24 h clock.

The asserted suite (controller_eval.py) runs 6 h for 7B/13B and mixes lambda shapes in. This runs
ALL motions at a fixed lambda over a full 24 h for all three models, which is the horizon the app
defaults to and the one the paper should quote: 24 h is the natural period of an interactive
workload and every controller timescale fits inside it (7B T_shrink ~5 min ~290x, 70B ~75 min ~19x).

The `brief spike` length is sized as a FRACTION OF T_grow, not of the horizon. At 24 h a naive
n/100 spike is 14 min while 7B's T_grow is 47 s -- 18x too long, so the controller grows and the
trace tests the opposite of what it claims. Both regimes are run here and asserted separately.

Output: results/dda32/motion_sweep_24h.csv
Usage:  python3 motion_sweep_24h.py [--dt 30] [--hours 24]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "explorer")))
import dse_metrics as M          # noqa: E402
import dda_controller as DC      # noqa: E402
import traces as TR              # noqa: E402

RECONFIG = {"GROW_B", "GROW_C", "SHRINK", "GREEDY"}

# what each motion must do; None = no hard assertion, reported for inspection
EXPECT = {
    "steady":        dict(max_reconfig=0, why="constant alpha - nothing may move"),
    "brief spike":   dict(max_reconfig=0, why="pulse shorter than T_grow - must be ignored"),
    "flapping":      dict(max_reconfig=0, why="hysteresis must suppress a square wave"),
    "step up":       dict(min_reconfig=1, why="sustained rise must grow (70B exempt)"),
    "step down":     dict(why="no immediate move; one bulk shrink after T_shrink"),
    "slow ramp":     dict(why="few grows, N_L broadly rising"),
    "diurnal":       dict(why="grow on the rise, bulk shrink on the fall"),
    "random walk":   dict(why="drift should not thrash"),
}


def run(model, motion, lam, n, dt, table, base, spike_ticks, period_h=24.0, hours=24.0, **kw):
    al = TR.alpha_motion(motion, n, mean=0.30, amp=0.25, period_h=period_h, hours=hours,
                         rng=np.random.default_rng(0), spike_ticks=spike_ticks)
    c, log = DC.run_trace(model, al, np.full(n, lam), dt=dt, table=table, **kw)
    if log.empty:
        return None, None
    lat = (log.load_L.fillna(0) * dt).sum()
    spill = (log.spilled.fillna(0) * dt).sum()
    unmet = (log.unmet.fillna(0) * dt).sum()
    w = (log.served_by_L.fillna(0) + log.spilled.fillna(0)) * dt
    eff = (log.eff_ttft_s.fillna(0) * w).sum() / w.sum() if w.sum() > 0 else np.nan
    nrec = int(log.action.isin(RECONFIG).sum())
    nbe = sum(1 for e in c.events if e.get("breakeven_t"))
    return c, dict(
        Model=model, motion=motion, policy=kw.get("policy", "hysteresis"),
        alpha_min=round(float(al.min()), 3), alpha_max=round(float(al.max()), 3),
        reconfigs=nrec,
        grow_B=int((log.action == "GROW_B").sum()), grow_C=int((log.action == "GROW_C").sum()),
        shrinks=int((log.action == "SHRINK").sum()),
        breakeven=f"{nbe}/{len(c.events)}" if c.events else "-",
        spill_frac=round(spill / lat, 4) if lat else 0.0,
        unmet_frac=round(unmet / lat, 4) if lat else 0.0,
        eff_TTFT_s=round(eff, 3) if eff == eff else None,
        tokens_lost=round(log.tokens_lost.sum(), 0),
        distinct_splits=int(log.split.replace("", np.nan).nunique()),
        N_L_range=f"{int(log.N_L.min())}-{int(log.N_L.max())}" if log.N_L.notna().any() else "-",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--out", default=os.path.join(M.DDA, "motion_sweep_24h.csv"))
    a = ap.parse_args()

    table = pd.read_csv(os.path.join(M.DDA, "dse_full_table.csv"))
    n = int(a.hours * 3600 / a.dt)
    rows, fails = [], []

    for model in M.MODELS:
        base = DC.make_controller(model, table)
        if base is None:
            continue
        cap = max(r.Tput_L + r.Tput_T for r in base.rungs)
        max_L = max(r.Tput_L for r in base.rungs)
        # lambda must stay servable by POOL L at the busiest alpha, or every trace is trivially
        # infeasible (70B's Pool L tops out at 117 tok/s while the fabric does 1121)
        lam = min(0.45 * cap, 0.85 * max_L / 0.55)
        short = max(1, int(0.5 * base.t_grow() / a.dt))     # < T_grow  -> must be ignored
        long = max(2, int(6.0 * base.t_grow() / a.dt))      # > T_grow  -> must be acted on

        # 'flapping' must FLAP: its half-period has to sit below T_grow. At period_h = horizon the
        # square wave has a 12 h half-period, which is a sustained regime change the controller is
        # right to act on -- the name promises a timescale the parameter does not deliver.
        fast_flap_h = max(1e-4, 2 * base.t_grow() / 3600.0)

        for motion in TR.MOTIONS:
            ph = fast_flap_h if motion == "flapping" else 24.0
            c, r = run(model, motion, lam, n, a.dt, table, base, short, period_h=ph)
            if r is None:
                continue
            r["lambda"] = round(lam, 1)
            r["spike_ticks"] = short if motion == "brief spike" else None
            r["period_h"] = round(ph, 4)
            rows.append(r)

            e = EXPECT.get(motion, {})
            msg = ""
            if "max_reconfig" in e and r["reconfigs"] > e["max_reconfig"]:
                msg = f"expected <= {e['max_reconfig']}, got {r['reconfigs']}"
            if ("min_reconfig" in e and model != "Llama2-70B"
                    and r["reconfigs"] < e["min_reconfig"]):
                msg = f"expected >= {e['min_reconfig']}, got {r['reconfigs']}"
            r["expectation"] = msg or "pass"
            if msg:
                fails.append(f"{model:<11} {motion:<13} {msg}   ({e.get('why','')})")

        # ---- controls: the same motions at a timescale the controller SHOULD act on ----
        for label, kwargs, spike in [
                ("brief spike (LONG >T_grow)", dict(period_h=24.0), long),
                ("flapping (SLOW 12h half-period)", dict(period_h=24.0), short)]:
            mot = "brief spike" if "spike" in label else "flapping"
            c, r = run(model, mot, lam, n, a.dt, table, base, spike, **kwargs)
            if r is None:
                continue
            r["motion"] = label
            r["lambda"] = round(lam, 1)
            r["spike_ticks"] = spike
            r["period_h"] = kwargs["period_h"]
            ok = r["reconfigs"] >= 1 or model == "Llama2-70B"
            r["expectation"] = "pass" if ok else "expected >= 1 reconfig"
            if not ok:
                fails.append(f"{model:<11} {label:<32} expected >=1 reconfig, got 0")
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    cols = ["Model", "motion", "reconfigs", "grow_B", "grow_C", "shrinks", "breakeven",
            "spill_frac", "unmet_frac", "eff_TTFT_s", "tokens_lost", "distinct_splits",
            "N_L_range", "expectation"]
    print(f"{'='*128}\n24 h sweep — every alpha motion, dt={a.dt:.0f}s, {n:,} ticks\n{'='*128}")
    for model, g in df.groupby("Model", sort=False):
        print(f"\n{model}   (lambda = {g['lambda'].iloc[0]:,.0f} tok/s)")
        print(g[cols[1:]].to_string(index=False))
    print(f"\nwrote {a.out}")
    if fails:
        print(f"\n{'!'*128}\nEXPECTATION FAILURES\n{'!'*128}")
        for f in fails:
            print("  " + f)
    else:
        print("\nAll motion expectations passed.")