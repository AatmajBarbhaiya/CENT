#!/usr/bin/env python3
"""baseline_comparison.py — DDA vs static deployments under the SAME dynamic workload.

Why this exists: `figure_pareto_dda_vs_baseline.py` compares DDA against static configs at a
SINGLE operating point. That is not a fair test. A static config is not choosing an operating
point -- it is stuck with one, and must serve BOTH request classes from one pool at every alpha
it ever sees. Comparing a static point against a DDA point hides the thing DDA is actually for.

So: replay the same alpha(t) through every deployment and score them on identical traffic.

The four deployments:
  1. native        the paper's device budget (7B=8, 13B=20, 70B=32), ONE pool, never changes
  2. cent32_best   best 32-device static config, ONE pool, never changes
  3. dda_static    the DDA split chosen once at mean alpha and never reconfigured
  4. dda_dynamic   the full controller: two pools + spillover + hysteresis

1 and 2 are single-pool: every request -- latency-tagged or not -- gets the same TTFT and the same
throughput budget, because there is no second pool to route to. That IS the handicap, and it is
the honest one: it is what "just extend every model to 32 devices" actually buys you.

3 isolates the value of the SPLIT from the value of RECONFIGURING. If 3 ~ 4, the controller is not
earning its complexity and the paper should say so.

Scoring is class-aware for every deployment: latency-tagged requests are charged the deployment's
latency-class latency, batch requests the batch latency, and anything over capacity is unmet.

Output: results/dda32/baseline_comparison.csv  (+ _by_motion.csv)
Usage:  python3 baseline_comparison.py [--dt 30] [--hours 24] [--max-spill-frac 0.10]

NOTE `--max-spill-frac` is a CONTROLLER POLICY knob, not a metric. It gates only the
`absorbed` test in Controller.step(), so changing it re-runs this script,
controller_eval.py and motion_sweep_24h.py -- and NOTHING else. The 118 pool
simulations, dse_full_table.csv and the frontier are all untouched by it.
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

NATIVE = {"Llama2-7B": 8, "Llama2-13B": 20, "Llama2-70B": 32}


def score_single_pool(cfg, al, lam, dt):
    """One pool serving both classes. `cfg` = dict(tput, ttft_s, req_lat_s, label).

    A single-pool deployment cannot specialise: the latency class gets the SAME latency as the
    batch class, because there is no faster pool to route it to. Capacity is shared, so an alpha
    excursion does not change what anything costs -- only whether it is served at all.
    """
    al = np.asarray(al, dtype=float)
    load = np.full(al.shape, float(lam))          # per-tick, not a scalar: both classes share one budget
    served = np.minimum(load, cfg["tput"])
    lat_load = al * lam                           # latency-tagged share of demand
    lat_served = np.minimum(lat_load, cfg["tput"] * al)   # proportional share of one capacity
    lat_unmet = lat_load - lat_served
    return dict(
        deployment=cfg["label"], devices=cfg["devices"],
        eff_TTFT_s=cfg["ttft_s"],                 # constant: one pool, one latency for everyone
        batch_lat_s=cfg["req_lat_s"],
        served_tok=float((served * dt).sum()),
        unmet_frac=float((lat_unmet * dt).sum() / max(1e-9, (lat_load * dt).sum())),
        spill_frac=np.nan,                        # meaningless with one pool
        reconfigs=0, tokens_lost=0.0,
        util=float(served.mean() / max(1e-9, cfg["tput"])),
    )


def score_dda(model, al, lam, dt, table, policy, **kw):
    c, log = DC.run_trace(model, al, np.full(len(al), lam), dt=dt, table=table,
                          policy=policy, **kw)
    if log.empty:
        return None
    lat_tok = (log.load_L.fillna(0) * dt).sum()
    spill = (log.spilled.fillna(0) * dt).sum()
    unmet = (log.unmet.fillna(0) * dt).sum()
    w = (log.served_by_L.fillna(0) + log.spilled.fillna(0)) * dt
    eff = (log.eff_ttft_s.fillna(0) * w).sum() / w.sum() if w.sum() > 0 else np.nan
    return dict(
        deployment="dda_dynamic" if policy == "hysteresis" else "dda_static", devices=32,
        eff_TTFT_s=round(float(eff), 3) if eff == eff else None,
        batch_lat_s=round(float(log.req_lat_T_s.mean()), 2),
        served_tok=float((log.served_total.fillna(0) * dt).sum()),
        unmet_frac=round(float(unmet / lat_tok), 4) if lat_tok else 0.0,
        spill_frac=round(float(spill / lat_tok), 4) if lat_tok else 0.0,
        reconfigs=int(len(c.events)), tokens_lost=round(float(log.tokens_lost.sum()), 0),
        util=float((log.served_total.fillna(0) * dt).sum()
                   / max(1e-9, ((log.cap_L.fillna(0) + log.cap_T.fillna(0)) * dt).sum())),
    )


def best_static(model, ndev, prompt_fraction=M.DEFAULT_PROMPT_FRACTION):
    """Pick the static config a sensible operator would deploy: max throughput among valid splits.

    Valid-split filtered (pp | num_layers) so the uniform-stage artifacts never win.
    """
    cfgs = M.static_configs(model, ndev, prompt_fraction)
    if not cfgs:
        return None
    b = max(cfgs, key=lambda c: c["tput"])
    return dict(tput=b["tput"], ttft_s=round(b["ttft_s"], 3), req_lat_s=round(b["req_lat_s"], 2),
                devices=ndev, label=f"static_{ndev}dev_PP{b['pp']}/TP{b['tp']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--out", default=M.DDA)
    ap.add_argument("--max-spill-frac", type=float, default=DC.Controller.max_spill_frac,
                    help="fraction of the latency class the controller tolerates on Pool T "
                         "before starting its grow timer. Chosen, not measured -- sweep it.")
    ap.add_argument("--grow-factor", type=float, default=DC.Controller.grow_factor)
    a = ap.parse_args()

    table = pd.read_csv(os.path.join(M.DDA, "dse_full_table.csv"))
    n = int(a.hours * 3600 / a.dt)
    rows = []

    for model in M.MODELS:
        base = DC.make_controller(model, table)
        if base is None:
            continue
        cap = max(r.Tput_L + r.Tput_T for r in base.rungs)
        max_L = max(r.Tput_L for r in base.rungs)
        lam = min(0.45 * cap, 0.85 * max_L / 0.55)
        spike = max(1, int(0.5 * base.t_grow() / a.dt))
        flap_h = max(1e-4, 2 * base.t_grow() / 3600.0)

        nat = best_static(model, NATIVE[model])
        c32 = best_static(model, 32)

        for motion in TR.MOTIONS:
            ph = flap_h if motion == "flapping" else 24.0
            al = TR.alpha_motion(motion, n, mean=0.30, amp=0.25, period_h=ph, hours=a.hours,
                                 rng=np.random.default_rng(0), spike_ticks=spike)
            for cfg in (nat, c32):
                if cfg:
                    r = score_single_pool(cfg, al, lam, a.dt)
                    r.update(Model=model, motion=motion, offered_tok=float(lam * n * a.dt))
                    rows.append(r)
            for pol in ("static", "hysteresis"):
                r = score_dda(model, al, lam, a.dt, table, pol,
                              max_spill_frac=a.max_spill_frac, grow_factor=a.grow_factor)
                if r:
                    r.update(Model=model, motion=motion, offered_tok=float(lam * n * a.dt))
                    rows.append(r)

    df = pd.DataFrame(rows)
    df["served_frac"] = df.served_tok / df.offered_tok
    df["max_spill_frac"] = a.max_spill_frac      # recorded so a sweep's outputs stay traceable
    df.to_csv(os.path.join(a.out, "baseline_comparison_by_motion.csv"), index=False)

    agg = (df.groupby(["Model", "deployment"], sort=False)
             .agg(devices=("devices", "first"), eff_TTFT_s=("eff_TTFT_s", "mean"),
                  batch_lat_s=("batch_lat_s", "mean"), served_frac=("served_frac", "mean"),
                  unmet_frac=("unmet_frac", "mean"), spill_frac=("spill_frac", "mean"),
                  reconfigs=("reconfigs", "sum"), tokens_lost=("tokens_lost", "sum"),
                  util=("util", "mean"))
             .round(4).reset_index())
    agg.to_csv(os.path.join(a.out, "baseline_comparison.csv"), index=False)

    print(f"{'='*118}\nSAME 24 h dynamic workload through every deployment "
          f"(mean over {len(TR.MOTIONS)} α motions)\n{'='*118}")
    for model, g in agg.groupby("Model", sort=False):
        lam_i = df[df.Model == model].offered_tok.iloc[0] / (n * a.dt)
        print(f"\n{model}   λ = {lam_i:,.0f} tok/s")
        print(g[["deployment", "devices", "eff_TTFT_s", "batch_lat_s", "served_frac",
                 "unmet_frac", "spill_frac", "reconfigs", "util"]].to_string(index=False))
        d = g[g.deployment == "dda_dynamic"]
        for _, s in g[g.deployment.str.startswith("static")].iterrows():
            if len(d):
                dd = d.iloc[0]
                print(f"   vs {s.deployment:<28} TTFT {s.eff_TTFT_s/dd.eff_TTFT_s:5.2f}× worse · "
                      f"batch {s.batch_lat_s/dd.batch_lat_s:5.2f}× · "
                      f"served {dd.served_frac/max(1e-9, s.served_frac):5.2f}× more")
    # ---- lambda sweep: the comparison only means something across the load range ----------
    # No winner column BY DESIGN. Declaring one needs a threshold on how much unmet difference
    # counts, and any such constant has to be defended; worse, a strict comparison lets a
    # two-tick drain transient (~0.02% unmet) override a 17% TTFT win. Both metrics are printed
    # side by side so the crossover is visible without being asserted.
    sweep = []
    for model in M.MODELS:
        base = DC.make_controller(model, table)
        if base is None:
            continue
        cap = max(r.Tput_L + r.Tput_T for r in base.rungs)
        c32 = best_static(model, 32)
        if not c32:
            continue
        al = TR.alpha_motion("diurnal", n, mean=0.30, amp=0.25, period_h=24.0, hours=a.hours,
                             rng=np.random.default_rng(0))
        for f in (0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95):
            lam = f * cap
            s = score_single_pool(c32, al, lam, a.dt)
            d = score_dda(model, al, lam, a.dt, table, "hysteresis",
                          max_spill_frac=a.max_spill_frac, grow_factor=a.grow_factor)
            if d is None:
                continue
            sweep.append(dict(Model=model, max_spill_frac=a.max_spill_frac,
                              lam_frac=f, lam=round(lam, 0),
                              static_cfg=c32["label"],
                              static_TTFT_s=s["eff_TTFT_s"], static_unmet=round(s["unmet_frac"], 4),
                              static_batch_s=s["req_lat_s"] if "req_lat_s" in s else c32["req_lat_s"],
                              dda_TTFT_s=d["eff_TTFT_s"], dda_unmet=d["unmet_frac"],
                              dda_spill=d["spill_frac"], dda_batch_s=d["batch_lat_s"],
                              dda_reconfigs=d["reconfigs"],
                              ttft_ratio=round(s["eff_TTFT_s"] / d["eff_TTFT_s"], 3)
                              if d["eff_TTFT_s"] else None))
    sw = pd.DataFrame(sweep)
    sw.to_csv(os.path.join(a.out, "baseline_lambda_sweep.csv"), index=False)

    print(f"\n{'='*118}\nλ SWEEP — same diurnal α, one scaled-up pool vs DDA. No winner column: "
          f"read both columns.\n{'='*118}")
    for model, g in sw.groupby("Model", sort=False):
        print(f"\n{model}   static32 = {g.static_cfg.iloc[0]}")
        print(g[["lam_frac", "lam", "static_TTFT_s", "static_unmet", "dda_TTFT_s",
                 "dda_unmet", "dda_spill", "dda_reconfigs", "ttft_ratio"]]
              .to_string(index=False))

    print(f"\nwrote baseline_comparison.csv, _by_motion.csv, baseline_lambda_sweep.csv to {a.out}")