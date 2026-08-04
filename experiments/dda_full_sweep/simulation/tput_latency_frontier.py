#!/usr/bin/env python3
"""tput_latency_frontier.py — at equal latency, how much TOTAL fabric throughput?

    x = latency experienced by the latency-tagged class
    y = TOTAL throughput served, Pool L + Pool T

Both deployments are scored on the same quantity: what all 32 devices deliver while holding that
latency for the latency class.

Three modelling decisions, each fixing an earlier mistake:

1. THE WHOLE STATIC CURVE, not one config. Earlier work compared DDA against the single
   max-throughput static config (7B PP32/TP1, TTFT 0.718 s) -- the one tuned for the OPPOSITE
   objective -- and concluded DDA's latency was unimpressive. 7B at 32 devices has six valid
   static configs from 0.398 s @ 1211 tok/s to 0.718 s @ 11505 tok/s; an operator picks a point
   on that curve, so the comparison must span all of it.

2. CAPACITY, not utilisation. At moderate lambda every deployment serves all the demand there is,
   so plotting SERVED throughput returns lambda for everyone and all differences vanish -- that
   measures the workload, not the deployment.

3. SPILL, not binary feasibility. An undersized Pool L is not a failure: the excess SPILLS to
   Pool T and is served there, just slowly. So the latency-class coordinate is the EFFECTIVE
   latency,

       eff = (served_by_L*TTFT_L + spilled*req_lat_T) / (served_by_L + spilled)

   computed by the same dda_controller.route() the controller uses. Only demand neither pool can
   take actually QUEUES. This is also what makes the frontier genuinely depend on (alpha, lambda):
   points MOVE as spill grows, instead of a feasible/infeasible flag flipping.

A static config has one pool, so it has nowhere to spill: its latency is the same for every
request and anything past capacity simply queues.

Outputs (results/dda32/):
  frontier_static.csv    every valid static config, both device budgets
  frontier_dda.csv       every ladder rung
  frontier_matched.csv   at each static latency bar, the best DDA split meeting it

Usage: python3 tput_latency_frontier.py [--alpha 0.3] [--lam-frac 0.45]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "explorer")))
import dse_metrics as M          # noqa: E402
import dda_controller as DC      # noqa: E402

NATIVE = {"Llama2-7B": 8, "Llama2-13B": 20, "Llama2-70B": 32}


def static_points(model, ndev, lam, alpha, kind):
    """Every valid static config as one point. ONE pool serves both classes at ONE latency."""
    out = []
    for c in M.static_configs(model, ndev):
        served = min(lam, c["tput"])
        out.append(dict(
            Model=model, devices=ndev, cfg=f"PP{c['pp']}/TP{c['tp']}",
            pp=c["pp"], tp=c["tp"],
            lat_TTFT_s=round(c["ttft_s"], 4),
            eff_lat_s=round(c["ttft_s"], 4),      # one pool -> nothing to spill to
            batch_lat_s=round(c["req_lat_s"], 2),
            total_tput=round(c["tput"], 1),
            served_total=round(served, 1),
            queued=round(max(0.0, lam - c["tput"]), 1),
            spill_frac=0.0,
            unmet_frac=round(max(0.0, lam - c["tput"]) / max(1e-9, lam), 4),
            # OVERLOAD, not "infeasible": with one pool there is nowhere to spill, so anything
            # past capacity queues and the backlog grows without bound
            feasible=bool(c["tput"] >= lam),
            # kind is passed in, never inferred: for 70B the native budget IS 32 devices, so
            # inferring would label both series "native" and leave static32 empty
            kind=kind))
    return sorted(out, key=lambda r: r["eff_lat_s"])


def dda_points(model, table, lam, alpha):
    """Every ladder rung, scored through the controller's own routing model (spill included)."""
    out = []
    lat_load = alpha * lam
    for r in DC.ladder(model, table):
        rt = DC.route(r, lam, alpha)
        total_cap = r.Tput_L + r.Tput_T
        out.append(dict(
            Model=model, devices=32, cfg=f"{r.j}xTP{r.m}/N_T{r.N_T}",
            N_L=r.N_L, N_T=r.N_T, k=r.j, TP=r.m,
            lat_TTFT_s=round(r.TTFT_L_s, 4),       # Pool L alone -- the brochure number
            eff_lat_s=round(rt["eff_ttft_s"], 4),  # what the class actually sees, spill included
            batch_lat_s=round(r.req_lat_T_s, 2),
            batch_tput=round(r.Tput_T, 1),
            total_tput=round(total_cap, 1),
            served_total=round(min(lam, total_cap), 1),
            queued=round(max(0.0, lam - total_cap), 1),
            spill_frac=round(rt["spilled"] / max(1e-9, lat_load), 4),
            unmet_frac=round(rt["unmet"] / max(1e-9, lat_load), 4),
            # the only hard failure: neither pool can take it, so it queues and the backlog grows
            feasible=bool(total_cap >= lam and rt["unmet"] <= 1e-9),
            kind="dda"))
    return sorted(out, key=lambda r: r["eff_lat_s"])


def matched(stat, dda, feasible_only=True):
    """At each static config's EFFECTIVE latency, the best DDA split meeting the same bar.

    Matched on effective latency, so a DDA split only counts if what the latency class actually
    experiences -- spill included -- clears the static config's bar.
    """
    S = [s for s in stat if s["feasible"]] if feasible_only else stat
    D = [d for d in dda if d["feasible"]] if feasible_only else dda
    rows = []
    for s in S:
        ok = [d for d in D if d["eff_lat_s"] <= s["eff_lat_s"] + 1e-9]
        if not ok:
            continue
        b = max(ok, key=lambda d: d["total_tput"])
        rows.append(dict(
            Model=s["Model"], ttft_bar_s=s["eff_lat_s"],
            static_cfg=s["cfg"], static_total_tput=s["total_tput"],
            static_batch_lat_s=s["batch_lat_s"],
            dda_cfg=b["cfg"], dda_total_tput=b["total_tput"],
            dda_batch_lat_s=b["batch_lat_s"], dda_spill_frac=b["spill_frac"],
            tput_ratio=round(b["total_tput"] / max(1e-9, s["total_tput"]), 2),
            batch_lat_ratio=round(b["batch_lat_s"] / max(1e-9, s["batch_lat_s"]), 2)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--lam-frac", type=float, default=0.45)
    ap.add_argument("--out", default=M.DDA)
    a = ap.parse_args()

    table = pd.read_csv(os.path.join(M.DDA, "dse_full_table.csv"))
    S, D, X = [], [], []

    for model in M.MODELS:
        base = DC.make_controller(model, table)
        if base is None:
            continue
        cap = max(r.Tput_L + r.Tput_T for r in base.rungs)
        lam = a.lam_frac * cap

        s32 = static_points(model, 32, lam, a.alpha, "static32")
        snat = (static_points(model, NATIVE[model], lam, a.alpha, "native")
                if NATIVE[model] != 32 else [])          # 70B: native == static32
        dd = dda_points(model, table, lam, a.alpha)
        for r in s32 + snat + dd:
            r["lambda"] = round(lam, 1)
            r["alpha"] = a.alpha
        S += s32 + snat
        D += dd
        X += matched(s32, dd)

    ps, pdd, px = pd.DataFrame(S), pd.DataFrame(D), pd.DataFrame(X)
    ps.to_csv(os.path.join(a.out, "frontier_static.csv"), index=False)
    pdd.to_csv(os.path.join(a.out, "frontier_dda.csv"), index=False)
    px.to_csv(os.path.join(a.out, "frontier_matched.csv"), index=False)

    print(f"{'='*116}\nAt equal latency-class latency, how much TOTAL fabric throughput?  "
          f"(α={a.alpha}, λ={a.lam_frac:.0%} of capacity)\n{'='*116}")
    for model, g in px.groupby("Model", sort=False):
        print(f"\n{model}")
        print(g[["ttft_bar_s", "static_cfg", "static_total_tput", "dda_cfg", "dda_total_tput",
                 "tput_ratio", "dda_spill_frac", "static_batch_lat_s", "dda_batch_lat_s",
                 "batch_lat_ratio"]].to_string(index=False))
        w = g[g.tput_ratio > 1.0]
        print(f"   DDA delivers more TOTAL throughput at {len(w)}/{len(g)} latency bars; "
              f"best {g.tput_ratio.max():.2f}×")
    for model, g in pdd.groupby("Model", sort=False):
        print(f"   {model:<11} {int(g.feasible.sum())}/{len(g)} rungs feasible · "
              f"spill {g.spill_frac.min():.0%}–{g.spill_frac.max():.0%} · "
              f"eff latency {g.eff_lat_s.min():.2f}–{g.eff_lat_s.max():.2f}s")
    print(f"\nwrote frontier_static.csv, frontier_dda.csv, frontier_matched.csv to {a.out}")