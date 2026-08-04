#!/usr/bin/env python3
"""dse_candidates_by_rate_alpha.py — EVERY split x EVERY (lambda, alpha), with all gate flags.

`best_split_by_rate_alpha.py` writes only the WINNER per (lambda, alpha). That is a decision log,
not a design-space exploration. This dumps the whole space: for each model, every split
(N_L x every j x m factorization x N_T) scored at every (lambda, alpha) grid point, with each gate
flagged independently so any of them can be relaxed in analysis.

NOTHING is filtered. Losing, saturated, KV-infeasible and overloaded splits all appear, each with
the flags that explain why it was not chosen.

Flags (all independent):
    meets_load   both pools absorb their class share at this (lambda, alpha)
    meets_kv     kv_slack >= kv_margin on BOTH pools
    unsaturated  TP_per_instance <= TP_sat
    in_eps_band  TTFT_L <= (1+epsilon) * best TTFT_L among load+kv+unsaturated candidates
    selectable   all four
    is_best      the split the optimiser actually picks (max lambda_max, then min Pool T latency)

Output: results/dda32/dse_candidates_by_rate_alpha.csv
Usage:  python3 dse_candidates_by_rate_alpha.py [--lambda-points 20] [--alpha-step 0.05]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "explorer")))
import dse_metrics as M   # noqa: E402

KEEP = ["Model", "N_L", "N_T", "n_instances", "TP_per_instance",
        "Tput_L", "Tput_T", "system_tput", "TTFT_L_s", "req_lat_T_packed_s",
        "req_lat_T_cent_s", "packing_gain_pct", "cpb_ceil", "cpb_floor", "idle_T",
        "kv_slack_L", "kv_slack_T", "C_max_4096_T", "C_need_T", "saturated"]


def build(table, alphas, lambdas_per_model, epsilon, kv_margin):
    out = []
    for model in M.MODELS:
        sub = table[(table.Model == model) & (table.N_L > 0)
                    & table.Tput_L.notna() & table.Tput_T.notna()].copy()
        if sub.empty:
            continue
        mean_S = M.mean_seqlen(M.load_pool_T(model, int(sub.N_T.iloc[0]))["seqlens"])
        for a_req in alphas:
            a_tok = M.alpha_token_share(a_req, mean_S, mean_S)
            for lam in lambdas_per_model[model]:
                best, ann = M.best_split(sub, lam, a_tok, epsilon, kv_margin)
                key = None if best is None else (best.N_L, best.n_instances, best.TP_per_instance)

                r = ann[KEEP].copy()
                r["alpha_req"] = round(a_req, 3)
                r["alpha_tok"] = round(a_tok, 4)
                r["lambda_tok_s"] = round(lam, 1)
                r["load_L"] = round(a_tok * lam, 1)
                r["load_T"] = round((1 - a_tok) * lam, 1)
                r["lambda_max"] = ann.lambda_max.round(1)
                r["headroom_ratio"] = (ann.lambda_max / lam).round(3)
                r["ttft_bar_s"] = ann.ttft_bar_s.round(4)
                for f in ["meets_load", "meets_kv", "unsaturated", "in_eps_band", "selectable"]:
                    r[f] = ann[f].values
                r["is_best"] = [key is not None and (nl, ni, tp) == key
                                for nl, ni, tp in zip(ann.N_L, ann.n_instances,
                                                      ann.TP_per_instance)]
                out.append(r)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-points", type=int, default=20)
    ap.add_argument("--alpha-step", type=float, default=0.05)
    ap.add_argument("--epsilon", type=float, default=M.EPSILON_DEFAULT)
    ap.add_argument("--kv-margin", type=float, default=M.KV_MARGIN_DEFAULT)
    ap.add_argument("--table", default=os.path.join(M.DDA, "dse_full_table.csv"))
    ap.add_argument("--out", default=os.path.join(M.DDA, "dse_candidates_by_rate_alpha.csv"))
    a = ap.parse_args()

    table = pd.read_csv(a.table)
    alphas = np.round(np.arange(a.alpha_step, 1.0, a.alpha_step), 3)

    lams = {}
    for model in M.MODELS:
        s = table[(table.Model == model) & (table.N_L > 0) & table.valid]
        if not s.empty:
            hi = 1.05 * s.system_tput.max()
            lams[model] = np.linspace(hi / a.lambda_points, hi, a.lambda_points)

    df = build(table, alphas, lams, a.epsilon, a.kv_margin)
    df.to_csv(a.out, index=False)
    sz = os.path.getsize(a.out) / 1024**2

    print(f"wrote {a.out}  ({len(df):,} rows, {sz:.1f} MB)")
    print(f"  grid: {len(alphas)} alphas (step {a.alpha_step}) x {a.lambda_points} lambdas "
          f"x every split | epsilon={a.epsilon} kv_margin={a.kv_margin}\n")
    for model, g in df.groupby("Model", sort=False):
        print(f"  {model:<11} {len(g):>8,} rows | splits "
              f"{g.groupby(['N_L','n_instances','TP_per_instance']).ngroups:3d}"
              f" | selectable {int(g.selectable.sum()):>7,} | best {int(g.is_best.sum()):>5,}")
        print(f"{'':<13}fails: load {int((~g.meets_load).sum()):>7,} | kv {int((~g.meets_kv).sum()):>6,}"
              f" | saturated {int((~g.unsaturated).sum()):>7,} | eps-band {int((~g.in_eps_band).sum()):>7,}")