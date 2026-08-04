#!/usr/bin/env python3
"""best_split_by_rate_alpha.py — optimal split over the full (lambda, alpha) grid.

lambda = total offered TOKEN rate (tokens/s), prompt and generated together -- CENT prices a
prompt token exactly like a generated one, so token demand is the right unit. Request rate is
lambda / E[S] under the mixture.

alpha_req = fraction of REQUESTS pre-tagged latency-class. Converted internally to alpha_tok
(token-demand share) because feasibility is a token-flow constraint; with the default identical
mixtures the two are equal, but the conversion keeps the model correct for asymmetric ones.

A split serves (lambda, alpha) iff BOTH pools can absorb their class's share AND the KV can hold
the concurrency the throughput assumes. Among feasible splits: MAX HEADROOM (largest lambda_max),
because a reshard costs ~21 s of drain, so distance from the capacity cliff is worth more than a
few ms of TTFT. Infeasible grid points are KEPT and flagged, reporting the max-lambda_max split
as the fallback.

All formulas come from explorer/dse_metrics.py.

Output: results/dda32/best_split_by_rate_alpha32.csv
Usage:  python3 best_split_by_rate_alpha.py [--prompt-fraction 0.125] [--lambda-points 80]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "explorer")))
import dse_metrics as M   # noqa: E402


def build(table, alphas, n_lambda, epsilon=M.EPSILON_DEFAULT, kv_margin=M.KV_MARGIN_DEFAULT):
    rows = []
    for model in M.MODELS:
        # every simulated split stays a candidate; the gates are FLAGS, not filters
        sub = table[(table.Model == model) & (table.N_L > 0)
                    & table.Tput_L.notna() & table.Tput_T.notna()].copy()
        if sub.empty:
            continue

        # lambda ceiling: no split can exceed its own combined pool throughput, and that is only
        # reachable at its balance point. 1.05x so the infeasible region is visible past the top.
        lam_hi = 1.05 * sub.system_tput.max()
        lambdas = np.linspace(lam_hi / n_lambda, lam_hi, n_lambda)

        # identical mixtures by default => alpha_tok == alpha_req; conversion kept for generality
        mean_S = M.mean_seqlen(M.load_pool_T(model, int(sub.N_T.iloc[0]))["seqlens"])

        # Reconfiguration is triggered by ALPHA drifting at a given offered rate, so the
        # previous choice must be tracked PER LAMBDA -- comparing across lambda instead would
        # price a load change as if it were a split change.
        prev_by_lambda = {}
        for a_req in alphas:
            a_tok = M.alpha_token_share(a_req, mean_S, mean_S)
            for lam in lambdas:
                best, ann = M.best_split(sub, lam, a_tok, epsilon, kv_margin)

                if best is None:
                    # nothing is selectable; report the max-headroom split as the fallback and
                    # say WHY nothing qualified, so the row is still informative
                    fb = ann.loc[ann.lambda_max.idxmax()]
                    rows.append(dict(
                        Model=model, alpha_req=round(a_req, 3), alpha_tok=round(a_tok, 4),
                        lambda_tok_s=round(lam, 1), selectable=False,
                        N_L=int(fb.N_L), N_T=int(fb.N_T),
                        n_instances=int(fb.n_instances), TP_per_instance=int(fb.TP_per_instance),
                        lambda_max=round(fb.lambda_max, 1),
                        headroom_ratio=round(fb.lambda_max / lam, 3),
                        Tput_L=fb.Tput_L, Tput_T=fb.Tput_T, system_tput=fb.system_tput,
                        TTFT_L_s=fb.TTFT_L_s, req_lat_T_packed_s=fb.req_lat_T_packed_s,
                        kv_slack_T=fb.kv_slack_T, kv_slack_L=fb.kv_slack_L,
                        bottleneck=M.lambda_max(fb.Tput_L, fb.Tput_T, a_tok)[1],
                        n_meets_load=int(ann.meets_load.sum()),
                        n_meets_kv=int(ann.meets_kv.sum()),
                        n_unsaturated=int(ann.unsaturated.sum()),
                        n_selectable=0, ttft_bar_s=round(float(ann.ttft_bar_s.iloc[0]), 3)
                        if pd.notna(ann.ttft_bar_s.iloc[0]) else None,
                        switch_from="", devices_draining=None, devices_kept=None,
                        T_drain_s=None, T_breakeven_s=None, free_move=None,
                        note="no split meets load+kv+unsaturated+eps at this rate"))
                    continue

                lmax, bott = M.lambda_max(best.Tput_L, best.Tput_T, a_tok)
                cur = (int(best.n_instances), int(best.TP_per_instance), int(best.N_T))

                # cooldown economics vs the previous alpha's choice at the same lambda
                sw = dict(draining=None, kept=None, T_drain_s=None,
                          T_breakeven_s=None, free_move=None)
                frm = ""
                prev = prev_by_lambda.get(round(lam, 1))
                if prev is not None and prev[0] != cur:
                    # Gain must be measured AT THE NEW ALPHA: the question is not "was the old
                    # split better under the old load" (it always was -- lambda_max falls as
                    # alpha rises) but "given alpha has moved, does switching beat standing pat".
                    lmax_stay, _ = M.lambda_max(prev[1], prev[2], a_tok)
                    sw = M.switch_economics(model, prev[0], cur, lmax_stay, lmax, prev[0][2])
                    frm = f"{prev[0][0]}xTP{prev[0][1]}/N_T{prev[0][2]}"

                rows.append(dict(
                    Model=model, alpha_req=round(a_req, 3), alpha_tok=round(a_tok, 4),
                    lambda_tok_s=round(lam, 1), selectable=True,
                    N_L=int(best.N_L), N_T=int(best.N_T),
                    n_instances=cur[0], TP_per_instance=cur[1],
                    lambda_max=round(lmax, 1), headroom_ratio=round(lmax / lam, 3),
                    Tput_L=best.Tput_L, Tput_T=best.Tput_T, system_tput=best.system_tput,
                    TTFT_L_s=best.TTFT_L_s, req_lat_T_packed_s=best.req_lat_T_packed_s,
                    kv_slack_T=best.kv_slack_T, kv_slack_L=best.kv_slack_L,
                    bottleneck=bott,
                    n_meets_load=int(ann.meets_load.sum()),
                    n_meets_kv=int(ann.meets_kv.sum()),
                    n_unsaturated=int(ann.unsaturated.sum()),
                    n_selectable=int(ann.selectable.sum()),
                    ttft_bar_s=round(float(ann.ttft_bar_s.iloc[0]), 3)
                    if pd.notna(ann.ttft_bar_s.iloc[0]) else None,
                    switch_from=frm,
                    devices_draining=sw["draining"], devices_kept=sw["kept"],
                    T_drain_s=round(sw["T_drain_s"], 2) if sw["T_drain_s"] is not None else None,
                    T_breakeven_s=round(sw["T_breakeven_s"], 2)
                    if sw["T_breakeven_s"] not in (None, float("inf")) else None,
                    free_move=sw["free_move"], note=""))
                prev_by_lambda[round(lam, 1)] = (cur, best.Tput_L, best.Tput_T)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-fraction", type=float, default=M.DEFAULT_PROMPT_FRACTION)
    ap.add_argument("--lambda-points", type=int, default=80)
    ap.add_argument("--epsilon", type=float, default=M.EPSILON_DEFAULT,
                    help="latency tolerance: admit splits within (1+eps) of best TTFT_L")
    ap.add_argument("--kv-margin", type=float, default=M.KV_MARGIN_DEFAULT,
                    help="required kv_slack (1.0 = physical wall; margin is reported either way)")
    ap.add_argument("--table", default=os.path.join(M.DDA, "dse_full_table.csv"))
    ap.add_argument("--out", default=os.path.join(M.DDA, "best_split_by_rate_alpha32.csv"))
    a = ap.parse_args()

    if not os.path.exists(a.table):
        sys.exit(f"{a.table} not found -- run dse_full_table.py first")
    table = pd.read_csv(a.table)

    alphas = np.round(np.arange(0.02, 0.99, 0.01), 2)
    df = build(table, alphas, a.lambda_points, a.epsilon, a.kv_margin)
    df.to_csv(a.out, index=False)

    print(f"wrote {a.out}  ({len(df)} rows: {len(alphas)} alphas x {a.lambda_points} lambdas "
          f"x models, epsilon={a.epsilon}, kv_margin={a.kv_margin})\n")
    for model in M.MODELS:
        s = df[df.Model == model]
        if s.empty:
            continue
        fe = s[s.selectable]
        print(f"{model:<11} selectable {len(fe):5d}/{len(s):5d}  "
              f"distinct splits chosen: {fe.groupby(['N_L','n_instances','TP_per_instance']).ngroups}")
        if len(fe):
            mid = fe[np.isclose(fe.lambda_tok_s, fe.lambda_tok_s.median(), rtol=0.15)]
            if len(mid):
                seq = mid.sort_values("alpha_req").groupby(
                    ["N_L", "n_instances", "TP_per_instance"], sort=False).alpha_req.agg(["min", "max"])
                print(f"{'':<11}   at lambda~{mid.lambda_tok_s.median():.0f} tok/s, split vs alpha:")
                for (nl, j, m), r in seq.iterrows():
                    print(f"{'':<11}     alpha {r['min']:.2f}-{r['max']:.2f} -> N_L={nl}"
                          f"({j}xTP{m})/N_T={32-nl}")
            free = fe[fe.free_move == True]   # noqa: E712
            paid = fe[fe.free_move == False]  # noqa: E712
            if len(free) or len(paid):
                print(f"{'':<11}   split changes: {len(free)} zero-drain, {len(paid)} paid"
                      + (f" (median drain {paid.devices_draining.median():.0f} dev, "
                         f"breakeven {paid.T_breakeven_s.median():.0f}s)" if len(paid) else ""))