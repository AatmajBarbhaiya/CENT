#!/usr/bin/env python3
"""interactive_load_sweep.py — throughput-optimal DDA split N_L*(alpha) vs interactive load.

NOTE: superseded by realistic_dda_sweep.py (which adds the DP-replica Pool L and packed
Pool T models). Kept as the simpler single-instance view over combined_metrics32.csv.

Given a single request stream where a fraction alpha is latency-critical (served by
Pool L, TP=N_L/PP=1) and 1-alpha is batch (Pool T, PP=layers/TP=1), the max sustainable
arrival rate for a split is the two-class load-balancing bound (parent DDA CLAUDE.md):

    lambda_max(N_L, alpha) = min( Tput_L(N_L) / alpha , Tput_T(N_T) / (1-alpha) )   N_T = 32 - N_L

For each model and each alpha we pick the split that maximises it. NO SLO gate — TTFT is
reported (TTFT_short_s) so the operator applies their own latency bar. The result is the
operating map: which N_L is throughput-optimal at each load and the alpha breakpoints.

Reads  ../results/dda32/combined_metrics32.csv   (Tput_L, Tput_T, TTFT_short per split)
Writes ../results/dda32/interactive_load_map.csv (per model x alpha)
       ../results/dda32/interactive_regions.csv  (alpha intervals per winning split)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "../results/dda32/combined_metrics32.csv")
MAP_OUT = os.path.join(HERE, "../results/dda32/interactive_load_map.csv")
REG_OUT = os.path.join(HERE, "../results/dda32/interactive_regions.csv")

ALPHAS = np.round(np.arange(0.02, 0.99, 0.01), 2)

df = pd.read_csv(SRC)
map_rows, reg_rows = [], []

for model in ["Llama2-7B", "Llama2-13B", "Llama2-70B"]:
    sub = df[df.Model == model].copy()
    if sub.empty:
        continue

    # No SLO gate — every capacity-feasible split is a candidate; TTFT is reported so the
    # operator applies their own latency bar.
    elig = sub.reset_index(drop=True)

    prev_NL, region_start = None, None
    for a in ALPHAS:
        lam = np.minimum(elig.Tput_L.values / a, elig.Tput_T.values / (1 - a))
        i = int(np.argmax(lam))
        r = elig.iloc[i]
        lam_star = lam[i]
        L_bound = (r.Tput_L / a) <= (r.Tput_T / (1 - a))   # Pool L saturates first
        map_rows.append({
            "Model": model, "alpha": a, "N_L": int(r.N_L), "N_T": int(r.N_T),
            "lambda_max": round(lam_star, 1),
            "Tput_L": round(r.Tput_L, 1), "Tput_T": round(r.Tput_T, 1),
            "TTFT_short_s": round(r.TTFT_short, 3),
            "bottleneck": "PoolL" if L_bound else "PoolT",
        })
        # track contiguous regions of constant N_L*
        if int(r.N_L) != prev_NL:
            if prev_NL is not None:
                reg_rows.append(dict(Model=model, N_L=prev_NL, N_T=32 - prev_NL,
                                     alpha_lo=region_start, alpha_hi=round(a - 0.01, 2)))
            prev_NL, region_start = int(r.N_L), a
    reg_rows.append(dict(Model=model, N_L=prev_NL, N_T=32 - prev_NL,
                         alpha_lo=region_start, alpha_hi=ALPHAS[-1]))

pd.DataFrame(map_rows).to_csv(MAP_OUT, index=False)
reg = pd.DataFrame(reg_rows)
reg.to_csv(REG_OUT, index=False)

# ---- report ----
print(f"{'='*100}\nOptimal DDA split N_L*(alpha) under interactive load  (N_total=32)\n{'='*100}")
mp = pd.DataFrame(map_rows)
for model in ["Llama2-7B", "Llama2-13B", "Llama2-70B"]:
    r = reg[reg.Model == model]
    if r.empty:
        continue
    print(f"\n### {model}")
    print(f"{'alpha range':>16}  {'N_L/N_T':>8}  {'lambda_max range (tok/s)':>26}  bottleneck")
    for _, x in r.iterrows():
        seg = mp[(mp.Model == model) & (mp.alpha >= x.alpha_lo) & (mp.alpha <= x.alpha_hi)]
        lo, hi = seg.lambda_max.min(), seg.lambda_max.max()
        bn = "/".join(seg.bottleneck.unique())
        print(f"  {x.alpha_lo:.2f}–{x.alpha_hi:.2f}      {int(x.N_L):>3}/{int(x.N_T):<3}   "
              f"{hi:>10.0f} → {lo:<10.0f}   {bn}")

print(f"\nSaved: {MAP_OUT}\n       {REG_OUT}")
