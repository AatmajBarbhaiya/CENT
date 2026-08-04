#!/usr/bin/env python3
"""figure_split_vs_alpha.py — which split the optimiser picks at each interactive load alpha.

REBASED 2026-07-29 onto results/dda32/best_split_by_rate_alpha32.csv (full DSE, alpha-weighted
normalised-latency objective). The previous version read the superseded best_split_by_alpha32.csv
produced by realistic_dda_sweep.py, which carried the packing-inflated Pool T throughput.

Top row : N_L (Pool L) and N_T (Pool T) vs alpha -- the "which split at which load" map.
Bottom  : the Pool L factorization actually chosen (instances k x TP), a second degree of freedom
          that N_L alone hides -- at 7B alpha 0.15->0.16 the split stays at N_L=4 but moves
          1xTP4 -> 2xTP2, buying 67% more Pool L throughput for 23% worse TTFT without taking a
          single device from Pool T.

Output: figures/figure_split_vs_alpha.pdf ; source data figure_source_data/figure_split_vs_alpha.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "explorer")))
import dse_metrics as M   # noqa: E402

FIG = os.path.join(ROOT, "figures")
SRC = os.path.join(ROOT, "figure_source_data")
for d in (FIG, SRC):
    os.makedirs(d, exist_ok=True)

MODELS = ["Llama2-7B", "Llama2-13B", "Llama2-70B"]
C = {"N_L": "#2a78d6", "N_T": "#eb6834", "inst": "#1baf7a", "tp": "#8a5cd6"}
INK, MUTED = "#1f2328", "#6b7280"

b = pd.read_csv(os.path.join(M.DDA, "best_split_by_rate_alpha32.csv"))

keep = []
for model in MODELS:
    g = b[(b.Model == model) & b.selectable]
    if g.empty:
        continue
    # Show the HIGHEST load that still exercises most of the alpha range. A median lambda leaves
    # 70B with 2 selectable alphas (empty panel); simply maximising the count picks the LOWEST
    # lambda, i.e. a trivially loaded system. Take the busiest lambda retaining >=60% of alphas.
    cnt = g.groupby("lambda_tok_s").size()
    ok = cnt[cnt >= 0.6 * b.alpha_req.nunique()]
    lam = ok.index.max() if len(ok) else cnt.idxmax()
    s = g[g.lambda_tok_s == lam].sort_values("alpha_req").copy()
    s["lambda_shown"] = lam
    keep.append(s)
src = pd.concat(keep, ignore_index=True)
src.to_csv(os.path.join(SRC, "figure_split_vs_alpha.csv"), index=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for col, model in enumerate(MODELS):
    s = src[src.Model == model]
    top, bot = axes[0][col], axes[1][col]
    if s.empty:
        top.set_visible(False)
        bot.set_visible(False)
        continue
    a = s.alpha_req.values

    top.step(a, s.N_L.values, where="post", color=C["N_L"], lw=2.0, label="N_L (Pool L)")
    top.step(a, s.N_T.values, where="post", color=C["N_T"], lw=2.0, ls="--", label="N_T (Pool T)")
    top.fill_between(a, 0, s.N_L.values, step="post", color=C["N_L"], alpha=0.10)
    top.set_ylim(0, 40)                              # headroom so the legend clears the steps
    top.set_ylabel("devices", fontsize=9, color=MUTED)
    top.set_title(f"{model}   (λ = {s.lambda_shown.iloc[0]:.0f} tok/s, "
                  f"{len(s)}/97 α selectable)", fontsize=11, color=INK, pad=8)
    top.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)

    # every split change is a reconfiguration point -- mark them on both rows
    key = (s.n_instances.astype(str) + "x" + s.TP_per_instance.astype(str)).values
    chg = np.flatnonzero(key[1:] != key[:-1]) + 1
    for i in chg:
        for ax in (top, bot):
            ax.axvline(a[i], color=MUTED, lw=0.5, alpha=0.3, zorder=0)

    bot.step(a, s.n_instances.values, where="post", color=C["inst"], lw=2.0, label="instances k")
    bot.step(a, s.TP_per_instance.values, where="post", color=C["tp"], lw=2.0, ls="--",
             label="TP per instance")
    bot.set_xlabel("α  (fraction of requests tagged latency-critical)", fontsize=9, color=MUTED)
    bot.set_ylabel("Pool L factorization", fontsize=9, color=MUTED)
    bot.set_ylim(0, max(s.n_instances.max(), s.TP_per_instance.max()) * 1.45)
    bot.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)

    # selective direct labels, spaced far enough apart that they cannot collide
    marks = [0] + list(chg)
    last_x = -1.0
    for i in marks:
        if a[i] - last_x < 0.16:
            continue
        bot.annotate(f"{s.n_instances.values[i]}×TP{s.TP_per_instance.values[i]}",
                     (a[i], s.n_instances.values[i]), fontsize=6.5, color=INK,
                     ha="left", va="bottom", xytext=(3, 5), textcoords="offset points")
        last_x = a[i]

    for ax in (top, bot):
        ax.grid(alpha=0.18, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#d4d4d8")
        ax.tick_params(labelsize=8, colors=MUTED)

fig.suptitle("Optimal split vs interactive load α  —  N_L grows with α; k and TP trade "
             "Pool L throughput against TTFT within a fixed N_L",
             fontsize=11.5, color=INK, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figure_split_vs_alpha.pdf"), bbox_inches="tight")
print(f"wrote figures/figure_split_vs_alpha.pdf  ({len(src)} source rows)")
