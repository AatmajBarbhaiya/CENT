#!/usr/bin/env python3
"""figure_baseline_dynamic.py — DDA vs a scaled-up static pool under the SAME dynamic workload.

This is the FAIR comparison, and it supersedes figure_pareto_dda_vs_baseline.py as the honest one.
That figure scores static configs at a single operating point; but a static deployment is not
choosing an operating point, it is stuck with one, and must serve BOTH request classes from ONE
pool at every alpha it ever sees. Here the same 24 h diurnal alpha(t) is replayed through both
deployments at each offered load.

Row 1  effective TTFT of the latency class vs offered load  (lower is better)
Row 2  unmet fraction of latency-class demand vs offered load  (lower is better; > 0 is a hard
       failure, unlike spill which is merely slow)

The result is deliberately NOT uniform in DDA's favour -- see finding 13. DDA's benefit tracks how
badly a uniform mapping wastes the fabric: large for 13B (40 layers strand 12 of 32 devices),
marginal for 7B (32 divides 32 exactly, so nothing is stranded), and negative for 70B (Pool L caps
at 117 tok/s = 9% of fabric throughput, so most of the latency class goes unserved).

Input:  results/dda32/baseline_lambda_sweep.csv   (run simulation/baseline_comparison.py first)
Output: figures/figure_baseline_dynamic.pdf ; source data figure_source_data/figure_baseline_dynamic.csv
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
C = {"static": "#eb6834", "dda": "#1baf7a", "bad": "#d1495b"}
INK, MUTED = "#1f2328", "#6b7280"

path = os.path.join(M.DDA, "baseline_lambda_sweep.csv")
if not os.path.exists(path):
    sys.exit(f"{path} not found — run simulation/baseline_comparison.py first")
sw = pd.read_csv(path)
sw.to_csv(os.path.join(SRC, "figure_baseline_dynamic.csv"), index=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex="col")

for col, model in enumerate(MODELS):
    g = sw[sw.Model == model].sort_values("lam_frac")
    top, bot = axes[0][col], axes[1][col]
    if g.empty:
        top.set_visible(False); bot.set_visible(False)
        continue
    x = g.lam_frac.values

    # ---- row 1: effective TTFT ----
    top.plot(x, g.static_TTFT_s, "-s", color=C["static"], lw=2, ms=6,
             mec="white", mew=0.8, label="static 32-dev (one pool)")
    top.plot(x, g.dda_TTFT_s, "-*", color=C["dda"], lw=2, ms=11,
             mec="white", mew=0.6, label="DDA (two pools + controller)")
    top.set_yscale("log")
    top.set_ylabel("effective TTFT (s)  ↓", fontsize=9.5, color=MUTED)
    top.set_title(f"{model}\n{g.static_cfg.iloc[0].replace('static_', '')}",
                  fontsize=10.5, color=INK, pad=8)

    # shade where DDA is worse on TTFT — the region the paper must not hide
    worse = g.dda_TTFT_s.values > g.static_TTFT_s.values
    for i in np.flatnonzero(worse):
        lo = x[i] - (x[i] - x[i - 1]) / 2 if i else x[i]
        hi = x[i] + (x[i + 1] - x[i]) / 2 if i < len(x) - 1 else x[i]
        top.axvspan(lo, hi, color=C["bad"], alpha=0.07, lw=0)

    best = float(np.nanmax(g.static_TTFT_s / g.dda_TTFT_s))
    top.annotate(f"best {best:.1f}× better", (x[int(np.nanargmax(g.static_TTFT_s / g.dda_TTFT_s))],
                                              g.dda_TTFT_s.min()),
                 fontsize=7.5, color=C["dda"], fontweight="bold",
                 ha="left", va="top", xytext=(4, -6), textcoords="offset points")

    # ---- row 2: unmet fraction ----
    bot.plot(x, 100 * g.static_unmet, "-s", color=C["static"], lw=2, ms=6,
             mec="white", mew=0.8, label="static 32-dev")
    bot.plot(x, 100 * g.dda_unmet, "-*", color=C["dda"], lw=2, ms=11,
             mec="white", mew=0.6, label="DDA")
    bot.plot(x, 100 * g.dda_spill, ":", color=C["dda"], lw=1.4, alpha=0.8,
             label="DDA spilled (slow, not lost)")
    bot.set_ylabel("unmet latency-class demand (%)  ↓", fontsize=9.5, color=MUTED)
    bot.set_xlabel("offered load  λ / fabric capacity", fontsize=9.5, color=MUTED)
    bot.set_ylim(-3, 100)

    for ax in (top, bot):
        ax.grid(alpha=0.18, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d4d4d8")
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.legend(fontsize=7.2, frameon=False, loc="best")

fig.suptitle("Same 24 h diurnal α through both deployments — DDA's gain tracks how badly a "
             "uniform mapping wastes the fabric, not model size\n"
             "(shaded = DDA worse on TTFT; 13B wins throughout, 7B only below ~0.6 load, "
             "70B's Pool L is too small to serve the latency class at all)",
             fontsize=11, color=INK, y=1.005)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figure_baseline_dynamic.pdf"), bbox_inches="tight")
print(f"wrote figures/figure_baseline_dynamic.pdf  ({len(sw)} source rows)")