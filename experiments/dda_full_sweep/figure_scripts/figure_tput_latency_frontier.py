#!/usr/bin/env python3
"""figure_tput_latency_frontier.py — at equal latency, how much TOTAL fabric throughput?

THE headline comparison.

    x = latency the latency-tagged class ACTUALLY experiences (spill included)
    y = TOTAL fabric throughput, Pool L + Pool T

Both deployments scored on the same quantity: what all 32 devices deliver while holding that
latency for the latency class.

Three modelling decisions, each correcting an earlier mistake:

1. THE WHOLE STATIC CURVE, not one config. Earlier work compared DDA against the single
   max-throughput static config (7B PP32/TP1, TTFT 0.718 s) -- the one tuned for the OPPOSITE
   objective -- and concluded DDA's latency was unimpressive. An operator picks a point on the
   static Pareto curve, so the comparison must span all of it.

2. TOTAL throughput, not batch-only, and CAPACITY not utilisation. At moderate lambda every
   deployment serves all the demand there is, so plotting SERVED throughput returns lambda for
   everyone and every difference vanishes -- that measures the workload, not the deployment.

3. SPILL, not binary feasibility. An undersized Pool L is not a failure: the excess spills to
   Pool T and is served there, just slowly. The x coordinate is therefore the EFFECTIVE latency
   from dda_controller.route(), so DDA points slide RIGHT as spill grows. Only demand neither
   pool can take actually queues -- that alone is drawn hollow.

Input:  results/dda32/frontier_{static,dda,matched}.csv  (run simulation/tput_latency_frontier.py)
Output: figures/figure_tput_latency_frontier.pdf ; source figure_source_data/figure_frontier.csv
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
NATIVE = {"Llama2-7B": 8, "Llama2-13B": 20, "Llama2-70B": 32}
C = {"native": "#2a78d6", "static": "#eb6834", "dda": "#1baf7a"}
INK, MUTED = "#1f2328", "#6b7280"

need = [os.path.join(M.DDA, f"frontier_{k}.csv") for k in ("static", "dda", "matched")]
if not all(os.path.exists(p) for p in need):
    sys.exit("run simulation/tput_latency_frontier.py first")
S, D, X = (pd.read_csv(p) for p in need)
pd.concat([S, D], ignore_index=True).to_csv(os.path.join(SRC, "figure_frontier.csv"), index=False)

ALPHA = S.alpha.iloc[0] if "alpha" in S else 0.30

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

for col, model in enumerate(MODELS):
    ax = axes[col]
    s32 = S[(S.Model == model) & (S.kind == "static32")]
    nat = S[(S.Model == model) & (S.kind == "native")]
    dd = D[D.Model == model]
    if dd.empty or s32.empty:
        ax.set_visible(False)
        continue
    has_native = not nat.empty      # 70B's native budget IS 32 dev -> no separate series

    def env(df):
        """Best TOTAL throughput reachable at or below each effective latency, over FEASIBLE
        points only. Monotone by construction -- raw rungs sorted by x zig-zag, because many
        share a latency with very different Pool T throughput."""
        d = df[df.feasible]
        if d.empty:
            return np.array([]), np.array([])
        d = d.sort_values("eff_lat_s")
        return d.eff_lat_s.values, np.maximum.accumulate(d.total_tput.values)

    ex, ey = env(dd)
    sx, sy = env(s32)
    if len(ex) and len(sx):
        grid = np.unique(np.concatenate([ex, sx]))
        di = np.interp(grid, ex, ey, left=np.nan, right=ey[-1])
        si = np.interp(grid, sx, sy, left=np.nan, right=sy[-1])
        ax.fill_between(grid, si, di, where=(di > si), color=C["dda"], alpha=0.18, lw=0,
                        zorder=1, label="reachable only by DDA")

    # hollow = cannot carry this (alpha, lambda) even WITH spill, so demand queues. Kept on
    # screen so the reader sees why the frontier shrank instead of points silently vanishing.
    for df, colr, sym, nm, big in ((s32, C["static"], "s", "static 32-dev (one pool)", False),
                                   (nat, C["native"], "o", f"native {NATIVE[model]}-dev", False),
                                   (dd, C["dda"], "*", "DDA rungs", True)):
        if df.empty:
            continue
        bad = df[~df.feasible]
        if len(bad):
            ax.plot(bad.eff_lat_s, bad.total_tput, sym, mfc="none", mec=colr,
                    ms=11 if big else 7, mew=1.2, alpha=0.55, zorder=2,
                    label=f"{nm} · overloaded")
        good = df[df.feasible]
        if len(good) and big:
            ax.plot(good.eff_lat_s, good.total_tput, sym, color=colr, ms=9, alpha=0.40,
                    mec="none", zorder=3, label=nm)

    if len(sx):
        ax.plot(sx, sy, "-s", color=C["static"], lw=2.2, ms=7, mec="white", mew=0.9,
                label="static 32-dev (one pool)", zorder=4)
    if has_native:
        nx, ny = env(nat)
        if len(nx):
            ax.plot(nx, ny, "-o", color=C["native"], lw=1.8, ms=7, mec="white", mew=0.9,
                    label=f"native {NATIVE[model]}-dev (one pool)", zorder=5)
    if len(ex):
        ax.plot(ex, ey, "-*", color=C["dda"], lw=2.4, ms=13, mec="white", mew=0.7,
                label="DDA frontier", zorder=6)

    xm = X[X.Model == model] if len(X) else pd.DataFrame()
    if len(xm):
        b = xm.loc[xm.tput_ratio.idxmax()]
        # placed in AXES coordinates: anchoring to the data point pushed the label outside the
        # axes whenever the best-ratio point sat near an edge
        ax.annotate(f"best matched gain: {b.tput_ratio:.2f}× total tput\n"
                    f"at {b.ttft_bar_s:.2f} s   ({b.static_cfg} → {b.dda_cfg})",
                    xy=(0.03, 0.03), xycoords="axes fraction", fontsize=7.4, color=INK,
                    fontweight="bold", ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d4d4d8", alpha=0.85))

    ax.set_xscale("log")
    ax.set_xlabel("effective latency-class latency (s)  ↓ better", fontsize=9.5, color=MUTED)
    ax.set_ylabel("TOTAL throughput, Pool L + Pool T (tok/s)  ↑ better",
                  fontsize=9.5, color=MUTED)
    ax.set_title(model, fontsize=12.5, color=INK, pad=8, fontweight="bold")
    ax.grid(alpha=0.2, lw=0.6, which="both")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d4d4d8")
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.legend(fontsize=6.8, frameon=False, loc="best")

fig.suptitle(f"At equal latency for the latency class, how much TOTAL fabric throughput?   "
             f"(α={ALPHA:.2f}, λ=45% of capacity · up-and-left is better)\n"
             "A static pool must meet the latency bar with the WHOLE fabric; DDA meets it in "
             "Pool L and spills the rest to Pool T — which is why its points slide right as α·λ grows",
             fontsize=11.5, color=INK, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figure_tput_latency_frontier.pdf"), bbox_inches="tight")
print(f"wrote figures/figure_tput_latency_frontier.pdf  ({len(S)+len(D)} source rows)")