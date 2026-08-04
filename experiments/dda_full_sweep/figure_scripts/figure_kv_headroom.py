#!/usr/bin/env python3
"""figure_kv_headroom.py — the KV capacity gate, per device class.

NEW 2026-07-29. The DSE's feasibility gate had no figure. Two panels:

LEFT  : per-device memory split at each Pool L TP degree -- weights vs KV headroom. The weight
        tax is set by TP degree m, NOT replica count j (j replicas also consume j*m devices, so
        the per-device ratio is (W_total/m)/16GiB). At m=1 a 7B replica spends 78% of the device
        on weights and can reach only ~7k context; at m=8, 10% and ~236k.

RIGHT : Pool T KV slack = C_max/C_need at S=4096, where C_need = num_layers because a pp-stage
        pipeline needs pp requests in flight and EVERY in-flight request holds KV on EVERY block.
        Nine configs sit under 1.3x and must never be quoted as comfortable -- including
        70B N_T=16..19 at 1.13x, which carries finding 8's +24% lambda_max result.

Output: figures/figure_kv_headroom.pdf ; source data figure_source_data/figure_kv_headroom.csv
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
C = {"Llama2-7B": "#2a78d6", "Llama2-13B": "#eb6834", "Llama2-70B": "#1baf7a"}
WEIGHT_C, KV_C = "#6b7280", "#2a78d6"
INK, MUTED = "#1f2328", "#6b7280"

# ---- left panel data: per-device weight vs KV headroom by TP degree -----------------
rows = []
for model in MODELS:
    f = M.model_facts(model)
    for m in range(f["cap_floor"], f["tp_sat"] + 1):
        w = f["w_total"] / m
        if w > M.DEV_BYTES:
            continue
        free = M.DEV_BYTES - w
        rows.append(dict(Model=model, TP=m,
                         weights_GiB=w / M.GiB, kv_free_GiB=free / M.GiB,
                         weight_share_pct=100 * w / M.DEV_BYTES,
                         max_context_tok=free * m / f["kv_tok_all"]))
left = pd.DataFrame(rows)

# ---- right panel data: Pool T KV slack ----------------------------------------------
d = pd.read_csv(os.path.join(M.DDA, "dse_full_table.csv"))
right = (d[d.N_L > 0].drop_duplicates(subset=["Model", "N_T"])
         .sort_values(["Model", "N_T"])
         [["Model", "N_T", "cpb_ceil", "kv_slack_T", "C_max_4096_T", "C_need_T"]])

pd.concat([left.assign(panel="poolL_memory"), right.assign(panel="poolT_kv_slack")],
          ignore_index=True).to_csv(os.path.join(SRC, "figure_kv_headroom.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
axL, axR = axes

# LEFT: stacked weights + KV per device, one group per model, x = TP degree
s7 = left[left.Model == "Llama2-7B"]
x = np.arange(len(s7))
axL.bar(x, s7.weights_GiB, width=0.62, color=WEIGHT_C, label="weights", zorder=3)
axL.bar(x, s7.kv_free_GiB, width=0.62, bottom=s7.weights_GiB, color=KV_C,
        label="KV headroom", zorder=3)
axL.set_xticks(x)
axL.set_xticklabels([f"TP{t}" for t in s7.TP], fontsize=8)
axL.axhline(16, color=INK, lw=0.9, ls="--", alpha=0.6)
axL.annotate("16 GiB device", (len(s7) - 0.45, 16), fontsize=7, color=MUTED,
             ha="right", va="top", xytext=(0, -3), textcoords="offset points")
for i, r in enumerate(s7.itertuples()):
    # weight-share % inside the grey block, max-context above the bar: two rows, no collision
    axL.annotate(f"{r.weight_share_pct:.0f}%", (i, r.weights_GiB / 2), fontsize=6.5,
                 color="white", ha="center", va="center", fontweight="bold")
    if i in (0, len(s7) - 1):
        axL.annotate(f"max ctx\n{r.max_context_tok/1000:.0f}k tok", (i, 16.6),
                     fontsize=6.5, color=INK, ha="center", va="bottom")
axL.set_ylabel("per-device memory (GiB)", fontsize=9, color=MUTED)
axL.set_xlabel("Pool L TP degree  (weight tax depends on TP, not replica count)",
               fontsize=9, color=MUTED)
axL.set_title("Llama2-7B: a low-TP replica spends the device on weights",
              fontsize=11, color=INK, pad=8)
axL.set_ylim(0, 20.5)
axL.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)

# RIGHT: Pool T KV slack, log y, danger band under 1.0
for model in MODELS:
    s = right[right.Model == model]
    if s.empty:
        continue
    axR.plot(s.N_T, s.kv_slack_T, "-o", color=C[model], lw=1.8, ms=4.5,
             mec="white", mew=0.7, label=model)
axR.axhspan(0.8, 1.0, color="#d1495b", alpha=0.10, zorder=0)
axR.axhline(1.0, color="#d1495b", lw=1.2, ls="--")
axR.annotate("OOM wall (kv_slack = 1)", (right.N_T.max(), 1.0), fontsize=7,
             color="#d1495b", ha="right", va="top", xytext=(0, -4),
             textcoords="offset points")
thin = right[right.kv_slack_T < 1.3]
if not thin.empty:
    axR.annotate(f"{len(thin)} configs under 1.3×\n7B 1.06 · 13B 1.09 · 70B 1.13",
                 (right.N_T.max(), 1.15), fontsize=7, color=INK,
                 ha="right", va="center")
axR.set_yscale("log")
axR.set_ylim(0.85, 6)
axR.set_yticks([1, 1.5, 2, 3, 4, 5])
axR.set_yticklabels(["1×", "1.5×", "2×", "3×", "4×", "5×"])
# log scale re-adds 2x10^0-style minor labels on top of the custom ticks
axR.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axR.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(
    ["1×", "1.5×", "2×", "3×", "4×", "5×"]))
axR.set_xlabel("N_T  (devices in Pool T)", fontsize=9, color=MUTED)
axR.set_ylabel("KV slack  = C_max / C_need   @ S=4096", fontsize=9, color=MUTED)
axR.set_title("Pool T capacity gate: every in-flight request holds KV on every block",
              fontsize=11, color=INK, pad=8)
axR.legend(fontsize=7.5, frameon=False, loc="best")

for ax in (axL, axR):
    ax.grid(alpha=0.18, lw=0.6, axis="y")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d4d4d8")
    ax.tick_params(labelsize=8, colors=MUTED)

fig.suptitle("KV capacity: what limits a split before latency or throughput does",
             fontsize=11.5, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figure_kv_headroom.pdf"), bbox_inches="tight")
print(f"wrote figures/figure_kv_headroom.pdf  ({len(left) + len(right)} source rows)")