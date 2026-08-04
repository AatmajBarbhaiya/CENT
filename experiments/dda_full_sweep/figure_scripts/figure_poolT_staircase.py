#!/usr/bin/env python3
"""figure_poolT_staircase.py — Pool T throughput is a staircase; packing buys latency, not tput.

NEW 2026-07-29. This is finding 4 + finding 7 + the 10a correction in one figure, and it is the
single most load-bearing plot for the DDA argument:

  LEFT  : Tput_T vs N_T -- flat across long runs of N_T, stepping only when bpd = ceil(L/N_T)
          drops. 7B has 5 throughput-distinct tiers across 26 device counts; N_T 16..31 is ONE
          value (6239 tok/s). Devices added inside a tier buy NO throughput.
  RIGHT : req_lat_T vs N_T -- but those same devices DO buy batch latency, linearly, via
          heterogeneous packing (7B: -0.302 s per device). The CENT uniform-mapping line is flat
          because CENT prices every device at cpb_ceil and never models the floor class.

Together they are the reason the optimiser hands spare devices to Pool T, and the reason
Tput_T must be derived from the UNPACKED latency (deriving it from packed inflated it by up to
+72%, reproducing the uniform-stage overestimate finding 4 warns about).

Output: figures/figure_poolT_staircase.pdf ; source data figure_source_data/figure_poolT_staircase.csv
"""
import os
import sys

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
INK, MUTED = "#1f2328", "#6b7280"

d = pd.read_csv(os.path.join(M.DDA, "dse_full_table.csv"))
src = (d[d.N_L > 0]
       .drop_duplicates(subset=["Model", "N_T"])
       .sort_values(["Model", "N_T"])
       [["Model", "N_T", "cpb_ceil", "cpb_floor", "n_ceil", "n_floor", "idle_T",
         "Tput_T", "req_lat_T_cent_s", "req_lat_T_packed_s", "packing_gain_pct"]])
src.to_csv(os.path.join(SRC, "figure_poolT_staircase.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axL, axR = axes

for model in MODELS:
    s = src[src.Model == model]
    if s.empty:
        continue
    axL.step(s.N_T, s.Tput_T, where="post", color=C[model], lw=2.0, label=model)
    axL.scatter(s.N_T, s.Tput_T, s=16, color=C[model], zorder=3)

    axR.plot(s.N_T, s.req_lat_T_packed_s, "-", color=C[model], lw=2.0, label=f"{model} packed")
    axR.plot(s.N_T, s.req_lat_T_cent_s, ":", color=C[model], lw=1.4, alpha=0.75,
             label=f"{model} CENT uniform")

# label the widest flat tier -- the headline
s7 = src[src.Model == "Llama2-7B"]
flat = s7[s7.Tput_T == s7.Tput_T.max()]
if len(flat) > 1:
    axL.annotate(f"N_T {int(flat.N_T.min())}–{int(flat.N_T.max())}: one throughput\n"
                 f"({flat.Tput_T.iloc[0]:.0f} tok/s across "
                 f"{int(flat.N_T.max() - flat.N_T.min()) + 1} device counts)",
                 (flat.N_T.median(), flat.Tput_T.iloc[0]), fontsize=7, color=INK,
                 ha="center", va="top", xytext=(0, -14), textcoords="offset points")
    lo = flat.req_lat_T_packed_s.max()
    hi = flat.req_lat_T_packed_s.min()
    axR.annotate(f"same tier: {lo:.1f}s → {hi:.1f}s\n(−0.302 s per device, free)",
                 (flat.N_T.max(), hi), fontsize=7, color=INK,
                 ha="right", va="bottom", xytext=(-6, 8), textcoords="offset points")

axL.set_xlabel("N_T  (devices in Pool T)", fontsize=9, color=MUTED)
axL.set_ylabel("Pool T throughput (tok/s)", fontsize=9, color=MUTED)
axL.set_title("Throughput: a staircase — flat inside a tier", fontsize=11, color=INK, pad=8)

axR.set_xlabel("N_T  (devices in Pool T)", fontsize=9, color=MUTED)
axR.set_ylabel("batch request latency (s)", fontsize=9, color=MUTED)
axR.set_yscale("log")
axR.set_title("Latency: falls linearly with every device added", fontsize=11, color=INK, pad=8)

for ax in (axL, axR):
    ax.grid(alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d4d4d8")
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="best")

fig.suptitle("Devices added to Pool T inside a throughput tier buy zero throughput "
             "and substantial batch latency", fontsize=11.5, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figure_poolT_staircase.pdf"), bbox_inches="tight")
print(f"wrote figures/figure_poolT_staircase.pdf  ({len(src)} source rows)")
