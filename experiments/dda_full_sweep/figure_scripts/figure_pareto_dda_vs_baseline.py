#!/usr/bin/env python3
"""figure_pareto_dda_vs_baseline.py — DDA32 vs CENT baselines, two-class Pareto.

REBASED 2026-07-29 onto the full-DSE tables. Every metric now comes from explorer/dse_metrics.py
(survival-weighted throughput, cumulative TTFT, ceil-bottlenecked Pool T throughput), so this
figure and the CSVs cannot drift. The previous version read the superseded
best_split_by_alpha32.csv and inherited the packing-inflated Pool T throughput (median +15%,
max +72%) plus the flat-mean latency error.

Per-model Pareto:
  x = TTFT of the latency-tagged class  — Pool L for DDA, the single config for static
  y = throughput available to the batch class — Pool T for DDA, the same config for static

A static config serves BOTH classes from one pool, so it can only sit on a Pareto curve. DDA runs
Pool L and Pool T simultaneously, so it lands in the ideal corner (low x, high y).

Static configs are valid-split filtered (pp | num_layers) to drop the uniform-stage artifacts.

Output: figures/figure_pareto_dda_vs_baseline.pdf ; source data figure_source_data/figure_pareto.csv
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
# validated categorical palette (dataviz validator, light surface): worst adjacent CVD dE 9.2,
# normal-vision dE 27.6. The green sits at 2.74:1 contrast -> relief via direct labels + the
# source CSV acting as the table view.
C = {"native": "#2a78d6", "static": "#eb6834", "dda": "#1baf7a"}
INK, MUTED = "#1f2328", "#6b7280"


def dda_frontier(model):
    """alpha-swept DDA points, straight off the DSE decision table.

    Load choice: the BUSIEST lambda that still serves >=90% of the alpha range, so every panel
    shows a near-complete alpha sweep rather than a stub. A median lambda leaves 70B covering
    only alpha 0.02-0.20 (19/97) because its tiny Pool L cannot serve high alpha at that rate.
    """
    b = pd.read_csv(os.path.join(M.DDA, "best_split_by_rate_alpha32.csv"))
    n_alpha = b.alpha_req.nunique()
    b = b[(b.Model == model) & b.selectable]
    if b.empty:
        return [], np.nan
    cnt = b.groupby("lambda_tok_s").size()
    ok = cnt[cnt >= 0.90 * n_alpha]
    lam = ok.index.max() if len(ok) else cnt.idxmax()
    b = b[b.lambda_tok_s == lam].sort_values("alpha_req")
    return [dict(alpha=float(r.alpha_req), N_L=int(r.N_L), N_T=int(r.N_T),
                 n_instances=int(r.n_instances), TP=int(r.TP_per_instance),
                 ttft=float(r.TTFT_L_s), tput=float(r.Tput_T)) for r in b.itertuples()], lam


rows = []
LAM = {}
for model in MODELS:
    for c in M.static_configs(model, 32):
        rows.append(dict(Model=model, series="cent32static", PP=c["pp"], TP=c["tp"],
                         ttft_s=c["ttft_s"], tput=c["tput"]))
    for c in M.static_configs(model, NATIVE[model]):
        rows.append(dict(Model=model, series="native", PP=c["pp"], TP=c["tp"],
                         ttft_s=c["ttft_s"], tput=c["tput"]))
    pts, LAM[model] = dda_frontier(model)
    for p in pts:
        rows.append(dict(Model=model, series="dda", PP="", TP=p["TP"], ttft_s=p["ttft"],
                         tput=p["tput"], alpha=p["alpha"], N_L=p["N_L"], N_T=p["N_T"],
                         n_instances=p["n_instances"], lambda_tok_s=LAM[model]))
src = pd.DataFrame(rows)
src.to_csv(os.path.join(SRC, "figure_pareto.csv"), index=False)


def _rows(model, series):
    return [r for r in rows if r["Model"] == model and r["series"] == series]


fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, model in zip(axes, MODELS):
    # ---- x-limit: keep the region where the data actually lives ------------------------
    # A few static configs sit at extreme TTFT (70B PP80/TP1 at ~18 s, an order of magnitude past
    # everything else). Letting them set the scale compresses the whole DDA cluster into a smear
    # -- at 70B it made two genuinely distinct rungs, N_L=9 (2.143 s) and N_L=12 (2.063 s),
    # overplot so the panel looked like it had 2 points when the sweep visits 3. Clip the axis to
    # the dense region and park off-scale points on the right edge with their true value.
    dd_pre = _rows(model, "dda")
    st_pre = _rows(model, "cent32static") + _rows(model, "native")
    dda_x = np.array([r["ttft_s"] for r in dd_pre]) if dd_pre else np.array([])
    all_x = np.array([r["ttft_s"] for r in st_pre] + list(dda_x))
    # Robust upper bound: keep everything within 3x the median, which retains the whole static32
    # Pareto curve and the DDA sweep, and exiles only the true outliers (7B 2.48 s, 13B 2.36 s,
    # 70B 17.8 s -- each an order off its own cluster).
    keep = all_x[all_x <= 3.0 * np.median(all_x)]
    x_lo = max(0.0, all_x.min() * 0.90)
    x_hi = (keep.max() if len(keep) else all_x.max()) * 1.15
    if x_hi <= x_lo:
        x_hi = all_x.max() * 1.05
    ax.set_xlim(x_lo, x_hi)

    def _plot_clipped(xs, ys, labels, **kw):
        """Draw in-range points normally; pin off-scale ones to the right edge as '>' markers."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        inr = xs <= x_hi
        ax.scatter(xs[inr], ys[inr], **kw)
        off = ~inr
        if off.any():
            edge = x_hi * 0.985
            kw2 = {k: v for k, v in kw.items() if k not in ("marker", "label", "s")}
            ax.scatter(np.full(off.sum(), edge), ys[off], marker=">", s=46, **kw2)
            for xv, yv, lb in zip(xs[off], ys[off], np.asarray(labels, object)[off]):
                ax.annotate(f"{lb} → {xv:.1f}s", (edge, yv), fontsize=5.8, color=MUTED,
                            ha="right", va="center", xytext=(-6, 0), textcoords="offset points")
        return inr

    st = _rows(model, "cent32static")
    if st:
        sx = np.array([r["ttft_s"] for r in st])
        sy = np.array([r["tput"] for r in st])
        o = np.argsort(sx)
        ax.plot(sx[o], sy[o], "-", color=C["static"], lw=1.6, alpha=0.45, zorder=2)
        inr = _plot_clipped(sx, sy, [f"PP{r['PP']}/TP{r['TP']}" for r in st],
                            s=52, c=C["static"], marker="s", edgecolor="white", lw=0.8,
                            label="CENT 32-static", zorder=3)
        vis = np.flatnonzero(inr)
        if len(vis):
            i_lat = vis[int(np.argmin(sx[vis]))]
            i_tp = vis[int(np.argmax(sy[vis]))]
            ax.annotate(f"PP{st[i_lat]['PP']}/TP{st[i_lat]['TP']}\nbest TTFT",
                        (sx[i_lat], sy[i_lat]), fontsize=6.5, color=MUTED,
                        ha="left", va="bottom", xytext=(6, 4), textcoords="offset points")
            ax.annotate(f"PP{st[i_tp]['PP']}/TP{st[i_tp]['TP']}\nbest tput",
                        (sx[i_tp], sy[i_tp]), fontsize=6.5, color=MUTED,
                        ha="right", va="top", xytext=(-6, -6), textcoords="offset points")

    dd = _rows(model, "dda")
    if dd:
        dx = np.array([r["ttft_s"] for r in dd])
        dy = np.array([r["tput"] for r in dd])
        al = np.array([r["alpha"] for r in dd])
        ax.plot(dx, dy, "-", color=C["dda"], lw=1.5, alpha=0.45, zorder=4)
        # 97 alphas collapse onto ~10 distinct (x,y) points, so plain markers hide the sweep.
        # Marker size encodes alpha (magnitude -> monotone ramp), making the coverage visible.
        ax.scatter(dx, dy, s=28 + 150 * (al - al.min()) / max(1e-9, np.ptp(al)),
                   c=C["dda"], marker="*", edgecolor="white", lw=0.6,
                   label=f"DDA32  α {al.min():.2f}→{al.max():.2f}  (marker size = α)", zorder=5)

        # Selective direct labels only: the two alpha extremes plus the best-throughput point.
        # The legend already states the alpha range and marker size carries the sweep, so
        # labelling every distinct split just produces a pile of overlapping text.
        picks = {int(np.argmin(al)): ("left", "bottom", (7, 5)),
                 int(np.argmax(al)): ("left", "top", (7, -6)),
                 int(np.argmax(dy)): ("right", "bottom", (-7, 6))}
        for i, (ha, va, off) in picks.items():
            r = dd[i]
            ax.annotate(f"α={r['alpha']:.2f}\n{r['n_instances']}×TP{r['TP']}/N_T{r['N_T']}",
                        (dx[i], dy[i]), fontsize=6.2, color=INK, fontweight="bold",
                        ha=ha, va=va, xytext=off, textcoords="offset points")

    nv = _rows(model, "native")
    if nv:                                        # drawn IN FRONT: else 70B hides under 32-static
        nx = np.array([r["ttft_s"] for r in nv])
        ny = np.array([r["tput"] for r in nv])
        on = np.argsort(nx)
        # connect the native budget into its own Pareto curve, same as the 32-static series
        ax.plot(nx[on], ny[on], "-", color=C["native"], lw=1.4, alpha=0.40, zorder=6)
        _plot_clipped(nx, ny, [f"PP{r['PP']}/TP{r['TP']}" for r in nv],
                      s=74, c=C["native"], marker="o", edgecolor="black", lw=0.9,
                      label=f"native ({NATIVE[model]} dev)", zorder=7)

    n_dda = len(dd)
    ax.set_title(f"{model}   (DDA at λ = {LAM[model]:.0f} tok/s, {n_dda}/97 α)",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel("latency-class TTFT (s)", fontsize=9, color=MUTED)
    ax.set_ylabel("batch-class throughput (tok/s)", fontsize=9, color=MUTED)
    ax.grid(alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d4d4d8")
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.legend(fontsize=7.5, frameon=False, loc="best")

fig.suptitle("DDA occupies the corner a single static config cannot reach  "
             "(← better TTFT, ↑ better batch throughput)",
             fontsize=11.5, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figure_pareto_dda_vs_baseline.pdf"), bbox_inches="tight")
print(f"wrote figures/figure_pareto_dda_vs_baseline.pdf  ({len(src)} source rows)")