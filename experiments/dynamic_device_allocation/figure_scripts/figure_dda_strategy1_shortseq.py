"""
figure_dda_strategy1_shortseq.py — DDA Strategy 1 results, per-model scatter layout.

SEQLEN NOTE:
  Short-request TTFT (X-axis) = mean(token_latency for seqlen ≤ 512) × 512 / 1000
  system_tput (Y-axis) = Pool_L_tput + Pool_T_tput (both averaged over ALL seqlens)
  DDA threshold T = 512 tokens.

Row 1 — three scatter panels (a)(b)(c), one per model:
  Static (PP,TP) configs as filled dots + Pareto line.
  All DDA splits as hollow stars.
  Best-gain DDA split as filled star.
  Ideal corner as diamond.
  Hatching between static Pareto curve and DDA star height (shows gain region).
  Arrow static → DDA + gain % label.
  1s / 5s SLO vertical lines.

Row 2 — bar chart (d):
  Best-gain DDA system_tput vs nearest static (TTFT ≤ DDA TTFT).
  Config labels on each bar. Gain multiplier inside DDA bar.

Run from repo root: python3 figure_scripts/figure_dda_strategy1_shortseq.py
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SIM_BASE = pd.read_csv('../../cent_simulation/simulation_results.csv')
COMBINED = pd.read_csv('results/combined_metrics.csv')

T_SHORT      = 512
X_MAX        = 6.5
MODELS       = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
MODEL_LABELS = ['Llama2-7B\n(8 dev)', 'Llama2-13B\n(20 dev)', 'Llama2-70B\n(32 dev)']
COLORS       = {'Llama2-7B': 'steelblue', 'Llama2-13B': 'darkorange', 'Llama2-70B': 'forestgreen'}
MARKERS      = {'Llama2-7B': 'o', 'Llama2-13B': 's', 'Llama2-70B': '^'}
TOTAL_DEV    = {'Llama2-7B': 8,   'Llama2-13B': 20,  'Llama2-70B': 32}
LAYERS       = {'Llama2-7B': 32,  'Llama2-13B': 40,  'Llama2-70B': 80}
PANEL_ABC    = ['(a)', '(b)', '(c)']

# ── Helpers ────────────────────────────────────────────────────────────────────

def baseline_pareto(model):
    b = SIM_BASE[SIM_BASE['Model'] == model]
    pts = []
    for (pp, tp), grp in b.groupby(['Pipeline parallelism', 'Tensor parallelism']):
        short = grp[grp['Sequence length'] <= T_SHORT]
        if short.empty: continue
        ttft = short['Token latency (ms)'].mean() * T_SHORT / 1000
        tput = grp['Throughput (tokens/s)'].mean()
        pts.append({'PP': pp, 'TP': tp, 'TTFT': ttft, 'Tput': tput})
    return pd.DataFrame(pts).sort_values('TTFT').reset_index(drop=True)


def ideal_corner(model, par):
    dev  = TOTAL_DEV[model]
    lat  = par[(par['PP'] == 1) & (par['TP'] == dev)]
    topt = par.sort_values('Tput', ascending=False).iloc[[0]]
    if lat.empty: return None, None
    return float(lat['TTFT'].iloc[0]), float(topt['Tput'].iloc[0])


def best_static_lte(par, dda_ttft):
    cands = par[par['TTFT'] <= dda_ttft + 0.01]
    if cands.empty: return par.iloc[0]
    return cands.sort_values('Tput', ascending=False).iloc[0]


def best_gain_split(model, par):
    dda = COMBINED[COMBINED['Model'] == model]
    best_gain, best_row = 0, None
    for _, row in dda.iterrows():
        bs   = best_static_lte(par, row['TTFT_short'])
        gain = row['system_tput'] / bs['Tput']
        if gain > best_gain:
            best_gain = gain
            best_row  = row
    return best_row, best_gain


# ── Pre-compute ────────────────────────────────────────────────────────────────
paretos     = {m: baseline_pareto(m) for m in MODELS}
ideal_pts   = {m: ideal_corner(m, paretos[m]) for m in MODELS}
highlighted = {}
for m in MODELS:
    row, gain = best_gain_split(m, paretos[m])
    bs = best_static_lte(paretos[m], row['TTFT_short'])
    highlighted[m] = {'dda_row': row, 'gain': gain, 'static_row': bs}


# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))  # Matched to motivation figure size

# Main grid: 2 rows, matching the exact height_ratios from the motivation plot
gs = GridSpec(2, 1, figure=fig, hspace=0.4, height_ratios=[1.0, 1.6])

# Row 0: 3 scatter panels, spaced identically to motivation
gs_top = gs[0].subgridspec(1, 3, wspace=0.45)
scatter_axes = [fig.add_subplot(gs_top[0, k]) for k in range(3)]

# Row 1: 1 bar chart, perfectly centered.
# Using a 1x4 subgrid and spanning columns 1 and 2 gives exactly 50% width.
# This gives the bar chart the exact same width and aspect ratio as 
# the bottom plots in your motivation figure!
gs_bot = gs[1].subgridspec(1, 4)
ax_bar = fig.add_subplot(gs_bot[0, 1:3])

Y_MAX = SIM_BASE['Throughput (tokens/s)'].max() * 1.12

# ══════════════════════════════════════════════════════════════════════════════
# Row 1: Per-model scatter panels (a)(b)(c)
# ══════════════════════════════════════════════════════════════════════════════
for idx, model in enumerate(MODELS):
    ax  = scatter_axes[idx]
    col = COLORS[model]
    mk  = MARKERS[model]
    par = paretos[model]
    dda = COMBINED[COMBINED['Model'] == model]
    hi  = highlighted[model]
    lbl = PANEL_ABC[idx]
    short = model.replace('Llama2-', 'L2-')

    bd  = hi['dda_row']
    bs  = hi['static_row']
    ix, iy = ideal_pts[model]

    # ── Static Pareto ────────────────────────────────────────────────────────
    vis = par[par['TTFT'] <= X_MAX].sort_values('TTFT')
    ax.plot(vis['TTFT'], vis['Tput'], color=col, linewidth=1.4, alpha=0.5, zorder=3)
    ax.scatter(vis['TTFT'], vis['Tput'], color=col, marker=mk, s=65,
               edgecolors='black', linewidths=0.7, zorder=4, alpha=0.8)

    # ── Hatching: region between Pareto curve and ideal-corner y ─────────────
    # Mirrors motivation figure: shows TTFT+tput space no single static config
    # can reach simultaneously. DDA stars in/above this region prove gain.
    if iy is not None:
        max_tput_row = vis.sort_values('Tput', ascending=False).iloc[0]
        hatch_x_max  = float(max_tput_row['TTFT'])
        clip_pts = vis[vis['TTFT'] <= hatch_x_max + 0.01].sort_values('TTFT')
        if not clip_pts.empty:
            ax.fill_between(clip_pts['TTFT'],
                            clip_pts['Tput'],
                            iy,
                            where=(clip_pts['Tput'] <= iy),
                            alpha=0.12, color=col, zorder=1,
                            hatch='///', edgecolor=col, linewidth=0.4)

    # ── Ideal corner ◆ + dotted drop-lines ───────────────────────────────────
    if ix is not None and ix <= X_MAX:
        ax.scatter(ix, iy, marker='D', s=100, zorder=7,
                   color=col, edgecolors='black', linewidths=1.4, alpha=0.5)
        ax.plot([ix, ix], [0, iy], color=col, linestyle=':', linewidth=1.0, alpha=0.5)
        ax.plot([0, ix], [iy, iy], color=col, linestyle=':', linewidth=1.0, alpha=0.5)

    # ── DDA splits: hollow stars (lower z) then filled star on top ───────────
    for _, row in dda.iterrows():
        if row['TTFT_short'] > X_MAX: continue
        if int(row['N_L']) == int(bd['N_L']): continue
        ax.scatter(row['TTFT_short'], row['system_tput'],
                   marker='*', s=260, zorder=8,
                   facecolors='white', edgecolors=col, linewidths=1.7)

    if bd['TTFT_short'] <= X_MAX:
        ax.scatter(bd['TTFT_short'], bd['system_tput'],
                   marker='*', s=520, zorder=10,
                   facecolors=col, edgecolors='black', linewidths=2.0)

    # ── Arrow + gain label ───────────────────────────────────────────────────
    if bd['TTFT_short'] <= X_MAX:
        ax.annotate('',
                    xy=(bd['TTFT_short'], bd['system_tput']),
                    xytext=(bs['TTFT'], bs['Tput']),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2.0))
        x_off = 0.18 if model == 'Llama2-13B' else 0.08
        mid_y = (bd['system_tput'] + bs['Tput']) / 2
        ax.text(bd['TTFT_short'] + x_off, mid_y,
                f'+{hi["gain"]-1:.0%}', fontsize=9.5, color=col,
                fontweight='bold', va='center')
        # Dotted line to ideal corner
        if ix is not None and ix <= X_MAX:
            ax.plot([bd['TTFT_short'], ix], [bd['system_tput'], iy],
                    color=col, linestyle=':', linewidth=0.9, alpha=0.45, zorder=3)

    # ── SLO lines ────────────────────────────────────────────────────────────
    ax.axvline(x=1.0, color='red',     linestyle='--', linewidth=1.4, label='1s SLO')
    ax.axvline(x=5.0, color='darkred', linestyle=':',  linewidth=1.3, label='5s SLO')
    ax.axvspan(0, 1.0, alpha=0.06, color='green', zorder=0)

    # ── Labels ───────────────────────────────────────────────────────────────
    if idx == 0:
        ax.set_ylabel('System Throughput (tok/s)  ↑ higher', fontsize=11)
    ax.set_xlabel('Short-Req TTFT (seqlen≤512, s)  ← lower', fontsize=10.5)
    ax.set_title(f'{lbl} {short} — DDA Strategy 1 Splits vs Static Configs',
                 fontsize=10.5)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.grid(linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=10)

    # Mini legend per panel
    legend_els = [
        mlines.Line2D([0],[0], marker=mk, color=col, linestyle='None', ms=8,
                      markeredgecolor='black', label=f'{short} static'),
        mlines.Line2D([0],[0], marker='*', color='white', linestyle='None', ms=13,
                      markeredgecolor=col, markeredgewidth=1.5, label='DDA splits'),
        mlines.Line2D([0],[0], marker='*', color=col, linestyle='None', ms=14,
                      markeredgecolor='black', markeredgewidth=1.5, label='Best-gain DDA'),
        mlines.Line2D([0],[0], color='red', linestyle='--', label='1s SLO'),
        mlines.Line2D([0],[0], color='darkred', linestyle=':', label='5s SLO'),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, loc='upper right')


# ══════════════════════════════════════════════════════════════════════════════
# Row 2: Bar chart (d) — same as original panel (b)
# ══════════════════════════════════════════════════════════════════════════════
x     = np.arange(len(MODELS))
width = 0.25

static_tput, static_ttft_v, static_cfg = [], [], []
dda_tput,    dda_ttft_v,    dda_cfg    = [], [], []
gain_vals = []

for model in MODELS:
    hi  = highlighted[model]
    bd  = hi['dda_row']
    bs  = hi['static_row']
    lay = LAYERS[model]

    static_tput.append(bs['Tput'])
    static_ttft_v.append(bs['TTFT'])
    static_cfg.append(f"PP={int(bs['PP'])}, TP={int(bs['TP'])}")

    dda_tput.append(bd['system_tput'])
    dda_ttft_v.append(bd['TTFT_short'])
    NL = int(bd['N_L']); NT = int(bd['N_T'])
    dda_cfg.append(f"N_L={NL} (PP=1,TP={NL})\nN_T={NT} (PP={lay},TP=1)")
    gain_vals.append(hi['gain'])

bars_s = ax_bar.bar(x - width/2, static_tput, width,
                    label='Static config',
                    color='lightsalmon', edgecolor='black')
bars_d = ax_bar.bar(x + width/2, dda_tput, width,
                    label='DDA Strategy 1\n(highest gain split per model)',
                    color=[COLORS[m] for m in MODELS], edgecolor='black', alpha=0.88)

y_top = max(dda_tput) * 1.38

# Annotate static bars
for bar, tput, ttft, cfg in zip(bars_s, static_tput, static_ttft_v, static_cfg):
    x_pos = bar.get_x() + bar.get_width()/2 - 0.02  # <-- DECREASE THIS TO MOVE FURTHER LEFT
    ax_bar.text(x_pos, tput + y_top*0.01,
                f'{tput:.0f} tok/s\n({ttft:.2f}s)\n{cfg}',
                ha='center', va='bottom', fontsize=7.5, color='black', linespacing=1.3)

# Annotate DDA bars
for bar, tput, ttft, cfg, gain in zip(bars_d, dda_tput, dda_ttft_v, dda_cfg, gain_vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, tput * 0.42,
                f'{gain:.1f}×', ha='center', va='center',
                fontsize=13, color='white', fontweight='bold')
    ax_bar.text(bar.get_x() + bar.get_width()/2, tput + y_top*0.01,
                f'{tput:.0f} tok/s\n({ttft:.2f}s)\n{cfg}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', linespacing=1.3)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(MODEL_LABELS, fontsize=12)
ax_bar.set_ylabel('System Throughput (tok/s)', fontsize=12)
ax_bar.set_title('(d) Best-Gain DDA Split vs Static — Same TTFT Constraint (seqlen≤512)',
                 fontsize=10.5)
ax_bar.legend(fontsize=9, loc='upper left')
ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
ax_bar.tick_params(axis='y', labelsize=11)
ax_bar.set_ylim(0, y_top)


plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)
os.makedirs('figure_source_data', exist_ok=True)
plt.savefig('figures/figure_dda_strategy1_shortseq.pdf', bbox_inches='tight')
print("Saved: figures/figure_dda_strategy1_shortseq.pdf")

# Summary
print(f"\nSeqlen: short-request TTFT = mean(token_lat for seqlen ≤ {T_SHORT}) × {T_SHORT}/1000")
for m in MODELS:
    hi = highlighted[m]
    bd, bs = hi['dda_row'], hi['static_row']
    print(f"  {m}: DDA({int(bd['N_L'])},{int(bd['N_T'])}) "
          f"tput={bd['system_tput']:.0f} TTFT={bd['TTFT_short']:.3f}s | "
          f"static PP={int(bs['PP'])},TP={int(bs['TP'])} tput={bs['Tput']:.0f} | "
          f"gain={hi['gain']:.2f}×")
