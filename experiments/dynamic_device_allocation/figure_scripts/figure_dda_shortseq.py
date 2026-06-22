"""
figure_dda_strategy1.py — DDA Strategy 1 results vs baseline static configs.

Panel (a): Zoomed Pareto [0–6s]. Static = filled markers. DDA = hollow stars.
           Filled star = the split highlighted in panel (b).
           Ideal corner = (lat-optimal TTFT, MAX tput config) per model.

Panel (b): Best-gain DDA split vs best static with TTFT <= DDA TTFT.
           Shows PP/TP config labels for static and N_L/N_T pool labels for DDA.

TTFT note: short-request TTFT is identical between static and DDA (by construction).
           Long-request TTFT is much worse in DDA (Pool T sacrifices latency for tput).

Run from repo root: python3 figure_scripts/figure_dda_strategy1.py
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SIM_BASE = pd.read_csv('../../cent_simulation/simulation_results.csv')
COMBINED = pd.read_csv('results/combined_metrics.csv')

T_SHORT = 512
MODELS       = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
MODEL_LABELS = ['Llama2-7B\n(8 dev)', 'Llama2-13B\n(20 dev)', 'Llama2-70B\n(32 dev)']
COLORS       = {'Llama2-7B': 'steelblue', 'Llama2-13B': 'darkorange', 'Llama2-70B': 'forestgreen'}
MARKERS      = {'Llama2-7B': 'o', 'Llama2-13B': 's', 'Llama2-70B': '^'}
TOTAL_DEV    = {'Llama2-7B': 8,   'Llama2-13B': 20,  'Llama2-70B': 32}
LAYERS       = {'Llama2-7B': 32,  'Llama2-13B': 40,  'Llama2-70B': 80}


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
    """Ideal = (lat-optimal TTFT, MAX tput config tput). Correct for all models."""
    dev = TOTAL_DEV[model]
    lat  = par[(par['PP'] == 1) & (par['TP'] == dev)]
    topt = par.sort_values('Tput', ascending=False).iloc[[0]]
    if lat.empty: return None, None
    return (float(lat['TTFT'].iloc[0]),
            float(topt['Tput'].iloc[0]))


def best_static_lte(par, dda_ttft):
    """Best static config with TTFT <= dda_ttft (10ms tolerance)."""
    cands = par[par['TTFT'] <= dda_ttft + 0.01]
    if cands.empty: return par.iloc[0]
    return cands.sort_values('Tput', ascending=False).iloc[0]


def best_gain_split(model, par):
    """DDA split with highest gain ratio vs best static at same TTFT."""
    dda = COMBINED[COMBINED['Model'] == model]
    best_gain, best_row = 0, None
    for _, row in dda.iterrows():
        bs   = best_static_lte(par, row['TTFT_short'])
        gain = row['system_tput'] / bs['Tput']
        if gain > best_gain:
            best_gain = gain
            best_row  = row
    return best_row, best_gain


# Pre-compute per-model data
paretos   = {m: baseline_pareto(m)    for m in MODELS}
ideal_pts = {m: ideal_corner(m, paretos[m]) for m in MODELS}
highlighted = {}   # best-gain split per model (for panel b + panel a filled star)
for m in MODELS:
    row, gain = best_gain_split(m, paretos[m])
    bs = best_static_lte(paretos[m], row['TTFT_short'])
    highlighted[m] = {
        'dda_row':     row,
        'gain':        gain,
        'static_row':  bs,
    }

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))


# ══════════════════════════════════════════════════════════════════════════════
# Panel (a): Pareto scatter zoomed [0–6s]
# ══════════════════════════════════════════════════════════════════════════════
X_MAX = 6.5

for model in MODELS:
    col = COLORS[model]
    mk  = MARKERS[model]
    par = paretos[model]
    dda = COMBINED[COMBINED['Model'] == model]
    hi  = highlighted[model]

    # Static Pareto curve
    vis = par[par['TTFT'] <= X_MAX]
    ax1.plot(vis['TTFT'], vis['Tput'], color=col, linewidth=1.3, alpha=0.5, zorder=3)
    ax1.scatter(vis['TTFT'], vis['Tput'], color=col, marker=mk, s=70,
                edgecolors='black', linewidths=0.8, zorder=4, label=f'{model} static')

    # Ideal corner diamond
    ix, iy = ideal_pts[model]
    if ix is not None and ix <= X_MAX:
        ax1.scatter(ix, iy, marker='D', s=110, zorder=7,
                    color=col, edgecolors='black', linewidths=1.5, alpha=0.45)

    # All DDA splits — hollow stars first (lower zorder), filled star on top
    for _, row in dda.iterrows():
        if row['TTFT_short'] > X_MAX: continue
        if int(row['N_L']) == int(hi['dda_row']['N_L']): continue  # draw last
        ax1.scatter(row['TTFT_short'], row['system_tput'],
                    marker='*', s=300, zorder=8,
                    facecolors='white', edgecolors=col, linewidths=1.8)

    # Highlighted (best-gain) star drawn last → always on top
    bd_row = hi['dda_row']
    if bd_row['TTFT_short'] <= X_MAX:
        ax1.scatter(bd_row['TTFT_short'], bd_row['system_tput'],
                    marker='*', s=550, zorder=10,
                    facecolors=col, edgecolors='black', linewidths=2.0)

    # Arrow + gain label for highlighted split
    bd = hi['dda_row']
    bs = hi['static_row']
    if bd['TTFT_short'] <= X_MAX:
        ax1.annotate('',
                     xy=(bd['TTFT_short'], bd['system_tput']),
                     xytext=(bs['TTFT'], bs['Tput']),
                     arrowprops=dict(arrowstyle='->', color=col, lw=2.0))
        x_off = 0.20 if model == 'Llama2-13B' else 0.08
        mid_y = (bd['system_tput'] + bs['Tput']) / 2
        ax1.text(bd['TTFT_short'] + x_off, mid_y,
                 f'+{hi["gain"]-1:.0%}', fontsize=9.5, color=col,
                 fontweight='bold', va='center')
        # Dashed line to ideal corner
        if ix is not None and ix <= X_MAX:
            ax1.plot([bd['TTFT_short'], ix], [bd['system_tput'], iy],
                     color=col, linestyle=':', linewidth=1.0, alpha=0.5, zorder=3)

# SLO lines
ax1.axvline(x=1.0, color='red',     linestyle='--', linewidth=1.5, label='1s SLO')
ax1.axvline(x=5.0, color='darkred', linestyle=':',  linewidth=1.5, label='5s SLO')
ax1.axvspan(0, 1.0, alpha=0.06, color='green', zorder=0)
ax1.text(X_MAX - 0.1, 80,
         '70B PP=80,TP=1\nTTFT=28.7s (off-scale)',
         ha='right', va='bottom', fontsize=7, color=COLORS['Llama2-70B'],
         style='italic', alpha=0.8)

legend_els = [
    mlines.Line2D([0],[0], marker=MARKERS[m], color=COLORS[m], linestyle='None',
                  ms=8, markeredgecolor='black', label=m.replace('Llama2-','L2-'))
    for m in MODELS
] + [
    mlines.Line2D([0],[0], marker='*', color='white', linestyle='None', ms=14,
                  markeredgecolor='black', markeredgewidth=1.5, label='DDA (hollow=other splits)'),
    mlines.Line2D([0],[0], marker='*', color='gray', linestyle='None', ms=14,
                  markeredgecolor='black', markeredgewidth=2.5, label='DDA (filled=best-gain, shown in b)'),
    mlines.Line2D([0],[0], marker='D', color='gray', alpha=0.5, linestyle='None', ms=8,
                  markeredgecolor='black', label='Ideal corner'),
    mlines.Line2D([0],[0], color='red',     linestyle='--', label='1s SLO'),
    mlines.Line2D([0],[0], color='darkred', linestyle=':',  label='5s SLO'),
]
ax1.legend(handles=legend_els, fontsize=7.5, loc='upper right', ncol=2)
ax1.set_xlabel('Short-Request TTFT (s)', fontsize=12)
ax1.set_ylabel('System Throughput (tok/s)', fontsize=12)
ax1.set_title('(a) DDA vs Static Pareto — Zoomed [0–6s]\n',
              fontsize=10.5)
ax1.set_xlim(0, X_MAX)
ax1.set_ylim(0, SIM_BASE['Throughput (tokens/s)'].max() * 1.12)
ax1.grid(linestyle='--', alpha=0.5)
ax1.tick_params(labelsize=11)


# ══════════════════════════════════════════════════════════════════════════════
# Panel (b): Best-gain split — throughput bars with config labels
# ══════════════════════════════════════════════════════════════════════════════
x     = np.arange(len(MODELS))
width = 0.35

static_tput, static_ttft_v, static_cfg = [], [], []
dda_tput,    dda_ttft_v,    dda_cfg    = [], [], []
gain_vals = []

for model in MODELS:
    hi   = highlighted[model]
    bd   = hi['dda_row']
    bs   = hi['static_row']
    lay  = LAYERS[model]

    static_tput.append(bs['Tput'])
    static_ttft_v.append(bs['TTFT'])
    static_cfg.append(f"PP={int(bs['PP'])}, TP={int(bs['TP'])}")

    dda_tput.append(bd['system_tput'])
    dda_ttft_v.append(bd['TTFT_short'])
    NL = int(bd['N_L']); NT = int(bd['N_T'])
    # Pool L lat-optimal: PP=1, TP=N_L; Pool T tput-optimal: PP=num_layers, TP=1
    dda_cfg.append(f"L: N_L={NL} (PP=1,TP={NL})\nT: N_T={NT} (PP={lay},TP=1)")
    gain_vals.append(hi['gain'])

bars_s = ax2.bar(x - width/2, static_tput, width,
                 label='Static config (best with TTFT ≤ DDA TTFT)',
                 color='lightsalmon', edgecolor='black')
bars_d = ax2.bar(x + width/2, dda_tput, width,
                 label='DDA Strategy 1 — Pool L + Pool T concurrent\n(highest gain split per model)',
                 color='steelblue', edgecolor='black', alpha=0.88)

y_top = max(dda_tput) * 1.38

# Annotate static: tput + TTFT + config
for bar, tput, ttft, cfg in zip(bars_s, static_tput, static_ttft_v, static_cfg):
    ax2.text(bar.get_x() + bar.get_width()/2, tput + y_top*0.01,
             f'{tput:.0f} tok/s\nTTFT={ttft:.2f}s\n{cfg}',
             ha='center', va='bottom', fontsize=7, color='dimgray', linespacing=1.3)

# Annotate DDA: gain inside bar, tput + TTFT + config above
for bar, tput, ttft, cfg, gain in zip(bars_d, dda_tput, dda_ttft_v, dda_cfg, gain_vals):
    # Gain multiplier inside bar
    ax2.text(bar.get_x() + bar.get_width()/2, tput * 0.42,
             f'{gain:.1f}×', ha='center', va='center',
             fontsize=12, color='white', fontweight='bold')
    # Tput + TTFT + config above bar
    ax2.text(bar.get_x() + bar.get_width()/2, tput + y_top*0.01,
             f'{tput:.0f} tok/s\nTTFT={ttft:.2f}s\n{cfg}',
             ha='center', va='bottom', fontsize=7, fontweight='bold', linespacing=1.3)

ax2.set_xticks(x)
ax2.set_xticklabels(MODEL_LABELS, fontsize=11)
ax2.set_ylabel('System Throughput (tok/s)', fontsize=12)
ax2.set_title('(b) Best-Gain DDA Split vs Static — Same TTFT Constraint\n',
              fontsize=10)
ax2.legend(fontsize=8.5, loc='upper left')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.tick_params(axis='y', labelsize=11)
ax2.set_ylim(0, y_top)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('figure_source_data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

rows = []
for model in MODELS:
    par = paretos[model]
    ix, iy = ideal_pts[model]
    for _, r in par.iterrows():
        rows.append({'Model': model, 'Type': 'static',
                     'PP': r['PP'], 'TP': r['TP'],
                     'TTFT_short': r['TTFT'], 'system_tput': r['Tput']})
    rows.append({'Model': model, 'Type': 'ideal_corner',
                 'TTFT_short': ix, 'system_tput': iy})
    dda = COMBINED[COMBINED['Model'] == model]
    for _, r in dda.iterrows():
        hi_flag = int(r['N_L']) == int(highlighted[model]['dda_row']['N_L'])
        rows.append({'Model': model, 'Type': 'dda', 'N_L': r['N_L'], 'N_T': r['N_T'],
                     'TTFT_short': r['TTFT_short'], 'system_tput': r['system_tput'],
                     'Tput_L': r['Tput_L'], 'Tput_T': r['Tput_T'],
                     'is_best_gain': hi_flag})

pd.DataFrame(rows).to_csv('figure_source_data/figure_dda_shortseq.csv', index=False)
plt.savefig('figures/figure_dda_shortseq.pdf', bbox_inches='tight')
print("Saved: figures/figure_dda_shortseq.pdf")

print("\nSummary (best-gain splits):")
for m in MODELS:
    hi = highlighted[m]
    bd, bs = hi['dda_row'], hi['static_row']
    ix, iy = ideal_pts[m]
    print(f"  {m}:")
    print(f"    Ideal corner: ({ix:.3f}s, {iy:.0f} tok/s)")
    print(f"    Static: PP={int(bs['PP'])},TP={int(bs['TP'])} TTFT={bs['TTFT']:.3f}s tput={bs['Tput']:.0f}")
    print(f"    DDA:    N_L={int(bd['N_L'])},N_T={int(bd['N_T'])} TTFT={bd['TTFT_short']:.3f}s sys_tput={bd['system_tput']:.0f}  gain={hi['gain']:.2f}×")
    slo = 'SLO-1s' if bd['SLO_1s_short'] else ('SLO-5s' if bd['SLO_5s_short'] else 'NO-SLO')
    print(f"    SLO: {slo}")
