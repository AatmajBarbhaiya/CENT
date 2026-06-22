"""
figure_dda_strategy2_tol.py — DDA Strategy 2 with 5% SLO Tolerance.

Same as figure_dda_strategy2.py but allows TTFT_L ≤ SLO × 1.05 (soft compliance).

Effect on 13B: N_L=6 (TTFT=1.003s = 0.3% above 1s SLO) now qualifies → S2 picks
N_L=6 (3020 tok/s) instead of N_L=10 (2506 tok/s). Same throughput as S1.
Bars annotated with "(~1s ±5%)" to indicate soft SLO compliance.

Bimodal SLO: interactive=1s (±5% tol), batch=5s.
Routing: SLO-1s requests → Pool L; SLO-5s requests → Pool T.
Optimal N_L = max system_tput subject to TTFT_L ≤ SLO_interactive × (1 + SLO_TOL).

Run from repo root: python3 figure_scripts/figure_dda_strategy2_tol.py

Key S2 vs S1 difference:
  S1 picks: argmax gain-ratio vs static at same TTFT
  S2 picks: argmax system_tput s.t. TTFT_L < SLO_interactive
  -> 70B: S2 N_L=4 (tput=1268) > S1 N_L=12 (tput=1057) because S2 objective is different
  -> 13B: S2 N_L=10 (tput=2506, SLO✓) < S1 N_L=6 (tput=3020, SLO✗)

Run from repo root: python3 figure_scripts/figure_dda_strategy2.py
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SIM_BASE = pd.read_csv('../../cent_simulation/simulation_results.csv')
COMBINED = pd.read_csv('results/combined_metrics.csv')

T_SHORT    = 512
SLO_INTER  = 1.0    # interactive SLO (s)
SLO_BATCH  = 5.0    # batch SLO (s)
SLO_TOL    = 0.05   # 5% soft tolerance: TTFT_L ≤ SLO_INTER × 1.05 counts as compliant

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


def best_static_lte(par, ttft_target):
    cands = par[par['TTFT'] <= ttft_target + 0.01]
    if cands.empty: return par.iloc[0]
    return cands.sort_values('Tput', ascending=False).iloc[0]


def s1_split(model):
    """S1: best gain-ratio split (from figure_dda_shortseq)."""
    dda = COMBINED[COMBINED['Model'] == model]
    par = baseline_pareto(model)
    best_gain, best_row = 0, None
    for _, row in dda.iterrows():
        bs = best_static_lte(par, row['TTFT_short'])
        gain = row['system_tput'] / bs['Tput']
        if gain > best_gain:
            best_gain = gain
            best_row = row
    return best_row, best_gain


def s2_split(model):
    """S2 (5% tol): max system_tput s.t. TTFT_L ≤ SLO_INTER×1.05. Fallback to 5s."""
    dda = COMBINED[COMBINED['Model'] == model]
    soft_limit = SLO_INTER * (1 + SLO_TOL)
    compliant = dda[dda['TTFT_short'] <= soft_limit]
    if not compliant.empty:
        row = compliant.sort_values('system_tput', ascending=False).iloc[0]
        # True = strict 1s met; '~1s' = within tolerance only
        strict = row['TTFT_short'] <= SLO_INTER
        return row, ('1s' if strict else '~1s')
    compliant_5s = dda[dda['SLO_5s_short']]
    row = compliant_5s.sort_values('system_tput', ascending=False).iloc[0]
    return row, '5s'


# Pre-compute
s1_data = {m: s1_split(m) for m in MODELS}
s2_data = {m: s2_split(m) for m in MODELS}
paretos  = {m: baseline_pareto(m) for m in MODELS}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))


# ══════════════════════════════════════════════════════════════════════════════
# Panel (a): Pareto scatter — all DDA splits + S1 and S2 highlighted
# ══════════════════════════════════════════════════════════════════════════════
X_MAX = 6.5

for model in MODELS:
    col = COLORS[model]
    par = paretos[model]
    dda = COMBINED[COMBINED['Model'] == model]
    s1_row, s1_gain       = s1_data[model]
    s2_row, s2_1s_met     = s2_data[model]

    # Static Pareto (faint)
    vis = par[par['TTFT'] <= X_MAX]
    ax1.plot(vis['TTFT'], vis['Tput'], color=col, linewidth=1.0, alpha=0.35, zorder=2)
    ax1.scatter(vis['TTFT'], vis['Tput'], color=col, marker=MARKERS[model],
                s=50, edgecolors='black', linewidths=0.6, alpha=0.45, zorder=3)

    # All DDA splits — small hollow stars
    for _, row in dda.iterrows():
        if row['TTFT_short'] > X_MAX: continue
        ax1.scatter(row['TTFT_short'], row['system_tput'],
                    marker='*', s=180, facecolors='white', edgecolors=col,
                    linewidths=1.4, zorder=4)

    # S1 point — filled star (same as strategy1 figure)
    if s1_row['TTFT_short'] <= X_MAX:
        ax1.scatter(s1_row['TTFT_short'], s1_row['system_tput'],
                    marker='*', s=500, facecolors=col, edgecolors='black',
                    linewidths=1.8, zorder=8, label=f'S1 ({model.replace("Llama2-","L2-")})')

    # S2 point — filled diamond
    if s2_row['TTFT_short'] <= X_MAX:
        ax1.scatter(s2_row['TTFT_short'], s2_row['system_tput'],
                    marker='D', s=180, facecolors=col, edgecolors='black',
                    linewidths=1.8, zorder=9)
        # Connect S1 → S2 with arrow (shows the trade-off)
        if s1_row['N_L'] != s2_row['N_L']:
            ax1.annotate('',
                         xy=(s2_row['TTFT_short'], s2_row['system_tput']),
                         xytext=(s1_row['TTFT_short'], s1_row['system_tput']),
                         arrowprops=dict(arrowstyle='->', color=col, lw=1.8,
                                         linestyle='dashed'))
            # # Label delta
            # mid_ttft = (s1_row['TTFT_short'] + s2_row['TTFT_short']) / 2
            # mid_tput = (s1_row['system_tput']  + s2_row['system_tput'])  / 2
            # delta_t  = (s2_row['system_tput'] - s1_row['system_tput']) / s1_row['system_tput'] * 100
            # ax1.text(mid_ttft + 0.08, mid_tput,
            #          f'{delta_t:+.0f}%\ntput', fontsize=8, color=col,
            #          fontweight='bold', va='center')

# SLO lines
ax1.axvline(x=SLO_INTER, color='red',     linestyle='--', linewidth=1.8,
            label=f'{SLO_INTER}s interactive SLO')
ax1.axvline(x=SLO_BATCH, color='darkred', linestyle=':',  linewidth=1.5,
            label=f'{SLO_BATCH}s batch SLO')
ax1.axvspan(0, SLO_INTER, alpha=0.06, color='green')

legend_els = [
    mlines.Line2D([0],[0], marker=MARKERS[m], color=COLORS[m], linestyle='None',
                  ms=8, markeredgecolor='black', label=m.replace('Llama2-','L2-'))
    for m in MODELS
] + [
    mlines.Line2D([0],[0], marker='*',  color='gray', ms=16,  linestyle='None',
                  markeredgecolor='black', markerfacecolor='gray',
                  label='S1: best-gain split (★ filled)'),
    mlines.Line2D([0],[0], marker='D',  color='gray', ms=10,  linestyle='None',
                  markeredgecolor='black', markerfacecolor='gray',
                  label='S2: SLO-optimal split (◆ filled)'),
    mlines.Line2D([0],[0], marker='*',  color='white', ms=14, linestyle='None',
                  markeredgecolor='black', label='Other DDA splits (★ hollow)'),
    mlines.Line2D([0],[0], color='red',     linestyle='--', label=f'{SLO_INTER}s interactive SLO'),
    mlines.Line2D([0],[0], color='darkred', linestyle=':',  label=f'{SLO_BATCH}s batch SLO'),
]
ax1.legend(handles=legend_els, fontsize=7.5, loc='upper right', ncol=2)
ax1.set_xlabel('Short-Request TTFT (s)', fontsize=12)
ax1.set_ylabel('System Throughput (tok/s)', fontsize=12)
ax1.set_title('(a) S1 vs S2 — Pareto Scatter [0–6s]\n',
              fontsize=10.5)
ax1.set_xlim(0, X_MAX)
ax1.set_ylim(0, SIM_BASE['Throughput (tokens/s)'].max() * 1.12)
ax1.grid(linestyle='--', alpha=0.5)
ax1.tick_params(labelsize=11)


# ══════════════════════════════════════════════════════════════════════════════
# Panel (b): Bar chart — S2 vs S1 vs static per model
# ══════════════════════════════════════════════════════════════════════════════
x     = np.arange(len(MODELS))
width = 0.25

static_tputs, s1_tputs, s2_tputs = [], [], []
static_ttfts, s1_ttfts, s2_ttfts = [], [], []
s1_slos, s2_slos = [], []

for model in MODELS:
    par = paretos[model]
    s1_row, _       = s1_data[model]
    s2_row, s2_1s   = s2_data[model]

    bs_s1 = best_static_lte(par, s1_row['TTFT_short'])
    bs_s2 = best_static_lte(par, s2_row['TTFT_short'])

    static_tputs.append(float(bs_s1['Tput']))
    static_ttfts.append(float(bs_s1['TTFT']))
    s1_tputs.append(s1_row['system_tput'])
    s1_ttfts.append(s1_row['TTFT_short'])
    s2_tputs.append(s2_row['system_tput'])
    s2_ttfts.append(s2_row['TTFT_short'])
    s1_slos.append('1s ✓' if s1_row['SLO_1s_short'] else
                   ('5s ✓' if s1_row['SLO_5s_short'] else '✗'))
    s2_slos.append(f'{s2_1s} ✓')

bars_stat = ax2.bar(x - width, static_tputs, width,
                    label='Balanced static', color='lightsalmon', edgecolor='black')
bars_s1   = ax2.bar(x,          s1_tputs,    width,
                    label='S1: best-gain split', color='steelblue',
                    edgecolor='black', alpha=0.88)
bars_s2   = ax2.bar(x + width,  s2_tputs,    width,
                    label='S2: SLO-optimal split', color='seagreen',
                    edgecolor='black', alpha=0.88)

y_top = max(s1_tputs + s2_tputs) * 1.38

# Annotate static
for bar, tput, ttft in zip(bars_stat, static_tputs, static_ttfts):
    ax2.text(bar.get_x() + bar.get_width()/2, tput + y_top*0.01,
             f'{tput:.0f}\n({ttft:.2f}s)',
             ha='center', va='bottom', fontsize=7, color='dimgray')

# Annotate S1
for bar, tput, ttft, slo in zip(bars_s1, s1_tputs, s1_ttfts, s1_slos):
    ax2.text(bar.get_x() + bar.get_width()/2, tput * 0.45,
             f'SLO:\n{slo}', ha='center', va='center',
             fontsize=8, color='white', fontweight='bold')
    ax2.text(bar.get_x() + bar.get_width()/2, tput + y_top*0.01,
             f'{tput:.0f}\n({ttft:.2f}s)',
             ha='center', va='bottom', fontsize=7, fontweight='bold')

# Annotate S2
for bar, tput, ttft, slo in zip(bars_s2, s2_tputs, s2_ttfts, s2_slos):
    ax2.text(bar.get_x() + bar.get_width()/2, tput * 0.45,
             f'SLO:\n{slo}', ha='center', va='center',
             fontsize=8, color='white', fontweight='bold')
    ax2.text(bar.get_x() + bar.get_width()/2, tput + y_top*0.01,
             f'{tput:.0f}\n({ttft:.2f}s)',
             ha='center', va='bottom', fontsize=7, fontweight='bold')

# Gain multipliers S2/static
for i, (s2, stat, col) in enumerate(zip(s2_tputs, static_tputs, [COLORS[m] for m in MODELS])):
    gain = s2 / stat
    ax2.text(x[i] + width, max(s2_tputs)*1.28,
             f'{gain:.1f}×', ha='center', fontsize=9, color=col, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(MODEL_LABELS, fontsize=11)
ax2.set_ylabel('System Throughput (tok/s)', fontsize=12)
ax2.set_title('(b) System Throughput: S2 vs S1 vs Static\n'
              f'Bimodal SLO: interactive=1s (±{int(SLO_TOL*100)}% tolerance), batch=5s\n',
              fontsize=10.5)
ax2.legend(fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.tick_params(axis='y', labelsize=11)
ax2.set_ylim(0, y_top)

plt.tight_layout()

os.makedirs('figures', exist_ok=True)
os.makedirs('figure_source_data', exist_ok=True)
plt.savefig('figures/figure_dda_strategy2_tol.pdf', bbox_inches='tight')
print("Saved: figures/figure_dda_strategy2_tol.pdf")

print("\nS1 vs S2 summary:")
for model in MODELS:
    s1_row, s1_g  = s1_data[model]
    s2_row, s2_1s = s2_data[model]
    par = paretos[model]
    bs  = best_static_lte(par, s2_row['TTFT_short'])
    print(f"  {model}:")
    print(f"    S1: N_L={int(s1_row['N_L'])} tput={s1_row['system_tput']:.0f} "
          f"TTFT={s1_row['TTFT_short']:.3f}s "
          f"SLO={'1s✓' if s1_row['SLO_1s_short'] else ('5s✓' if s1_row['SLO_5s_short'] else '✗')}")
    print(f"    S2: N_L={int(s2_row['N_L'])} tput={s2_row['system_tput']:.0f} "
          f"TTFT={s2_row['TTFT_short']:.3f}s "
          f"SLO={s2_1s}✓ "
          f"vs static {bs['Tput']:.0f} → {s2_row['system_tput']/bs['Tput']:.1f}×")
