"""
figure_dda_longseq_sweep.py — Sweep all DDA splits for long-request performance.

Asks: for seqlen > 512, which (N_L, N_T) split maximises Pool T throughput?
Answer: largest N_T at each breakpoint (channels_per_block changes).

Panel (a): Pareto scatter (Pool T TTFT @ seqlen=1024 vs Pool T Tput @ seqlen=1024)
           for ALL simulated DDA splits per model.
           Filled star = best long-seq split (max Pool T tput).
           Static balanced config shown as ◆ for reference.
           Mirrors panel (a) of figure_dda_strategy1 but for long requests.

Panel (b): Throughput vs seqlen (seqlen > 512) comparing:
           - Balanced static (same as strategy1)
           - Best SHORT-seq split Pool T (from strategy1)
           - Best LONG-seq split Pool T (this figure)

Panel (c): TTFT vs seqlen (same three configs).

Run from repo root: python3 figure_scripts/figure_dda_longseq_sweep.py
"""

import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SIM_BASE = pd.read_csv('../../cent_simulation/simulation_results.csv')
RESULTS  = 'results'

T_SHORT   = 512
REF_SEQLEN = 1024   # reference seqlen for panel (a) scatter
MODELS    = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
COLORS    = {'Llama2-7B': 'steelblue', 'Llama2-13B': 'darkorange', 'Llama2-70B': 'forestgreen'}
MARKERS   = {'Llama2-7B': 'o', 'Llama2-13B': 's', 'Llama2-70B': '^'}
LAYERS    = {'Llama2-7B': 32, 'Llama2-13B': 40, 'Llama2-70B': 80}
TOTAL_DEV = {'Llama2-7B': 8,  'Llama2-13B': 20, 'Llama2-70B': 32}
STATIC_CMP  = {'Llama2-7B': (4,2), 'Llama2-13B': (2,10), 'Llama2-70B': (2,16)}
SHORT_SPLIT = {'Llama2-7B': (2,6), 'Llama2-13B': (6,14), 'Llama2-70B': (12,20)}

# Best long-seq split = max Pool T tput (may differ from best short-seq split)
# Computed below; stored here after first pass.
LONG_SPLIT  = {}


def pool_T_at_seqlen(dfT, lay, s):
    """Pool T tput-optimal (PP=lay,TP=1): TTFT and Tput at seqlen s."""
    grp = dfT[(dfT['Pipeline parallelism']==lay) & (dfT['Tensor parallelism']==1)]
    sub = grp[grp['Sequence length'] <= s]
    row = grp[grp['Sequence length'] == s]
    if sub.empty or row.empty:
        return None, None
    ttft = sub['Token latency (ms)'].mean() * s / 1000
    tput = float(row['Throughput (tokens/s)'].iloc[0])
    return ttft, tput


def pool_T_series(dfT, lay, seqlen_min=T_SHORT):
    """Running TTFT and per-step Tput for Pool T (seqlen >= seqlen_min)."""
    grp = dfT[(dfT['Pipeline parallelism']==lay) & (dfT['Tensor parallelism']==1)]
    grp = grp.sort_values('Sequence length')
    seqlens, ttfts, tputs = [], [], []
    for s in sorted(grp['Sequence length'].unique()):
        if s < seqlen_min: continue
        sub = grp[grp['Sequence length'] <= s]
        row = grp[grp['Sequence length'] == s]
        seqlens.append(s)
        ttfts.append(sub['Token latency (ms)'].mean() * s / 1000)
        tputs.append(float(row['Throughput (tokens/s)'].iloc[0]))
    return np.array(seqlens), np.array(ttfts), np.array(tputs)


def static_series(b, pp, tp, seqlen_min=T_SHORT):
    grp = b[(b['Pipeline parallelism']==pp) & (b['Tensor parallelism']==tp)]
    grp = grp.sort_values('Sequence length')
    seqlens, ttfts, tputs = [], [], []
    for s in sorted(grp['Sequence length'].unique()):
        if s < seqlen_min: continue
        sub = grp[grp['Sequence length'] <= s]
        row = grp[grp['Sequence length'] == s]
        seqlens.append(s)
        ttfts.append(sub['Token latency (ms)'].mean() * s / 1000)
        tputs.append(float(row['Throughput (tokens/s)'].iloc[0]))
    return np.array(seqlens), np.array(ttfts), np.array(tputs)


# ── Pre-compute panel (a) data + find best long-seq split ─────────────────────
scatter_data = {}   # model → list of {NT, ttft, tput}

for model in MODELS:
    lay = LAYERS[model]
    pts = []
    for f in sorted(glob.glob(f'{RESULTS}/sim_{model}_poolT_*.csv')):
        NT  = int(f.split('_poolT_')[1].replace('.csv',''))
        dfT = pd.read_csv(f)
        ttft, tput = pool_T_at_seqlen(dfT, lay, REF_SEQLEN)
        if ttft is None: continue
        pts.append({'NT': NT, 'ttft': ttft, 'tput': tput})
    scatter_data[model] = pts
    # best long-seq split = max tput (ties broken by max NT = most devices for Pool T)
    best = max(pts, key=lambda x: (x['tput'], x['NT']))
    LONG_SPLIT[model] = (TOTAL_DEV[model] - best['NT'], best['NT'])

print("Best long-seq splits:", LONG_SPLIT)
print("Best short-seq splits:", SHORT_SPLIT)


# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 5.5))
ax1 = fig.add_subplot(1, 3, 1)   # panel (a): pareto scatter
ax2 = fig.add_subplot(1, 3, 2)   # panel (b): throughput vs seqlen
ax3 = fig.add_subplot(1, 3, 3)   # panel (c): TTFT vs seqlen


# ══════════════════════════════════════════════════════════════════════════════
# Panel (a): Pareto scatter — Pool T TTFT vs Tput at REF_SEQLEN
# ══════════════════════════════════════════════════════════════════════════════
for model in MODELS:
    col = COLORS[model]
    mk  = MARKERS[model]
    lay = LAYERS[model]
    pts = scatter_data[model]
    NL_long, NT_long = LONG_SPLIT[model]
    NL_short, NT_short = SHORT_SPLIT[model]

    # Group pts by (rounded ttft, rounded tput) to detect overlaps
    from collections import defaultdict
    groups = defaultdict(list)
    for p in pts:
        key = (round(p['ttft'], 1), round(p['tput'], 0))
        groups[key].append(p)

    # All DDA splits — hollow markers; stagger annotations within overlapping groups
    for p in pts:
        is_long  = (p['NT'] == NT_long)
        face = col if is_long else 'white'
        sz   = 300 if is_long else 140
        zord = 8   if is_long else 4
        ax1.scatter(p['ttft'], p['tput'],
                    marker='*', s=sz, facecolors=face, edgecolors=col,
                    linewidths=2.0 if is_long else 1.5, zorder=zord)

    # Annotate — stagger vertically within overlapping groups
    for key, group in groups.items():
        n = len(group)
        for idx, p in enumerate(sorted(group, key=lambda x: x['NT'])):
            is_long = (p['NT'] == NT_long)
            # vertical offsets: spread labels -8*(n//2) to +8*(n//2)
            y_off = ((idx - (n - 1) / 2) * 10) - 5
            x_off = 10
            ax1.annotate(f'N_T={p["NT"]}', (p['ttft'], p['tput']),
                         xytext=(x_off, y_off), textcoords='offset points',
                         fontsize=7, color=col, alpha=0.9,
                         fontweight='bold' if is_long else 'normal')

    # Highlight short-seq split with a ring
    for p in pts:
        if p['NT'] == NT_short and NT_short != NT_long:
            ax1.scatter(p['ttft'], p['tput'],
                        marker='*', s=250, facecolors='white',
                        edgecolors=col, linewidths=2.5, zorder=7)

    # Balanced static reference ◆
    b = SIM_BASE[SIM_BASE['Model'] == model]
    pp_s, tp_s = STATIC_CMP[model]
    ttft_s, tput_s = pool_T_at_seqlen(
        b[(b['Pipeline parallelism']==pp_s) & (b['Tensor parallelism']==tp_s)]
        .rename(columns={'Pipeline parallelism': 'Pipeline parallelism',
                         'Tensor parallelism':   'Tensor parallelism'}),
        pp_s, REF_SEQLEN)
    # compute manually
    sub_s = b[(b['Pipeline parallelism']==pp_s) & (b['Tensor parallelism']==tp_s)]
    sub_s_fil = sub_s[sub_s['Sequence length'] <= REF_SEQLEN]
    row_s = sub_s[sub_s['Sequence length'] == REF_SEQLEN]
    if not sub_s_fil.empty and not row_s.empty:
        ttft_s = sub_s_fil['Token latency (ms)'].mean() * REF_SEQLEN / 1000
        tput_s = float(row_s['Throughput (tokens/s)'].iloc[0])
        ax1.scatter(ttft_s, tput_s, marker='D', s=100, color=col,
                    edgecolors='black', linewidths=1.3, zorder=9, alpha=0.7)
        ax1.annotate(f'Static\nPP={pp_s},TP={tp_s}',
                     (ttft_s, tput_s), xytext=(5, -15),
                     textcoords='offset points', fontsize=6.5, color=col, alpha=0.8)
0
ax1.set_xlabel(f'Pool T TTFT at seqlen={REF_SEQLEN} (s)', fontsize=10.5)
ax1.set_ylabel('Pool T Throughput (tok/s)', fontsize=10.5)
ax1.set_title(f'(a) DDA Split Sweep — Long-Request Pareto\n'
              f'(seqlen={REF_SEQLEN}, best long-seq, best short-seq, static)',
              fontsize=10)
ax1.grid(linestyle='--', alpha=0.5)
ax1.tick_params(labelsize=10)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

# Per-model color patches in legend
legend_els = [
    mlines.Line2D([0],[0], marker=MARKERS[m], color=COLORS[m], linestyle='None',
                  ms=8, markeredgecolor='black',
                  label=m.replace('Llama2-','L2-'))
    for m in MODELS
] + [
    mlines.Line2D([0],[0], marker='*', color='gray', linestyle='None',
                  markerfacecolor='gray', markeredgecolor='black', ms=14,
                  label='best long-seq split'),
    mlines.Line2D([0],[0], marker='*', color='white', linestyle='None', ms=12,
                  markeredgecolor='black', markeredgewidth=1.8,
                  label='best short-seq split'),
    mlines.Line2D([0],[0], marker='D', color='gray', linestyle='None', ms=8,
                  markeredgecolor='black', alpha=0.7, label='balanced static'),
]
ax1.legend(handles=legend_els, fontsize=7.5, loc='upper right')


# ══════════════════════════════════════════════════════════════════════════════
# Panels (b)+(c): Throughput and TTFT vs seqlen
# ══════════════════════════════════════════════════════════════════════════════
LSTYLE = {'Llama2-7B': '-', 'Llama2-13B': '--', 'Llama2-70B': (0,(3,1,1,1))}
COL_STATIC    = 'steelblue'
COL_SHORT_DDA = 'seagreen'
COL_LONG_DDA  = '#cc0000'
LW = 2.0

for model in MODELS:
    lay = LAYERS[model]
    ls  = LSTYLE[model]
    b   = SIM_BASE[SIM_BASE['Model'] == model]
    pp_s, tp_s         = STATIC_CMP[model]
    NL_short, NT_short = SHORT_SPLIT[model]
    NL_long,  NT_long  = LONG_SPLIT[model]
    short_str = model.replace('Llama2-','L2-')

    dfT_short = pd.read_csv(f'{RESULTS}/sim_{model}_poolT_{NT_short}.csv')
    dfT_long  = pd.read_csv(f'{RESULTS}/sim_{model}_poolT_{NT_long}.csv')

    # Balanced static
    ss, ttft_ss, tput_ss = static_series(b, pp_s, tp_s)
    ax2.plot(ss, tput_ss, color=COL_STATIC,    linestyle=ls, linewidth=LW,
             label=f'Static PP={pp_s},TP={tp_s} ({short_str})')
    ax3.plot(ss, ttft_ss, color=COL_STATIC,    linestyle=ls, linewidth=LW,
             label=f'Static PP={pp_s},TP={tp_s} ({short_str})')

    # Best short-seq DDA split Pool T
    ds, ttft_ds, tput_ds = pool_T_series(dfT_short, lay)
    ax2.plot(ds, tput_ds, color=COL_SHORT_DDA, linestyle=ls, linewidth=LW,
             label=f'Pool T short-split N_T={NT_short} ({short_str})')
    ax3.plot(ds, ttft_ds, color=COL_SHORT_DDA, linestyle=ls, linewidth=LW,
             label=f'Pool T short-split N_T={NT_short} ({short_str})')

    # Best long-seq DDA split Pool T
    dl, ttft_dl, tput_dl = pool_T_series(dfT_long, lay)
    ax2.plot(dl, tput_dl, color=COL_LONG_DDA,  linestyle=ls, linewidth=LW+0.5,
             label=f'Pool T long-split N_T={NT_long} ({short_str})')
    ax3.plot(dl, ttft_dl, color=COL_LONG_DDA,  linestyle=ls, linewidth=LW+0.5,
             label=f'Pool T long-split N_T={NT_long} ({short_str})')

    # Shade: long-split gain over short-split
    min_len = min(len(ds), len(dl))
    ax2.fill_between(dl[:min_len], tput_ds[:min_len], tput_dl[:min_len],
                     alpha=0.10, color=COL_LONG_DDA)

ax2.set_xlabel('Sequence Length (tokens)', fontsize=11)
ax2.set_ylabel('Throughput (tok/s)', fontsize=11)
ax2.set_title('(b) Throughput vs Seqlen — Long Requests\n'
              'Shaded = gain of best long-seq split over short-seq split Pool T',
              fontsize=10)
ax2.set_xlim(T_SHORT, 4096)
ax2.set_ylim(bottom=0)
ax2.grid(linestyle='--', alpha=0.4)
ax2.tick_params(labelsize=10)

ax3.set_yscale('log')
ax3.set_xlabel('Sequence Length (tokens)', fontsize=11)
ax3.set_ylabel('TTFT (s, log scale)', fontsize=11)
ax3.set_title('(c) TTFT vs Seqlen — Long Requests\n'
              'Long-seq split has lower TTFT (more Pool T devices)',
              fontsize=10)
ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=1.2, alpha=0.8, label='1s SLO')
ax3.axhline(y=5.0, color='olive', linestyle=':',  linewidth=1.2, alpha=0.8, label='5s SLO')
ax3.set_xlim(T_SHORT, 4096)
ax3.grid(linestyle='--', alpha=0.4, which='both')
ax3.tick_params(labelsize=10)

# Shared legend for b + c
legend_bc = [
    mlines.Line2D([0],[0], color=COL_STATIC,    linewidth=2,   label='Balanced static (same as strategy1)'),
    mlines.Line2D([0],[0], color=COL_SHORT_DDA, linewidth=2,   label='Pool T — best SHORT-seq split'),
    mlines.Line2D([0],[0], color=COL_LONG_DDA,  linewidth=2.5, label='Pool T — best LONG-seq split'),
    mlines.Line2D([0],[0], color='gray', linewidth=2, linestyle='-',           label='L2-7B  (short N_T=6,  long N_T=6)'),
    mlines.Line2D([0],[0], color='gray', linewidth=2, linestyle='--',          label='L2-13B (short N_T=14, long N_T=18)'),
    mlines.Line2D([0],[0], color='gray', linewidth=2, linestyle=(0,(3,1,1,1)), label='L2-70B (short N_T=20, long N_T=28)'),
    mlines.Line2D([0],[0], color='green', linewidth=1.2, linestyle='--', label='1s SLO'),
    mlines.Line2D([0],[0], color='olive', linewidth=1.2, linestyle=':',  label='5s SLO'),
]
fig.legend(handles=legend_bc, fontsize=8, loc='lower center',
           ncol=4, bbox_to_anchor=(0.65, -0.12), frameon=True)

plt.suptitle('DDA Strategy 1 — Long-Request Split Sweep\n'
             'Best short-seq split vs best long-seq split vs balanced static',
             fontsize=11)
plt.tight_layout(rect=[0, 0.08, 1, 1])

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure_dda_longseq.pdf', bbox_inches='tight')
print("Saved: figures/figure_dda_longseq.pdf")

print(f"\nKey comparison at seqlen={REF_SEQLEN}:")
for model in MODELS:
    lay = LAYERS[model]
    NL_short, NT_short = SHORT_SPLIT[model]
    NL_long,  NT_long  = LONG_SPLIT[model]
    pp_s, tp_s = STATIC_CMP[model]
    b    = SIM_BASE[SIM_BASE['Model']==model]
    dfTs = pd.read_csv(f'{RESULTS}/sim_{model}_poolT_{NT_short}.csv')
    dfTl = pd.read_csv(f'{RESULTS}/sim_{model}_poolT_{NT_long}.csv')
    ttft_s, tput_s = pool_T_at_seqlen(dfTs, lay, REF_SEQLEN)
    ttft_l, tput_l = pool_T_at_seqlen(dfTl, lay, REF_SEQLEN)
    sub_stat = b[(b['Pipeline parallelism']==pp_s)&(b['Tensor parallelism']==tp_s)&(b['Sequence length']<=REF_SEQLEN)]
    row_stat = b[(b['Pipeline parallelism']==pp_s)&(b['Tensor parallelism']==tp_s)&(b['Sequence length']==REF_SEQLEN)]
    ttft_st = sub_stat['Token latency (ms)'].mean()*REF_SEQLEN/1000
    tput_st = float(row_stat['Throughput (tokens/s)'].iloc[0])
    print(f"  {model}: static={tput_st:.0f}tok/s@{ttft_st:.1f}s | "
          f"short-split(N_T={NT_short})={tput_s:.0f}@{ttft_s:.1f}s | "
          f"long-split(N_T={NT_long})={tput_l:.0f}@{ttft_l:.1f}s")
