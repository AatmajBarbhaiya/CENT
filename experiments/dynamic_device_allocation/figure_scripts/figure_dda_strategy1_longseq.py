"""
figure_dda_strategy1_longseq.py — DDA Strategy 1: short-split vs long-split comparison.

Models: Llama2-13B and Llama2-70B only (7B has only one valid split — no comparison possible).

Splits compared:
  Short split = best-gain split from figure_dda_strategy1_shortseq panel (d):
    13B: N_L=6,  N_T=14  Pool L: PP=1,TP=6   Pool T: PP=40,TP=1  sys_tput=3020
    70B: N_L=12, N_T=20  Pool L: PP=1,TP=12  Pool T: PP=80,TP=1  sys_tput=1057

  Long split = max Pool T tput (best for long-seq workloads):
    13B: N_L=2,  N_T=18  Pool L: PP=1,TP=2   Pool T: PP=40,TP=1  sys_tput=2927
    70B: N_L=4,  N_T=28  Pool L: PP=1,TP=4   Pool T: PP=80,TP=1  sys_tput=1268

Panel (a): Adaptive total throughput vs seqlen (ONE curve per model):
           seqlen ≤ 512: short split → Pool_L_short_tput(s) + Tput_T_short_const
           seqlen > 512: long split  → Tput_L_long_const + Pool_T_long_tput(s)
Panel (b): Adaptive Pool_L TTFT vs seqlen (ONE curve per model):
           seqlen ≤ 512: Pool_L_short TTFT(s)
           seqlen > 512: Pool_L_long TTFT(s)
Panel (c): Bar chart — system_tput (Pool_L + Pool_T) per split vs balanced static.
           Annotated with Pool_L TTFT@512 for short and long splits.

Run from repo root: python3 figure_scripts/figure_dda_strategy1_longseq.py
"""

import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SIM_BASE = pd.read_csv('../../cent_simulation/simulation_results.csv')
COMBINED = pd.read_csv('results/combined_metrics.csv')
RESULTS  = 'results'

T_SHORT    = 512
MODELS     = ['Llama2-13B', 'Llama2-70B']
COLORS     = {'Llama2-13B': 'darkorange', 'Llama2-70B': 'forestgreen'}
LAYERS     = {'Llama2-13B': 40, 'Llama2-70B': 80}
STATIC_CMP = {'Llama2-13B': (2, 10), 'Llama2-70B': (2, 16)}

SPLITS = {
    'Llama2-13B': {
        'short': {'N_L': 6,  'N_T': 14, 'PP_L': 1, 'TP_L': 6,  'PP_T': 40},
        'long':  {'N_L': 2,  'N_T': 18, 'PP_L': 1, 'TP_L': 2,  'PP_T': 40},
    },
    'Llama2-70B': {
        'short': {'N_L': 12, 'N_T': 20, 'PP_L': 1, 'TP_L': 12, 'PP_T': 80},
        'long':  {'N_L': 4,  'N_T': 28, 'PP_L': 1, 'TP_L': 4,  'PP_T': 80},
    },
}


def running_ttft_series(df, pp, tp, seqlen_max=4096):
    grp = df[(df['Pipeline parallelism']==pp) & (df['Tensor parallelism']==tp)]
    grp = grp.sort_values('Sequence length')
    seqlens, ttfts = [], []
    for s in sorted(grp['Sequence length'].unique()):
        if s > seqlen_max: continue
        sub = grp[grp['Sequence length'] <= s]
        seqlens.append(s)
        ttfts.append(sub['Token latency (ms)'].mean() * s / 1000)
    return np.array(seqlens), np.array(ttfts)


def tput_series(df, pp, tp, seqlen_min=None, seqlen_max=4096):
    grp = df[(df['Pipeline parallelism']==pp) & (df['Tensor parallelism']==tp)]
    grp = grp.sort_values('Sequence length')
    seqlens, tputs = [], []
    for s in sorted(grp['Sequence length'].unique()):
        if seqlen_min and s < seqlen_min: continue
        if s > seqlen_max: continue
        seqlens.append(s)
        tputs.append(float(grp[grp['Sequence length']==s]['Throughput (tokens/s)'].iloc[0]))
    return np.array(seqlens), np.array(tputs)


# Pre-load pool CSVs
pool_data = {}
for model in MODELS:
    for kind in ['poolL', 'poolT']:
        for f in glob.glob(f'{RESULTS}/sim_{model}_{kind}_*.csv'):
            N = int(f.split(f'_{kind}_')[1].replace('.csv', ''))
            pool_data[(model, kind, N)] = pd.read_csv(f)

# Get metrics from combined
cm = {}
for model in MODELS:
    for split_name, sp in SPLITS[model].items():
        row = COMBINED[(COMBINED['Model']==model) & (COMBINED['N_L']==sp['N_L']) &
                       (COMBINED['N_T']==sp['N_T'])]
        if not row.empty:
            cm[(model, split_name)] = {
                'sys_tput': float(row['system_tput'].iloc[0]),
                'Tput_L':   float(row['Tput_L'].iloc[0]),
                'Tput_T':   float(row['Tput_T'].iloc[0]),
                'TTFT_L':   float(row['TTFT_short'].iloc[0]),
            }

# ── Figure: 3-column layout ───────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 5.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, left=0.06, right=0.99,
                        bottom=0.18, top=0.92, wspace=0.38,
                        width_ratios=[1.2, 1.2, 1])
ax_a = fig.add_subplot(gs[0, 0])   # throughput vs seqlen
ax_b = fig.add_subplot(gs[0, 1])   # Pool_L TTFT vs seqlen
ax_c = fig.add_subplot(gs[0, 2])   # bar chart

leg_a = {}   # handles for panel (a) legend
leg_b = {}   # handles for panel (b) legend

# ══════════════════════════════════════════════════════════════════════════════
# Panel (a): Adaptive total throughput vs seqlen (ONE curve per model)
# seqlen ≤ 512: short split → Pool_L_short_tput(s) + Tput_T_short_const
# seqlen > 512: long split  → Tput_L_long_const + Pool_T_long_tput(s)
# ══════════════════════════════════════════════════════════════════════════════
for model in MODELS:
    col      = COLORS[model]
    sp_short = SPLITS[model]['short']
    sp_long  = SPLITS[model]['long']
    df_L_sh  = pool_data.get((model, 'poolL', sp_short['N_L']))
    df_T_lg  = pool_data.get((model, 'poolT', sp_long['N_T']))
    if df_L_sh is None or df_T_lg is None: continue

    Tput_T_sh_const = cm[(model, 'short')]['Tput_T']
    Tput_L_lg_const = cm[(model, 'long')]['Tput_L']

    sL, tL = tput_series(df_L_sh, sp_short['PP_L'], sp_short['TP_L'], seqlen_max=T_SHORT)
    sT, tT = tput_series(df_T_lg, sp_long['PP_T'], 1, seqlen_min=T_SHORT+1)
    s_all  = np.concatenate([sL, sT])
    t_all  = np.concatenate([tL + Tput_T_sh_const, tT + Tput_L_lg_const])

    short_lbl = model.replace('Llama2-', 'L2-')
    lbl = f'{short_lbl} DDA adaptive'
    h, = ax_a.plot(s_all, t_all, color=col, linestyle='-', linewidth=2.2, label=lbl)
    leg_a[lbl] = h

for model in MODELS:
    col = COLORS[model]
    b   = SIM_BASE[SIM_BASE['Model']==model]
    pp_s, tp_s = STATIC_CMP[model]
    grp = b[(b['Pipeline parallelism']==pp_s) &
             (b['Tensor parallelism']==tp_s)].sort_values('Sequence length')
    short_lbl = model.replace('Llama2-', 'L2-')
    lbl = f'{short_lbl} static PP={pp_s},TP={tp_s}'
    h, = ax_a.plot(grp['Sequence length'], grp['Throughput (tokens/s)'],
                   color=col, linestyle=':', linewidth=1.4, alpha=0.6, label=lbl)
    leg_a[lbl] = h

h_thresh_a = ax_a.axvline(x=T_SHORT, color='purple', linestyle='--', linewidth=1.6,
                           label=f'DDA threshold T={T_SHORT}')
leg_a[f'DDA threshold T={T_SHORT}'] = h_thresh_a

ax_a.set_xlabel('Sequence Length (tokens)', fontsize=11)
ax_a.set_ylabel('Total System Throughput (tok/s)', fontsize=11)
ax_a.set_title('(a) Total Throughput vs Seqlen\n'
               '≤512: short-split total  |  >512: long-split total', fontsize=10)
ax_a.set_xlim(0, 4096)
ax_a.set_ylim(bottom=0)
ax_a.grid(linestyle='--', alpha=0.4)
ax_a.tick_params(labelsize=10)
ax_a.legend(handles=list(leg_a.values()), labels=list(leg_a.keys()),
            loc='lower right', fontsize=7.5, framealpha=0.9)

# ══════════════════════════════════════════════════════════════════════════════
# Panel (b): Adaptive Pool_L TTFT vs seqlen (ONE curve per model)
# seqlen ≤ 512: short-split Pool_L TTFT(s)  — larger N_L, lower latency
# seqlen > 512: long-split  Pool_L TTFT(s)  — smaller N_L, higher latency
# ══════════════════════════════════════════════════════════════════════════════
for model in MODELS:
    col      = COLORS[model]
    sp_short = SPLITS[model]['short']
    sp_long  = SPLITS[model]['long']
    df_L_sh  = pool_data.get((model, 'poolL', sp_short['N_L']))
    df_L_lg  = pool_data.get((model, 'poolL', sp_long['N_L']))
    if df_L_sh is None or df_L_lg is None: continue

    sL, tL = running_ttft_series(df_L_sh, sp_short['PP_L'], sp_short['TP_L'], T_SHORT)
    sR, tR = running_ttft_series(df_L_lg, sp_long['PP_L'],  sp_long['TP_L'],  4096)
    mask = sR > T_SHORT
    sR, tR = sR[mask], tR[mask]

    s_all = np.concatenate([sL, sR])
    t_all = np.concatenate([tL, tR])

    short_lbl = model.replace('Llama2-', 'L2-')
    lbl = f'{short_lbl} Pool_L adaptive'
    h, = ax_b.plot(s_all, t_all, color=col, linestyle='-', linewidth=2.2, label=lbl)
    leg_b[lbl] = h

for model in MODELS:
    col = COLORS[model]
    b = SIM_BASE[SIM_BASE['Model'] == model]
    pp_s, tp_s = STATIC_CMP[model]
    sS, tS = running_ttft_series(b, pp_s, tp_s, seqlen_max=4096)
    short_lbl = model.replace('Llama2-', 'L2-')
    lbl = f'{short_lbl} static PP={pp_s},TP={tp_s}'
    h, = ax_b.plot(sS, tS, color=col, linestyle=':', linewidth=1.4, alpha=0.6, label=lbl)
    leg_b[lbl] = h

h_thr_b = ax_b.axvline(x=T_SHORT, color='purple', linestyle='--', linewidth=1.6,
                        label=f'DDA threshold T={T_SHORT}')
h_slo1  = ax_b.axhline(y=1.0, color='red',     linestyle='--', linewidth=1.2,
                        label='1s SLO', alpha=0.8)
h_slo5  = ax_b.axhline(y=5.0, color='darkred', linestyle=':',  linewidth=1.2,
                        label='5s SLO', alpha=0.8)
leg_b[f'DDA threshold T={T_SHORT}'] = h_thr_b
leg_b['1s SLO'] = h_slo1
leg_b['5s SLO'] = h_slo5

ax_b.set_yscale('log')
ax_b.set_xlabel('Sequence Length (tokens)', fontsize=11)
ax_b.set_ylabel('Pool_L TTFT (s, log scale)', fontsize=11)
ax_b.set_title('(b) Pool_L TTFT vs Seqlen\n'
               '≤512: short-split Pool_L  |  >512: long-split Pool_L',
               fontsize=10)
ax_b.set_xlim(0, 4096)
ax_b.grid(linestyle='--', alpha=0.4, which='both')
ax_b.tick_params(labelsize=10)
ax_b.legend(handles=list(leg_b.values()), labels=list(leg_b.keys()),
            loc='lower right', fontsize=7.5, framealpha=0.9)

# ══════════════════════════════════════════════════════════════════════════════
# Panel (c): Bar chart — short-split vs long-split vs balanced static
# Pool_L TTFT@T_SHORT annotated above short and long split bars.
# ══════════════════════════════════════════════════════════════════════════════
model_list = MODELS
x     = np.arange(len(model_list))
width = 0.22

static_tputs, short_tputs, long_tputs = [], [], []
short_ttfts, long_ttfts = [], []
for model in model_list:
    b = SIM_BASE[SIM_BASE['Model']==model]
    pp_s, tp_s = STATIC_CMP[model]
    static_tputs.append(b[(b['Pipeline parallelism']==pp_s) &
                           (b['Tensor parallelism']==tp_s)]['Throughput (tokens/s)'].mean())
    short_tputs.append(cm[(model, 'short')]['sys_tput'])
    long_tputs.append(cm[(model, 'long')]['sys_tput'])
    short_ttfts.append(cm[(model, 'short')]['TTFT_L'])
    long_ttfts.append(cm[(model, 'long')]['TTFT_L'])

bars_stat  = ax_c.bar(x - width, static_tputs, width, label='Balanced static',
                      color='lightsalmon', edgecolor='black')
bars_short = ax_c.bar(x,          short_tputs, width, label='Short-seq split',
                      color=[COLORS[m] for m in model_list], edgecolor='black', alpha=0.88)
bars_long  = ax_c.bar(x + width,  long_tputs,  width, label='Long-seq split',
                      color=[COLORS[m] for m in model_list], edgecolor='black',
                      alpha=0.55, hatch='///')

y_top = max(short_tputs + long_tputs) * 1.38
# Throughput labels
for bars, vals in [(bars_stat, static_tputs), (bars_short, short_tputs), (bars_long, long_tputs)]:
    for bar, val in zip(bars, vals):
        ax_c.text(bar.get_x() + bar.get_width()/2, val + y_top*0.01,
                  f'{val:.0f}', ha='center', va='bottom', fontsize=8)

# Pool_L TTFT annotations above short and long bars
for bar, ttft in zip(bars_short, short_ttfts):
    ax_c.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + y_top*0.09,
              f'TTFT\n{ttft:.2f}s', ha='center', va='bottom',
              fontsize=7.5, fontweight='bold', color='black')

for bar, ttft in zip(bars_long, long_ttfts):
    ax_c.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + y_top*0.09,
              f'TTFT\n{ttft:.2f}s', ha='center', va='bottom',
              fontsize=7.5, color='black', fontweight='bold')

# Multi-line tick labels embedding split configs
tick_labels = []
for model in model_list:
    sp_s   = SPLITS[model]['short']
    sp_l   = SPLITS[model]['long']
    devs   = 20 if '13B' in model else 32
    mlabel = 'L2-13B' if '13B' in model else 'L2-70B'
    tick_labels.append(
        f'{mlabel} ({devs} dev)\n'
        f'Sh: N_L={sp_s["N_L"]}, N_T={sp_s["N_T"]}\n'
        f'Lg: N_L={sp_l["N_L"]}, N_T={sp_l["N_T"]}'
    )
ax_c.set_xticks(x)
ax_c.set_xticklabels(tick_labels, fontsize=8.5)
ax_c.set_ylabel('System Throughput (tok/s)', fontsize=11)
ax_c.set_title('(c) System Throughput: Pool_L + Pool_T\n'
               'Short-split vs Long-split vs Balanced static', fontsize=10.5)
ax_c.legend(fontsize=9)
ax_c.set_ylim(0, y_top)
ax_c.grid(axis='y', linestyle='--', alpha=0.5)
ax_c.tick_params(axis='y', labelsize=10)


# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure_dda_strategy1_longseq.pdf', bbox_inches='tight')
print("Saved: figures/figure_dda_strategy1_longseq.pdf")
print()
print("Key numbers:")
for model in MODELS:
    for split_name in ['short', 'long']:
        d  = cm[(model, split_name)]
        sp = SPLITS[model][split_name]
        print(f"  {model} {split_name}: N_L={sp['N_L']},N_T={sp['N_T']} "
              f"sys_tput={d['sys_tput']:.0f} TTFT_L={d['TTFT_L']:.3f}s")