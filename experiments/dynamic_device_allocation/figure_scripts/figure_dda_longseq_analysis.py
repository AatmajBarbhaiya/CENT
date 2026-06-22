"""
figure_dda_longseq.py — DDA Pool T vs static configs for seqlen > 512 (long requests).

Panel (a): TTFT vs seqlen — log scale.
Panel (b): Throughput vs seqlen.

Colors: blue = static lat-optimal (max TP), orange = static tput-optimal (max PP),
        red = DDA Pool T (tput-optimal config of N_T sub-pool).
Linestyle: solid=7B, dashed=13B, dotted=70B.

Key message: Pool T (DDA) achieves higher throughput than static lat-optimal
for the same request class, but at much higher TTFT (batch workload trade-off).
Pool T throughput is below static tput-optimal (fewer devices).
The net DDA gain comes from Pool L running concurrently for short requests.

Run from repo root: python3 figure_scripts/figure_dda_longseq.py
"""

import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SIM_BASE = pd.read_csv('../../cent_simulation/simulation_results.csv')
RESULTS  = 'results'

T_SHORT = 512
MODELS  = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
LAYERS  = {'Llama2-7B': 32, 'Llama2-13B': 40, 'Llama2-70B': 80}
TOTAL_DEV = {'Llama2-7B': 8,  'Llama2-13B': 20, 'Llama2-70B': 32}
SPLITS  = {'Llama2-7B': (2,6), 'Llama2-13B': (6,14), 'Llama2-70B': (12,20)}

# Same balanced static configs used in strategy1 comparison
# (best static with TTFT <= DDA short-request TTFT)
STATIC_CMP = {'Llama2-7B': (4,2), 'Llama2-13B': (2,10), 'Llama2-70B': (2,16)}

# Color by config type; linestyle by model
COL_STATIC = 'steelblue'    # balanced static (same as strategy1 comparison)
COL_DDA    = '#cc0000'      # DDA Pool T
LSTYLE     = {'Llama2-7B': '-', 'Llama2-13B': '--', 'Llama2-70B': (0,(3,1,1,1))}
LW_MAIN  = 2.2
SLO_1S, SLO_5S = 1.0, 5.0


def get_series(df, pp, tp, seqlen_min=None):
    """Extract (seqlen, TTFT, Throughput) series for a (PP,TP) config."""
    grp = df[(df['Pipeline parallelism']==pp) & (df['Tensor parallelism']==tp)]
    grp = grp.sort_values('Sequence length')
    if seqlen_min is not None:
        grp = grp[grp['Sequence length'] >= seqlen_min]
    seqlens = sorted(grp['Sequence length'].unique())
    ttft, tput = [], []
    for s in seqlens:
        sub = grp[grp['Sequence length'] <= s]
        ttft.append(sub['Token latency (ms)'].mean() * s / 1000)
        tput.append(grp[grp['Sequence length']==s]['Throughput (tokens/s)'].values[0])
    return np.array(seqlens), np.array(ttft), np.array(tput)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

for model in MODELS:
    NL, NT   = SPLITS[model]
    lay      = LAYERS[model]
    ls       = LSTYLE[model]
    pp_s, tp_s = STATIC_CMP[model]
    short    = model.replace('Llama2-', 'L2-')

    b   = SIM_BASE[SIM_BASE['Model'] == model]
    dfT = pd.read_csv(f'{RESULTS}/sim_{model}_poolT_{NT}.csv')

    # Balanced static (same config as strategy1 comparison)
    ss, ttft_ss, tput_ss = get_series(b, pp_s, tp_s, seqlen_min=T_SHORT)
    ax1.plot(ss, ttft_ss, color=COL_STATIC, linestyle=ls, linewidth=LW_MAIN, zorder=4,
             label=f'Static PP={pp_s},TP={tp_s} ({short})')
    ax2.plot(ss, tput_ss, color=COL_STATIC, linestyle=ls, linewidth=LW_MAIN, zorder=4,
             label=f'Static PP={pp_s},TP={tp_s} ({short})')

    # DDA Pool T: PP=lay, TP=1 on N_T devices
    dt, ttft_dt, tput_dt = get_series(dfT, lay, 1, seqlen_min=T_SHORT)
    ax1.plot(dt, ttft_dt, color=COL_DDA, linestyle=ls, linewidth=LW_MAIN+0.5, zorder=6,
             label=f'DDA Pool T PP={lay},TP=1 N_T={NT} ({short})')
    ax2.plot(dt, tput_dt, color=COL_DDA, linestyle=ls, linewidth=LW_MAIN+0.5, zorder=6,
             label=f'DDA Pool T N_T={NT} ({short})')

    # Shade gap: Pool T tput gain over balanced static
    min_len = min(len(ss), len(dt))
    ax2.fill_between(dt[:min_len], tput_ss[:min_len], tput_dt[:min_len],
                     alpha=0.12, color=COL_DDA, zorder=2)

# ── Panel (a): TTFT ───────────────────────────────────────────────────────────
ax1.axhline(y=SLO_1S, color='green', linestyle='--', linewidth=1.3,
            alpha=0.8, label='1s SLO')
ax1.axhline(y=SLO_5S, color='olive', linestyle=':', linewidth=1.3,
            alpha=0.8, label='5s SLO')
ax1.set_yscale('log')
ax1.set_xlabel('Sequence Length (tokens)', fontsize=12)
ax1.set_ylabel('TTFT (s, log scale)', fontsize=12)
ax1.set_title('(a) TTFT vs Seqlen — Long Requests (seqlen > 512)\n'
              'Same balanced static as Strategy 1 comparison (PP=4,TP=2 / PP=2,TP=10 / PP=2,TP=16)',
              fontsize=10.5)
ax1.set_xlim(T_SHORT, 4096)
ax1.grid(linestyle='--', alpha=0.4, which='both')
ax1.tick_params(labelsize=10)

# ── Panel (b): Throughput ─────────────────────────────────────────────────────
ax2.set_xlabel('Sequence Length (tokens)', fontsize=12)
ax2.set_ylabel('Throughput (tok/s)', fontsize=12)
ax2.set_title('(b) Throughput vs Seqlen — Long Requests (seqlen > 512)\n'
              'Shaded = Pool T throughput gain over balanced static\n'
              '7B: ~1× (negligible), 13B: ~3×, 70B: ~4×',
              fontsize=10.5)
ax2.set_xlim(T_SHORT, 4096)
ax2.set_ylim(bottom=0)
ax2.grid(linestyle='--', alpha=0.4)
ax2.tick_params(labelsize=10)

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_els = [
    mlines.Line2D([0],[0], color=COL_STATIC, linewidth=2,
                  label='Balanced static (same config as Strategy 1 comparison)'),
    mlines.Line2D([0],[0], color=COL_DDA,    linewidth=2.5,
                  label='DDA Pool T (PP=max_layers, TP=1, N_T devices)'),
    mlines.Line2D([0],[0], color='gray', linewidth=2, linestyle='-',
                  label='Llama2-7B  — static PP=4,TP=2  | Pool T PP=32,TP=1 N_T=6'),
    mlines.Line2D([0],[0], color='gray', linewidth=2, linestyle='--',
                  label='Llama2-13B — static PP=2,TP=10 | Pool T PP=40,TP=1 N_T=14'),
    mlines.Line2D([0],[0], color='gray', linewidth=2, linestyle=(0,(3,1,1,1)),
                  label='Llama2-70B — static PP=2,TP=16 | Pool T PP=80,TP=1 N_T=20'),
    mlines.Line2D([0],[0], color='green', linewidth=1.3, linestyle='--', label='1s SLO'),
    mlines.Line2D([0],[0], color='olive', linewidth=1.3, linestyle=':',  label='5s SLO'),
]
fig.legend(handles=legend_els, fontsize=8.5, loc='lower center',
           ncol=4, bbox_to_anchor=(0.5, -0.13), frameon=True)

plt.suptitle('DDA Strategy 1 — Long-Request Performance: Pool T vs Balanced Static\n'
             'Same static configs used in Strategy 1 short-request comparison',
             fontsize=11)
plt.tight_layout(rect=[0, 0.08, 1, 1])

os.makedirs('figures', exist_ok=True)
os.makedirs('figure_source_data', exist_ok=True)
plt.savefig('figures/figure_dda_longseq_analysis.pdf', bbox_inches='tight')
print("Saved: figures/figure_dda_longseq_analysis.pdf")

# Print key comparison at seqlen=1024
print("\nAt seqlen=1024 (balanced static vs Pool T):")
for model in MODELS:
    NL, NT = SPLITS[model]; lay=LAYERS[model]
    pp_s, tp_s = STATIC_CMP[model]
    b = SIM_BASE[SIM_BASE['Model']==model]
    dfT = pd.read_csv(f'{RESULTS}/sim_{model}_poolT_{NT}.csv')
    def at(df, pp, tp, s):
        sub = df[(df['Pipeline parallelism']==pp)&(df['Tensor parallelism']==tp)&(df['Sequence length']<=s)]
        tput_row = df[(df['Pipeline parallelism']==pp)&(df['Tensor parallelism']==tp)&(df['Sequence length']==s)]
        return sub['Token latency (ms)'].mean()*s/1000, float(tput_row['Throughput (tokens/s)'].iloc[0])
    sl, sp = at(b,   pp_s, tp_s, 1024)
    dl, dp = at(dfT, lay,  1,    1024)
    print(f"  {model}: static PP={pp_s},TP={tp_s} TTFT={sl:.1f}s tput={sp:.0f}  |  Pool T TTFT={dl:.1f}s tput={dp:.0f}  |  tput_gain={dp/sp:.2f}x TTFT_penalty={dl/sl:.1f}x")
