import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Data sources ──────────────────────────────────────────────────────────────
sim      = pd.read_csv('../../cent_simulation/simulation_results.csv')
proc     = pd.read_csv('../../cent_simulation/processed_results_multiseqlen.csv')
ttft_csv = pd.read_csv('../../cent_simulation/processed_results_ttft.csv')

models       = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
model_labels = ['Llama2-7B\n(8 dev)', 'Llama2-13B\n(20 dev)', 'Llama2-70B\n(32 dev)']
devices      = {'Llama2-7B': 8,  'Llama2-13B': 20,  'Llama2-70B': 32}
layers       = {'Llama2-7B': 32, 'Llama2-13B': 40,  'Llama2-70B': 80}
colors       = {'Llama2-7B': 'steelblue', 'Llama2-13B': 'darkorange', 'Llama2-70B': 'forestgreen'}
markers      = {'Llama2-7B': 'o',          'Llama2-13B': 's',          'Llama2-70B': '^'}

SLO_interactive = 1.0   # 1 s
SLO_relaxed     = 5.0   # 5 s
DDA_THRESHOLD   = 512   # seqlen boundary: short < 512, long >= 512

# ── Panel (a) helpers ─────────────────────────────────────────────────────────
# Reads processed_results_multiseqlen.csv at Seqlen=4096, Phase=prefill/end2end

def get_extreme_configs(model):
    dev = devices[model]
    pref = proc[(proc['Model'] == model) & (proc['Seqlen'] == 4096)]

    # Lat-optimal: PP=1, TP=max_devices
    lat_row  = pref[(pref['Phase'] == 'prefill') & (pref['Pipeline parallelism'] == 1) & (pref['Tensor parallelism'] == dev)]
    e2e_lat  = pref[(pref['Phase'] == 'end2end') & (pref['Pipeline parallelism'] == 1) & (pref['Tensor parallelism'] == dev)]

    # Tput-optimal: TP=1 config with MAX end2end throughput (PP=32 beats PP=80 for 70B)
    tp1_e2e = pref[(pref['Phase'] == 'end2end') & (pref['Tensor parallelism'] == 1)]
    tput_pp = int(tp1_e2e.sort_values('Throughput (tokens/s)', ascending=False).iloc[0]['Pipeline parallelism'])
    tput_row = pref[(pref['Phase'] == 'prefill') & (pref['Pipeline parallelism'] == tput_pp) & (pref['Tensor parallelism'] == 1)]
    e2e_tput = pref[(pref['Phase'] == 'end2end') & (pref['Pipeline parallelism'] == tput_pp) & (pref['Tensor parallelism'] == 1)]

    return {
        'lat_ttft':      float(lat_row['Total Latency (s)'].values[0]),
        'tput_ttft':     float(tput_row['Total Latency (s)'].values[0]),
        'lat_e2e_tput':  float(e2e_lat['Throughput (tokens/s)'].values[0]),
        'tput_e2e_tput': float(e2e_tput['Throughput (tokens/s)'].values[0]),
        'tput_pp':       tput_pp,   # actual PP used (PP=32 for 70B, not PP=80)
    }


# ── Panel (b) helpers ─────────────────────────────────────────────────────────
# TTFT(s) = prefill latency at seqlen=s from processed_results_ttft.csv

def compute_running_ttft(model):
    sub = ttft_csv[ttft_csv['Model'] == model].copy()
    sub = sub.rename(columns={
        'Seqlen': 'seqlen',
        'Total Latency (s)': 'TTFT',
        'Pipeline parallelism': 'PP',
        'Tensor parallelism': 'TP',
    })
    return sub[['PP', 'TP', 'seqlen', 'TTFT']]


# ── Panel (c) helpers ─────────────────────────────────────────────────────────
# short_TTFT: prefill latency at Seqlen=DDA_THRESHOLD from processed_results_ttft.csv
# long_Tput:  mean decode throughput for seqlen > DDA_THRESHOLD from simulation_results.csv

def compute_dda_scatter(model):
    sub      = sim[sim['Model'] == model]
    ttft_sub = ttft_csv[(ttft_csv['Model'] == model) & (ttft_csv['Seqlen'] == DDA_THRESHOLD)]
    pts = []
    for (pp, tp), grp in sub.groupby(['Pipeline parallelism', 'Tensor parallelism']):
        long_ = grp[grp['Sequence length'] > DDA_THRESHOLD]
        ttft_row = ttft_sub[(ttft_sub['Pipeline parallelism'] == pp) & (ttft_sub['Tensor parallelism'] == tp)]
        if len(long_) == 0 or len(ttft_row) == 0:
            continue
        short_ttft = float(ttft_row['Total Latency (s)'].values[0])
        long_tput  = long_['Throughput (tokens/s)'].mean()
        pts.append({'PP': pp, 'TP': tp, 'short_TTFT': short_ttft, 'long_Tput': long_tput})
    return pd.DataFrame(pts)


# ── Figure ────────────────────────────────────────────────────────────────────
from matplotlib.gridspec import GridSpec
# Row 0: scatter panels (a)(b)(c) — per-model DDA scatter  ← now on top
# Row 1: bar chart (d) + seqlen line (e)                   ← now on bottom
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(2, 6, figure=fig, hspace=0.5, wspace=0.45,
               height_ratios=[1.0, 1.6])   # scatter row compact; bar+line row taller
ax3s = [fig.add_subplot(gs[0, k*2:(k+1)*2]) for k in range(3)]  # (a)(b)(c): scatter, top row
ax1  = fig.add_subplot(gs[1, 0:3])        # panel (d): bar chart, bottom-left
ax2  = fig.add_subplot(gs[1, 3:6])        # panel (e): seqlen line, bottom-right

# Shared scale for all 3 scatter panels
# x goes to 31s to show 70B PP=80 at 28.7s; hatching still clips at PP=32 (9.94s)
SCATTER_XLIM = (0, 31.0)
SCATTER_YLIM = (0, 4500)


# ══════════════════════════════════════════════════════════════════════════════
# Panel (a): TTFT at extreme configs — bar chart
# ══════════════════════════════════════════════════════════════════════════════
ttft_minlat, ttft_maxtput, tput_minlat, tput_maxtput = [], [], [], []

for model in models:
    cfg = get_extreme_configs(model)
    ttft_minlat.append(cfg['lat_ttft'])
    ttft_maxtput.append(cfg['tput_ttft'])
    tput_minlat.append(cfg['lat_e2e_tput'])
    tput_maxtput.append(cfg['tput_e2e_tput'])

x = np.arange(len(models))
width = 0.35

bars_ml = ax1.bar(x - width/2, ttft_minlat, width,
                  label='Latency-optimal',
                  color='steelblue', edgecolor='black')
bars_mt = ax1.bar(x + width/2, ttft_maxtput, width,
                  label='Throughput-optimal',
                  color='lightsalmon', edgecolor='black')

ax1.axhline(y=SLO_interactive, color='red',     linestyle='--', linewidth=1.0,
            label=f'Interactive SLO ({SLO_interactive}s)')
ax1.axhline(y=SLO_relaxed,     color='darkred', linestyle=':',  linewidth=1.0,
            label=f'Relaxed SLO ({SLO_relaxed}s)')

for bar, val in zip(bars_mt, ttft_maxtput):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.12,
             f'{val:.2f}s', ha='center', va='bottom', fontsize=9)


for i in range(len(models)):
    loss_pct = (1 - tput_minlat[i] / tput_maxtput[i]) * 100
    bar = bars_ml[i]
    val = ttft_minlat[i]
    x_pos = bar.get_x() + bar.get_width() / 2

    # 1. Place the time (in seconds) right on top of the blue bar
    ax1.text(x_pos, val + 0.15, f'{val:.2f}s',
             ha='center', va='bottom', fontsize=9, color='black')

    # 2. Place the throughput loss % directly above the time text
    # We add an offset (~0.65) so it floats cleanly above the time string
    ax1.text(x_pos, val + 0.65, f'−{loss_pct:.0f}%\ntput',
             ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

# for i in range(len(models)):
#     loss_pct = (1 - tput_minlat[i] / tput_maxtput[i]) * 100
#     bar = bars_ml[i]
#     val = ttft_minlat[i]
#     height = bar.get_height()
#     mid = height / 2
#     x_pos = bar.get_x() + bar.get_width()/2

#     if height > 1.5:
#         ax1.text(x_pos, val + 0.12, f'{val:.2f}s',
#                  ha='center', va='bottom', fontsize=9)
#         ax1.text(x_pos, mid, f'−{loss_pct:.0f}%\ntput',
#                  ha='center', va='center', fontsize=8, color='white', fontweight='bold')
#     else:
#         loss_y = 0.2
#         ax1.text(x_pos, loss_y + 1, f'−{loss_pct:.0f}%\ntput',
#                  ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
#         ax1.text(x_pos, loss_y + 3.5, f'{val:.2f}s',
#                  ha='center', va='bottom', fontsize=9)

ax1.set_xticks(x)
ax1.set_xticklabels(model_labels, fontsize=11)
ax1.set_ylabel('Time to First Token (TTFT), (s)', fontsize=11)
ax1.set_title('(d) TTFT: Latency-Optimal vs Throughput-Optimal Config', fontsize=11)
ax1.set_ylim(0, max(ttft_maxtput) * 1.22)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.tick_params(axis='y', labelsize=11)


# ══════════════════════════════════════════════════════════════════════════════
# Panel (b): TTFT vs seqlen — Llama2-70B, all (PP,TP) configs
# ══════════════════════════════════════════════════════════════════════════════
ttft_df = compute_running_ttft('Llama2-70B')

# CHANGE: Using a qualitative colormap for high contrast between lines
tp_values = sorted(ttft_df['TP'].unique())
cmap = plt.cm.get_cmap('tab10') 
tp_color = {tp: cmap(i % 10) for i, tp in enumerate(tp_values)}

for (pp, tp), grp in ttft_df.groupby(['PP', 'TP']):
    grp = grp.sort_values('seqlen')
    is_lat  = (pp == 1 and tp == max(tp_values))   # lat-optimal: PP=1,TP=32
    is_tput = (pp == 32 and tp == 1)               # tput-optimal: PP=32,TP=1
    is_pp80 = (pp == 80 and tp == 1)               # PP=80: plot faint (lower tput)
    lw    = 2.2 if (is_lat or is_tput) else 0.8
    alpha = 1.0 if (is_lat or is_tput) else (0.35 if is_pp80 else 0.5)
    ax2.plot(grp['seqlen'], grp['TTFT'],
             color=tp_color[tp], linewidth=lw, alpha=alpha,
             label=f'PP={pp}, TP={tp}')

# DDA threshold vertical line
ax2.axvline(x=DDA_THRESHOLD, color='purple', linestyle='--', linewidth=1.8,
            label=f'DDA threshold T={DDA_THRESHOLD}')
ax2.axhline(y=SLO_interactive, color='red',     linestyle='--', linewidth=1.5,
            label=f'SLO {SLO_interactive}s')
ax2.axhline(y=SLO_relaxed,     color='darkred', linestyle=':',  linewidth=1.5,
            label=f'SLO {SLO_relaxed}s')

dev70  = devices['Llama2-70B']
# Tput-optimal for 70B is PP=32 (higher throughput than PP=80); lay70 updated accordingly
lay70  = 32   # was 80; PP=32,TP=1 gives tput=1340 vs PP=80,TP=1 tput=1185

# Log scale — data spans ~3s to ~280s; SLO lines must stay visible
ax2.set_yscale('log')

# Shade short-request region (must draw after yscale set)
ax2.axvspan(0, DDA_THRESHOLD, alpha=0.06, color='green')

# Text position: derive from data max so it scales correctly
y_max_data = ttft_df['TTFT'].max()
ax2.text(DDA_THRESHOLD * 0.45, y_max_data * 0.25,
         'short\nreqs', ha='center', va='top', fontsize=8,
         color='green', style='italic')

# Annotate the two extreme config endpoints
lat_end  = ttft_df[(ttft_df['PP']==1)  & (ttft_df['TP']==dev70)].sort_values('seqlen').iloc[-1]
tput_end = ttft_df[(ttft_df['PP']==lay70) & (ttft_df['TP']==1)].sort_values('seqlen').iloc[-1]

ax2.annotate(f'PP=1, TP={dev70}\n(latency-optimal)',
             xy=(lat_end['seqlen'], lat_end['TTFT']),
             xytext=(-60, -40), textcoords='offset points',
             fontsize=8, color=tp_color[dev70],
             arrowprops=dict(arrowstyle='->', color=tp_color[dev70], lw=1.2))
ax2.annotate(f'PP={lay70}, TP=1\n(throughput-optimal)',
             xy=(tput_end['seqlen'], tput_end['TTFT']),
             xytext=(-150, -5), textcoords='offset points',
             fontsize=8, color=tp_color[1],
             arrowprops=dict(arrowstyle='->', color=tp_color[1], lw=1.2))

ax2.set_xlabel('Sequence Length (tokens)', fontsize=12)
ax2.set_ylabel('TTFT (s, log scale)', fontsize=12)
ax2.set_title('(e) TTFT vs Seqlen — Llama2-70B, All (PP, TP) Configs', fontsize=11)
ax2.set_xlim(left=0)
ax2.legend(fontsize=7, loc='lower right', ncol=1)
ax2.grid(linestyle='--', alpha=0.5, which='both')
ax2.tick_params(labelsize=11)


# ══════════════════════════════════════════════════════════════════════════════
# Panels (c)(d)(e): Per-model scatter — one panel per model.
# Each panel: static (PP,TP) configs as dots + DDA ideal star +
#             hatched region between Pareto curve and ideal corner
#             (shows the throughput + TTFT space uniquely reachable by DDA).
# ══════════════════════════════════════════════════════════════════════════════
all_rows = []

PANEL_LABELS = ['(a)', '(b)', '(c)']

for idx, model in enumerate(models):
    ax3 = ax3s[idx]
    pts = compute_dda_scatter(model)
    pts['Model'] = model
    dev = devices[model]
    lay = layers[model]
    col = colors[model]

    sorted_pts = pts.sort_values('short_TTFT').reset_index(drop=True)

    # DDA ideal corner: lat-opt TTFT + max-tput throughput
    dda_x = float(pts[pts['PP'] == 1]['short_TTFT'].min())
    dda_y = float(pts[pts['TP'] == 1]['long_Tput'].max())

    # Best-tput config (max Y) and its x-coordinate — hatching cutoff
    max_tput_row = pts.sort_values('long_Tput', ascending=False).iloc[0]
    hatch_x_max  = float(max_tput_row['short_TTFT'])  # hatch only LEFT of this x

    # ── Hatching: region between Pareto curve and ideal corner ──────────────
    # Clip to x ≤ hatch_x_max (left of max-tput config, avoids extending right
    # into lower-tput configs like 70B PP=80 which has HIGHER TTFT but LOWER tput)
    clip_pts = sorted_pts[sorted_pts['short_TTFT'] <= hatch_x_max + 0.01]
    if not clip_pts.empty:
        ax3.fill_between(clip_pts['short_TTFT'],
                         clip_pts['long_Tput'],
                         dda_y,
                         where=(clip_pts['long_Tput'] <= dda_y),
                         alpha=0.12, color=col, zorder=1,
                         hatch='///', edgecolor=col, linewidth=0.4)

    # Pareto curve + all static configs (small dots)
    ax3.plot(sorted_pts['short_TTFT'], sorted_pts['long_Tput'],
             color=col, linestyle='-', linewidth=1.5, alpha=0.4, zorder=3)
    ax3.scatter(pts['short_TTFT'], pts['long_Tput'],
                color=col, marker=markers[model],
                s=55, edgecolors='black', linewidths=0.6, zorder=4, alpha=0.7)

    # ── Annotate 2 key configs: best TTFT (lat-opt) and best throughput ─────
    lat_static   = pts.sort_values('short_TTFT').iloc[0]
    max_tput_cfg = pts.sort_values('long_Tput', ascending=False).iloc[0]

    ax3.annotate(f'PP={int(lat_static.PP)},TP={int(lat_static.TP)}\n(best TTFT)',
                 (lat_static['short_TTFT'], lat_static['long_Tput']),
                 xytext=(10, -4), textcoords='offset points', fontsize=7.5, color=col,
                 fontweight='bold')
    ax3.annotate(f'PP={int(max_tput_cfg.PP)},TP={int(max_tput_cfg.TP)}\n(best tput)',
                 (max_tput_cfg['short_TTFT'], max_tput_cfg['long_Tput']),
                 xytext=(6, 5), textcoords='offset points', fontsize=7.5, color=col,
                 fontweight='bold')

    # DDA ideal star
    ax3.scatter(dda_x, dda_y, color=col, marker='*',
                s=400, edgecolors='black', linewidths=0.9, zorder=8,
                label='DDA ideal ★')

    # Dotted lines to axes from DDA star
    ax3.plot([dda_x, dda_x], [0, dda_y], color=col, linestyle=':', linewidth=1.0, alpha=0.5)
    ax3.plot([0, dda_x],     [dda_y, dda_y], color=col, linestyle=':', linewidth=1.0, alpha=0.5)

    lbl = PANEL_LABELS[idx]
    ax3.set_xlabel(f'Short-Req TTFT (seqlen≤{DDA_THRESHOLD}), (s)', fontsize=10)
    if idx == 0:
        ax3.set_ylabel(f'Long-Req Throughput (seqlen>{DDA_THRESHOLD}), (tok/s)', fontsize=10)
    short = model.replace('Llama2-', 'L2-')
    ax3.set_title(f'{lbl} {short}', fontsize=10)
    ax3.set_xlim(*SCATTER_XLIM)
    ax3.set_ylim(*SCATTER_YLIM)
    ax3.grid(linestyle='--', alpha=0.5)
    ax3.tick_params(labelsize=10)
    ax3.legend(fontsize=8, loc='lower right')

    all_rows.append(pts)

plt.tight_layout()


# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('figure_source_data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Consolidated CSV: all panels' data
csv_rows = []

# Panel (a) data
for i, model in enumerate(models):
    csv_rows.append({
        'Panel': 'a', 'Model': model, 'Config': 'lat-optimal',
        'short_TTFT': ttft_minlat[i], 'Throughput': tput_minlat[i],
        'Source': 'proc_seqlen4096_prefill'
    })
    csv_rows.append({
        'Panel': 'a', 'Model': model, 'Config': 'tput-optimal',
        'short_TTFT': ttft_maxtput[i], 'Throughput': tput_maxtput[i],
        'Source': 'proc_seqlen4096_prefill'
    })

# Panel (b) data
ttft_df['Panel'] = 'b'
ttft_df['Model'] = 'Llama2-70B'

# Panel (c) data
for pts in all_rows:
    for _, row in pts.iterrows():
        csv_rows.append({
            'Panel': 'c', 'Model': row['Model'],
            'Pipeline parallelism': row['PP'], 'Tensor parallelism': row['TP'],
            'short_TTFT': row['short_TTFT'], 'long_Tput': row['long_Tput'],
            'Source': f'sim_threshold{DDA_THRESHOLD}'
        })

pd.DataFrame(csv_rows).to_csv('figure_source_data/figure_motivation_dda.csv', index=False)
ttft_df.to_csv('figure_source_data/figure_motivation_dda_panel_b.csv', index=False)

plt.savefig('figures/figure_motivation_dda.pdf', bbox_inches='tight')
print("Saved: figures/figure_motivation_dda.pdf")
print("Saved: figure_source_data/figure_motivation_dda.csv")
print("Saved: figure_source_data/figure_motivation_dda_panel_b.csv")
