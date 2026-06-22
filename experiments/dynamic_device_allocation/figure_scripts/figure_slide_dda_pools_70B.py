"""
figure_slide_dda_pools_70B.py — DDA Pools slide for Llama2-70B (32 devices).

Best-gain DDA split:
  Pool L: N_L=12 devices, TP=12, PP=1  → TTFT=3.363s (5s SLO ✓, 1s SLO ✗)
  Pool T: N_T=20 devices, TP=1, PP=80  → 955.8 tok/s
  system_tput = 101.6 + 955.8 = 1057 tok/s  vs  static PP=2,TP=16: 216 tok/s  → 4.9×

Pool L: 12 devices in 2 rows of 6 (TP=12, PP=1, all 80 layers in 1 stage)
Pool T: 20 devices stacked (TP=1, PP=80, 4 layers per device)

Run from repo root: python3 figure_scripts/figure_slide_dda_pools_70B.py
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(18, 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

# ── Colors ────────────────────────────────────────────────────────────────────
LAT_COLOR   = '#DBEAFE'; LAT_EDGE   = '#1D4ED8'; LAT_DEV   = '#93C5FD'
TPUT_COLOR  = '#FEF3C7'; TPUT_EDGE  = '#B45309'; TPUT_DEV  = '#FDE68A'
TOKEN_EDGE  = '#7C3AED'; TOKEN_COLOR = '#F3E8FF'
CXL_COL     = '#F59E0B'; TITLE_C    = '#111827'

def dev_box(ax, x, y, w, h, label, sublabel, fc, ec, fs=8.5, sfs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04',
                                facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3))
    if sublabel:
        ax.text(x+w/2, y+h*0.64, label, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='#1E3A5F', zorder=4)
        ax.text(x+w/2, y+h*0.28, sublabel, ha='center', va='center', fontsize=sfs,
                color='#374151', zorder=4)
    else:
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='#1E3A5F', zorder=4)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(9, 9.62, 'DDA: Dynamic Device Allocation — Latency Pool + Throughput Pool',
        ha='center', va='center', fontsize=19, fontweight='bold', color=TITLE_C)
ax.text(9, 9.25, 'Llama2-70B  |  80 Decoder Layers  |  embed=8192  |  32 total devices split: N_L=12 + N_T=20',
        ha='center', va='center', fontsize=12.5, color='#4B5563')

ax.plot([9, 9], [0.3, 8.95], color='#9CA3AF', linewidth=1.5, linestyle='--', zorder=1)
ax.text(9, 9.05, '── split ──', ha='center', va='center', fontsize=9.5, color='#9CA3AF')

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT: Pool L
# ═══════════════════════════════════════════════════════════════════════════════
LL, LR = 0.3, 8.7; LM = (LL + LR) / 2

ax.add_patch(FancyBboxPatch((LL, 0.40), LR-LL, 8.50, boxstyle='round,pad=0.1',
                            facecolor=LAT_COLOR, edgecolor=LAT_EDGE,
                            linewidth=2.5, alpha=0.35, zorder=0))

ax.text(LM, 8.67, '[L] Pool L — Latency Pool',
        ha='center', va='center', fontsize=14, fontweight='bold', color=LAT_EDGE)
ax.text(LM, 8.33, 'N_L = 12 devices  |  TP = 12  |  PP = 1',
        ha='center', va='center', fontsize=11, color='#1E3A5F')

ax.text(LM, 7.95, 'Interactive Request  (5s relaxed SLO)',
        ha='center', va='center', fontsize=10.5, color=TOKEN_EDGE,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=TOKEN_COLOR,
                  edgecolor=TOKEN_EDGE, linewidth=1.5))

# 12 devices: 2 rows of 6
dw, dh, dgap, rgap = 1.12, 0.72, 0.10, 0.14
n_per_row, n_rows = 6, 2
total_dw = n_per_row * dw + (n_per_row - 1) * dgap
sx = LL + (LR - LL - total_dw) / 2
row1_top = 7.48

for row in range(n_rows):
    yt = row1_top - row * (dh + rgap)
    for col in range(n_per_row):
        dev = row * n_per_row + col + 1
        x = sx + col * (dw + dgap)
        dev_box(ax, x, yt - dh, dw, dh,
                f'Dev {dev}', f'1/12\nweights', LAT_DEV, LAT_EDGE)

# All-reduce arrow
ar_y = row1_top - 2 * dh - rgap - 0.22
ax.annotate('', xy=(sx + total_dw + 0.1, ar_y), xytext=(sx - 0.1, ar_y),
            arrowprops=dict(arrowstyle='<->', color=CXL_COL, lw=2.0))
ax.text(LM, ar_y + 0.14,
        'Broadcast  →  parallel GEMV (each holds 1/12 FC weight rows)  →  All-Reduce (CXL)',
        ha='center', va='center', fontsize=9, color='#92400E', fontweight='bold')

# PP=1 label
pp1_y = ar_y - 0.38
ax.text(LM, pp1_y, 'PP = 1:  All 80 Decoder Layers processed in ONE stage',
        ha='center', va='center', fontsize=10, color='#1E3A5F', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=LAT_EDGE, linewidth=1.2))

# Layer block
lb_top = pp1_y - 0.28
ax.add_patch(FancyBboxPatch((LL + 0.5, lb_top - 0.65), LR - LL - 1.0, 0.65,
                            boxstyle='round,pad=0.05',
                            facecolor='#BFDBFE', edgecolor=LAT_EDGE, linewidth=1.5))
ax.text(LM, lb_top - 0.33, 'Layers 1 – 80  (all in one stage, processed by TP-12 group)',
        ha='center', va='center', fontsize=9.5, color=LAT_EDGE, fontweight='bold')

ax.annotate('', xy=(LM, lb_top - 0.88), xytext=(LM, lb_top - 0.65),
            arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.8))
ax.text(LM, lb_top - 1.05, '1 token in-flight  (PP=1, no pipeline depth)',
        ha='center', va='center', fontsize=9.5, color='#374151', style='italic')

# Token dots
for t_i, t_x in enumerate([LM - 0.8, LM, LM + 0.8]):
    ax.scatter([t_x], [lb_top - 1.52], s=140, color=TOKEN_EDGE,
               alpha=1.0 - t_i * 0.3, zorder=5)
ax.text(LM, lb_top - 1.78, 'Tokens served one at a time', ha='center',
        fontsize=8.5, color='#374151')

# Result box
rb_y = lb_top - 2.12
ax.add_patch(FancyBboxPatch((LL + 0.3, rb_y - 0.65), LR - LL - 0.6, 0.65,
                            boxstyle='round,pad=0.07',
                            facecolor='#EFF6FF', edgecolor=LAT_EDGE, linewidth=2.0))
ax.text(LM, rb_y - 0.20, 'TTFT = 3.363 s  ✓  Meets 5s SLO  (1s SLO not achievable for 70B)',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color=LAT_EDGE)
ax.text(LM, rb_y - 0.48, 'Pool L tput = 101.6 tok/s  (low — single-token serving)',
        ha='center', va='center', fontsize=9.5, color='#374151')

ax.text(LM, rb_y - 0.90,
        'High TP → each token uses all 12 devices → best achievable latency',
        ha='center', va='center', fontsize=9, color='#4B5563', style='italic')

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT: Pool T
# ═══════════════════════════════════════════════════════════════════════════════
TL, TR = 9.3, 17.7; TM = (TL + TR) / 2

ax.add_patch(FancyBboxPatch((TL, 0.40), TR-TL, 8.50, boxstyle='round,pad=0.1',
                            facecolor=TPUT_COLOR, edgecolor=TPUT_EDGE,
                            linewidth=2.5, alpha=0.35, zorder=0))

ax.text(TM, 8.67, '[T] Pool T — Throughput Pool',
        ha='center', va='center', fontsize=14, fontweight='bold', color=TPUT_EDGE)
ax.text(TM, 8.33, 'N_T = 20 devices  |  TP = 1  |  PP = 80',
        ha='center', va='center', fontsize=11, color='#78350F')

ax.text(TM, 7.95, 'Batch Requests  (high-throughput workload)',
        ha='center', va='center', fontsize=10.5, color=TPUT_EDGE,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFBEB',
                  edgecolor=TPUT_EDGE, linewidth=1.5))

# 20 devices stacked — each handles 4 layers (80/20=4)
pipe_top  = 7.52
stage_h   = 0.24
stage_gap = 0.03
pipe_w   = TR - TL - 1.0
pipe_left = TL + 0.5

token_positions = [0, 5, 10, 15]   # tokens at dev 1, 6, 11, 16

for i in range(20):
    sy = pipe_top - i * (stage_h + stage_gap)
    layer_s = i * 4 + 1
    layer_e = layer_s + 3
    fc = '#FDE68A' if i % 2 == 0 else '#FEF3C7'
    ax.add_patch(FancyBboxPatch((pipe_left, sy - stage_h), pipe_w, stage_h,
                                boxstyle='round,pad=0.03',
                                facecolor=fc, edgecolor=TPUT_EDGE,
                                linewidth=1.0, zorder=3))
    ax.text(pipe_left + 0.28, sy - stage_h/2, f'Dev {i+1}',
            ha='left', va='center', fontsize=8, fontweight='bold', color='#78350F')
    ax.text(pipe_left + pipe_w/2 + 0.3, sy - stage_h/2,
            f'Layers {layer_s}–{layer_e}  (TP=1)',
            ha='center', va='center', fontsize=8, color='#374151')

    if i in token_positions:
        tok_x = pipe_left + pipe_w - 0.22
        ax.scatter([tok_x], [sy - stage_h/2], s=100, color=TOKEN_EDGE,
                   zorder=6, marker='D')
        ax.text(tok_x + 0.10, sy - stage_h/2,
                f'Tok {token_positions.index(i)+1}',
                va='center', fontsize=7.5, color=TOKEN_EDGE, fontweight='bold')

# Pipeline mini-arrows
for i in range(19):
    sy = pipe_top - i * (stage_h + stage_gap) - stage_h - stage_gap/2
    ax.annotate('', xy=(TM, sy - 0.01), xytext=(TM, sy + 0.01),
                arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=0.7))

# Bottom of pipeline
bottom_pipe = pipe_top - 19 * (stage_h + stage_gap) - stage_h

tokens_y = bottom_pipe - 0.20
ax.text(TM, tokens_y,
        '4 tokens in-flight simultaneously  (pipeline parallelism)',
        ha='center', va='center', fontsize=9.5, color='#92400E',
        fontweight='bold', style='italic')

# Result box placed snugly below tokens label
rb_bot = tokens_y - 0.25   # top of result box content
ax.add_patch(FancyBboxPatch((TL + 0.3, rb_bot - 0.58), TR - TL - 0.6, 0.58,
                            boxstyle='round,pad=0.07',
                            facecolor='#FFFBEB', edgecolor=TPUT_EDGE, linewidth=2.0))
ax.text(TM, rb_bot - 0.18, 'Pool T tput = 955.8 tok/s  (high pipeline throughput)',
        ha='center', va='center', fontsize=11, fontweight='bold', color=TPUT_EDGE)
ax.text(TM, rb_bot - 0.44, 'TTFT >> 5s  (batch / offline workloads only)',
        ha='center', va='center', fontsize=9.5, color='#374151')

# ── Bottom bar ────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((1.5, 0.44), 15, 0.60, boxstyle='round,pad=0.07',
                            facecolor='#F0FDF4', edgecolor='#166534', linewidth=2.0))
ax.text(9, 0.74,
        'DDA System Throughput = Pool_L + Pool_T = 101.6 + 955.8 = 1057 tok/s'
        '    vs    Best static (PP=2, TP=16): 216 tok/s    →    4.9× gain',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#166534')

plt.tight_layout(pad=0)
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure_slide_dda_pools_70B.pdf', bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved: figures/figure_slide_dda_pools_70B.pdf")
