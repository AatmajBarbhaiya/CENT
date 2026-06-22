"""
figure_slide_dda_pools.py — Presentation Slide 2.

Shows DDA device partitioning for Llama2-13B:
  Pool L (Latency):   N_L=10 devices, TP=10, PP=1  → TTFT=0.882s  (1s SLO ✓)
  Pool T (Throughput): N_T=10 devices, TP=1,  PP=40 → 2506 tok/s

Pool L: all 10 devices serve ONE request together (high TP reduces latency).
Pool T: 10 devices pipeline 40 stages (4 layers/device); many tokens in-flight.

Run from repo root: python3 figure_scripts/figure_slide_dda_pools.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig = plt.figure(figsize=(18, 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

# ── Colors ────────────────────────────────────────────────────────────────────
LAT_COLOR    = '#DBEAFE'   # light blue — latency pool
LAT_EDGE     = '#1D4ED8'
LAT_DEV      = '#93C5FD'
TPUT_COLOR   = '#FEF3C7'   # light amber — throughput pool
TPUT_EDGE    = '#B45309'
TPUT_DEV     = '#FDE68A'
TOKEN_COLOR  = '#F3E8FF'   # purple for tokens
TOKEN_EDGE   = '#7C3AED'
TITLE_COLOR  = '#111827'
CXL_COLOR    = '#F59E0B'

def dev_box(ax, x, y, w, h, label, sublabel, facecolor, edgecolor,
            fontsize=9.5, sub_fontsize=8):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.04',
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='#1E3A5F', zorder=4)
        ax.text(x + w/2, y + h*0.28, sublabel, ha='center', va='center',
                fontsize=sub_fontsize, color='#374151', zorder=4)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='#1E3A5F', zorder=4)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(9, 9.6, 'DDA: Dynamic Device Allocation — Latency Pool + Throughput Pool',
        ha='center', va='center', fontsize=20, fontweight='bold', color=TITLE_COLOR)
ax.text(9, 9.22, 'Llama2-13B  |  40 Decoder Layers  |  20 total CXL devices split: N_L=10 + N_T=10',
        ha='center', va='center', fontsize=13, color='#4B5563')

# ── Dividing line ─────────────────────────────────────────────────────────────
ax.plot([9, 9], [0.3, 8.9], color='#9CA3AF', linewidth=1.5, linestyle='--', zorder=1)
ax.text(9, 9.0, '─── split ───', ha='center', va='center',
        fontsize=10, color='#9CA3AF')

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL: Pool L (Latency)
# ═══════════════════════════════════════════════════════════════════════════════
L_LEFT, L_RIGHT = 0.3, 8.7
L_MID = (L_LEFT + L_RIGHT) / 2

# Panel background
bg = FancyBboxPatch((L_LEFT, 0.4), L_RIGHT - L_LEFT, 8.45,
                    boxstyle='round,pad=0.1',
                    facecolor=LAT_COLOR, edgecolor=LAT_EDGE,
                    linewidth=2.5, alpha=0.35, zorder=0)
ax.add_patch(bg)

ax.text(L_MID, 8.65, '[L] Pool L — Latency Pool',
        ha='center', va='center', fontsize=14, fontweight='bold', color=LAT_EDGE)
ax.text(L_MID, 8.32, 'N_L = 10 devices  |  TP = 10  |  PP = 1',
        ha='center', va='center', fontsize=11, color='#1E3A5F')

# Interactive request token (incoming)
ax.text(L_MID, 7.92,
        '⬇  Interactive Request  (SLO = 1s)',
        ha='center', va='center', fontsize=10.5, color=TOKEN_EDGE,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=TOKEN_COLOR,
                  edgecolor=TOKEN_EDGE, linewidth=1.5))

# 10 devices in a single TP group (1 row)
dw, dh, gap = 0.72, 0.80, 0.08
n_dev = 10
total_dev_w = n_dev * dw + (n_dev - 1) * gap
dev_start_x = L_LEFT + (L_RIGHT - L_LEFT - total_dev_w) / 2
dev_y = 6.35

for i in range(n_dev):
    dx = dev_start_x + i * (dw + gap)
    dev_box(ax, dx, dev_y, dw, dh,
            f'Dev {i+1}', f'1/{n_dev}\nweights',
            LAT_DEV, LAT_EDGE, fontsize=8.5, sub_fontsize=7.5)

# TP group bracket
bx1 = dev_start_x - 0.05
bx2 = dev_start_x + total_dev_w + 0.05
bracket_y = dev_y - 0.22
ax.annotate('', xy=(bx2, bracket_y), xytext=(bx1, bracket_y),
            arrowprops=dict(arrowstyle='<->', color=CXL_COLOR, lw=2.0))
ax.text(L_MID, bracket_y - 0.20,
        'Broadcast input → parallel GEMV (each handles 1/10 FC weights) → All-Reduce',
        ha='center', va='center', fontsize=8.5, color='#92400E', fontweight='bold')

# Single pipeline stage label
ax.text(L_MID, 5.65, 'PP = 1:  All 40 Decoder Layers processed in ONE stage',
        ha='center', va='center', fontsize=10, color='#1E3A5F', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=LAT_EDGE, linewidth=1.2))

# Show all 40 layers as a single block
layer_block = FancyBboxPatch((L_LEFT + 0.5, 4.55), L_RIGHT - L_LEFT - 1.0, 0.72,
                             boxstyle='round,pad=0.05',
                             facecolor='#BFDBFE', edgecolor=LAT_EDGE,
                             linewidth=1.5)
ax.add_patch(layer_block)
ax.text(L_MID, 4.91, 'Layers 1 – 40  (all in one stage, processed by TP group)',
        ha='center', va='center', fontsize=9.5, color=LAT_EDGE, fontweight='bold')

# Output arrow
ax.annotate('', xy=(L_MID, 4.20), xytext=(L_MID, 4.55),
            arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.8))

# Timeline: 1 token, fast
ax.text(L_MID, 3.95, '1 token in-flight at a time  (PP=1, no pipeline)',
        ha='center', va='center', fontsize=9.5, color='#374151', style='italic')

# Token flow diagram
for t_i, t_x in enumerate([L_MID - 0.8, L_MID, L_MID + 0.8]):
    alpha = 1.0 - t_i * 0.3
    ax.scatter([t_x], [3.45], s=160, color=TOKEN_EDGE, alpha=alpha, zorder=5)
ax.text(L_MID, 3.18, 'Tokens served one at a time', ha='center', fontsize=8.5, color='#374151')

# Result box
result = FancyBboxPatch((L_LEFT + 0.3, 2.25), L_RIGHT - L_LEFT - 0.6, 0.72,
                        boxstyle='round,pad=0.07',
                        facecolor='#EFF6FF', edgecolor=LAT_EDGE, linewidth=2.0)
ax.add_patch(result)
ax.text(L_MID, 2.73, 'TTFT = 0.882 s  ✓  Meets 1s SLO',
        ha='center', va='center', fontsize=11, fontweight='bold', color=LAT_EDGE)
ax.text(L_MID, 2.48, 'Pool L tput = 390 tok/s  (low — single-token throughput)',
        ha='center', va='center', fontsize=9.5, color='#374151')

ax.text(L_MID, 1.90, 'High TP → each token uses all 10 devices → fastest per-token latency',
        ha='center', va='center', fontsize=9, color='#4B5563', style='italic')

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL: Pool T (Throughput)
# ═══════════════════════════════════════════════════════════════════════════════
T_LEFT, T_RIGHT = 9.3, 17.7
T_MID = (T_LEFT + T_RIGHT) / 2

# Panel background
bg2 = FancyBboxPatch((T_LEFT, 0.4), T_RIGHT - T_LEFT, 8.45,
                     boxstyle='round,pad=0.1',
                     facecolor=TPUT_COLOR, edgecolor=TPUT_EDGE,
                     linewidth=2.5, alpha=0.35, zorder=0)
ax.add_patch(bg2)

ax.text(T_MID, 8.65, '[T] Pool T — Throughput Pool',
        ha='center', va='center', fontsize=14, fontweight='bold', color=TPUT_EDGE)
ax.text(T_MID, 8.32, 'N_T = 10 devices  |  TP = 1  |  PP = 40',
        ha='center', va='center', fontsize=11, color='#78350F')

# Batch request token
ax.text(T_MID, 7.92,
        '⬇  Batch Requests  (high-throughput workload)',
        ha='center', va='center', fontsize=10.5, color=TPUT_EDGE,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFBEB',
                  edgecolor=TPUT_EDGE, linewidth=1.5))

# 10 devices in a pipeline (vertical stack, each handles 4 layers)
# Show 10 device stages stacked vertically
pipe_top = 7.52
stage_h  = 0.40
stage_gap = 0.05
pipe_w = T_RIGHT - T_LEFT - 1.0
pipe_left = T_LEFT + 0.5

# Token positions (4 tokens at different pipeline stages for illustration)
token_positions = [0, 3, 6, 9]  # at stage indices

for i in range(10):
    sy = pipe_top - i * (stage_h + stage_gap)
    layer_start = i * 4 + 1
    layer_end = layer_start + 3

    # Stage box
    fc = '#FDE68A' if i % 2 == 0 else '#FEF3C7'
    stage_rect = FancyBboxPatch((pipe_left, sy - stage_h), pipe_w, stage_h,
                                boxstyle='round,pad=0.03',
                                facecolor=fc, edgecolor=TPUT_EDGE,
                                linewidth=1.2, zorder=3)
    ax.add_patch(stage_rect)
    ax.text(pipe_left + 0.3, sy - stage_h/2,
            f'Dev {i+1}',
            ha='left', va='center', fontsize=8.5, fontweight='bold', color='#78350F')
    ax.text(pipe_left + pipe_w/2 + 0.2, sy - stage_h/2,
            f'Layers {layer_start}–{layer_end}  (TP=1, no sharding)',
            ha='center', va='center', fontsize=8.5, color='#374151')

    # Show token marker if token is at this stage
    if i in token_positions:
        tok_x = pipe_left + pipe_w - 0.25
        ax.scatter([tok_x], [sy - stage_h/2], s=120, color=TOKEN_EDGE,
                   zorder=6, marker='D')
        ax.text(tok_x + 0.12, sy - stage_h/2,
                f'Tok {token_positions.index(i)+1}',
                va='center', fontsize=7.5, color=TOKEN_EDGE, fontweight='bold')

# Pipeline arrows (between stages)
for i in range(9):
    sy = pipe_top - i * (stage_h + stage_gap) - stage_h - stage_gap/2
    ax.annotate('', xy=(T_MID, sy - 0.02), xytext=(T_MID, sy + 0.02),
                arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=0.8))

# Compute bottom of pipeline dynamically
bottom_pipe = pipe_top - 9 * (stage_h + stage_gap) - stage_h

# "4 tokens" label directly below pipeline, with gap
tokens_label_y = bottom_pipe - 0.22
ax.text(T_MID, tokens_label_y,
        '4 tokens in-flight simultaneously  (pipeline parallelism)',
        ha='center', va='center', fontsize=9.5, color='#92400E', fontweight='bold',
        style='italic')

# Result box — placed below tokens label with clear gap
rb_top = tokens_label_y - 0.30
result2 = FancyBboxPatch((T_LEFT + 0.3, rb_top - 0.65), T_RIGHT - T_LEFT - 0.6, 0.65,
                         boxstyle='round,pad=0.07',
                         facecolor='#FFFBEB', edgecolor=TPUT_EDGE, linewidth=2.0)
ax.add_patch(result2)
ax.text(T_MID, rb_top - 0.22, 'Pool T tput = 2116 tok/s  (high pipeline throughput)',
        ha='center', va='center', fontsize=11, fontweight='bold', color=TPUT_EDGE)
ax.text(T_MID, rb_top - 0.50, 'TTFT >> 1s  (batch/offline only — not SLO-critical)',
        ha='center', va='center', fontsize=9.5, color='#374151')

ax.text(T_MID, rb_top - 0.85,
        'High PP → many tokens in-flight → maximizes device utilization',
        ha='center', va='center', fontsize=9, color='#4B5563', style='italic')

# ── Bottom: Combined system_tput ──────────────────────────────────────────────
result3 = FancyBboxPatch((2.5, 0.45), 13, 0.62,
                         boxstyle='round,pad=0.07',
                         facecolor='#F0FDF4', edgecolor='#166534', linewidth=2.0)
ax.add_patch(result3)
ax.text(9, 0.76,
        'DDA System Throughput = Pool_L_tput + Pool_T_tput = 390 + 2116 = 2506 tok/s'
        '    vs    Best static (PP=2,TP=10): 780 tok/s    →    3.2× gain',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color='#166534')

plt.tight_layout(pad=0)
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure_slide_dda_pools.pdf', bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved: figures/figure_slide_dda_pools.pdf")
