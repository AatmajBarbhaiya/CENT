"""
figure_slide_dda_style.py — DDA split slide matching presentation style.

Matches the TP-PP hybrid slide: white bg, purple header, numbered boxes,
colored pools, simple arrows. Shows 70B DDA: N_L=12 (Pool L) + N_T=20 (Pool T).

Pool L: 12 devices, TP=12, PP=1  — salmon boxes, all-reduce arrow
Pool T: 20 devices, TP=1, PP=80  — blue boxes stacked, tokens flowing

Run from repo root: python3 figure_scripts/figure_slide_dda_style.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Purple header bar ─────────────────────────────────────────────────────────
ax.add_patch(mpatches.Rectangle((0, 8.25), 16, 0.75,
                                 facecolor='#5B2C8D', zorder=10))
ax.text(8, 8.62, 'DDA Split: Pool L (TP=12, PP=1) + Pool T (TP=1, PP=80)',
        ha='center', va='center', fontsize=20, fontweight='bold',
        color='white', zorder=11)

# Bottom purple bar
ax.add_patch(mpatches.Rectangle((0, 0), 16, 0.22, facecolor='#5B2C8D', zorder=10))

# ── Sub-title ─────────────────────────────────────────────────────────────────
ax.text(8, 8.08, 'Llama2-70B  |  80 Decoder Layers  |  32 CXL Devices split: N_L=12 + N_T=20',
        ha='center', va='center', fontsize=11.5, color='#444444')

# ── Divider ───────────────────────────────────────────────────────────────────
ax.plot([8.1, 8.1], [0.25, 7.95], color='#CCCCCC', linewidth=1.5, linestyle='--')
ax.text(8.1, 7.82, 'device split', ha='center', fontsize=9, color='#888888')

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT: Pool L — 12 devices, TP=12, PP=1
# Salmon boxes matching original slide style
# ═══════════════════════════════════════════════════════════════════════════════
SALMON   = '#F4A0A0'   # regular TP device
SALMON_H = '#E05050'   # master/highlighted device
BLUE_H   = '#4A90D9'   # Pool T highlighted
BLUE_L   = '#A8CFF0'   # Pool T regular
ARROW    = '#3A7FD5'   # arrow color
TEXT_BOX_LAT  = '#FFF3CD'  # amber for latency result
TEXT_BOX_TPUT = '#D4EDDA'  # green for throughput result

def device_box(ax, x, y, w, h, num, color, fontsize=14):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.07',
                          facecolor=color, edgecolor='#333333',
                          linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, str(num),
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#111111')

def thick_arrow(ax, x1, y1, x2, y2, color=ARROW, lw=3.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=22))

# Pool L header
ax.text(4.0, 7.62, 'Pool L — Latency Pool',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#C0392B')
ax.text(4.0, 7.32, 'N_L = 12 devices  |  TP = 12  |  PP = 1',
        ha='center', va='center', fontsize=11, color='#555555')

# Input arrow
thick_arrow(ax, 4.0, 7.08, 4.0, 6.82)
ax.add_patch(FancyBboxPatch((2.5, 6.82), 3.0, 0.35,
                            boxstyle='round,pad=0.05',
                            facecolor='#FFA500', edgecolor='#CC7700', linewidth=1.5))
ax.text(4.0, 6.995, 'Interactive Request',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color='white')

# 12 devices: 2 rows of 6
# Stage box (rounded rect grouping)
ax.add_patch(FancyBboxPatch((0.25, 4.35), 7.65, 2.22,
                            boxstyle='round,pad=0.1',
                            facecolor='#FFF5F5', edgecolor='#E05050',
                            linewidth=2.0, linestyle='-'))

dw, dh = 1.08, 0.75
gap = 0.12
sx = 0.45
for row in range(2):
    for col in range(6):
        devn = row * 6 + col + 1
        x = sx + col * (dw + gap)
        y = 5.52 - row * (dh + 0.16)
        c = SALMON_H if devn == 1 else SALMON
        device_box(ax, x, y, dw, dh, devn, c, fontsize=13)

# All-reduce arrow
ax.annotate('', xy=(7.6, 4.78), xytext=(0.45, 4.78),
            arrowprops=dict(arrowstyle='<->', color='#F59E0B', lw=2.2))
ax.text(4.0, 4.60, 'Broadcast  →  parallel GEMV  →  All-Reduce (CXL)',
        ha='center', va='center', fontsize=9, color='#92400E', fontweight='bold')

# Stage label
ax.text(4.0, 4.42, 'Stage 1: Each device holds 1/12 of FC weights (PP=1 — all 80 layers in one pass)',
        ha='center', va='center', fontsize=9.5, color='#333333')

# Output arrow + result
thick_arrow(ax, 4.0, 4.30, 4.0, 3.88)

ax.add_patch(FancyBboxPatch((1.0, 3.22), 6.0, 0.70,
                            boxstyle='round,pad=0.07',
                            facecolor=TEXT_BOX_LAT, edgecolor='#C05050',
                            linewidth=1.8))
ax.text(4.0, 3.68, 'TTFT = 3.363 s  ✓  Meets 5s SLO',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#C0392B')
ax.text(4.0, 3.40, 'Pool L tput = 101.6 tok/s  |  1 token in-flight at a time',
        ha='center', va='center', fontsize=10, color='#555555')

# Single token visual
for i, (tx, ta) in enumerate([(3.0, 1.0), (4.0, 0.65), (5.0, 0.35)]):
    ax.scatter([tx], [2.88], s=200, color='#C0392B', alpha=ta, zorder=5)
ax.text(4.0, 2.62, 'Tokens served one at a time  (no pipeline)',
        ha='center', va='center', fontsize=9.5, color='#555555', style='italic')

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT: Pool T — 20 devices, TP=1, PP=80, 4 layers each
# Blue boxes stacked showing pipeline, tokens at different stages
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(12.35, 7.62, 'Pool T — Throughput Pool',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#1A5276')
ax.text(12.35, 7.32, 'N_T = 20 devices  |  TP = 1  |  PP = 80',
        ha='center', va='center', fontsize=11, color='#555555')

thick_arrow(ax, 12.35, 7.08, 12.35, 6.82)
ax.add_patch(FancyBboxPatch((10.55, 6.82), 3.6, 0.35,
                            boxstyle='round,pad=0.05',
                            facecolor='#2E86C1', edgecolor='#1A5276', linewidth=1.5))
ax.text(12.35, 6.995, 'Batch Request (many tokens)',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color='white')

# 20 devices stacked — pipeline diagram
dw_t = 4.5
dh_t = 0.22
gap_t = 0.022
tx_start = 8.35
pipe_top_y = 6.72

tok_colors = {0: '#8E44AD', 5: '#8E44AD', 10: '#8E44AD', 15: '#8E44AD'}

for i in range(20):
    x = tx_start
    y = pipe_top_y - i * (dh_t + gap_t)
    ls = i * 4 + 1; le = ls + 3
    c = BLUE_H if i % 5 == 0 else BLUE_L
    rect = FancyBboxPatch((x, y - dh_t), dw_t, dh_t,
                          boxstyle='round,pad=0.03',
                          facecolor=c, edgecolor='#1A5276',
                          linewidth=1.1)
    ax.add_patch(rect)
    ax.text(x + 0.28, y - dh_t/2, f'Dev {i+1:2d}',
            ha='left', va='center', fontsize=8, fontweight='bold', color='#1A3A5C')
    ax.text(x + dw_t/2 + 0.4, y - dh_t/2,
            f'Layers {ls:2d}–{le:2d}',
            ha='center', va='center', fontsize=8, color='#1A3A5C')

    # Token markers
    if i in [0, 5, 10, 15]:
        tok_n = i // 5 + 1
        ax.text(x + dw_t + 0.1, y - dh_t/2,
                f'← Token {tok_n}',
                ha='left', va='center', fontsize=8.5,
                fontweight='bold', color='#8E44AD')

# Mini arrows between devices
for i in range(19):
    y = pipe_top_y - i * (dh_t + gap_t) - dh_t - gap_t/2
    ax.annotate('', xy=(10.1, y - 0.005), xytext=(10.1, y + 0.005),
                arrowprops=dict(arrowstyle='->', color='#7FB3D3', lw=0.8))

# Result box Pool T
pipe_bot = pipe_top_y - 19 * (dh_t + gap_t) - dh_t

ax.text(12.35, pipe_bot - 0.16,
        'Up to 20 tokens in pipeline simultaneously',
        ha='center', va='center', fontsize=9.5, color='#8E44AD',
        fontweight='bold', style='italic')

ax.add_patch(FancyBboxPatch((8.55, pipe_bot - 0.76), 7.6, 0.52,
                            boxstyle='round,pad=0.06',
                            facecolor=TEXT_BOX_TPUT, edgecolor='#1A7040',
                            linewidth=1.8))
ax.text(12.35, pipe_bot - 0.44, 'Pool T tput = 955.8 tok/s',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#1A5276')
ax.text(12.35, pipe_bot - 0.67, 'TTFT >> 5s  |  batch / offline only  |  1 output/device-step at steady state',
        ha='center', va='center', fontsize=9, color='#333333')

# ── Bottom result bar ─────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.5, 0.28), 15, 0.60,
                            boxstyle='round,pad=0.06',
                            facecolor='#EAF5EA', edgecolor='#1A7040', linewidth=2.0))
ax.text(8, 0.58,
        'DDA System Throughput = 101.6 + 955.8 = 1057 tok/s'
        '    vs    Static (PP=2, TP=16): 216 tok/s    →    4.9× gain',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#1A5276')

plt.tight_layout(pad=0)
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure_slide_dda_style.pdf', bbox_inches='tight',
            facecolor='white')
print("Saved: figures/figure_slide_dda_style.pdf")
