"""
figure_slide_static_config.py — Presentation Slide 1.
Static Config: 32 CXL Devices, TP=16, PP=2 (Llama2-70B).
Layout: Stages on left (x=0.4..13.0), Config Summary on right (x=13.3..17.6).
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
S1_BG    = '#FFF9C4';  S1_EDGE  = '#1D4ED8'
S2_BG    = '#C8F7C5';  S2_EDGE  = '#166534'
DEV1_FC  = '#BFDBFE'
DEV2_FC  = '#BBF7D0'
DEV_EC   = '#374151'
CXL_COL  = '#F59E0B'
SUM_BG   = '#F1F5F9'
SUM_EDGE = '#475569'
TITLE_C  = '#111827'

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(6.7, 9.65, 'Static Hybrid TP-PP: 32 CXL Devices, TP=16, PP=2',
        ha='center', va='center', fontsize=19, fontweight='bold', color=TITLE_C)
ax.text(6.7, 9.28, 'Llama2-70B  |  80 Decoder Layers  |  Embedding dim = 8192',
        ha='center', va='center', fontsize=12, color='#4B5563')

# ── Stage dimensions ─────────────────────────────────────────────────────────
# Stages span x=[0.4, 13.1], Config Summary x=[13.4, 17.6]
SX_L, SX_R = 0.4, 13.1   # stage box x range
SX_MID = (SX_L + SX_R) / 2

def draw_devices(ax, n_per_row, dev_offset, row1_top, dw, dh, gap, fc, lbl_color):
    total_w = n_per_row * dw + (n_per_row - 1) * gap
    start_x = SX_L + (SX_R - SX_L - total_w) / 2
    for row in range(2):
        y_top = row1_top - row * (dh + 0.16)
        for i in range(n_per_row):
            x = start_x + i * (dw + gap)
            d = i + row * n_per_row + dev_offset
            r = FancyBboxPatch((x, y_top - dh), dw, dh,
                               boxstyle='round,pad=0.04',
                               facecolor=fc, edgecolor=DEV_EC, linewidth=1.2)
            ax.add_patch(r)
            ax.text(x + dw/2, y_top - dh/2, f'Dev {d}',
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color=lbl_color)
    return start_x, total_w, row1_top - 2 * dh - 0.16

def draw_stage(ax, stage_num, y_top, y_bot, layers, bg, edge, dev_fc,
               dev_lbl_color, dev_offset):
    # Stage bounding box
    r = FancyBboxPatch((SX_L, y_bot), SX_R - SX_L, y_top - y_bot,
                       boxstyle='round,pad=0.07',
                       facecolor=bg, edgecolor=edge, linewidth=2.5, zorder=1)
    ax.add_patch(r)

    # Stage vertical label on left margin
    ax.text(SX_L - 0.2, (y_top + y_bot)/2, f'Stage {stage_num}',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=edge, rotation=90)

    # Title inside box
    ax.text(SX_MID, y_top - 0.22, f'Pipeline Stage {stage_num}  —  {layers}',
            ha='center', va='center', fontsize=12, fontweight='bold', color=edge)

    # Devices: 8 per row, 2 rows
    dw, dh, gap = 1.35, 0.60, 0.10
    dev_row1_top = y_top - 0.52
    start_x, total_w, dev_bottom = draw_devices(
        ax, 8, dev_offset, dev_row1_top, dw, dh, gap, dev_fc, dev_lbl_color)

    # Weight label
    wlabel_y = dev_bottom - 0.16
    ax.text(SX_MID, wlabel_y,
            'Each device: 1/16 FC weight matrix rows  |  GEMV shard',
            ha='center', va='center', fontsize=9, color='#374151',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                      edgecolor=edge, alpha=0.85, linewidth=0.9))

    # All-reduce arrow
    ar_y = wlabel_y - 0.32
    x1, x2 = start_x - 0.1, start_x + total_w + 0.1
    ax.annotate('', xy=(x2, ar_y), xytext=(x1, ar_y),
                arrowprops=dict(arrowstyle='<->', color=CXL_COL, lw=2.0))
    ax.text(SX_MID, ar_y + 0.15,
            'Broadcast input  →  parallel GEMV  →  All-Reduce (CXL)',
            ha='center', va='center', fontsize=9.5,
            color='#92400E', fontweight='bold')

# ── Token input ───────────────────────────────────────────────────────────────
ax.text(SX_MID, 8.88, 'Single Token Request',
        ha='center', va='center', fontsize=12, color='#1E3A5F',
        bbox=dict(boxstyle='round,pad=0.32', facecolor='#EFF6FF',
                  edgecolor=S1_EDGE, linewidth=1.5))
ax.annotate('', xy=(SX_MID, 8.62), xytext=(SX_MID, 8.75),
            arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.8))

# ── Stage 1 ───────────────────────────────────────────────────────────────────
draw_stage(ax, 1, 8.60, 4.95, 'Layers 1–40',
           S1_BG, S1_EDGE, DEV1_FC, '#1E3A5F', dev_offset=1)

# ── Activation transfer ───────────────────────────────────────────────────────
ax.annotate('', xy=(SX_MID, 4.72), xytext=(SX_MID, 4.95),
            arrowprops=dict(arrowstyle='->', color='#6B7280', lw=2.0))
ax.text(SX_MID, 4.83,
        'Activation (8192-dim)  transferred via CXL switch  →  Stage 2',
        ha='center', va='center', fontsize=9.5, color='#374151',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#FEF3C7',
                  edgecolor=CXL_COL, alpha=0.9, linewidth=1.2))

# ── Stage 2 ───────────────────────────────────────────────────────────────────
draw_stage(ax, 2, 4.70, 0.90, 'Layers 41–80',
           S2_BG, S2_EDGE, DEV2_FC, '#14532D', dev_offset=17)

# ── Output ────────────────────────────────────────────────────────────────────
ax.annotate('', xy=(SX_MID, 0.65), xytext=(SX_MID, 0.90),
            arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.8))
ax.text(SX_MID, 0.50, 'Output Token',
        ha='center', va='center', fontsize=12, color='#14532D',
        bbox=dict(boxstyle='round,pad=0.32', facecolor='#F0FDF4',
                  edgecolor=S2_EDGE, linewidth=1.5))

# ── Config Summary (right column) ────────────────────────────────────────────
CX_L, CX_R = 13.4, 17.65
CX_MID = (CX_L + CX_R) / 2

# Summary background
sb = FancyBboxPatch((CX_L, 0.4), CX_R - CX_L, 8.50,
                    boxstyle='round,pad=0.1',
                    facecolor=SUM_BG, edgecolor=SUM_EDGE,
                    linewidth=2.0)
ax.add_patch(sb)

ax.text(CX_MID, 8.65, 'Config Summary',
        ha='center', va='center', fontsize=13, fontweight='bold', color=TITLE_C)

entries = [
    ('TP = 16', 'devices per pipeline stage'),
    ('PP = 2', 'pipeline stages total'),
    ('', ''),
    ('32 devices', '16 × 2 = total'),
    ('', ''),
    ('FC layers', '→ sharded across 16 devs'),
    ('Norm / Attn', '→ single master device'),
    ('', ''),
    ('Tokens in-flight', '2  (one per PP stage)'),
    ('TTFT', '≈ 2 × stage latency'),
    ('', ''),
    ('CXL comm.', 'All-Reduce per FC layer'),
    ('', '135 KB per transformer\nblock (Llama2-70B)'),
]

y = 8.22
for key, val in entries:
    if not key:
        y -= 0.10
        continue
    ax.text(CX_L + 0.3, y, key,
            ha='left', va='center', fontsize=10, fontweight='bold', color='#1E3A5F')
    ax.text(CX_R - 0.2, y, val,
            ha='right', va='center', fontsize=9, color='#374151')
    ax.plot([CX_L + 0.2, CX_R - 0.2], [y - 0.25, y - 0.25],
            color='#E2E8F0', linewidth=0.7)
    y -= 0.52

# Throughput insight box at bottom of summary
tb = FancyBboxPatch((CX_L + 0.15, 0.55), CX_R - CX_L - 0.30, 1.60,
                    boxstyle='round,pad=0.08',
                    facecolor='#DBEAFE', edgecolor=S1_EDGE, linewidth=1.5)
ax.add_patch(tb)
ax.text(CX_MID, 2.00, 'Why PP=2?',
        ha='center', va='center', fontsize=10, fontweight='bold', color=S1_EDGE)
ax.text(CX_MID, 1.60,
        'While Stage 2 processes\nToken A, Stage 1 starts\nToken B in parallel',
        ha='center', va='center', fontsize=9, color='#1E3A5F')
ax.text(CX_MID, 0.88,
        '→ 2× throughput vs PP=1',
        ha='center', va='center', fontsize=9.5, fontweight='bold', color=S1_EDGE)

plt.tight_layout(pad=0)
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure_slide_static_config.pdf', bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved: figures/figure_slide_static_config.pdf")
