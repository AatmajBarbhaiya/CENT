"""
figure_dda_strategy2_alpha.py

Panel (a): Optimal N_L vs α — capacity-driven split selection.
           λ_max(N_L, α) = min(Pool_L_tput/α, Pool_T_tput/(1-α))
           Optimal N_L = argmax λ_max at each α.
           Regions annotated with SLO status (1s vs 5s).

Panel (b): Long-seq Pool T throughput vs seqlen.
           S2 uses the N_T paired with optimal N_L at α=0.5.
           Compares S2 Pool T vs S1 Pool T vs balanced static.

Panel (c): Long-seq Pool T TTFT vs seqlen (log scale), same three configs.

Key insight: S2's optimal N_L shifts right (more Pool L devices) as α grows,
because Pool L becomes the capacity bottleneck at high interactive fractions.
Crossover points (where optimal N_L jumps) are model-specific.

Run from repo root: python3 figure_scripts/figure_dda_strategy2_alpha.py
"""

import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

COMBINED = pd.read_csv('results/combined_metrics.csv')
SIM_BASE = pd.read_csv('../../cent_simulation/simulation_results.csv')
RESULTS  = 'results'

MODELS    = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
COLORS    = {'Llama2-7B': 'steelblue', 'Llama2-13B': 'darkorange', 'Llama2-70B': 'forestgreen'}
SLO_COLORS = {
    '1s': '#d1f2eb',  # Light mint/green
    '5s': '#fef9e7',  # Light yellow
    'NoSLO': '#fdedec' # Light red (optional)
}
LAYERS    = {'Llama2-7B': 32, 'Llama2-13B': 40, 'Llama2-70B': 80}
TOTAL_DEV = {'Llama2-7B': 8,  'Llama2-13B': 20, 'Llama2-70B': 32}
STATIC_CMP = {'Llama2-7B': (4,2), 'Llama2-13B': (2,10), 'Llama2-70B': (2,16)}
S1_SPLIT  = {'Llama2-7B': (2,6), 'Llama2-13B': (6,14), 'Llama2-70B': (12,20)}
T_SHORT   = 512
ALPHA_REF = 0.35   # reference α for long-seq panel

# ── Helpers ────────────────────────────────────────────────────────────────────

def lambda_max(pl, pt, alpha):
    """Max sustainable request rate given Pool L tput pl, Pool T tput pt, fraction α."""
    if alpha <= 0: return pt
    if alpha >= 1: return pl
    return min(pl / alpha, pt / (1 - alpha))


def optimal_nl_curve(model, alphas):
    """Optimal N_L at each α (max λ_max). Returns arrays: nl, lambda_max, slo_status."""
    m = COMBINED[COMBINED['Model'] == model]
    splits = [(int(r['N_L']), r['Tput_L'], r['Tput_T'],
               r['SLO_1s_short'], r['SLO_5s_short']) for _, r in m.iterrows()]
    nl_arr, lam_arr, slo_arr = [], [], []
    for alpha in alphas:
        best_lam, best_nl, best_slo = 0, None, None
        for nl, pl, pt, slo1, slo5 in splits:
            lam = lambda_max(pl, pt, alpha)
            if lam > best_lam:
                best_lam = lam
                best_nl = nl
                best_slo = '1s' if slo1 else ('5s' if slo5 else 'NoSLO')
        nl_arr.append(best_nl)
        lam_arr.append(best_lam)
        slo_arr.append(best_slo)
    return np.array(nl_arr), np.array(lam_arr), slo_arr


def pool_T_series(dfT, lay, seqlen_min=T_SHORT):
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


# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.5))
alphas = np.linspace(0.001, 0.999, 500)

# ══════════════════════════════════════════════════════════════════════════════
# Panel (a): Optimal N_L vs α
# ══════════════════════════════════════════════════════════════════════════════
for model in MODELS:
    col = COLORS[model]
    nl_arr, lam_arr, slo_arr = optimal_nl_curve(model, alphas)

    slo_arr_np = np.array(slo_arr)

    # Draw full path dashed first (base layer — no gaps)
    ax1.step(alphas, nl_arr, where='post', color=col, linewidth=1.8, linestyle='--',
             label=model.replace('Llama2-', 'L2-') + ' — solid=1s✓, dashed=5s✓')

    # Overdraw solid where SLO-1s is met (on top, no gap risk)
    i = 0
    while i < len(alphas) - 1:
        if slo_arr_np[i] == '1s':
            j = i + 1
            while j < len(alphas) and slo_arr_np[j] == '1s':
                j += 1
            # include one extra point so the step bridges correctly
            seg_a  = alphas[max(0, i-1):j+1]
            seg_nl = nl_arr[max(0, i-1):j+1]
            ax1.step(seg_a, seg_nl, where='post', color=col, linewidth=2.5, linestyle='-')
            i = j
        else:
            i += 1

    # Single light background shade: where ANY model hits 1s-SLO
    # Only 13B and 7B ever reach 1s — shade once globally (done after loop)

    # Crossover annotations — no vertical lines
    # 70B: place left of transition; 13B: right; 7B: static label
    prev_nl = nl_arr[0]
    for k, (a, nl, slo) in enumerate(zip(alphas, nl_arr, slo_arr)):
        if nl != prev_nl:
            slo_label = '1s✓' if slo == '1s' else ('5s✓' if slo == '5s' else '✗')
            if model == 'Llama2-70B':
                ax1.text(a + 0.02, nl + 0.2,
                         f'N_L={nl} ({slo_label})',
                         ha='right', fontsize=6.5, color=col, va='bottom')
            else:
                ax1.text(a + 0.015, nl - 0.4,
                         f'N_L={nl} ({slo_label})',
                         ha='left', fontsize=6.5, color=col, va='top')
            prev_nl = nl

    # Static 7B label (never changes)
    if model == 'Llama2-7B':
        ax1.text(0.97, nl_arr[-1] + 0.3,
                 'N_L=2 (1s✓)\n7B fixed',
                 ha='right', va='bottom', fontsize=6.5, color=col)

    # Mark α=0.5 reference
    nl_at_ref = nl_arr[np.argmin(np.abs(alphas - ALPHA_REF))]
    ax1.scatter([ALPHA_REF], [nl_at_ref], color=col, s=80,
                zorder=8, marker='v', edgecolors='black', linewidths=0.8)

# Background shade: green where 13B (and 7B) meet interactive 1s SLO
# 7B: always meets 1s (all α). 13B: meets from α≈0.15 onward. 70B: never.
# Shade the region α≥first-1s-crossover for 13B
nl_13b, _, slo_13b = optimal_nl_curve('Llama2-13B', alphas)
idx_1s = np.where(np.array(slo_13b) == '1s')
if len(idx_1s[0]) > 0:
    a_1s_start = alphas[idx_1s[0][0]]
    ax1.axvspan(a_1s_start, 1.0, alpha=0.08, color='limegreen', zorder=-1,
                label='13B meets 1s SLO (solid line)')
    ax1.axvspan(0, a_1s_start, alpha=0.06, color='gold', zorder=-1,
                label='13B/70B SLO-5s only (dashed line)')

ax1.axvline(x=ALPHA_REF, color='gray', linestyle='--', linewidth=1.2, alpha=0.7,
            label=f'α={ALPHA_REF} (ref for b+c)')
ax1.set_xlabel('α — Fraction of Interactive (SLO=1s) Requests', fontsize=11)
ax1.set_ylabel('Optimal N_L (devices in Pool L)', fontsize=11)
ax1.set_title('(a) S2 Optimal Pool L Size vs Workload Mix α\n',
              fontsize=10)
ax1.legend(fontsize=8, loc='upper right')
ax1.set_xlim(0, 1)
ax1.grid(linestyle='--', alpha=0.4)
ax1.tick_params(labelsize=10)


# ══════════════════════════════════════════════════════════════════════════════
# Panels (b)+(c): Long-seq Pool T comparison at α=ALPHA_REF
# ══════════════════════════════════════════════════════════════════════════════
LSTYLE = {'Llama2-7B': '-', 'Llama2-13B': '--', 'Llama2-70B': (0,(3,1,1,1))}
COL_STATIC = 'steelblue'
COL_S1     = 'seagreen'
COL_S2     = '#cc0000'
LW = 2.0

for model in MODELS:
    lay  = LAYERS[model]
    ls   = LSTYLE[model]
    b    = SIM_BASE[SIM_BASE['Model'] == model]
    pp_s, tp_s = STATIC_CMP[model]
    short = model.replace('Llama2-', 'L2-')

    # S1 N_T (from S1 split)
    NL_s1, NT_s1 = S1_SPLIT[model]
    # S2 N_T at α=ALPHA_REF
    nl_at_ref = optimal_nl_curve(model, np.array([ALPHA_REF]))[0][0]
    NT_s2 = TOTAL_DEV[model] - nl_at_ref

    dfT_s1 = pd.read_csv(f'{RESULTS}/sim_{model}_poolT_{NT_s1}.csv')
    dfT_s2 = pd.read_csv(f'{RESULTS}/sim_{model}_poolT_{NT_s2}.csv')

    # # Balanced static
    # ss, ttft_ss, tput_ss = static_series(b, pp_s, tp_s)
    # ax2.plot(ss, tput_ss, color=COL_STATIC, linestyle=ls, linewidth=LW, zorder=3,
    #          label=f'Static PP={pp_s},TP={tp_s} ({short})')
    # ax3.plot(ss, ttft_ss, color=COL_STATIC, linestyle=ls, linewidth=LW, zorder=3)

    # S1 Pool T
    ds1, ttft_s1, tput_s1 = pool_T_series(dfT_s1, lay)
    ax2.plot(ds1, tput_s1, color=COL_S1, linestyle=ls, linewidth=LW,
             label=f'S1 Pool T N_T={NT_s1} ({short})')
    ax3.plot(ds1, ttft_s1, color=COL_S1, linestyle=ls, linewidth=LW)

    # S2 Pool T
    ds2, ttft_s2, tput_s2 = pool_T_series(dfT_s2, lay)
    ax2.plot(ds2, tput_s2, color=COL_S2, linestyle=ls, linewidth=LW+0.5,
             label=f'S2 Pool T N_T={NT_s2} ({short}) [α={ALPHA_REF}]')
    ax3.plot(ds2, ttft_s2, color=COL_S2, linestyle=ls, linewidth=LW+0.5)

    # Shade: S2 gain over S1 in throughput
    min_len = min(len(ds1), len(ds2))
    ax2.fill_between(ds2[:min_len],
                     tput_s1[:min_len], tput_s2[:min_len],
                     alpha=0.10, color=COL_S2 if tput_s2.mean() > tput_s1.mean() else COL_S1)

    # Print NT comparison
    print(f'{model}: S1 N_T={NT_s1}, S2 N_T={NT_s2} at α={ALPHA_REF}')

ax2.set_xlabel('Sequence Length (tokens)', fontsize=11)
ax2.set_ylabel('Throughput (tok/s)', fontsize=11)
ax2.set_title(f'(b) Long-Seq Pool T Throughput\n'
              f'S2 N_T = optimal at α={ALPHA_REF}',
              fontsize=10)
ax2.set_xlim(T_SHORT, 4096)
ax2.set_ylim(bottom=0)
ax2.grid(linestyle='--', alpha=0.4)
ax2.tick_params(labelsize=10)

ax3.set_yscale('log')
ax3.set_xlabel('Sequence Length (tokens)', fontsize=11)
ax3.set_ylabel('TTFT (s, log scale)', fontsize=11)
ax3.set_title(f'(c) Long-Seq Pool T TTFT\n'
              f'S2 N_T = optimal at α={ALPHA_REF}',
              fontsize=10)
ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=1.2, alpha=0.8)
ax3.axhline(y=5.0, color='olive', linestyle=':',  linewidth=1.2, alpha=0.8)
ax3.set_xlim(T_SHORT, 4096)
ax3.grid(linestyle='--', alpha=0.4, which='both')
ax3.tick_params(labelsize=10)

# Shared legend for b+c
legend_bc = [
#    mlines.Line2D([0],[0], color=COL_STATIC, linewidth=2,  label='Balanced static'),
    mlines.Line2D([0],[0], color=COL_S1,     linewidth=2,  label='S1 Pool T (best-gain split)'),
    mlines.Line2D([0],[0], color=COL_S2,     linewidth=2.5,label=f'S2 Pool T (optimal at α={ALPHA_REF})'),
    mlines.Line2D([0],[0], color='gray', linewidth=2, linestyle='-',           label='L2-7B'),
    mlines.Line2D([0],[0], color='gray', linewidth=2, linestyle='--',          label='L2-13B'),
    mlines.Line2D([0],[0], color='gray', linewidth=2, linestyle=(0,(3,1,1,1)), label='L2-70B'),
    mlines.Line2D([0],[0], color='green', linewidth=1.2, linestyle='--', label='1s SLO'),
    mlines.Line2D([0],[0], color='olive', linewidth=1.2, linestyle=':',  label='5s SLO'),
]
fig.legend(handles=legend_bc, fontsize=8, loc='lower center',
           ncol=4, bbox_to_anchor=(0.65, -0.12))

plt.suptitle(f'DDA Strategy 2 — α-Dependent Split Selection & Long-Request Performance\n'
             f'Bimodal SLO: interactive=1s, batch=5s | '
             f'panels (b)+(c) at reference α={ALPHA_REF}',
             fontsize=11)
plt.tight_layout(rect=[0, 0.08, 1, 1])

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure_dda_strategy2_alpha.pdf', bbox_inches='tight')
print("Saved: figures/figure_dda_strategy2_alpha.pdf")
