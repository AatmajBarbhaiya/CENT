"""
figure_dda_strategy3_v2.py — DDA Strategy 3: α-adaptive Pareto-optimal split selection.

Only 2 Pareto-efficient splits exist for 13B:
  N_L=6  (N_T=14, Pool_T=2666, Pool_L=354, TTFT_L=1.003s): best at low α (large Pool T)
  N_L=10 (N_T=10, Pool_T=2116, Pool_L=390, TTFT_L=0.882s): best at high α (large Pool L)
  Crossover at α=0.143 (λ_max(6,α) = λ_max(10,α)).

Strategy comparison:
  S1 = N_L=6  (Strategy-1 best-gain, static)  → TTFT=1.003s, sys_tput=3020 tok/s
  S2 = N_L=10 (Strategy-2 SLO-optimal, static) → TTFT=0.882s, sys_tput=2506 tok/s
  S3 = adaptive: N_L=6 when α<0.143, N_L=10 when α≥0.143
       → Best of both: S1's throughput at low α, S2's SLO compliance at high α.

α ramp: 0.05 → 0.35 → 0.05.  α=0.143 crossover is visible during ramp.

M/M/1: R_L = TTFT_L + ρ/(μ_L × (1−ρ)), where ρ=α×λ/μ_L.
At λ=350 tok/s all ρ<<1 → queue negligible → TTFT_L dominates R_L.

Panels:
  (a) α(t) ramp + crossover threshold marker
  (b) System throughput = Pool_L_tput + Pool_T_tput
      S3 matches S1 at low α, drops to S2 level at high α
  (c) Interactive response time R_L + 1s SLO line
      S1 always violates; S2 always meets; S3 adapts
  (d) S3 N_L(t) — single step 6→10 at α=0.143, back 10→6 on descent

Run from repo root: python3 figure_scripts/figure_dda_strategy3_v2.py
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

COMBINED = pd.read_csv('results/combined_metrics.csv')

# ── Parameters ─────────────────────────────────────────────────────────────────
MODEL   = 'Llama2-13B'
LAMBDA  = 350.0
DT      = 0.1
T_MAX   = 80.0

ALPHA_BASE = 0.05
ALPHA_PEAK = 0.35
RAMP_UP_START, RAMP_UP_END = 15.0, 35.0
HOLD_END                   = 55.0
RAMP_DOWN_END              = 70.0
SLO_INTER                  = 1.0

# Fixed strategy N_L
NL_S1 = 6    # Strategy-1 best-gain
NL_S2 = 10   # Strategy-2 SLO-optimal (actual S2 result from figure_dda_strategy2.py)

# Pareto-efficient N_L for 13B: {6, 10} (all others dominated)
PARETO_NL = [6, 10]

# ── Load split data ─────────────────────────────────────────────────────────────
m = COMBINED[COMBINED['Model'] == MODEL].sort_values('N_L').reset_index(drop=True)
SPLITS = {int(r['N_L']): {'TTFT_L':    r['TTFT_short'],
                           'mu_L':      r['Tput_L'],
                           'mu_T':      r['Tput_T'],
                           'sys_tput':  r['system_tput'],
                           'slo_1s':    r['SLO_1s_short']}
          for _, r in m.iterrows()}


def lambda_max(nl, alpha):
    s = SPLITS[nl]
    if alpha <= 0: return s['mu_T']
    if alpha >= 1: return s['mu_L']
    return min(s['mu_L'] / alpha, s['mu_T'] / (1 - alpha))


# Crossover between N_L=6 and N_L=10
lo, hi = 0.001, 0.999
for _ in range(60):
    mid = (lo + hi) / 2
    if lambda_max(6, mid) > lambda_max(10, mid):
        lo = mid
    else:
        hi = mid
ALPHA_CROSSOVER = mid
print(f"13B Pareto crossover: α={ALPHA_CROSSOVER:.4f}  (N_L=6 → N_L=10)")
print(f"  N_L=6:  TTFT={SPLITS[6]['TTFT_L']:.3f}s  sys_tput={SPLITS[6]['sys_tput']:.0f}")
print(f"  N_L=10: TTFT={SPLITS[10]['TTFT_L']:.3f}s  sys_tput={SPLITS[10]['sys_tput']:.0f}")


def s3_target_nl(alpha):
    """Pareto-optimal N_L: N_L=6 below crossover, N_L=10 above."""
    return 10 if alpha >= ALPHA_CROSSOVER else 6


def r_l(nl, alpha):
    """M/M/1 interactive response time."""
    s   = SPLITS[nl]
    rho = alpha * LAMBDA / s['mu_L']
    if rho >= 1.0:
        return 5.0
    return s['TTFT_L'] + rho / (s['mu_L'] * (1 - rho))


def alpha_at(t):
    if t < RAMP_UP_START:  return ALPHA_BASE
    if t < RAMP_UP_END:    return ALPHA_BASE + (ALPHA_PEAK - ALPHA_BASE) * (t - RAMP_UP_START) / (RAMP_UP_END - RAMP_UP_START)
    if t < HOLD_END:       return ALPHA_PEAK
    if t < RAMP_DOWN_END:  return ALPHA_PEAK - (ALPHA_PEAK - ALPHA_BASE) * (t - HOLD_END) / (RAMP_DOWN_END - HOLD_END)
    return ALPHA_BASE


def simulate(nl_fixed=None, use_s3=False):
    times, R_L_arr, NL_arr, TPUT_arr = [], [], [], []
    nl       = nl_fixed if nl_fixed else NL_S1
    cooldown = 0.0
    COOLDOWN = 2.0

    for step in range(int(T_MAX / DT)):
        t     = step * DT
        alpha = alpha_at(t)

        RL   = r_l(nl, alpha)
        tput = SPLITS[nl]['sys_tput']   # static system_tput for this split

        if use_s3:
            cooldown = max(0.0, cooldown - DT)
            if cooldown <= 0:
                target = s3_target_nl(alpha)
                if target != nl:
                    nl = target
                    cooldown = COOLDOWN

        times.append(t)
        R_L_arr.append(RL)
        NL_arr.append(nl)
        TPUT_arr.append(SPLITS[nl]['sys_tput'])

    return (np.array(times), np.array(R_L_arr),
            np.array(NL_arr), np.array(TPUT_arr))


t_s1, R_s1, NL_s1, T_s1 = simulate(nl_fixed=NL_S1)
t_s2, R_s2, NL_s2, T_s2 = simulate(nl_fixed=NL_S2)
t_s3, R_s3, NL_s3, T_s3 = simulate(use_s3=True)

ts     = np.array([step * DT for step in range(int(T_MAX / DT))])
alphas = np.array([alpha_at(t) for t in ts])

# ── Figure ─────────────────────────────────────────────────────────────────────
# Create a 2x2 grid and adjust the figsize to be taller and slightly narrower
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Flatten the 2D axes array into a 1D list so we can unpack it the exact same way
ax_a, ax_b, ax_c, ax_d = axes.flatten()
COL_S1, COL_S2, COL_S3 = 'steelblue', 'seagreen', '#cc0000'

def shade_ramp(ax):
    ax.axvspan(RAMP_UP_START, RAMP_DOWN_END, alpha=0.10, color='tomato', zorder=0)

# (a) α(t)
shade_ramp(ax_a)
ax_a.plot(ts, alphas, color='black', linewidth=2)
ax_a.axhline(y=ALPHA_CROSSOVER, color='purple', linestyle='--', linewidth=1.5,
             label=f'Crossover α={ALPHA_CROSSOVER:.2f}\n(N_L=6→10)')
ax_a.text(2, ALPHA_CROSSOVER + 0.005, f'α={ALPHA_CROSSOVER:.2f} → N_L switches',
          fontsize=8, color='purple', va='bottom')
ax_a.set_xlabel('Time (s)', fontsize=11)
ax_a.set_ylabel('α (interactive fraction)', fontsize=11)
ax_a.set_title(f'(a) Workload Mix α(t)\nRamp {ALPHA_BASE}→{ALPHA_PEAK}→{ALPHA_BASE}', fontsize=10.5)
ax_a.set_ylim(0, ALPHA_PEAK * 1.3)
ax_a.set_xlim(0, T_MAX)
ax_a.legend(fontsize=8.5, loc='upper right')
ax_a.grid(linestyle='--', alpha=0.4)
ax_a.tick_params(labelsize=10)

# (b) System throughput
shade_ramp(ax_b)
ax_b.plot(t_s1, T_s1, color=COL_S1, linewidth=2.0,
          label=f'S1 N_L={NL_S1} (fixed)')
ax_b.plot(t_s2, T_s2, color=COL_S2, linewidth=2.0,
          label=f'S2 N_L={NL_S2} (fixed)')
ax_b.plot(t_s3, T_s3, color=COL_S3, linewidth=2.5,
          label='S3 adaptive', zorder=5)
# Annotate throughput values
for nl, col in [(NL_S1, COL_S1), (NL_S2, COL_S2)]:
    ax_b.text(T_MAX - 1, SPLITS[nl]['sys_tput'] + 20,
              f"{SPLITS[nl]['sys_tput']:.0f}", ha='right', color=col, fontsize=8.5)
ax_b.set_xlabel('Time (s)', fontsize=11)
ax_b.set_ylabel('System Throughput (tok/s)', fontsize=11)
ax_b.set_title('(b) System Throughput\nS3 = S1 at low α; = S2 at high α', fontsize=10.5)
ax_b.set_xlim(0, T_MAX)
ax_b.set_ylim(bottom=0)
ax_b.legend(fontsize=8.5)
ax_b.grid(linestyle='--', alpha=0.4)
ax_b.tick_params(labelsize=10)

# (c) R_L
shade_ramp(ax_c)
ax_c.plot(t_s1, R_s1, color=COL_S1, linewidth=2.0, label=f'S1 N_L={NL_S1}')
ax_c.plot(t_s2, R_s2, color=COL_S2, linewidth=2.0, label=f'S2 N_L={NL_S2}')
ax_c.plot(t_s3, R_s3, color=COL_S3, linewidth=2.5, label='S3 adaptive', zorder=5)
ax_c.axhline(y=SLO_INTER, color='red', linestyle='--', linewidth=1.8, label='1s SLO')
ax_c.set_xlabel('Time (s)', fontsize=11)
ax_c.set_ylabel('Response Time R_L (s)', fontsize=11)
ax_c.set_title('(c) Interactive Response Time R_L\n'
               'S1 always violates; S2 always meets; S3 adapts', fontsize=10.5)
ax_c.set_xlim(0, T_MAX)
ax_c.set_ylim(bottom=0)
ax_c.legend(fontsize=8.5)
ax_c.grid(linestyle='--', alpha=0.4)
ax_c.tick_params(labelsize=10)

# (d) S3 N_L
shade_ramp(ax_d)
ax_d.step(t_s3, NL_s3, where='post', color=COL_S3, linewidth=2.5,
          label='S3 N_L(t)')
ax_d.axhline(y=NL_S1, color=COL_S1, linestyle='--', linewidth=1.5,
             label=f'S1 fixed N_L={NL_S1}')
ax_d.axhline(y=NL_S2, color=COL_S2, linestyle='--', linewidth=1.5,
             label=f'S2 fixed N_L={NL_S2}')
ax_d.axhline(y=ALPHA_CROSSOVER * 0 + 0, alpha=0)   # invisible, just for spacing
# Mark crossover transition
for t_idx in range(1, len(NL_s3)):
    if NL_s3[t_idx] != NL_s3[t_idx - 1]:
        ax_d.axvline(x=ts[t_idx], color='purple', linestyle=':', linewidth=1.5, alpha=0.8)
ax_d.set_xlabel('Time (s)', fontsize=11)
ax_d.set_ylabel('N_L (Pool L devices)', fontsize=11)
ax_d.set_title('(d) S3 Dynamic N_L\nN_L=6↔10 at α=0.14 crossover', fontsize=10.5)
ax_d.set_xlim(0, T_MAX)
ax_d.set_ylim(0, 14)
ax_d.legend(fontsize=8.5)
ax_d.grid(linestyle='--', alpha=0.4)
ax_d.tick_params(labelsize=10)

plt.suptitle(f'DDA Strategy 3 — Pareto-Adaptive Split Selection ({MODEL})\n'
             f'λ={LAMBDA:.0f} tok/s | Only 2 Pareto-efficient splits: '
             f'N_L=6 (tput-opt) ↔ N_L=10 (SLO-opt), crossover at α={ALPHA_CROSSOVER:.3f}',
             fontsize=11)
plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figure_dda_strategy3_v2.pdf', bbox_inches='tight')
print("Saved: figures/figure_dda_strategy3_v2.pdf")

print(f"\nSummary (during ramp α={ALPHA_BASE}→{ALPHA_PEAK}→{ALPHA_BASE}):")
ramp_mask  = (ts >= RAMP_UP_START) & (ts < RAMP_DOWN_END)
low_mask   = (ts < RAMP_UP_START) | (ts >= RAMP_DOWN_END)
for name, R, NL, T in [('S1', R_s1, NL_s1, T_s1),
                        ('S2', R_s2, NL_s2, T_s2),
                        ('S3', R_s3, NL_s3, T_s3)]:
    viol  = (R[ramp_mask] > SLO_INTER).mean() * 100
    t_low = T[low_mask].mean()
    t_ramp= T[ramp_mask].mean()
    nl_r  = f"[{NL[ramp_mask].min()},{NL[ramp_mask].max()}]"
    print(f"  {name}: SLO_viol_ramp={viol:.0f}%  tput_low={t_low:.0f}  tput_ramp={t_ramp:.0f}  N_L_ramp={nl_r}")
