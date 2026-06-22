"""
combine_pools.py — build DDA combined metrics from Pool L + Pool T simulation results.

For each valid (N_L, N_T) split per model:
  - Pool L lat-optimal config: PP=1, TP=N_L → TTFT_short, Tput_L
  - Pool T tput-optimal config: PP=num_layers, TP=1 → Tput_T
  - Both pools run concurrently → system_tput = Tput_L + Tput_T
  - TTFT_short = Pool L's prefill latency (short requests always routed to Pool L)

Output: ../results/combined_metrics.csv
"""

import os
import glob
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
OUT_PATH    = os.path.join(RESULTS_DIR, "combined_metrics.csv")

LAYERS     = {'Llama2-7B': 32, 'Llama2-13B': 40, 'Llama2-70B': 80}
TOTAL_DEV  = {'Llama2-7B': 8,  'Llama2-13B': 20,  'Llama2-70B': 32}
T_SHORT    = 512   # seqlen threshold: short < T_SHORT, long >= T_SHORT
PREFILL    = 512
DECODING   = 3584

SLO_1S = 1.0
SLO_5S = 5.0


def pool_L_metrics(df, N_L):
    """From Pool L CSV: pick lat-optimal config (PP=1, TP=N_L), compute TTFT and Tput."""
    lat = df[(df['Pipeline parallelism'] == 1) & (df['Tensor parallelism'] == N_L)]
    if lat.empty:
        return None

    # TTFT: mean token latency over short seqlens × T_SHORT
    short_rows = lat[lat['Sequence length'] <= T_SHORT]
    if short_rows.empty:
        return None
    ttft_short = short_rows['Token latency (ms)'].mean() * T_SHORT / 1000

    # Tput: mean over all seqlens (overall throughput capacity of Pool L)
    tput_L = lat['Throughput (tokens/s)'].mean()

    # PIM/CXL breakdown at T_SHORT for analysis
    short_pim = short_rows['PIM latency'].mean()
    short_cxl = short_rows['CXL latency'].mean()

    return {
        'TTFT_short': ttft_short,
        'Tput_L': tput_L,
        'PIM_L': short_pim,
        'CXL_L': short_cxl,
        'SLO_1s': ttft_short < SLO_1S,
        'SLO_5s': ttft_short < SLO_5S,
    }


def pool_T_metrics(df, num_layers):
    """From Pool T CSV: pick tput-optimal config (PP=num_layers, TP=1), compute Tput."""
    tpt = df[(df['Pipeline parallelism'] == num_layers) & (df['Tensor parallelism'] == 1)]
    if tpt.empty:
        return None

    tput_T = tpt['Throughput (tokens/s)'].mean()

    # End-to-end latency for long requests
    long_rows = tpt[tpt['Sequence length'] > T_SHORT]
    if long_rows.empty:
        long_rows = tpt
    ttft_long = long_rows['Token latency (ms)'].mean() * (PREFILL + DECODING) / 1000

    return {
        'Tput_T': tput_T,
        'TTFT_long': ttft_long,
    }


rows = []

for model in ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']:
    lay = LAYERS[model]
    N_total = TOTAL_DEV[model]

    pool_L_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"sim_{model}_poolL_*.csv")))
    pool_T_files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"sim_{model}_poolT_*.csv")))

    pool_T_map = {}
    for f in pool_T_files:
        NT = int(f.split('_poolT_')[1].replace('.csv', ''))
        df = pd.read_csv(f)
        m  = pool_T_metrics(df, lay)
        if m:
            pool_T_map[NT] = m

    for f in pool_L_files:
        NL = int(f.split('_poolL_')[1].replace('.csv', ''))
        NT = N_total - NL
        if NT not in pool_T_map:
            print(f"  WARNING: no Pool T data for {model} N_T={NT}, skipping N_L={NL}")
            continue

        df_L = pd.read_csv(f)
        m_L  = pool_L_metrics(df_L, NL)
        if m_L is None:
            print(f"  WARNING: Pool L config PP=1,TP={NL} missing for {model}, skipping")
            continue

        m_T = pool_T_map[NT]

        system_tput = m_L['Tput_L'] + m_T['Tput_T']

        rows.append({
            'Model':        model,
            'N_total':      N_total,
            'N_L':          NL,
            'N_T':          NT,
            'TTFT_short':   m_L['TTFT_short'],
            'Tput_L':       m_L['Tput_L'],
            'Tput_T':       m_T['Tput_T'],
            'system_tput':  system_tput,
            'TTFT_long':    m_T['TTFT_long'],
            'PIM_L':        m_L['PIM_L'],
            'CXL_L':        m_L['CXL_L'],
            'SLO_1s_short': m_L['SLO_1s'],
            'SLO_5s_short': m_L['SLO_5s'],
        })

combined = pd.DataFrame(rows).sort_values(['Model', 'N_L']).reset_index(drop=True)
combined.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
print()
print(combined[['Model','N_L','N_T','TTFT_short','Tput_L','Tput_T',
                'system_tput','SLO_1s_short']].to_string(index=False))
