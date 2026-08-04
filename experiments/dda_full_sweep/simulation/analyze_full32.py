"""
analyze_full32.py — does giving 7B / 13B the full 32-device fabric pay off?

Reads ../results/sim_{model}_{pp|mp}_{N}dev_lanes{32|N}.csv and answers:

  Q1  Scale-up gain: one instance on 32 devices vs the paper's 8 / 20.
  Q2  Data-parallel alternative: k instances of N devices on the same fabric.
      Is aggregate throughput really k x the single-instance number?
  Q3  Where the two meet: throughput per device, and the latency each buys.

Throughput note: CENT's `Throughput (tokens/s)` = 1000 / token_latency * pp is
PER INSTANCE and carries no DP term. A k-way data-parallel deployment of
independent instances is therefore k x that column -- the multiply is done here,
explicitly, not by the simulator.

Utilisation note: PP quantises to blocks_per_device = ceil(layers / N). A config
that leaves devices idle is reported at its *fabric* cost (N), not its utilised
device count, since the idle devices are still bought and powered.

Output: ../results/full32_summary.csv + a printed report.
"""

import os
import glob
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
LAYERS  = {'Llama2-7B': 32, 'Llama2-13B': 40}
PAPER_N = {'Llama2-7B': 8,  'Llama2-13B': 20}
FABRIC  = 32
PREFILL = 512

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)


def load(model, mode, N, lanes):
    p = os.path.join(RESULTS, f"sim_{model}_{mode}_{N}dev_lanes{lanes}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def pp_row(model, N, lanes):
    """Pure pipeline config: PP=layers, TP=1 — the throughput-optimal point."""
    df = load(model, 'pp', N, lanes)
    if df is None:
        return None
    r = df[(df['Pipeline parallelism'] == LAYERS[model]) & (df['Tensor parallelism'] == 1)]
    return r if not r.empty else None


def summarise(r, model, N, label):
    """Collapse a per-seqlen frame into one row of headline metrics."""
    short = r[r['Sequence length'] <= PREFILL]
    return {
        'Model':        model,
        'Config':       label,
        'N_dev':        N,
        'cpb':          int(r['Channels per block'].iloc[0]),
        'PP':           int(r['Pipeline parallelism'].iloc[0]),
        'TP':           int(r['Tensor parallelism'].iloc[0]),
        'dev_util':     float(r['Device utilization'].iloc[0]),
        'tok_lat_ms':   r['Token latency (ms)'].mean(),
        'TTFT_s':       short['Token latency (ms)'].mean() * PREFILL / 1000,
        'tput_inst':    r['Throughput (tokens/s)'].mean(),
        'energy_mJ':    r['Token energy (mJ)'].mean(),
        'power_W':      r['Total power (W)'].mean(),
    }


def main():
    rows = []
    for model in LAYERS:
        for N in sorted({8, 16, 20, 32}):
            for lanes in ('32', 'N'):
                r = pp_row(model, N, lanes)
                if r is None:
                    continue
                s = summarise(r, model, N, f"PP x{LAYERS[model]}, lanes={lanes}")
                s['lanes'] = lanes
                rows.append(s)
    if not rows:
        print("No results yet.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "full32_summary.csv"), index=False)

    for model in LAYERS:
        base_N = PAPER_N[model]
        sub = df[(df['Model'] == model) & (df['lanes'] == '32')]
        if sub.empty:
            continue

        print(f"\n{'='*100}\n{model}  —  pipeline configs on a shared {FABRIC}-device fabric (4 PCIe lanes/dev)\n{'='*100}")

        base = sub[sub['N_dev'] == base_N]
        if base.empty:
            print(f"  (no {base_N}-device baseline)")
            continue
        b = base.iloc[0]

        out = []
        for _, r in sub.iterrows():
            k = FABRIC // r['N_dev']                     # instances that fit on the fabric
            out.append({
                'N_dev':       int(r['N_dev']),
                'cpb':         int(r['cpb']),
                'dev_util':    f"{r['dev_util']:.0%}",
                'tok_lat_ms':  round(r['tok_lat_ms'], 3),
                'TTFT_s':      round(r['TTFT_s'], 3),
                'lat_speedup': round(b['tok_lat_ms'] / r['tok_lat_ms'], 2),
                'tput_inst':   round(r['tput_inst'], 1),
                'DP_k':        k,
                'tput_fabric': round(r['tput_inst'] * k, 1),
                'tput/dev':    round(r['tput_inst'] * k / FABRIC, 1),
            })
        print(pd.DataFrame(out).to_string(index=False))
        print(f"\n  baseline = {base_N} devices (paper). lat_speedup vs that baseline.")
        print(f"  tput_fabric = per-instance throughput x DP_k instances filling the {FABRIC}-device fabric.")

    print(f"\n{'='*100}\nPCIe sensitivity: same silicon, standalone box (144/N lanes) vs shared fabric (4 lanes)\n{'='*100}")
    piv = df.pivot_table(index=['Model', 'N_dev'], columns='lanes',
                         values=['tok_lat_ms', 'tput_inst'])
    print(piv.to_string())


if __name__ == "__main__":
    main()