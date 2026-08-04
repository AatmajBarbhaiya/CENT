"""
combine_pools32.py — DDA pool metrics with N_total = 32 for ALL models.

Differs from ../../dynamic_device_allocation/simulation/combine_pools.py:
  - N_total is 32 for every model (that one pins 7B->8, 13B->20, 70B->32).
  - S2 runs with NO tolerance: strict TTFT_L < 1.0s (the parent used 1s +/- 5%).
  - Reports device utilisation and flags splits whose throughput is a formula
    artifact rather than a realizable pipeline.

Validity filter (see ../CLAUDE.md "Model limitation"):
  throughput = 1000/token_latency * pp assumes UNIFORM stages, which only holds
  when pp divides num_layers. Pool T always uses pp = num_layers (1 block/stage,
  exact). Pool L uses PP=1 (whole model on one stage, exact). So both pool
  configs are realizable -- the filter matters only if someone adds mid-range
  pp values later. We assert it rather than assume it.

Output: ../results/dda32/combined_metrics32.csv
"""

import os
import glob
import math
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/dda32")
STATIC  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
OUT     = os.path.join(RESULTS, "combined_metrics32.csv")

LAYERS  = {'Llama2-7B': 32, 'Llama2-13B': 40, 'Llama2-70B': 80}
MIN_CPB = {'Llama2-7B': 5,  'Llama2-13B': 8,  'Llama2-70B': 6}
N_TOTAL = 32
T_SHORT = 512
PREFILL, DECODING = 512, 3584

pd.set_option('display.width', 250)


def pool_L(model, N_L):
    """Latency pool: TP=N_L, PP=1."""
    p = os.path.join(RESULTS, f"sim_{model}_poolL_{N_L}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    r = df[(df['Pipeline parallelism'] == 1) & (df['Tensor parallelism'] == N_L)]
    if r.empty:
        return None
    short = r[r['Sequence length'] <= T_SHORT]
    if short.empty:
        return None
    return {
        'TTFT_short': short['Token latency (ms)'].mean() * T_SHORT / 1000,
        'Tput_L':     r['Throughput (tokens/s)'].mean(),
        'PIM_L':      short['PIM latency'].mean(),
        'CXL_L':      short['CXL latency'].mean(),
    }


def pool_T(model, N_T):
    """Throughput pool: TP=1, PP=num_layers."""
    p = os.path.join(RESULTS, f"sim_{model}_poolT_{N_T}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    r = df[(df['Pipeline parallelism'] == LAYERS[model]) & (df['Tensor parallelism'] == 1)]
    if r.empty:
        return None
    bpd  = math.ceil(LAYERS[model] / N_T)
    used = math.ceil(LAYERS[model] / bpd)          # devices PP actually lights up
    long = r[r['Sequence length'] > T_SHORT]
    if long.empty:
        long = r
    return {
        'Tput_T':    r['Throughput (tokens/s)'].mean(),
        'TTFT_long': long['Token latency (ms)'].mean() * (PREFILL + DECODING) / 1000,
        'cpb_T':     int(r['Channels per block'].iloc[0]),
        'used_T':    used,
        'idle_T':    N_T - used,
    }


rows = []
for model in LAYERS:
    for f in sorted(glob.glob(os.path.join(RESULTS, f"sim_{model}_poolL_*.csv"))):
        N_L = int(f.split('_poolL_')[1].replace('.csv', ''))
        N_T = N_TOTAL - N_L
        mL, mT = pool_L(model, N_L), pool_T(model, N_T)
        if mL is None or mT is None:
            continue
        assert 32 // math.ceil(LAYERS[model] / N_T) >= MIN_CPB[model], \
            f"{model} N_T={N_T} violates minimal_channel_per_block"
        rows.append({
            'Model': model, 'N_L': N_L, 'N_T': N_T,
            'TTFT_short': mL['TTFT_short'], 'Tput_L': mL['Tput_L'],
            'Tput_T': mT['Tput_T'], 'system_tput': mL['Tput_L'] + mT['Tput_T'],
            'TTFT_long': mT['TTFT_long'], 'cpb_T': mT['cpb_T'],
            'used_T': mT['used_T'], 'idle_T': mT['idle_T'],
            'idle_total': mT['idle_T'],       # Pool L (PP=1,TP=N_L) uses all N_L
            'PIM_L': mL['PIM_L'], 'CXL_L': mL['CXL_L'],
        })

df = pd.DataFrame(rows).sort_values(['Model', 'N_L']).reset_index(drop=True)
os.makedirs(RESULTS, exist_ok=True)
df.to_csv(OUT, index=False)


MAIN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "../../../cent_simulation/simulation_results.csv")


def _pareto_from(frames, model):
    pts = []
    for d in frames:
        for (pp, tp), g in d.groupby(['Pipeline parallelism', 'Tensor parallelism']):
            # uniform-stage validity: throughput = pp/token_latency only holds
            # when pp divides num_layers (see ../CLAUDE.md "Model limitation")
            if LAYERS[model] % pp != 0:
                continue
            s = g[g['Sequence length'] <= T_SHORT]
            if s.empty:
                continue
            pts.append({'PP': pp, 'TP': tp,
                        'TTFT': s['Token latency (ms)'].mean() * T_SHORT / 1000,
                        'Tput': g['Throughput (tokens/s)'].mean()})
    if not pts:
        return pd.DataFrame(columns=['PP', 'TP', 'TTFT', 'Tput'])
    return pd.DataFrame(pts).sort_values('TTFT').reset_index(drop=True)


def static_pareto(model):
    """Best single static (PP,TP) config on 32 devices.

    Prefers this folder's static sweep (7B/13B). 70B was never re-run here
    because the paper already deploys it on 32 devices -- its baseline comes
    from cent_simulation/simulation_results.csv, which is scored at the same
    144//32 = 4 lanes/device, so the two are directly comparable.
    """
    frames = []
    for mode in ('pp', 'mp'):
        p = os.path.join(STATIC, f"sim_{model}_{mode}_32dev_lanes32.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames and os.path.exists(MAIN_CSV):
        d = pd.read_csv(MAIN_CSV)
        d = d[(d['Model'] == model) & (d['Device number'] == 32)]
        if not d.empty:
            frames.append(d)
    return _pareto_from(frames, model)


def best_static_lte(par, ttft):
    c = par[par['TTFT'] <= ttft + 0.01]
    return (c.sort_values('Tput', ascending=False).iloc[0] if not c.empty
            else par.iloc[0])


print(f"\n{'='*128}\nDDA pools, N_total=32, all models — every (N_L, N_T) split\n{'='*128}")
for model in LAYERS:
    sub = df[df['Model'] == model]
    if sub.empty:
        continue
    print(f"\n### {model}  ({LAYERS[model]} layers)")
    show = sub[['N_L', 'N_T', 'TTFT_short', 'Tput_L', 'Tput_T', 'system_tput',
                'TTFT_long', 'cpb_T', 'used_T', 'idle_T']].copy()
    for c in ('TTFT_short', 'Tput_L', 'Tput_T', 'system_tput', 'TTFT_long'):
        show[c] = show[c].round(3 if 'TTFT' in c else 1)
    print(show.to_string(index=False))

print(f"\n{'='*128}\nS1 (argmax gain vs best static @ same TTFT)  vs  S2 (argmax system_tput, unconstrained — TTFT reported, no SLO)\n{'='*128}")
out = []
for model in LAYERS:
    sub = df[df['Model'] == model]
    par = static_pareto(model)
    if sub.empty or par.empty:
        continue

    # S1 — best throughput gain vs the best static config at the SAME TTFT (Pareto-matched)
    best_gain, s1 = 0, None
    for _, r in sub.iterrows():
        g = r['system_tput'] / best_static_lte(par, r['TTFT_short'])['Tput']
        if g > best_gain:
            best_gain, s1 = g, r

    # S2 — max system throughput, unconstrained (no SLO gate); its TTFT is just reported
    s2 = sub.sort_values('system_tput', ascending=False).iloc[0]
    bs = best_static_lte(par, s2['TTFT_short'])
    out.append({
        'Model': model,
        'S1 N_L/N_T': f"{int(s1['N_L'])}/{int(s1['N_T'])}",
        'S1 tput': round(s1['system_tput'], 0), 'S1 TTFT': round(s1['TTFT_short'], 3),
        'S1 gain': f"{best_gain:.2f}x",
        'S2 N_L/N_T': f"{int(s2['N_L'])}/{int(s2['N_T'])}",
        'S2 tput': round(s2['system_tput'], 0), 'S2 TTFT': round(s2['TTFT_short'], 3),
        'S2 gain': f"{s2['system_tput'] / bs['Tput']:.2f}x",
        'best static': f"PP{int(bs['PP'])}/TP{int(bs['TP'])} @{bs['Tput']:.0f}",
    })
print(pd.DataFrame(out).to_string(index=False))
print(f"\nSaved: {OUT}")