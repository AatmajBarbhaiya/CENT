#!/usr/bin/env python3
"""Realistic heterogeneous-packing model for the static 32-device sweep.

CENT assumes a UNIFORM pipeline: every device holds blocks_per_device = ceil(L/N)
blocks at one cpb, utilized_devices = ceil(L/bpd) (the rest idle). When L does not
divide N this is wrong two ways:
  - PP: floor devices could each hold FEWER blocks at HIGHER cpb (faster block).
    CENT prices every block at the slow cpb_ceil -> over-states LATENCY. Throughput
    is still bottlenecked by the ceil device, so it is unchanged.
  - MP: a stage's blocks run serially (all 32 ch per block). CENT prices pp stages
    of 1 block, but the bottleneck stage holds ceil(L/pp) -> over-states THROUGHPUT
    by ceil(L/pp)/(L/pp). Latency is unchanged (all blocks cpb=32).

Two device classes (finding 4 / finding 6):
    b_ceil = ceil(L/N), b_floor = floor(L/N), r = L mod N
    r devices hold b_ceil (cpb_ceil = 32//b_ceil), N-r hold b_floor (cpb_floor).

CALIBRATION: the realistic numbers are a *delta on CENT's own output*, never a
from-scratch reconstruction, so a uniform config reproduces CENT exactly and the
only reported change is the packing effect. Per-block latency at a given cpb is
CENT's own effective per-block cost, recovered from the raw sweep:
    tbl(cpb) = (token_latency(cpb) - embedding(cpb) - InOut) / L        [per seqlen]

PP:  tok_real  = tok_cent - (n_floor * b_floor) * (tbl(cpb_ceil) - tbl(cpb_floor))
     tput_real = tput_cent                              (bottleneck = ceil device)
MP:  tok_real  = tok_cent
     tput_real = tput_cent * (L/pp) / ceil(L/pp)        (uneven-stage fix)

Outputs results/full32_realistic.csv. Raw CSVs untouched.
"""
import csv, glob, math, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LAYERS = {"Llama2-7B": 32, "Llama2-13B": 40, "Llama2-70B": 80}
INOUT = 0.15  # ms, hardcoded host top-K (utils.py:7)
BASELINE_N = {"Llama2-7B": 8, "Llama2-13B": 20}  # paper's per-model device budget


def model_of(path):
    return "Llama2-7B" if "7B" in path else ("Llama2-13B" if "13B" in path else "Llama2-70B")


# ---- per-block latency lookup: (model,cpb) -> {seqlen: tbl}, source tag ----
# tbl = CENT's own effective per-block cost at that cpb (acc quirk and all), so a
# uniform config rebuilt from it reproduces CENT exactly.
tbl, src = {}, {}

def register(model, cpb, df, source):
    key = (model, cpb)
    if key in tbl and src[key] == "pp":
        return  # PP authoritative; MP fallback never overwrites it
    L = LAYERS[model]
    tbl[key] = {int(r["Sequence length"]):
                (r["Token latency (ms)"] - r["Embedding latency"] - INOUT) / L
                for _, r in df.iterrows()}
    src[key] = source

for f in sorted(glob.glob(os.path.join(RES, "sim_*_pp_*dev_lanes32.csv"))):
    df = pd.read_csv(f)
    register(model_of(f), int(df["Channels per block"].iloc[0]), df, "pp")
for f in sorted(glob.glob(os.path.join(RES, "sim_*_mp_*dev_lanes32.csv"))):
    df = pd.read_csv(f)
    tp1 = df[df["Tensor parallelism"] == 1]
    if len(tp1):
        register(model_of(f), 32, tp1, "mp_tp1")   # cpb=32 fallback for 13B floor


# ---- config models ----
rows = []

def pp_config(model, N, df):
    L = LAYERS[model]
    b_ceil, b_floor = math.ceil(L / N), L // N
    r = L % N
    n_ceil, n_floor = (r, N - r) if r else (N, 0)
    cpb_ceil, cpb_floor = 32 // b_ceil, 32 // b_floor
    floor_blocks = n_floor * b_floor
    lat, tput = [], []
    for _, row in df.iterrows():
        sl = int(row["Sequence length"])
        save = floor_blocks * (tbl[(model, cpb_ceil)][sl] - tbl[(model, cpb_floor)][sl])
        lat.append(row["Token latency (ms)"] - save)
        tput.append(row["Throughput (tokens/s)"])          # unchanged: ceil bottleneck
    return dict(mode="PP", TP=1, pp=L, b_ceil=b_ceil, b_floor=b_floor,
                n_ceil=n_ceil, n_floor=n_floor, cpb_ceil=cpb_ceil, cpb_floor=cpb_floor,
                tok_cent=df["Token latency (ms)"].mean(), tput_cent=df["Throughput (tokens/s)"].mean(),
                tok_real=sum(lat)/len(lat), tput_real=sum(tput)/len(tput),
                sbl_src=(src[(model, cpb_floor)] if n_floor else "pp"),
                note="uniform" if r == 0 else "heterogeneous")

def mp_config(model, N, TP, g):
    L = LAYERS[model]
    pp = N // TP
    b_ceil, b_floor = math.ceil(L / pp), L // pp
    r = L % pp
    n_ceil, n_floor = (r, pp - r) if r else (pp, 0)
    factor = (L / pp) / b_ceil                              # <=1, over-estimate fix
    cent_lat, cent_tput = g["Token latency (ms)"].mean(), g["Throughput (tokens/s)"].mean()
    return dict(mode="MP", TP=TP, pp=pp, b_ceil=b_ceil, b_floor=b_floor,
                n_ceil=n_ceil, n_floor=n_floor, cpb_ceil=32, cpb_floor=32,
                tok_cent=cent_lat, tput_cent=cent_tput,
                tok_real=cent_lat, tput_real=cent_tput * factor,
                sbl_src="cent", note="uniform" if r == 0 else "heterogeneous")

FABRIC = 32  # runs beyond the fabric are per-block-latency SOURCES only, never deployments
for f in sorted(glob.glob(os.path.join(RES, "sim_*_pp_*dev_lanes32.csv"))):
    df = pd.read_csv(f); m = model_of(f); N = int(df["Device number"].iloc[0])
    if N > FABRIC:
        continue  # 13B/40-dev exists only to supply the real cpb=32 per-block latency
    d = pp_config(m, N, df); d.update(Model=m, N=N); rows.append(d)
for f in sorted(glob.glob(os.path.join(RES, "sim_*_mp_*dev_lanes32.csv"))):
    df = pd.read_csv(f); m = model_of(f); N = int(df["Device number"].iloc[0])
    for TP, g in df.groupby("Tensor parallelism"):
        d = mp_config(m, N, int(TP), g); d.update(Model=m, N=N); rows.append(d)

# ---- derived ratios ----
base = {d["Model"]: d for d in rows if d["mode"] == "PP" and d["N"] == BASELINE_N.get(d["Model"])}
for d in rows:
    d["lat_improve_pct"] = 100.0 * (d["tok_cent"] - d["tok_real"]) / d["tok_cent"]
    d["tput_ratio_real_cent"] = d["tput_real"] / d["tput_cent"]
    b = base.get(d["Model"])
    d["tput_vs_baseline"] = d["tput_real"] / b["tput_real"] if b else float("nan")
    d["lat_vs_baseline"] = d["tok_real"] / b["tok_real"] if b else float("nan")

cols = ["Model", "mode", "N", "TP", "pp", "b_ceil", "b_floor", "n_ceil", "n_floor",
        "cpb_ceil", "cpb_floor", "tok_cent", "tok_real", "lat_improve_pct",
        "tput_cent", "tput_real", "tput_ratio_real_cent",
        "tput_vs_baseline", "lat_vs_baseline", "sbl_src", "note"]
out = os.path.join(RES, "full32_realistic.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for d in sorted(rows, key=lambda x: (x["Model"], x["mode"], x["N"], x["TP"])):
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})

print(f"wrote {out}  ({len(rows)} configs)\n")
hdr = (f"{'model':<11}{'md':<3}{'N':>3}{'TP':>3}{'ceil/flr':>9}{'cpb c/f':>9}"
       f"{'lat_c':>8}{'lat_r':>8}{'dLat%':>7}{'tput_c':>8}{'tput_r':>8}{'dTput':>7}  note")
print(hdr); print("-" * (len(hdr) + 8))
for d in sorted(rows, key=lambda x: (x["Model"], x["mode"], x["N"], x["TP"])):
    print(f"{d['Model']:<11}{d['mode']:<3}{d['N']:>3}{d['TP']:>3}"
          f"{str(d['b_ceil'])+'/'+str(d['b_floor']):>9}"
          f"{str(d['cpb_ceil'])+'/'+str(d['cpb_floor']):>9}"
          f"{d['tok_cent']:>8.2f}{d['tok_real']:>8.2f}{d['lat_improve_pct']:>7.1f}"
          f"{d['tput_cent']:>8.0f}{d['tput_real']:>8.0f}{d['tput_ratio_real_cent']:>7.2f}"
          f"  {d['note']}" + (f" [{d['sbl_src']}]" if d['note'][0] == 'h' else ""))
