#!/usr/bin/env python3
"""realistic_dda_sweep.py — full DDA performance spectrum, all 3 models, 32 devices.

Splits 32 devices into Pool L (latency) + Pool T (throughput) and reports the WHOLE
spectrum of splits with the realistic models we derived, not just the winners.

Pool L  (latency): "TP = max, accounting saturation." A single TP=N_L instance
  saturates past the knee (finding 3), so past TP_sat we run DATA-PARALLEL replicas
  of the TP_sat config instead of one fat instance:
      k = N_L // TP_sat replicas ;  Tput_L = k * Tput(TP_sat) ;  TTFT_L = TTFT(TP_sat)
  Remainder devices (N_L mod TP_sat) are handed to Pool T (no idle).
  TP_sat = in_dim/512 (finding 3):  7B=8, 13B=10, 70B=16.

Pool T  (throughput): heterogeneous packing (finding 7). CENT prices every device at
  cpb_ceil; the floor devices actually run at cpb_floor (faster), so batch latency
  drops while throughput (ceil-bottlenecked) is unchanged:
      tok_packed = tok_cent - (n_floor*b_floor)*(tbl(cpb_ceil) - tbl(cpb_floor))
  computed as a calibrated delta on CENT (uniform N_T reproduces CENT exactly).

Interactive load alpha (fraction latency-critical), FIFO two-class bound:
      lambda_max(k, alpha) = min( Tput_L / alpha , Tput_T / (1-alpha) )

Outputs (results/dda32/):
  poolL_spectrum32.csv       Pool L per N_L: single-instance vs DP-replica (saturation)
  poolT_spectrum32.csv       Pool T per N_T: CENT vs packed, staircase, idle devices
  dda_split_spectrum32.csv   deployments k replicas -> (N_L,N_T): system metrics
  best_split_by_alpha32.csv  optimal deployment per alpha
"""
import os, glob, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DDA = os.path.join(RES, "dda32")

LAYERS  = {"Llama2-7B": 32, "Llama2-13B": 40, "Llama2-70B": 80}
TP_SAT  = {"Llama2-7B": 8,  "Llama2-13B": 10, "Llama2-70B": 16}
MIN_CPB = {"Llama2-7B": 5,  "Llama2-13B": 8,  "Llama2-70B": 6}
# fp16 weight footprint (finding: Footprints table). Pool L (TP=N_L) shards weights
# across N_L devices -> need N_L >= ceil(weights / 16 GiB). This wall makes 70B's
# sub-10 poolL sims physically invalid (CENT never checks capacity).
WEIGHTS_GIB = {"Llama2-7B": 12.1, "Llama2-13B": 23.6, "Llama2-70B": 145.0}
DEV_GIB = 16.0

def _cap_floor(model):
    return math.ceil(WEIGHTS_GIB[model] / DEV_GIB)
INOUT = 0.15
T_SHORT = 512
PREFILL, DECODING = 512, 3584
FABRIC = 32
ALPHAS = np.round(np.arange(0.02, 0.99, 0.01), 2)


def _model(path):
    for m in LAYERS:
        if m in path:
            return m
    return None


# ---- per-block latency ladder: (model,cpb) -> {seqlen: tbl}, for finding-7 packing ----
tbl, tbl_src = {}, {}

def _register(model, cpb, df, source):
    key = (model, cpb)
    if key in tbl and tbl_src[key] == "pp":
        return
    L = LAYERS[model]
    tbl[key] = {int(r["Sequence length"]):
                (r["Token latency (ms)"] - r["Embedding latency"] - INOUT) / L
                for _, r in df.iterrows()}
    tbl_src[key] = source

# static PP files (incl. cpb probes), then Pool T files — all PP (TP=1) sources.
for f in sorted(glob.glob(os.path.join(RES, "sim_*_pp_*dev_lanes32.csv"))):
    df = pd.read_csv(f)
    _register(_model(f), int(df["Channels per block"].iloc[0]), df, "pp")
for f in sorted(glob.glob(os.path.join(DDA, "sim_*_poolT_*.csv"))):
    df = pd.read_csv(f)
    r = df[df["Tensor parallelism"] == 1]
    if len(r):
        _register(_model(f), int(r["Channels per block"].iloc[0]), r, "poolT")


def _have_cpb(model, cpb):
    return (model, cpb) in tbl


# ---- Pool L ----
def pool_L_single(model, N_L):
    """Measured single TP=N_L instance (PP=1)."""
    p = os.path.join(DDA, f"sim_{model}_poolL_{N_L}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    r = df[(df["Pipeline parallelism"] == 1) & (df["Tensor parallelism"] == N_L)]
    if r.empty:
        return None
    short = r[r["Sequence length"] <= T_SHORT]
    return {"Tput": r["Throughput (tokens/s)"].mean(),
            "TTFT": short["Token latency (ms)"].mean() * T_SHORT / 1000}


def pool_L_dp(model, N_L):
    """DP replicas of TP_sat. Remainder devices fold into Pool T."""
    ts = TP_SAT[model]
    base = pool_L_single(model, ts)
    if base is None:
        return None
    k = N_L // ts
    used = k * ts
    return {"k": k, "N_L_used": used, "rem": N_L - used,
            "Tput": k * base["Tput"], "TTFT": base["TTFT"],
            "Tput_per_replica": base["Tput"]}


# ---- Pool T ----
def pool_T(model, N_T):
    """CENT + finding-7 packed latency for Pool T at N_T (TP=1, PP=layers)."""
    p = os.path.join(DDA, f"sim_{model}_poolT_{N_T}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    r = df[(df["Pipeline parallelism"] == LAYERS[model]) & (df["Tensor parallelism"] == 1)]
    if r.empty:
        return None
    L = LAYERS[model]
    b_ceil, b_floor = math.ceil(L / N_T), L // N_T
    rem = L % N_T
    n_ceil, n_floor = (rem, N_T - rem) if rem else (N_T, 0)
    cpb_ceil, cpb_floor = 32 // b_ceil, 32 // b_floor
    floor_blocks = n_floor * b_floor
    bpd = math.ceil(L / N_T)
    used = math.ceil(L / bpd)

    packed_ok = _have_cpb(model, cpb_ceil) and _have_cpb(model, cpb_floor)
    lat_cent, lat_packed = [], []
    for _, row in r.iterrows():
        sl = int(row["Sequence length"])
        tok = row["Token latency (ms)"]
        lat_cent.append(tok)
        if rem and packed_ok:
            save = floor_blocks * (tbl[(model, cpb_ceil)][sl] - tbl[(model, cpb_floor)][sl])
            lat_packed.append(tok - save)
        else:
            lat_packed.append(tok)   # uniform (rem=0) or no ladder -> no change
    r2 = pd.DataFrame({"sl": r["Sequence length"].values,
                       "cent": lat_cent, "packed": lat_packed})
    lng = r2[r2.sl > T_SHORT]
    if lng.empty:
        lng = r2
    full = (PREFILL + DECODING) / 1000
    return {"Tput_T": r["Throughput (tokens/s)"].mean(),
            "cpb_ceil": cpb_ceil, "cpb_floor": cpb_floor,
            "n_ceil": n_ceil, "n_floor": n_floor, "used": used, "idle": N_T - used,
            "hetero": bool(rem), "packed_ok": (packed_ok or not rem),
            "TTFT_long_cent": lng.cent.mean() * full,
            "TTFT_long_packed": lng.packed.mean() * full,
            "lat_improve_pct": 100 * (r2.cent.mean() - r2.packed.mean()) / r2.cent.mean()}


# ---- build the four views ----
poolL_rows, poolT_rows, split_rows = [], [], []

for model in LAYERS:
    ts, L = TP_SAT[model], LAYERS[model]

    # View A: Pool L spectrum (single vs DP) over measured N_L
    for f in sorted(glob.glob(os.path.join(DDA, f"sim_{model}_poolL_*.csv")),
                    key=lambda x: int(x.split("_poolL_")[1].replace(".csv", ""))):
        N_L = int(f.split("_poolL_")[1].replace(".csv", ""))
        s, d = pool_L_single(model, N_L), pool_L_dp(model, N_L)
        if s is None:
            continue
        poolL_rows.append(dict(
            Model=model, N_L=N_L, TP_sat=ts,
            Tput_single=round(s["Tput"], 1), TTFT_single=round(s["TTFT"], 3),
            k_replicas=d["k"] if d else None, N_L_used=d["N_L_used"] if d else None,
            Tput_dp=round(d["Tput"], 1) if d else None,
            TTFT_dp=round(d["TTFT"], 3) if d else None,
            dp_gain=round(d["Tput"] / s["Tput"], 2) if d and s["Tput"] else None))

    # View B: Pool T spectrum (CENT vs packed) over measured N_T
    for f in sorted(glob.glob(os.path.join(DDA, f"sim_{model}_poolT_*.csv")),
                    key=lambda x: int(x.split("_poolT_")[1].replace(".csv", ""))):
        N_T = int(f.split("_poolT_")[1].replace(".csv", ""))
        t = pool_T(model, N_T)
        if t is None:
            continue
        poolT_rows.append(dict(
            Model=model, N_T=N_T, cpb_ceil=t["cpb_ceil"], cpb_floor=t["cpb_floor"],
            n_ceil=t["n_ceil"], n_floor=t["n_floor"], used=t["used"], idle=t["idle"],
            Tput_T=round(t["Tput_T"], 1),
            TTFT_long_cent=round(t["TTFT_long_cent"], 3),
            TTFT_long_packed=round(t["TTFT_long_packed"], 3),
            lat_improve_pct=round(t["lat_improve_pct"], 1),
            hetero=t["hetero"], packed_ok=t["packed_ok"]))

    # View C: deployment spectrum. Pool L is EITHER
    #   - one sub-saturation instance TP=m (floor <= m <= TP_sat): variable 1st instance
    #     for low alpha, frees devices to Pool T (may cross a Tput_T staircase step), or
    #   - k homogeneous TP_sat replicas (k>=2): throughput scaling for higher alpha.
    # Variable 2nd instance dropped (finding 8b): heterogeneous replicas break Tput_L=k*..
    base = pool_L_single(model, ts)
    if base is None:
        continue
    cap = _cap_floor(model)   # capacity wall (physics) is the ONLY floor; no SLO policy gate

    # Pool L options: (N_L, Tput_L, TTFT_L, mode). Capacity-gated only. TTFT_L is REPORTED,
    # never gated — the operator applies whatever latency bar they want (no 1s/5s here).
    opts = [(0, 0.0, float("nan"), "none")]     # N_L=0 = all Pool T (alpha->0 reference)
    for f in sorted(glob.glob(os.path.join(DDA, f"sim_{model}_poolL_*.csv")),
                    key=lambda x: int(x.split("_poolL_")[1].replace(".csv", ""))):
        m = int(f.split("_poolL_")[1].replace(".csv", ""))
        if m > ts:
            continue                            # past TP_sat -> use DP replicas instead
        s = pool_L_single(model, m)
        if s is None or m < cap:
            continue                            # only the capacity wall is a hard floor
        opts.append((m, s["Tput"], s["TTFT"], "single"))
    for k in range(2, FABRIC // ts + 1):        # homogeneous DP replicas
        opts.append((k * ts, k * base["Tput"], base["TTFT"], f"dp_k{k}"))

    for N_L, Tput_L, TTFT_L, mode in opts:
        N_T = FABRIC - N_L
        t = pool_T(model, N_T) if N_T > 0 else None
        Tput_T = t["Tput_T"] if t else 0.0
        split_rows.append(dict(
            Model=model, mode=mode, N_L=N_L, N_T=N_T,
            TP_per_replica=(ts if mode.startswith("dp") else N_L),
            Tput_L=round(Tput_L, 1), TTFT_L_s=round(TTFT_L, 3) if N_L else None,
            Tput_T=round(Tput_T, 1),
            TTFT_T_cent=round(t["TTFT_long_cent"], 3) if t else None,
            TTFT_T_packed=round(t["TTFT_long_packed"], 3) if t else None,
            idle_T=t["idle"] if t else 0,
            system_tput=round(Tput_L + Tput_T, 1),
            cap_floor=cap))

split = pd.DataFrame(split_rows)

# View D: throughput-optimal deployment per alpha (max lambda_max). NO SLO gate — the
# chosen split's TTFT_L is REPORTED so the operator can accept/reject against their own bar.
# Tiebreak: among equal lambda_max, pick the SMALLEST N_L (-> largest N_T -> best batch
# latency via packing, and staircase headroom). Sorting candidates by N_L ascending makes
# np.argmax return the min-N_L winner. This is where the variable 1st instance pays off.
best_rows = []
for model in LAYERS:
    sub = split[(split.Model == model) & (split.N_L > 0)].copy()
    sub = sub.sort_values("N_L").reset_index(drop=True)
    if sub.empty:
        continue
    for a in ALPHAS:
        # An empty pool has 0 capacity for its class -> its bound is 0, not inf.
        bound_L = sub.Tput_L.values / a
        bound_T = sub.Tput_T.values / (1 - a)
        lam = np.minimum(bound_L, bound_T)
        i = int(np.argmax(lam))
        r = sub.iloc[i]
        best_rows.append(dict(Model=model, alpha=a, mode=r["mode"],
                              N_L=int(r.N_L), N_T=int(r.N_T),
                              lambda_max=round(lam[i], 1),
                              Tput_L=r.Tput_L, Tput_T=r.Tput_T,
                              TTFT_L_s=r.TTFT_L_s, TTFT_T_packed_s=r.TTFT_T_packed,
                              bottleneck="PoolL" if bound_L[i] <= bound_T[i] else "PoolT"))

# ---- write ----
os.makedirs(DDA, exist_ok=True)
pd.DataFrame(poolL_rows).to_csv(os.path.join(DDA, "poolL_spectrum32.csv"), index=False)
pd.DataFrame(poolT_rows).to_csv(os.path.join(DDA, "poolT_spectrum32.csv"), index=False)
split.to_csv(os.path.join(DDA, "dda_split_spectrum32.csv"), index=False)
pd.DataFrame(best_rows).to_csv(os.path.join(DDA, "best_split_by_alpha32.csv"), index=False)

# ---- report ----
print(f"{'='*90}\nPool L saturation (single instance vs DP replicas of TP_sat)\n{'='*90}")
print(pd.DataFrame(poolL_rows).to_string(index=False))
print(f"\n{'='*90}\nPool T staircase + heterogeneous packing (finding 7)\n{'='*90}")
print(pd.DataFrame(poolT_rows).to_string(index=False))
print(f"\n{'='*90}\nDeployment spectrum (k Pool L replicas)\n{'='*90}")
print(split.to_string(index=False))
print(f"\nSaved 4 CSVs to {DDA}")