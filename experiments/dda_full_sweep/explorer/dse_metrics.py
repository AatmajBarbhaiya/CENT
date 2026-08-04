#!/usr/bin/env python3
"""dse_metrics.py — SINGLE SOURCE OF TRUTH for every derived DDA metric.

Imported by BOTH the CSV writers (simulation/dse_full_table.py,
simulation/best_split_by_rate_alpha.py) and the Streamlit explorer (explorer/app.py).
Never reimplement these formulas anywhere else — the CSVs are paper-submission evidence and
the explorer is the reasoning tool, so a divergence between them is a correctness bug.

Model (see ../CLAUDE.md "Full Design-Space Exploration"):

  Requests are PRE-TAGGED as latency-class (L) or throughput-class (T) by the client/SLA.
  Class is independent of length, so BOTH pools see the full 128..4096 seqlen range.

  Seqlen mixture: uniform over the simulated grid, identical for both classes. A request
  sweeps UPWARD through occupancies, so the weight on row s is the survival function
  P(S >= s) -- every request visits s=128, only the longest reach s=4096.

  seqlen = prompt + generated. prompt_fraction splits it; it affects TTFT ONLY (throughput
  and KV capacity depend on total S). Default 0.125 is CENT's arbitrary CLI default
  (run_sim.py: --prefill 512 --decoding 3584), NOT a principled value.

  Capacity: a pp-stage pipeline needs pp requests in flight to deliver its reported
  throughput, and EVERY in-flight request holds KV on EVERY block. A split whose KV cannot
  hold that concurrency is not a deployment -> kv_feasible=False, excluded from the optimizer.
"""
import os, glob, math
from functools import lru_cache

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
DDA = os.path.join(RES, "dda32")

# ---------------------------------------------------------------- hardware / model constants
GiB, MiB = 1024**3, 1024**2
DEV_BYTES = 16 * GiB          # 32 ch x 512 MiB (GDDR6.cpp: density 128Gb)
CHANNELS = 32
BANKS_PER_CH = 16
FABRIC = 32                   # devices in the full fabric
INOUT_MS = 0.15               # utils.py:7 host top-K sampling, per token, never scales
VOCAB = 32000
DTYPE_B = 2                   # fp16

# dim, n_heads, gqa_factor, ffn_dim, layers, minimal_channel_per_block  (cent_simulation/utils.py)
MODELS = {
    "Llama2-7B":  dict(d=4096, n_heads=32, gqa=1, ffn=11008, L=32, min_cpb=5),
    "Llama2-13B": dict(d=5120, n_heads=40, gqa=1, ffn=13824, L=40, min_cpb=8),
    "Llama2-70B": dict(d=8192, n_heads=64, gqa=8, ffn=28672, L=80, min_cpb=6),
}

DEFAULT_PROMPT_FRACTION = 0.125


def model_facts(model):
    """Weight/KV footprints derived from architecture. NEVER hand-set these.

    Wk/Wv are GQA-REDUCED in CENT (function_sim.py:29-30 -- `wk = zeros((head_dim*n_kv_heads), dim)`),
    so 70B's per-block weight is 1632 MiB, not the 1856 MiB an all-full-width formula gives.
    That error put 70B's cap_floor at 10 instead of 9 and cost a legal Pool L configuration.
    """
    m = MODELS[model]
    d, L, ffn = m["d"], m["L"], m["ffn"]
    head_dim = d // m["n_heads"]
    n_kv_heads = m["n_heads"] // m["gqa"]

    w_block = (2 * d * d + 2 * d * head_dim * n_kv_heads + 3 * d * ffn) * DTYPE_B
    w_total = w_block * L + 2 * VOCAB * d * DTYPE_B
    kv_tok_layer = 2 * n_kv_heads * head_dim * DTYPE_B      # K and V, one layer, one token

    return dict(
        model=model, d=d, L=L, ffn=ffn, head_dim=head_dim, n_kv_heads=n_kv_heads,
        min_cpb=m["min_cpb"],
        w_block=w_block, w_total=w_total, kv_tok_layer=kv_tok_layer,
        kv_tok_all=kv_tok_layer * L,
        cap_floor=math.ceil(w_total / DEV_BYTES),           # 7B=1, 13B=2, 70B=9
        tp_sat=math.ceil(d / 512),                          # 7B=8, 13B=10, 70B=16
    )


# ---------------------------------------------------------------- seqlen mixture & weighting
def survival_weights(seqlens):
    """P(S >= s) for request length S uniform over the simulated grid, normalized.

    Every request visits the lowest occupancy; only the longest reach the top. Weight on the
    k-th of K grid points is (K-k)/K. A FLAT mean instead of this overstates latency ~9.8%
    and understates throughput ~10.9% on 7B/32-dev -- same order as the effects being measured.
    """
    K = len(seqlens)
    surv = np.array([(K - k) / K for k in range(K)], dtype=float)
    return surv / surv.sum()


def weighted_token_latency(seqlens, latencies):
    """Mixture-mean cost of ONE token-step, weighted by how often each occupancy occurs."""
    w = survival_weights(np.asarray(seqlens))
    return float((w * np.asarray(latencies, dtype=float)).sum())


def pool_throughput(seqlens, latencies, pp):
    """tokens/s. Derived from weighted LATENCY, never by averaging the CSV throughput column
    (that column is 1000/lat*pp per row; averaging reciprocals is not the reciprocal of the mean)."""
    return 1000.0 / weighted_token_latency(seqlens, latencies) * pp


def _cumulative_seconds(seqlens, latencies):
    """cum[k] = seconds to traverse occupancies 1..seqlens[k], stepping the sim grid."""
    s = np.asarray(seqlens, dtype=float)
    gap = s[1] - s[0] if len(s) > 1 else s[0]
    return np.cumsum(np.asarray(latencies, dtype=float) * gap) / 1000.0


def ttft_seconds(seqlens, latencies, prompt_fraction=DEFAULT_PROMPT_FRACTION):
    """Mixture-mean time-to-first-token.

    CUMULATIVE SUM over the prompt, not mean x length -- each prompt token costs more as the
    cache fills. Averaged over the request-length mixture, so requests with short prompts pull
    it down. The old `mean(lat[s<=512]) * 512` assumed EVERY request has a full 512-token prompt.
    """
    s = np.asarray(seqlens, dtype=float)
    cum = _cumulative_seconds(s, latencies)
    gap = s[1] - s[0] if len(s) > 1 else s[0]
    per_request = [cum[max(0, int(math.ceil(prompt_fraction * S / gap)) - 1)] for S in s]
    return float(np.mean(per_request))


def request_latency_seconds(seqlens, latencies):
    """Mixture-mean end-to-end completion latency of a whole request."""
    return float(np.mean(_cumulative_seconds(seqlens, latencies)))


def ttft_seconds_no_prefill_sampling(seqlens, latencies,
                                     prompt_fraction=DEFAULT_PROMPT_FRACTION):
    """TTFT with the host top-K sampling charge removed from PROMPT tokens.

    CENT bills InOut_latency=0.15 ms on every token including all prompt tokens, but sampling
    happens once per request at the end of prefill. Reported as a SEPARATE column -- never
    silently applied to the baseline.
    """
    lat = np.asarray(latencies, dtype=float) - INOUT_MS
    return ttft_seconds(seqlens, lat, prompt_fraction)


def mean_seqlen(seqlens):
    return float(np.mean(seqlens))


# ---------------------------------------------------------------- KV capacity feasibility
def kv_capacity(model, cpb, blocks_on_device, seqlen):
    """Concurrent sequences a device holding `blocks_on_device` blocks at `cpb` can serve.

    Per-block allocation is cpb/32 of a 16 GiB device. Each in-flight request holds KV on
    EVERY block it has traversed (layer i's attention needs layer i's history), so per-block
    KV demand is C * kv_per_token_per_layer * seqlen.
    """
    f = model_facts(model)
    alloc = cpb * (DEV_BYTES / CHANNELS)
    free = alloc - f["w_block"]
    if free <= 0:
        return dict(kv_free=free, c_max=0.0)
    return dict(kv_free=free, c_max=free / (f["kv_tok_layer"] * seqlen))


def pool_T_capacity(model, N_T, seqlen):
    """Worst-case (ceil-class device) KV capacity for a Pool T deployment at N_T devices."""
    f = model_facts(model)
    L = f["L"]
    b_ceil = math.ceil(L / N_T)
    cpb_ceil = CHANNELS // b_ceil
    cap = kv_capacity(model, cpb_ceil, b_ceil, seqlen)
    c_need = L                       # pp = num_layers requests in flight
    return dict(kv_free_per_block=cap["kv_free"], c_max=cap["c_max"], c_need=c_need,
                kv_slack=cap["c_max"] / c_need if c_need else float("inf"))


def pool_L_capacity(model, tp, k_replicas, seqlen):
    """Pool L: PP=1 so only k requests are in flight; weights shard across tp devices."""
    f = model_facts(model)
    w_dev = f["w_total"] / tp
    if w_dev > DEV_BYTES:
        return dict(kv_free_per_device=DEV_BYTES - w_dev, c_max=0.0,
                    c_need=k_replicas, kv_slack=0.0)
    free_fabric = (DEV_BYTES - w_dev) * tp * k_replicas
    c_max = free_fabric / (f["kv_tok_all"] * seqlen)
    return dict(kv_free_per_device=DEV_BYTES - w_dev, c_max=c_max,
                c_need=k_replicas, kv_slack=c_max / k_replicas if k_replicas else float("inf"))


# ---------------------------------------------------------------- raw simulation data loaders
@lru_cache(maxsize=None)
def _read(path):
    return pd.read_csv(path)


def load_pool_L(model, tp):
    """Measured single instance at TP=tp, PP=1. Returns the per-seqlen curve."""
    p = os.path.join(DDA, f"sim_{model}_poolL_{tp}.csv")
    if not os.path.exists(p):
        return None
    df = _read(p)
    r = df[(df["Pipeline parallelism"] == 1) & (df["Tensor parallelism"] == tp)]
    if r.empty:
        return None
    r = r.sort_values("Sequence length")
    return dict(seqlens=r["Sequence length"].values,
                latencies=r["Token latency (ms)"].values,
                pp=1, embedding=r["Embedding latency"].values)


def load_pool_T(model, N_T):
    """Measured Pool T at N_T devices: TP=1, PP=num_layers."""
    p = os.path.join(DDA, f"sim_{model}_poolT_{N_T}.csv")
    if not os.path.exists(p):
        return None
    df = _read(p)
    r = df[(df["Pipeline parallelism"] == MODELS[model]["L"]) & (df["Tensor parallelism"] == 1)]
    if r.empty:
        return None
    r = r.sort_values("Sequence length")
    return dict(seqlens=r["Sequence length"].values,
                latencies=r["Token latency (ms)"].values,
                pp=MODELS[model]["L"],
                cpb=int(r["Channels per block"].iloc[0]),
                embedding=r["Embedding latency"].values)


@lru_cache(maxsize=None)
def cpb_ladder(model):
    """(model, cpb) -> per-block latency by seqlen, for finding-7 heterogeneous packing.

    tbl(cpb) = (token_latency - embedding - InOut) / num_layers, recovered from measured runs
    so the packing correction is a calibrated DELTA on CENT (a uniform config reproduces CENT
    exactly and the only reported change is the packing effect).
    """
    L = MODELS[model]["L"]
    ladder = {}

    def register(df, prefer):
        cpb = int(df["Channels per block"].iloc[0])
        if cpb in ladder and ladder[cpb]["src"] == "pp":
            return
        d = df.sort_values("Sequence length")
        ladder[cpb] = dict(
            src=prefer,
            tbl={int(s): (t - e - INOUT_MS) / L
                 for s, t, e in zip(d["Sequence length"], d["Token latency (ms)"],
                                    d["Embedding latency"])})

    for f in sorted(glob.glob(os.path.join(RES, f"sim_{model}_pp_*dev_lanes32.csv"))):
        register(_read(f), "pp")
    for f in sorted(glob.glob(os.path.join(DDA, f"sim_{model}_poolT_*.csv"))):
        df = _read(f)
        r = df[df["Tensor parallelism"] == 1]
        if len(r):
            register(r, "poolT")
    return ladder


# ---------------------------------------------------------------- static (single-pool) baselines
def static_configs(model, n_devices, prompt_fraction=DEFAULT_PROMPT_FRACTION, valid_only=True):
    """CENT static baselines at `n_devices`, scored with the SAME metrics as every DDA split.

    A static config serves BOTH classes from one pool, so it reports one TTFT and one throughput
    -- which is exactly why it can only sit on a Pareto curve while DDA occupies the corner.

    valid_only drops configs where pp does not divide num_layers: `throughput = pp/token_latency`
    assumes uniform stages, so those rows are formula artifacts (13B MP TP=1 at pp=32 overestimates
    ~1.6x). This is the #1 correctness trap in the codebase.
    """
    L = MODELS[model]["L"]
    frames = []
    for mode in ("pp", "mp"):
        p = os.path.join(RES, f"sim_{model}_{mode}_{n_devices}dev_lanes32.csv")
        if os.path.exists(p):
            frames.append(_read(p))
    if not frames:
        return []
    df = pd.concat(frames)

    out = []
    for (pp, tp), g in df.groupby(["Pipeline parallelism", "Tensor parallelism"]):
        if valid_only and L % int(pp) != 0:
            continue
        # concatenating the pp and mp files can repeat a (pp,tp,seqlen) row; duplicates would
        # make gap = s[1]-s[0] = 0 and blow up the cumulative TTFT integration
        g = (g.drop_duplicates(subset=["Sequence length"])
               .sort_values("Sequence length"))
        s, lat = g["Sequence length"].values, g["Token latency (ms)"].values
        if len(s) < 2 or s[1] == s[0]:
            continue
        out.append(dict(
            pp=int(pp), tp=int(tp), n_devices=n_devices,
            ttft_s=ttft_seconds(s, lat, prompt_fraction),
            req_lat_s=request_latency_seconds(s, lat),
            tput=pool_throughput(s, lat, int(pp))))
    return out


# ---------------------------------------------------------------- Pool T: heterogeneous packing
def pool_T_metrics(model, N_T, prompt_fraction=DEFAULT_PROMPT_FRACTION, seqlen_cap=None):
    """Pool T at N_T devices with finding-7 heterogeneous packing applied.

    CENT maps every device uniformly at cpb_ceil. Reality when L % N_T != 0: TWO device classes.
    `r = L mod N_T` devices hold b_ceil blocks (cpb_ceil), the other N_T-r hold b_floor blocks
    at a HIGHER cpb -> faster blocks. Throughput is UNCHANGED (still bottlenecked by the ceil
    device -- the staircase); only batch latency drops.
    """
    d = load_pool_T(model, N_T)
    if d is None:
        return None
    f = model_facts(model)
    L = f["L"]

    b_ceil, b_floor = math.ceil(L / N_T), L // N_T
    rem = L % N_T
    n_ceil, n_floor = (rem, N_T - rem) if rem else (N_T, 0)
    cpb_ceil = CHANNELS // b_ceil
    cpb_floor = CHANNELS // b_floor if b_floor else cpb_ceil
    used = math.ceil(L / b_ceil)

    ladder = cpb_ladder(model)
    packed_ok = (cpb_ceil in ladder) and (cpb_floor in ladder)
    floor_blocks = n_floor * b_floor

    seqlens, lat_cent = d["seqlens"], d["latencies"]
    lat_packed = []
    for s, tok in zip(seqlens, lat_cent):
        if rem and packed_ok and int(s) in ladder[cpb_ceil]["tbl"] and int(s) in ladder[cpb_floor]["tbl"]:
            save = floor_blocks * (ladder[cpb_ceil]["tbl"][int(s)] - ladder[cpb_floor]["tbl"][int(s)])
            lat_packed.append(tok - save)
        else:
            lat_packed.append(tok)
    lat_packed = np.array(lat_packed)

    cap4096 = pool_T_capacity(model, N_T, 4096)
    capmean = pool_T_capacity(model, N_T, mean_seqlen(seqlens))

    return dict(
        N_T=N_T, cpb_ceil=cpb_ceil, cpb_floor=cpb_floor, n_ceil=n_ceil, n_floor=n_floor,
        b_ceil=b_ceil, b_floor=b_floor, used=used, idle=N_T - used, hetero=bool(rem),
        packed_ok=packed_ok or not rem,
        # THROUGHPUT IS CEIL-BOTTLENECKED, so it comes from the UNPACKED latency.
        # Pipeline rate = 1/max_stage_latency, and the slowest stage is a block on a ceil
        # device at cpb_ceil -- exactly what CENT's uniform mapping already prices. Packing
        # only speeds up the FLOOR devices, which lowers end-to-end request latency but cannot
        # move the bottleneck (finding 4/7: "throughput unchanged").
        # Deriving Tput_T from lat_packed instead inflates it by up to 72% (median 15%) --
        # the same uniform-stage overestimate finding 4 warns about.
        Tput_T=pool_throughput(seqlens, lat_cent, d["pp"]),
        Tput_T_cent=pool_throughput(seqlens, lat_cent, d["pp"]),
        req_lat_T_packed_s=request_latency_seconds(seqlens, lat_packed),
        req_lat_T_cent_s=request_latency_seconds(seqlens, lat_cent),
        ttft_T_packed_s=ttft_seconds(seqlens, lat_packed, prompt_fraction),
        lat_improve_pct=100 * (lat_cent.mean() - lat_packed.mean()) / lat_cent.mean(),
        kv_free_per_block_MiB=cap4096["kv_free_per_block"] / MiB,
        C_max_4096=cap4096["c_max"], C_max_mean=capmean["c_max"],
        C_need=cap4096["c_need"], kv_slack=cap4096["kv_slack"],
        kv_feasible=cap4096["kv_slack"] >= 1.0,
    )


# ---------------------------------------------------------------- Pool L: every factorization
def pool_L_factorizations(model, N_L):
    """All homogeneous packings j instances x TP=m with j*m = N_L and m >= cap_floor.

    Configs with m > TP_sat are KEPT and flagged saturated -- showing throughput flat-or-dropping
    past the knee is a RESULT (7B TP16->TP32: 1119->556 measured), not a reason to exclude.
    Heterogeneous mixes stay excluded (finding 8b): they break Tput_L = sum additivity under FIFO
    with unequal servers, and the partition search is combinatorial.
    """
    f = model_facts(model)
    out = []
    for m in range(1, N_L + 1):
        if N_L % m:
            continue
        if m < f["cap_floor"]:
            continue
        out.append(dict(j=N_L // m, m=m, saturated=m > f["tp_sat"]))
    return out


def pool_L_metrics(model, j, m, prompt_fraction=DEFAULT_PROMPT_FRACTION):
    """j data-parallel replicas, each a TP=m instance (PP=1). Tput adds; TTFT is per-instance."""
    d = load_pool_L(model, m)
    if d is None:
        return None
    seqlens, lat = d["seqlens"], d["latencies"]
    per = pool_throughput(seqlens, lat, d["pp"])
    cap4096 = pool_L_capacity(model, m, j, 4096)
    capmean = pool_L_capacity(model, m, j, mean_seqlen(seqlens))
    return dict(
        n_instances=j, TP_per_instance=m, N_L=j * m,
        Tput_L=per * j, Tput_per_instance=per,
        TTFT_L_s=ttft_seconds(seqlens, lat, prompt_fraction),
        TTFT_L_no_prefill_sampling_s=ttft_seconds_no_prefill_sampling(seqlens, lat, prompt_fraction),
        req_lat_L_s=request_latency_seconds(seqlens, lat),
        saturated=m > model_facts(model)["tp_sat"],
        kv_free_per_device_GiB=cap4096["kv_free_per_device"] / GiB,
        C_max_4096=cap4096["c_max"], C_max_mean=capmean["c_max"],
        C_need=cap4096["c_need"], kv_slack=cap4096["kv_slack"],
        kv_feasible=cap4096["kv_slack"] >= 1.0,
    )


# ---------------------------------------------------------------- load model & optimizer
def alpha_token_share(alpha_req, mean_S_L, mean_S_T):
    """Requests tagged L are a fraction alpha_req of ARRIVALS; feasibility is a TOKEN-flow
    constraint, so convert to token-demand share. Identical mixtures => alpha_tok == alpha_req."""
    num = alpha_req * mean_S_L
    den = num + (1 - alpha_req) * mean_S_T
    return num / den if den else 0.0


def lambda_max(Tput_L, Tput_T, alpha_tok):
    """Max sustainable total token rate: each pool must absorb its own class's share."""
    bl = Tput_L / alpha_tok if alpha_tok > 0 else float("inf")
    bt = Tput_T / (1 - alpha_tok) if alpha_tok < 1 else float("inf")
    return min(bl, bt), ("PoolL" if bl <= bt else "PoolT")


def feasible(row, lam, alpha_tok):
    """A split serves (lambda, alpha) only if BOTH pools have the capacity AND the KV to do it."""
    return (row["Tput_L"] >= alpha_tok * lam
            and row["Tput_T"] >= (1 - alpha_tok) * lam
            and row["kv_feasible"])


EPSILON_DEFAULT = 0.20      # latency tolerance for the epsilon-constraint objective
KV_MARGIN_DEFAULT = 1.0     # gate at the physical wall; report the margin, don't invent one


def annotate_candidates(table, lam, alpha_tok, epsilon=EPSILON_DEFAULT,
                        kv_margin=KV_MARGIN_DEFAULT):
    """Score EVERY split and FLAG which gates it passes. Nothing is ever dropped.

    Full-DSE discipline: a split that fails a gate is still data (it shows what the gate costs).
    Flags added, all independent so any one can be relaxed downstream or in the explorer:

      meets_load    both pools can absorb their class's share at this (lambda, alpha)
      meets_kv      kv_slack >= kv_margin on BOTH pools (default 1.0 = the physical wall)
      unsaturated   m <= TP_sat -- past the knee both axes go flat within ~3% jitter
                    (7B TP8..TP26 all ~600 tok/s, ~0.40 s), so a fat instance there is not a
                    trade, it is idle silicon: 1xTP24 gives 617 tok/s where 3xTP8 gives 1797
      in_eps_band   TTFT_L <= (1+epsilon) * best TTFT_L among load+kv+unsaturated candidates
      selectable    all four -- the set the optimiser is allowed to choose from
    """
    t = table.copy()
    # vectorised: this runs once per (lambda, alpha) grid point, so a row-wise apply here
    # would cost ~2M python-level calls over the full sweep
    tl = t["Tput_L"].to_numpy(dtype=float)
    tt = t["Tput_T"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bl = tl / alpha_tok if alpha_tok > 0 else np.full_like(tl, np.inf)
        bt = tt / (1 - alpha_tok) if alpha_tok < 1 else np.full_like(tt, np.inf)
    t["lambda_max"] = np.minimum(bl, bt)
    t["meets_load"] = (tl >= alpha_tok * lam) & (tt >= (1 - alpha_tok) * lam)
    t["meets_kv"] = (t["kv_slack_T"].fillna(-1).to_numpy(dtype=float) >= kv_margin) & \
                    (t["kv_slack_L"].fillna(-1).to_numpy(dtype=float) >= kv_margin)
    t["unsaturated"] = ~t["saturated"].astype(bool) if "saturated" in t.columns else True

    base = t[t.meets_load & t.meets_kv & t.unsaturated]
    lat = base["TTFT_L_s"].dropna() if len(base) else pd.Series(dtype=float)
    bar = lat.min() * (1 + epsilon) if len(lat) else float("inf")
    t["ttft_bar_s"] = bar
    t["in_eps_band"] = t["TTFT_L_s"].le(bar) & t["TTFT_L_s"].notna()
    t["selectable"] = t.meets_load & t.meets_kv & t.unsaturated

    # ---- balanced objective J: alpha-weighted NORMALISED latency ----------------------
    # J = a*(TTFT_L / best_TTFT_L) + (1-a)*(req_lat_T / best_req_lat_T)
    #
    # Each class is scored against the best achievable FOR THAT CLASS, so the ~20x absolute
    # scale gap between an interactive TTFT (~0.4 s) and a batch completion (~7 s) cannot let
    # one term dominate by units alone. J is the expected *relative* latency degradation seen
    # by a random request under the class mix.
    #
    # Why this and not a fixed epsilon band on TTFT_L: the band ignores alpha, so it blocked
    # trades that are overwhelmingly correct at low alpha. At alpha=0.05 shrinking 7B Pool L
    # from N_L=4 to N_L=1 costs 0.324 s of TTFT_L but saves 0.91 s of batch latency across 95%
    # of requests. J takes that trade and refuses it once alpha is high enough to matter.
    #
    # Emergent properties (verified): N_L starts at cap_floor, rises MONOTONICALLY with alpha,
    # and every device not needed by Pool L goes to Pool T where packing converts it into
    # -0.302 s/device of batch latency.
    if len(base):
        bl = base["TTFT_L_s"].min()
        bt = base["req_lat_T_packed_s"].min()
    else:
        bl = bt = np.nan
    t["J"] = (alpha_tok * t["TTFT_L_s"] / bl
              + (1 - alpha_tok) * t["req_lat_T_packed_s"] / bt)
    t["ttft_ratio"] = t["TTFT_L_s"] / bl
    t["req_lat_T_ratio"] = t["req_lat_T_packed_s"] / bt
    return t


def best_split(table, lam, alpha_tok, epsilon=EPSILON_DEFAULT, kv_margin=KV_MARGIN_DEFAULT,
               n_l_floor=0):
    """EPSILON-CONSTRAINT: max headroom among splits within epsilon of the best latency.

    Two objectives (max throughput, min latency) cannot both be optimised, so the standard
    multi-objective move is to turn one into a constraint:
        1. TTFT_best = best achievable TTFT_L among load+kv+unsaturated candidates
        2. bar       = (1 + epsilon) * TTFT_best
        3. among splits under the bar, take MAX lambda_max (headroom)

    epsilon has an operational meaning ("most headroom among splits within 20% of the best
    latency anyone can offer"): epsilon=0 collapses to pure latency, epsilon=inf to pure
    throughput. Max headroom decides the winner because a reshard costs seconds of drain, so
    distance from the capacity cliff is worth more than a few ms of TTFT.

    Returns (chosen_row_or_None, fully_annotated_table). The table ALWAYS contains every split
    with its flags -- callers report the whole thing, not just the winner.
    """
    t = annotate_candidates(table, lam, alpha_tok, epsilon, kv_margin)
    t["meets_nl_floor"] = t["N_L"] >= n_l_floor
    sel = t[t.selectable & t.meets_nl_floor]
    if sel.empty:
        # floor unsatisfiable at this (lambda, alpha) -> fall back to unconstrained so the grid
        # point still reports a split rather than a hole
        sel = t[t.selectable]
    if sel.empty:
        return None, t

    # SELECT = min J (alpha-weighted normalised latency). See annotate_candidates for the
    # derivation. Tiebreak min N_L, so ties hand devices to Pool T.
    #
    # Not "max lambda_max": lambda_max ties constantly because Tput_T is a staircase (7B: every
    # N_T in 16..31 is one tier at 6239 tok/s), so it decides almost nothing and left the choice
    # to an arbitrary tiebreak. J discriminates inside a tier using the thing that actually
    # varies there -- latency. headroom_ratio is still reported for every candidate.
    #
    # OLD TIEBREAK NOTE (kept as the reason J exists):
    #
    # lambda_max ties constantly, because Tput_T is a staircase: for 7B every N_T in 16..31 is one
    # tier at 6239 tok/s. Within a tier, devices moved from Pool L to Pool T cannot add throughput
    # -- but finding-7 packing means they DO cut batch latency, linearly and hard: 7B loses
    # 0.302 s of req_lat_T per device handed to Pool T (10.832 s at N_T=16 -> 6.299 s at N_T=31).
    #
    # Meanwhile Pool L saturates: TP4..TP7 give TTFT_L 0.458/0.450/0.445/0.446 -- flat within 3%.
    # So breaking ties on TTFT_L over-invests in Pool L. Concretely at alpha=0.05, 1xTP8/N_T24
    # (TTFT_L 0.404, req_lat_T 8.41) vs 1xTP4/N_T28 (0.458, 7.21): +54 ms on the latency class to
    # save 1200 ms on the batch class -- 22x more latency gained than lost, same lambda_max.
    #
    # The epsilon band still protects the latency class (TTFT_L stays within (1+eps) of best), so
    # this spends only the SLACK the band allows, never the guarantee.
    sel = sel.sort_values(["J", "N_L"], ascending=[True, True])
    return sel.iloc[0], t


# ---------------------------------------------------------------- reconfiguration / cooldown
def device_roles(model, n_instances, tp, N_T):
    """Multiset of device ROLES for a split. A role is what fixes a device's physical layout:

        Pool L: ("L", m, shard)      weight row i -> bank i % (512*m); m fixes the modulus and
                                     shard fixes which rows. Instances at equal m are
                                     interchangeable, so only (m, shard) matters.
        Pool T: ("T", blocks, cpb)   block b -> a device holding cpb channels.

    A Counter, so the diff is over role TYPES rather than device indices -- devices are freely
    permutable, so the scheduler can always assign roles to minimise movement.
    """
    from collections import Counter
    roles = Counter()
    for s in range(tp):
        roles[("L", tp, s)] += n_instances

    f = model_facts(model)
    L = f["L"]
    if N_T > 0:
        b_ceil, b_floor = math.ceil(L / N_T), L // N_T
        rem = L % N_T
        n_ceil, n_floor = (rem, N_T - rem) if rem else (N_T, 0)
        cpb_ceil = CHANNELS // b_ceil
        cpb_floor = CHANNELS // b_floor if b_floor else cpb_ceil
        b = 0
        for _ in range(n_ceil):
            if b >= L:
                break
            roles[("T", tuple(range(b, min(b + b_ceil, L))), cpb_ceil)] += 1
            b += b_ceil
        for _ in range(n_floor):
            if b >= L:
                break
            roles[("T", tuple(range(b, min(b + b_floor, L))), cpb_floor)] += 1
            b += b_floor
    return roles


def reconfig_diff(model, split_a, split_b):
    """Devices that must DRAIN moving split_a -> split_b. Each split is (n_instances, tp, N_T).

    Cost is wildly asymmetric:
      LOSES a role -> drain (seconds..20 s): in-flight requests must finish, and autoregressive
                      decode cannot pipeline its own tokens, so you pay the full remaining
                      generation, not a pipeline-rate flush.
      GAINS a role -> weight load only (~23-270 ms over CXL); nothing is in flight.
      KEEPS a role -> free.
    Drain dominates weight transfer ~1000x, so cost ~ (devices losing a role).

    Cheap axes this exposes for the controller:
      * fixed m, changing replica count -> existing instances untouched -> ZERO drain
        (2xTP8 -> 3xTP8 adds 8 fresh devices and re-shards nothing)
      * changing m -> modulus 512*m changes -> EVERY Pool L device re-shards
      * Pool T under heterogeneous packing re-distributes even at constant bpd, so same-bpd
        N_T moves are CHEAPER BUT NOT FREE (7B N_T 16->20: 12 devices keep 2blk@cpb16,
        4 drain, 8 arrive empty).
    """
    ra = device_roles(model, *split_a)
    rb = device_roles(model, *split_b)
    kept = sum((ra & rb).values())
    return dict(devices_before=sum(ra.values()), devices_after=sum(rb.values()),
                kept=kept, draining=sum(ra.values()) - kept,
                arriving=sum(rb.values()) - kept)


def drain_seconds(model, N_T):
    """Worst-case drain: a request that just entered must run to full context before its device
    can be repurposed. Cumulative token latency 1..4096 for that Pool T config."""
    d = load_pool_T(model, N_T)
    if d is None:
        return None
    return float(_cumulative_seconds(d["seqlens"], d["latencies"])[-1])


def switch_economics(model, split_a, split_b, lam_max_a, lam_max_b, N_T_a):
    """Is a split change worth paying for?

        loss        = lambda_max_old * T_drain     tokens not served during the switch
        T_breakeven = loss / (lambda_max_new - lambda_max_old)

    Switch only if the new alpha is expected to persist longer than T_breakeven -- the
    hysteresis that stops split-thrashing. A move gaining 5% capacity but costing 20 s of drain
    needs the new load to hold for minutes before it pays for itself.
    """
    diff = reconfig_diff(model, split_a, split_b)
    t_drain = drain_seconds(model, N_T_a) if diff["draining"] else 0.0
    if t_drain is None:
        t_drain = 0.0
    gain = lam_max_b - lam_max_a
    loss = lam_max_a * t_drain
    return dict(draining=diff["draining"], kept=diff["kept"], arriving=diff["arriving"],
                T_drain_s=t_drain, tokens_lost=loss, gain_tok_s=gain,
                T_breakeven_s=(loss / gain) if gain > 0 else float("inf"),
                free_move=diff["draining"] == 0)