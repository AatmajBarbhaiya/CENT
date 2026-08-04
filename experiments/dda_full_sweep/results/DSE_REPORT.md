# Full Design-Space Exploration — DDA on a 32-Device CENT Fabric

**Date**: 2026-07-29 (findings 12–15 appended 2026-08-04) · **Scope**: Llama2-7B / 13B / 70B, N_total = 32 devices
**Data**: 118 pool simulations (0 failures) → 154 splits → 23,280 (λ, α) decisions
**Reproduce**: `bash simulation_dse_fill.sh 8 128 && python3 dse_full_table.py && python3 best_split_by_rate_alpha.py`

---

## 1. What was asked and what was built

Exhaustive exploration of every valid Pool L / Pool T split, with the throughput of both pools
and the combined system, the latency of the latency pool, and every way of packing instances
into Pool L — **including splits that perform badly**, since the point is a complete map.

Four artifacts, all in `results/dda32/`:

| file | contents |
|---|---|
| `sim_{model}_pool{L,T}_{N}.csv` | 118 raw per-seqlen simulation outputs |
| `dse_full_table.csv` | 154 splits × 35 columns — every metric, every gate flag |
| `best_split_by_rate_alpha32.csv` | optimal split over 97 α × 80 λ per model |
| `dse_fill_failures.log` | empty — no config failed |

Shared metric code lives in `explorer/dse_metrics.py` and is imported by both the CSV writers
and (next) the Streamlit explorer, so there is exactly one implementation of every formula.

### Coverage

| model | splits | N_L range | N_T range | distinct (j × TP) | saturated | KV-infeasible |
|---|---|---|---|---|---|---|
| 7B | 92 | 1–26 | 6–31 | 91 | 23 | 0 |
| 13B | 53 | 2–22 | 10–30 | 52 | 13 | 0 |
| 70B | 9 | 9–16 | 16–23 | 8 | 0 | 0 |

Bounds are `cpb = 32//ceil(L/N_T) ≥ min_cpb` for Pool T and `m ≥ cap_floor` for Pool L.
Every legal integer split is present. Saturated configs (`m > TP_sat`) are **kept and flagged**,
never dropped.

---

## 2. Corrections to prior work

### 2.1 🔴 70B weight footprint was wrong — cap_floor is 9, not 10

`realistic_dda_sweep.py` hand-set `WEIGHTS_GIB["Llama2-70B"] = 145.0`, pricing Wk/Wv at full
width. CENT stores them GQA-reduced — `function_sim.py:29-30`:

```python
"wk": torch.zeros((head_dim * n_kv_heads), dim),   # 1024 × 8192, not 8192 × 8192
```

Correct per-block weight is `2d² + 2·d·head_dim·n_kv_heads + 3·d·ffn` = **1632 MiB**, total
**128.5 GiB**, so `cap_floor = ceil(128.5/16) = 9`.

**Impact**: unlocks `1×TP9 / N_T=23`, which delivers **1213 tok/s vs finding 8's 1057 — +15%**.
N_T=23 had never been simulated. All footprints are now derived from architecture in code;
none are hand-set.

### 2.2 `minimal_channel_per_block` is also the KV capacity wall

CLAUDE.md asserted min_cpb was purely tiling legality, "not capacity-driven — a batch-1 block
needs <4 ch". That assumed batch-1 concurrency, but a PP pipeline needs `pp = num_layers`
requests in flight to deliver its reported throughput, and every in-flight request holds KV on
**every** block. Testing one cpb below each hand-set floor at S=4096:

| model | cpb below floor | fits | needs | | at floor | fits | needs |
|---|---|---|---|---|---|---|---|
| 7B | 4 | 26 | 32 ✗ | | **5** | 34 | 32 ✓ |
| 13B | 7 | 37 | 40 ✗ | | **8** | 44 | 40 ✓ |
| 70B | 5 | 58 | 80 ✗ | | **6** | 90 | 80 ✓ |

Three for three, tight on both sides. Arithmetic verified; authors' intent not.

### 2.3 🔴 Pool T throughput must come from the UNPACKED latency (found in final verification)

`pool_T_metrics` initially derived `Tput_T` from the **packed** latency. Wrong: pipeline rate is
`1/max_stage_latency`, and the slowest stage is a block on a **ceil** device at `cpb_ceil` —
exactly what CENT's uniform mapping already prices. Packing only speeds up the **floor** devices,
lowering end-to-end request latency without moving the bottleneck. Finding 7 says so explicitly
("throughput unchanged").

Inflation from the bug: **median +15%, max +72%** (7B N_T=31: 10,728 vs the true 6,239) — a fresh
instance of the uniform-stage overestimate finding 4 warns about. Fixed; both tables regenerated.
`Tput_T == Tput_T_cent` is now asserted for every row in the verification pass.

### 2.4 Metric definitions that were quietly wrong

| old | problem | now |
|---|---|---|
| flat mean over seqlen rows | treats occupancy 4096 as likely as 128 when it is 32× rarer | **survival weighting** `P(S ≥ s)` |
| `mean(lat[s≤512]) × 512` | assumes every request has a 512-token prompt | **cumulative sum** to `0.125·S` |
| `TTFT_T_cent/packed` | never a TTFT — it is completion latency | renamed `req_lat_T_*` |
| class = seqlen threshold | conflates "short" with "urgent" | **pre-tagged classes** |

Survival weighting alone moves 7B/32-dev from 3.084 ms / 10,378 tok/s to
**2.781 ms / 11,505 tok/s** — latency −9.8%, throughput +10.9%, the same order as the effects
being measured.

---

## 3. Model

**Pre-tagged classes.** Class (latency L / batch T) is assigned by the client/SLA before arrival,
independent of length, so both pools see the full 128–4096 range.

**Mixture.** Uniform over the simulated grid, identical for both classes — the scientific control,
so any result is attributable to device allocation rather than a length mix baked into the input.
Zero free parameters, and `α_tok = α_req` exactly.

**Prompt fraction 0.125** is CENT's arbitrary CLI default (`--prefill 512 --decoding 3584`), not a
principled value. It affects **TTFT only**; throughput and KV depend on total `S`. Exposed as a
slider, sensitivity to be reported at 0.125 / 0.5 / 0.9.

**Capacity gate.**
```
Pool T:  KV_free = cpb_ceil × 0.5 GiB − W_block ;  C_need = L
Pool L:  KV_free = (16 GiB − W_total/m) × m × j ;  C_need = j
kv_slack = C_max / C_need   at S = 4096
```

**Objective — ε-constraint, ε = 0.20.** Among splits passing load + KV + unsaturated gates, admit
those within `(1+ε)` of the best achievable TTFT_L, then take **max λ_max**. ε=0 is pure latency,
ε=∞ pure throughput. Max headroom decides the winner because a reshard costs seconds of drain.

---

## 4. Results

### 4.1 Best by criterion (unsaturated, KV-feasible)

| model | max system throughput | min TTFT_L |
|---|---|---|
| 7B | **11,605 tok/s** — 16×TP1 / N_T16, TTFT 0.782 s | **0.404 s** — 1×TP8 / N_T24, tput 6,838 |
| 13B | **6,020 tok/s** — 6×TP2 / N_T20, TTFT 0.946 s | **0.607 s** — 1×TP10 / N_T22, tput 4,791 |
| 70B | **1,121 tok/s** — 1×TP12 / N_T20, TTFT 2.063 s | **1.891 s** — 1×TP16 / N_T16, tput 874 |

Max servable λ: **7B 11,424 · 13B 5,926 · 70B 1,104 tok/s.**

#### The 70B cap_floor fix is a correctness fix, not a throughput win

An earlier draft claimed cap_floor 10 → 9 unlocked `1×TP9/N_T23 = 1213 tok/s`, "+15% over
finding 8". **That was an artifact of the §2.3 inflation** — N_T=23 is heterogeneous
(80 mod 23 = 11) and was over-credited. Corrected:

| N_L | split | Tput_T | system tput |
|---|---|---|---|
| 9 | 1×TP9 / N_T23 | 1010.9 | 1118.2 |
| **12** | **1×TP12 / N_T20** | 1010.9 | **1121.2** ← best |

N_T ∈ {20…23} share one staircase tier (`bpd=4, cpb=8`), so N_L=12 wins only on its slightly
larger Pool L. cap_floor=9 adds a legal option marginally *worse* than one already available at
cap_floor=10. The GQA weight correction (145 → 128.5 GiB) remains necessary and correct — it just
does not buy throughput.

### 4.2 🔴 70B admits no data parallelism at all

`cap_floor = 9` and `N_L ≤ 16` force `j = 1` for every legal N_L — **all 8 of 70B's Pool L
configs are single instances.** 70B has only the TP axis on this fabric; its split map is 5 points
wide where 7B's is 55. Any DP-based argument is 7B/13B only.

### 4.3 ε=0.2 avoids both failure modes

The objective settles on TP3–TP5 through most of the α range, avoiding the TP=1 weight-tax
extreme (78% of the device is weights, max context 7,064 tokens) and the saturated fat instances.
Split map at λ ≈ 50% of each ceiling:

| model | α → split |
|---|---|
| 7B | 0.02→1×TP4/N_T28 · 0.07→2×TP4/N_T24 · 0.20→4×TP3/N_T20 · 0.44→9×TP2/N_T14 · 0.71→13×TP2/N_T6 · 0.94→26×TP1/N_T6 |
| 13B | 0.02→1×TP5/N_T27 · 0.14→3×TP4/N_T20 · 0.34→6×TP3/N_T14 · 0.55→11×TP2/N_T10 |
| 70B | 0.02→1×TP9/N_T23 · 0.10→1×TP12/N_T20 · 0.13→1×TP16/N_T16 |

N_L rises monotonically with α, reaching TP=1 only at α ≥ 0.94 where Pool L truly dominates.

### 4.4 Why fat instances past TP_sat are waste, not a trade

Past `TP_sat = d/512`, **both axes go flat** (7B, single instance):

| TP | 8 | 12 | 16 | 20 | 24 | 26 |
|---|---|---|---|---|---|---|
| tput | 599 | 608 | 605 | 596 | 617 | 612 |
| TTFT | 0.404 | 0.396 | 0.398 | 0.406 | 0.388 | 0.392 |

±3% non-monotonic jitter, no trend. So in 24 devices, `1×TP24` yields 617 tok/s @ 0.388 s while
`3×TP8` yields **1797 tok/s @ 0.404 s** — 2.9× the throughput for 1.6% more latency. These rows
stay in the table as evidence and are excluded from selection only.

### 4.5 🔴 Reconfiguration: fine-grained α-chasing never pays

Gain measured **at the new α** — "does switching beat standing pat now", not "was the old split
better before":

| model | devices draining (median) | T_drain (median) | T_breakeven p25 / **p50** / p95 |
|---|---|---|---|
| 7B | 8 | 23.5 s | 374 s / **995 s** / 9,524 s |
| 13B | 14 | 41.1 s | 558 s / **1,234 s** / 13,536 s |
| 70B | 21 | 348.2 s | 12,895 s / **42,889 s** / 128,019 s |

**α must hold ~17 min (7B), ~21 min (13B), ~12 hours (70B) before a split change repays its
drain.** Adjacent-α moves buy tiny capacity deltas at the cost of a full drain. The controller
must work on coarse α bands with long dwell times; for 70B the honest recommendation is **pick
one split and never reconfigure**.

### 4.6 🔴 No zero-drain moves exist on a saturated fabric

**0 of 2,129 split changes were free.** The hypothesised cheap axis — add replicas at fixed `m`,
leaving existing instances untouched — does not exist here, because Pool L and Pool T share a
fixed 32-device budget: growing Pool L shrinks Pool T, which redistributes its heterogeneous
packing and re-shards. The fixed-m insight remains correct in isolation and would apply on a
fabric with spare devices.

Related correction: same-bpd N_T moves are **cheaper but not free**. Under uniform CENT mapping
cpb is unchanged, but under finding-7 packing the ceil/floor counts shift — 7B N_T 16→20 keeps 12
devices at 2blk@cpb16, drains 4, and brings 8 in empty. Role-diff still beats a full-fabric charge
by ~8×.

### 4.7 Thin-KV configurations

Nine splits sit under 1.3× slack at S=4096 and must never be quoted as comfortable:

| model | N_T | cpb | kv_slack | fits / needs |
|---|---|---|---|---|
| 7B | 6 | 5 | **1.06** | 34 / 32 |
| 13B | 10–13 | 8 | **1.09** | 43.6 / 40 |
| 70B | 16–19 | 6 | **1.13** | 90 / 80 |

Gated at `kv_margin = 1.0` (physics only), consistent with the SLO gate already removed. A 1.2
margin rejects all nine — including 70B N_T=16, which carries finding 8's +24% λ_max result.

---

### 4.8 The alternative to reconfiguration — spillover, and the Δα break-even rule

Since split changes rarely pay (4.5), reconfiguration must be the **last** response to an α shift,
not the first. Four options, cheapest first:

1. **Stand pat** — while both pools stay under capacity the mismatch is only queueing delay. Free.
2. **🟢 Spillover** — route excess latency-class requests to Pool T. A software policy: zero drain,
   zero reshard, instant. Spilled requests get Pool T's worse TTFT, so the cost is service quality,
   not capacity. **This is the right first response.**
3. **Admission control** — shed or defer excess.
4. **Reconfigure** — only for sustained regime change.

#### Break-even rule

A split chosen for α_old has `Tput_L ≈ α_old·λ`. If α drifts by Δα and we stand pat, Pool L is
overloaded by `Δα·λ`, accumulating `Δα·λ·T` backlog over dwell T. Reconfiguring instead forfeits
`λ·T_drain`. Equating:

```
Δα · λ · T = λ · T_drain     →     Δα_breakeven = T_drain / T_dwell
```

**λ cancels.** The threshold depends only on drain time and how long the new α persists:

| dwell | 7B (T_drain 23.5 s) | 13B (41.1 s) | 70B (407 s) |
|---|---|---|---|
| 1 min | 0.391 | 0.685 | 6.78 ✗ |
| 5 min | 0.078 | 0.137 | 1.36 ✗ |
| 10 min | 0.039 | 0.069 | 0.678 |
| 30 min | 0.013 | 0.023 | 0.226 |
| 1 hour | 0.007 | 0.011 | 0.113 |

✗ = exceeds the entire α range → reconfiguration can never pay at that dwell. **70B cannot justify
a resplit for anything shorter than a ~10-minute regime change.**

#### Spillover headroom

`spare_T = Tput_T − (1−α)·λ`, so spillover absorbs `Δα ≤ spare_T / λ`:

| model | at 50% of max λ | at 80% of max λ |
|---|---|---|
| 7B (α=0.2) | Δα ≤ **0.28** | Δα ≤ **−0.13** (none — Pool T already oversubscribed) |
| 13B (α=0.2) | Δα ≤ 0.66 | Δα ≤ 0.11 |
| 70B (α=0.2) | Δα ≤ 1.02 (any) | Δα ≤ 0.34 |

⚠️ **Spillover room shrinks exactly when it is most needed.** At 50% load it absorbs α swings far
larger than anything worth reconfiguring for; at 80% load 7B has *negative* spare, leaving only
admission control or a resplit. Practical policy: **spillover below ~70% load, reconfigure only on
sustained drift above it.**

---

## 5. Batching — checked, and it is a result rather than a gap

`grep -c batch cent_simulation/*.py` = **0**; batch-1 is structural
(`cache_k = zeros((1, seqlen, …))`). Batching would barely help: GPU batching pays because the
weight matrix is fetched from HBM once and amortised over B tokens, but **in AiM the weights never
move** — `MAC_ABK` computes inside the bank against a row that stays put. Nothing to amortise, so
B tokens cost B× the time and **PIM throughput is approximately batch-invariant**.

That is a strong argument for the paper: on GPUs you buy throughput with batch size; on PIM that
lever does not exist, so **the only throughput lever is device allocation** — the DDA thesis. It
also explains why CENT was written batch-1.

Separately, `InOut_latency = 0.15 ms` (host top-K sampling) is charged on every token including
all prompt tokens, but sampling happens once per request. That inflates TTFT by ~77 ms on a
512-token prompt (~5%). Reported as a separate column, never silently applied.

---

## 6. Open items

1. **`explorer/app.py`** — Streamlit, sliders for model / λ / α_req / prompt fraction / ε /
   kv-margin; best-split card, ranked candidate table with all gate flags, Pareto scatter,
   32-device KV strip.
2. **Regenerate paper figures** from `dse_full_table.csv` (supersedes finding 8's spectrum).
3. **Prompt-fraction sensitivity** at 0.125 / 0.5 / 0.9 — post-processing only, no new sims.
4. **Numbers differ from finding 8 by design** — survival weighting +11% throughput, cumulative
   TTFT lower, tagged mixture removes the seqlen filters. Both column sets retained so the diff
   is auditable.

---

## 7. Later findings (2026-07-29 → 08-04) — see CLAUDE.md for the full text

| # | finding |
|---|---|
| 12 | **Controller implemented and evaluated.** 27/27 asserted traces pass. Hysteresis costs ~1.5× TTFT and saves ~32× the drain vs greedy; beats static outright. Only **5/21 (7B), 9/16 (13B), 0/4 (70B)** rungs are ever worth switching between, and 4 of 7B's 5 share N_T=16 — the work happens in the **k** dimension at constant Pool T size. |
| 13 | **Fair baseline under a dynamic workload.** Replaying the same 24 h α(t) through native / static-32 / DDA-static / DDA-dynamic. DDA's benefit tracks how badly a uniform mapping wastes the fabric, not model size. |
| 13b | **The headline comparison** — at equal latency for the latency class, how much TOTAL fabric throughput? Compared against the *whole* static Pareto curve, not one hand-picked config. |
| 14 | **24 h motion sweep**, 30/30 pass. Two timescale bugs found, both the same shape: *a motion's name does not guarantee its timescale*. `brief spike` was 18× longer than T_grow; `flapping` at `period_h = horizon` was a single step with a 12 h half-period. |
| 15 | **Spill, not binary feasibility.** An undersized Pool L is not a failure — its excess spills to Pool T and is served slowly. Only demand neither pool can take queues. This is also what makes the frontier depend on (α, λ) at all. |

### Corrections this report's own numbers are subject to

Section 4's effective-TTFT comparison (finding 13) uses a **single hand-picked static config** and
a binary feasibility rule. Both were later corrected — findings 13b and 15. **Section 4 remains
valid as an honesty check** ("under one dynamic workload, whose effective TTFT is lower"), but
**finding 13b is the headline**: at a fixed latency bar, who has more total capacity. The two
measure different things and both should appear in the paper, with 13b leading.

### Current artifact map

- `results/dda32/frontier_{static,dda,matched}.csv` — the headline comparison
- `results/dda32/controller_*.csv` — controller evaluation
- `results/dda32/motion_sweep_24h.csv` — every α motion on one 24 h clock
- `figures/figure_tput_latency_frontier.pdf` — the headline figure
- `app/app.py` — where the (α, λ) exploration actually happens; the PDFs are single frames of it
