# DDA Split-Selection Algorithm & Reconfiguration Controller — Plan

**Date**: 2026-07-29 · **Status**: design, not yet implemented
**Grounding data**: `results/dda32/dse_full_table.csv` (154 splits, 118 sims), findings 4/7/10/11

---

## 0. The three facts that force the design

**F1 — Pool T throughput is a staircase.** For 7B every `N_T ∈ 16…31` is one tier at 6239 tok/s;
13B has 3 tiers, 70B has 2. Throughput only moves when `bpd = ceil(L/N_T)` changes. So there are
**very few throughput-distinct Pool T sizes** — the real search space is tiny.

**F2 — Only spillover is free. Every split change re-shards BOTH pools.**
- Pool L: weight row `i → bank i % (512·m)`. Changing `m` moves every row.
- Pool T: `cpb = 32//bpd` plus the finding-7 ceil/floor packing split. Changing `N_T` redistributes.
- Since `N_L + N_T = 32`, **any N_L change perturbs Pool T too.** Measured: 0 of 1284 transitions
  were zero-drain. Median drain 12 devices (7B), 14 (13B), 21 (70B); T_drain 23.5 / 63.3 / 348 s.

**F3 — Replica freedom collapses with model size.** Per-device weight share is `(W_total/m)/16 GiB`
— independent of `j`. But `cap_floor ≤ m` and `N_L ≤ 32 − N_T_min` bound `j = N_L/m`:

| model | cap_floor | TP_sat | j range | note |
|---|---|---|---|---|
| 7B | 1 | 8 | **1…26** | full freedom |
| 13B | 2 | 10 | 1…11 | moderate |
| 70B | 9 | 16 | **1 only** | k is not a variable; only `m` is |

---

## 1. Phase 0 — Offline: build the split ladder (once per model)

Do not search 92 splits at runtime. Precompute a **short ladder of rungs** so reconfiguration is
rare by construction.

```
LADDER(model):
  cands = { (N_L, j, m) : j*m = N_L,  cap_floor <= m <= TP_sat,
                          N_T = 32 - N_L in legal Pool T range,
                          kv_slack_L >= 1 and kv_slack_T >= 1 }          # physics only

  # F1: collapse Pool T to throughput-distinct tiers. Within a tier, the SMALLEST N_T gives the
  # same Tput_T using the fewest devices, freeing the rest for Pool L; the LARGEST gives the best
  # batch latency. Keep both endpoints of every tier as rungs.
  tiers    = group cands by Tput_T
  rung_NTs = { min(N_T), max(N_T) for each tier }  U  { 32 - cap_floor }

  # For each rung N_L, keep only the Pareto-efficient factorizations on (Tput_L up, TTFT_L down).
  # This is where k is chosen; for 70B the set is a singleton.
  for N_L in {32 - nt for nt in rung_NTs}:
      keep pareto_front over (Tput_L, -TTFT_L) among factorizations of N_L

  return ladder sorted ascending by N_L        # ~5-8 rungs per model, not 92
```

**Why this shrinks reconfigurations**: rungs sit only where throughput actually changes, so α drift
inside a tier never triggers a move.

### The actual ladders (computed, not estimated)

⚠️ The "~5–8 rungs" estimate above was wrong. Collapsing Pool T to tier endpoints does cut the
N_T axis hard, but each surviving N_T still carries a **Pareto front of Pool L factorizations**,
and that is where the real width is. Measured: **7B 21 rungs · 13B 16 · 70B 4** (vs 92/53/9 splits
in the full table — a 4×/3×/2× reduction, not 12×).

**Llama2-70B — 4 rungs** (cap_floor 9, TP_sat 16, 2 throughput tiers, `j ≡ 1`):

| N_T | N_L | cfg | Tput_L | Tput_T | TTFT_L | req_lat_T | kv_T |
|---|---|---|---|---|---|---|---|
| 16 | 16 | 1×TP16 | 117 | 756 | 1.891 | 223.4 | 1.12 |
| 19 | 13 | 1×TP13 | 111 | 756 | 2.056 | 181.5 | 1.12 |
| 20 | 12 | 1×TP12 | 110 | 1011 | 2.063 | 167.2 | 1.93 |
| 23 | 9 | 1×TP9 | 107 | 1011 | 2.143 | 152.8 | 1.93 |

Exactly the tier endpoints, since `j ≡ 1` leaves nothing to factorize. Pool L throughput barely
moves (107→117, **9% across the entire ladder**) while batch latency swings 153→223 s. So for 70B
the ladder is almost purely a *batch-latency* choice — another reason its policy is "provision once."

**Llama2-13B — 16 rungs** across N_T ∈ {10, 12, 14, 18, 20, 30}; richest at N_T=12 and N_T=20
(4 factorizations each, e.g. N_L=12 as 6×TP2 / 4×TP3 / 3×TP4 / 2×TP6 — Tput_L 1618→715 as TTFT_L
improves 0.946→0.678).

**Llama2-7B — 21 rungs** across N_T ∈ {6, 7, 8, 10, 11, 15, 16, 31}; richest at N_T=8, where
N_L=24 factorizes six ways:

| cfg | Tput_L | TTFT_L | KV slack (Pool L) |
|---|---|---|---|
| 24×TP1 | 8050 | 0.782 | 1.7 |
| 12×TP2 | 5414 | 0.562 | 9.7 |
| 8×TP3 | 3960 | 0.507 | 17.7 |
| 6×TP4 | 3238 | 0.458 | 25.7 |
| 4×TP6 | 2220 | 0.445 | 41.7 |
| 3×TP8 | 1797 | 0.404 | 57.7 |

A clean 4.5× throughput ↔ 1.9× TTFT trade at **constant device count** — the k dimension doing
exactly the job it exists for.

**Cold-start floors are all reachable**: the ladder's smallest rung is N_L=1 (7B), 2 (13B), 9 (70B)
— each equal to `cap_floor`, so `PROVISION` can always start at the minimum.

**If 21 rungs proves too many** (watch trace 6, flapping): cap the Pareto front to the 3 most
distinct factorizations per N_T. Do this only if the eval shows thrash — the hysteresis is supposed
to handle it, and pruning trades away real operating points.

---

## 2. Phase 1 — Cold start: minimum feasible N_L, remainder to Pool T

```
PROVISION(model, lambda, alpha):
  load_L = alpha * lambda
  load_T = (1 - alpha) * lambda
  for (N_L, j, m) in LADDER ascending by N_L:            # smallest Pool L first
      if Tput_L(j,m) >= load_L and Tput_T(32-N_L) >= load_T:
          return (N_L, j, m)                            # first feasible = minimum N_L
  return INFEASIBLE                                     # needs admission control
```

Minimum N_L is correct here because within a Pool T tier the freed devices are **not** idle — via
finding-7 packing each one buys **−0.302 s of batch latency** (7B: 10.832 s at N_T=16 → 6.299 s at
N_T=31), at zero throughput cost. Measured floors: 7B N_L=1, 13B N_L=2, 70B N_L=9 — all equal to
`cap_floor`.

**Tie-break among factorizations of the chosen N_L**: minimise
`J = α·(TTFT_L/best_TTFT_L) + (1−α)·(req_lat_T/best_req_lat_T)`
— α-weighted *normalised* latency, so the ~20× absolute scale gap between an interactive TTFT
(~0.4 s) and a batch completion (~7 s) cannot let one term win on units alone.

---

## 3. Phase 2 — Runtime: three tiers, cheapest first

α is observed, noisy, and drifting. Respond in escalating cost order and **never skip a tier**.

### Tier A — Spillover (free, instant, always first)

Route excess latency-class load to Pool T. Zero drain, zero reshard; spilled requests simply get
Pool T's latency instead of Pool L's.

```
spare_T   = Tput_T - load_T
excess_L  = max(0, load_L - Tput_L)
if excess_L <= spare_T:  spill(excess_L);  return NO_RECONFIG
```

Capacity (finding 11): at 50% of max λ this absorbs Δα ≤ 0.28 (7B) / 0.66 (13B) / 1.02 (70B) —
larger than any drift worth resharding for. ⚠️ **But at 80% load 7B's spare goes negative**, so
spillover is a low/mid-load mechanism. Above ~70% load, Tier C becomes reachable sooner.

### Tier B — Grow k at fixed m (cheapest real reconfiguration)

Adding a replica at unchanged `m` leaves existing instances bit-identical (same modulus `512·m`),
so **no Pool L instance drains**. Only the `m` devices arriving from Pool T pay a weight load
(~23–270 ms) — plus Pool T's own repacking, which is unavoidable under F2.

```
if exists (N_L + m, j+1, m) in LADDER and it clears load:
    cost = drain(Pool T devices whose (blocks,cpb) role changed)     # Pool L: 0
    prefer this over Tier C
```
Available for 7B and 13B. **Not available for 70B** (j≡1) — it goes straight to Tier C.

### Tier C — Full reconfiguration (change m, or jump rungs)

Both pools re-shard. Gate it behind the commit rule below.

---

## 4. Phase 3 — The commit rule (hysteresis)

From finding 11, standing pat accrues backlog `Δα·λ·T` while reconfiguring forfeits `λ·T_drain`:

```
Δα_breakeven = T_drain / T_dwell          # lambda cancels
```

so **commit only if the drift is both large enough and persistent enough**:

```
COMMIT(model, alpha_now, alpha_smoothed):
  # EWMA over a window W >= T_drain, so the estimate cannot be faster than the actuator
  W = max(2 * T_drain, 60 s)
  if spillover still absorbs the excess:            return HOLD
  if |alpha_smoothed - alpha_deployed| < T_drain / T_hold:  return HOLD
  if sustained_for(excess > spare_T) < T_hold:      return HOLD
  return RECONFIGURE(to lowest-cost tier that clears load)
```

**Asymmetric thresholds — the key to not thrashing:**

| direction | trigger | sizing | why |
|---|---|---|---|
| **grow** N_L | spillover exhausted for `T_grow ≈ 2·T_drain` | **minimum** rung that clears load | latency class is being missed; act now |
| **shrink** N_L | over-provisioned for `T_shrink` (below) | **one large step**, not a trickle | pure QoS gain; never urgent |

Monotone-ish N_L then **emerges** from the asymmetry rather than being imposed — which matters,
because forcing strict monotonicity would itself cause a reshard on every α tick.

### Shrinking: yes, N_L must come back down — but slowly and in bulk

When α falls, Pool L is over-provisioned. Nothing is being missed; the only prize is batch latency
(each device returned to Pool T is worth `Δ_lat` below). The drain, however, delays roughly
`λ·T_drain/E[S]` requests by ~`T_drain/2`. Repaying that gives a **quadratic** threshold:

```
T_shrink  =  T_drain² / ( 2 · (1−α) · Δ_lat · devices_freed )
```

| model | Δ_lat per device | shrink 2 dev | shrink 4 dev | shrink 8 dev |
|---|---|---|---|---|
| 7B | 0.302 s | ~11 min | ~5 min | ~3 min |
| 13B | 0.439 s | ~54 min | ~27 min | ~14 min |
| 70B | **4.798 s** | **~2.5 h** | **~75 min** | ~38 min |

(at α=0.3; scales mildly with α)

**Design consequence — shrink in bulk.** Freeing 8 devices repays in 3 min on 7B; freeing 2 takes
11. So the controller should *wait* and take one large step down rather than trickling devices
back — the exact opposite of the grow path, which takes the minimum rung that clears the load.

**70B effectively never shrinks**: ~2.5 h for a 2-device move, with only 2 rungs and `j≡1`. Its
honest policy stays "provision once, spill, leave it alone."

On any shrink, `k` is **re-chosen jointly** — the shrink re-runs `PROVISION`, re-optimising the
whole `(N_L, j, m)` triple against the new α, not inheriting the old factorization.

Concrete dwell requirements (median T_drain):

| model | T_drain | Δα needed @10 min dwell | @1 h | practical policy |
|---|---|---|---|---|
| 7B | 23.5 s | 0.039 | 0.007 | reconfigure on shift ≥0.05 held ≥10 min |
| 13B | 63.3 s | 0.069 | 0.011 | ≥0.10 held ≥10 min |
| 70B | 348 s | 0.678 | 0.113 | **never reconfigure**; provision once, spill always |

---

## 5. Full loop

```
ladder = LADDER(model)                                  # Phase 0, offline
split  = PROVISION(model, lambda_0, alpha_0)            # Phase 1, minimum N_L
deploy(split)

every control period (>= T_drain):
    lambda, alpha = observe()
    alpha_s       = ewma(alpha, W)
    if TierA_absorbs(split, lambda, alpha):   continue          # free
    if not COMMIT(model, alpha, alpha_s):     continue          # hold, keep spilling
    target = TierB_candidate(split) or PROVISION(model, lambda, alpha_s)
    if lambda_max(target) - lambda_max(split) <= 0:  continue   # no capacity gain, don't pay
    drain(role_changed_devices);  reshard();  deploy(target)
```

---

## 6. Implementation plan

| file | contents |
|---|---|
| `explorer/dda_controller.py` | `LADDER`, `PROVISION`, `TierA/B/C`, `COMMIT`, `run_trace` |
| `simulation/controller_eval.py` | replay synthetic α(t) traces → reconfig count, spill fraction, SLO misses, tokens lost |

---

## 7. Evaluation plan — the α(t) trace suite

**Purpose**: show *when and where* requests get routed after spillover, and *where* reconfiguration
actually fires. Each trace targets a specific corner; the "expected" column is an **assertion**, so
a controller bug shows up as a failed expectation rather than a plausible-looking number.

### Measured timescales that set trace lengths

| model | T_drain | T_grow = 2·T_drain | Δ_lat/device | T_shrink (2 / 4 / 8 dev freed) | rungs |
|---|---|---|---|---|---|
| 7B | 23.5 s | 47 s | 0.566 s | 10.9 / 5.4 / 2.7 min | 5 |
| 13B | 41.1 s | 82 s | 1.045 s | 54 / 27 / 14 min | 3 |
| 70B | 407 s | 814 s | 9.393 s | 150 / 75 / 38 min | 2 |

Tick `dt = 30 s`. 7B/13B run **6 h** (720 ticks); 70B runs **24 h** (2880 ticks) or its shrink
threshold never elapses.

### The nine traces

| # | trace | α(t) · λ | corner it pins | **expected** |
|---|---|---|---|---|
| 1 | **steady** | α, λ constant mid-range | nothing should ever move | 0 reconfigs, 0 spill |
| 2 | **brief spike** | α 0.1→0.6 for < T_grow, back | anti-thrash | **0 reconfigs**, spill absorbs all |
| 3 | **step up** | α 0.1→0.6, held | grow path | spill → 1 grow after ~T_grow, then quiet |
| 4 | **step down** | α 0.6→0.1, held | shrink path | no immediate move; 1 bulk shrink after T_shrink |
| 5 | **slow ramp** | α 0.05→0.9 over 6 h | monotone growth | few grows, N_L non-decreasing, spill only in the lag |
| 6 | **flapping** | α square wave, period ≈ T_grow | thrash resistance | reconfigs ≪ half-periods; hysteresis suppresses |
| 7 | **diurnal** | α sinusoid over 24 h | realistic duty cycle | grow on the rise, 1 bulk shrink on the fall |
| 8 | **spillover saturation** | λ ≈ 0.85 × max, α rising | **max-spillover corner** | `spare_T` → negative, Tier A fails at once, forced grow |
| 9 | **overload** | λ > any rung's capacity | admission control | `INFEASIBLE` flagged, no thrashing |

Traces 1, 2, 6 are the ones that must produce **no** reconfiguration — they are the real test,
since any controller can grow when pushed.

### Per-tick log (the routing record you asked for)

`t · λ · α · α_smoothed · action · split · N_L · N_T · j · m · Tput_L · Tput_T ·
load_L · load_T · spare_T · excess_L · spill_tok_s · served_by_L · served_by_T ·
devices_draining · devices_kept · tokens_lost`

`action ∈ {PROVISION, HOLD, SPILL_HOLD, GROW_B, GROW_C, SHRINK_WAIT, SHRINK, INFEASIBLE}` —
so the log answers directly *when* spillover started, *how much* was rerouted to Pool T, and
*where* a split change fired.

### Per-trace summary metrics

- reconfigurations, split by tier (B = k-grow at fixed m, C = full)
- **spill fraction** = latency-class tokens served by Pool T ÷ all latency-class tokens
- **effective TTFT** = mix of Pool L TTFT (routed) and Pool T latency (spilled) — the real QoS
- time-average batch latency; total tokens lost to drain; total devices drained
- unmet demand (ticks where neither pool could take the load)

### Baselines

| baseline | expectation |
|---|---|
| **static-best** — provision once at mean α, never move | fine on 1/2/6, misses badly on 3/5/7 |
| **greedy** — re-optimise every tick, no hysteresis | best QoS, **worst tokens-lost**; the argument for hysteresis |
| **oracle** — perfect foresight, moves only when it repays | upper bound on achievable |

The headline the eval must produce: **hysteresis costs a little QoS and saves a lot of drain**,
and **70B should sit at ~0 reconfigurations on every trace** — its T_shrink is 38–150 min and it
has 2 rungs, so "provision once and spill" is the correct policy, not a failure to act.
| `results/dda32/controller_ladder.csv` | the ladder per model (the paper table) |
| `results/dda32/controller_trace_eval.csv` | per-trace outcomes vs two baselines |

Reuses `dse_metrics.py` unchanged for every metric — no formula is reimplemented.

**Baselines to beat**: (a) static best-λ_max split, never reconfigure; (b) greedy per-tick optimal
(no hysteresis) — expected to lose badly on drain cost, which is the point.

**α(t) traces**: step change · slow ramp · diurnal sinusoid · bursty (Poisson-modulated). Report
reconfig count, mean/p99 TTFT for the latency class, batch latency, and tokens lost to drain.

**Open question to settle with the eval**: whether Tier B is worth its complexity for 7B/13B, given
that Pool T re-shards regardless under F2. If the Pool-T-only drain dominates, Tier B collapses
into Tier C and the controller simplifies to spillover + gated full reconfiguration.
