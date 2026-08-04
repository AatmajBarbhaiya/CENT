# Streamlit trace-replay view — plan

**Date**: 2026-07-29 · **Status**: BUILT 2026-08-04 — `app/app.py`. Kept as the design record; deviations from it are listed at the bottom.
**Depends on**: `explorer/dda_controller.py`, `explorer/dse_metrics.py`, `results/dda32/dse_full_table.csv`

Replay α(t) at a chosen fixed λ through the full controller and show, on one shared time axis,
which splits were selected, where requests actually went, and whether each reconfiguration ever
paid for itself.

---

## 1. What must be added to the controller first

The controller books a reconfiguration as a lump `tokens_lost` at the commit tick. That is enough
to *rank* policies but not to *prove* a gain. Three additions:

**(a) Drain as a transient, not a lump.** For `T_drain` seconds after a commit the draining
fraction of the fabric serves nothing. Model it as a capacity derate over those ticks:
`Tput_eff = Tput * (1 - draining/32)` while `t < t_commit + T_drain`. The ledger then falls out of
normal accounting instead of being a separate charge, and the dip is visible on the throughput panel.

**(b) Shadow (counterfactual) split.** On every commit, keep simulating the split we left behind.
Without it there is no honest "gain", only an assumption:
```
served_actual(t) = tokens/s delivered by the deployed split (derated while draining)
served_shadow(t) = tokens/s the OLD split would have delivered at the same (lambda, alpha)
```
Shadow resets at each new commit — a chained counterfactual, compared against the split actually
left behind rather than the cold start.

**(c) Cumulative ledger + break-even marker.**
```
ledger(t)  = ∫ (served_actual − served_shadow) dt      # negative during drain
breakeven  = first t after a commit where ledger(t) >= 0
```
`breakeven = never` is a legitimate and important outcome: it means the move should not have been
made, and the plot must say so rather than hide it.

---

## 2. Time horizon — 24 h for everything

**All α motions run a full 24 h**, for every model. That is the natural period of an interactive
workload, so the diurnal shape is the honest default and every other motion is judged on the same
clock. At `dt = 30 s` that is 2880 ticks — cheap.

Controller timescales all fit comfortably inside it, so break-even is actually observable:

| model | T_drain | T_grow | T_shrink (4 dev) | fits in 24 h |
|---|---|---|---|---|
| 7B | 23.5 s | 47 s | ~5 min | ~290× |
| 13B | 41.1 s | 82 s | ~27 min | ~53× |
| 70B | 407 s | 814 s | ~75 min | ~19× |

Horizon stays a slider (1–48 h) for zooming a single event, but **24 h is the default and the
number quoted in the paper**.

---

## 3. The plots — all share ONE x axis (time)

One Plotly figure, `make_subplots(shared_xaxes=True)`, a single `rangeslider` on the bottom panel
driving every other panel, plus a window preset row (15 min / 1 h / 6 h / 24 h). Reconfiguration
events are drawn as vertical rules across **every** panel, so a cause in one lines up with its
effect in another.

| # | panel | content | question it answers |
|---|---|---|---|
| 1 | **Drivers** | α(t) raw + α_smoothed (EWMA) | what the controller was reacting to |
| 2 | **Offered load** | λ(t) — its own panel, never a second y-axis | the load being applied |
| 3 | **Split timeline** | N_L / N_T step areas; k and TP as a second step pair | which split was deployed when |
| 4 | **Routing** ⭐ | stacked area: served_by_L · spilled_to_T · unmet | **where requests actually went** |
| 5 | **Pool headroom** | load_L vs Tput_L, load_T vs Tput_T, spare_T | which pool was binding |
| 6 | **Latency** | effective TTFT (what the latency class really saw) · Pool L TTFT · Pool T req_lat | QoS cost of spilling |
| 7 | **Throughput** | delivered vs offered λ, with the drain derate visible as a dip | the visible cost of a reconfiguration |
| 8 | **Token ledger** ⭐ | cumulative (actual − shadow), zero line, **break-even marker** per reconfig | **did the move pay for itself** |
| 9 | **Action ribbon** | categorical strip: HOLD · SPILL_HOLD · GROW_B · GROW_C · SHRINK_WAIT · SHRINK · INFEASIBLE | the state machine at a glance |
| 10 | **KV slack** | kv_slack_L, kv_slack_T, 1.0 danger line | did a reconfiguration move into thin KV |

Panels 4 and 8 are the two actually asked for; the rest exist to explain them.

### Alignment / scrolling rules
- one `rangeslider` (bottom panel only); every other panel sets `xaxis.matches`
- `hovermode="x unified"` so a single timestamp reads every metric at once
- reconfig vlines labelled (`2×TP8 → 16×TP1`) on panel 3 only, to avoid clutter on nine panels
- panel 9 is a heatmap strip on the same axis; panel 8 uses a signed fill (red below 0, green above)

---

## 4. Controls (sidebar)

| control | range | note |
|---|---|---|
| model | 7B / 13B / 70B | |
| **λ (fixed)** | 0 → 1.05 × max system tput | the "set lambda" |
| α motion | steady · step up · step down · brief spike · slow ramp · flapping · **diurnal** · random walk · custom | reuses the 9 eval traces; diurnal is the default |
| α mean / amplitude / period / noise | per-motion | shape knobs |
| horizon · dt | 1–48 h (**default 24 h**) · 10–120 s | |
| policy | hysteresis · greedy · static | side-by-side comparison |
| ε · kv_margin · prompt fraction | | fed to `dse_metrics` |
| **max_spill_frac** | 0–0.5, default 0.10 | ⚠️ a policy constant chosen, not measured — needs a sensitivity sweep before the paper |
| T_grow factor · EWMA half-life · min shrink devices | | controller tuning |

## 5. Headline counters (above the plots)

Reconfigurations · tokens lost to drain · tokens gained vs shadow · **net** · **break-even reached
(n of m)** · spill fraction · effective TTFT · unmet fraction · distinct rungs used.

"n of m reconfigurations reached break-even" is the honest summary — finding 12 predicts it will
be low, and that is the result, not a bug.

---

## 6. Build order

1. controller additions (a) drain transient, (b) shadow split, (c) ledger — in `dda_controller.py`
2. `explorer/app.py` sidebar + α-motion generators (import from `controller_eval.py`)
3. panels 1–4 + 8 (the core story), then 5–7, 9–10
4. the DSE explorer tabs (split table, Pareto scatter, KV strip) from the earlier plan

---

## 7. Built — what deviated from this plan

| planned | built | why |
|---|---|---|
| separate app for the frontier | **tab in `app/app.py`** | user call; one app, four tabs |
| 10 panels incl. batch-latency row | 10 panels, batch latency folded into `eff_lat` | the frontier's y axis became TOTAL throughput, so a separate batch-latency row was redundant |
| drain as a lump charge | **capacity derate for `T_drain`** + `DRAINING` action | makes the dip visible in throughput and lets the ledger fall out of ordinary accounting |
| one ledger | **two ledgers** (tokens, request-seconds) | devices only move *between* pools, so total throughput is ~conserved and a token-only ledger reports "never repaid" for almost every move — measuring the metric, not the controller |
| horizon default 12 h | **24 h for everything** | the natural period of an interactive workload; every controller timescale fits inside it |
| automatic theme | **explicit light/dark toggle** | Streamlit cannot reliably report its active theme to the script |
| — | **spike length as a multiple of T_grow** | the n/100 default was 14 min against a 47 s T_grow, so the "brief" spike tested the opposite of its claim |

Still open: `max_spill_frac` (default 0.10) is a policy constant **chosen, not measured**. It
directly sets how eagerly the controller reconfigures and needs a sensitivity sweep before the
paper, same as ε and the KV margin.
