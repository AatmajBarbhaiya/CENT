# DDA explorer (Streamlit)

```bash
pip install streamlit plotly
cd experiments/dda_full_sweep/app
streamlit run app.py
```

## What it is

Replays α(t) at a fixed λ through the full DDA controller and shows, on **one shared time axis**,
which splits were selected, where requests actually went, and whether each reconfiguration ever
paid for itself.

Every performance number comes from `../explorer/dse_metrics.py` — the same module the CSV writers
and paper figures import — so the app and the paper data cannot drift apart. The α shapes come from
`../explorer/traces.py`, shared with `../simulation/controller_eval.py`, so the interactive view and
the asserted trace suite use identical waveforms.

## Layout

| file | role |
|---|---|
| `app.py` | UI, sidebar controls, the 10-panel figure |
| `../explorer/dse_metrics.py` | all derived metrics (single source of truth) |
| `../explorer/dda_controller.py` | ladder, provisioning, Tier A/B/C, hysteresis, ledgers |
| `../explorer/traces.py` | α(t) shapes, shared with the eval suite |
| `../results/dda32/dse_full_table.csv` | the 154-split design space |

## Tabs

**Throughput vs latency** — the headline comparison, with α and λ sliders. x = effective
latency-class latency (spill included), y = TOTAL fabric throughput (Pool L + Pool T). Every
marker hovers with its full configuration; hollow means the deployment cannot carry that (α, λ)
even with spill, so demand queues. Below it: the matched-latency-bar table, and an α sweep from
0 to 1 in 0.1 steps at the current λ.

⚠️ In that sweep, `tput ×` and `latency ×` compare the two *best-throughput* configs — unmatched,
so a gain there can come from merely sitting at a different latency. **`MATCHED tput ×` is the
defensible number**: the largest total-throughput advantage DDA holds at an *identical* latency bar.

**Trace replay** — 10 stacked panels sharing one x axis (hours), one range slider on the bottom
panel driving all of them, `hovermode="x unified"` so a single timestamp reads every metric.
Reconfiguration events are vertical rules across every panel.

1. α raw + EWMA · 2. λ offered · 3. N_L / N_T · 4. Pool L factorization (k, TP) ·
5. **Routing** — served by Pool L / spilled to Pool T / unmet · 6. Pool headroom ·
7. Latency (log) · 8. Throughput delivered vs offered · 9. **Ledger** · 10. KV slack

**Ladder** — the rungs this model may deploy (7B 21, 13B 16, 70B 4), which were used in the current
trace, and the k↔TP trade plotted at constant device count.

**All splits** — the full 154-row DSE table, including `saturated` rows kept as evidence.

## Reading the ledger (panel 9)

Two currencies, because the two directions pay differently:

- **tokens** — `∫(served_actual − served_shadow)`. Only turns positive when a move relieves
  genuinely *unmet* demand. Devices merely move between pools, so total throughput is roughly
  conserved and most moves never repay here.
- **request-seconds** — `∫(latency_shadow − latency_actual) × served`. This is where a shrink
  (devices back to Pool T for batch latency) and most grows (traffic off the slow spill path)
  actually pay.

Break-even fires on **either**, and the events table records which one paid. `NEVER` is a real
result, not a bug — finding 12 predicts most moves are marginal.

The shadow is **chained**: it resets at each commit, so every move is judged against the split it
replaced rather than the cold start.

## Caveat

`max_spill_frac` (default 0.10) is a policy constant **chosen, not measured**. It directly sets how
eagerly the controller reconfigures. It needs a sensitivity sweep before the paper, same as ε and
the KV margin — the slider is there to make that easy.

## Why the sliders matter

On total throughput the plotted points are pure deployment properties, so α and λ would do nothing
on their own. What the workload actually controls is **spill**: an undersized Pool L is not a
failure — its excess is served by Pool T at Pool T's much slower latency. So the DDA points
*slide right* as α·λ grows. Only demand neither pool can take actually queues.

This is why the static PDFs in `figures/` are single frames: they show one (α, λ). The exploration
lives here.

## α EWMA vs raw α

`α` is instantaneous; `α EWMA` is the smoothed estimate the controller actually acts on:

    α_s ← α_s + w·(α − α_s),    w = 1 − exp(−ln2 · Δt / t½)

The lag is deliberate. Reconfiguration costs seconds-to-minutes of drain, so an estimate that moves
faster than the actuator can respond is how thrash starts. The visible gap between the two lines is
the controller's blind spot — a spike living entirely inside it is one the controller is *right*
to ignore.
