# Idea A: tp/pp Pareto Frontier — Latency vs Throughput Under QoS

## Goal
Sweep all (tp, pp) configurations for each model and identify the Pareto-optimal device allocation that maximizes throughput subject to a latency SLO (e.g. 100ms/token for interactive, 1000ms/token for batch).

## Hypothesis
For interactive workloads, higher tp (more weight sharding) reduces per-token latency despite CXL overhead. For batch/throughput workloads, higher pp (pipeline depth) is better since CXL traffic scales with tp but throughput scales with pp.

## Design Space
- Llama2-7B: tp ∈ {1, 2, 4, 8}, pp = 8/tp
- Llama2-13B: tp ∈ {1, 2, 4, 5, 10, 20}, pp = 20/tp
- Llama2-70B: tp ∈ {1, 2, 4, 8, 16, 32}, pp = 32/tp

## Key Files to Modify
- `cent_simulation/run_sim.py` — add tp sweep loop
- `cent_simulation/simulation.sh` — new sweep commands
- `figure_scripts/` — new Pareto plot script

## Session Log
| Date | Action | Finding |
|---|---|---|
| 2026-04-24 | Idea created | — |

## Status
- [ ] Run baseline sweep (seqlen_gap=128)
- [ ] Plot latency vs throughput for each model
- [ ] Identify Pareto frontier per QoS constraint
- [ ] Write analysis