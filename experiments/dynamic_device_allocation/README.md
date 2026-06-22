# Dynamic Device Allocation (DDA)

Partitions CXL-PIM devices into two concurrent pools to serve mixed interactive + batch LLM inference workloads simultaneously, achieving up to **5.9× throughput** over the best single static (PP, TP) configuration.

- **Pool L** (`N_L` devices): TP=N_L, PP=1 — minimises TTFT for short/interactive requests
- **Pool T** (`N_T` devices): TP=1, PP=num_layers — maximises throughput for long/batch requests

Three routing strategies: S1 seqlen heuristic → S2 SLO-tagged → S3 Pareto-adaptive.

---

## Folder Structure

```
experiments/dynamic_device_allocation/
├── README.md                    ← this file
├── run_figures.sh               ← regenerate all 6 paper figures
│
├── simulation/
│   ├── simulation_dda.sh        ← run all Pool L + Pool T sweeps (calls process_dda.sh on finish)
│   ├── process_dda.sh           ← build combined_metrics.csv from sim CSVs
│   ├── clean_dda.sh             ← wipe traces and/or results for a fresh run
│   ├── run_sim_dda.py           ← modified run_sim.py with per-pool trace paths + PCIe lanes
│   └── combine_pools.py         ← aggregates pool CSVs → combined_metrics.csv
│
├── figure_scripts/              ← one script per paper figure (run from this folder)
│   ├── figure_motivation_dda.py
│   ├── figure_dda_strategy1_shortseq.py
│   ├── figure_dda_strategy1_longseq.py
│   ├── figure_dda_strategy2_tol.py
│   ├── figure_dda_strategy2_alpha.py
│   └── figure_dda_strategy3_v2.py
│
├── figures/                     ← output PDFs + PPTX
├── figure_source_data/          ← intermediate CSVs written by figure scripts
├── results/                     ← simulation CSVs (gitignored: sim_*.csv, combined_metrics.csv)
│   ├── sim_{model}_poolL_{N}.csv
│   ├── sim_{model}_poolT_{N}.csv
│   └── combined_metrics.csv
└── traces/                      ← generated traces (gitignored, ~2.4 GB)
```

---

## Quickstart: Full Run from Scratch

```bash
cd experiments/dynamic_device_allocation/simulation

# 1. Clean old artifacts (if any)
bash clean_dda.sh

# 2. Run sub-pool simulations (~4h on 8 threads, seqlen_gap=128)
bash simulation_dda.sh 8 128
# Automatically runs process_dda.sh at the end → builds combined_metrics.csv

# 3. Generate all paper figures
cd .. && bash run_figures.sh
```

Figures land in `figures/`.

---

## Script Reference

### `simulation/simulation_dda.sh`

Sweeps all valid (N_L, N_T) splits for all three models.

```bash
# Run from: experiments/dynamic_device_allocation/simulation/
bash simulation_dda.sh <threads> [seqlen_gap]

# Examples:
bash simulation_dda.sh 8 128    # fast (~4h), seqlen_gap=128
bash simulation_dda.sh 4 1      # full run (~24h), seqlen_gap=1
```

**What it does:**
- For each valid split, runs Pool L (`--model_parallel`) and Pool T (`--pipeline_parallel`) separately via `run_sim_dda.py`
- Writes one CSV per pool per split: `results/sim_{model}_pool{L|T}_{N}.csv`
- Auto-calls `process_dda.sh` on completion

**Valid splits per model:**

| Model | N_total | Valid N_L | Corresponding N_T |
|-------|---------|-----------|-------------------|
| Llama2-7B  | 8  | {2}              | {6}              |
| Llama2-13B | 20 | {2, 4, 6, 8, 10} | {18,16,14,12,10} |
| Llama2-70B | 32 | {4, 8, 12, 16}   | {28,24,20,16}    |

Splits constrained by `minimal_channel_per_block` in `cent_simulation/utils.py` (Pool T needs enough channels/block for KV-cache).

---

### `simulation/process_dda.sh`

Builds `combined_metrics.csv` from the per-pool CSVs. Run this after simulation if you need to re-run post-processing without re-simulating.

```bash
# Run from: experiments/dynamic_device_allocation/simulation/
bash process_dda.sh
```

Wraps `combine_pools.py`, which:
- Reads all `results/sim_*.csv`
- Extracts Pool L: PP=1, TP=N_L config → TTFT_short, Tput_L
- Extracts Pool T: PP=num_layers, TP=1 config → Tput_T
- Computes `system_tput = Tput_L + Tput_T`
- Writes `results/combined_metrics.csv`

---

### `simulation/clean_dda.sh`

Removes simulation artifacts to prepare for a fresh run.

```bash
# Run from: experiments/dynamic_device_allocation/simulation/
bash clean_dda.sh              # remove both traces + results (default)
bash clean_dda.sh --traces     # remove traces/ only (~2.4 GB)
bash clean_dda.sh --results    # remove results/sim_*.csv + combined_metrics.csv only
bash clean_dda.sh --all        # same as default
```

Does **not** touch `figures/` or `figure_source_data/`.

---

### `run_figures.sh`

Regenerates all six paper figures from `results/combined_metrics.csv` and `../../cent_simulation/simulation_results.csv`.

```bash
# Run from: experiments/dynamic_device_allocation/
bash run_figures.sh
```

Runs these scripts in order:

| Script | Output | What it shows |
|--------|--------|---------------|
| `figure_motivation_dda.py` | `figures/figure_motivation_dda.pdf` | No static config meets both SLOs; 9.7× latency spread at T=512 |
| `figure_dda_strategy1_shortseq.py` | `figures/figure_dda_strategy1_shortseq.pdf` | S1 short-req Pareto scatter + 1.3–4.9× throughput gain bars |
| `figure_dda_strategy1_longseq.py` | `figures/figure_dda_strategy1_longseq.pdf` | Adaptive tput+TTFT vs seqlen; short/long-split bar comparison |
| `figure_dda_strategy2_tol.py` | `figures/figure_dda_strategy2_tol.pdf` | S2 ±5% SLO tolerance; S1 vs S2 Pareto + bars (5.9× 70B) |
| `figure_dda_strategy2_alpha.py` | `figures/figure_dda_strategy2_alpha.pdf` | Optimal N_L vs α; all transitions before α=0.20 |
| `figure_dda_strategy3_v2.py` | `figures/figure_dda_strategy3_v2.pdf` | S3 Pareto-adaptive: 4-panel time-series; α ramp 0.05→0.35→0.05 |

**Prerequisite:** `results/combined_metrics.csv` must exist (run `process_dda.sh` first).

---

## Common Workflows

### Re-run figures only (no new simulation)
```bash
cd experiments/dynamic_device_allocation
bash run_figures.sh
```

### Re-run combine + figures (traces already exist)
```bash
cd experiments/dynamic_device_allocation/simulation
bash clean_dda.sh --results
bash process_dda.sh
cd .. && bash run_figures.sh
```

### Re-run everything from scratch
```bash
cd experiments/dynamic_device_allocation/simulation
bash clean_dda.sh
bash simulation_dda.sh 8 128
cd .. && bash run_figures.sh
```

### Run one model only (manual)
```bash
cd experiments/dynamic_device_allocation/simulation

# Example: 13B N_L=6, N_T=14
python3 run_sim_dda.py --model Llama2-13B --num_devices 6 --total_devices 20 \
    --model_parallel --generate_trace --simulate_trace --process_results --update_csv \
    --run_simulation_max_workers 8 --generate_trace_max_workers 8 --seqlen_gap 128 \
    --simulation_result_path ../results/sim_Llama2-13B_poolL_6.csv

python3 run_sim_dda.py --model Llama2-13B --num_devices 14 --total_devices 20 \
    --generate_trace --simulate_trace --process_results --update_csv \
    --run_simulation_max_workers 8 --generate_trace_max_workers 8 --seqlen_gap 128 \
    --simulation_result_path ../results/sim_Llama2-13B_poolT_14.csv

bash process_dda.sh
```

---

## Key Results Summary

| Model | Best Static | S1 gain | S2 gain | S2 TTFT | S2 SLO |
|-------|------------|---------|---------|---------|--------|
| Llama2-7B  | 1822 tok/s | 1.3× | 1.3× | 0.82s | 1s ✓ |
| Llama2-13B | 780 tok/s  | 3.9× | 3.9× | 1.00s | ~1s ✓ |
| Llama2-70B | 216 tok/s  | 4.9× | **5.9×** | 4.38s | 5s ✓ |

S3 (13B only): achieves S1 throughput (3020 tok/s) at low α and S2 SLO compliance (TTFT=0.882s) at high α, switching at Pareto crossover α=0.143.

---

## Dependencies

All scripts share baselines from `../../cent_simulation/simulation_results.csv` (must exist — generated by the main CENT simulation). The `run_sim_dda.py` requires the C++ simulator built at `../../aim_simulator/build/ramulator2`.

See the root [CENT README](../../README.md) for build instructions.