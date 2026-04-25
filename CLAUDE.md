# CENT — CXL-Enabled GPU-Free LLM Inference Simulator

**Paper**: "PIM Is All You Need: A CXL-Enabled GPU-Free System for LLM Inference" (ASPLOS 2025)

CENT simulates LLM inference on SK Hynix AiM (Accelerator-in-Memory) GDDR6 devices connected over CXL. The full pipeline goes: Python trace generation → C++ cycle-accurate DRAM simulation → latency/power post-processing → figure generation.

---

## Architecture Overview

```
User (CLI args)
    │
    ▼
cent_simulation/run_sim.py          ← master orchestrator
    │  ├─ generate_trace()          → calls function_sim.py per config (threaded)
    │  ├─ simulate_trace()          → runs aim_simulator/build/ramulator2 (C++)
    │  ├─ process_results()         → aggregates simulator outputs
    │  └─ update_csv()              → computes latency, throughput, energy
    │
    ├─ cent_simulation/function_sim.py      ← single transformer block trace
    │       └─ TransformerBlock.py / Llama.py / GPT.py
    │              └─ aim_sim.py            ← bank/channel/DIMM model + AiM ISR emitter
    │
    ├─ cent_simulation/cxl_latency.py       ← CXL interconnect latency model
    └─ cent_simulation/cent_power_calculator.py  ← power/energy from cmd counts
```

**Hardware model**: Each CXL device = 32-channel GDDR6 with AiM PIM. Devices linked via CXL switch (PCIe bandwidth model). Transformer weights live entirely in DRAM; compute is in-memory.

---

## Core File Deep-Dives

### `aim_sim.py` — PIM Hardware Model & Trace Emitter (392 lines)

**Role**: Lowest level of the stack. Defines the physical memory hierarchy and emits `.trace` files.

**Class hierarchy**:
```
Bank          → holds DRAM_row × DRAM_column array + MAC latch + AF register
Channel(Bank) → holds num_banks Banks + 1 Global Buffer (GB, on-chip SRAM scratchpad)
DIMM(Channel) → holds num_channels Channels
PIM           → top-level device; holds pim_device dict of DIMMs
```

**Two modes** (controlled by `args.only_trace`):
- **Functional mode**: actually allocates PyTorch tensors, performs real compute for correctness checking
- **Trace-only mode**: skips tensor allocation, just writes the instruction sequence to file

**Trace format** written to `.trace` file:
```
W MEM <channel> <bank> <row>          # DRAM write (256-bit burst)
R MEM <channel> <bank> <row>          # DRAM read
AiM WR_BIAS 0 <channel_mask>          # Write to MAC bias register
AiM MAC_ABK <op_size> <mask> <row>    # MAC across all banks
AiM RD_MAC 0 <channel_mask>           # Read MAC accumulator
AiM EWMUL <op_size> <mask> <row>      # Element-wise multiply (bank0×bank1→bank2)
AiM EWADD <op_size> 0 0               # Element-wise add
AiM AF <channel_mask>                 # Activation via LUT (sigmoid/SiLU)
AiM RD_AF 0 <channel_mask>            # Read activation register
AiM COPY_BKGB <sz> <mask> <bk> <row> # Bank → Global Buffer
AiM COPY_GBBK <sz> <mask> <bk> <row> # Global Buffer → Bank
AiM SYNC                              # Barrier
AiM EOC                               # End of compute
```

**Channel mask**: bitmask of which channels receive the instruction (enables broadcast to all channels in one op).

**Timing constants** (nanoseconds, hardcoded from AiM datasheet):
- `WR_BIAS=37.5`, `MAC_ABK=49`, `RD_MAC=37.5`, `EWMUL=47`, `AF=60`, `COPY_BK_GB=42.5`, `COPY_GB_BK=45.5`
- Per-op time = constant + `op_size` (number of DRAM burst chunks)

**Key insight for modifications**: Every AiM ISR method has two variants — `foo()` (functional, computes result) and `foo_only_trace()` (trace-only, just writes to file). When adding new ops, add both. The `self.time` dict accumulates timing estimates independently of the C++ simulator.

---

### `TransformerBlock.py` — Base Class: Memory Layout & DRAM Access (1257 lines)

**Role**: Bridges the logical transformer (tensors, dimensions) to the physical DRAM layout (banks, channels, rows). Inherits from `PIM`.

**Initialization (`__init__`)**:
- Reads model config from `dic_model` dict (dim, n_heads, n_kv_heads, TP_param, etc.)
- Resolves parallelism mode: sets `self.channels_per_block`, `self.FC_total_banks`, `self.intra_device_attention`
- Loads all weight/activation tensors from `dic_model` as float tensors
- Key flags: `model_parallel`, `pipeline_parallel`, `FC_devices` (= tp factor), `TP_param`

**Memory mapping (`memory_mapping()`)** — must be called before any trace generation:
- Computes `dic_size`, `dic_row`, `dic_shape` for every tensor (x, wq, wk, wv, wo, w1, w2, w3, cache_k, cache_v, scores, etc.)
- Assigns contiguous row index offsets (e.g. `wq_row_index`, `cache_k_row_index`) so tensors don't overlap in DRAM
- Prints a layout summary and reports task-level parallelism (how many inference tasks can co-reside)

**Weight layout (critical for TP)**:
- In pipeline-parallel: `FC_total_banks = total_banks`, weights fit in one device's channels
- In tensor-parallel: `FC_total_banks = total_banks × FC_devices`, rows are split across `FC_devices` devices
- Weight matrix row `i` maps to bank `i % FC_total_banks`, row offset `i // FC_total_banks`

**KV-cache layout** — two modes:
- `intra_device_attention=True` (default for small seqlen): each channel holds heads[channel×N : (channel+1)×N], dim distributed across 16 banks, seq dimension along rows. Efficient for short context.
- `intra_device_attention=False` (long context): banks distributed across devices proportional to seqlen. `banks_per_head` scales with `FC_total_banks / n_kv_heads`. At seqlen=32k: 2048 banks/head, spanning 4 devices.

**Key helpers used by subclasses**:
- `store_to_DRAM_multi_channel(data, row_index, mode, op_trace)` — physical write with trace emission
- `load_from_DRAM_multi_channel(shape, row_index, mode, size, op_trace)` — physical read
- `Vector_Matrix_Mul_weight_pim(x, row_index, in_dim, out_dim, total_banks, op_trace, timing)` — full GEMV via AiM MAC
- `Vector_Matrix_Mul_weight_pim_only_trace(...)` — trace-only GEMV (no actual compute)
- `Vector_Matrix_Mul_score_pim_only_trace(...)` — Q×K^T attention score GEMV
- `Vector_Matrix_Mul_output_pim_only_trace(...)` — score×V output GEMV
- `store_for_neighbor_bank_input_only_trace(...)` — stores input vector across neighboring banks for MAC_ABK
- `store_for_EWMUL_input_only_trace(...)` — stores inputs for element-wise ops

**What to modify here** when adding new parallelism strategies: update `bank_index()`, `store_to_DRAM_multi_channel()`, and `memory_mapping()` to reflect new layout assumptions.

---

### `Llama.py` — LLaMA-Specific AiM Execution (1236 lines)

**Role**: Implements one complete transformer block for LLaMA-2 on AiM hardware. Two execution paths: functional (verifies correctness) and trace-only (fast, for simulation).

**Class**: `TransformerBlockLlama(TransformerBlock)`

**`self_attention()`** — reference PyTorch implementation for correctness comparison:
- RMSNorm → Q/K/V linear → rotary embedding → KV cache update → Q×K^T scores → softmax → score×V → Wo projection → residual add
- Uses `compare()` at each step to validate intermediate values

**`self_attention_aim()`** — AiM-mapped attention (functional + trace):
Step-by-step AiM execution:
1. **RMSNorm**: `MAC_BK_BK` (x² sum across neighbor banks) → CXL reduction → `EWMUL` (scale by 1/sqrt) → `COPY_BK_GB / COPY_GB_BK` (distribute scalar) → second `EWMUL` (apply weight) → result in `SANorm_row_index`
2. **Q/K/V GEMV**: `Vector_Matrix_Mul_weight_pim()` with `MAC_BK_GB` (input x in GB, weight rows in banks → partial sums in MAC latch) → `RD_MAC`
3. **Rotary embedding**: `EWMUL` on xq/xk with cos/sin vectors (handled via CXL port in trace-only)
4. **KV cache store**: `WR_SBK` for xk (into `cache_k_row_index`), `WR_ABK` for xv (into `cache_v_row_index`)
5. **Q×K score GEMV**: `Vector_Matrix_Mul_score_pim_only_trace()` — xq broadcast across channels, K cache as weight rows
6. **Softmax**: stored to/from DRAM rows for the scores (`store_for_score_only_trace`, `load_for_score_only_trace`)
7. **Score×V output GEMV**: `Vector_Matrix_Mul_output_pim_only_trace()`
8. **Wo projection**: `Vector_Matrix_Mul_weight_pim()`
9. **Residual add**: `EWADD`

**`FFN_aim()`** — AiM-mapped feed-forward:
1. **FFNNorm**: same RMSNorm pattern as attention norm
2. **W1/W3 GEMV + SiLU gate**: `Vector_Matrix_Mul_weight_af_pim()` → `AF` (sigmoid LUT) → `EWMUL` (SiLU = x × sigmoid(x))
3. **W2 GEMV**: `Vector_Matrix_Mul_weight_pim()` on gated activation

**`trace_only()`** — fast path, delegates to `GPT.trace_only()` effectively (LLaMA uses GPT.py's trace_only logic via inheritance chain)

**What to modify here** when adding new ops (e.g. new norm type, new activation): add the AiM instruction sequence in `self_attention_aim()` or `FFN_aim()`, add a corresponding `*_only_trace()` helper in `TransformerBlock.py`, and update timing in `aim_sim.py`.

---

### `GPT.py` — GPT/Trace-Only Transformer Block (510 lines)

**Role**: Provides the `trace_only()`, `trace_only_embedding()`, and `trace_only_FC()` methods used by the production simulation path. Despite the name, this is the fast simulation path for all models (LLaMA uses this via `function_sim.py` when `--only-trace` is passed).

**Class**: `TransformerBlockGPT(TransformerBlock)`

**Three methods**:

`trace_only()` — full transformer block trace (attention + FFN):
- Calls `store_for_neighbor_bank_input_only_trace` + `MAC_ABK_only_trace` + `RD_MAC_only_trace` for RMSNorm
- Calls `Vector_Matrix_Mul_weight_pim_only_trace` for Q/K/V/O GEMVs
- Stores xk via `W_MEM_only_trace`, stores xv via `WR_ABK_only_trace` (two code paths: `intra_device_attention` vs cross-device for long context)
- Calls `Vector_Matrix_Mul_score_pim_only_trace` for Q×K, `Vector_Matrix_Mul_output_pim_only_trace` for score×V
- Calls `Vector_Matrix_Mul_weight_af_pim_only_trace` for w1 (FFN gate with activation) and `Vector_Matrix_Mul_weight_pim_only_trace` for w2
- Long-context V-cache store has 3 branches based on `banks_per_head` vs `num_banks`: intra-device (small seqlen), single-device-per-head (medium), multi-device-per-head (large seqlen like 32k)

`trace_only_embedding()` — embedding layer trace:
- One GEMV for input embedding lookup (vocab_size → dim)
- RMSNorm
- One GEMV for output projection (dim → vocab_size)

`trace_only_FC()` — FC-only trace (used for model_parallel_FC traces):
- Q/K/V/O GEMVs only, no attention, no norm, no FFN activation — just the weight GEMVs
- Used to isolate FC latency in tensor-parallel sensitivity analysis

`memory_mapping()` — overrides base class to add vocab_size fields and print layout:
- Maps all tensors to row indices (same as TransformerBlock but with explicit assertions)
- Computes `task_level_parallelism` = how many inference tasks fit simultaneously

**Key structural note**: `trace_only()` has a large `if False:` block (lines 156–194) for softmax — this was the old explicit softmax trace (scale → exp → sum → normalize). It's currently dead code; softmax is handled implicitly by the simulator via score load/store.

---

### `function_sim.py` — Entry Point for One Transformer Block (76 lines)

**Role**: CLI entry point called by `run_sim.py` in a subprocess/thread for each (model, seqlen, parallelism config) combination. Instantiates the correct TransformerBlock subclass and runs either trace generation or functional verification.

**Flow**:
1. Calls `get_args()` to parse CLI flags
2. If `--filename` given: loads `dic_model` from a `.pt` file (for functional verification with real weights)
3. Otherwise: constructs a zero-initialized `dic_model` dict from CLI dimensions (for trace-only runs — no real weights needed)
4. Key `dic_model` fields: `dim`, `n_heads`, `TP_param`, `wq/wk/wv/wo/w1/w2/w3` (weight tensors), `cache_k/cache_v` (KV caches), `x` (input activation), `start_pos` (= seqlen-1, current decode step)
5. Instantiates `TransformerBlockLlama` (for `--Llama-GQA` or `--Llama`) or `TransformerBlockGPT` (default)
6. Calls `TB.memory_mapping()` — must always run first to set row indices
7. Dispatch:
   - `--only-trace --embedding` → `TB.trace_only_embedding()` + `TB.finish()`
   - `--only-trace --only-FC` → `TB.trace_only_FC()` + `TB.finish()`
   - `--only-trace` → `TB.trace_only()` + `TB.finish()`
   - `--pim-memory-mapping` → `TB.self_attention_aim()` + `TB.FFN_aim()` (functional AiM verification)
   - (default) → `TB.self_attention()` + `TB.FFN()` (reference PyTorch)

**What to add here** when introducing new trace variants (e.g. a new parallelism mode or a new op set): add a new `elif args.new_flag:` branch dispatching to a new method on the TransformerBlock subclass.

---

## Key Files (Summary)

| File | Role |
|---|---|
| `cent_simulation/run_sim.py` | Top-level orchestrator — CLI parsing, threading, CSV output |
| `cent_simulation/function_sim.py` | Entry point for one transformer block trace |
| `cent_simulation/TransformerBlock.py` | Base class: maps tensors → DRAM banks, emits AiM trace ops |
| `cent_simulation/Llama.py` | LLaMA-specific attention + FFN → AiM ISR mappings |
| `cent_simulation/GPT.py` | GPT-style trace-only flows (production simulation path) |
| `cent_simulation/aim_sim.py` | Bank/Channel/DIMM/PIM class hierarchy; writes `.trace` files |
| `cent_simulation/cxl_latency.py` | Broadcast/gather latency for tensor-parallel over CXL |
| `cent_simulation/cent_power_calculator.py` | Parses simulator logs → power & energy estimates |
| `cent_simulation/utils.py` | Model configs, CLI arg parser |
| `cent_simulation/scaling_study_various_DP.py` | Data-parallel scalability studies |
| `aim_simulator/build/ramulator2` | **Compiled C++ simulator** — must be built before running |
| `cost_model/cost_model.py` | Die/packaging/NRE cost calculations |
| `cost_model/supply_chain_model.py` | Fab cost models |
| `data/GPU_*.csv` | GPU baseline latency, energy, power, throughput |
| `figure_scripts/figure_*.py` | One script per paper figure (12–15) |
| `figure_source_data/figure_*.csv` | Pre-computed data for each figure |
| `figures/figure_*.pdf` | Generated PDF figures |

---

## Supported Models

| Model | Layers | Heads | Embedding | FFN size | GQA factor | Default devices |
|---|---|---|---|---|---|---|
| Llama2-7B | 32 | 32 | 4096 | 11008 | 1 | 8 |
| Llama2-13B | 40 | 40 | 5120 | 13824 | 1 | 20 |
| Llama2-70B | 80 | 64 | 8192 | 28672 | 8 | 32 |

All use 32 channels per CXL device.

---

## Parallelism Modes

**Pipeline Parallelism (`--pipeline-parallel`)**
- Transformer blocks distributed sequentially across devices
- `pp = num_layers`, `tp = 1`
- Each device runs full-width attention/FFN; no weight sharding

**Tensor / Model Parallelism (`--model-parallel`)**
- Weight matrices (W_Q/K/V/O, W_1/W_2/W_3) sharded across `FC_devices`
- `tp = FC_devices`, `pp = num_devices / FC_devices`
- CXL latency incurred per layer for broadcast/gather

The full design space is `tp ∈ {1 … num_devices}`, with `pp = num_devices / tp`. Higher tp → less per-device memory, more CXL traffic.

---

## AiM Instruction Set (trace ops)

| Op | Meaning |
|---|---|
| `MAC_ABK / MAC_BK_GB` | Parallel MAC across all banks / bank→GlobalBuffer |
| `EWMUL / EWADD` | Element-wise multiply / add (softmax scaling, residual) |
| `AF` | Activation function via LUT (SiLU/sigmoid) |
| `WR_GB / WR_BIAS` | Write to Global Buffer / MAC bias register |
| `RD_MAC / RD_AF` | Read MAC accumulators / activation output |
| `COPY_BKGB / COPY_GBBK` | Bank ↔ Global Buffer transfers |
| `SYNC / EOC` | Synchronization / end-of-compute |
| `W MEM / R MEM` | Conventional DRAM write / read (256-bit rows) |

---

## Simulation Results Schema

**`cent_simulation/simulation_results.csv`** — per-sequence-length raw results:
```
Model, Device number, Pipeline parallelism, Tensor parallelism,
Channels per device, Channels per block, Sequence length,
PIM latency, CXL latency, Acc latency, TransformerBlock latency, Embedding latency,
Token latency (ms), Throughput (tokens/s), Token energy (mJ), Total power (W), Device utilization
```

**`cent_simulation/processed_results.csv`** — aggregated by phase (prefill / decoding / end2end):
```
Model, Device number, Seqlen, Pipeline parallelism, Tensor parallelism, Phase,
Total Latency (s), Throughput (tokens/s), Energy per Token (mJ), Total power (W)
```

### Key Results (from paper figures)
- **Latency speedup vs GPU**: 6.3× (7B), 4.7× (13B), 3.2× (70B)
- **TCO-normalized throughput**: 6.7× (7B), 7.4× (13B), 2.8× (70B)
- **Throughput vs GPU**: 0.33×–2.97× depending on model and batch

---

## Build & Run

```bash
# 1. Build C++ simulator (one-time)
cd aim_simulator && mkdir build && cd build && cmake .. && make -j4

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run simulation (quick: seqlen_gap=128, ~2-3h; full: seqlen_gap=1, ~24h)
cd cent_simulation
bash simulation.sh <num_threads> <seqlen_gap>   # e.g. bash simulation.sh 8 128

# 4. Aggregate results
bash process_results.sh

# 5. Generate figures
cd .. && bash generate_figures.sh
```

**Resource estimates (full run, seqlen_gap=1):**
- Time: ~24h on 8 threads
- RAM: ~8 GB
- Disk: ~100 GB (traces)

---

## Trace Directory Layout

```
trace/32_channels_per_device/
├── pipeline_parallel/          # pp mode traces
├── model_parallel/             # tp mode traces (full block)
├── model_parallel_embedding/   # tp mode embedding traces
└── model_parallel_FC/          # tp mode FC-only traces
```

---

## Current Explorations

- **Idea A**: Sensitivity analysis of device allocation for latency-vs-throughput oriented tasks — sweep tp/pp across the Pareto frontier and identify the optimal operating point for each model under QoS constraints.
- **Idea B**: (add your next idea here)

---

## Glossary

| Term | Meaning |
|---|---|
| AiM | Accelerator-in-Memory (SK Hynix PIM technology on GDDR6) |
| CXL | Compute Express Link (PCIe-based coherent interconnect) |
| PP | Pipeline Parallelism — blocks distributed across devices |
| TP | Tensor Parallelism — weights sharded across devices |
| GEMV | General Matrix-Vector multiply (dominant op in single-token decoding) |
| GQA | Grouped Query Attention (used in Llama2-70B, factor=8) |
| GB | Global Buffer — on-chip SRAM scratchpad in AiM device |
| ISR | Instruction Set Register (AiM's compute instruction format) |
