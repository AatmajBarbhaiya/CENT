#!/usr/bin/env bash
# DDA pool sweep with EVERY model on the full 32-device fabric.
# Run from: experiments/dda_full_sweep/simulation/
# Usage: bash simulation_dda32.sh <threads> [seqlen_gap]
#
# Differs from experiments/dynamic_device_allocation/simulation/simulation_dda.sh:
#   that one pins N_total per model (7B->8, 13B->20, 70B->32, the paper's device
#   budgets). Here N_total = 32 for all three, so 7B and 13B get the same fabric
#   70B already had.
#
# Pool L: model_parallel, N_L devices, TP=N_L, PP=1       -> minimises TTFT
# Pool T: pipeline_parallel, N_T devices, TP=1, PP=layers  -> maximises throughput
# Both scored at 144/32 = 4 PCIe lanes/device (shared fabric).
#
# --fc_devices N_L restricts Pool L's TP sweep to the one degree it reads
# (PP=1/TP=N_L) instead of every factor of N_L -> ~5x fewer traces.
#
# Valid N_T (needs cpb = 32 // ceil(layers/N_T) >= min_cpb, see utils.py):
#   7B (32 layers, min 5): N_T >= 6 | 13B (40, min 8): N_T >= 10 | 70B (80, min 6): N_T >= 16
#
# Pool T throughput is a STAIRCASE in N_T -- see ../CLAUDE.md "PP staircase".
# The N_L grid straddles the step thresholds.

set -e

THREADS=${1:-8}
SEQLEN_GAP=${2:-128}
OUTDIR="../results/dda32"
mkdir -p "$OUTDIR"

run_pool_L() {
    local model=$1 N_L=$2
    local OUT="$OUTDIR/sim_${model}_poolL_${N_L}.csv"
    [ -f "$OUT" ] && { echo "skip Pool L $model N_L=$N_L (exists)"; return; }
    echo ""
    echo "=== Pool L | $model | N_L=$N_L / 32 | TP=$N_L, PP=1 ==="
    python3 run_sim_full32.py \
        --model "$model" \
        --num_devices "$N_L" \
        --total_devices 32 \
        --fc_devices "$N_L" \
        --model_parallel \
        --generate_trace --simulate_trace --process_results --update_csv \
        --run_simulation_max_workers "$THREADS" \
        --generate_trace_max_workers "$THREADS" \
        --seqlen_gap "$SEQLEN_GAP" \
        --simulation_result_path "$OUT"
}

run_pool_T() {
    local model=$1 N_T=$2
    local OUT="$OUTDIR/sim_${model}_poolT_${N_T}.csv"
    [ -f "$OUT" ] && { echo "skip Pool T $model N_T=$N_T (exists)"; return; }
    echo ""
    echo "=== Pool T | $model | N_T=$N_T / 32 | TP=1, PP=layers ==="
    python3 run_sim_full32.py \
        --model "$model" \
        --num_devices "$N_T" \
        --total_devices 32 \
        --generate_trace --simulate_trace --process_results --update_csv \
        --run_simulation_max_workers "$THREADS" \
        --generate_trace_max_workers "$THREADS" \
        --seqlen_gap "$SEQLEN_GAP" \
        --simulation_result_path "$OUT"
}

# ── Llama2-7B: N_L grid, N_T = 32 - N_L ─────────────────────────────────────
for N_L in 2 4 6 8 12 16 20 24; do
    run_pool_L Llama2-7B "$N_L"
    run_pool_T Llama2-7B $(( 32 - N_L ))
done

# ── Llama2-13B: N_L <= 22 (N_T >= 10) ───────────────────────────────────────
for N_L in 2 4 6 8 12 16 20 22; do
    run_pool_L Llama2-13B "$N_L"
    run_pool_T Llama2-13B $(( 32 - N_L ))
done

# ── Llama2-70B: N_L <= 16 (N_T >= 16) ───────────────────────────────────────
for N_L in 2 4 6 8 12 16; do
    run_pool_L Llama2-70B "$N_L"
    run_pool_T Llama2-70B $(( 32 - N_L ))
done

echo ""
echo "=== DDA N=32 sweep complete ==="
ls "$OUTDIR"/sim_*.csv | wc -l