#!/usr/bin/env bash
# DDA sub-pool simulation sweep.
# Run from: experiments/dynamic_device_allocation/simulation/
# Usage: bash simulation_dda.sh <threads> [seqlen_gap]
#
# Generates Pool L (model_parallel, N_L devices) and Pool T (pipeline_parallel, N_T devices)
# for each valid split per model. Results → ../results/sim_{model}_pool{L|T}_{N}.csv
#
# PCIe lanes: proportional allocation.
#   Pool L gets 144 * N_L / N_total lanes (total), same per-device rate as full system.
#   Pool T gets 144 * N_T / N_total lanes (total).
#
# Valid splits (limited by minimal_channel_per_block constraint):
#   7B  (N=8):  N_L ∈ {2}                → N_T ∈ {6}
#   13B (N=20): N_L ∈ {2,4,6,8,10}       → N_T ∈ {18,16,14,12,10}
#   70B (N=32): N_L ∈ {4,8,12,16}        → N_T ∈ {28,24,20,16}

set -e

THREADS=${1:-4}
SEQLEN_GAP=${2:-128}

# --total_devices tells run_sim_dda.py the full system size so it computes
# PCIe_lanes_per_device = 144 // total_devices (same per-device rate as baseline).
# No manual PCIE math needed here.

run_pool_L() {
    local model=$1 N_L=$2 N_TOTAL=$3
    local OUT="../results/sim_${model}_poolL_${N_L}.csv"
    echo ""
    echo "=== Pool L | model=$model | N_L=$N_L / $N_TOTAL ==="
    python3 run_sim_dda.py \
        --model "$model" \
        --num_devices "$N_L" \
        --total_devices "$N_TOTAL" \
        --model_parallel \
        --generate_trace \
        --simulate_trace \
        --process_results \
        --update_csv \
        --run_simulation_max_workers "$THREADS" \
        --generate_trace_max_workers "$THREADS" \
        --seqlen_gap "$SEQLEN_GAP" \
        --simulation_result_path "$OUT"
}

run_pool_T() {
    local model=$1 N_T=$2 N_TOTAL=$3
    local OUT="../results/sim_${model}_poolT_${N_T}.csv"
    echo ""
    echo "=== Pool T | model=$model | N_T=$N_T / $N_TOTAL ==="
    python3 run_sim_dda.py \
        --model "$model" \
        --num_devices "$N_T" \
        --total_devices "$N_TOTAL" \
        --generate_trace \
        --simulate_trace \
        --process_results \
        --update_csv \
        --run_simulation_max_workers "$THREADS" \
        --generate_trace_max_workers "$THREADS" \
        --seqlen_gap "$SEQLEN_GAP" \
        --simulation_result_path "$OUT"
}

# ── Llama2-7B (N=8): only valid split N_L=2, N_T=6 ──────────────────────────
run_pool_L Llama2-7B 2 8
run_pool_T Llama2-7B 6 8

# ── Llama2-13B (N=20): N_L ∈ {2,4,6,8,10} ───────────────────────────────────
for N_L in 2 4 6 8 10; do
    N_T=$(( 20 - N_L ))
    run_pool_L Llama2-13B "$N_L" 20
    run_pool_T Llama2-13B "$N_T" 20
done

# ── Llama2-70B (N=32): N_L ∈ {4,8,12,16} ────────────────────────────────────
for N_L in 4 8 12 16; do
    N_T=$(( 32 - N_L ))
    run_pool_L Llama2-70B "$N_L" 32
    run_pool_T Llama2-70B "$N_T" 32
done

echo ""
echo "=== DDA simulation sweep complete ==="
echo "Results in: experiments/dynamic_device_allocation/results/"
ls ../results/sim_*.csv 2>/dev/null || true

echo ""
echo "=== Post-processing: building combined_metrics.csv ==="
bash "$(dirname "$0")/process_dda.sh"
