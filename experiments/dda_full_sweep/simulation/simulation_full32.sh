#!/usr/bin/env bash
# Full-fabric deployment sweep: what do 7B / 13B gain from all 32 devices?
# Run from: experiments/dda_full_sweep/simulation/
# Usage: bash simulation_full32.sh <threads> [seqlen_gap]
#
# Question: the paper pins 7B->8 devices and 13B->20 devices. On a 32-device
# fabric, is it better to (a) scale ONE instance across all 32, or (b) run
# several smaller instances data-parallel (7B: 4x8, 13B: 2x16)?
#
# Each (model, N, mode) is simulated once, then scored under TWO PCIe budgets:
#   lanes32 : N devices carved out of a shared 32-device fabric -> 144/32 = 4 lanes/dev
#   lanesN  : N devices as a standalone box -> 144/N lanes/dev (published baseline)
# The PCIe budget only feeds the CXL latency model, not the traces, so the
# second scoring pass re-reads the same logs and is nearly free.
#
# PP device utilisation is quantised by blocks_per_device = ceil(layers / N):
#   7B  (32 layers): N=8 -> 4 blk/dev, cpb=8  | N=16 -> 2 blk/dev, cpb=16 | N=32 -> 1 blk/dev, cpb=32
#   13B (40 layers): N=16 -> 3 blk/dev, cpb=10 (only 14/16 used)
#                    N=20 -> 2 blk/dev, cpb=16 (paper config)
#                    N=32 -> 2 blk/dev, cpb=16, still only 20 devices used <- 12 idle
# MP runs sweep every factor of N, so one MP run per N covers all TP splits.

set -e

THREADS=${1:-8}
SEQLEN_GAP=${2:-128}
OUTDIR="../results"
mkdir -p "$OUTDIR"

# $1=model  $2=N  $3=pp|mp
run_cfg() {
    local model=$1 N=$2 mode=$3
    local mp_flag=""
    [ "$mode" = "mp" ] && mp_flag="--model_parallel"

    echo ""
    echo "=== $model | N=$N | mode=$mode | scoring at 4 lanes/dev (shared 32-dev fabric) ==="
    python3 run_sim_full32.py \
        --model "$model" \
        --num_devices "$N" \
        --total_devices 32 \
        $mp_flag \
        --generate_trace \
        --simulate_trace \
        --process_results \
        --update_csv \
        --run_simulation_max_workers "$THREADS" \
        --generate_trace_max_workers "$THREADS" \
        --seqlen_gap "$SEQLEN_GAP" \
        --simulation_result_path "$OUTDIR/sim_${model}_${mode}_${N}dev_lanes32.csv"

    echo "=== $model | N=$N | mode=$mode | re-scoring at $((144 / N)) lanes/dev (standalone box) ==="
    python3 run_sim_full32.py \
        --model "$model" \
        --num_devices "$N" \
        --total_devices "$N" \
        $mp_flag \
        --update_csv \
        --seqlen_gap "$SEQLEN_GAP" \
        --simulation_result_path "$OUTDIR/sim_${model}_${mode}_${N}dev_lanesN.csv"
}

# ── Llama2-7B: 8 (paper) / 16 / 32 devices ──────────────────────────────────
for N in 8 16 32; do
    run_cfg Llama2-7B "$N" pp
    run_cfg Llama2-7B "$N" mp
done

# ── Llama2-13B: 16 / 20 (paper) / 32 devices ────────────────────────────────
for N in 16 20 32; do
    run_cfg Llama2-13B "$N" pp
    run_cfg Llama2-13B "$N" mp
done

echo ""
echo "=== full-32 sweep complete ==="
ls "$OUTDIR"/sim_*.csv