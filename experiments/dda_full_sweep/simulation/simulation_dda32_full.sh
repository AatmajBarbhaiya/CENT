#!/usr/bin/env bash
# Comprehensive DDA pool sweep — FULL N_L/N_T spectrum for ALL 3 models.
# Extends the sparse simulation_dda32.sh grid to a dense step-2 grid so the
# realistic split analysis sees the ENTIRE performance spectrum, whether or not
# a split beats the baseline.
#
# Run from: experiments/dda_full_sweep/simulation/
# Usage: bash simulation_dda32_full.sh <threads> [seqlen_gap]
# Skips any (model,pool,N) whose CSV already exists -> resumes/extends safely.
# NOT set -e: an illegal config (min_cpb wall) is skipped, not fatal.
#
# HETEROGENEOUS PACKING NOTE: a single CENT run maps every device uniformly at
# one cpb, so packing (finding 7) is NOT produced here — this only COLLECTS the
# primitives (CENT base latency + Tput_T per N_T, the cpb ladder, Pool L per TP).
# realistic_dda_sweep.py APPLIES packing to Pool T and DP-saturation to Pool L.
#
# Valid Pool T N_T (cpb = 32//ceil(L/N_T) >= min_cpb):
#   7B  (32L, min 5): N_T 6..30   | 13B (40L, min 8): N_T 10..30 | 70B (80L, min 6): N_T 16..30
# Pool L N_L (TP=N_L, PP=1; cpb always 32): swept 2..(32 - min N_T).

THREADS=${1:-8}
SEQLEN_GAP=${2:-128}
OUTDIR="../results/dda32"
mkdir -p "$OUTDIR"

run_pool_L() {
    local model=$1 N_L=$2
    local OUT="$OUTDIR/sim_${model}_poolL_${N_L}.csv"
    [ -f "$OUT" ] && { echo "skip Pool L $model N_L=$N_L"; return; }
    echo "=== Pool L | $model | N_L=$N_L | TP=$N_L, PP=1 ==="
    python3 run_sim_full32.py --model "$model" --num_devices "$N_L" --total_devices 32 \
        --fc_devices "$N_L" --model_parallel \
        --generate_trace --simulate_trace --process_results --update_csv \
        --run_simulation_max_workers "$THREADS" --generate_trace_max_workers "$THREADS" \
        --seqlen_gap "$SEQLEN_GAP" --simulation_result_path "$OUT" \
        || echo "!! Pool L $model N_L=$N_L FAILED (skipped)"
}

run_pool_T() {
    local model=$1 N_T=$2
    local OUT="$OUTDIR/sim_${model}_poolT_${N_T}.csv"
    [ -f "$OUT" ] && { echo "skip Pool T $model N_T=$N_T"; return; }
    echo "=== Pool T | $model | N_T=$N_T | TP=1, PP=layers ==="
    python3 run_sim_full32.py --model "$model" --num_devices "$N_T" --total_devices 32 \
        --generate_trace --simulate_trace --process_results --update_csv \
        --run_simulation_max_workers "$THREADS" --generate_trace_max_workers "$THREADS" \
        --seqlen_gap "$SEQLEN_GAP" --simulation_result_path "$OUT" \
        || echo "!! Pool T $model N_T=$N_T FAILED (skipped)"
}

# cpb-ladder probe: num_devices > fabric, purely to supply a floor-block cpb that
# no in-fabric N_T can produce. Written as a pp static file; realistic_dda_sweep.py
# reads it as a cpb source and its FABRIC guard drops N>32 from deployments.
run_cpb_probe() {
    local model=$1 N=$2
    local OUT="../results/sim_${model}_pp_${N}dev_lanes32.csv"
    [ -f "$OUT" ] && { echo "skip cpb probe $model N=$N"; return; }
    echo "=== cpb probe | $model | N=$N ==="
    python3 run_sim_full32.py --model "$model" --num_devices "$N" --total_devices 32 \
        --generate_trace --simulate_trace --process_results --update_csv \
        --run_simulation_max_workers "$THREADS" --generate_trace_max_workers "$THREADS" \
        --seqlen_gap "$SEQLEN_GAP" --simulation_result_path "$OUT" \
        || echo "!! cpb probe $model N=$N FAILED (skipped)"
}

# $1=model  $2=maxNL  $3=minNT
sweep_model() {
    local model=$1 maxNL=$2 minNT=$3
    for N in $(seq 2 2 "$maxNL"); do run_pool_L "$model" "$N"; done
    for N in $(seq "$minNT" 2 30); do run_pool_T "$model" "$N"; done
}

echo "############ Llama2-7B ############";  sweep_model Llama2-7B  26 6
echo "############ Llama2-13B ###########";  sweep_model Llama2-13B 22 10
run_pool_L Llama2-13B 10                     # TP=10 saturation knee (off step-2 grid)
echo "############ Llama2-70B ###########";  sweep_model Llama2-70B 16 16
run_cpb_probe Llama2-70B 40                   # -> cpb=16 floor block for 70B N_T 27..30

echo ""
echo "=== comprehensive pool sweep complete ==="
echo "poolL: $(ls $OUTDIR/sim_*_poolL_*.csv 2>/dev/null | wc -l)  poolT: $(ls $OUTDIR/sim_*_poolT_*.csv 2>/dev/null | wc -l)"