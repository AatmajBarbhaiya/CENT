#!/usr/bin/env bash
# simulation_dse_fill.sh — fill every gap in the DDA split grid for the FULL design-space
# exploration. simulation_dda32_full.sh laid a step-2 grid; this adds the missing integers so
# EVERY valid (N_L, N_T) split is backed by a real simulation. Nothing is derived or reused —
# the raw CSVs are paper-submission data.
#
# Run from: experiments/dda_full_sweep/simulation/
# Usage: bash simulation_dse_fill.sh <threads> [seqlen_gap]
# Skips any (model,pool,N) whose CSV already exists -> resumes safely.
# NOT set -e: an illegal config (min_cpb / tiling wall) is recorded and skipped, not fatal.
#
# Bounds (cpb = 32//ceil(L/N_T) >= min_cpb ; N_L >= cap_floor):
#   7B  (L=32, min_cpb=5, cap_floor=1): N_T  6..31 | N_L 1..26
#   13B (L=40, min_cpb=8, cap_floor=2): N_T 10..30 | N_L 2..22
#   70B (L=80, min_cpb=6, cap_floor=9): N_T 16..23 | N_L 9..16
#
# 70B cap_floor is 9, NOT 10: the old 145 GiB weight figure priced Wk/Wv at full width, but CENT
# stores them GQA-reduced (function_sim.py:29-30) -> 128.5 GiB -> ceil(128.5/16) = 9.
# The existing 70B poolL_{2,4,6,8} sims are below the capacity floor and stay excluded downstream.
#
# Failures are appended to ../results/dda32/dse_fill_failures.log so invalid configs land in the
# DSE table as explicit `invalid` rows rather than silently vanishing.

THREADS=${1:-8}
SEQLEN_GAP=${2:-128}
OUTDIR="../results/dda32"
FAILLOG="$OUTDIR/dse_fill_failures.log"
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
        || { echo "!! Pool L $model N_L=$N_L FAILED (skipped)"; \
             echo "poolL,$model,$N_L,$(date -Is)" >> "$FAILLOG"; }
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
        || { echo "!! Pool T $model N_T=$N_T FAILED (skipped)"; \
             echo "poolT,$model,$N_T,$(date -Is)" >> "$FAILLOG"; }
}

echo "############ Llama2-7B  (N_L 1..26, N_T 6..31) ############"
for N in 1 3 5 7 9 11 13 15 17 19 21 23 25;      do run_pool_L Llama2-7B  "$N"; done
for N in 7 9 11 13 15 17 19 21 23 25 27 29 31;   do run_pool_T Llama2-7B  "$N"; done

echo "############ Llama2-13B (N_L 2..22, N_T 10..30) ###########"
for N in 3 5 7 9 11 13 15 17 19 21;              do run_pool_L Llama2-13B "$N"; done
for N in 11 13 15 17 19 21 23 25 27 29;          do run_pool_T Llama2-13B "$N"; done

echo "############ Llama2-70B (N_L 9..16, N_T 16..23) ###########"
for N in 9 11 13 15;                             do run_pool_L Llama2-70B "$N"; done
for N in 17 19 21 23;                            do run_pool_T Llama2-70B "$N"; done

echo ""
echo "=== DSE gap-fill complete ==="
echo "poolL: $(ls $OUTDIR/sim_*_poolL_*.csv 2>/dev/null | wc -l)  poolT: $(ls $OUTDIR/sim_*_poolT_*.csv 2>/dev/null | wc -l)"
[ -f "$FAILLOG" ] && { echo "failures logged:"; cat "$FAILLOG"; } || echo "no failures"
