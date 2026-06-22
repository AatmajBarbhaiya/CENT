#!/usr/bin/env bash
# Reprocess simulation_results.csv at multiple total seqlens.
# No new simulation — pure Python aggregation. Fast (~seconds).
# Writes: cent_simulation/processed_results_multiseqlen.csv
#
# Seqlen split: PREFILL = SEQLEN/8 (rounded to 128 grid), DECODING = SEQLEN - PREFILL
# Seqlens: 512 (PREFILL=128,DEC=384), 1024 (128,896), 2048 (256,1792), 4096 (512,3584)

set -e

OUT="processed_results_multiseqlen.csv"

declare -A PREFILL=([512]=128 [1024]=128 [2048]=256 [4096]=512)
declare -A DECODING=([512]=384 [1024]=896 [2048]=1792 [4096]=3584)

for SEQLEN in 512 1024 2048 4096; do
    P=${PREFILL[$SEQLEN]}
    D=${DECODING[$SEQLEN]}
    echo "=== seqlen=$SEQLEN  prefill=$P  decoding=$D ==="
    for PHASE in prefill decoding end2end; do
        # Pipeline parallel
        python3 run_sim.py --model Llama2-7B  --num_devices 8  --process_throughputs --phase $PHASE --prefill $P --decoding $D --simulation_result_path simulation_results.csv --processed_result_path $OUT
        python3 run_sim.py --model Llama2-13B --num_devices 20 --process_throughputs --phase $PHASE --prefill $P --decoding $D --simulation_result_path simulation_results.csv --processed_result_path $OUT
        python3 run_sim.py --model Llama2-70B --num_devices 32 --process_throughputs --phase $PHASE --prefill $P --decoding $D --simulation_result_path simulation_results.csv --processed_result_path $OUT
        # Model parallel (tensor parallelism)
        python3 run_sim.py --model Llama2-7B  --num_devices 8  --model_parallel --process_throughputs --phase $PHASE --prefill $P --decoding $D --simulation_result_path simulation_results.csv --processed_result_path $OUT
        python3 run_sim.py --model Llama2-13B --num_devices 20 --model_parallel --process_throughputs --phase $PHASE --prefill $P --decoding $D --simulation_result_path simulation_results.csv --processed_result_path $OUT
        python3 run_sim.py --model Llama2-70B --num_devices 32 --model_parallel --process_throughputs --phase $PHASE --prefill $P --decoding $D --simulation_result_path simulation_results.csv --processed_result_path $OUT
    done
done

echo "Done: $OUT"
python3 -c "
import pandas as pd
df = pd.read_csv('$OUT')
print('Rows:', len(df))
print(df.groupby(['Model','Seqlen','Phase']).size().to_string())
"
