#!/usr/bin/env bash
# Generate TTFT (prefill-only) latency for all seqlens in simulation_results.csv.
# Uses --prefill seqlen --decoding 0 so Total Latency = pure TTFT at each seqlen.
# No new simulation — pure Python aggregation. Fast (~seconds).
# Writes: cent_simulation/processed_results_ttft.csv

set -e

OUT="processed_results_ttft.csv"
rm -f "$OUT"

SEQLENS=(128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096)

for S in "${SEQLENS[@]}"; do
    echo "=== seqlen=$S ==="
    # Pipeline parallel
    python3 run_sim.py --model Llama2-7B  --num_devices 8  --process_throughputs --phase prefill --prefill $S --decoding 0 --simulation_result_path simulation_results.csv --processed_result_path $OUT
    python3 run_sim.py --model Llama2-13B --num_devices 20 --process_throughputs --phase prefill --prefill $S --decoding 0 --simulation_result_path simulation_results.csv --processed_result_path $OUT
    python3 run_sim.py --model Llama2-70B --num_devices 32 --process_throughputs --phase prefill --prefill $S --decoding 0 --simulation_result_path simulation_results.csv --processed_result_path $OUT
    # Model parallel (tensor parallelism)
    python3 run_sim.py --model Llama2-7B  --num_devices 8  --model_parallel --process_throughputs --phase prefill --prefill $S --decoding 0 --simulation_result_path simulation_results.csv --processed_result_path $OUT
    python3 run_sim.py --model Llama2-13B --num_devices 20 --model_parallel --process_throughputs --phase prefill --prefill $S --decoding 0 --simulation_result_path simulation_results.csv --processed_result_path $OUT
    python3 run_sim.py --model Llama2-70B --num_devices 32 --model_parallel --process_throughputs --phase prefill --prefill $S --decoding 0 --simulation_result_path simulation_results.csv --processed_result_path $OUT
done

echo "Done: $OUT"
python3 -c "
import pandas as pd
df = pd.read_csv('$OUT')
print('Rows:', len(df))
print(df.groupby(['Model','Seqlen']).size().to_string())
"
