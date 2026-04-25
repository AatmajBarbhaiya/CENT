#!/usr/bin/env bash
set -e
# Wrapper: generate a single-token LLaMA trace (trace-only, op-trace)
# Run from inside the cent_simulation directory or call this script directly.

python3 function_sim.py \
  --only-trace \
  --op-trace \
  --Llama \
  --trace-file sample_aim.trace \
  --seqlen 1 \
  --n_heads 32 \
  --ffn_dim 11008 \
  --trace-prepare \
  --trace-norm \
  --trace-fc-kqvo \
  --trace-attention \
  --trace-softmax \
  --trace-fc-ffn \
  --trace-activation

echo "Trace generated: sample_aim.trace"
