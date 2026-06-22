#!/usr/bin/env bash
# Build combined_metrics.csv from Pool L + Pool T simulation results.
# Run from: experiments/dynamic_device_allocation/simulation/
# Requires: results/sim_{model}_pool{L|T}_{N}.csv to exist (run simulation_dda.sh first).

set -e
cd "$(dirname "$0")"

echo "=== Building combined_metrics.csv ==="
python3 combine_pools.py
echo "=== Done: results/combined_metrics.csv ==="