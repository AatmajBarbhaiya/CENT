#!/usr/bin/env bash
# Remove DDA simulation artifacts to prepare for a fresh run.
# Run from: experiments/dynamic_device_allocation/simulation/
#
# Usage:
#   bash clean_dda.sh            # removes both traces and results (default)
#   bash clean_dda.sh --traces   # traces only  (~2.4 GB)
#   bash clean_dda.sh --results  # results/sim_*.csv + combined_metrics.csv only
#   bash clean_dda.sh --all      # same as default

set -e
cd "$(dirname "$0")"

CLEAN_TRACES=0
CLEAN_RESULTS=0

if [[ $# -eq 0 ]]; then
    CLEAN_TRACES=1
    CLEAN_RESULTS=1
fi

for arg in "$@"; do
    case "$arg" in
        --traces)  CLEAN_TRACES=1 ;;
        --results) CLEAN_RESULTS=1 ;;
        --all)     CLEAN_TRACES=1; CLEAN_RESULTS=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [[ $CLEAN_TRACES -eq 1 ]]; then
    if [[ -d ../traces ]]; then
        echo "Removing traces/ (~$(du -sh ../traces 2>/dev/null | cut -f1)) ..."
        rm -rf ../traces/
        echo "  Done."
    else
        echo "traces/ not found, skipping."
    fi
fi

if [[ $CLEAN_RESULTS -eq 1 ]]; then
    echo "Removing results/sim_*.csv and combined_metrics.csv ..."
    rm -f ../results/sim_*.csv ../results/combined_metrics.csv
    echo "  Done."
fi

echo "Clean complete."