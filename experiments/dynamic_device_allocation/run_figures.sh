#!/usr/bin/env bash
# Run from: experiments/dynamic_device_allocation/
# Outputs go to: figures/
set -e
cd "$(dirname "$0")"

python3 figure_scripts/figure_motivation_dda.py
python3 figure_scripts/figure_dda_strategy1_shortseq.py
python3 figure_scripts/figure_dda_strategy1_longseq.py
python3 figure_scripts/figure_dda_strategy2_tol.py
python3 figure_scripts/figure_dda_strategy2_alpha.py
python3 figure_scripts/figure_dda_strategy3_v2.py
