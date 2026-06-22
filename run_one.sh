#!/usr/bin/env bash
# HTCondor per-job wrapper: run ONE scenario's trial-chunk on a single slot.
#
# Invoked (via campaign.sub) as:
#   bash run_one.sh <label> <exp_indep_charge_mode> <run_offset> <num_runs>
#
# Cluster-specific paths — edit HOME / REPO if you relocate.
# NOTE: Condor runs jobs in a minimal environment with no $HOME, so we set it
# explicitly (conda needs it). We use `set -e` but NOT `set -u`: conda's own
# activation scripts reference variables that can be unset in that sparse env.
set -eo pipefail

export HOME=/export/home/adesai
REPO="$HOME/Projects/sensei_charge_traps"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sensei_charge_traps
cd "$REPO"

exec python run_campaign.py \
    --flavor minimal_caldet \
    --binning-factors 32 \
    --exp-indep-charge-mode "$2" \
    --only "$1" \
    --run-offset "$3" \
    --num_runs "$4" \
    --num_workers 1
