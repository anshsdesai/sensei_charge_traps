#!/usr/bin/env bash
# HTCondor per-job wrapper: run ONE trap-only (zero dark-current) scenario's
# trial-chunk on a single slot. This is the BRACKET campaign -- it runs both
# populations:
#   effcorr -> efficiency-corrected MLE point estimate (empty tau bins zeroed;
#              lower edge of the trap-only prediction)
#   upper   -> 90% CL completeness-corrected count (empty bins filled; upper edge)
# both with the injected single-electron dark current zeroed (--zero-exp-dep),
# so trap emission is the only exposure-dependent single-e source.
#
# Invoked (via campaign_effcorr.sub) as:
#   bash run_one_effcorr.sh <label> <exp_indep_charge_mode> <run_offset> <num_runs>
#
# Needs both seed files present in the repo:
#   tau_at_135k_hist_minimal_caldet_efficiency_corrected.npz  (rsync from dev box)
#   tau_at_135k_hist_minimal_caldet_upper_limit.npz           (already on cluster)
#
# set -e but NOT -u: conda's activation scripts reference vars unset in Condor's
# sparse environment.
set -eo pipefail

export HOME=/export/home/adesai
REPO="$HOME/Projects/sensei_charge_traps"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sensei_charge_traps
cd "$REPO"

# --vp-scan always passed so the V_p band (vp3/vp10) is part of the enumeration
# universe; --only selects the single scenario this job owns. clear-modes and
# exposure-order-policy are left at run_campaign.py's own defaults (same as
# run_one.sh) so this campaign's scenario grid matches the headline one.
exec python run_campaign.py \
    --flavor minimal_caldet \
    --populations effcorr upper \
    --zero-exp-dep \
    --vp-scan \
    --binning-factors 32 \
    --exp-indep-charge-mode "$2" \
    --only "$1" \
    --run-offset "$3" \
    --num_runs "$4" \
    --num_workers 1
