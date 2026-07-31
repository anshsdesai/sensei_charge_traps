#!/usr/bin/env bash
# HTCondor per-job wrapper: run ONE physical BASELINE scenario chunk.
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

# $HOME here is an NFS mount (hepatl30:/home). HDF5 >=1.10 takes an flock() on
# every file it opens, and on NFS those calls fail with BlockingIOError
# (errno 11, "Resource temporarily unavailable") under concurrent access or
# stale lock state -- which kills whole trial chunks and makes valid outputs
# look unreadable. Disabling the lock is safe for this workload: each trial
# writes its own uniquely-named file through a temp+atomic-rename, so no two
# processes ever write the same file and readers only see completed files.
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
POPULATION_FILE=trap_population_esigma_minimal_caldet.npz
if [ ! -f "$POPULATION_FILE" ]; then
    echo "Missing $REPO/$POPULATION_FILE; sync it from the analysis machine." >&2
    exit 2
fi

# --vp-scan is always passed so the V_p band scenarios (vp3/vp10) are part of
# the enumeration universe; --only still selects the single scenario this job
# owns, so headline vp1 jobs are unaffected. Whether band jobs actually get
# queued is controlled by make_joblist.sh (VP_SCAN=1).
exec python run_campaign.py \
    --flavor minimal_caldet \
    --population-model esigma \
    --populations baseline \
    --binning-factors 32 \
    --vp-scan \
    --exp-indep-charge-mode "$2" \
    --only "$1" \
    --run-offset "$3" \
    --num_runs "$4" \
    --num_workers 1
