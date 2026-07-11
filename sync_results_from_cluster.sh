#!/usr/bin/env bash
# Pull completed campaign results from the cluster back to this machine for
# plotting. Run from WSL (it can reach the Windows tree under /mnt/c).
#
#   bash sync_results_from_cluster.sh
#
# Safe to re-run: it is additive (no --delete), resumes partial transfers
# (--partial), and only pulls the per-trial HDF5 outputs + their directory
# structure (skips Condor logs, scratch, and any large intermediates).
set -euo pipefail

REMOTE="adesai@hepatl31.uoregon.edu"
REMOTE_CAMPAIGN="/export/home/adesai/Projects/sensei_charge_traps/campaign/"
LOCAL_CAMPAIGN="/mnt/c/Users/Ansh/Projects/sensei_charge_traps/campaign/"

mkdir -p "$LOCAL_CAMPAIGN"
echo "Pulling results:"
echo "  from $REMOTE:$REMOTE_CAMPAIGN"
echo "  to   $LOCAL_CAMPAIGN"

rsync -avzh --partial --info=progress2 \
  --include='*/' \
  --include='ccd_traps_run*.h5' \
  --exclude='*' \
  "$REMOTE:$REMOTE_CAMPAIGN" "$LOCAL_CAMPAIGN"

echo "Done. Results are under $LOCAL_CAMPAIGN"
