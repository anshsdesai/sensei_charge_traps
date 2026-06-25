#!/usr/bin/env bash
# Generate joblist_effcorr.txt for campaign_effcorr.sub: one line per
# (scenario, charge-mode, trial-offset) chunk for the trap-only (zero
# dark-current) BRACKET campaign (effcorr + upper populations).
#
# Labels come straight from `run_campaign.py --list` with the SAME flags
# run_one_effcorr.sh uses, so the job set always matches the real enumeration
# (no hand-maintained list to drift).
#
# Override defaults via env, e.g.:  TOTAL=100 CHUNK=5 bash make_joblist_effcorr.sh
set -euo pipefail

REPO=/export/home/adesai/Projects/sensei_charge_traps
cd "$REPO"

TOTAL=${TOTAL:-200}                       # trials per scenario
CHUNK=${CHUNK:-10}                        # trials per Condor job (~CHUNK*6 min)
MODES=${MODES:-"pre_readout post_readout"}
FLAVOR=${FLAVOR:-minimal_caldet}

: > joblist_effcorr.txt
for mode in $MODES; do
  python run_campaign.py --flavor "$FLAVOR" --populations effcorr upper --zero-exp-dep \
      --clear-modes sequencer three_hour --exposure-order-policy all --vp-scan \
      --exp-indep-charge-mode "$mode" --list \
    | awk '/^(minos|snolab)_/{print $1}' \
    | while read -r label; do
        off=0
        while [ "$off" -lt "$TOTAL" ]; do
          echo "$label, $mode, $off, $CHUNK" >> joblist_effcorr.txt
          off=$((off + CHUNK))
        done
      done
done

echo "Wrote $(wc -l < joblist_effcorr.txt) jobs to joblist_effcorr.txt (TOTAL=$TOTAL, CHUNK=$CHUNK, modes: $MODES)."
