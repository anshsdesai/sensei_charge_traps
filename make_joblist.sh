#!/usr/bin/env bash
# Generate joblist.txt for campaign.sub: one line per (scenario, charge-mode,
# trial-offset) chunk, formatted "<label>, <mode>, <offset>, <chunk>".
#
# Labels come straight from `run_campaign.py --list`, so the job set always
# matches the real scenario enumeration (no hand-maintained list to drift).
#
# Override defaults via env, e.g.:  TOTAL=100 CHUNK=5 bash make_joblist.sh
# Set VP_SCAN=1 to also queue the V_p systematic band (vp1, vp10) — adds 48
# scenarios (24 per mode). run_one.sh always passes --vp-scan, so these jobs
# run correctly. Default (unset) queues only the central V_p=3 headline set.
set -euo pipefail

REPO=/export/home/adesai/Projects/sensei_charge_traps
cd "$REPO"

TOTAL=${TOTAL:-200}                       # trials per scenario
CHUNK=${CHUNK:-10}                        # trials per Condor job (~CHUNK*6 min)
MODES=${MODES:-"pre_readout post_readout"}
FLAVOR=${FLAVOR:-minimal_caldet}
BINNING=${BINNING:-32}
VP_SCAN_FLAG=""
[ -n "${VP_SCAN:-}" ] && VP_SCAN_FLAG="--vp-scan"

: > joblist.txt
for mode in $MODES; do
  python run_campaign.py --flavor "$FLAVOR" --binning-factors "$BINNING" $VP_SCAN_FLAG \
      --exp-indep-charge-mode "$mode" --list \
    | awk '/^(minos|snolab)_/{print $1}' \
    | while read -r label; do
        off=0
        while [ "$off" -lt "$TOTAL" ]; do
          echo "$label, $mode, $off, $CHUNK" >> joblist.txt
          off=$((off + CHUNK))
        done
      done
done

echo "Wrote $(wc -l < joblist.txt) jobs to joblist.txt (TOTAL=$TOTAL, CHUNK=$CHUNK, modes: $MODES, vp_scan: ${VP_SCAN:-off})."
