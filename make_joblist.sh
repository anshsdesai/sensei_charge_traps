#!/usr/bin/env bash
# Generate the physical BASELINE joblist for campaign.sub: one line per
# (scenario, charge-mode, trial-offset) chunk, formatted "<label>, <mode>, <offset>, <chunk>".
#
# Labels come straight from `run_campaign.py --list`, so the job set always
# matches the real scenario enumeration (no hand-maintained list to drift).
#
# Override defaults via env, e.g.:  TOTAL=100 CHUNK=5 bash make_joblist.sh
# Set VP_SCAN=1 to also queue the V_p systematic band (vp3, vp10) -- adds 24
# baseline scenarios across both charge modes (12 per mode). run_one.sh always
# passes --vp-scan; the job list controls whether those labels are queued.
# Upper-limit and efficiency-corrected jobs are intentionally excluded; use
# make_joblist_effcorr.sh and campaign_effcorr.sub for that legacy bracket.
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
case "$FLAVOR" in
  legacy) POPULATION_FILE=trap_population_esigma.npz ;;
  minimal_caldet) POPULATION_FILE=trap_population_esigma_minimal_caldet.npz ;;
  *) echo "Unsupported FLAVOR=$FLAVOR" >&2; exit 2 ;;
esac
if [ ! -f "$POPULATION_FILE" ]; then
  echo "Missing $REPO/$POPULATION_FILE; sync the generated physical population before submitting." >&2
  exit 2
fi
: > joblist.txt
for mode in $MODES; do
  python run_campaign.py --flavor "$FLAVOR" --population-model esigma --populations baseline \
      --binning-factors "$BINNING" $VP_SCAN_FLAG \
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

echo "Wrote $(wc -l < joblist.txt) jobs to joblist.txt (TOTAL=$TOTAL, CHUNK=$CHUNK, modes: $MODES, vp_scan: ${VP_SCAN:-off}, population: physical baseline)."
