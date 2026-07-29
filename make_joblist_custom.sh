#!/usr/bin/env bash
# Generate jobs for the custom physical E/sigma campaign. Defaults are the
# campaign defaults and queue both normal and trap-only dark-current cases.
set -euo pipefail

REPO=/export/home/adesai/Projects/sensei_charge_traps
cd "$REPO"

TOTAL=${TOTAL:-200}
CHUNK=${CHUNK:-10}
MODES=${MODES:-pre_readout}
EXP_DEP_CASES=${EXP_DEP_CASES:-"with_exp_dep zero_exp_dep"}
POPULATION_FILE=${POPULATION_FILE:-trap_population_custom.npz}

if [ ! -f "$POPULATION_FILE" ]; then
  echo "Missing $REPO/$POPULATION_FILE; generate and sync it before submitting." >&2
  exit 2
fi

: > joblist_custom.txt
for expdep in $EXP_DEP_CASES; do
  zero_exp_dep=()
  [ "$expdep" = zero_exp_dep ] && zero_exp_dep=(--zero-exp-dep)
  for mode in $MODES; do
    python run_custom_campaign.py --population-file "$POPULATION_FILE" \
        "${zero_exp_dep[@]}" --exp-indep-charge-mode "$mode" --list \
      | while read -r label; do
          off=0
          while [ "$off" -lt "$TOTAL" ]; do
            echo "$label, $mode, $expdep, $off, $CHUNK" >> joblist_custom.txt
            off=$((off + CHUNK))
          done
        done
  done
done

echo "Wrote $(wc -l < joblist_custom.txt) jobs (TOTAL=$TOTAL, CHUNK=$CHUNK, modes: $MODES, exp-dep cases: $EXP_DEP_CASES)."