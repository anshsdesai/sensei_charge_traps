#!/usr/bin/env bash
# Run one custom-population Condor chunk. expdep is with_exp_dep or zero_exp_dep.
set -eo pipefail

export HOME=/export/home/adesai
REPO="$HOME/Projects/sensei_charge_traps"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate sensei_charge_traps
cd "$REPO"

POPULATION_FILE=trap_population_custom.npz
if [ ! -f "$POPULATION_FILE" ]; then
    echo "Missing $REPO/$POPULATION_FILE; generate and sync it before submitting." >&2
    exit 2
fi

zero_exp_dep=()
case "$3" in
  with_exp_dep) ;;
  zero_exp_dep) zero_exp_dep=(--zero-exp-dep) ;;
  *) echo "Unknown expdep case: $3" >&2; exit 2 ;;
esac

exec python run_custom_campaign.py \
    --population-file "$POPULATION_FILE" \
    --exp-indep-charge-mode "$2" \
    --only "$1" \
    --run-offset "$4" \
    --num-runs "$5" \
    --num-workers 1 \
    "${zero_exp_dep[@]}"