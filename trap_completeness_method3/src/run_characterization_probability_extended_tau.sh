#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="/home/ansh/miniforge3/envs/sensei_charge_traps/bin/python"

cd "${REPO_ROOT}"

exec "${PYTHON}" trap_completeness_method3/src/characterization_probability.py \
  --stage08-h5 trap_completeness_method3/cache/08_pdet_grid_tau1000_v1.h5 \
  --stage08-summary trap_completeness_method3/cache/08_pdet_grid_tau1000_summary.json \
  --output-h5 trap_completeness_method3/cache/09_characterization_probability_tau1000_v1.h5 \
  --output-summary trap_completeness_method3/cache/09_characterization_probability_tau1000_summary.json \
  --smoke-summary trap_completeness_method3/cache/09_characterization_probability_tau1000_smoke_summary.json \
  --figure-prefix 09_tau1000 \
  "$@"
