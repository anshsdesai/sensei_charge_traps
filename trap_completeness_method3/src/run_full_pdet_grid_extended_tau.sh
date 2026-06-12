#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="/home/ansh/miniforge3/envs/sensei_charge_traps/bin/python"

cd "${REPO_ROOT}"

exec "${PYTHON}" trap_completeness_method3/src/full_pdet_grid.py \
  --realizations 100 \
  --seed 2026052210 \
  --extended-long-tau \
  "$@"
