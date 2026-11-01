#!/usr/bin/env bash
# One task and one training/inference process. Resources come from the submit helper.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${CF_ENV_SETUP:-}" ]]; then source "$CF_ENV_SETUP"; fi
: "${CF_PYTHON:?Set CF_PYTHON to the absolute path of your prepared Python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ "${1:-}" == matrix ]]; then
  : "${CF_MATRIX_DIR:?Set the directory produced by build_experiment_configs.py}"
  srun --ntasks=1 --cpu-bind=none "$CF_PYTHON" -m contrastforge.cli train \
    --config "$CF_MATRIX_DIR/run_${SLURM_ARRAY_TASK_ID:?}.json"
else
  srun --ntasks=1 --cpu-bind=none "$CF_PYTHON" -m contrastforge.cli "$@"
fi
