#!/usr/bin/env bash
# Resources are passed by submit.sh. One Slurm task launches all local ranks.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${PF_ENV_SETUP:-}" ]]; then source "$PF_ENV_SETUP"; fi
: "${PF_PYTHON:?Set an absolute path to your prepared Python environment}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
stage="$1"
config="$2"
shift 2
"$PF_PYTHON" -m probeforge.cli doctor
case "$stage" in
  sft|dpo)
    srun --ntasks=1 --cpu-bind=none "$PF_PYTHON" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc_per_node="${PF_GPUS:-4}" \
      -m probeforge.cli train --config "$config" --stage "$stage" "$@"
    ;;
  reference)
    srun --ntasks=1 --cpu-bind=none "$PF_PYTHON" -m probeforge.cli cache-reference --config "$config" "$@"
    ;;
  collect|branch|evaluate)
    srun --ntasks=1 --cpu-bind=none "$PF_PYTHON" -m probeforge.cli "$stage" "$@"
    ;;
  *) echo "Unknown stage" >&2; exit 2 ;;
esac
