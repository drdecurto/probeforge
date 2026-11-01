#!/usr/bin/env bash
# Usage: bash slurm/submit_contrast.sh train --config configs/contrast_sft_mn5.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CF_ACCOUNT:?Set your BSC project account}"
: "${CF_QOS:?Set an accelerated QoS authorized for your account}"
: "${1:?Provide a contrastforge subcommand}"
export CF_PYTHON="${CF_PYTHON:-$PWD/.venv/bin/python}"
export CF_GPUS="${CF_GPUS:-1}"
if [[ "$CF_PYTHON" != /* ]]; then echo 'CF_PYTHON must be absolute' >&2; exit 2; fi
mkdir -p logs
opts=(--parsable --account="$CF_ACCOUNT" --qos="$CF_QOS" --nodes=1 --ntasks=1
      --gres="${CF_GRES:-gpu:$CF_GPUS}" --cpus-per-task="${CF_CPUS:-8}"
      --time="${CF_TIME:-04:00:00}" --job-name="contrast-$1"
      --output="$PWD/logs/%x-%A_%a.out" --error="$PWD/logs/%x-%A_%a.err")
if [[ -n "${CF_PARTITION:-}" ]]; then opts+=(--partition="$CF_PARTITION"); fi
if [[ -n "${CF_DEPENDENCY:-}" ]]; then opts+=(--dependency="afterok:$CF_DEPENDENCY"); fi
if [[ -n "${CF_ARRAY:-}" ]]; then opts+=(--array="$CF_ARRAY"); fi
sbatch "${opts[@]}" slurm/contrast_job.sbatch "$@"
