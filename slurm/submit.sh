#!/usr/bin/env bash
# Usage: PF_ACCOUNT=... PF_QOS=... bash slurm/submit.sh sft configs/sft_mn5.json [CLI overrides]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${PF_ACCOUNT:?Set PF_ACCOUNT to your BSC project account}"
: "${PF_QOS:?Set PF_QOS to an accelerated QoS authorized for your account}"
stage="${1:?stage: sft, dpo, reference, collect, branch, evaluate}"
config="${2:?config path (use - for collect/branch/evaluate)}"
shift 2
case "$stage" in
  sft|dpo) default_gpus=4 ;;
  reference|collect|branch|evaluate) default_gpus=1 ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
export PF_GPUS="${PF_GPUS:-$default_gpus}"
export PF_PYTHON="${PF_PYTHON:-$PWD/.venv/bin/python}"
mkdir -p logs
opts=(--parsable --account="$PF_ACCOUNT" --qos="$PF_QOS" --nodes=1 --ntasks=1
      --gres="gpu:$PF_GPUS" --cpus-per-task="${PF_CPUS:-32}" --time="${PF_TIME:-04:00:00}"
      --job-name="probe-$stage" --output="$PWD/logs/%x-%j.out" --error="$PWD/logs/%x-%j.err")
if [[ -n "${PF_PARTITION:-}" ]]; then opts+=(--partition="$PF_PARTITION"); fi
if [[ -n "${PF_DEPENDENCY:-}" ]]; then opts+=(--dependency="afterok:$PF_DEPENDENCY"); fi
sbatch "${opts[@]}" slurm/job.sbatch "$stage" "$config" "$@"
