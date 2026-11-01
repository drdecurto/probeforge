#!/usr/bin/env bash
#SBATCH --job-name=cf-collect-base
#SBATCH --account=bsc21
#SBATCH --qos=acc_bsccase
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%A_%a.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%A_%a.err
set -euo pipefail
source ~/probeforge/slurm/environment.sh
S=/gpfs/scratch/bsc21/bsc021885/cf; D=$S/data/contrast60; K=$SLURM_ARRAY_TASK_ID
cd ~/probeforge
echo "shard $K on $(hostname) gpu=$CUDA_VISIBLE_DEVICES start $(date)"
srun --ntasks=1 --cpu-bind=none contrastforge collect \
  --inputs $D/train.jsonl $D/validation.jsonl \
  --only-ids $D/shard$K.json \
  --out $S/data/student_base.shard$K.jsonl --model $S/models/Qwen3-8B \
  --budgets 512 --samples 1
echo "shard $K end $(date)"
