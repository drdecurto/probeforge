#!/usr/bin/env bash
#SBATCH --job-name=cf-dpo-v2
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; P=$S/data/curriculav2/mined_dpo; cd ~/probeforge
echo "start $(date)"
[ -f $P/reference_train.jsonl ] || srun --ntasks=1 --cpu-bind=none contrastforge cache-reference --config configs/v2_dpo_mined.json --input $P/preferences_train.jsonl --output $P/reference_train.jsonl
[ -f $P/reference_validation.jsonl ] || srun --ntasks=1 --cpu-bind=none contrastforge cache-reference --config configs/v2_dpo_mined.json --input $P/preferences_validation.jsonl --output $P/reference_validation.jsonl
echo "reference cached $(date)"; wc -l $P/reference_*.jsonl
R=""; ls $S/runs/dpov2_mined 2>/dev/null | grep -q checkpoint- && R="--resume latest"
srun --ntasks=1 --cpu-bind=none contrastforge train --config configs/v2_dpo_mined.json --stage dpo $R 2>&1 | tail -5
echo "end $(date)"
