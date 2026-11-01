#!/usr/bin/env bash
#SBATCH --job-name=cf-sft60-4arms
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh
S=/gpfs/scratch/bsc21/bsc021885/cf
cd ~/probeforge
echo "node $(hostname) start $(date)"
K=0
for arm in translated random mined no_counterfactual; do
  CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none \
    --output=$S/logs/sft60-$arm.out --error=$S/logs/sft60-$arm.err \
    contrastforge train --config configs/contrast_sft_mn5.json --model $S/models/Qwen3-8B \
      --train-data $S/data/curricula60/$arm/sft_train.jsonl --validation-data $S/data/curricula60/$arm/sft_validation.jsonl \
      --output-dir $S/runs/sft60_$arm &
  K=$((K+1))
done
wait
echo "all arms finished $(date)"; ls $S/runs/
