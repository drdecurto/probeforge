#!/usr/bin/env bash
#SBATCH --job-name=cf-sft14b-seeds
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
S=/gpfs/scratch/bsc21/bsc021885/cf; cd ~/probeforge
echo "node $(hostname) start $(date)"; K=0
for cfg in v14b_random_s1 v14b_random_s2 v14b_mined_s1 v14b_mined_s2; do
  CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none \
    --output=$S/logs/sft14b-$cfg.out --error=$S/logs/sftv2-$cfg.err contrastforge train --config configs/$cfg.json &
  K=$((K+1))
done
wait; echo "end $(date)"
