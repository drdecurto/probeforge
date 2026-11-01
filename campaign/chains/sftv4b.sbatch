#!/usr/bin/env bash
#SBATCH --job-name=cf-sftv4b
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
source ~/probeforge/slurm/environment.sh; cd ~/probeforge; K=0
for n in 8b phi4mini; do R=""; ls /gpfs/scratch/bsc21/bsc021885/cf/runs/sftv4b_$n 2>/dev/null | grep -q checkpoint- && R="--resume latest"
  [ -f /gpfs/scratch/bsc21/bsc021885/cf/runs/sftv4b_$n/metrics.json ] || CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/sftv4b-$n.out --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/sftv4b-$n.err contrastforge train --config configs/v4b_$n.json $R &
  K=$((K+1)); done; wait; echo "end $(date)"
