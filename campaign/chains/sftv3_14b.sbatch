#!/usr/bin/env bash
#SBATCH --job-name=cf-sftv3-14b
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
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; cd ~/probeforge; K=0
for arm in random mined; do
  R=""; [ -d $S/runs/sftv3_14b_$arm ] && ls $S/runs/sftv3_14b_$arm | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sftv3_14b_$arm/metrics.json ] || CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/sftv3_14b-$arm.out --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/sftv3_14b-$arm.err contrastforge train --config configs/v3_14b_$arm.json $R &
  K=$((K+1))
done
wait; echo "end $(date)"
