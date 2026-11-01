#!/usr/bin/env bash
#SBATCH --job-name=cf-sftv3-14b-seeds
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
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; cd ~/probeforge; K=0
for n in random_s2 random_s3 mined_s2 mined_s3; do
  R=""; [ -d $S/runs/sftv3_14b_$n ] && ls $S/runs/sftv3_14b_$n | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sftv3_14b_$n/metrics.json ] || CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/sftv3_14b-$n.out --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/sftv3_14b-$n.err contrastforge train --config configs/v3_14b_$n.json $R &
  K=$((K+1))
done
wait; echo "end $(date)"
