#!/usr/bin/env bash
#SBATCH --job-name=cf-sftphi4mini
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:3
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.err
set -uo pipefail
source ~/probeforge/slurm/environment.sh; S=/gpfs/scratch/bsc21/bsc021885/cf; cd ~/probeforge; K=0
for arm in translated random mined; do
  R=""; [ -d $S/runs/sftphi4mini_$arm ] && ls $S/runs/sftphi4mini_$arm | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sftphi4mini_$arm/metrics.json ] || CUDA_VISIBLE_DEVICES=$K srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/sftphi4mini-$arm.out --error=/gpfs/scratch/bsc21/bsc021885/cf/logs/sftphi4mini-$arm.err contrastforge train --config configs/vphi4mini_$arm.json $R &
  K=$((K+1))
done
wait; echo "end $(date)"
