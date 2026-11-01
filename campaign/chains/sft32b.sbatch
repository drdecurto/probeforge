#!/usr/bin/env bash
#SBATCH --job-name=cf-sft32b
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; source ~/probeforge/slurm/environment.sh; cd ~/probeforge
for n in random v4c; do R=""; ls $S/runs/sft32b_$n 2>/dev/null | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sft32b_$n/metrics.json ] || srun --exact --ntasks=1 --cpus-per-task=40 --gres=gpu:1 --cpu-bind=none --output=$L/sft32b-$n.out --error=$L/sft32b-$n.err contrastforge train --config configs/v32b_$n.json $R < /dev/null &
done; wait
