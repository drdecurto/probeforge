#!/usr/bin/env bash
#SBATCH --job-name=cf-sft32bs
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; source ~/probeforge/slurm/environment.sh; cd ~/probeforge
for a in random_s2 random_s3 v4c_s2 v4c_s3; do R=""; ls $S/runs/sft32b_$a 2>/dev/null | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sft32b_$a/metrics.json ] || srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sft32b-$a.out --error=$L/sft32b-$a.err contrastforge train --config configs/v32b_$a.json $R < /dev/null &
done; wait
