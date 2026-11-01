#!/usr/bin/env bash
#SBATCH --job-name=cf-sftphi4s
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
while IFS='|' read name cfg seed; do R=""; ls $S/runs/sftphi4_$name 2>/dev/null | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sftphi4_$name/metrics.json ] || srun --exact --ntasks=1 --cpus-per-task=20 --gres=gpu:1 --cpu-bind=none --output=$L/sftphi4s-$name.out --error=$L/sftphi4s-$name.err contrastforge train --config configs/vphi4_$name.json $R < /dev/null &
done < $S/v4/phi4_seed_runs.txt; wait
