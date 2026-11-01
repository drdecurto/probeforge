#!/usr/bin/env bash
#SBATCH --job-name=cf-sftv4c
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; L=$S/logs; source ~/probeforge/slurm/environment.sh; cd ~/probeforge
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); i=0
while IFS='|' read name base cur cfg seed; do n=$((i/4)); R=""; ls $S/runs/sftv4c_$name 2>/dev/null | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sftv4c_$name/metrics.json ] || srun --nodes=1 --ntasks=1 -w ${NODES[$n]} --exact --gres=gpu:1 --cpus-per-task=20 --cpu-bind=none --output=$L/sftv4c-$name.out --error=$L/sftv4c-$name.err contrastforge train --config configs/v4c_$name.json $R &
  i=$((i+1)); done < $S/v4/v4c_runs.txt
wait; echo "end $(date)"
