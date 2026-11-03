#!/usr/bin/env bash
#SBATCH --job-name=cf-sftr4
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
while IFS='|' read cfg run; do n=$((i/4)); R=""; ls $S/runs/$run 2>/dev/null | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/$run/metrics.json ] || srun --nodes=1 --ntasks=1 -w ${NODES[$n]} --exact --gres=gpu:1 --cpus-per-task=20 --cpu-bind=none --output=$L/r4-$run.out --error=$L/r4-$run.err contrastforge train --config $cfg $R < /dev/null &
  i=$((i+1)); done < $S/v4/train_list4.txt
wait; echo "end $(date)"
