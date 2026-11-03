#!/usr/bin/env bash
#SBATCH --job-name=cf-kolsft
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclude=as04r5b02
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; K=$S/kolibri; source $K/env.sh; cd ~/probeforge
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); i=0
for a in $ARMS; do R=""; ls $S/runs/sftkol_$a 2>/dev/null | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sftkol_$a/metrics.json ] || srun --nodes=1 --ntasks=1 -w ${NODES[$i]} --exact --cpus-per-task=80 --gres=gpu:4 --cpu-bind=none --output=$K/logs/sft_$a.$SLURM_JOB_ID.out --error=$K/logs/sft_$a.$SLURM_JOB_ID.err bash -c "source $K/env.sh; cd ~/probeforge; nvidia-smi -L | wc -l; python -m contrastforge.cli train --config configs/vkol_$a.json $R" < /dev/null &
  i=$((i+1)); done; wait; echo "end $(date)"
