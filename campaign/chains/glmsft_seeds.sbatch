#!/usr/bin/env bash
#SBATCH --job-name=cf-glmsfts
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; G=$S/glm; source $G/env.sh; cd ~/probeforge
for a in $ARMS; do R=""; ls $S/runs/sftglm_$a 2>/dev/null | grep -q checkpoint- && R="--resume latest"
  [ -f $S/runs/sftglm_$a/metrics.json ] || srun --exact --ntasks=1 --cpus-per-task=40 --gres=gpu:2 --cpu-bind=none --output=$G/logs/sft_$a.out --error=$G/logs/sft_$a.err bash -c "source $G/env.sh; cd ~/probeforge; python -m contrastforge.cli train --config configs/vglm_$a.json $R" < /dev/null &
done; wait
