#!/usr/bin/env bash
#SBATCH --job-name=stdmerger4
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; source ~/probeforge/slurm/environment.sh; export HF_HUB_OFFLINE=1; i=0
while IFS='|' read name base run; do M=$S/models/merged/$name; [ -f $M/config.json ] && continue
  ( CUDA_VISIBLE_DEVICES=$((i%4)) python $S/stdbench/merge.py $S/models/$base $S/runs/$run $M < /dev/null > $S/stdbench/logs/merge_$name.log 2>&1; echo "merge $name $([ -f $M/config.json ] && echo ok || echo FAIL)" ) &
  i=$((i+1)); [ $((i%4)) = 0 ] && wait; done < $S/v4/eval_list4.txt
wait; echo "end $(date)"
