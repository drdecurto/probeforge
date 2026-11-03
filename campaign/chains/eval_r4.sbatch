#!/usr/bin/env bash
#SBATCH --job-name=cf-evalr4
#SBATCH --account=bsc21
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/bsc21/bsc021885/cf/logs/%x-%j.out
S=/gpfs/scratch/bsc21/bsc021885/cf; E=$S/evalr4; T=$S/data/contrast13m_200/test_eval12.jsonl; mkdir -p $E; source ~/probeforge/slurm/environment.sh; cd ~/probeforge
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST)); i=0
while IFS='|' read name base run; do n=$((i/4))
  [ -f $E/test_$name.jsonl.manifest.json ] || srun --nodes=1 --ntasks=1 -w ${NODES[$n]} --exact --gres=gpu:1 --cpus-per-task=20 --cpu-bind=none --output=$E/collect_$name.out --error=$E/collect_$name.err \
     contrastforge collect --inputs $T --out $E/test_$name.jsonl --model $S/models/$base --adapter $S/runs/$run --budgets 512 --samples 1 < /dev/null &
  i=$((i+1)); done < $S/v4/eval_list4.txt
wait; echo "end $(date)"
